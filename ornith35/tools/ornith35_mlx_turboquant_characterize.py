#!/usr/bin/env python3
"""Characterize real Ornith-35 K/V and evaluate bounded TurboQuant profiles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any, Sequence

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_layer as layer
import ornith35_mlx_model as model
import ornith35_mlx_turboquant as turboquant
import ornith35_turboquant_reference as reference
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TextTokenizer,
    load_text_tokenizer,
    render_text_prompt,
)


FORMAT = "ornith35-turboquant-characterization-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = DEFAULT_ROOT / "experiments" / "turboquant-characterize-v1" / "report.json"
EXPECTED_MLX_VERSION = "0.32.0"
CALIBRATION_CHANNELS = 128
QUERY_SAMPLES_PER_PROMPT = 8
NATIVE_TOKENS = 262_144
YARN_TOKENS = 524_288


@dataclass(frozen=True)
class PromptSpec:
    name: str
    role: str
    path: Path


@dataclass(frozen=True)
class EncodedPrompt:
    spec: PromptSpec
    rendered: str
    token_ids: tuple[int, ...]
    rendered_sha256: str
    token_sha256: str


@dataclass(frozen=True)
class LayerTrace:
    layer_index: int
    queries: mx.array
    gates: mx.array
    keys: mx.array
    values: mx.array


@dataclass(frozen=True)
class PromptTrace:
    prompt: EncodedPrompt
    layers: tuple[LayerTrace, ...]
    hidden: mx.array
    state: model.TextModelState


@dataclass(frozen=True)
class Profile:
    name: str
    key_qjl: bool
    key_bits: tuple[int, ...]
    value_bits: tuple[int, ...]
    norm_dtype: str
    description: str

    @property
    def split(self) -> bool:
        return len(self.key_bits) == 2


PROFILES = (
    Profile(
        "bf16-control",
        False,
        (16,),
        (16,),
        "bf16",
        "Uncompressed BF16 control through the same evaluator.",
    ),
    Profile(
        "bf16-k-v4-mse",
        False,
        (16,),
        (4,),
        "bf16",
        "BF16 keys with four-bit MSE values to isolate value error.",
    ),
    Profile(
        "k4-mse-v-bf16",
        False,
        (4,),
        (16,),
        "bf16",
        "Four-bit MSE keys with BF16 values to isolate score error.",
    ),
    Profile(
        "k4-mse-v4-bf16norm",
        False,
        (4,),
        (4,),
        "bf16",
        "Four-bit MSE keys and values; conservative non-QJL control.",
    ),
    Profile(
        "k5-mse-v4-bf16norm",
        False,
        (5,),
        (4,),
        "bf16",
        "Five-bit MSE keys with four-bit MSE values to recover score precision.",
    ),
    Profile(
        "k4-mse-v5-bf16norm",
        False,
        (4,),
        (5,),
        "bf16",
        "Four-bit MSE keys with five-bit MSE values to isolate value recovery.",
    ),
    Profile(
        "k5-mse-v5-bf16norm",
        False,
        (5,),
        (5,),
        "bf16",
        "Five-bit MSE keys and values as the symmetric recovery control.",
    ),
    Profile(
        "k6-mse-v5-bf16norm",
        False,
        (6,),
        (5,),
        "bf16",
        "Six-bit MSE keys with five-bit MSE values to isolate score recovery.",
    ),
    Profile(
        "k5-mse-v6-bf16norm",
        False,
        (5,),
        (6,),
        "bf16",
        "Five-bit MSE keys with six-bit MSE values to isolate value recovery.",
    ),
    Profile(
        "k6-mse-v6-bf16norm",
        False,
        (6,),
        (6,),
        "bf16",
        "Six-bit MSE keys and values as the symmetric high-fidelity control.",
    ),
    Profile(
        "k4-qjl-v4-bf16norm",
        True,
        (4,),
        (4,),
        "bf16",
        "Four-bit product keys and four-bit MSE values.",
    ),
    Profile(
        "k4-qjl-v3-bf16norm",
        True,
        (4,),
        (3,),
        "bf16",
        "Asymmetric four-bit product keys and three-bit MSE values.",
    ),
    Profile(
        "k4-mse-v3-bf16norm",
        False,
        (4,),
        (3,),
        "bf16",
        "Asymmetric four-bit MSE keys and three-bit MSE values.",
    ),
    Profile(
        "k4-mse-v2-bf16norm",
        False,
        (4,),
        (2,),
        "bf16",
        "Four-bit MSE keys with aggressive two-bit MSE values.",
    ),
    Profile(
        "split35-qjl-bf16norm",
        True,
        (4, 3),
        (4, 3),
        "bf16",
        "Calibration-ranked 128/128 channel split at effective 3.5 bits.",
    ),
    Profile(
        "split35-qjl-fp32norm",
        True,
        (4, 3),
        (4, 3),
        "fp32",
        "The 3.5-bit split with conservative FP32 vector norms.",
    ),
    Profile(
        "split35-mse-bf16norm",
        False,
        (4, 3),
        (4, 3),
        "bf16",
        "Calibration-ranked 3.5-bit split using MSE keys and values.",
    ),
    Profile(
        "k3-qjl-v3-bf16norm",
        True,
        (3,),
        (3,),
        "bf16",
        "Three-bit product keys and three-bit MSE values.",
    ),
    Profile(
        "k3-mse-v3-bf16norm",
        False,
        (3,),
        (3,),
        "bf16",
        "Three-bit MSE keys and values; non-QJL low-bit control.",
    ),
    Profile(
        "k3-qjl-v2-bf16norm",
        True,
        (3,),
        (2,),
        "bf16",
        "Aggressive three-bit product keys and two-bit MSE values.",
    ),
)


DEFAULT_PROMPTS = (
    PromptSpec(
        "calibration-code-audit",
        "calibration",
        REPOSITORY_ROOT / "tests" / "test-vectors" / "prompts" / "long_code_audit.txt",
    ),
    PromptSpec(
        "calibration-memory-archive",
        "calibration",
        REPOSITORY_ROOT / "tests" / "test-vectors" / "prompts" / "long_memory_archive.txt",
    ),
    PromptSpec(
        "holdout-security",
        "holdout",
        REPOSITORY_ROOT / "tests" / "long_context_security_prompt.txt",
    ),
    PromptSpec(
        "holdout-story",
        "holdout",
        REPOSITORY_ROOT / "tests" / "long_context_story_prompt.txt",
    ),
)


TRANSFORM_SEEDS = {
    "key_whole_rotation": 202_607_180_101,
    "key_whole_projection": 202_607_180_102,
    "value_whole_rotation": 202_607_180_103,
    "key_high_rotation": 202_607_180_111,
    "key_low_rotation": 202_607_180_112,
    "key_high_projection": 202_607_180_113,
    "key_low_projection": 202_607_180_114,
    "value_high_rotation": 202_607_180_121,
    "value_low_rotation": 202_607_180_122,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise reference.TurboQuantError(message)


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"{timestamp} {message}", flush=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def repository_revision() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(len(revision) == 40, "invalid repository revision")
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def encode_bounded_prompt(
    tokenizer: TextTokenizer,
    spec: PromptSpec,
    max_tokens: int,
) -> EncodedPrompt:
    require(spec.role in ("calibration", "holdout"), "invalid prompt role")
    require(spec.path.is_file(), f"missing prompt source: {spec.path}")
    text = spec.path.read_text(encoding="utf-8").strip()
    require(text, f"empty prompt source: {spec.path}")

    def render_prefix(characters: int) -> tuple[str, tuple[int, ...]]:
        rendered = render_text_prompt(text[:characters], enable_thinking=True)
        return rendered, tokenizer.encode(rendered)

    high = min(len(text), max_tokens * 64)
    rendered, token_ids = render_prefix(high)
    if len(token_ids) <= max_tokens:
        while high < len(text) and len(token_ids) <= max_tokens:
            next_high = min(len(text), high * 2)
            candidate_rendered, candidate_ids = render_prefix(next_high)
            if len(candidate_ids) > max_tokens:
                break
            high = next_high
            rendered, token_ids = candidate_rendered, candidate_ids
        if high == len(text):
            require(token_ids, "rendered prompt encoded to no tokens")
            return EncodedPrompt(
                spec=spec,
                rendered=rendered,
                token_ids=token_ids,
                rendered_sha256=sha256_bytes(rendered.encode("utf-8")),
                token_sha256=token_sha256(token_ids),
            )

    low = 1
    best: tuple[str, tuple[int, ...]] | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate = render_prefix(middle)
        if len(candidate[1]) <= max_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    require(best is not None and best[1], "token bound is too small for chat framing")
    rendered, token_ids = best
    return EncodedPrompt(
        spec=spec,
        rendered=rendered,
        token_ids=token_ids,
        rendered_sha256=sha256_bytes(rendered.encode("utf-8")),
        token_sha256=token_sha256(token_ids),
    )


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    require(values, "cannot summarize an empty metric")
    ordered = sorted(float(value) for value in values)
    require(all(math.isfinite(value) for value in ordered), "metric is not finite")

    def percentile(fraction: float) -> float:
        index = math.ceil((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": math.fsum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def profile_storage(profile: Profile) -> dict[str, int | float | str]:
    require(profile.norm_dtype in ("bf16", "fp32"), "invalid profile norm dtype")
    require(
        len(profile.key_bits) == len(profile.value_bits) and len(profile.key_bits) in (1, 2),
        "invalid profile group geometry",
    )
    dimensions = (256,) if len(profile.key_bits) == 1 else (128, 128)
    scalar_bytes = 2 if profile.norm_dtype == "bf16" else 4
    if profile.key_qjl:
        key_bytes = reference.split_product_vector_bytes(
            dimensions,
            profile.key_bits,
            scalar_bytes=scalar_bytes,
            alignment=4,
        )
    else:
        key_bytes = sum(
            size * 2
            if bits == 16
            else reference.mse_vector_bytes(
                size,
                bits,
                scalar_bytes=scalar_bytes,
                alignment=4,
            )
            for size, bits in zip(dimensions, profile.key_bits)
        )
    value_bytes = sum(
        size * 2
        if bits == 16
        else reference.mse_vector_bytes(
            size,
            bits,
            scalar_bytes=scalar_bytes,
            alignment=4,
        )
        for size, bits in zip(dimensions, profile.value_bits)
    )
    native_bytes = reference.cache_payload_bytes(
        NATIVE_TOKENS,
        10,
        2,
        key_bytes,
        value_bytes,
    )
    yarn_bytes = reference.cache_payload_bytes(
        YARN_TOKENS,
        10,
        2,
        key_bytes,
        value_bytes,
    )
    weighted_key_bits = sum(size * bits for size, bits in zip(dimensions, profile.key_bits)) / 256
    weighted_value_bits = (
        sum(size * bits for size, bits in zip(dimensions, profile.value_bits)) / 256
    )
    return {
        "name": profile.name,
        "description": profile.description,
        "norm_dtype": profile.norm_dtype,
        "key_bytes_per_vector": key_bytes,
        "value_bytes_per_vector": value_bytes,
        "coordinate_key_bits": weighted_key_bits,
        "coordinate_value_bits": weighted_value_bits,
        "physical_bits_per_kv_channel": (key_bytes + value_bytes) * 8 / 512,
        "native_cache_bytes": native_bytes,
        "native_cache_gib": native_bytes / 2**30,
        "yarn_cache_bytes": yarn_bytes,
        "yarn_cache_gib": yarn_bytes / 2**30,
        "native_compression_ratio": (5 * 2**30) / native_bytes,
    }


def build_transforms() -> dict[str, turboquant.MLXTransform]:
    return {
        "key_whole_rotation": turboquant.haar_rotation(
            256, TRANSFORM_SEEDS["key_whole_rotation"]
        ),
        "key_whole_projection": turboquant.qjl_projection(
            256, TRANSFORM_SEEDS["key_whole_projection"]
        ),
        "value_whole_rotation": turboquant.haar_rotation(
            256, TRANSFORM_SEEDS["value_whole_rotation"]
        ),
        "key_high_rotation": turboquant.haar_rotation(
            128, TRANSFORM_SEEDS["key_high_rotation"]
        ),
        "key_low_rotation": turboquant.haar_rotation(
            128, TRANSFORM_SEEDS["key_low_rotation"]
        ),
        "key_high_projection": turboquant.qjl_projection(
            128, TRANSFORM_SEEDS["key_high_projection"]
        ),
        "key_low_projection": turboquant.qjl_projection(
            128, TRANSFORM_SEEDS["key_low_projection"]
        ),
        "value_high_rotation": turboquant.haar_rotation(
            128, TRANSFORM_SEEDS["value_high_rotation"]
        ),
        "value_low_rotation": turboquant.haar_rotation(
            128, TRANSFORM_SEEDS["value_low_rotation"]
        ),
    }


def transform_report(
    transforms: dict[str, turboquant.MLXTransform],
) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "dimension": transform.matrix.shape[0],
            "kind": transform.kind,
            "seed": transform.seed,
            "sha256": transform.sha256,
        }
        for name, transform in transforms.items()
    }


def capture_prompt(
    prompt: EncodedPrompt,
    weights: model.TextModelWeights,
) -> PromptTrace:
    started = time.perf_counter()
    state = model.initial_state(weights, model.PRODUCTION_CONFIG, context.NATIVE_PROFILE_ID)
    result = model.prefill_hidden_chunk_with_attention_inputs(
        prompt.token_ids,
        state,
        weights,
        use_steel=False,
    )
    traces: list[LayerTrace] = []
    for layer_index, hidden in zip(result.attention_layer_indices, result.attention_inputs):
        layer_weights = weights.layers[layer_index]
        layer_state = result.state.layers[layer_index]
        require(
            isinstance(layer_weights, layer.AttentionLayerWeights),
            f"attention weight mismatch at layer {layer_index}",
        )
        require(
            isinstance(layer_state, attention.MLXAttentionState),
            f"attention state mismatch at layer {layer_index}",
        )
        projected = attention.project_prefill_qkv_for_analysis(
            hidden,
            layer_weights.token_mixer,
            0,
            context_profile=context.NATIVE_PROFILE_ID,
        )
        expected_keys = mx.swapaxes(projected.keys, 0, 1)
        expected_values = mx.swapaxes(projected.values, 0, 1)
        mx.eval(
            projected.queries,
            projected.gates,
            projected.keys,
            projected.values,
            layer_state.keys,
            layer_state.values,
        )
        require(
            bool(mx.array_equal(expected_keys, layer_state.keys).item()),
            f"projected keys differ from authoritative cache at layer {layer_index}",
        )
        require(
            bool(mx.array_equal(expected_values, layer_state.values).item()),
            f"projected values differ from authoritative cache at layer {layer_index}",
        )
        traces.append(
            LayerTrace(
                layer_index=layer_index,
                queries=projected.queries,
                gates=projected.gates,
                keys=projected.keys,
                values=projected.values,
            )
        )
    require(len(traces) == 10, "full-attention trace count mismatch")
    mx.eval(result.hidden)
    log(
        f"capture-done prompt={prompt.spec.name} role={prompt.spec.role} "
        f"tokens={len(prompt.token_ids)} layers={len(traces)} "
        f"elapsed_s={time.perf_counter() - started:.3f}"
    )
    return PromptTrace(
        prompt=prompt,
        layers=tuple(traces),
        hidden=result.hidden,
        state=result.state,
    )


def channel_statistics(vectors: mx.array) -> dict[str, list[float]]:
    require(vectors.ndim == 2 and vectors.shape[1] == 256, "channel-stat geometry mismatch")
    source = vectors.astype(mx.float32)
    mean = mx.mean(source, axis=0)
    centered = source - mean
    variance = mx.mean(centered * centered, axis=0)
    rms = mx.sqrt(mx.mean(source * source, axis=0))
    maximum = mx.max(mx.abs(source), axis=0)
    kurtosis = mx.mean(centered**4, axis=0) / mx.maximum(variance * variance, 1e-20)
    mx.eval(mean, variance, rms, maximum, kurtosis)
    return {
        "mean": mean.tolist(),
        "std": mx.sqrt(variance).tolist(),
        "rms": rms.tolist(),
        "max_abs": maximum.tolist(),
        "kurtosis": kurtosis.tolist(),
    }


def top_channels(values: Sequence[float], count: int) -> tuple[int, ...]:
    require(len(values) == 256 and 0 < count < 256, "channel selection geometry mismatch")
    ranked = sorted(range(256), key=lambda index: (-float(values[index]), index))
    return tuple(sorted(ranked[:count]))


def sampled_query_positions(tokens: int, count: int = QUERY_SAMPLES_PER_PROMPT) -> tuple[int, ...]:
    require(tokens > 0 and count > 0, "query sampling geometry mismatch")
    if tokens <= count:
        return tuple(range(tokens))
    positions = tuple(
        min(tokens - 1, math.ceil((index + 1) * tokens / count) - 1)
        for index in range(count)
    )
    require(len(set(positions)) == count and positions[-1] == tokens - 1, "query sampling drift")
    return positions


def calibrate_channels(
    captures: Sequence[PromptTrace],
) -> tuple[dict[str, Any], dict[tuple[int, int, str], turboquant.MLXChannelSplit]]:
    require(captures, "calibration capture set is empty")
    require(all(capture.prompt.spec.role == "calibration" for capture in captures), "role leak")
    report: dict[str, Any] = {}
    splits: dict[tuple[int, int, str], turboquant.MLXChannelSplit] = {}
    for trace_index in range(10):
        layer_index = captures[0].layers[trace_index].layer_index
        require(
            all(capture.layers[trace_index].layer_index == layer_index for capture in captures),
            "calibration layer order mismatch",
        )
        layer_report: dict[str, Any] = {}
        for kv_head in range(2):
            keys = mx.concatenate(
                tuple(capture.layers[trace_index].keys[:, kv_head, :] for capture in captures),
                axis=0,
            )
            values = mx.concatenate(
                tuple(capture.layers[trace_index].values[:, kv_head, :] for capture in captures),
                axis=0,
            )
            queries = mx.concatenate(
                tuple(
                    capture.layers[trace_index].queries[:, kv_head * 8 : (kv_head + 1) * 8, :]
                    .reshape(-1, 256)
                    for capture in captures
                ),
                axis=0,
            )
            key_stats = channel_statistics(keys)
            value_stats = channel_statistics(values)
            query_rms = channel_statistics(queries)["rms"]
            key_importance = [
                key_rms * query_scale
                for key_rms, query_scale in zip(key_stats["rms"], query_rms)
            ]
            value_importance = list(value_stats["rms"])
            key_high = top_channels(key_importance, CALIBRATION_CHANNELS)
            value_high = top_channels(value_importance, CALIBRATION_CHANNELS)
            splits[(layer_index, kv_head, "key")] = turboquant.channel_split(256, key_high)
            splits[(layer_index, kv_head, "value")] = turboquant.channel_split(256, value_high)

            def energy_fraction(rms: Sequence[float], selected: Sequence[int]) -> float:
                energy = [float(value) ** 2 for value in rms]
                return math.fsum(energy[index] for index in selected) / math.fsum(energy)

            layer_report[str(kv_head)] = {
                "samples": keys.shape[0],
                "key": {
                    **key_stats,
                    "query_rms": query_rms,
                    "importance": key_importance,
                    "high_channels": list(key_high),
                    "high_energy_fraction": energy_fraction(key_stats["rms"], key_high),
                },
                "value": {
                    **value_stats,
                    "importance": value_importance,
                    "high_channels": list(value_high),
                    "high_energy_fraction": energy_fraction(value_stats["rms"], value_high),
                },
            }
        report[str(layer_index)] = layer_report
        log(f"calibration-layer-done layer={layer_index}")
    return report, splits


def candidate_head(
    profile: Profile,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    key_split: turboquant.MLXChannelSplit,
    value_split: turboquant.MLXChannelSplit,
    transforms: dict[str, turboquant.MLXTransform],
) -> tuple[mx.array, mx.array]:
    norm_dtype = mx.bfloat16 if profile.norm_dtype == "bf16" else mx.float32
    if profile.split:
        if profile.key_qjl:
            key_encoding = turboquant.quantize_split_product(
                keys,
                key_split,
                profile.key_bits[0],
                profile.key_bits[1],
                transforms["key_high_rotation"],
                transforms["key_low_rotation"],
                transforms["key_high_projection"],
                transforms["key_low_projection"],
                norm_dtype=norm_dtype,
            )
            scores = turboquant.split_product_inner_products(
                queries,
                key_encoding,
                transforms["key_high_rotation"],
                transforms["key_low_rotation"],
                transforms["key_high_projection"],
                transforms["key_low_projection"],
            )
        else:
            key_encoding = turboquant.quantize_split_mse(
                keys,
                key_split,
                profile.key_bits[0],
                profile.key_bits[1],
                transforms["key_high_rotation"],
                transforms["key_low_rotation"],
                norm_dtype=norm_dtype,
            )
            scores = turboquant.split_mse_inner_products(
                queries,
                key_encoding,
                transforms["key_high_rotation"],
                transforms["key_low_rotation"],
            )
        value_encoding = turboquant.quantize_split_mse(
            values,
            value_split,
            profile.value_bits[0],
            profile.value_bits[1],
            transforms["value_high_rotation"],
            transforms["value_low_rotation"],
            norm_dtype=norm_dtype,
        )
        reconstructed_values = turboquant.dequantize_split_mse(
            value_encoding,
            transforms["value_high_rotation"],
            transforms["value_low_rotation"],
        )
        return scores, reconstructed_values

    if profile.key_bits[0] == 16:
        require(len(profile.key_bits) == 1 and not profile.key_qjl, "invalid BF16 key profile")
        scores = queries.astype(mx.float32) @ mx.swapaxes(keys.astype(mx.float32), -2, -1)
    elif profile.key_qjl:
        key_encoding = turboquant.quantize_product(
            keys,
            profile.key_bits[0],
            transforms["key_whole_rotation"],
            transforms["key_whole_projection"],
            norm_dtype=norm_dtype,
        )
        scores = turboquant.product_inner_products(
            queries,
            key_encoding,
            transforms["key_whole_rotation"],
            transforms["key_whole_projection"],
        )
    else:
        key_encoding = turboquant.quantize_mse(
            keys,
            profile.key_bits[0],
            transforms["key_whole_rotation"],
            norm_dtype=norm_dtype,
        )
        scores = turboquant.mse_inner_products(
            queries,
            key_encoding,
            transforms["key_whole_rotation"],
        )
    if profile.value_bits[0] == 16:
        require(len(profile.value_bits) == 1, "invalid BF16 value profile")
        reconstructed_values = values.astype(mx.float32)
    else:
        value_encoding = turboquant.quantize_mse(
            values,
            profile.value_bits[0],
            transforms["value_whole_rotation"],
            norm_dtype=norm_dtype,
        )
        reconstructed_values = turboquant.dequantize_mse(
            value_encoding,
            transforms["value_whole_rotation"],
        )
    return scores, reconstructed_values


def reconstruct_mse_segment(
    vectors: mx.array,
    profile: Profile,
    layer_index: int,
    kv_head: int,
    kind: str,
    splits: dict[tuple[int, int, str], turboquant.MLXChannelSplit],
    transforms: dict[str, turboquant.MLXTransform],
) -> mx.array:
    require(kind in ("key", "value"), "invalid K/V segment kind")
    require(not profile.key_qjl, "QJL keys require direct-score evaluation")
    bits = profile.key_bits if kind == "key" else profile.value_bits
    if bits == (16,):
        return vectors.astype(mx.bfloat16)
    norm_dtype = mx.bfloat16 if profile.norm_dtype == "bf16" else mx.float32
    if len(bits) == 2:
        rotation_prefix = "key" if kind == "key" else "value"
        encoding = turboquant.quantize_split_mse(
            vectors,
            splits[(layer_index, kv_head, kind)],
            bits[0],
            bits[1],
            transforms[f"{rotation_prefix}_high_rotation"],
            transforms[f"{rotation_prefix}_low_rotation"],
            norm_dtype=norm_dtype,
        )
        result = turboquant.dequantize_split_mse(
            encoding,
            transforms[f"{rotation_prefix}_high_rotation"],
            transforms[f"{rotation_prefix}_low_rotation"],
        )
    else:
        rotation = transforms[f"{kind}_whole_rotation"]
        encoding = turboquant.quantize_mse(
            vectors,
            bits[0],
            rotation,
            norm_dtype=norm_dtype,
        )
        result = turboquant.dequantize_mse(encoding, rotation)
    return result.astype(mx.bfloat16)


def compress_state_prefix(
    state: model.TextModelState,
    profile: Profile,
    splits: dict[tuple[int, int, str], turboquant.MLXChannelSplit],
    transforms: dict[str, turboquant.MLXTransform],
    *,
    exact_tail: int = 1,
) -> model.TextModelState:
    require(not profile.key_qjl, "QJL state injection is not representable as BF16 K/V")
    require(0 <= exact_tail <= state.position, "invalid exact K/V tail")
    compressed_length = state.position - exact_tail
    next_layers = []
    arrays: list[mx.array] = []
    for layer_index, layer_state in enumerate(state.layers):
        if not isinstance(layer_state, attention.MLXAttentionState):
            next_layers.append(layer_state)
            continue
        keys = []
        values = []
        for kv_head in range(2):
            key_prefix = reconstruct_mse_segment(
                layer_state.keys[kv_head, :compressed_length],
                profile,
                layer_index,
                kv_head,
                "key",
                splits,
                transforms,
            ) if compressed_length else layer_state.keys[kv_head, :0]
            value_prefix = reconstruct_mse_segment(
                layer_state.values[kv_head, :compressed_length],
                profile,
                layer_index,
                kv_head,
                "value",
                splits,
                transforms,
            ) if compressed_length else layer_state.values[kv_head, :0]
            keys.append(mx.concatenate((key_prefix, layer_state.keys[kv_head, compressed_length:])))
            values.append(
                mx.concatenate((value_prefix, layer_state.values[kv_head, compressed_length:]))
            )
        next_state = attention.MLXAttentionState(
            keys=mx.stack(tuple(keys)),
            values=mx.stack(tuple(values)),
            context_profile=layer_state.context_profile,
        )
        arrays.extend((next_state.keys, next_state.values))
        next_layers.append(next_state)
    mx.eval(*arrays)
    return model.TextModelState(
        position=state.position,
        layers=tuple(next_layers),
        context_profile=state.context_profile,
    )


def advance_compressed_state(
    previous: model.TextModelState,
    advanced: model.TextModelState,
    profile: Profile,
    splits: dict[tuple[int, int, str], turboquant.MLXChannelSplit],
    transforms: dict[str, turboquant.MLXTransform],
) -> model.TextModelState:
    require(advanced.position == previous.position + 1, "trajectory state did not advance once")
    if profile.key_bits == (16,) and profile.value_bits == (16,):
        return advanced
    require(previous.position > 0 and not profile.key_qjl, "invalid compressed trajectory")
    stable_length = previous.position - 1
    next_layers = []
    arrays: list[mx.array] = []
    for layer_index, (previous_layer, advanced_layer) in enumerate(
        zip(previous.layers, advanced.layers)
    ):
        if not isinstance(previous_layer, attention.MLXAttentionState):
            next_layers.append(advanced_layer)
            continue
        require(
            isinstance(advanced_layer, attention.MLXAttentionState),
            f"advanced attention state mismatch at layer {layer_index}",
        )
        keys = []
        values = []
        for kv_head in range(2):
            compressed_key = reconstruct_mse_segment(
                previous_layer.keys[kv_head, stable_length : stable_length + 1],
                profile,
                layer_index,
                kv_head,
                "key",
                splits,
                transforms,
            )
            compressed_value = reconstruct_mse_segment(
                previous_layer.values[kv_head, stable_length : stable_length + 1],
                profile,
                layer_index,
                kv_head,
                "value",
                splits,
                transforms,
            )
            keys.append(
                mx.concatenate(
                    (
                        previous_layer.keys[kv_head, :stable_length],
                        compressed_key,
                        advanced_layer.keys[kv_head, previous.position : previous.position + 1],
                    )
                )
            )
            values.append(
                mx.concatenate(
                    (
                        previous_layer.values[kv_head, :stable_length],
                        compressed_value,
                        advanced_layer.values[kv_head, previous.position : previous.position + 1],
                    )
                )
            )
        next_state = attention.MLXAttentionState(
            keys=mx.stack(tuple(keys)),
            values=mx.stack(tuple(values)),
            context_profile=previous_layer.context_profile,
        )
        arrays.extend((next_state.keys, next_state.values))
        next_layers.append(next_state)
    mx.eval(*arrays)
    return model.TextModelState(
        position=advanced.position,
        layers=tuple(next_layers),
        context_profile=advanced.context_profile,
    )


def vector_metrics(source: mx.array, candidate: mx.array) -> dict[str, float]:
    source = source.astype(mx.float32).reshape(-1)
    candidate = candidate.astype(mx.float32).reshape(-1)
    require(source.shape == candidate.shape and source.size > 0, "vector metric geometry mismatch")
    error = candidate - source
    source_norm = mx.sqrt(mx.sum(source * source))
    candidate_norm = mx.sqrt(mx.sum(candidate * candidate))
    error_norm = mx.sqrt(mx.sum(error * error))
    cosine = mx.sum(source * candidate) / mx.maximum(source_norm * candidate_norm, 1e-20)
    mse = mx.mean(error * error)
    maximum = mx.max(mx.abs(error))
    mx.eval(source_norm, error_norm, cosine, mse, maximum)
    return {
        "relative_l2": float((error_norm / mx.maximum(source_norm, 1e-20)).item()),
        "cosine": float(cosine.item()),
        "mse": float(mse.item()),
        "max_abs": float(maximum.item()),
    }


def logit_metrics(source_logits: mx.array, candidate_logits: mx.array) -> dict[str, float | int]:
    require(
        source_logits.ndim == candidate_logits.ndim == 1
        and source_logits.shape == candidate_logits.shape,
        "logit metric geometry mismatch",
    )
    source = source_logits.astype(mx.float32)
    candidate = candidate_logits.astype(mx.float32)
    source_top = mx.argmax(source)
    candidate_top = mx.argmax(candidate)
    top_k = 8
    source_indices = mx.argpartition(source, source.size - top_k)[-top_k:]
    candidate_indices = mx.argpartition(candidate, candidate.size - top_k)[-top_k:]
    source_log = source - mx.logsumexp(source)
    candidate_log = candidate - mx.logsumexp(candidate)
    divergence = mx.sum(mx.exp(source_log) * (source_log - candidate_log))
    mx.eval(source_top, candidate_top, source_indices, candidate_indices, divergence)
    source_set = set(source_indices.tolist())
    candidate_set = set(candidate_indices.tolist())
    report: dict[str, float | int] = vector_metrics(source, candidate)
    report.update(
        {
            "kl": max(0.0, float(divergence.item())),
            "top1_agreement": int(source_top.item()) == int(candidate_top.item()),
            "top8_recall": len(source_set & candidate_set) / top_k,
            "source_top_token": int(source_top.item()),
            "candidate_top_token": int(candidate_top.item()),
        }
    )
    return report


def build_baseline_trajectory(
    capture: PromptTrace,
    weights: model.TextModelWeights,
    steps: int,
) -> tuple[tuple[int, ...], tuple[mx.array, ...]]:
    require(steps > 0, "trajectory must contain at least one step")
    prompt_logits = model.project_lm_head(weights.lm_head, capture.hidden[-1])
    next_token = mx.argmax(prompt_logits)
    mx.eval(next_token)
    token_id = int(next_token.item())
    state = capture.state
    inputs: list[int] = []
    logits: list[mx.array] = []
    for _ in range(steps):
        inputs.append(token_id)
        result = model.forward_token(token_id, state, weights)
        mx.eval(result.logits)
        logits.append(result.logits)
        next_token = mx.argmax(result.logits)
        mx.eval(next_token)
        token_id = int(next_token.item())
        state = result.state
    return tuple(inputs), tuple(logits)


def evaluate_compressed_trajectory(
    capture: PromptTrace,
    weights: model.TextModelWeights,
    profile: Profile,
    input_tokens: Sequence[int],
    baseline_logits: Sequence[mx.array],
    splits: dict[tuple[int, int, str], turboquant.MLXChannelSplit],
    transforms: dict[str, turboquant.MLXTransform],
) -> list[dict[str, Any]]:
    require(
        input_tokens and len(input_tokens) == len(baseline_logits),
        "baseline trajectory geometry mismatch",
    )
    state = compress_state_prefix(
        capture.state,
        profile,
        splits,
        transforms,
        exact_tail=1,
    )
    reports: list[dict[str, Any]] = []
    for step, (token_id, source_logits) in enumerate(zip(input_tokens, baseline_logits)):
        result = model.forward_token(token_id, state, weights)
        mx.eval(result.logits)
        report: dict[str, Any] = logit_metrics(source_logits, result.logits)
        report.update(
            {
                "prompt": capture.prompt.spec.name,
                "step": step,
                "input_token": token_id,
                "position": state.position,
            }
        )
        reports.append(report)
        if step + 1 < len(input_tokens):
            state = advance_compressed_state(
                state,
                result.state,
                profile,
                splits,
                transforms,
            )
    return reports


def aggregate_trajectory(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(reports, "trajectory report is empty")
    metric_names = (
        "relative_l2",
        "cosine",
        "mse",
        "max_abs",
        "kl",
        "top1_agreement",
        "top8_recall",
    )
    return {
        name: summarize([float(report[name]) for report in reports])
        for name in metric_names
    }


def attention_metrics(
    source_scores: mx.array,
    candidate_scores: mx.array,
    source_probabilities: mx.array,
    candidate_probabilities: mx.array,
) -> dict[str, float]:
    require(
        source_scores.ndim == candidate_scores.ndim == 1
        and source_scores.shape == candidate_scores.shape
        and source_scores.shape == source_probabilities.shape
        and source_scores.shape == candidate_probabilities.shape,
        "attention metric geometry mismatch",
    )
    error = candidate_scores.astype(mx.float32) - source_scores.astype(mx.float32)
    source = source_scores.astype(mx.float32)
    source_scale = mx.sqrt(mx.mean(source * source))
    rmse = mx.sqrt(mx.mean(error * error))
    bias = mx.mean(error)
    maximum = mx.max(mx.abs(error))
    probability_l1 = mx.sum(mx.abs(candidate_probabilities - source_probabilities))
    source_safe = mx.maximum(source_probabilities, 1e-30)
    candidate_safe = mx.maximum(candidate_probabilities, 1e-30)
    kl = mx.sum(source_safe * (mx.log(source_safe) - mx.log(candidate_safe)))
    mx.eval(rmse, source_scale, bias, maximum, probability_l1, kl)
    source_list = source_probabilities.tolist()
    candidate_list = candidate_probabilities.tolist()
    source_ranked = sorted(range(len(source_list)), key=lambda index: (-source_list[index], index))
    candidate_ranked = sorted(
        range(len(candidate_list)), key=lambda index: (-candidate_list[index], index)
    )
    top = min(8, len(source_list))
    top_recall = len(set(source_ranked[:top]) & set(candidate_ranked[:top])) / top
    return {
        "score_rmse": float(rmse.item()),
        "score_relative_rmse": float((rmse / mx.maximum(source_scale, 1e-20)).item()),
        "score_bias": float(bias.item()),
        "score_max_abs": float(maximum.item()),
        "probability_l1": float(probability_l1.item()),
        "probability_kl": max(0.0, float(kl.item())),
        "top1_agreement": float(source_ranked[0] == candidate_ranked[0]),
        "top8_recall": top_recall,
    }


def evaluate_profile_layer(
    prompt_name: str,
    trace: LayerTrace,
    layer_weights: layer.AttentionLayerWeights,
    profile: Profile,
    splits: dict[tuple[int, int, str], turboquant.MLXChannelSplit],
    transforms: dict[str, turboquant.MLXTransform],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = sampled_query_positions(trace.queries.shape[0])
    position_indices = mx.array(positions, dtype=mx.uint32)
    source_attended: list[mx.array] = []
    candidate_attended: list[mx.array] = []
    head_reports: list[dict[str, Any]] = []
    scale = model.PRODUCTION_CONFIG.attention.head_dim**-0.5
    for kv_head in range(2):
        query_start = kv_head * 8
        queries = trace.queries[position_indices, query_start : query_start + 8, :].reshape(
            -1, 256
        ).astype(mx.float32)
        keys = trace.keys[:, kv_head, :].astype(mx.float32)
        values = trace.values[:, kv_head, :].astype(mx.float32)
        source_scores = (queries @ mx.swapaxes(keys, -2, -1)) * scale
        candidate_scores, reconstructed_values = candidate_head(
            profile,
            queries,
            keys,
            values,
            splits[(trace.layer_index, kv_head, "key")],
            splits[(trace.layer_index, kv_head, "value")],
            transforms,
        )
        candidate_scores = candidate_scores * scale
        row_positions = mx.array(
            [position for position in positions for _ in range(8)],
            dtype=mx.int32,
        )
        columns = mx.arange(keys.shape[0], dtype=mx.int32)[None, :]
        causal = columns <= row_positions[:, None]
        source_probabilities = mx.softmax(
            mx.where(causal, source_scores, -float("inf")),
            axis=-1,
        )
        candidate_probabilities = mx.softmax(
            mx.where(causal, candidate_scores, -float("inf")),
            axis=-1,
        )
        source_head_output = source_probabilities @ values
        candidate_head_output = candidate_probabilities @ reconstructed_values
        mx.eval(
            source_scores,
            candidate_scores,
            source_probabilities,
            candidate_probabilities,
            source_head_output,
            candidate_head_output,
        )
        source_attended.append(source_head_output.reshape(len(positions), 8, 256))
        candidate_attended.append(candidate_head_output.reshape(len(positions), 8, 256))
        for position_index, position in enumerate(positions):
            prefix = position + 1
            for local_head in range(8):
                row = position_index * 8 + local_head
                report = attention_metrics(
                    source_scores[row, :prefix],
                    candidate_scores[row, :prefix],
                    source_probabilities[row, :prefix],
                    candidate_probabilities[row, :prefix],
                )
                report.update(
                    {
                        "prompt": prompt_name,
                        "layer": trace.layer_index,
                        "position": position,
                        "kv_head": kv_head,
                        "query_head": query_start + local_head,
                    }
                )
                head_reports.append(report)

    source = mx.concatenate(tuple(source_attended), axis=1)
    candidate = mx.concatenate(tuple(candidate_attended), axis=1)
    gates = mx.sigmoid(trace.gates[position_indices].astype(mx.float32))
    source_output = mx.stack(
        tuple(
            layer_weights.token_mixer.o_proj.astype(mx.float32)
            @ (source[index] * gates[index]).reshape(-1)
            for index in range(len(positions))
        )
    )
    candidate_output = mx.stack(
        tuple(
            layer_weights.token_mixer.o_proj.astype(mx.float32)
            @ (candidate[index] * gates[index]).reshape(-1)
            for index in range(len(positions))
        )
    )
    mx.eval(source_output, candidate_output)
    mixer_reports: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        mixer_report: dict[str, Any] = vector_metrics(
            source_output[index],
            candidate_output[index],
        )
        mixer_report.update(
            {"prompt": prompt_name, "layer": trace.layer_index, "position": position}
        )
        mixer_reports.append(mixer_report)
    return head_reports, mixer_reports


def aggregate_profile(
    head_reports: Sequence[dict[str, Any]],
    mixer_reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    require(head_reports and mixer_reports, "profile report is empty")
    head_names = (
        "score_rmse",
        "score_relative_rmse",
        "score_bias",
        "score_max_abs",
        "probability_l1",
        "probability_kl",
        "top1_agreement",
        "top8_recall",
    )
    mixer_names = ("relative_l2", "cosine", "mse", "max_abs")
    return {
        "attention_heads": {
            name: summarize([float(report[name]) for report in head_reports])
            for name in head_names
        },
        "mixer_outputs": {
            name: summarize([float(report[name]) for report in mixer_reports])
            for name in mixer_names
        },
    }


def evaluate_holdout(
    capture: PromptTrace,
    weights: model.TextModelWeights,
    splits: dict[tuple[int, int, str], turboquant.MLXChannelSplit],
    transforms: dict[str, turboquant.MLXTransform],
    accumulators: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    require(capture.prompt.spec.role == "holdout", "calibration prompt entered holdout")
    for profile in PROFILES:
        started = time.perf_counter()
        for trace in capture.layers:
            layer_weights = weights.layers[trace.layer_index]
            require(
                isinstance(layer_weights, layer.AttentionLayerWeights),
                f"attention weight mismatch at layer {trace.layer_index}",
            )
            head, mixers = evaluate_profile_layer(
                capture.prompt.spec.name,
                trace,
                layer_weights,
                profile,
                splits,
                transforms,
            )
            accumulators[profile.name]["heads"].extend(head)
            accumulators[profile.name]["mixers"].extend(mixers)
        log(
            f"profile-done prompt={capture.prompt.spec.name} profile={profile.name} "
            f"elapsed_s={time.perf_counter() - started:.3f}"
        )


def source_identity(root: Path) -> dict[str, Any]:
    state_path = root / "source-nvfp4-state.json"
    require(state_path.is_file(), f"missing verified source state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    require(state.get("format") == "ornith35-source-verified-v1", "source is not verified")
    return {
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "repository": state.get("repository"),
        "revision": state.get("revision"),
        "weight": state.get("weight"),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--tokens-per-prompt", type=int, default=128)
    result.add_argument("--trajectory-steps", type=int, default=4)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        require(32 <= args.tokens_per_prompt <= 512, "prompt token bound must be in [32, 512]")
        require(1 <= args.trajectory_steps <= 16, "trajectory steps must be in [1, 16]")
        require(version("mlx") == EXPECTED_MLX_VERSION, "MLX version mismatch")
        source = source_identity(args.root)
        plan = {
            "format": FORMAT,
            "model_root": str(args.root),
            "output": str(args.output),
            "tokens_per_prompt": args.tokens_per_prompt,
            "trajectory_steps": args.trajectory_steps,
            "prompt_count": len(DEFAULT_PROMPTS),
            "calibration_prompts": sum(prompt.role == "calibration" for prompt in DEFAULT_PROMPTS),
            "holdout_prompts": sum(prompt.role == "holdout" for prompt in DEFAULT_PROMPTS),
            "profiles": [profile_storage(profile) for profile in PROFILES],
            "source": source,
            "persistent_trace_bytes": 0,
        }
        if args.plan_only:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        revision, dirty = repository_revision()
        tokenizer = load_text_tokenizer(args.root)
        prompts = [
            encode_bounded_prompt(tokenizer, prompt, args.tokens_per_prompt)
            for prompt in DEFAULT_PROMPTS
        ]
        require(
            len({prompt.rendered_sha256 for prompt in prompts}) == len(prompts),
            "prompt identities overlap",
        )
        log(
            f"run-start prompts={len(prompts)} tokens_bound={args.tokens_per_prompt} "
            f"free_gib={os.statvfs(args.root).f_bavail * os.statvfs(args.root).f_frsize / 2**30:.3f}"
        )
        mx.reset_peak_memory()
        transforms = build_transforms()
        log("transforms-ready")
        weights = model.load_text_model(
            args.root,
            map_embedding=True,
            quantize_lm_head=True,
        )
        log(f"model-loaded active_gib={mx.get_active_memory() / 2**30:.3f}")

        calibration_captures = [
            capture_prompt(prompt, weights)
            for prompt in prompts
            if prompt.spec.role == "calibration"
        ]
        calibration, splits = calibrate_channels(calibration_captures)
        del calibration_captures
        mx.clear_cache()

        accumulators = {
            profile.name: {"heads": [], "mixers": [], "trajectory": []}
            for profile in PROFILES
        }
        for prompt in prompts:
            if prompt.spec.role != "holdout":
                continue
            capture = capture_prompt(prompt, weights)
            evaluate_holdout(capture, weights, splits, transforms, accumulators)
            input_tokens, baseline_logits = build_baseline_trajectory(
                capture,
                weights,
                args.trajectory_steps,
            )
            for profile in PROFILES:
                if profile.key_qjl:
                    continue
                started = time.perf_counter()
                reports = evaluate_compressed_trajectory(
                    capture,
                    weights,
                    profile,
                    input_tokens,
                    baseline_logits,
                    splits,
                    transforms,
                )
                accumulators[profile.name]["trajectory"].extend(reports)
                log(
                    f"trajectory-done prompt={prompt.spec.name} profile={profile.name} "
                    f"steps={len(reports)} elapsed_s={time.perf_counter() - started:.3f}"
                )
            del baseline_logits
            del capture
            mx.clear_cache()

        runtime_files = (
            Path(__file__),
            REPOSITORY_ROOT / "ornith35" / "tools" / "ornith35_mlx_turboquant.py",
            REPOSITORY_ROOT / "ornith35" / "tools" / "ornith35_turboquant_reference.py",
            REPOSITORY_ROOT / "ornith35" / "tools" / "ornith35_mlx_attention.py",
            REPOSITORY_ROOT / "ornith35" / "tools" / "ornith35_mlx_model.py",
        )
        profile_reports: dict[str, Any] = {}
        for profile in PROFILES:
            values = accumulators[profile.name]
            profile_reports[profile.name] = {
                "storage": profile_storage(profile),
                "aggregate": aggregate_profile(values["heads"], values["mixers"]),
                "head_cases": values["heads"],
                "mixer_cases": values["mixers"],
                "trajectory": (
                    {
                        "status": "measured",
                        "exact_tail_tokens": 1,
                        "aggregate": aggregate_trajectory(values["trajectory"]),
                        "cases": values["trajectory"],
                    }
                    if values["trajectory"]
                    else {
                        "status": "not-representable-by-bf16-state-injection",
                        "reason": "QJL keys require direct packed score evaluation.",
                    }
                ),
            }
        report = {
            "format": FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "characterization-only",
            "authority": "BF16 cache; no production cache behavior changed",
            "trajectory_contract": (
                "Teacher-forced baseline-greedy inputs with one exact recent K/V token; "
                "older MSE cache entries are reconstructed once and never requantized."
            ),
            "paper_ambiguities": [
                "The paper does not specify a reproducible outlier-channel selector.",
                "Its stated 32x3 plus 96x2 example computes to 2.25, not 2.5 bits.",
            ],
            "source": source,
            "repository": {
                "revision": revision,
                "dirty": dirty,
                "runtime_files": {
                    str(path.relative_to(REPOSITORY_ROOT)): {
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in runtime_files
                },
            },
            "runtime": {
                "mlx_version": version("mlx"),
                "active_gib": mx.get_active_memory() / 2**30,
                "peak_gib": mx.get_peak_memory() / 2**30,
            },
            "prompts": [
                {
                    "name": prompt.spec.name,
                    "role": prompt.spec.role,
                    "source_path": str(prompt.spec.path.relative_to(REPOSITORY_ROOT)),
                    "source_sha256": sha256_file(prompt.spec.path),
                    "rendered_sha256": prompt.rendered_sha256,
                    "token_sha256": prompt.token_sha256,
                    "tokens": len(prompt.token_ids),
                }
                for prompt in prompts
            ],
            "prompt_set_sha256": canonical_sha256(
                [
                    {
                        "name": prompt.spec.name,
                        "role": prompt.spec.role,
                        "rendered_sha256": prompt.rendered_sha256,
                        "token_sha256": prompt.token_sha256,
                    }
                    for prompt in prompts
                ]
            ),
            "transforms": transform_report(transforms),
            "calibration": calibration,
            "profiles": profile_reports,
        }
        atomic_json(args.output, report)
        report_sha256 = sha256_file(args.output)
        log(
            f"run-done output={args.output} bytes={args.output.stat().st_size} "
            f"sha256={report_sha256} peak_gib={mx.get_peak_memory() / 2**30:.3f}"
        )
        return 0
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        reference.TurboQuantError,
    ) as exc:
        print(f"ornith35 TurboQuant characterization failed: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

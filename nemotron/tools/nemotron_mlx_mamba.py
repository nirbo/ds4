#!/usr/bin/env python3
"""Load and execute one official Nemotron Mamba2 layer through MLX."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock
from mlx_lm.models.ssm import ssm_update

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear, ModelOptFP8Linear, fp8_matvec, fp8_matvec_custom


SEQUENCE_RELATIVE_L2_LIMIT = 1e-7
SEQUENCE_MAX_ABS_LIMIT = 2e-7
SEQUENCE_STATE_MAX_ABS_LIMIT = 4e-5


def load_mamba_projection(tensors: dict[str, mx.array], prefix: str, implementation=fp8_matvec):
    weight_name = f"{prefix}.weight"
    require(weight_name in tensors, f"missing Mamba projection: {weight_name}")
    weight = tensors[weight_name]
    if weight.dtype == mx.bfloat16:
        return ModelOptBF16Linear(weight)
    require(weight.dtype == mx.uint8, f"unsupported Mamba projection dtype: {weight_name} {weight.dtype}")
    scale_name = f"{prefix}.weight_scale"
    require(scale_name in tensors, f"missing Mamba FP8 scale: {scale_name}")
    return ModelOptFP8Linear(weight, tensors[scale_name], implementation)


def layer_tensors(source_dir: Path, layer: int) -> dict[str, mx.array]:
    index = load_json(source_dir / "model.safetensors.index.json")
    prefix = f"backbone.layers.{layer}."
    shard_names = sorted(
        {shard for name, shard in index.get("weight_map", {}).items() if name.startswith(prefix)}
    )
    require(shard_names, f"no tensors found for layer {layer}")
    result: dict[str, mx.array] = {}
    for shard_name in shard_names:
        result.update(
            (name, value)
            for name, value in mx.load(str(source_dir / shard_name)).items()
            if name.startswith(prefix) and ".experts." not in name
        )
    return result


def load_mamba_layer(
    source_dir: Path,
    layer: int,
    implementation: Callable[[mx.array, mx.array, mx.array], mx.array] = fp8_matvec,
) -> NemotronHBlock:
    config = load_json(source_dir / "config.json")
    args = ModelArgs.from_dict(config)
    require(args.hybrid_override_pattern[layer] == "M", f"layer {layer} is not Mamba2")
    tensors = layer_tensors(source_dir, layer)
    base = f"backbone.layers.{layer}"
    mixer = f"{base}.mixer"
    required = [
        f"{base}.norm.weight",
        f"{mixer}.A_log",
        f"{mixer}.D",
        f"{mixer}.dt_bias",
        f"{mixer}.conv1d.weight",
        f"{mixer}.conv1d.bias",
        f"{mixer}.norm.weight",
        f"{mixer}.in_proj.weight",
        f"{mixer}.out_proj.weight",
    ]
    for name in required:
        require(name in tensors, f"missing Mamba tensor: {name}")

    block = NemotronHBlock(args, "M")
    block.norm.weight = tensors[f"{base}.norm.weight"]
    block.mixer.A_log = tensors[f"{mixer}.A_log"]
    block.mixer.D = tensors[f"{mixer}.D"]
    block.mixer.dt_bias = tensors[f"{mixer}.dt_bias"]
    block.mixer.conv1d.weight = tensors[f"{mixer}.conv1d.weight"].moveaxis(2, 1)
    block.mixer.conv1d.bias = tensors[f"{mixer}.conv1d.bias"]
    block.mixer.norm.weight = tensors[f"{mixer}.norm.weight"]
    block.mixer.in_proj = load_mamba_projection(tensors, f"{mixer}.in_proj", implementation)
    block.mixer.out_proj = load_mamba_projection(tensors, f"{mixer}.out_proj", implementation)
    block.eval()
    return block


def mamba_sequence_exact(
    block: NemotronHBlock,
    x: mx.array,
    cache: ArraysCache,
    mask: mx.array | None = None,
    capture_token: int | None = None,
    captured_state: list[mx.array] | None = None,
) -> mx.array:
    """Batch projection work while preserving one-token SSM recurrence order."""

    require(x.ndim == 3 and x.shape[1] > 1, "exact Mamba sequence requires multiple tokens")
    mixer = block.mixer
    require(
        (capture_token is None and captured_state is None)
        or (
            capture_token is not None
            and captured_state is not None
            and 0 <= capture_token < x.shape[1]
            and not captured_state
        ),
        "invalid Mamba intermediate-state capture request",
    )
    residual = x
    hidden = block.norm(x)
    projected = mixer.in_proj(hidden)
    gate, conv_input, dt = mx.split(
        projected,
        [mixer.intermediate_size, mixer.intermediate_size + mixer.conv_dim],
        axis=-1,
    )
    initial_conv_state = cache[0]
    conv_output = mixer._conv(conv_input, cache, mask)
    captured_conv_state = None
    if capture_token is not None:
        require(cache.lengths is None, "Mamba capture does not support per-request lengths")
        if initial_conv_state is None:
            initial_conv_state = mx.zeros(
                (conv_input.shape[0], mixer.conv_kernel_size - 1, mixer.conv_dim),
                dtype=conv_input.dtype,
            )
        conv_history = mx.concatenate([initial_conv_state, conv_input], axis=1)
        start = capture_token + 1
        captured_conv_state = conv_history[
            :, start : start + mixer.conv_kernel_size - 1, :
        ]
    hidden_ssm, B, C = mx.split(
        conv_output,
        [mixer.intermediate_size, mixer.intermediate_size + mixer.n_groups * mixer.ssm_state_size],
        axis=-1,
    )
    batch_size, sequence_length, _ = hidden_ssm.shape
    hidden_ssm = hidden_ssm.reshape(
        batch_size,
        sequence_length,
        mixer.num_heads,
        mixer.head_dim,
    )
    B = B.reshape(batch_size, sequence_length, mixer.n_groups, mixer.ssm_state_size)
    C = C.reshape(batch_size, sequence_length, mixer.n_groups, mixer.ssm_state_size)

    state = cache[1]
    outputs = []
    captured_ssm_state = None
    for token in range(sequence_length):
        token_mask = mask[:, token : token + 1] if mask is not None else None
        output, state = ssm_update(
            hidden_ssm[:, token : token + 1],
            mixer.A_log,
            B[:, token : token + 1],
            C[:, token : token + 1],
            mixer.D.astype(hidden_ssm.dtype),
            dt[:, token : token + 1],
            mixer.dt_bias,
            state,
            mixer.time_step_limit,
            token_mask,
        )
        outputs.append(output)
        if token == capture_token:
            captured_ssm_state = state
    cache[1] = state
    cache.advance(sequence_length)
    if captured_state is not None:
        require(
            captured_conv_state is not None and captured_ssm_state is not None,
            "Mamba intermediate state was not captured",
        )
        captured_state.extend([captured_conv_state, captured_ssm_state])
    output = mx.concatenate(outputs, axis=1).reshape(
        batch_size,
        sequence_length,
        mixer.intermediate_size,
    )
    output = mixer.norm(output, gate)
    return residual + mixer.out_proj(output)


def compare_implementations(source_dir: Path, layer: int) -> dict[str, float]:
    native = load_mamba_layer(source_dir, layer, fp8_matvec)
    reference = load_mamba_layer(source_dir, layer, fp8_matvec_custom)
    hidden_size = native.norm.weight.size
    native_cache = ArraysCache(size=2)
    reference_cache = ArraysCache(size=2)
    native_output = reference_output = None
    for step in range(4):
        x = mx.array(
            [
                math.sin(index * 0.009 + step * 0.13) * 0.3
                + math.cos(index * 0.017 - step * 0.07) * 0.1
                for index in range(hidden_size)
            ],
            dtype=mx.float32,
        ).reshape(1, 1, hidden_size)
        native_output = native(x, mask=None, cache=native_cache)
        reference_output = reference(x, mask=None, cache=reference_cache)
    require(native_output is not None and reference_output is not None, "Mamba comparison produced no output")
    mx.eval(native_output, reference_output, native_cache.state, reference_cache.state)
    difference = native_output - reference_output
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(reference_output)))
    state_error = max(
        float(mx.max(mx.abs(left - right)))
        for left, right in zip(native_cache.state, reference_cache.state)
    )
    return {
        "relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "max_abs": float(mx.max(mx.abs(difference))),
        "state_max_abs": state_error,
    }


def sequence_parity(source_dir: Path, layer: int, tokens: int = 4) -> dict[str, float]:
    batched_block = load_mamba_layer(source_dir, layer)
    incremental_block = load_mamba_layer(source_dir, layer)
    hidden_size = batched_block.norm.weight.size
    sequence = mx.array(
        [
            [
                math.sin(index * 0.009 + token * 0.13) * 0.3
                + math.cos(index * 0.017 - token * 0.07) * 0.1
                for index in range(hidden_size)
            ]
            for token in range(tokens)
        ],
        dtype=mx.float32,
    ).reshape(1, tokens, hidden_size)
    batched_cache = ArraysCache(size=2)
    incremental_cache = ArraysCache(size=2)
    captured = []
    batched = mamba_sequence_exact(
        batched_block,
        sequence,
        batched_cache,
        capture_token=0,
        captured_state=captured,
    )
    incremental_outputs = []
    first_incremental_state = None
    for token in range(tokens):
        incremental_outputs.append(
            incremental_block(
                sequence[:, token : token + 1],
                mask=None,
                cache=incremental_cache,
            )
        )
        if token == 0:
            first_incremental_state = tuple(incremental_cache.state)
    incremental = mx.concatenate(incremental_outputs, axis=1)
    require(first_incremental_state is not None, "incremental Mamba state was not captured")
    mx.eval(
        batched,
        incremental,
        batched_cache.state,
        incremental_cache.state,
        captured,
        first_incremental_state,
    )
    difference = batched - incremental
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(incremental)))
    state_error = max(
        float(mx.max(mx.abs(left - right)))
        for left, right in zip(batched_cache.state, incremental_cache.state)
    )
    captured_state_error = max(
        float(mx.max(mx.abs(left - right)))
        for left, right in zip(captured, first_incremental_state)
    )
    return {
        "sequence_relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "sequence_max_abs": float(mx.max(mx.abs(difference))),
        "sequence_state_max_abs": state_error,
        "captured_state_max_abs": captured_state_error,
    }


def benchmark(source_dir: Path, layer: int, repeats: int) -> dict[str, float]:
    block = load_mamba_layer(source_dir, layer)
    hidden_size = block.norm.weight.size
    x = mx.array([math.sin(index * 0.01) * 0.2 for index in range(hidden_size)], dtype=mx.float32).reshape(
        1, 1, hidden_size
    )
    cache = ArraysCache(size=2)
    warm = block(x, mask=None, cache=cache)
    mx.eval(warm, cache.state)
    mx.synchronize()

    started = time.perf_counter()
    outputs = []
    for _ in range(repeats):
        output = block(x, mask=None, cache=cache)
        outputs.append(output)
    mx.eval(*outputs, cache.state)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "ms": elapsed * 1000 / repeats,
        "checksum": float(outputs[-1].sum()),
        "active_mib": mx.get_active_memory() / 2**20,
        "peak_mib": mx.get_peak_memory() / 2**20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        comparison = compare_implementations(args.source_dir, args.layer)
        comparison.update(sequence_parity(args.source_dir, args.layer))
        performance = benchmark(args.source_dir, args.layer, args.repeats)
        print(
            f"mlx mamba: layer={args.layer} ms={performance['ms']:.6f} "
            f"active={performance['active_mib']:.1f}MiB peak={performance['peak_mib']:.1f}MiB "
            f"relative_l2={comparison['relative_l2']:.9g} max_abs={comparison['max_abs']:.9g} "
            f"state_max_abs={comparison['state_max_abs']:.9g} "
            f"sequence_relative_l2={comparison['sequence_relative_l2']:.9g} "
            f"sequence_max_abs={comparison['sequence_max_abs']:.9g} "
            f"sequence_state_max_abs={comparison['sequence_state_max_abs']:.9g} "
            f"captured_state_max_abs={comparison['captured_state_max_abs']:.9g} "
            f"checksum={performance['checksum']:.9g}"
        )
        require(
            comparison["relative_l2"] <= 2e-5
            and comparison["max_abs"] <= 2e-4
            and comparison["state_max_abs"] <= 2e-4,
            "Mamba optimized/reference drift exceeds tolerance",
        )
        require(
            comparison["sequence_relative_l2"] <= SEQUENCE_RELATIVE_L2_LIMIT
            and comparison["sequence_max_abs"] <= SEQUENCE_MAX_ABS_LIMIT
            and comparison["sequence_state_max_abs"] <= SEQUENCE_STATE_MAX_ABS_LIMIT
            and comparison["captured_state_max_abs"] <= SEQUENCE_STATE_MAX_ABS_LIMIT,
            "Mamba recurrent-order sequence drift exceeds the validated envelope",
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        print(f"nemotron MLX Mamba error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

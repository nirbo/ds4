#!/usr/bin/env python3
"""Fit one-bit MTP expert endpoints to routed teacher activations."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import (
    NemotronMTPSidecar,
    load_indexed_tensors,
    sidecar_quantization_settings,
)
from nemotron_mlx_mtp_bench import TRACE_FORMAT
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-mtp-binary-fit-v1"
SIDECAR_FORMAT = "nemotron-mlx-mtp-sidecar-v1"
UP_WEIGHT = "mtp.layers.1.mixer.switch_mlp.up_proj.weight"
DOWN_WEIGHT = "mtp.layers.1.mixer.switch_mlp.down_proj.weight"


def atomic_safetensors(path: Path, tensors: dict[str, mx.array], format_name: str) -> None:
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(str(temporary), tensors, metadata={"format": format_name})
    temporary.replace(path)


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_sidecar(sidecar_dir: Path) -> tuple[dict, dict[str, mx.array], Path, dict]:
    config = load_json(sidecar_dir / "config.json")
    runtime = config.get("nemotron_mtp_runtime", {})
    require(runtime.get("format") == SIDECAR_FORMAT, f"invalid MTP sidecar: {sidecar_dir}")
    index = load_json(sidecar_dir / "model.safetensors.index.json")
    shard_names = set(index.get("weight_map", {}).values())
    require(len(shard_names) == 1, f"MTP sidecar must occupy one shard: {sidecar_dir}")
    payload_path = sidecar_dir / next(iter(shard_names))
    tensors = mx.load(str(payload_path))
    require(set(tensors) == set(index["weight_map"]), f"MTP sidecar index mismatch: {sidecar_dir}")
    report = load_json(sidecar_dir / "nemotron_mtp_pack_report.json")
    require(
        report.get("format") == SIDECAR_FORMAT and report.get("status") == "complete",
        f"MTP sidecar report is incomplete: {sidecar_dir}",
    )
    return config, tensors, payload_path, report


def unpack_binary(weight: mx.array, width: int) -> mx.array:
    require(weight.dtype == mx.uint32, "one-bit expert weights must use packed uint32 storage")
    require(width % 32 == 0 and weight.shape[-1] == width // 32, "one-bit weight shape mismatch")
    shifts = mx.arange(32, dtype=mx.uint32)
    return ((weight[..., None] >> shifts) & 1).reshape(*weight.shape[:-1], width)


def reconstruct_binary(codes: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    group_size = codes.shape[-1] // scales.shape[-1]
    return (
        codes.reshape(*codes.shape[:-1], scales.shape[-1], group_size).astype(mx.float32)
        * scales.astype(mx.float32)[..., None]
        + biases.astype(mx.float32)[..., None]
    ).reshape(codes.shape)


def pack_binary(codes: mx.array) -> mx.array:
    require(codes.shape[-1] % 32 == 0, "binary codes do not align to uint32 words")
    shifts = mx.arange(32, dtype=mx.uint32)
    grouped = codes.astype(mx.uint32).reshape(*codes.shape[:-1], -1, 32)
    return mx.sum(grouped << shifts, axis=-1).astype(mx.uint32)


def weighted_projection_error(
    inputs: mx.array,
    target: mx.array,
    weight: mx.array,
    sample_weights: mx.array,
) -> float:
    difference = inputs.astype(mx.float32) @ weight.astype(mx.float32).T - target.astype(mx.float32)
    weights = sample_weights.astype(mx.float32)
    weights = weights / mx.maximum(mx.mean(weights), 1e-8)
    error = mx.sum(weights[:, None] * mx.square(difference))
    mx.eval(error)
    return float(error)


def fit_group_endpoints(
    student_input: mx.array,
    teacher_input: mx.array,
    teacher_weight: mx.array,
    codes: mx.array,
    initial_scales: mx.array,
    initial_biases: mx.array,
    sample_weights: mx.array,
    ridge: float,
    endpoint_margin: float = 0.25,
) -> tuple[mx.array, mx.array]:
    """Solve independent activation-weighted two-level fits for every weight group."""

    require(student_input.ndim == teacher_input.ndim == 2, "binary fit inputs must be matrices")
    require(student_input.shape[0] == teacher_input.shape[0], "binary fit row mismatch")
    require(teacher_weight.shape == codes.shape and teacher_weight.ndim == 2, "binary fit weight mismatch")
    require(student_input.shape[1] == codes.shape[1], "binary fit student width mismatch")
    require(teacher_input.shape[1] == teacher_weight.shape[1], "binary fit teacher width mismatch")
    require(initial_scales.shape == initial_biases.shape, "binary fit endpoint shape mismatch")
    require(initial_scales.shape[0] == codes.shape[0], "binary fit output width mismatch")
    require(sample_weights.shape == (student_input.shape[0],), "binary fit sample weight mismatch")
    require(ridge >= 0.0 and endpoint_margin >= 0.0, "binary fit regularizers must be nonnegative")
    groups = initial_scales.shape[-1]
    group_size = codes.shape[-1] // groups
    require(group_size * groups == codes.shape[-1], "binary fit group size mismatch")

    student_grouped = student_input.astype(mx.float32).reshape(-1, groups, group_size)
    teacher_grouped = teacher_input.astype(mx.float32).reshape(-1, groups, group_size)
    code_grouped = codes.astype(mx.float32).reshape(codes.shape[0], groups, group_size)
    weight_grouped = teacher_weight.astype(mx.float32).reshape(
        teacher_weight.shape[0], groups, group_size
    )
    scale_feature = mx.einsum("ngk,ogk->ong", student_grouped, code_grouped)
    bias_feature = mx.broadcast_to(
        mx.sum(student_grouped, axis=-1)[None, :, :], scale_feature.shape
    )
    target = mx.einsum("ngk,ogk->ong", teacher_grouped, weight_grouped)
    initial_scale = initial_scales.astype(mx.float32)
    initial_bias = initial_biases.astype(mx.float32)
    residual = (
        target
        - scale_feature * initial_scale[:, None, :]
        - bias_feature * initial_bias[:, None, :]
    )
    weights = sample_weights.astype(mx.float32)
    weights = weights / mx.maximum(mx.mean(weights), 1e-8)
    weights = weights[None, :, None]
    aa = mx.sum(weights * scale_feature * scale_feature, axis=1)
    ab = mx.sum(weights * scale_feature * bias_feature, axis=1)
    bb = mx.sum(weights * bias_feature * bias_feature, axis=1)
    ar = mx.sum(weights * scale_feature * residual, axis=1)
    br = mx.sum(weights * bias_feature * residual, axis=1)
    aa_regularized = aa * (1.0 + ridge) + 1e-8
    bb_regularized = bb * (1.0 + ridge) + 1e-8
    determinant = aa_regularized * bb_regularized - ab * ab
    valid = mx.abs(determinant) > 1e-12
    delta_scale = mx.where(
        valid,
        (ar * bb_regularized - br * ab) / determinant,
        0.0,
    )
    delta_bias = mx.where(
        valid,
        (aa_regularized * br - ab * ar) / determinant,
        0.0,
    )
    low = initial_bias + delta_bias
    high = initial_bias + initial_scale + delta_bias + delta_scale
    teacher_low = mx.min(weight_grouped, axis=-1)
    teacher_high = mx.max(weight_grouped, axis=-1)
    span = mx.maximum(teacher_high - teacher_low, 1e-8)
    lower_bound = teacher_low - endpoint_margin * span
    upper_bound = teacher_high + endpoint_margin * span
    low = mx.clip(low, lower_bound, upper_bound)
    high = mx.clip(high, lower_bound, upper_bound)
    high = mx.maximum(high, low)
    return (high - low).astype(mx.bfloat16), low.astype(mx.bfloat16)


def fit_projection_endpoints(
    student_input: mx.array,
    target_output: mx.array,
    teacher_weight: mx.array,
    codes: mx.array,
    initial_scales: mx.array,
    initial_biases: mx.array,
    sample_weights: mx.array,
    ridge: float,
    endpoint_margin: float = 0.25,
) -> tuple[mx.array, mx.array]:
    """Jointly fit every group endpoint against complete projection outputs."""

    require(student_input.ndim == target_output.ndim == 2, "projection fit inputs must be matrices")
    require(student_input.shape[0] == target_output.shape[0], "projection fit row mismatch")
    require(teacher_weight.shape == codes.shape and teacher_weight.ndim == 2, "projection fit weight mismatch")
    require(student_input.shape[1] == codes.shape[1], "projection fit input width mismatch")
    require(target_output.shape[1] == codes.shape[0], "projection fit output width mismatch")
    require(initial_scales.shape == initial_biases.shape, "projection fit endpoint shape mismatch")
    require(initial_scales.shape[0] == codes.shape[0], "projection fit endpoint width mismatch")
    require(sample_weights.shape == (student_input.shape[0],), "projection fit sample weight mismatch")
    require(ridge >= 0.0 and endpoint_margin >= 0.0, "projection fit regularizers must be nonnegative")
    groups = initial_scales.shape[-1]
    group_size = codes.shape[-1] // groups
    require(group_size * groups == codes.shape[-1], "projection fit group size mismatch")

    grouped_input = student_input.astype(mx.float32).reshape(-1, groups, group_size)
    grouped_codes = codes.astype(mx.float32).reshape(codes.shape[0], groups, group_size)
    scale_feature = mx.einsum("ngk,ogk->ong", grouped_input, grouped_codes)
    bias_feature = mx.broadcast_to(
        mx.sum(grouped_input, axis=-1)[None, :, :], scale_feature.shape
    )
    features = mx.concatenate((scale_feature, bias_feature), axis=-1)
    initial = mx.concatenate(
        (initial_scales.astype(mx.float32), initial_biases.astype(mx.float32)),
        axis=-1,
    )
    residual = target_output.astype(mx.float32).T - mx.sum(
        features * initial[:, None, :], axis=-1
    )
    weights = sample_weights.astype(mx.float32)
    weights = weights / mx.maximum(mx.mean(weights), 1e-8)
    weighted_features = features * weights[None, :, None]
    transposed = mx.swapaxes(features, -1, -2)
    gram = transposed @ weighted_features
    diagonal = mx.diagonal(gram, axis1=-2, axis2=-1)
    identity = mx.eye(features.shape[-1], dtype=mx.float32)[None, :, :]
    regularizer = identity * (ridge * mx.maximum(diagonal, 1e-8))[:, None, :]
    gram = gram + regularizer + identity * 1e-8
    rhs = transposed @ (weights[None, :, None] * residual[..., None])
    mx.eval(gram, rhs)
    delta = mx.linalg.solve(gram, rhs, stream=mx.cpu).squeeze(-1)
    mx.eval(delta)
    fitted = initial + delta
    low = fitted[:, groups:]
    high = low + fitted[:, :groups]
    grouped_teacher = teacher_weight.astype(mx.float32).reshape(
        teacher_weight.shape[0], groups, group_size
    )
    teacher_low = mx.min(grouped_teacher, axis=-1)
    teacher_high = mx.max(grouped_teacher, axis=-1)
    span = mx.maximum(teacher_high - teacher_low, 1e-8)
    lower_bound = teacher_low - endpoint_margin * span
    upper_bound = teacher_high + endpoint_margin * span
    low = mx.clip(low, lower_bound, upper_bound)
    high = mx.clip(high, lower_bound, upper_bound)
    high = mx.maximum(high, low)
    return (high - low).astype(mx.bfloat16), low.astype(mx.bfloat16)


def refine_projection_codes(
    inputs: mx.array,
    target_output: mx.array,
    teacher_weight: mx.array,
    codes: mx.array,
    scales: mx.array,
    biases: mx.array,
    sample_weights: mx.array,
    ridge: float,
    endpoint_margin: float,
    iterations: int,
    flip_fraction: float,
) -> tuple[mx.array, mx.array, mx.array, list[dict[str, float]]]:
    """Refine fixed-size binary codes using exact single-flip projection loss deltas."""

    require(iterations >= 0, "binary code iteration count must be nonnegative")
    require(0.0 < flip_fraction <= 0.25, "binary code flip fraction is out of range")
    current_codes = codes.astype(mx.uint32)
    current_scales = scales
    current_biases = biases
    current_weight = reconstruct_binary(current_codes, current_scales, current_biases)
    current_error = weighted_projection_error(
        inputs, target_output, current_weight, sample_weights
    )
    history = []
    weighted = sample_weights.astype(mx.float32)
    weighted = weighted / mx.maximum(mx.mean(weighted), 1e-8)
    input32 = inputs.astype(mx.float32)
    for iteration in range(iterations):
        residual = input32 @ current_weight.T - target_output.astype(mx.float32)
        correlation = (residual * weighted[:, None]).T @ input32
        input_energy = mx.sum(weighted[:, None] * mx.square(input32), axis=0)
        group_size = current_codes.shape[-1] // current_scales.shape[-1]
        expanded_scale = mx.broadcast_to(
            current_scales.astype(mx.float32)[..., None],
            (*current_scales.shape, group_size),
        ).reshape(current_codes.shape)
        direction = 1.0 - 2.0 * current_codes.astype(mx.float32)
        delta_weight = direction * expanded_scale
        delta_loss = (
            2.0 * delta_weight * correlation
            + mx.square(delta_weight) * input_energy[None, :]
        )
        flips_per_row = max(1, int(current_codes.shape[-1] * flip_fraction))
        threshold = mx.sort(delta_loss, axis=-1)[:, flips_per_row - 1 : flips_per_row]
        flip_mask = (delta_loss <= threshold) & (delta_loss < 0.0)
        candidate_codes = mx.where(flip_mask, 1 - current_codes, current_codes).astype(mx.uint32)
        candidate_scales, candidate_biases = fit_projection_endpoints(
            inputs,
            target_output,
            teacher_weight,
            candidate_codes,
            current_scales,
            current_biases,
            sample_weights,
            ridge,
            endpoint_margin,
        )
        candidate_weight = reconstruct_binary(
            candidate_codes, candidate_scales, candidate_biases
        )
        candidate_error = weighted_projection_error(
            inputs, target_output, candidate_weight, sample_weights
        )
        mx.eval(flip_mask, candidate_codes, candidate_scales, candidate_biases)
        flipped = int(mx.sum(flip_mask))
        history.append(
            {
                "iteration": iteration,
                "before_error2": current_error,
                "after_error2": candidate_error,
                "flipped_bits": flipped,
            }
        )
        if candidate_error >= current_error:
            break
        current_codes = candidate_codes
        current_scales = candidate_scales
        current_biases = candidate_biases
        current_weight = candidate_weight
        current_error = candidate_error
        mx.clear_cache()
    return current_codes, current_scales, current_biases, history


def context_signature(source_dir: Path, context_sidecar: Path, traces: list[Path]) -> dict:
    _, _, payload, report = load_sidecar(context_sidecar)
    return {
        "format": FORMAT,
        "source_dir": str(source_dir.resolve()),
        "source_revision": report["source_revision"],
        "context_sidecar": str(context_sidecar.resolve()),
        "context_sidecar_sha256": sha256_file(payload),
        "traces": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in traces
        ],
        "tool_sha256": sha256_file(Path(__file__)),
    }


def capture_contexts(
    source_dir: Path,
    context_sidecar: Path,
    traces: list[Path],
    cache_path: Path,
    operation_log: OperationLog,
) -> dict[str, mx.array]:
    signature = context_signature(source_dir, context_sidecar, traces)
    state_path = cache_path.with_suffix(".state.json")
    if cache_path.exists() and state_path.exists():
        state = load_json(state_path)
        if (
            state.get("signature") == signature
            and state.get("artifact_sha256") == sha256_file(cache_path)
        ):
            operation_log.write(f"mtp-binary-context-resume path={cache_path}")
            return mx.load(str(cache_path))

    operation_log.write(
        f"mtp-binary-context-start path={cache_path} traces={len(traces)}"
    )
    globals_ = load_indexed_tensors(source_dir, {"backbone.embeddings.weight"})
    embeddings = globals_["backbone.embeddings.weight"]
    model = NemotronMTPSidecar(
        context_sidecar,
        embeddings,
        ModelOptBF16Linear(embeddings),
    )
    latent_rows = []
    index_rows = []
    score_rows = []
    split_rows = []
    for split, trace_path in enumerate(traces):
        arrays, metadata = mx.load(str(trace_path), return_metadata=True)
        require(metadata.get("format") == TRACE_FORMAT, f"unsupported MTP trace: {trace_path}")
        required = {"target_hidden", "accepted_token_ids"}
        require(required <= set(arrays), f"MTP trace is incomplete: {trace_path}")
        rows = arrays["target_hidden"].shape[0]
        require(arrays["accepted_token_ids"].shape[0] == rows, "MTP trace row mismatch")
        for row in range(rows):
            attention = model._attention_step(
                arrays["target_hidden"][row],
                int(arrays["accepted_token_ids"][row]),
                cache=None,
            )
            hidden = mx.fast.rms_norm(
                attention,
                model.moe.norm_weight,
                model.moe.epsilon,
            )
            indices, scores = model.moe.route(hidden)
            latent = model.moe.fc1_latent(hidden)
            latent = latent.reshape(-1).astype(mx.float32)
            indices = indices.reshape(-1).astype(mx.int32)
            scores = scores.reshape(-1).astype(mx.float32)
            mx.eval(latent, indices, scores)
            latent_rows.append(latent)
            index_rows.append(indices)
            score_rows.append(scores)
            split_rows.append(split)
        operation_log.write(
            f"mtp-binary-context-trace path={trace_path} rows={rows}"
        )
    output = {
        "latent": mx.stack(latent_rows),
        "indices": mx.stack(index_rows),
        "scores": mx.stack(score_rows),
        "trace_indices": mx.array(split_rows, dtype=mx.int32),
    }
    mx.eval(*output.values())
    atomic_safetensors(cache_path, output, FORMAT)
    atomic_json(
        state_path,
        {
            "format": FORMAT,
            "status": "complete",
            "signature": signature,
            "artifact_sha256": sha256_file(cache_path),
            "rows": len(latent_rows),
        },
    )
    operation_log.write(
        f"mtp-binary-context-complete path={cache_path} rows={len(latent_rows)}"
    )
    del model, globals_, embeddings
    gc.collect()
    mx.clear_cache()
    return output


def expert_weight(
    config: dict,
    tensors: dict[str, mx.array],
    name: str,
    expert: int,
) -> mx.array:
    weight = tensors[name][expert]
    if weight.dtype == mx.bfloat16:
        return weight.astype(mx.float32)
    quantization = config.get("nemotron_mtp_runtime", {}).get("quantization")
    require(quantization is not None, f"quantized teacher tensor has no settings: {name}")
    settings = sidecar_quantization_settings(quantization, name)
    prefix = name[: -len(".weight")]
    return mx.dequantize(
        weight,
        tensors[f"{prefix}.scales"][expert],
        tensors.get(f"{prefix}.biases")[expert]
        if f"{prefix}.biases" in tensors
        else None,
        group_size=settings["group_size"],
        bits=settings["bits"],
        mode=settings["mode"],
        dtype=mx.float32,
    )


def expert_context_rows(
    contexts: dict[str, mx.array], expert: int
) -> tuple[np.ndarray, mx.array, mx.array]:
    indices = np.asarray(contexts["indices"])
    scores = np.asarray(contexts["scores"])
    rows, slots = np.where(indices == expert)
    require(np.unique(rows).size == rows.size, "an expert was routed more than once in one row")
    return (
        rows,
        contexts["latent"][mx.array(rows)],
        mx.array(scores[rows, slots], dtype=mx.float32),
    )


def expert_rows(contexts: dict[str, mx.array], expert: int) -> tuple[mx.array, mx.array]:
    _, latent, scores = expert_context_rows(contexts, expert)
    return latent, scores


def updated_expert_weight(
    config: dict,
    tensors: dict[str, mx.array],
    updates: dict[int, dict[str, mx.array]],
    projection: str,
    expert: int,
) -> mx.array:
    update = updates.get(expert)
    if update is None:
        name = UP_WEIGHT if projection == "up" else DOWN_WEIGHT
        return expert_weight(config, tensors, name, expert)
    scales = update[f"{projection}_scales"]
    biases = update[f"{projection}_biases"]
    width = int(scales.shape[-1]) * 128
    codes = unpack_binary(update[f"{projection}_weight"], width)
    return reconstruct_binary(codes, scales, biases)


def routed_aggregate(
    contexts: dict[str, mx.array],
    config: dict,
    tensors: dict[str, mx.array],
    updates: dict[int, dict[str, mx.array]] | None = None,
    operation_log: OperationLog | None = None,
    label: str = "aggregate",
) -> np.ndarray:
    """Evaluate the score-weighted expert sum without retaining dense expert weights."""

    updates = updates or {}
    rows = int(contexts["latent"].shape[0])
    latent_width = int(contexts["latent"].shape[-1])
    output = np.zeros((rows, latent_width), dtype=np.float32)
    routed = np.asarray(contexts["indices"])
    experts = sorted(set(int(value) for value in routed.reshape(-1)))
    for position, expert in enumerate(experts, 1):
        row_indices, latent, scores = expert_context_rows(contexts, expert)
        up = updated_expert_weight(config, tensors, updates, "up", expert)
        down = updated_expert_weight(config, tensors, updates, "down", expert)
        hidden = mx.square(mx.maximum(latent.astype(mx.float32) @ up.T, 0.0))
        expert_output = hidden @ down.T
        contribution = expert_output * scores[:, None]
        mx.eval(contribution)
        output[row_indices] += np.asarray(contribution)
        if operation_log is not None and (position % 64 == 0 or position == len(experts)):
            operation_log.write(
                f"mtp-binary-{label}-progress experts={position}/{len(experts)}"
            )
        del up, down, hidden, expert_output, contribution
        mx.clear_cache()
    return output


def aggregate_projection_target(
    current_aggregate: mx.array,
    teacher_aggregate: mx.array,
    current_expert_output: mx.array,
    scores: mx.array,
) -> mx.array:
    """Return the exact expert contribution needed to match the routed teacher sum."""

    require(current_aggregate.shape == teacher_aggregate.shape, "aggregate target shape mismatch")
    require(current_expert_output.shape == current_aggregate.shape, "expert output shape mismatch")
    require(scores.shape == (current_aggregate.shape[0],), "aggregate score shape mismatch")
    current_without_expert = current_aggregate - scores[:, None] * current_expert_output
    return teacher_aggregate - current_without_expert


def replaced_aggregate(
    current_aggregate: mx.array,
    current_expert_output: mx.array,
    candidate_expert_output: mx.array,
    scores: mx.array,
) -> mx.array:
    require(current_aggregate.shape == current_expert_output.shape, "aggregate replacement shape mismatch")
    require(candidate_expert_output.shape == current_expert_output.shape, "candidate output shape mismatch")
    require(scores.shape == (current_aggregate.shape[0],), "aggregate replacement score mismatch")
    return current_aggregate + scores[:, None] * (
        candidate_expert_output - current_expert_output
    )


def squared_error(candidate: mx.array, target: mx.array) -> float:
    require(candidate.shape == target.shape, "aggregate metric shape mismatch")
    value = mx.sum(mx.square(candidate.astype(mx.float32) - target.astype(mx.float32)))
    mx.eval(value)
    return float(value)


def expert_metric(
    latent: mx.array,
    sample_weights: mx.array,
    teacher_up: mx.array,
    teacher_down: mx.array,
    student_up: mx.array,
    student_down: mx.array,
) -> tuple[float, float]:
    if latent.shape[0] == 0:
        return 0.0, 0.0
    teacher_hidden = mx.square(mx.maximum(latent.astype(mx.float32) @ teacher_up.T, 0.0))
    teacher = teacher_hidden @ teacher_down.T
    student_hidden = mx.square(mx.maximum(latent.astype(mx.float32) @ student_up.T, 0.0))
    student = student_hidden @ student_down.T
    weight = sample_weights.astype(mx.float32)[:, None]
    error2 = mx.sum(weight * mx.square(student - teacher))
    reference2 = mx.sum(weight * mx.square(teacher))
    mx.eval(error2, reference2)
    return float(error2), float(reference2)


def progress_payload(updates: dict[int, dict[str, mx.array]]) -> dict[str, mx.array]:
    payload = {}
    for expert, tensors in sorted(updates.items()):
        for name, value in tensors.items():
            payload[f"expert_{expert:03d}.{name}"] = value
    return payload


def load_progress(path: Path) -> dict[int, dict[str, mx.array]]:
    if not path.exists():
        return {}
    updates: dict[int, dict[str, mx.array]] = {}
    for name, value in mx.load(str(path)).items():
        expert_name, tensor_name = name.split(".", 1)
        expert = int(expert_name.removeprefix("expert_"))
        updates.setdefault(expert, {})[tensor_name] = value
    return updates


def save_progress(path: Path, updates: dict[int, dict[str, mx.array]]) -> None:
    atomic_safetensors(path, progress_payload(updates), FORMAT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--base-sidecar", required=True, type=Path)
    parser.add_argument("--teacher-sidecar", required=True, type=Path)
    parser.add_argument("--context-sidecar", required=True, type=Path)
    parser.add_argument("--train-trace", required=True, action="append", type=Path)
    parser.add_argument("--validation-trace", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--context-cache-dir",
        type=Path,
        help="share immutable routed-context captures across compatible fit jobs",
    )
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--endpoint-margin", type=float, default=0.25)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--max-experts", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument(
        "--fit-objective",
        choices=("projection", "group", "aggregate"),
        default="projection",
    )
    parser.add_argument("--code-iterations", type=int, default=0)
    parser.add_argument("--flip-fraction", type=float, default=0.005)
    parser.add_argument(
        "--allow-validation-regression",
        action="store_true",
        help="retain a fitted expert even when its local held-out error worsens",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.ridge >= 0.0, "ridge must be nonnegative")
        require(args.endpoint_margin >= 0.0, "endpoint margin must be nonnegative")
        require(args.min_samples > 0, "minimum sample count must be positive")
        require(args.max_experts >= 0, "maximum expert count must be nonnegative")
        require(args.checkpoint_every > 0, "checkpoint interval must be positive")
        require(args.code_iterations >= 0, "code iteration count must be nonnegative")
        require(0.0 < args.flip_fraction <= 0.25, "code flip fraction is out of range")
        require(
            args.code_iterations == 0 or args.fit_objective in ("projection", "aggregate"),
            "binary code refinement requires a joint projection objective",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        context_cache_dir = args.context_cache_dir or args.output_dir
        context_cache_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "binary-fit.log")
        base_config, base_tensors, base_payload, base_report = load_sidecar(args.base_sidecar)
        teacher_config, teacher_tensors, teacher_payload, teacher_report = load_sidecar(
            args.teacher_sidecar
        )
        context_config, _, context_payload, context_report = load_sidecar(args.context_sidecar)
        revision = base_report["source_revision"]
        require(
            teacher_report["source_revision"] == revision
            and context_report["source_revision"] == revision,
            "MTP binary fit sidecar revisions differ",
        )
        expert_count = base_config["n_routed_experts"]
        require(
            teacher_config["n_routed_experts"] == expert_count
            and context_config["n_routed_experts"] == expert_count,
            "MTP binary fit expert counts differ",
        )
        quantization = base_config["nemotron_mtp_runtime"].get("quantization")
        require(quantization is not None, "base MTP sidecar is not quantized")
        for name in (UP_WEIGHT, DOWN_WEIGHT):
            settings = sidecar_quantization_settings(quantization, name)
            require(
                settings["mode"] == "affine"
                and settings["bits"] == 1
                and settings["group_size"] == 128,
                f"base tensor is not one-bit affine: {name}",
            )

        job = {
            "format": FORMAT,
            "source_dir": str(args.source_dir.resolve()),
            "source_revision": revision,
            "base_sidecar": str(args.base_sidecar.resolve()),
            "base_sidecar_sha256": sha256_file(base_payload),
            "teacher_sidecar": str(args.teacher_sidecar.resolve()),
            "teacher_sidecar_sha256": sha256_file(teacher_payload),
            "context_sidecar": str(args.context_sidecar.resolve()),
            "context_sidecar_sha256": sha256_file(context_payload),
            "train_traces": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.train_trace
            ],
            "validation_traces": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.validation_trace
            ],
            "ridge": args.ridge,
            "endpoint_margin": args.endpoint_margin,
            "min_samples": args.min_samples,
            "max_experts": args.max_experts,
            "fit_objective": args.fit_objective,
            "code_iterations": args.code_iterations,
            "flip_fraction": args.flip_fraction,
            "allow_validation_regression": args.allow_validation_regression,
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": getattr(mx, "__version__", "unknown"),
        }
        job_hash = canonical_hash(job)
        state_path = args.output_dir / "fit-state.json"
        progress_path = args.output_dir / "fit-progress.safetensors"
        state = load_json(state_path) if state_path.exists() else None
        if state is not None:
            require(state.get("job_hash") == job_hash, "existing MTP binary fit job differs")
            if state.get("status") == "complete":
                output_path = args.output_dir / "mtp.safetensors"
                require(
                    state.get("artifact_sha256") == sha256_file(output_path),
                    "completed MTP binary fit artifact changed",
                )
                operation_log.write(f"mtp-binary-fit-already-complete job={job_hash}")
                return 0
        else:
            state = {
                "format": FORMAT,
                "status": "running",
                "job": job,
                "job_hash": job_hash,
                "completed_experts": [],
                "expert_metrics": [],
            }
            atomic_json(state_path, state)

        train = capture_contexts(
            args.source_dir,
            args.context_sidecar,
            args.train_trace,
            context_cache_dir / "train-contexts.safetensors",
            operation_log,
        )
        validation = capture_contexts(
            args.source_dir,
            args.context_sidecar,
            args.validation_trace,
            context_cache_dir / "validation-contexts.safetensors",
            operation_log,
        )
        train_indices = np.asarray(train["indices"])
        train_scores = np.asarray(train["scores"])
        counts = np.bincount(train_indices.reshape(-1), minlength=expert_count)
        masses = np.bincount(
            train_indices.reshape(-1),
            weights=train_scores.reshape(-1),
            minlength=expert_count,
        )
        ordered = sorted(range(expert_count), key=lambda expert: (-masses[expert], -counts[expert], expert))
        selected = [expert for expert in ordered if counts[expert] >= args.min_samples]
        if args.max_experts:
            selected = selected[: args.max_experts]
        require(selected, "no routed experts satisfy the binary fit constraints")
        updates = load_progress(progress_path)
        completed = set(state["completed_experts"])
        require(completed == set(updates), "MTP binary fit progress/state mismatch")
        metric_rows = {row["expert"]: row for row in state["expert_metrics"]}
        operation_log.write(
            f"mtp-binary-fit-start job={job_hash} selected={len(selected)} resumed={len(completed)}"
        )

        base_prefixes = {
            "up": UP_WEIGHT[: -len(".weight")],
            "down": DOWN_WEIGHT[: -len(".weight")],
        }
        aggregate_train_teacher = None
        aggregate_train_current = None
        aggregate_validation_teacher = None
        aggregate_validation_current = None
        aggregate_validation_baseline_error2 = None
        aggregate_validation_error2 = None
        aggregate_validation_reference2 = None
        if args.fit_objective == "aggregate":
            operation_log.write("mtp-binary-aggregate-initialize")
            aggregate_train_teacher = routed_aggregate(
                train,
                teacher_config,
                teacher_tensors,
                operation_log=operation_log,
                label="aggregate-train-teacher",
            )
            aggregate_validation_teacher = routed_aggregate(
                validation,
                teacher_config,
                teacher_tensors,
                operation_log=operation_log,
                label="aggregate-validation-teacher",
            )
            aggregate_train_base = routed_aggregate(
                train,
                base_config,
                base_tensors,
                operation_log=operation_log,
                label="aggregate-train-base",
            )
            aggregate_validation_base = routed_aggregate(
                validation,
                base_config,
                base_tensors,
                operation_log=operation_log,
                label="aggregate-validation-base",
            )
            if updates:
                aggregate_train_current = routed_aggregate(
                    train,
                    base_config,
                    base_tensors,
                    updates,
                    operation_log,
                    "aggregate-train-resume",
                )
                aggregate_validation_current = routed_aggregate(
                    validation,
                    base_config,
                    base_tensors,
                    updates,
                    operation_log,
                    "aggregate-validation-resume",
                )
            else:
                aggregate_train_current = aggregate_train_base.copy()
                aggregate_validation_current = aggregate_validation_base.copy()
            aggregate_validation_baseline_error2 = float(
                np.sum(
                    np.square(
                        aggregate_validation_base.astype(np.float64)
                        - aggregate_validation_teacher.astype(np.float64)
                    )
                )
            )
            aggregate_validation_error2 = float(
                np.sum(
                    np.square(
                        aggregate_validation_current.astype(np.float64)
                        - aggregate_validation_teacher.astype(np.float64)
                    )
                )
            )
            aggregate_validation_reference2 = float(
                np.sum(np.square(aggregate_validation_teacher.astype(np.float64)))
            )
            expected_baseline = state.get("aggregate_validation_baseline_error2")
            if expected_baseline is not None:
                require(
                    abs(expected_baseline - aggregate_validation_baseline_error2)
                    <= max(aggregate_validation_baseline_error2, 1.0) * 1e-7,
                    "resumed aggregate baseline metric changed",
                )
            state["aggregate_validation_baseline_error2"] = (
                aggregate_validation_baseline_error2
            )
            operation_log.write(
                "mtp-binary-aggregate-ready "
                f"baseline_relative_l2="
                f"{(aggregate_validation_baseline_error2 / max(aggregate_validation_reference2, 1e-30)) ** 0.5:.9g} "
                f"current_relative_l2="
                f"{(aggregate_validation_error2 / max(aggregate_validation_reference2, 1e-30)) ** 0.5:.9g}"
            )
        for position, expert in enumerate(selected, 1):
            if expert in completed:
                continue
            started = time.perf_counter()
            train_row_indices, train_latent, train_weight = expert_context_rows(train, expert)
            validation_row_indices, validation_latent, validation_weight = expert_context_rows(
                validation, expert
            )
            teacher_up = expert_weight(teacher_config, teacher_tensors, UP_WEIGHT, expert)
            teacher_down = expert_weight(teacher_config, teacher_tensors, DOWN_WEIGHT, expert)
            up_codes = unpack_binary(base_tensors[UP_WEIGHT][expert], teacher_up.shape[-1])
            down_codes = unpack_binary(base_tensors[DOWN_WEIGHT][expert], teacher_down.shape[-1])
            up_codes0 = up_codes
            down_codes0 = down_codes
            up_scale0 = base_tensors[f"{base_prefixes['up']}.scales"][expert]
            up_bias0 = base_tensors[f"{base_prefixes['up']}.biases"][expert]
            down_scale0 = base_tensors[f"{base_prefixes['down']}.scales"][expert]
            down_bias0 = base_tensors[f"{base_prefixes['down']}.biases"][expert]
            base_up = reconstruct_binary(up_codes0, up_scale0, up_bias0)
            base_down = reconstruct_binary(down_codes0, down_scale0, down_bias0)
            if args.fit_objective == "aggregate":
                up_scale, up_bias = up_scale0, up_bias0
            elif args.fit_objective == "projection":
                teacher_up_output = train_latent.astype(mx.float32) @ teacher_up.T
                up_scale, up_bias = fit_projection_endpoints(
                    train_latent,
                    teacher_up_output,
                    teacher_up,
                    up_codes,
                    up_scale0,
                    up_bias0,
                    train_weight,
                    args.ridge,
                    args.endpoint_margin,
                )
            else:
                up_scale, up_bias = fit_group_endpoints(
                    train_latent,
                    train_latent,
                    teacher_up,
                    up_codes,
                    up_scale0,
                    up_bias0,
                    train_weight,
                    args.ridge,
                    args.endpoint_margin,
                )
            up_history = []
            if args.code_iterations and args.fit_objective != "aggregate":
                teacher_up_output = train_latent.astype(mx.float32) @ teacher_up.T
                up_codes, up_scale, up_bias, up_history = refine_projection_codes(
                    train_latent,
                    teacher_up_output,
                    teacher_up,
                    up_codes,
                    up_scale,
                    up_bias,
                    train_weight,
                    args.ridge,
                    args.endpoint_margin,
                    args.code_iterations,
                    args.flip_fraction,
                )
            student_up = reconstruct_binary(up_codes, up_scale, up_bias)
            teacher_hidden = mx.square(
                mx.maximum(train_latent.astype(mx.float32) @ teacher_up.T, 0.0)
            )
            student_hidden = mx.square(
                mx.maximum(train_latent.astype(mx.float32) @ student_up.T, 0.0)
            )
            if args.fit_objective == "aggregate":
                require(
                    aggregate_train_current is not None
                    and aggregate_train_teacher is not None,
                    "aggregate training state is unavailable",
                )
                current_train_rows = mx.array(
                    aggregate_train_current[train_row_indices], dtype=mx.float32
                )
                teacher_train_rows = mx.array(
                    aggregate_train_teacher[train_row_indices], dtype=mx.float32
                )
                current_train_output = student_hidden @ base_down.T
                target_contribution = aggregate_projection_target(
                    current_train_rows,
                    teacher_train_rows,
                    current_train_output,
                    train_weight,
                )
                scaled_hidden = student_hidden * train_weight[:, None]
                down_scale, down_bias = fit_projection_endpoints(
                    scaled_hidden,
                    target_contribution,
                    teacher_down,
                    down_codes,
                    down_scale0,
                    down_bias0,
                    mx.ones((scaled_hidden.shape[0],), dtype=mx.float32),
                    args.ridge,
                    args.endpoint_margin,
                )
            elif args.fit_objective == "projection":
                teacher_down_output = teacher_hidden @ teacher_down.T
                down_scale, down_bias = fit_projection_endpoints(
                    student_hidden,
                    teacher_down_output,
                    teacher_down,
                    down_codes,
                    down_scale0,
                    down_bias0,
                    train_weight,
                    args.ridge,
                    args.endpoint_margin,
                )
            else:
                down_scale, down_bias = fit_group_endpoints(
                    student_hidden,
                    teacher_hidden,
                    teacher_down,
                    down_codes,
                    down_scale0,
                    down_bias0,
                    train_weight,
                    args.ridge,
                    args.endpoint_margin,
                )
            down_history = []
            if args.code_iterations:
                if args.fit_objective == "aggregate":
                    code_inputs = scaled_hidden
                    code_target = target_contribution
                    code_weights = mx.ones((scaled_hidden.shape[0],), dtype=mx.float32)
                else:
                    code_inputs = student_hidden
                    code_target = teacher_hidden @ teacher_down.T
                    code_weights = train_weight
                down_codes, down_scale, down_bias, down_history = refine_projection_codes(
                    code_inputs,
                    code_target,
                    teacher_down,
                    down_codes,
                    down_scale,
                    down_bias,
                    code_weights,
                    args.ridge,
                    args.endpoint_margin,
                    args.code_iterations,
                    args.flip_fraction,
                )
            fitted_down = reconstruct_binary(down_codes, down_scale, down_bias)
            if args.fit_objective == "aggregate":
                require(
                    aggregate_validation_current is not None
                    and aggregate_validation_teacher is not None
                    and aggregate_validation_error2 is not None
                    and aggregate_validation_reference2 is not None,
                    "aggregate validation state is unavailable",
                )
                before_error2 = aggregate_validation_error2
                reference2 = aggregate_validation_reference2
                after_error2 = before_error2
                current_validation_output = None
                candidate_validation_output = None
                candidate_validation_rows = None
                if validation_latent.shape[0] > 0:
                    validation_hidden = mx.square(
                        mx.maximum(validation_latent.astype(mx.float32) @ base_up.T, 0.0)
                    )
                    current_validation_output = validation_hidden @ base_down.T
                    candidate_validation_output = validation_hidden @ fitted_down.T
                    current_validation_rows = mx.array(
                        aggregate_validation_current[validation_row_indices],
                        dtype=mx.float32,
                    )
                    teacher_validation_rows = mx.array(
                        aggregate_validation_teacher[validation_row_indices],
                        dtype=mx.float32,
                    )
                    candidate_validation_rows = replaced_aggregate(
                        current_validation_rows,
                        current_validation_output,
                        candidate_validation_output,
                        validation_weight,
                    )
                    old_rows_error2 = squared_error(
                        current_validation_rows, teacher_validation_rows
                    )
                    new_rows_error2 = squared_error(
                        candidate_validation_rows, teacher_validation_rows
                    )
                    after_error2 = max(
                        0.0,
                        before_error2 - old_rows_error2 + new_rows_error2,
                    )
                accepted = validation_latent.shape[0] > 0 and (
                    args.allow_validation_regression or after_error2 < before_error2
                )
            else:
                before_error2, reference2 = expert_metric(
                    validation_latent,
                    validation_weight,
                    teacher_up,
                    teacher_down,
                    base_up,
                    base_down,
                )
                after_error2, after_reference2 = expert_metric(
                    validation_latent,
                    validation_weight,
                    teacher_up,
                    teacher_down,
                    student_up,
                    fitted_down,
                )
                require(
                    abs(after_reference2 - reference2) <= max(reference2, 1.0) * 1e-5,
                    "metric reference drift",
                )
                accepted = validation_latent.shape[0] > 0 and (
                    args.allow_validation_regression or after_error2 < before_error2
                )
            if not accepted:
                up_codes = up_codes0
                down_codes = down_codes0
                up_scale, up_bias = up_scale0, up_bias0
                down_scale, down_bias = down_scale0, down_bias0
            elif args.fit_objective == "aggregate":
                candidate_train_output = student_hidden @ fitted_down.T
                current_train_rows = replaced_aggregate(
                    current_train_rows,
                    current_train_output,
                    candidate_train_output,
                    train_weight,
                )
                mx.eval(current_train_rows)
                aggregate_train_current[train_row_indices] = np.asarray(current_train_rows)
                require(candidate_validation_rows is not None, "accepted aggregate has no validation rows")
                mx.eval(candidate_validation_rows)
                aggregate_validation_current[validation_row_indices] = np.asarray(
                    candidate_validation_rows
                )
                aggregate_validation_error2 = after_error2
            deployed_after_error2 = after_error2 if accepted else before_error2
            updates[expert] = {
                "up_weight": pack_binary(up_codes),
                "up_scales": up_scale,
                "up_biases": up_bias,
                "down_weight": pack_binary(down_codes),
                "down_scales": down_scale,
                "down_biases": down_bias,
            }
            row = {
                "expert": expert,
                "train_samples": int(train_latent.shape[0]),
                "validation_samples": int(validation_latent.shape[0]),
                "route_score_mass": float(masses[expert]),
                "validation_before_error2": before_error2,
                "validation_candidate_after_error2": after_error2,
                "validation_after_error2": deployed_after_error2,
                "validation_reference2": reference2,
                "accepted": accepted,
                "up_code_history": up_history,
                "down_code_history": down_history,
            }
            metric_rows[expert] = row
            completed.add(expert)
            state["completed_experts"] = sorted(completed)
            state["expert_metrics"] = [metric_rows[key] for key in sorted(metric_rows)]
            operation_log.write(
                f"mtp-binary-fit-expert expert={expert} train={train_latent.shape[0]} "
                f"validation={validation_latent.shape[0]} "
                f"before_rel={before_error2 / max(reference2, 1e-30):.9g} "
                f"after_rel={after_error2 / max(reference2, 1e-30):.9g} "
                f"accepted={accepted} "
                f"elapsed={time.perf_counter() - started:.3f}s"
            )
            if position % args.checkpoint_every == 0 or position == len(selected):
                save_progress(progress_path, updates)
                atomic_json(state_path, state)
                operation_log.write(
                    f"mtp-binary-fit-checkpoint completed={len(completed)} path={progress_path}"
                )
            mx.clear_cache()

        require(set(selected) <= completed, "MTP binary fit did not complete every selected expert")
        output = dict(base_tensors)
        for projection, weight_name in (("up", UP_WEIGHT), ("down", DOWN_WEIGHT)):
            prefix = base_prefixes[projection]
            output[weight_name] = mx.stack(
                [
                    updates.get(expert, {}).get(
                        f"{projection}_weight",
                        base_tensors[weight_name][expert],
                    )
                    for expert in range(expert_count)
                ]
            )
            output[f"{prefix}.scales"] = mx.stack(
                [
                    updates.get(expert, {}).get(
                        f"{projection}_scales",
                        base_tensors[f"{prefix}.scales"][expert],
                    )
                    for expert in range(expert_count)
                ]
            )
            output[f"{prefix}.biases"] = mx.stack(
                [
                    updates.get(expert, {}).get(
                        f"{projection}_biases",
                        base_tensors[f"{prefix}.biases"][expert],
                    )
                    for expert in range(expert_count)
                ]
            )
        mx.eval(output[f"{base_prefixes['up']}.scales"], output[f"{base_prefixes['up']}.biases"])
        mx.eval(output[f"{base_prefixes['down']}.scales"], output[f"{base_prefixes['down']}.biases"])
        output_path = args.output_dir / "mtp.safetensors"
        atomic_safetensors(output_path, output, "nemotron-mtp-sidecar-quant-v1")
        loaded = mx.load(str(output_path))
        require(set(loaded) == set(output), "fitted MTP sidecar tensor set mismatch")
        modified = {
            UP_WEIGHT,
            DOWN_WEIGHT,
            f"{base_prefixes['up']}.scales",
            f"{base_prefixes['up']}.biases",
            f"{base_prefixes['down']}.scales",
            f"{base_prefixes['down']}.biases",
        }
        for name in sorted(output):
            expected = output[name] if name in modified else base_tensors[name]
            require(bool(mx.array_equal(loaded[name], expected)), f"fitted MTP tensor mismatch: {name}")

        output_config = copy.deepcopy(base_config)
        fit_provenance = {
            "format": FORMAT,
            "job_hash": job_hash,
            "teacher_sidecar_sha256": job["teacher_sidecar_sha256"],
            "context_sidecar_sha256": job["context_sidecar_sha256"],
            "train_trace_sha256": [row["sha256"] for row in job["train_traces"]],
            "ridge": args.ridge,
            "endpoint_margin": args.endpoint_margin,
            "fit_objective": args.fit_objective,
            "code_iterations": args.code_iterations,
            "flip_fraction": args.flip_fraction,
            "fitted_experts": sorted(selected),
            "base_binary_fit": base_config["nemotron_mtp_runtime"]["quantization"].get(
                "binary_fit"
            ),
        }
        output_config["nemotron_mtp_runtime"]["quantization"]["binary_fit"] = fit_provenance
        atomic_json(args.output_dir / "config.json", output_config)
        payload_bytes = sum(value.nbytes for value in loaded.values())
        atomic_json(
            args.output_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": payload_bytes},
                "weight_map": {name: output_path.name for name in loaded},
            },
        )
        rows = [metric_rows[expert] for expert in selected]
        if args.fit_objective == "aggregate":
            require(
                aggregate_validation_baseline_error2 is not None
                and aggregate_validation_error2 is not None
                and aggregate_validation_reference2 is not None,
                "aggregate final metric is unavailable",
            )
            before_error2 = aggregate_validation_baseline_error2
            after_error2 = aggregate_validation_error2
            reference2 = aggregate_validation_reference2
        else:
            before_error2 = sum(row["validation_before_error2"] for row in rows)
            after_error2 = sum(row["validation_after_error2"] for row in rows)
            reference2 = sum(row["validation_reference2"] for row in rows)
        output_report = copy.deepcopy(base_report)
        for stale in ("full_tensor_error", "max_relative_l2", "max_abs"):
            output_report.pop(stale, None)
        output_report.update(
            {
                "status": "complete",
                "payload_bytes": payload_bytes,
                "payload_gib": payload_bytes / 2**30,
                "source_sidecar_sha256": job["base_sidecar_sha256"],
                "artifact_sha256": sha256_file(output_path),
                "binary_fit": fit_provenance,
                "binary_fit_validation": {
                    "experts": len(selected),
                    "accepted_experts": sum(row["accepted"] for row in rows),
                    "before_relative_l2": (before_error2 / max(reference2, 1e-30)) ** 0.5,
                    "after_relative_l2": (after_error2 / max(reference2, 1e-30)) ** 0.5,
                    "expert_metrics": rows,
                },
            }
        )
        atomic_json(args.output_dir / "nemotron_mtp_pack_report.json", output_report)
        state.update(
            {
                "status": "complete",
                "artifact_sha256": output_report["artifact_sha256"],
                "report_sha256": sha256_file(args.output_dir / "nemotron_mtp_pack_report.json"),
            }
        )
        atomic_json(state_path, state)
        progress_path.unlink(missing_ok=True)
        operation_log.write(
            f"mtp-binary-fit-complete experts={len(selected)} "
            f"before_relative_l2={output_report['binary_fit_validation']['before_relative_l2']:.9g} "
            f"after_relative_l2={output_report['binary_fit_validation']['after_relative_l2']:.9g} "
            f"sha256={output_report['artifact_sha256']}"
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, KeyError) as exc:
        if operation_log is not None:
            operation_log.write(f"mtp-binary-fit-failed error={exc}")
        print(f"nemotron MTP binary fit error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

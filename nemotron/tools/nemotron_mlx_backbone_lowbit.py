#!/usr/bin/env python3
"""Activation-fit low-bit Nemotron backbone experts from BF16 teacher weights."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from nemotron_metadata import require


FORMAT = "nemotron-backbone-lowbit-expert-v1"


@dataclass
class AffineWeight:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int
    rows: int
    columns: int

    def validate(self) -> None:
        require(self.bits in (1, 2, 3, 4), "unsupported affine bit width")
        require(self.group_size > 0 and self.columns % self.group_size == 0, "invalid affine group size")
        require(self.weight.dtype == mx.uint32, "affine codes must use packed uint32 storage")
        require(self.scales.dtype == self.biases.dtype == mx.bfloat16, "affine endpoints must be BF16")
        require(self.weight.shape == (self.rows, self.columns * self.bits // 32), "packed affine shape mismatch")
        require(
            self.scales.shape == self.biases.shape == (self.rows, self.columns // self.group_size),
            "affine endpoint shape mismatch",
        )

    @property
    def payload_bytes(self) -> int:
        return self.weight.nbytes + self.scales.nbytes + self.biases.nbytes


@dataclass
class BinaryExpert:
    up: AffineWeight
    down: AffineWeight

    def validate(self) -> None:
        self.up.validate()
        self.down.validate()
        require(self.up.bits == self.down.bits == 1, "binary expert has non-binary weights")
        require(self.up.columns == self.down.rows, "binary expert latent width mismatch")
        require(self.up.rows == self.down.columns, "binary expert hidden width mismatch")

    @property
    def payload_bytes(self) -> int:
        return self.up.payload_bytes + self.down.payload_bytes


def pack_codes(codes: mx.array, bits: int) -> mx.array:
    require(bits in (1, 2, 3, 4), "unsupported packed bit width")
    require(codes.ndim == 2, "packed codes must be a matrix")
    require((codes.shape[1] * bits) % 32 == 0, "code rows do not align to uint32")
    require(bool(mx.all((codes >= 0) & (codes < (1 << bits)))), "code value is out of range")
    if bits in (1, 2, 4):
        values_per_word = 32 // bits
        shifts = mx.arange(values_per_word, dtype=mx.uint32) * bits
        grouped = codes.astype(mx.uint32).reshape(codes.shape[0], -1, values_per_word)
        return mx.sum(grouped << shifts, axis=-1).astype(mx.uint32)

    # Three-bit rows can cross word boundaries. Packing through uint64 keeps
    # the definition simple and is only used while building an artifact.
    rows = []
    for row in codes.tolist():
        words = []
        accumulator = 0
        occupied = 0
        for code in row:
            accumulator |= int(code) << occupied
            occupied += 3
            if occupied >= 32:
                words.append(accumulator & 0xFFFFFFFF)
                accumulator >>= 32
                occupied -= 32
        if occupied:
            words.append(accumulator)
        require(len(words) == codes.shape[1] * 3 // 32, "three-bit packing mismatch")
        rows.append(words)
    return mx.array(rows, dtype=mx.uint32)


def unpack_codes(weight: mx.array, columns: int, bits: int) -> mx.array:
    require(weight.dtype == mx.uint32 and weight.ndim == 2, "invalid packed affine codes")
    require(weight.shape[1] == columns * bits // 32, "packed affine row width mismatch")
    mask = (1 << bits) - 1
    if bits in (1, 2, 4):
        values_per_word = 32 // bits
        shifts = mx.arange(values_per_word, dtype=mx.uint32) * bits
        return ((weight[..., None] >> shifts) & mask).reshape(weight.shape[0], columns)

    rows = []
    for words in weight.tolist():
        values = []
        accumulator = 0
        occupied = 0
        word_index = 0
        while len(values) < columns:
            while occupied < 3:
                accumulator |= int(words[word_index]) << occupied
                occupied += 32
                word_index += 1
            values.append(accumulator & mask)
            accumulator >>= 3
            occupied -= 3
        rows.append(values)
    return mx.array(rows, dtype=mx.uint32)


def reconstruct_affine(codes: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    require(codes.ndim == 2, "affine reconstruction requires a code matrix")
    require(scales.shape == biases.shape and scales.shape[0] == codes.shape[0], "endpoint shape mismatch")
    group_size = codes.shape[1] // scales.shape[1]
    require(group_size * scales.shape[1] == codes.shape[1], "code groups do not divide row width")
    return (
        codes.reshape(codes.shape[0], scales.shape[1], group_size).astype(mx.float32)
        * scales.astype(mx.float32)[..., None]
        + biases.astype(mx.float32)[..., None]
    ).reshape(codes.shape)


def dequantize_affine(weight: AffineWeight) -> mx.array:
    weight.validate()
    if weight.bits > 1:
        return mx.dequantize(
            weight.weight,
            weight.scales,
            weight.biases,
            group_size=weight.group_size,
            bits=weight.bits,
            mode="affine",
            dtype=mx.float32,
        )
    return reconstruct_affine(
        unpack_codes(weight.weight, weight.columns, weight.bits),
        weight.scales,
        weight.biases,
    )


def kmeans_affine_codes(
    value: mx.array,
    bits: int,
    group_size: int,
    iterations: int = 8,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fit uniformly spaced affine levels and return unpacked integer codes."""

    require(value.ndim == 2, "affine quantization requires a matrix")
    require(bits in (1, 2, 3, 4), "unsupported affine bit width")
    require(value.shape[1] % group_size == 0, "affine group size does not divide matrix")
    require(iterations > 0, "affine quantization iterations must be positive")
    levels = (1 << bits) - 1
    grouped = value.astype(mx.float32).reshape(value.shape[0], -1, group_size)
    low = mx.min(grouped, axis=-1)
    high = mx.max(grouped, axis=-1)
    scale = mx.maximum((high - low) / levels, 1e-8)
    bias = low
    codes = mx.zeros(grouped.shape, dtype=mx.uint32)
    for _ in range(iterations):
        codes = mx.clip(mx.round((grouped - bias[..., None]) / scale[..., None]), 0, levels).astype(
            mx.uint32
        )
        code32 = codes.astype(mx.float32)
        count = float(group_size)
        sum_code = mx.sum(code32, axis=-1)
        sum_code2 = mx.sum(mx.square(code32), axis=-1)
        sum_value = mx.sum(grouped, axis=-1)
        sum_code_value = mx.sum(code32 * grouped, axis=-1)
        determinant = count * sum_code2 - mx.square(sum_code)
        valid = mx.abs(determinant) > 1e-12
        fitted_scale = (count * sum_code_value - sum_code * sum_value) / mx.maximum(
            determinant, 1e-12
        )
        fitted_bias = (sum_value - fitted_scale * sum_code) / count
        scale = mx.where(valid, mx.maximum(fitted_scale, 1e-8), scale)
        bias = mx.where(valid, fitted_bias, bias)
    return (
        codes.reshape(value.shape).astype(mx.uint32),
        scale.astype(mx.bfloat16),
        bias.astype(mx.bfloat16),
    )


def fit_projection_endpoints(
    inputs: mx.array,
    target_output: mx.array,
    teacher_weight: mx.array,
    codes: mx.array,
    initial_scales: mx.array,
    initial_biases: mx.array,
    sample_weights: mx.array,
    ridge: float = 1e-3,
    endpoint_margin: float = 0.25,
) -> tuple[mx.array, mx.array]:
    """Fit every group endpoint against complete activation projections."""

    require(inputs.ndim == target_output.ndim == 2, "projection fit requires matrices")
    require(inputs.shape[0] == target_output.shape[0], "projection fit row mismatch")
    require(teacher_weight.shape == codes.shape and teacher_weight.ndim == 2, "teacher/code mismatch")
    require(inputs.shape[1] == codes.shape[1], "projection input width mismatch")
    require(target_output.shape[1] == codes.shape[0], "projection output width mismatch")
    require(initial_scales.shape == initial_biases.shape, "projection endpoint mismatch")
    require(initial_scales.shape[0] == codes.shape[0], "projection endpoint rows mismatch")
    require(sample_weights.shape == (inputs.shape[0],), "projection sample-weight mismatch")
    require(ridge >= 0.0 and endpoint_margin >= 0.0, "projection regularizers must be nonnegative")
    groups = initial_scales.shape[1]
    group_size = codes.shape[1] // groups
    require(group_size * groups == codes.shape[1], "projection group size mismatch")

    grouped_input = inputs.astype(mx.float32).reshape(-1, groups, group_size)
    grouped_codes = codes.astype(mx.float32).reshape(codes.shape[0], groups, group_size)
    scale_feature = mx.einsum("ngk,ogk->ong", grouped_input, grouped_codes)
    bias_feature = mx.broadcast_to(mx.sum(grouped_input, axis=-1)[None, :, :], scale_feature.shape)
    features = mx.concatenate((scale_feature, bias_feature), axis=-1)
    initial = mx.concatenate(
        (initial_scales.astype(mx.float32), initial_biases.astype(mx.float32)), axis=-1
    )
    residual = target_output.astype(mx.float32).T - mx.sum(features * initial[:, None, :], axis=-1)
    weights = sample_weights.astype(mx.float32)
    weights = weights / mx.maximum(mx.mean(weights), 1e-8)
    weighted_features = features * weights[None, :, None]
    transposed = mx.swapaxes(features, -1, -2)
    gram = transposed @ weighted_features
    diagonal = mx.diagonal(gram, axis1=-2, axis2=-1)
    identity = mx.eye(features.shape[-1], dtype=mx.float32)[None, :, :]
    regularizer = identity * (ridge * mx.maximum(diagonal, 1e-8))[:, None, :]
    rhs = transposed @ (weights[None, :, None] * residual[..., None])
    mx.eval(gram, rhs)
    delta = mx.linalg.solve(gram + regularizer + identity * 1e-8, rhs, stream=mx.cpu).squeeze(-1)
    fitted = initial + delta
    low = fitted[:, groups:]
    high = low + fitted[:, :groups]
    grouped_teacher = teacher_weight.astype(mx.float32).reshape(
        teacher_weight.shape[0], groups, group_size
    )
    teacher_low = mx.min(grouped_teacher, axis=-1)
    teacher_high = mx.max(grouped_teacher, axis=-1)
    span = mx.maximum(teacher_high - teacher_low, 1e-8)
    low = mx.clip(low, teacher_low - endpoint_margin * span, teacher_high + endpoint_margin * span)
    high = mx.clip(high, teacher_low - endpoint_margin * span, teacher_high + endpoint_margin * span)
    high = mx.maximum(high, low)
    return (high - low).astype(mx.bfloat16), low.astype(mx.bfloat16)


def affine_weight(
    value: mx.array,
    bits: int,
    group_size: int = 128,
    iterations: int = 8,
) -> AffineWeight:
    if bits == 1:
        codes, scales, biases = kmeans_affine_codes(value, bits, group_size, iterations)
        packed = pack_codes(codes, bits)
    else:
        packed, scales, biases = mx.quantize(
            value,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
    result = AffineWeight(
        weight=packed,
        scales=scales,
        biases=biases,
        bits=bits,
        group_size=group_size,
        rows=value.shape[0],
        columns=value.shape[1],
    )
    result.validate()
    return result


def _projection_fit(
    teacher_weight: mx.array,
    inputs: mx.array,
    target: mx.array,
    sample_weights: mx.array,
    group_size: int,
    ridge: float,
    endpoint_margin: float,
) -> AffineWeight:
    codes, scales, biases = kmeans_affine_codes(teacher_weight, 1, group_size)
    fitted_scales, fitted_biases = fit_projection_endpoints(
        inputs,
        target,
        teacher_weight,
        codes,
        scales,
        biases,
        sample_weights,
        ridge,
        endpoint_margin,
    )
    result = AffineWeight(
        weight=pack_codes(codes, 1),
        scales=fitted_scales,
        biases=fitted_biases,
        bits=1,
        group_size=group_size,
        rows=teacher_weight.shape[0],
        columns=teacher_weight.shape[1],
    )
    result.validate()
    return result


def expert_output(latent: mx.array, up: mx.array, down: mx.array) -> mx.array:
    hidden = mx.square(mx.maximum(latent.astype(mx.float32) @ up.astype(mx.float32).T, 0.0))
    return hidden @ down.astype(mx.float32).T


def output_error(
    candidate: mx.array,
    teacher: mx.array,
    sample_weights: mx.array,
) -> dict[str, float]:
    require(candidate.shape == teacher.shape and candidate.ndim == 2, "expert output shape mismatch")
    require(sample_weights.shape == (candidate.shape[0],), "expert metric weight mismatch")
    weights = sample_weights.astype(mx.float32)
    weights = weights / mx.maximum(mx.mean(weights), 1e-8)
    difference = candidate.astype(mx.float32) - teacher.astype(mx.float32)
    error2 = mx.sum(weights[:, None] * mx.square(difference))
    reference2 = mx.sum(weights[:, None] * mx.square(teacher.astype(mx.float32)))
    maximum = mx.max(mx.abs(difference))
    mx.eval(error2, reference2, maximum)
    error = float(error2)
    reference = float(reference2)
    return {
        "error2": error,
        "reference2": reference,
        "relative_l2": math.sqrt(error / max(reference, 1e-30)),
        "max_abs": float(maximum),
    }


def fit_binary_expert(
    teacher_up: mx.array,
    teacher_down: mx.array,
    train_latent: mx.array,
    validation_latent: mx.array,
    train_weights: mx.array | None = None,
    validation_weights: mx.array | None = None,
    group_size: int = 128,
    ridge: float = 1e-3,
    endpoint_margin: float = 0.25,
    *,
    target_up: mx.array | None = None,
    target_down: mx.array | None = None,
) -> tuple[BinaryExpert, dict[str, dict[str, float]]]:
    """Fit BF16-derived codes to a teacher or an explicit QAT target function."""

    require(teacher_up.ndim == teacher_down.ndim == 2, "expert teacher weights must be matrices")
    require(teacher_up.shape[1] == teacher_down.shape[0], "expert teacher latent width mismatch")
    require(teacher_up.shape[0] == teacher_down.shape[1], "expert teacher hidden width mismatch")
    require(train_latent.ndim == validation_latent.ndim == 2, "expert contexts must be matrices")
    require(train_latent.shape[1] == validation_latent.shape[1] == teacher_up.shape[1], "context width mismatch")
    require(train_latent.shape[0] >= 2 and validation_latent.shape[0] >= 1, "insufficient expert contexts")
    target_up = teacher_up if target_up is None else target_up
    target_down = teacher_down if target_down is None else target_down
    require(target_up.shape == teacher_up.shape, "expert target up shape mismatch")
    require(target_down.shape == teacher_down.shape, "expert target down shape mismatch")
    train_weights = (
        mx.ones((train_latent.shape[0],), dtype=mx.float32)
        if train_weights is None
        else train_weights
    )
    validation_weights = (
        mx.ones((validation_latent.shape[0],), dtype=mx.float32)
        if validation_weights is None
        else validation_weights
    )
    require(bool(mx.all(train_weights >= 0)) and bool(mx.any(train_weights > 0)), "invalid train weights")
    require(
        bool(mx.all(validation_weights >= 0)) and bool(mx.any(validation_weights > 0)),
        "invalid validation weights",
    )

    teacher_train_pre = train_latent.astype(mx.float32) @ target_up.astype(mx.float32).T
    up = _projection_fit(
        teacher_up,
        train_latent,
        teacher_train_pre,
        train_weights,
        group_size,
        ridge,
        endpoint_margin,
    )
    fitted_up = dequantize_affine(up)
    student_train_hidden = mx.square(mx.maximum(train_latent.astype(mx.float32) @ fitted_up.T, 0.0))
    teacher_train_output = expert_output(train_latent, target_up, target_down)
    down = _projection_fit(
        teacher_down,
        student_train_hidden,
        teacher_train_output,
        train_weights,
        group_size,
        ridge,
        endpoint_margin,
    )
    result = BinaryExpert(up=up, down=down)
    result.validate()
    fitted_down = dequantize_affine(down)
    initial_up = affine_weight(teacher_up, 1, group_size)
    initial_down = affine_weight(teacher_down, 1, group_size)
    initial_up_dense = dequantize_affine(initial_up)
    initial_down_dense = dequantize_affine(initial_down)

    metrics = {}
    for label, latent, weights in (
        ("train", train_latent, train_weights),
        ("validation", validation_latent, validation_weights),
    ):
        teacher = expert_output(latent, target_up, target_down)
        initial = expert_output(
            latent,
            initial_up_dense,
            initial_down_dense,
        )
        candidate = expert_output(latent, fitted_up, fitted_down)
        metrics[label] = {
            **{f"initial_{key}": value for key, value in output_error(initial, teacher, weights).items()},
            **{f"fitted_{key}": value for key, value in output_error(candidate, teacher, weights).items()},
        }
    return result, metrics


def precision_metrics(
    teacher_up: mx.array,
    teacher_down: mx.array,
    latent: mx.array,
    sample_weights: mx.array,
    bits: tuple[int, ...] = (2, 3, 4),
    group_size: int = 128,
    *,
    target_up: mx.array | None = None,
    target_down: mx.array | None = None,
) -> dict[str, dict[str, float]]:
    """Measure held-out function error for ordinary affine precision tiers."""

    target_up = teacher_up if target_up is None else target_up
    target_down = teacher_down if target_down is None else target_down
    require(target_up.shape == teacher_up.shape, "precision target up shape mismatch")
    require(target_down.shape == teacher_down.shape, "precision target down shape mismatch")
    teacher = expert_output(latent, target_up, target_down)
    result = {}
    for bit_width in bits:
        up = affine_weight(teacher_up, bit_width, group_size)
        down = affine_weight(teacher_down, bit_width, group_size)
        candidate = expert_output(latent, dequantize_affine(up), dequantize_affine(down))
        result[str(bit_width)] = {
            **output_error(candidate, teacher, sample_weights),
            "payload_bytes": up.payload_bytes + down.payload_bytes,
        }
    return result


def _ste_binary_dense(
    logits: mx.array,
    scales: mx.array,
    biases: mx.array,
    temperature: float,
) -> mx.array:
    require(temperature > 0.0, "binary refinement temperature must be positive")
    probability = mx.sigmoid(logits.astype(mx.float32) / temperature)
    hard = (logits >= 0).astype(mx.float32)
    codes = probability + mx.stop_gradient(hard - probability)
    return reconstruct_affine(codes, scales, biases)


def _binary_from_parameters(
    parameters: dict[str, mx.array],
    up_shape: tuple[int, int],
    down_shape: tuple[int, int],
    group_size: int,
) -> BinaryExpert:
    def projection(name: str, shape: tuple[int, int]) -> AffineWeight:
        codes = (parameters[f"{name}_logits"] >= 0).astype(mx.uint32)
        result = AffineWeight(
            weight=pack_codes(codes, 1),
            scales=parameters[f"{name}_scales"].astype(mx.bfloat16),
            biases=parameters[f"{name}_biases"].astype(mx.bfloat16),
            bits=1,
            group_size=group_size,
            rows=shape[0],
            columns=shape[1],
        )
        result.validate()
        return result

    result = BinaryExpert(projection("up", up_shape), projection("down", down_shape))
    result.validate()
    return result


def _endpoint_bounds(
    teacher: mx.array,
    group_size: int,
    margin: float,
) -> tuple[mx.array, mx.array]:
    grouped = teacher.astype(mx.float32).reshape(teacher.shape[0], -1, group_size)
    low = mx.min(grouped, axis=-1)
    high = mx.max(grouped, axis=-1)
    span = mx.maximum(high - low, 1e-8)
    return low - margin * span, high + margin * span


def _clamp_binary_parameters(
    parameters: dict[str, mx.array],
    bounds: dict[str, tuple[mx.array, mx.array]],
) -> dict[str, mx.array]:
    result = dict(parameters)
    for name in ("up", "down"):
        minimum, maximum = bounds[name]
        low = mx.clip(parameters[f"{name}_biases"], minimum, maximum)
        high = mx.clip(
            parameters[f"{name}_biases"] + parameters[f"{name}_scales"],
            minimum,
            maximum,
        )
        high = mx.maximum(high, low + 1e-8)
        result[f"{name}_biases"] = low
        result[f"{name}_scales"] = high - low
        result[f"{name}_logits"] = mx.clip(parameters[f"{name}_logits"], -8.0, 8.0)
    return result


def refine_binary_expert(
    initial: BinaryExpert,
    teacher_up: mx.array,
    teacher_down: mx.array,
    train_latent: mx.array,
    validation_latent: mx.array,
    train_weights: mx.array,
    validation_weights: mx.array,
    *,
    target_up: mx.array | None = None,
    target_down: mx.array | None = None,
    steps: int = 16,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    code_learning_rate: float = 3e-3,
    temperature: float = 1.0,
    logit_margin: float = 0.25,
    code_anchor: float = 1e-4,
    endpoint_margin: float = 0.25,
    evaluate_every: int = 4,
    code_warmup_steps: int = 8,
) -> tuple[BinaryExpert, dict]:
    """Safely refine binary code assignments with held-out checkpoint selection."""

    from nemotron_mlx_gefen import GefenMLX

    initial.validate()
    require(
        steps >= 0
        and batch_size > 0
        and evaluate_every > 0
        and code_warmup_steps >= 0,
        "invalid binary refinement schedule",
    )
    require(
        learning_rate > 0.0
        and code_learning_rate > 0.0
        and temperature > 0.0
        and logit_margin > 0.0,
        "invalid binary refinement optimizer",
    )
    require(code_anchor >= 0.0 and endpoint_margin >= 0.0, "invalid binary refinement regularizer")
    require(teacher_up.shape == (initial.up.rows, initial.up.columns), "refinement up shape mismatch")
    require(teacher_down.shape == (initial.down.rows, initial.down.columns), "refinement down shape mismatch")
    target_up = teacher_up if target_up is None else target_up
    target_down = teacher_down if target_down is None else target_down
    require(target_up.shape == teacher_up.shape and target_down.shape == teacher_down.shape, "refinement target mismatch")
    require(train_latent.shape[0] >= 2 and validation_latent.shape[0] >= 1, "insufficient refinement contexts")
    require(train_weights.shape == (train_latent.shape[0],), "refinement train-weight mismatch")
    require(validation_weights.shape == (validation_latent.shape[0],), "refinement validation-weight mismatch")
    effective_warmup_steps = min(code_warmup_steps, steps)

    up_codes = unpack_codes(initial.up.weight, initial.up.columns, 1).astype(mx.float32)
    down_codes = unpack_codes(initial.down.weight, initial.down.columns, 1).astype(mx.float32)
    initial_codes = {"up": up_codes, "down": down_codes}
    parameters = {
        "up_logits": mx.where(up_codes > 0, logit_margin, -logit_margin),
        "up_scales": initial.up.scales.astype(mx.float32),
        "up_biases": initial.up.biases.astype(mx.float32),
        "down_logits": mx.where(down_codes > 0, logit_margin, -logit_margin),
        "down_scales": initial.down.scales.astype(mx.float32),
        "down_biases": initial.down.biases.astype(mx.float32),
    }
    bounds = {
        "up": _endpoint_bounds(teacher_up, initial.up.group_size, endpoint_margin),
        "down": _endpoint_bounds(teacher_down, initial.down.group_size, endpoint_margin),
    }
    train_target = expert_output(train_latent, target_up, target_down)
    validation_target = expert_output(validation_latent, target_up, target_down)
    mx.eval(*parameters.values(), *bounds["up"], *bounds["down"], train_target, validation_target)

    def loss_fn(values: dict[str, mx.array], rows: mx.array) -> mx.array:
        up = _ste_binary_dense(
            values["up_logits"],
            values["up_scales"],
            values["up_biases"],
            temperature,
        )
        down = _ste_binary_dense(
            values["down_logits"],
            values["down_scales"],
            values["down_biases"],
            temperature,
        )
        candidate = expert_output(train_latent[rows], up, down)
        target = train_target[rows]
        weights = train_weights[rows].astype(mx.float32)
        weights = weights / mx.maximum(mx.mean(weights), 1e-8)
        error = mx.sum(weights[:, None] * mx.square(candidate - target))
        reference = mx.sum(weights[:, None] * mx.square(target))
        anchor = 0.0
        for name in ("up", "down"):
            probability = mx.sigmoid(values[f"{name}_logits"] / temperature)
            anchor = anchor + mx.mean(mx.square(probability - initial_codes[name]))
        return error / mx.maximum(reference, 1e-30) + code_anchor * anchor

    initial_dense_up = dequantize_affine(initial.up)
    initial_dense_down = dequantize_affine(initial.down)
    initial_metrics = output_error(
        expert_output(validation_latent, initial_dense_up, initial_dense_down),
        validation_target,
        validation_weights,
    )
    best = initial
    best_metrics = initial_metrics
    best_step = 0
    history = [{"step": 0, **initial_metrics, "code_flips": 0}]
    if steps == 0:
        return best, {
            "optimizer": "gefen",
            "steps": 0,
            "best_step": 0,
            "improved": False,
            "initial": initial_metrics,
            "best": best_metrics,
            "history": history,
        }

    code_names = ("up_logits", "down_logits")
    endpoint_names = ("up_scales", "up_biases", "down_scales", "down_biases")
    code_optimizer = GefenMLX(code_learning_rate, period_cap_by_shape=True)
    endpoint_optimizer = GefenMLX(learning_rate, period_cap_by_shape=True)
    value_and_grad = mx.value_and_grad(loss_fn)
    train_rows = train_latent.shape[0]
    for step in range(1, steps + 1):
        start = ((step - 1) * batch_size) % train_rows
        row_ids = [(start + offset) % train_rows for offset in range(min(batch_size, train_rows))]
        rows = mx.array(row_ids, dtype=mx.int32)
        loss, gradients = value_and_grad(parameters, rows)
        code_values = (
            code_optimizer.update(
                {name: parameters[name] for name in code_names},
                {name: gradients[name] for name in code_names},
            )
            if step > effective_warmup_steps
            else {name: parameters[name] for name in code_names}
        )
        endpoint_values = endpoint_optimizer.update(
            {name: parameters[name] for name in endpoint_names},
            {name: gradients[name] for name in endpoint_names},
        )
        parameters = _clamp_binary_parameters({**code_values, **endpoint_values}, bounds)
        mx.eval(loss, *parameters.values())
        if step % evaluate_every != 0 and step != steps:
            continue
        candidate = _binary_from_parameters(
            parameters,
            teacher_up.shape,
            teacher_down.shape,
            initial.up.group_size,
        )
        candidate_output = expert_output(
            validation_latent,
            dequantize_affine(candidate.up),
            dequantize_affine(candidate.down),
        )
        metrics = output_error(candidate_output, validation_target, validation_weights)
        flips = int(
            mx.sum(
                unpack_codes(candidate.up.weight, candidate.up.columns, 1).astype(mx.int32)
                != up_codes.astype(mx.int32)
            )
        ) + int(
            mx.sum(
                unpack_codes(candidate.down.weight, candidate.down.columns, 1).astype(mx.int32)
                != down_codes.astype(mx.int32)
            )
        )
        history.append({"step": step, **metrics, "code_flips": flips})
        if metrics["error2"] < best_metrics["error2"]:
            best = BinaryExpert(
                AffineWeight(
                    mx.array(np.asarray(candidate.up.weight)),
                    mx.array(np.asarray(candidate.up.scales.astype(mx.float32))).astype(mx.bfloat16),
                    mx.array(np.asarray(candidate.up.biases.astype(mx.float32))).astype(mx.bfloat16),
                    1,
                    candidate.up.group_size,
                    candidate.up.rows,
                    candidate.up.columns,
                ),
                AffineWeight(
                    mx.array(np.asarray(candidate.down.weight)),
                    mx.array(np.asarray(candidate.down.scales.astype(mx.float32))).astype(mx.bfloat16),
                    mx.array(np.asarray(candidate.down.biases.astype(mx.float32))).astype(mx.bfloat16),
                    1,
                    candidate.down.group_size,
                    candidate.down.rows,
                    candidate.down.columns,
                ),
            )
            best.validate()
            mx.eval(
                best.up.weight,
                best.up.scales,
                best.up.biases,
                best.down.weight,
                best.down.scales,
                best.down.biases,
            )
            best_metrics = metrics
            best_step = step
    best_up_codes = unpack_codes(best.up.weight, best.up.columns, 1)
    best_down_codes = unpack_codes(best.down.weight, best.down.columns, 1)
    final_flips = int(mx.sum(best_up_codes != up_codes.astype(mx.uint32))) + int(
        mx.sum(best_down_codes != down_codes.astype(mx.uint32))
    )
    return best, {
        "optimizer": "gefen",
        "optimizer_revision": "704034f0d62871cc651a5ebae7b5547c55e0fc37",
        "optimizer_state_bytes": code_optimizer.state_bytes() + endpoint_optimizer.state_bytes(),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "code_learning_rate": code_learning_rate,
        "temperature": temperature,
        "logit_margin": logit_margin,
        "code_anchor": code_anchor,
        "evaluate_every": evaluate_every,
        "code_warmup_steps": effective_warmup_steps,
        "best_step": best_step,
        "best_code_flips": final_flips,
        "improved": best_step > 0,
        "initial": initial_metrics,
        "best": best_metrics,
        "history": history,
    }

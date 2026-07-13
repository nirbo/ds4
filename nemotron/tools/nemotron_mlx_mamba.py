#!/usr/bin/env python3
"""Load and execute one official Nemotron Mamba2 layer through MLX."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock
from mlx_lm.models.ssm import compute_dt, ssm_update

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear, ModelOptFP8Linear, fp8_matvec, fp8_matvec_custom


SEQUENCE_RELATIVE_L2_LIMIT = 1e-7
SEQUENCE_MAX_ABS_LIMIT = 2e-7
SEQUENCE_STATE_MAX_ABS_LIMIT = 4e-5
SHORT_SEQUENCE_LIMIT = 8
SHORT_SEQUENCE_ENABLED = os.environ.get("NEMOTRON_DISABLE_SHORT_SSM") != "1"


SSM_SEQUENCE_SOURCE = r"""
uint batch_head = thread_position_in_grid.z;
uint batch = batch_head / H;
uint head = batch_head - batch * H;
uint group = head / HEADS_PER_GROUP;
uint channel = thread_position_in_grid.y;
uint state_lane = thread_position_in_threadgroup.x;
uint state_base = ((batch * H + head) * HEAD_DIM + channel) * STATE_DIM;
float state_values[STATE_DIM / 32];
for (uint item = 0; item < STATE_DIM / 32; ++item) {
    uint state_index = state_lane + item * 32;
    state_values[item] = float(state_in[state_base + state_index]);
}
float transition = -fast::exp(float(A_log[head]));
for (uint token = 0; token < TOKENS; ++token) {
    uint token_head = (batch * TOKENS + token) * H + head;
    uint x_base = token_head * HEAD_DIM;
    uint bc_base = ((batch * TOKENS + token) * GROUPS + group) * STATE_DIM;
    float delta = dt[token_head];
    float decay = fast::exp(transition * delta);
    float input = float(X[x_base + channel]);
    float sum = 0.0f;
    for (uint item = 0; item < STATE_DIM / 32; ++item) {
        uint state_index = state_lane + item * 32;
        float updated = decay * state_values[item]
            + input * delta * float(B[bc_base + state_index]);
        sum += updated * float(C[bc_base + state_index]);
        state_values[item] = float(U(updated));
    }
    sum = simd_sum(sum);
    if (thread_index_in_simdgroup == 0) {
        output[x_base + channel] = T(sum + input * float(D[head]));
    }
}
for (uint item = 0; item < STATE_DIM / 32; ++item) {
    uint state_index = state_lane + item * 32;
    state_out[state_base + state_index] = U(state_values[item]);
}
"""


SSM_SEQUENCE_CAPTURE_SOURCE = SSM_SEQUENCE_SOURCE.replace(
    "        state_values[item] = float(U(updated));",
    """        state_values[item] = float(U(updated));
        if (token == CAPTURE_TOKEN) {
            captured_state[state_base + state_index] = U(state_values[item]);
        }""",
)


_ssm_sequence_kernel = mx.fast.metal_kernel(
    name="nemotron_ssm_short_sequence",
    input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
    output_names=["output", "state_out"],
    source=SSM_SEQUENCE_SOURCE,
)


_ssm_sequence_capture_kernel = mx.fast.metal_kernel(
    name="nemotron_ssm_short_sequence_capture",
    input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
    output_names=["output", "state_out", "captured_state"],
    source=SSM_SEQUENCE_CAPTURE_SOURCE,
)


def ssm_short_sequence(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: tuple[float, float],
    capture_token: int | None = None,
) -> tuple[mx.array, mx.array, mx.array | None]:
    """Run a short Mamba recurrence in one Metal launch."""

    batch, tokens, heads, head_dim = hidden_states.shape
    groups, state_dim = B.shape[-2:]
    require(
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and 2 <= tokens <= SHORT_SEQUENCE_LIMIT,
        "short SSM kernel requires a 2-8 token Metal sequence",
    )
    require(
        B.shape == C.shape == (batch, tokens, groups, state_dim)
        and dt.shape == (batch, tokens, heads)
        and state.shape == (batch, heads, head_dim, state_dim)
        and heads % groups == 0
        and state_dim % 32 == 0,
        "short SSM kernel shape mismatch",
    )
    require(
        capture_token is None or 0 <= capture_token < tokens,
        "short SSM capture token is out of range",
    )
    dt = compute_dt(dt, dt_bias, time_step_limit)
    template = [
        ("T", hidden_states.dtype),
        ("U", state.dtype),
        ("TOKENS", tokens),
        ("H", heads),
        ("GROUPS", groups),
        ("HEADS_PER_GROUP", heads // groups),
        ("HEAD_DIM", head_dim),
        ("STATE_DIM", state_dim),
    ]
    output_shapes = [hidden_states.shape, state.shape]
    output_dtypes = [hidden_states.dtype, state.dtype]
    kernel = _ssm_sequence_kernel
    if capture_token is not None:
        template.append(("CAPTURE_TOKEN", capture_token))
        output_shapes.append(state.shape)
        output_dtypes.append(state.dtype)
        kernel = _ssm_sequence_capture_kernel
    outputs = kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=template,
        grid=(32, head_dim, heads * batch),
        threadgroup=(32, 8, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    return outputs[0], outputs[1], outputs[2] if capture_token is not None else None


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
    capture_tokens: tuple[int, ...] | None = None,
    captured_states: dict[int, list[mx.array]] | None = None,
) -> mx.array:
    """Batch projection work while preserving one-token SSM recurrence order."""

    require(x.ndim == 3 and x.shape[1] > 1, "exact Mamba sequence requires multiple tokens")
    mixer = block.mixer
    single_capture = capture_token is not None or captured_state is not None
    multi_capture = capture_tokens is not None or captured_states is not None
    require(not (single_capture and multi_capture), "Mamba capture modes are mutually exclusive")
    if single_capture:
        require(
            capture_token is not None
            and captured_state is not None
            and 0 <= capture_token < x.shape[1]
            and not captured_state,
            "invalid Mamba intermediate-state capture request",
        )
        requested_tokens = (capture_token,)
    elif multi_capture:
        require(
            capture_tokens is not None
            and captured_states is not None
            and capture_tokens
            and tuple(sorted(set(capture_tokens))) == capture_tokens
            and all(0 <= token < x.shape[1] for token in capture_tokens)
            and not captured_states,
            "invalid Mamba multi-state capture request",
        )
        requested_tokens = capture_tokens
    else:
        requested_tokens = ()
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
    captured_conv_states = {}
    if requested_tokens:
        require(cache.lengths is None, "Mamba capture does not support per-request lengths")
        if initial_conv_state is None:
            initial_conv_state = mx.zeros(
                (conv_input.shape[0], mixer.conv_kernel_size - 1, mixer.conv_dim),
                dtype=conv_input.dtype,
            )
        conv_history = mx.concatenate([initial_conv_state, conv_input], axis=1)
        for token in requested_tokens:
            start = token + 1
            captured_conv_states[token] = conv_history[
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
    output = None
    outputs = []
    captured_ssm_states = {}
    if (
        SHORT_SEQUENCE_ENABLED
        and state is not None
        and mask is None
        and sequence_length <= SHORT_SEQUENCE_LIMIT
        and len(requested_tokens) <= 1
    ):
        output, state, captured_ssm_state = ssm_short_sequence(
            hidden_ssm,
            mixer.A_log,
            B,
            C,
            mixer.D.astype(hidden_ssm.dtype),
            dt,
            mixer.dt_bias,
            state,
            mixer.time_step_limit,
            requested_tokens[0] if requested_tokens else None,
        )
        if requested_tokens:
            require(captured_ssm_state is not None, "short SSM state capture failed")
            captured_ssm_states[requested_tokens[0]] = captured_ssm_state
    else:
        for token in range(sequence_length):
            token_mask = mask[:, token : token + 1] if mask is not None else None
            token_output, state = ssm_update(
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
            outputs.append(token_output)
            if token in requested_tokens:
                captured_ssm_states[token] = state
    cache[1] = state
    cache.advance(sequence_length)
    if captured_state is not None:
        require(
            capture_token in captured_conv_states and capture_token in captured_ssm_states,
            "Mamba intermediate state was not captured",
        )
        captured_state.extend(
            [captured_conv_states[capture_token], captured_ssm_states[capture_token]]
        )
    if captured_states is not None:
        require(
            set(captured_conv_states) == set(capture_tokens)
            and set(captured_ssm_states) == set(capture_tokens),
            "Mamba intermediate states were not captured",
        )
        captured_states.update(
            {
                token: [captured_conv_states[token], captured_ssm_states[token]]
                for token in capture_tokens
            }
        )
    if output is None:
        output = mx.concatenate(outputs, axis=1)
    output = output.reshape(batch_size, sequence_length, mixer.intermediate_size)
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
    multi_capture_block = load_mamba_layer(source_dir, layer)
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
    multi_capture_cache = ArraysCache(size=2)
    incremental_cache = ArraysCache(size=2)
    prefix = mx.array(
        [
            math.sin(index * 0.011 - 0.19) * 0.2
            + math.cos(index * 0.015 + 0.23) * 0.1
            for index in range(hidden_size)
        ],
        dtype=mx.float32,
    ).reshape(1, 1, hidden_size)
    prefix_outputs = [
        block(prefix, mask=None, cache=cache)
        for block, cache in (
            (batched_block, batched_cache),
            (multi_capture_block, multi_capture_cache),
            (incremental_block, incremental_cache),
        )
    ]
    mx.eval(
        *prefix_outputs,
        batched_cache.state,
        multi_capture_cache.state,
        incremental_cache.state,
    )
    captured = []
    batched = mamba_sequence_exact(
        batched_block,
        sequence,
        batched_cache,
        capture_token=0,
        captured_state=captured,
    )
    multi_captured = {}
    multi_batched = mamba_sequence_exact(
        multi_capture_block,
        sequence,
        multi_capture_cache,
        capture_tokens=tuple(range(tokens - 1)),
        captured_states=multi_captured,
    )
    incremental_outputs = []
    first_incremental_state = None
    incremental_states = {}
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
        if token < tokens - 1:
            incremental_states[token] = tuple(incremental_cache.state)
    incremental = mx.concatenate(incremental_outputs, axis=1)
    require(first_incremental_state is not None, "incremental Mamba state was not captured")
    mx.eval(
        batched,
        multi_batched,
        incremental,
        batched_cache.state,
        incremental_cache.state,
        captured,
        first_incremental_state,
        multi_captured,
        incremental_states,
    )
    difference = batched - incremental
    multi_difference = multi_batched - incremental
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
    multi_capture_error = max(
        float(mx.max(mx.abs(left - right)))
        for token in multi_captured
        for left, right in zip(multi_captured[token], incremental_states[token])
    )
    return {
        "sequence_relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "sequence_max_abs": float(mx.max(mx.abs(difference))),
        "sequence_state_max_abs": state_error,
        "captured_state_max_abs": captured_state_error,
        "multi_captured_state_max_abs": multi_capture_error,
        "multi_sequence_max_abs": float(mx.max(mx.abs(multi_difference))),
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
            f"multi_captured_state_max_abs={comparison['multi_captured_state_max_abs']:.9g} "
            f"multi_sequence_max_abs={comparison['multi_sequence_max_abs']:.9g} "
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
            and comparison["captured_state_max_abs"] <= SEQUENCE_STATE_MAX_ABS_LIMIT
            and comparison["multi_captured_state_max_abs"] <= SEQUENCE_STATE_MAX_ABS_LIMIT
            and comparison["multi_sequence_max_abs"] <= SEQUENCE_MAX_ABS_LIMIT,
            "Mamba recurrent-order sequence drift exceeds the validated envelope",
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        print(f"nemotron MLX Mamba error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

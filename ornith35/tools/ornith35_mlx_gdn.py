#!/usr/bin/env python3
"""One-token MLX GatedDeltaNet composition for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ornith35_gdn_reference import GDNConfig, require
from ornith35_nvfp4 import SafetensorsFile


PRODUCTION_CONFIG = GDNConfig(
    hidden_size=2048,
    num_k_heads=16,
    num_v_heads=32,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel_size=4,
    rms_norm_eps=1e-6,
)


CONV_KERNEL_SOURCE = r"""
uint channel = thread_position_in_grid.x;
if (channel >= 8192u) return;
uint base = channel * 4u;
bfloat16_t first = conv_state[base + 1u];
bfloat16_t second = conv_state[base + 2u];
bfloat16_t third = conv_state[base + 3u];
bfloat16_t fourth = mixed[channel];
output_state[base] = first;
output_state[base + 1u] = second;
output_state[base + 2u] = third;
output_state[base + 3u] = fourth;
float total = 0.0f;
volatile float product0 = float(first) * float(weight[base]);
total += product0;
volatile float product1 = float(second) * float(weight[base + 1u]);
total += product1;
volatile float product2 = float(third) * float(weight[base + 2u]);
total += product2;
volatile float product3 = float(fourth) * float(weight[base + 3u]);
total += product3;
output_convolved[channel] = total;
"""


_conv_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_conv4_bf16_f32",
    input_names=["conv_state", "mixed", "weight"],
    output_names=["output_state", "output_convolved"],
    source=CONV_KERNEL_SOURCE,
)


# The column loop and shuffle reduction match MLX 0.32.0's BF16 GEMV. One row
# per SIMD group avoids register pressure from the fused convolution and gate.
QKV_CONV_SILU_KERNEL_SOURCE = r"""
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x * 8u + group;
if (row >= 8192u) return;
float sum = 0.0f;
for (uint column = lane * 4u; column < 2048u; column += 128u) {
    float input0 = float(hidden[column]);
    float input1 = float(hidden[column + 1u]);
    float input2 = float(hidden[column + 2u]);
    float input3 = float(hidden[column + 3u]);
    uint weight_base = row * 2048u + column;
    sum += float(projection[weight_base]) * input0;
    sum += float(projection[weight_base + 1u]) * input1;
    sum += float(projection[weight_base + 2u]) * input2;
    sum += float(projection[weight_base + 3u]) * input3;
}
for (ushort offset = 16; offset >= 1; offset >>= 1) {
    sum += simd_shuffle_down(sum, offset);
}
if (lane == 0u) {
    uint state_base = row * 4u;
    bfloat16_t first = conv_state[state_base + 1u];
    bfloat16_t second = conv_state[state_base + 2u];
    bfloat16_t third = conv_state[state_base + 3u];
    bfloat16_t fourth = bfloat16_t(sum);
    output_state[state_base] = first;
    output_state[state_base + 1u] = second;
    output_state[state_base + 2u] = third;
    output_state[state_base + 3u] = fourth;
    float total = 0.0f;
    volatile float product0 = float(first) * float(conv_weight[state_base]);
    total += product0;
    volatile float product1 =
        float(second) * float(conv_weight[state_base + 1u]);
    total += product1;
    volatile float product2 =
        float(third) * float(conv_weight[state_base + 2u]);
    total += product2;
    volatile float product3 =
        float(fourth) * float(conv_weight[state_base + 3u]);
    total += product3;
    float y = 1.0f / (
        1.0f + metal::precise::exp(metal::abs(total))
    );
    float sigmoid_value = total < 0.0f ? y : 1.0f - y;
    volatile float silu_value = total * sigmoid_value;
    output_convolved[row] = bfloat16_t(silu_value);
}
"""


_qkv_conv_silu_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_qkv_conv_silu_bf16_rows1_sg8",
    input_names=["projection", "hidden", "conv_state", "conv_weight"],
    output_names=["output_state", "output_convolved"],
    source=QKV_CONV_SILU_KERNEL_SOURCE,
)


# The volatile products retain MLX's materialized FP32 multiply/add boundaries;
# allowing Metal to contract them changes the authoritative recurrent state.
RECURRENCE_KERNEL_SOURCE = r"""
uint head = threadgroup_position_in_grid.x;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
float decay_value = decay[head];
float beta_value = beta[head];
for (uint value_index = group; value_index < 128u; value_index += SIMDGROUPS) {
    float memory = 0.0f;
    uint state_indices[4];
    float decayed_values[4];
    float key_values[4];
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint key_index = lane + offset * 32u;
        uint state_index = (head * 128u + key_index) * 128u + value_index;
        float decayed = recurrent[state_index] * decay_value;
        float key_value = key[head * 128u + key_index];
        state_indices[offset] = state_index;
        decayed_values[offset] = decayed;
        key_values[offset] = key_value;
        volatile float memory_term = decayed * key_value;
        memory += memory_term;
    }
    memory = simd_sum(memory);
    memory = simd_broadcast_first(memory);
    float delta = (value[head * 128u + value_index] - memory) * beta_value;
    float core = 0.0f;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint key_index = lane + offset * 32u;
        volatile float update = key_values[offset] * delta;
        float next = decayed_values[offset] + update;
        output_recurrent[state_indices[offset]] = next;
        volatile float core_term = next * query[head * 128u + key_index];
        core += core_term;
    }
    core = simd_sum(core);
    if (lane == 0u) output_core[head * 128u + value_index] = core;
}
"""


_recurrence_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_recurrence_f32",
    input_names=["recurrent", "key", "query", "value", "beta", "decay"],
    output_names=["output_recurrent", "output_core"],
    source=RECURRENCE_KERNEL_SOURCE,
)


RECURRENCE_CORE_GATE_KERNEL_SOURCE = r"""
uint head = threadgroup_position_in_grid.x;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
threadgroup float core_values[128];
threadgroup float inverse_variance[1];
float decay_value = decay[head];
float beta_value = beta[head];
for (uint value_index = group; value_index < 128u; value_index += SIMDGROUPS) {
    float memory = 0.0f;
    uint state_indices[4];
    float decayed_values[4];
    float key_values[4];
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint key_index = lane + offset * 32u;
        uint state_index = (head * 128u + key_index) * 128u + value_index;
        float decayed = recurrent[state_index] * decay_value;
        float key_value = key[head * 128u + key_index];
        state_indices[offset] = state_index;
        decayed_values[offset] = decayed;
        key_values[offset] = key_value;
        volatile float memory_term = decayed * key_value;
        memory += memory_term;
    }
    memory = simd_sum(memory);
    memory = simd_broadcast_first(memory);
    float delta = (value[head * 128u + value_index] - memory) * beta_value;
    float core = 0.0f;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint key_index = lane + offset * 32u;
        volatile float update = key_values[offset] * delta;
        float next = decayed_values[offset] + update;
        output_recurrent[state_indices[offset]] = next;
        volatile float core_term = next * query[head * 128u + key_index];
        core += core_term;
    }
    core = simd_sum(core);
    if (lane == 0u) core_values[value_index] = core;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    float total = 0.0f;
    uint base = lane * 4u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        float core = core_values[base + offset];
        volatile float square = core * core;
        total += square;
    }
    total = simd_sum(total);
    if (lane == 0u) {
        volatile float mean = total / 128.0f;
        volatile float adjusted = mean + 1.0e-6f;
        inverse_variance[0] = metal::precise::rsqrt(adjusted);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    uint base = lane * 4u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint value_index = base + offset;
        uint index = head * 128u + value_index;
        volatile float normalized = core_values[value_index] * inverse_variance[0];
        bfloat16_t normalized_bf16 = bfloat16_t(normalized);
        volatile float weighted_product =
            float(normalized_bf16) * float(norm[value_index]);
        bfloat16_t weighted = bfloat16_t(weighted_product);
        float z_value = float(z[index]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(z_value)));
        float sigmoid_value = z_value < 0.0f ? y : 1.0f - y;
        volatile float silu_value = z_value * sigmoid_value;
        volatile float gated_value = float(weighted) * silu_value;
        output_gated[index] = bfloat16_t(gated_value);
    }
}
"""


_recurrence_core_gate_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_recurrence_core_gate",
    input_names=[
        "recurrent",
        "key",
        "query",
        "value",
        "beta",
        "decay",
        "z",
        "norm",
    ],
    output_names=["output_recurrent", "output_gated"],
    source=RECURRENCE_CORE_GATE_KERNEL_SOURCE,
)


CONV_CHUNK_KERNEL_SOURCE = r"""
uint channel = thread_position_in_grid.x;
if (channel >= 8192u) return;
uint base = channel * 4u;
bfloat16_t first = conv_state[base];
bfloat16_t second = conv_state[base + 1u];
bfloat16_t third = conv_state[base + 2u];
bfloat16_t fourth = conv_state[base + 3u];
for (uint token = 0u; token < TOKENS; ++token) {
    first = second;
    second = third;
    third = fourth;
    fourth = mixed[token * 8192u + channel];
    float total = 0.0f;
    volatile float product0 = float(first) * float(weight[base]);
    total += product0;
    volatile float product1 = float(second) * float(weight[base + 1u]);
    total += product1;
    volatile float product2 = float(third) * float(weight[base + 2u]);
    total += product2;
    volatile float product3 = float(fourth) * float(weight[base + 3u]);
    total += product3;
    output_convolved[token * 8192u + channel] = total;
}
output_state[base] = first;
output_state[base + 1u] = second;
output_state[base + 2u] = third;
output_state[base + 3u] = fourth;
"""


_conv_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_conv4_chunk_bf16_f32",
    input_names=["conv_state", "mixed", "weight"],
    output_names=["output_state", "output_convolved"],
    source=CONV_CHUNK_KERNEL_SOURCE,
)


RECURRENCE_CORE_GATE_CHUNK_KERNEL_SOURCE = r"""
uint head = threadgroup_position_in_grid.x;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
threadgroup float core_values[128];
threadgroup float inverse_variance[1];
for (uint token = 0u; token < TOKENS; ++token) {
    uint head_index = token * 32u + head;
    float decay_value = decay[head_index];
    float beta_value = beta[head_index];
    for (uint value_index = group; value_index < 128u; value_index += SIMDGROUPS) {
        float memory = 0.0f;
        uint state_indices[4];
        float decayed_values[4];
        float key_values[4];
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint key_index = lane + offset * 32u;
            uint state_index = (head * 128u + key_index) * 128u + value_index;
            float previous = token == 0u ? recurrent[state_index] : output_recurrent[state_index];
            float decayed = previous * decay_value;
            float key_value = key[head_index * 128u + key_index];
            state_indices[offset] = state_index;
            decayed_values[offset] = decayed;
            key_values[offset] = key_value;
            volatile float memory_term = decayed * key_value;
            memory += memory_term;
        }
        memory = simd_sum(memory);
        memory = simd_broadcast_first(memory);
        float delta = (value[head_index * 128u + value_index] - memory) * beta_value;
        float core = 0.0f;
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint key_index = lane + offset * 32u;
            volatile float update = key_values[offset] * delta;
            float next = decayed_values[offset] + update;
            output_recurrent[state_indices[offset]] = next;
            volatile float core_term = next * query[head_index * 128u + key_index];
            core += core_term;
        }
        core = simd_sum(core);
        if (lane == 0u) core_values[value_index] = core;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0u) {
        float total = 0.0f;
        uint base = lane * 4u;
        for (uint offset = 0u; offset < 4u; ++offset) {
            float core = core_values[base + offset];
            volatile float square = core * core;
            total += square;
        }
        total = simd_sum(total);
        if (lane == 0u) {
            volatile float mean = total / 128.0f;
            volatile float adjusted = mean + 1.0e-6f;
            inverse_variance[0] = metal::precise::rsqrt(adjusted);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0u) {
        uint base = lane * 4u;
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint value_index = base + offset;
            uint index = head_index * 128u + value_index;
            volatile float normalized = core_values[value_index] * inverse_variance[0];
            bfloat16_t normalized_bf16 = bfloat16_t(normalized);
            volatile float weighted_product =
                float(normalized_bf16) * float(norm[value_index]);
            bfloat16_t weighted = bfloat16_t(weighted_product);
            float z_value = float(z[index]);
            float y = 1.0f / (1.0f + metal::exp(metal::abs(z_value)));
            float sigmoid_value = z_value < 0.0f ? y : 1.0f - y;
            volatile float silu_value = z_value * sigmoid_value;
            volatile float gated_value = float(weighted) * silu_value;
            output_gated[index] = bfloat16_t(gated_value);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}
"""


_recurrence_core_gate_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_recurrence_core_gate_chunk",
    input_names=[
        "recurrent",
        "key",
        "query",
        "value",
        "beta",
        "decay",
        "z",
        "norm",
    ],
    output_names=["output_recurrent", "output_gated"],
    source=RECURRENCE_CORE_GATE_CHUNK_KERNEL_SOURCE,
)


RECURRENCE_COLUMN_CHUNK_KERNEL_SOURCE = r"""
uint head = threadgroup_position_in_grid.x;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
for (uint value_index = group; value_index < 128u; value_index += SIMDGROUPS) {
    uint state_indices[4];
    float state_values[4];
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint key_index = lane + offset * 32u;
        uint state_index = (head * 128u + key_index) * 128u + value_index;
        state_indices[offset] = state_index;
        state_values[offset] = recurrent[state_index];
    }
    for (uint token = 0u; token < TOKENS; ++token) {
        uint head_index = token * 32u + head;
        float decay_value = decay[head_index];
        float beta_value = beta[head_index];
        float decayed_values[4];
        float key_values[4];
        float memory = 0.0f;
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint key_index = lane + offset * 32u;
            float decayed = state_values[offset] * decay_value;
            float key_value = key[head_index * 128u + key_index];
            decayed_values[offset] = decayed;
            key_values[offset] = key_value;
            volatile float memory_term = decayed * key_value;
            memory += memory_term;
        }
        memory = simd_sum(memory);
        memory = simd_broadcast_first(memory);
        float delta =
            (value[head_index * 128u + value_index] - memory) * beta_value;
        float core = 0.0f;
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint key_index = lane + offset * 32u;
            volatile float update = key_values[offset] * delta;
            float next = decayed_values[offset] + update;
            state_values[offset] = next;
            volatile float core_term =
                next * query[head_index * 128u + key_index];
            core += core_term;
        }
        core = simd_sum(core);
        if (lane == 0u) {
            output_core[head_index * 128u + value_index] = core;
        }
    }
    for (uint offset = 0u; offset < 4u; ++offset) {
        output_recurrent[state_indices[offset]] = state_values[offset];
    }
}
"""


_recurrence_column_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_recurrence_column_chunk",
    input_names=["recurrent", "key", "query", "value", "beta", "decay"],
    output_names=["output_recurrent", "output_core"],
    source=RECURRENCE_COLUMN_CHUNK_KERNEL_SOURCE,
)


CORE_GATE_CHUNK_KERNEL_SOURCE = r"""
uint head_index = threadgroup_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
threadgroup float inverse_variance[1];
float total = 0.0f;
uint base = lane * 4u;
for (uint offset = 0u; offset < 4u; ++offset) {
    float value = core[head_index * 128u + base + offset];
    volatile float square = value * value;
    total += square;
}
total = simd_sum(total);
if (lane == 0u) {
    volatile float mean = total / 128.0f;
    volatile float adjusted = mean + 1.0e-6f;
    inverse_variance[0] = metal::precise::rsqrt(adjusted);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint offset = 0u; offset < 4u; ++offset) {
    uint value_index = base + offset;
    uint index = head_index * 128u + value_index;
    volatile float normalized = core[index] * inverse_variance[0];
    bfloat16_t normalized_bf16 = bfloat16_t(normalized);
    volatile float weighted_product =
        float(normalized_bf16) * float(norm[value_index]);
    bfloat16_t weighted = bfloat16_t(weighted_product);
    float z_value = float(z[index]);
    float y = 1.0f / (1.0f + metal::exp(metal::abs(z_value)));
    float sigmoid_value = z_value < 0.0f ? y : 1.0f - y;
    volatile float silu_value = z_value * sigmoid_value;
    volatile float gated_value = float(weighted) * silu_value;
    output_gated[index] = bfloat16_t(gated_value);
}
"""


_core_gate_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_core_gate_chunk",
    input_names=["core", "z", "norm"],
    output_names=["output_gated"],
    source=CORE_GATE_CHUNK_KERNEL_SOURCE,
)


@dataclass(frozen=True)
class MLXGDNWeights:
    in_proj_qkv: mx.array
    in_proj_z: mx.array
    in_proj_b: mx.array
    in_proj_a: mx.array
    conv1d: mx.array
    dt_bias: mx.array
    a_log: mx.array
    norm: mx.array
    out_proj: mx.array


@dataclass(frozen=True)
class MLXGDNState:
    conv: mx.array
    recurrent: mx.array


def validate_weights(weights: MLXGDNWeights, config: GDNConfig) -> None:
    expected = {
        "in_proj_qkv": (config.conv_dim, config.hidden_size),
        "in_proj_z": (config.value_dim, config.hidden_size),
        "in_proj_b": (config.num_v_heads, config.hidden_size),
        "in_proj_a": (config.num_v_heads, config.hidden_size),
        "conv1d": (config.conv_dim, config.conv_kernel_size),
        "dt_bias": (config.num_v_heads,),
        "a_log": (config.num_v_heads,),
        "norm": (config.head_v_dim,),
        "out_proj": (config.hidden_size, config.value_dim),
    }
    for name, shape in expected.items():
        require(getattr(weights, name).shape == shape, f"{name} shape mismatch")
    model_dtype = weights.in_proj_qkv.dtype
    require(
        model_dtype in (mx.bfloat16, mx.float32),
        "GDN model dtype must be BF16 or reference FP32",
    )
    require(
        all(array.dtype == model_dtype for array in weights.__dict__.values()),
        "GDN weight dtype mismatch",
    )


def validate_state(state: MLXGDNState, config: GDNConfig) -> None:
    require(
        state.conv.shape == (config.conv_dim, config.conv_kernel_size),
        "convolution state shape mismatch",
    )
    require(
        state.recurrent.shape
        == (config.num_v_heads, config.head_k_dim, config.head_v_dim),
        "recurrent state shape mismatch",
    )
    require(state.recurrent.dtype == mx.float32, "recurrent state must be FP32")


def zeros_state(config: GDNConfig, conv_dtype: mx.Dtype = mx.bfloat16) -> MLXGDNState:
    return MLXGDNState(
        conv=mx.zeros((config.conv_dim, config.conv_kernel_size), dtype=conv_dtype),
        recurrent=mx.zeros(
            (config.num_v_heads, config.head_k_dim, config.head_v_dim),
            dtype=mx.float32,
        ),
    )


def _linear(weight: mx.array, vector: mx.array) -> mx.array:
    return mx.matmul(weight, vector)


def _linear_batch(weight: mx.array, vectors: mx.array) -> mx.array:
    return mx.vmap(lambda vector: _linear(weight, vector))(vectors)


def _softplus(value: mx.array) -> mx.array:
    return mx.maximum(value, 0.0) + mx.log1p(mx.exp(-mx.abs(value)))


def _silu(value: mx.array) -> mx.array:
    return value * mx.sigmoid(value)


def _l2norm(value: mx.array) -> mx.array:
    value32 = value.astype(mx.float32)
    return value32 * mx.rsqrt(mx.sum(value32 * value32, axis=-1, keepdims=True) + 1e-6)


def fused_conv_step(
    conv_state: mx.array,
    mixed: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Shift the production convolution state and evaluate its exact FP32 dot."""
    require(
        conv_state.dtype == mx.bfloat16 and conv_state.shape == (8192, 4),
        "fused convolution state mismatch",
    )
    require(
        mixed.dtype == mx.bfloat16 and mixed.shape == (8192,),
        "fused convolution input mismatch",
    )
    require(
        weight.dtype == mx.bfloat16 and weight.shape == (8192, 4),
        "fused convolution weight mismatch",
    )
    next_state, convolved = _conv_kernel(
        inputs=[conv_state, mixed, weight],
        grid=(8192, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(8192, 4), (8192,)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return next_state, convolved


def fused_qkv_conv_silu_step(
    hidden: mx.array,
    conv_state: mx.array,
    projection: mx.array,
    conv_weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Project QKV and emit the exact production convolution and BF16 SiLU."""
    require(
        hidden.dtype == mx.bfloat16 and hidden.shape == (2048,),
        "fused QKV hidden mismatch",
    )
    require(
        conv_state.dtype == mx.bfloat16 and conv_state.shape == (8192, 4),
        "fused QKV convolution state mismatch",
    )
    require(
        projection.dtype == mx.bfloat16 and projection.shape == (8192, 2048),
        "fused QKV projection mismatch",
    )
    require(
        conv_weight.dtype == mx.bfloat16 and conv_weight.shape == (8192, 4),
        "fused QKV convolution weight mismatch",
    )
    next_state, convolved = _qkv_conv_silu_kernel(
        inputs=[projection, hidden, conv_state, conv_weight],
        grid=(262_144, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(8192, 4), (8192,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return next_state, convolved


def fused_recurrence_step(
    recurrent: mx.array,
    key: mx.array,
    query: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array,
    *,
    simdgroups: int = 32,
) -> tuple[mx.array, mx.array]:
    """Evaluate the exact production recurrence without temporary state tensors."""
    expected = {
        "recurrent": (recurrent, (32, 128, 128)),
        "key": (key, (32, 128)),
        "query": (query, (32, 128)),
        "value": (value, (32, 128)),
        "beta": (beta, (32,)),
        "decay": (decay, (32,)),
    }
    for name, (array, shape) in expected.items():
        require(array.dtype == mx.float32, f"fused recurrence {name} must be FP32")
        require(array.shape == shape, f"fused recurrence {name} shape mismatch")
    require(simdgroups in (8, 16, 32), "invalid recurrence SIMD-group count")
    threads = simdgroups * 32
    next_recurrent, core = _recurrence_kernel(
        inputs=[recurrent, key, query, value, beta, decay],
        template=[("SIMDGROUPS", simdgroups)],
        grid=(32 * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(32, 128, 128), (32, 128)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return next_recurrent, core


def fused_recurrence_core_gate_step(
    recurrent: mx.array,
    key: mx.array,
    query: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array,
    z: mx.array,
    norm: mx.array,
    *,
    simdgroups: int = 32,
) -> tuple[mx.array, mx.array]:
    """Keep the exact recurrence core resident through normalization and gating."""
    expected = {
        "recurrent": (recurrent, mx.float32, (32, 128, 128)),
        "key": (key, mx.float32, (32, 128)),
        "query": (query, mx.float32, (32, 128)),
        "value": (value, mx.float32, (32, 128)),
        "beta": (beta, mx.float32, (32,)),
        "decay": (decay, mx.float32, (32,)),
        "z": (z, mx.bfloat16, (32, 128)),
        "norm": (norm, mx.bfloat16, (128,)),
    }
    for name, (array, dtype, shape) in expected.items():
        require(array.dtype == dtype, f"fused recurrence/core {name} dtype mismatch")
        require(array.shape == shape, f"fused recurrence/core {name} shape mismatch")
    require(simdgroups in (8, 16, 32), "invalid recurrence SIMD-group count")
    threads = simdgroups * 32
    next_recurrent, gated = _recurrence_core_gate_kernel(
        inputs=[recurrent, key, query, value, beta, decay, z, norm],
        template=[("SIMDGROUPS", simdgroups)],
        grid=(32 * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(32, 128, 128), (32, 128)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )
    return next_recurrent, gated


def fused_conv_chunk(
    conv_state: mx.array,
    mixed: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Evaluate a nonempty production convolution chunk in token order."""
    require(
        conv_state.dtype == mx.bfloat16 and conv_state.shape == (8192, 4),
        "fused convolution chunk state mismatch",
    )
    require(
        mixed.dtype == mx.bfloat16 and mixed.ndim == 2 and mixed.shape[0] > 0
        and mixed.shape[1] == 8192,
        "fused convolution chunk input mismatch",
    )
    require(
        weight.dtype == mx.bfloat16 and weight.shape == (8192, 4),
        "fused convolution chunk weight mismatch",
    )
    tokens = mixed.shape[0]
    next_state, convolved = _conv_chunk_kernel(
        inputs=[conv_state, mixed, weight],
        template=[("TOKENS", tokens)],
        grid=(8192, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(8192, 4), (tokens, 8192)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return next_state, convolved


def fused_recurrence_core_gate_chunk(
    recurrent: mx.array,
    key: mx.array,
    query: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array,
    z: mx.array,
    norm: mx.array,
    *,
    simdgroups: int = 32,
) -> tuple[mx.array, mx.array]:
    """Advance the exact production recurrence through a token chunk."""
    require(key.ndim == 3 and key.shape[0] > 0, "fused recurrence chunk is empty")
    tokens = key.shape[0]
    expected = {
        "recurrent": (recurrent, mx.float32, (32, 128, 128)),
        "key": (key, mx.float32, (tokens, 32, 128)),
        "query": (query, mx.float32, (tokens, 32, 128)),
        "value": (value, mx.float32, (tokens, 32, 128)),
        "beta": (beta, mx.float32, (tokens, 32)),
        "decay": (decay, mx.float32, (tokens, 32)),
        "z": (z, mx.bfloat16, (tokens, 32, 128)),
        "norm": (norm, mx.bfloat16, (128,)),
    }
    for name, (array, dtype, shape) in expected.items():
        require(array.dtype == dtype, f"fused recurrence chunk {name} dtype mismatch")
        require(array.shape == shape, f"fused recurrence chunk {name} shape mismatch")
    require(simdgroups in (8, 16, 32), "invalid recurrence chunk SIMD-group count")
    threads = simdgroups * 32
    next_recurrent, gated = _recurrence_core_gate_chunk_kernel(
        inputs=[recurrent, key, query, value, beta, decay, z, norm],
        template=[("TOKENS", tokens), ("SIMDGROUPS", simdgroups)],
        grid=(32 * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(32, 128, 128), (tokens, 32, 128)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )
    return next_recurrent, gated


def fused_recurrence_core_gate_column_chunk(
    recurrent: mx.array,
    key: mx.array,
    query: mx.array,
    value: mx.array,
    beta: mx.array,
    decay: mx.array,
    z: mx.array,
    norm: mx.array,
    *,
    simdgroups: int = 32,
) -> tuple[mx.array, mx.array]:
    """Keep independent recurrent columns in registers through a token chunk."""
    require(key.ndim == 3 and key.shape[0] > 0, "column recurrence chunk is empty")
    tokens = key.shape[0]
    expected = {
        "recurrent": (recurrent, mx.float32, (32, 128, 128)),
        "key": (key, mx.float32, (tokens, 32, 128)),
        "query": (query, mx.float32, (tokens, 32, 128)),
        "value": (value, mx.float32, (tokens, 32, 128)),
        "beta": (beta, mx.float32, (tokens, 32)),
        "decay": (decay, mx.float32, (tokens, 32)),
        "z": (z, mx.bfloat16, (tokens, 32, 128)),
        "norm": (norm, mx.bfloat16, (128,)),
    }
    for name, (array, dtype, shape) in expected.items():
        require(array.dtype == dtype, f"column recurrence chunk {name} dtype mismatch")
        require(array.shape == shape, f"column recurrence chunk {name} shape mismatch")
    require(simdgroups in (8, 16, 32), "invalid column recurrence SIMD-group count")
    threads = simdgroups * 32
    next_recurrent, core = _recurrence_column_chunk_kernel(
        inputs=[recurrent, key, query, value, beta, decay],
        template=[("TOKENS", tokens), ("SIMDGROUPS", simdgroups)],
        grid=(32 * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(32, 128, 128), (tokens, 32, 128)],
        output_dtypes=[mx.float32, mx.float32],
    )
    gated = _core_gate_chunk_kernel(
        inputs=[core, z, norm],
        grid=(tokens * 32 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(tokens, 32, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return next_recurrent, gated


def decode_step(
    hidden: mx.array,
    state: MLXGDNState,
    weights: MLXGDNWeights,
    config: GDNConfig = PRODUCTION_CONFIG,
    *,
    fused_convolution: bool = True,
    fused_recurrence: bool = True,
    fused_core_gate_output: bool = True,
    _validated: bool = False,
) -> tuple[mx.array, MLXGDNState]:
    """Append one token without mutating the caller's rollback state."""
    require(hidden.ndim == 1 and hidden.shape == (config.hidden_size,), "hidden-state shape mismatch")
    if not _validated:
        validate_state(state, config)
        validate_weights(weights, config)

    model_dtype = weights.in_proj_qkv.dtype
    require(state.conv.dtype == model_dtype, "convolution state dtype mismatch")
    hidden = hidden.astype(model_dtype)
    z = _linear(weights.in_proj_z, hidden)
    b = _linear(weights.in_proj_b, hidden)
    a = _linear(weights.in_proj_a, hidden)

    production = config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16
    if fused_convolution and production:
        next_conv, convolved = fused_qkv_conv_silu_step(
            hidden,
            state.conv,
            weights.in_proj_qkv,
            weights.conv1d,
        )
    else:
        mixed = _linear(weights.in_proj_qkv, hidden)
        next_conv = mx.concatenate([state.conv[:, 1:], mixed[:, None]], axis=1)
        convolved32 = mx.sum(
            next_conv.astype(mx.float32) * weights.conv1d.astype(mx.float32),
            axis=1,
        )
        convolved = _silu(convolved32).astype(model_dtype)

    query_end = config.key_dim
    key_end = query_end + config.key_dim
    query = convolved[:query_end].reshape(config.num_k_heads, config.head_k_dim)
    key = convolved[query_end:key_end].reshape(config.num_k_heads, config.head_k_dim)
    value = convolved[key_end:].reshape(config.num_v_heads, config.head_v_dim)
    query = _l2norm(query)
    key = _l2norm(key)
    repeats = config.num_v_heads // config.num_k_heads
    if repeats > 1:
        query = mx.repeat(query, repeats, axis=0)
        key = mx.repeat(key, repeats, axis=0)

    beta = mx.sigmoid(b.astype(mx.float32))
    decay_log = -mx.exp(weights.a_log.astype(mx.float32)) * _softplus(
        a.astype(mx.float32) + weights.dt_bias.astype(mx.float32)
    )
    query = query * (config.head_k_dim**-0.5)
    decay = mx.exp(decay_log)
    value32 = value.astype(mx.float32)
    z_heads = z.reshape(config.num_v_heads, config.head_v_dim)
    if fused_recurrence and fused_core_gate_output and production:
        recurrent, gated = fused_recurrence_core_gate_step(
            state.recurrent,
            key,
            query,
            value32,
            beta,
            decay,
            z_heads,
            weights.norm,
        )
    else:
        if fused_recurrence and production:
            recurrent, core = fused_recurrence_step(
                state.recurrent,
                key,
                query,
                value32,
                beta,
                decay,
            )
        else:
            decayed = state.recurrent * decay[:, None, None]
            memory = mx.sum(decayed * key[:, :, None], axis=1)
            delta = (value32 - memory) * beta[:, None]
            recurrent = decayed + key[:, :, None] * delta[:, None, :]
            core = mx.sum(recurrent * query[:, :, None], axis=1)

        variance = mx.mean(core * core, axis=-1, keepdims=True)
        normalized = core * mx.rsqrt(variance + config.rms_norm_eps)
        weighted = (
            normalized.astype(model_dtype) * weights.norm.astype(model_dtype)
        ).astype(model_dtype)
        gated = (
            weighted.astype(mx.float32) * _silu(z_heads.astype(mx.float32))
        ).astype(model_dtype)
    output = _linear(weights.out_proj, gated.reshape(config.value_dim))
    return output, MLXGDNState(conv=next_conv, recurrent=recurrent)


def prefill_chunk(
    hidden: mx.array,
    state: MLXGDNState,
    weights: MLXGDNWeights,
    config: GDNConfig = PRODUCTION_CONFIG,
) -> tuple[mx.array, MLXGDNState]:
    """Evaluate a nonempty token chunk and return only its final cache state."""
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0 and hidden.shape[1] == config.hidden_size,
        "GDN prefill hidden-state shape mismatch",
    )
    validate_state(state, config)
    validate_weights(weights, config)
    model_dtype = weights.in_proj_qkv.dtype
    require(state.conv.dtype == model_dtype, "convolution state dtype mismatch")
    if config != PRODUCTION_CONFIG or model_dtype != mx.bfloat16:
        outputs = []
        next_state = state
        for token in hidden:
            output, next_state = decode_step(token, next_state, weights, config)
            outputs.append(output)
        return mx.stack(outputs), next_state

    hidden = hidden.astype(model_dtype)
    mixed = _linear_batch(weights.in_proj_qkv, hidden)
    z = _linear_batch(weights.in_proj_z, hidden)
    b = _linear_batch(weights.in_proj_b, hidden)
    a = _linear_batch(weights.in_proj_a, hidden)

    next_conv, convolved32 = fused_conv_chunk(state.conv, mixed, weights.conv1d)
    convolved = _silu(convolved32).astype(model_dtype)
    tokens = hidden.shape[0]
    query_end = config.key_dim
    key_end = query_end + config.key_dim
    query = convolved[:, :query_end].reshape(tokens, config.num_k_heads, config.head_k_dim)
    key = convolved[:, query_end:key_end].reshape(tokens, config.num_k_heads, config.head_k_dim)
    value = convolved[:, key_end:].reshape(tokens, config.num_v_heads, config.head_v_dim)
    query = _l2norm(query)
    key = _l2norm(key)
    repeats = config.num_v_heads // config.num_k_heads
    if repeats > 1:
        query = mx.repeat(query, repeats, axis=1)
        key = mx.repeat(key, repeats, axis=1)

    beta = mx.sigmoid(b.astype(mx.float32))
    decay_log = -mx.exp(weights.a_log.astype(mx.float32))[None, :] * _softplus(
        a.astype(mx.float32) + weights.dt_bias.astype(mx.float32)[None, :]
    )
    query = query * (config.head_k_dim**-0.5)
    decay = mx.exp(decay_log)
    recurrent, gated = fused_recurrence_core_gate_column_chunk(
        state.recurrent,
        key,
        query,
        value.astype(mx.float32),
        beta,
        decay,
        z.reshape(tokens, config.num_v_heads, config.head_v_dim),
        weights.norm,
    )
    output = _linear_batch(weights.out_proj, gated.reshape(tokens, config.value_dim))
    return output, MLXGDNState(conv=next_conv, recurrent=recurrent)


def _load_bf16(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "BF16", f"expected BF16 tensor: {name}")
    require(entry.get("shape") == list(shape), f"tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    expected_bytes = 2
    for size in shape:
        expected_bytes *= size
    require(len(payload) == expected_bytes, f"tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.bfloat16).reshape(shape)


def load_layer(source_path: Path, layer: int) -> MLXGDNWeights:
    require(layer >= 0 and layer % 4 != 3 and layer < 40, "layer is not an Ornith GDN layer")
    prefix = f"model.language_model.layers.{layer}.linear_attn"
    with SafetensorsFile(source_path) as source:
        weights = MLXGDNWeights(
            in_proj_qkv=_load_bf16(source, f"{prefix}.in_proj_qkv.weight", (8192, 2048)),
            in_proj_z=_load_bf16(source, f"{prefix}.in_proj_z.weight", (4096, 2048)),
            in_proj_b=_load_bf16(source, f"{prefix}.in_proj_b.weight", (32, 2048)),
            in_proj_a=_load_bf16(source, f"{prefix}.in_proj_a.weight", (32, 2048)),
            conv1d=_load_bf16(source, f"{prefix}.conv1d.weight", (8192, 1, 4)).reshape(8192, 4),
            dt_bias=_load_bf16(source, f"{prefix}.dt_bias", (32,)),
            a_log=_load_bf16(source, f"{prefix}.A_log", (32,)),
            norm=_load_bf16(source, f"{prefix}.norm.weight", (128,)),
            out_proj=_load_bf16(source, f"{prefix}.out_proj.weight", (2048, 4096)),
        )
        mx.eval(*weights.__dict__.values())
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights

#!/usr/bin/env python3
"""One-token MLX GatedDeltaNet composition for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

import ornith35_mlx_dense as dense
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


QKV_Z_CONV_SILU_CHUNK_KERNEL_SOURCE = r"""
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x * SIMDGROUPS + group;
if (row >= 12288u) return;
float sums[TOKENS];
for (uint token = 0u; token < TOKENS; ++token) {
    sums[token] = 0.0f;
}
for (uint column = lane * 4u; column < 2048u; column += 128u) {
    uint weight_base = (row < 8192u ? row : row - 8192u) * 2048u + column;
    float weight0;
    float weight1;
    float weight2;
    float weight3;
    if (row < 8192u) {
        weight0 = float(projection[weight_base]);
        weight1 = float(projection[weight_base + 1u]);
        weight2 = float(projection[weight_base + 2u]);
        weight3 = float(projection[weight_base + 3u]);
    } else {
        weight0 = float(z_projection[weight_base]);
        weight1 = float(z_projection[weight_base + 1u]);
        weight2 = float(z_projection[weight_base + 2u]);
        weight3 = float(z_projection[weight_base + 3u]);
    }
    for (uint token = 0u; token < TOKENS; ++token) {
        uint input_base = token * 2048u + column;
        sums[token] += weight0 * float(hidden[input_base]);
        sums[token] += weight1 * float(hidden[input_base + 1u]);
        sums[token] += weight2 * float(hidden[input_base + 2u]);
        sums[token] += weight3 * float(hidden[input_base + 3u]);
    }
}
for (uint token = 0u; token < TOKENS; ++token) {
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        sums[token] += simd_shuffle_down(sums[token], offset);
    }
}
if (lane == 0u && row < 8192u) {
    uint state_base = row * 4u;
    bfloat16_t first = conv_state[state_base];
    bfloat16_t second = conv_state[state_base + 1u];
    bfloat16_t third = conv_state[state_base + 2u];
    bfloat16_t fourth = conv_state[state_base + 3u];
    for (uint token = 0u; token < TOKENS; ++token) {
        first = second;
        second = third;
        third = fourth;
        fourth = bfloat16_t(sums[token]);
        float total = 0.0f;
        volatile float product0 =
            float(first) * float(conv_weight[state_base]);
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
        output_convolved[token * 8192u + row] = bfloat16_t(silu_value);
    }
    output_state[state_base] = first;
    output_state[state_base + 1u] = second;
    output_state[state_base + 2u] = third;
    output_state[state_base + 3u] = fourth;
} else if (lane == 0u) {
    uint z_row = row - 8192u;
    for (uint token = 0u; token < TOKENS; ++token) {
        output_z[token * 4096u + z_row] = bfloat16_t(sums[token]);
    }
}
"""


_qkv_z_conv_silu_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_qkv_z_conv_silu_chunk_bf16",
    input_names=[
        "projection",
        "z_projection",
        "hidden",
        "conv_state",
        "conv_weight",
    ],
    output_names=["output_state", "output_convolved", "output_z"],
    source=QKV_Z_CONV_SILU_CHUNK_KERNEL_SOURCE,
)


# QKV/z use MLX's one-SIMD-per-row GEMV. The 32-row b/a projections use eight
# SIMD groups across K and four output rows per threadgroup; both trees remain
# distinct inside this joined dispatch so their BF16 boundaries stay exact.
QKV_CONV_SILU_TRANSITION_KERNEL_SOURCE = r"""
uint task = threadgroup_position_in_grid.x;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
threadgroup float b_partials[32];
threadgroup float a_partials[32];
if (task < 1024u) {
    uint row = task * 8u + group;
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
    return;
}

if (task < 1536u) {
    uint row = (task - 1024u) * 8u + group;
    float sum = 0.0f;
    for (uint column = lane * 4u; column < 2048u; column += 128u) {
        float input0 = float(hidden[column]);
        float input1 = float(hidden[column + 1u]);
        float input2 = float(hidden[column + 2u]);
        float input3 = float(hidden[column + 3u]);
        uint weight_base = row * 2048u + column;
        sum += float(z_projection[weight_base]) * input0;
        sum += float(z_projection[weight_base + 1u]) * input1;
        sum += float(z_projection[weight_base + 2u]) * input2;
        sum += float(z_projection[weight_base + 3u]) * input3;
    }
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        sum += simd_shuffle_down(sum, offset);
    }
    if (lane == 0u) output_z[row] = bfloat16_t(sum);
    return;
}

uint output_group = task - 1536u;
float b_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float a_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint iteration = 0u; iteration < 2u; ++iteration) {
    uint column = iteration * 1024u + group * 128u + lane * 4u;
    float inputs[4] = {
        float(hidden[column]),
        float(hidden[column + 1u]),
        float(hidden[column + 2u]),
        float(hidden[column + 3u])
    };
    for (uint row_offset = 0u; row_offset < 4u; ++row_offset) {
        uint row = output_group * 4u + row_offset;
        uint weight_base = row * 2048u + column;
        b_sums[row_offset] += float(b_projection[weight_base]) * inputs[0];
        b_sums[row_offset] +=
            float(b_projection[weight_base + 1u]) * inputs[1];
        b_sums[row_offset] +=
            float(b_projection[weight_base + 2u]) * inputs[2];
        b_sums[row_offset] +=
            float(b_projection[weight_base + 3u]) * inputs[3];
        a_sums[row_offset] += float(a_projection[weight_base]) * inputs[0];
        a_sums[row_offset] +=
            float(a_projection[weight_base + 1u]) * inputs[1];
        a_sums[row_offset] +=
            float(a_projection[weight_base + 2u]) * inputs[2];
        a_sums[row_offset] +=
            float(a_projection[weight_base + 3u]) * inputs[3];
    }
}
for (uint row_offset = 0u; row_offset < 4u; ++row_offset) {
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        b_sums[row_offset] += simd_shuffle_down(b_sums[row_offset], offset);
        a_sums[row_offset] += simd_shuffle_down(a_sums[row_offset], offset);
    }
    if (lane == 0u) {
        b_partials[group * 4u + row_offset] = b_sums[row_offset];
        a_partials[group * 4u + row_offset] = a_sums[row_offset];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u && lane == 0u) {
    for (uint row_offset = 0u; row_offset < 4u; ++row_offset) {
        float b_total = b_partials[row_offset];
        float a_total = a_partials[row_offset];
        for (uint partial = 1u; partial < 8u; ++partial) {
            b_total += b_partials[partial * 4u + row_offset];
            a_total += a_partials[partial * 4u + row_offset];
        }
        uint row = output_group * 4u + row_offset;
        float b_value = float(bfloat16_t(b_total));
        float sigmoid_inverse = 1.0f / (
            1.0f + metal::precise::exp(metal::abs(b_value))
        );
        output_beta[row] = b_value < 0.0f
            ? sigmoid_inverse
            : 1.0f - sigmoid_inverse;

        float combined = float(bfloat16_t(a_total)) + float(dt_bias[row]);
        float exponent = metal::precise::exp(-metal::abs(combined));
        float exponent_plus_one = 1.0f + exponent;
        float logarithm;
        if (exponent_plus_one == Limits<float>::max) {
            logarithm = Limits<float>::max;
        } else if (exponent_plus_one == 1.0f) {
            logarithm = exponent;
        } else {
            logarithm = exponent * (
                metal::precise::log(exponent_plus_one) /
                (exponent_plus_one - 1.0f)
            );
        }
        float softplus = max(combined, 0.0f) + logarithm;
        float decay_log = -metal::precise::exp(float(a_log[row])) * softplus;
        output_decay[row] = metal::precise::exp(decay_log);
    }
}
"""


_qkv_conv_silu_transition_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_qkv_conv_silu_transition_exact",
    input_names=[
        "projection",
        "hidden",
        "conv_state",
        "conv_weight",
        "z_projection",
        "b_projection",
        "a_projection",
        "dt_bias",
        "a_log",
    ],
    output_names=[
        "output_state",
        "output_convolved",
        "output_z",
        "output_beta",
        "output_decay",
    ],
    source=QKV_CONV_SILU_TRANSITION_KERNEL_SOURCE,
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


RECURRENCE_CONVOLVED_CORE_GATE_KERNEL_SOURCE = r"""
uint head = threadgroup_position_in_grid.x;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
threadgroup float core_values[128];
threadgroup float inverse_norm[2];
threadgroup float inverse_variance[1];
uint source_head = head / 2u;
if (group == 0u) {
    float query_total = 0.0f;
    float key_total = 0.0f;
    uint query_base = source_head * 128u + lane * 4u;
    uint key_base = 2048u + query_base;
    for (uint offset = 0u; offset < 4u; ++offset) {
        float query_value = float(convolved[query_base + offset]);
        float key_value = float(convolved[key_base + offset]);
        volatile float query_square = query_value * query_value;
        volatile float key_square = key_value * key_value;
        query_total += query_square;
        key_total += key_square;
    }
    query_total = simd_sum(query_total);
    key_total = simd_sum(key_total);
    if (lane == 0u) {
        volatile float query_adjusted = query_total + 1.0e-6f;
        volatile float key_adjusted = key_total + 1.0e-6f;
        inverse_norm[0] = metal::precise::rsqrt(query_adjusted);
        inverse_norm[1] = metal::precise::rsqrt(key_adjusted);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float decay_value = decay[head];
float beta_value = beta[head];
for (uint value_index = group; value_index < 128u; value_index += SIMDGROUPS) {
    float memory = 0.0f;
    uint state_indices[4];
    float decayed_values[4];
    float key_values[4];
    float query_values[4];
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint key_index = lane + offset * 32u;
        uint state_index = (head * 128u + key_index) * 128u + value_index;
        float decayed = recurrent[state_index] * decay_value;
        uint source_index = source_head * 128u + key_index;
        volatile float key_value =
            float(convolved[2048u + source_index]) * inverse_norm[1];
        volatile float query_normalized =
            float(convolved[source_index]) * inverse_norm[0];
        volatile float query_value = query_normalized * 0.08838834764831845f;
        state_indices[offset] = state_index;
        decayed_values[offset] = decayed;
        key_values[offset] = key_value;
        query_values[offset] = query_value;
        volatile float memory_term = decayed * key_value;
        memory += memory_term;
    }
    memory = simd_sum(memory);
    memory = simd_broadcast_first(memory);
    float value = float(convolved[4096u + head * 128u + value_index]);
    float delta = (value - memory) * beta_value;
    float core = 0.0f;
    for (uint offset = 0u; offset < 4u; ++offset) {
        volatile float update = key_values[offset] * delta;
        float next = decayed_values[offset] + update;
        output_recurrent[state_indices[offset]] = next;
        volatile float core_term = next * query_values[offset];
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


_recurrence_convolved_core_gate_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_recurrence_convolved_core_gate",
    input_names=["recurrent", "convolved", "beta", "decay", "z", "norm"],
    output_names=["output_recurrent", "output_gated"],
    source=RECURRENCE_CONVOLVED_CORE_GATE_KERNEL_SOURCE,
)


BETA_DECAY_KERNEL_SOURCE = r"""
uint head = thread_position_in_grid.x;
if (head >= 32u) return;
float b_value = float(b[head]);
float sigmoid_inverse = 1.0f / (
    1.0f + metal::precise::exp(metal::abs(b_value))
);
output_beta[head] = b_value < 0.0f
    ? sigmoid_inverse
    : 1.0f - sigmoid_inverse;

float combined = float(a[head]) + float(dt_bias[head]);
float exponent = metal::precise::exp(-metal::abs(combined));
float exponent_plus_one = 1.0f + exponent;
float logarithm;
if (exponent_plus_one == Limits<float>::max) {
    logarithm = Limits<float>::max;
} else if (exponent_plus_one == 1.0f) {
    logarithm = exponent;
} else {
    logarithm = exponent * (
        metal::precise::log(exponent_plus_one) / (exponent_plus_one - 1.0f)
    );
}
float softplus = max(combined, 0.0f) + logarithm;
float decay_log = -metal::precise::exp(float(a_log[head])) * softplus;
output_decay[head] = metal::precise::exp(decay_log);
"""


_beta_decay_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_beta_decay_exact",
    input_names=["b", "a", "dt_bias", "a_log"],
    output_names=["output_beta", "output_decay"],
    source=BETA_DECAY_KERNEL_SOURCE,
)


BETA_DECAY_CHUNK_KERNEL_SOURCE = r"""
uint index = thread_position_in_grid.x;
if (index >= TOKENS * 32u) return;
uint head = index % 32u;
float b_value = float(b[index]);
float sigmoid_inverse = 1.0f / (
    1.0f + metal::precise::exp(metal::abs(b_value))
);
output_beta[index] = b_value < 0.0f
    ? sigmoid_inverse
    : 1.0f - sigmoid_inverse;

float combined = float(a[index]) + float(dt_bias[head]);
float exponent = metal::precise::exp(-metal::abs(combined));
float exponent_plus_one = 1.0f + exponent;
float logarithm;
if (exponent_plus_one == Limits<float>::max) {
    logarithm = Limits<float>::max;
} else if (exponent_plus_one == 1.0f) {
    logarithm = exponent;
} else {
    logarithm = exponent * (
        metal::precise::log(exponent_plus_one) / (exponent_plus_one - 1.0f)
    );
}
float softplus = max(combined, 0.0f) + logarithm;
float decay_log = -metal::precise::exp(float(a_log[head])) * softplus;
output_decay[index] = metal::precise::exp(decay_log);
"""


_beta_decay_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_beta_decay_chunk_exact",
    input_names=["b", "a", "dt_bias", "a_log"],
    output_names=["output_beta", "output_decay"],
    source=BETA_DECAY_CHUNK_KERNEL_SOURCE,
)


BA_BETA_DECAY_CHUNK_KERNEL_SOURCE = r"""
uint task = threadgroup_position_in_grid.x;
uint token = task / 8u;
uint output_group = task - token * 8u;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
threadgroup float b_partials[32];
threadgroup float a_partials[32];
float b_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float a_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint iteration = 0u; iteration < 2u; ++iteration) {
    uint column = iteration * 1024u + group * 128u + lane * 4u;
    uint input_base = token * 2048u + column;
    float inputs[4] = {
        float(hidden[input_base]),
        float(hidden[input_base + 1u]),
        float(hidden[input_base + 2u]),
        float(hidden[input_base + 3u])
    };
    for (uint row_offset = 0u; row_offset < 4u; ++row_offset) {
        uint row = output_group * 4u + row_offset;
        uint weight_base = row * 2048u + column;
        b_sums[row_offset] += float(b_projection[weight_base]) * inputs[0];
        b_sums[row_offset] +=
            float(b_projection[weight_base + 1u]) * inputs[1];
        b_sums[row_offset] +=
            float(b_projection[weight_base + 2u]) * inputs[2];
        b_sums[row_offset] +=
            float(b_projection[weight_base + 3u]) * inputs[3];
        a_sums[row_offset] += float(a_projection[weight_base]) * inputs[0];
        a_sums[row_offset] +=
            float(a_projection[weight_base + 1u]) * inputs[1];
        a_sums[row_offset] +=
            float(a_projection[weight_base + 2u]) * inputs[2];
        a_sums[row_offset] +=
            float(a_projection[weight_base + 3u]) * inputs[3];
    }
}
for (uint row_offset = 0u; row_offset < 4u; ++row_offset) {
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        b_sums[row_offset] += simd_shuffle_down(b_sums[row_offset], offset);
        a_sums[row_offset] += simd_shuffle_down(a_sums[row_offset], offset);
    }
    if (lane == 0u) {
        b_partials[group * 4u + row_offset] = b_sums[row_offset];
        a_partials[group * 4u + row_offset] = a_sums[row_offset];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u && lane == 0u) {
    for (uint row_offset = 0u; row_offset < 4u; ++row_offset) {
        uint row = output_group * 4u + row_offset;
        uint output_index = token * 32u + row;
        float b_total = b_partials[row_offset];
        float a_total = a_partials[row_offset];
        for (uint partial = 1u; partial < 8u; ++partial) {
            b_total += b_partials[partial * 4u + row_offset];
            a_total += a_partials[partial * 4u + row_offset];
        }
        float b_value = float(bfloat16_t(b_total));
        float sigmoid_inverse = 1.0f / (
            1.0f + metal::precise::exp(metal::abs(b_value))
        );
        output_beta[output_index] = b_value < 0.0f
            ? sigmoid_inverse
            : 1.0f - sigmoid_inverse;

        float combined = float(bfloat16_t(a_total)) + float(dt_bias[row]);
        float exponent = metal::precise::exp(-metal::abs(combined));
        float exponent_plus_one = 1.0f + exponent;
        float logarithm;
        if (exponent_plus_one == Limits<float>::max) {
            logarithm = Limits<float>::max;
        } else if (exponent_plus_one == 1.0f) {
            logarithm = exponent;
        } else {
            logarithm = exponent * (
                metal::precise::log(exponent_plus_one) /
                (exponent_plus_one - 1.0f)
            );
        }
        float softplus = max(combined, 0.0f) + logarithm;
        float decay_log = -metal::precise::exp(float(a_log[row])) * softplus;
        output_decay[output_index] = metal::precise::exp(decay_log);
    }
}
"""


_ba_beta_decay_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_ba_beta_decay_chunk_exact",
    input_names=["hidden", "b_projection", "a_projection", "dt_bias", "a_log"],
    output_names=["output_beta", "output_decay"],
    source=BA_BETA_DECAY_CHUNK_KERNEL_SOURCE,
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


RECURRENCE_COLUMN_CORE_GATE_SMALL_CHUNK_KERNEL_SOURCE = r"""
threadgroup float core_values[TOKENS * 128];
threadgroup float inverse_variance[TOKENS];
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
            core_values[token * 128u + value_index] = core;
        }
    }
    for (uint offset = 0u; offset < 4u; ++offset) {
        output_recurrent[state_indices[offset]] = state_values[offset];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group < TOKENS) {
    uint token = group;
    uint head_index = token * 32u + head;
    uint base = lane * 4u;
    float total = 0.0f;
    for (uint offset = 0u; offset < 4u; ++offset) {
        float core = core_values[token * 128u + base + offset];
        volatile float square = core * core;
        total += square;
    }
    total = simd_sum(total);
    if (lane == 0u) {
        volatile float mean = total / 128.0f;
        volatile float adjusted = mean + 1.0e-6f;
        inverse_variance[token] = metal::precise::rsqrt(adjusted);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group < TOKENS) {
    uint token = group;
    uint head_index = token * 32u + head;
    uint base = lane * 4u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint value_index = base + offset;
        uint index = head_index * 128u + value_index;
        volatile float normalized =
            core_values[token * 128u + value_index] * inverse_variance[token];
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


_recurrence_column_core_gate_small_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_recurrence_column_core_gate_small_chunk",
    input_names=["recurrent", "key", "query", "value", "beta", "decay", "z", "norm"],
    output_names=["output_recurrent", "output_gated"],
    source=RECURRENCE_COLUMN_CORE_GATE_SMALL_CHUNK_KERNEL_SOURCE,
)


RECURRENCE_CONVOLVED_COLUMN_CORE_GATE_SMALL_CHUNK_KERNEL_SOURCE = r"""
threadgroup float inverse_query[TOKENS];
threadgroup float inverse_key[TOKENS];
threadgroup float core_values[TOKENS * 128];
threadgroup float inverse_variance[TOKENS];
uint head = threadgroup_position_in_grid.x;
uint source_head = head / 2u;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
if (group == 0u) {
    for (uint token = 0u; token < TOKENS; ++token) {
        uint token_base = token * 8192u;
        uint query_base = token_base + source_head * 128u + lane * 4u;
        uint key_base = query_base + 2048u;
        float query_total = 0.0f;
        float key_total = 0.0f;
        for (uint offset = 0u; offset < 4u; ++offset) {
            float query_value = float(convolved[query_base + offset]);
            float key_value = float(convolved[key_base + offset]);
            volatile float query_square = query_value * query_value;
            volatile float key_square = key_value * key_value;
            query_total += query_square;
            key_total += key_square;
        }
        query_total = simd_sum(query_total);
        key_total = simd_sum(key_total);
        if (lane == 0u) {
            volatile float query_adjusted = query_total + 1.0e-6f;
            volatile float key_adjusted = key_total + 1.0e-6f;
            inverse_query[token] = metal::precise::rsqrt(query_adjusted);
            inverse_key[token] = metal::precise::rsqrt(key_adjusted);
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
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
        uint token_base = token * 8192u;
        float decay_value = decay[head_index];
        float beta_value = beta[head_index];
        float decayed_values[4];
        float key_values[4];
        float query_values[4];
        float memory = 0.0f;
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint key_index = lane + offset * 32u;
            float decayed = state_values[offset] * decay_value;
            uint source_index = token_base + source_head * 128u + key_index;
            volatile float key_value =
                float(convolved[source_index + 2048u]) * inverse_key[token];
            volatile float query_normalized =
                float(convolved[source_index]) * inverse_query[token];
            volatile float query_value =
                query_normalized * 0.08838834764831845f;
            decayed_values[offset] = decayed;
            key_values[offset] = key_value;
            query_values[offset] = query_value;
            volatile float memory_term = decayed * key_value;
            memory += memory_term;
        }
        memory = simd_sum(memory);
        memory = simd_broadcast_first(memory);
        float value = float(
            convolved[token_base + 4096u + head * 128u + value_index]
        );
        float delta = (value - memory) * beta_value;
        float core = 0.0f;
        for (uint offset = 0u; offset < 4u; ++offset) {
            volatile float update = key_values[offset] * delta;
            float next = decayed_values[offset] + update;
            state_values[offset] = next;
            volatile float core_term = next * query_values[offset];
            core += core_term;
        }
        core = simd_sum(core);
        if (lane == 0u) {
            core_values[token * 128u + value_index] = core;
        }
    }
    for (uint offset = 0u; offset < 4u; ++offset) {
        output_recurrent[state_indices[offset]] = state_values[offset];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group < TOKENS) {
    uint token = group;
    uint head_index = token * 32u + head;
    uint base = lane * 4u;
    float total = 0.0f;
    for (uint offset = 0u; offset < 4u; ++offset) {
        float core = core_values[token * 128u + base + offset];
        volatile float square = core * core;
        total += square;
    }
    total = simd_sum(total);
    if (lane == 0u) {
        volatile float mean = total / 128.0f;
        volatile float adjusted = mean + 1.0e-6f;
        inverse_variance[token] = metal::precise::rsqrt(adjusted);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group < TOKENS) {
    uint token = group;
    uint head_index = token * 32u + head;
    uint base = lane * 4u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint value_index = base + offset;
        uint index = head_index * 128u + value_index;
        volatile float normalized =
            core_values[token * 128u + value_index] * inverse_variance[token];
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


_recurrence_convolved_column_core_gate_small_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_gdn_recurrence_convolved_column_core_gate_small_chunk",
    input_names=["recurrent", "convolved", "beta", "decay", "z", "norm"],
    output_names=["output_recurrent", "output_gated"],
    source=RECURRENCE_CONVOLVED_COLUMN_CORE_GATE_SMALL_CHUNK_KERNEL_SOURCE,
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


def fused_qkv_z_conv_silu_chunk(
    hidden: mx.array,
    conv_state: mx.array,
    projection: mx.array,
    z_projection: mx.array,
    conv_weight: mx.array,
    *,
    simdgroups: int = 16,
) -> tuple[mx.array, mx.array]:
    """Project up to eight tokens through exact convolution and BF16 SiLU."""
    require(
        hidden.dtype == mx.bfloat16
        and hidden.ndim == 2
        and 1 <= hidden.shape[0] <= 8
        and hidden.shape[1] == 2048,
        "fused QKV chunk hidden mismatch",
    )
    require(
        conv_state.dtype == mx.bfloat16 and conv_state.shape == (8192, 4),
        "fused QKV chunk convolution state mismatch",
    )
    require(
        projection.dtype == mx.bfloat16 and projection.shape == (8192, 2048),
        "fused QKV chunk projection mismatch",
    )
    require(
        z_projection.dtype == mx.bfloat16 and z_projection.shape == (4096, 2048),
        "fused QKV chunk z projection mismatch",
    )
    require(
        conv_weight.dtype == mx.bfloat16 and conv_weight.shape == (8192, 4),
        "fused QKV chunk convolution weight mismatch",
    )
    require(simdgroups in (8, 16, 32), "invalid fused QKV chunk SIMD groups")
    tokens = hidden.shape[0]
    threads = simdgroups * 32
    return _qkv_z_conv_silu_chunk_kernel(
        inputs=[projection, z_projection, hidden, conv_state, conv_weight],
        template=[("TOKENS", tokens), ("SIMDGROUPS", simdgroups)],
        grid=(((12288 + simdgroups - 1) // simdgroups) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(8192, 4), (tokens, 8192), (tokens, 4096)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )


def fused_qkv_conv_silu_transition_step(
    hidden: mx.array,
    conv_state: mx.array,
    projection: mx.array,
    conv_weight: mx.array,
    z_projection: mx.array,
    b_projection: mx.array,
    a_projection: mx.array,
    dt_bias: mx.array,
    a_log: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Project all GDN inputs and emit exact beta/decay transition scalars."""
    require(
        hidden.dtype == mx.bfloat16 and hidden.shape == (2048,),
        "fused transition hidden mismatch",
    )
    require(
        conv_state.dtype == mx.bfloat16 and conv_state.shape == (8192, 4),
        "fused transition convolution state mismatch",
    )
    require(
        projection.dtype == mx.bfloat16 and projection.shape == (8192, 2048),
        "fused transition QKV projection mismatch",
    )
    require(
        conv_weight.dtype == mx.bfloat16 and conv_weight.shape == (8192, 4),
        "fused transition convolution weight mismatch",
    )
    require(
        z_projection.dtype == mx.bfloat16 and z_projection.shape == (4096, 2048),
        "fused transition z projection mismatch",
    )
    require(
        b_projection.dtype == mx.bfloat16 and b_projection.shape == (32, 2048),
        "fused transition b projection mismatch",
    )
    require(
        a_projection.dtype == mx.bfloat16 and a_projection.shape == (32, 2048),
        "fused transition a projection mismatch",
    )
    require(
        dt_bias.dtype == mx.bfloat16 and dt_bias.shape == (32,),
        "fused transition dt bias mismatch",
    )
    require(
        a_log.dtype == mx.bfloat16 and a_log.shape == (32,),
        "fused transition A-log mismatch",
    )
    next_state, convolved, z, beta, decay = _qkv_conv_silu_transition_kernel(
        inputs=[
            projection,
            hidden,
            conv_state,
            conv_weight,
            z_projection,
            b_projection,
            a_projection,
            dt_bias,
            a_log,
        ],
        grid=(395_264, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(8192, 4), (8192,), (4096,), (32,), (32,)],
        output_dtypes=[
            mx.bfloat16,
            mx.bfloat16,
            mx.bfloat16,
            mx.float32,
            mx.float32,
        ],
    )
    return next_state, convolved, z, beta, decay


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


def fused_recurrence_convolved_core_gate_step(
    recurrent: mx.array,
    convolved: mx.array,
    beta: mx.array,
    decay: mx.array,
    z: mx.array,
    norm: mx.array,
    *,
    simdgroups: int = 32,
) -> tuple[mx.array, mx.array]:
    """Normalize convolved Q/K inside the exact recurrence/core dispatch."""
    expected = {
        "recurrent": (recurrent, mx.float32, (32, 128, 128)),
        "convolved": (convolved, mx.bfloat16, (8192,)),
        "beta": (beta, mx.float32, (32,)),
        "decay": (decay, mx.float32, (32,)),
        "z": (z, mx.bfloat16, (32, 128)),
        "norm": (norm, mx.bfloat16, (128,)),
    }
    for name, (array, dtype, shape) in expected.items():
        require(array.dtype == dtype, f"convolved recurrence {name} dtype mismatch")
        require(array.shape == shape, f"convolved recurrence {name} shape mismatch")
    require(simdgroups in (8, 16, 32), "invalid recurrence SIMD-group count")
    threads = simdgroups * 32
    next_recurrent, gated = _recurrence_convolved_core_gate_kernel(
        inputs=[recurrent, convolved, beta, decay, z, norm],
        template=[("SIMDGROUPS", simdgroups)],
        grid=(32 * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(32, 128, 128), (32, 128)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )
    return next_recurrent, gated


def fused_beta_decay(
    b: mx.array,
    a: mx.array,
    dt_bias: mx.array,
    a_log: mx.array,
) -> tuple[mx.array, mx.array]:
    """Evaluate the production 32-head beta and decay formulas on Metal."""
    for name, array in {
        "b": b,
        "a": a,
        "dt_bias": dt_bias,
        "a_log": a_log,
    }.items():
        require(array.dtype == mx.bfloat16, f"beta/decay {name} dtype mismatch")
        require(array.shape == (32,), f"beta/decay {name} shape mismatch")
    beta, decay = _beta_decay_kernel(
        inputs=[b, a, dt_bias, a_log],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32,), (32,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return beta, decay


def fused_beta_decay_chunk(
    b: mx.array,
    a: mx.array,
    dt_bias: mx.array,
    a_log: mx.array,
) -> tuple[mx.array, mx.array]:
    """Evaluate exact beta and decay formulas for a token block."""
    require(
        b.dtype == mx.bfloat16 and b.ndim == 2 and b.shape[1] == 32,
        "beta/decay chunk b mismatch",
    )
    tokens = b.shape[0]
    require(tokens > 0, "beta/decay chunk is empty")
    require(a.dtype == mx.bfloat16 and a.shape == b.shape, "beta/decay chunk a mismatch")
    require(
        dt_bias.dtype == mx.bfloat16 and dt_bias.shape == (32,),
        "beta/decay chunk bias mismatch",
    )
    require(
        a_log.dtype == mx.bfloat16 and a_log.shape == (32,),
        "beta/decay chunk A-log mismatch",
    )
    elements = tokens * 32
    return _beta_decay_chunk_kernel(
        inputs=[b, a, dt_bias, a_log],
        template=[("TOKENS", tokens)],
        grid=(((elements + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, 32), (tokens, 32)],
        output_dtypes=[mx.float32, mx.float32],
    )


def fused_ba_beta_decay_chunk(
    hidden: mx.array,
    b_projection: mx.array,
    a_projection: mx.array,
    dt_bias: mx.array,
    a_log: mx.array,
) -> tuple[mx.array, mx.array]:
    """Project B/A and evaluate exact beta/decay for up to eight tokens."""
    require(
        hidden.dtype == mx.bfloat16
        and hidden.ndim == 2
        and 1 <= hidden.shape[0] <= 8
        and hidden.shape[1] == 2048,
        "fused B/A chunk hidden mismatch",
    )
    require(
        b_projection.dtype == mx.bfloat16 and b_projection.shape == (32, 2048),
        "fused B chunk projection mismatch",
    )
    require(
        a_projection.dtype == mx.bfloat16 and a_projection.shape == (32, 2048),
        "fused A chunk projection mismatch",
    )
    require(
        dt_bias.dtype == mx.bfloat16 and dt_bias.shape == (32,),
        "fused B/A chunk bias mismatch",
    )
    require(
        a_log.dtype == mx.bfloat16 and a_log.shape == (32,),
        "fused B/A chunk A-log mismatch",
    )
    tokens = hidden.shape[0]
    return _ba_beta_decay_chunk_kernel(
        inputs=[hidden, b_projection, a_projection, dt_bias, a_log],
        template=[("TOKENS", tokens)],
        grid=(tokens * 8 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, 32), (tokens, 32)],
        output_dtypes=[mx.float32, mx.float32],
    )


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
    fused_small_chunk: bool = True,
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
    if fused_small_chunk and tokens <= 8 and simdgroups == 32:
        return _recurrence_column_core_gate_small_chunk_kernel(
            inputs=[recurrent, key, query, value, beta, decay, z, norm],
            template=[("TOKENS", tokens), ("SIMDGROUPS", simdgroups)],
            grid=(32 * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(32, 128, 128), (tokens, 32, 128)],
            output_dtypes=[mx.float32, mx.bfloat16],
        )
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


def fused_recurrence_convolved_core_gate_column_small_chunk(
    recurrent: mx.array,
    convolved: mx.array,
    beta: mx.array,
    decay: mx.array,
    z: mx.array,
    norm: mx.array,
) -> tuple[mx.array, mx.array]:
    """Advance up to eight tokens directly from convolved BF16 Q/K/V."""
    require(
        convolved.dtype == mx.bfloat16
        and convolved.ndim == 2
        and 1 <= convolved.shape[0] <= 8
        and convolved.shape[1] == 8192,
        "convolved column recurrence chunk mismatch",
    )
    tokens = convolved.shape[0]
    expected = {
        "recurrent": (recurrent, mx.float32, (32, 128, 128)),
        "beta": (beta, mx.float32, (tokens, 32)),
        "decay": (decay, mx.float32, (tokens, 32)),
        "z": (z, mx.bfloat16, (tokens, 32, 128)),
        "norm": (norm, mx.bfloat16, (128,)),
    }
    for name, (array, dtype, shape) in expected.items():
        require(array.dtype == dtype, f"convolved column {name} dtype mismatch")
        require(array.shape == shape, f"convolved column {name} shape mismatch")
    threads = 32 * 32
    return _recurrence_convolved_column_core_gate_small_chunk_kernel(
        inputs=[recurrent, convolved, beta, decay, z, norm],
        template=[("TOKENS", tokens), ("SIMDGROUPS", 32)],
        grid=(32 * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(32, 128, 128), (tokens, 32, 128)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )


def decode_step(
    hidden: mx.array,
    state: MLXGDNState,
    weights: MLXGDNWeights,
    config: GDNConfig = PRODUCTION_CONFIG,
    *,
    fused_convolution: bool = True,
    fused_recurrence: bool = True,
    fused_core_gate_output: bool = True,
    fused_recurrence_inputs: bool = True,
    fused_beta_decay_output: bool = True,
    fused_input_transition: bool = True,
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

    production = config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16
    if (
        fused_convolution
        and fused_beta_decay_output
        and fused_input_transition
        and production
    ):
        next_conv, convolved, z, beta, decay = fused_qkv_conv_silu_transition_step(
            hidden,
            state.conv,
            weights.in_proj_qkv,
            weights.conv1d,
            weights.in_proj_z,
            weights.in_proj_b,
            weights.in_proj_a,
            weights.dt_bias,
            weights.a_log,
        )
        transition_prepared = True
    elif fused_convolution and production:
        z = _linear(weights.in_proj_z, hidden)
        b = _linear(weights.in_proj_b, hidden)
        a = _linear(weights.in_proj_a, hidden)
        next_conv, convolved = fused_qkv_conv_silu_step(
            hidden,
            state.conv,
            weights.in_proj_qkv,
            weights.conv1d,
        )
        transition_prepared = False
    else:
        z = _linear(weights.in_proj_z, hidden)
        b = _linear(weights.in_proj_b, hidden)
        a = _linear(weights.in_proj_a, hidden)
        mixed = _linear(weights.in_proj_qkv, hidden)
        next_conv = mx.concatenate([state.conv[:, 1:], mixed[:, None]], axis=1)
        convolved32 = mx.sum(
            next_conv.astype(mx.float32) * weights.conv1d.astype(mx.float32),
            axis=1,
        )
        convolved = _silu(convolved32).astype(model_dtype)
        transition_prepared = False

    z_heads = z.reshape(config.num_v_heads, config.head_v_dim)
    use_convolved_recurrence = (
        fused_recurrence
        and fused_core_gate_output
        and fused_recurrence_inputs
        and production
    )
    if not transition_prepared:
        if fused_beta_decay_output and production:
            beta, decay = fused_beta_decay(
                b,
                a,
                weights.dt_bias,
                weights.a_log,
            )
        else:
            beta = mx.sigmoid(b.astype(mx.float32))
            decay_log = -mx.exp(weights.a_log.astype(mx.float32)) * _softplus(
                a.astype(mx.float32) + weights.dt_bias.astype(mx.float32)
            )
            decay = mx.exp(decay_log)
    if use_convolved_recurrence:
        recurrent, gated = fused_recurrence_convolved_core_gate_step(
            state.recurrent,
            convolved,
            beta,
            decay,
            z_heads,
            weights.norm,
        )
    else:
        query_end = config.key_dim
        key_end = query_end + config.key_dim
        query = convolved[:query_end].reshape(
            config.num_k_heads,
            config.head_k_dim,
        )
        key = convolved[query_end:key_end].reshape(
            config.num_k_heads,
            config.head_k_dim,
        )
        value = convolved[key_end:].reshape(
            config.num_v_heads,
            config.head_v_dim,
        )
        query = _l2norm(query)
        key = _l2norm(key)
        repeats = config.num_v_heads // config.num_k_heads
        if repeats > 1:
            query = mx.repeat(query, repeats, axis=0)
            key = mx.repeat(key, repeats, axis=0)
        query = query * (config.head_k_dim**-0.5)
        value32 = value.astype(mx.float32)
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
    *,
    token_tiled_projections: bool = True,
    fused_qkv_convolution: bool = True,
    fused_convolved_recurrence: bool = True,
    fused_beta_decay_output: bool = True,
    fused_ba_transition: bool = True,
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
    tokens = hidden.shape[0]
    fused_qkv = (
        fused_qkv_convolution
        and token_tiled_projections
        and hidden.shape[0] <= 8
    )
    if token_tiled_projections:
        if fused_qkv:
            next_conv, convolved, z = fused_qkv_z_conv_silu_chunk(
                hidden,
                state.conv,
                weights.in_proj_qkv,
                weights.in_proj_z,
                weights.conv1d,
            )
        else:
            mixed = dense.token_tiled_matvec(
                weights.in_proj_qkv,
                hidden,
                token_tile=8,
                simdgroups_per_threadgroup=16,
            )
            z = dense.token_tiled_matvec(
                weights.in_proj_z,
                hidden,
                token_tile=8,
                simdgroups_per_threadgroup=16,
            )
    else:
        mixed = _linear_batch(weights.in_proj_qkv, hidden)
        z = _linear_batch(weights.in_proj_z, hidden)
    fused_ba = fused_ba_transition and tokens <= 8
    if fused_ba:
        beta, decay = fused_ba_beta_decay_chunk(
            hidden,
            weights.in_proj_b,
            weights.in_proj_a,
            weights.dt_bias,
            weights.a_log,
        )
    else:
        b = _linear_batch(weights.in_proj_b, hidden)
        a = _linear_batch(weights.in_proj_a, hidden)

    if not fused_qkv:
        next_conv, convolved32 = fused_conv_chunk(state.conv, mixed, weights.conv1d)
        convolved = _silu(convolved32).astype(model_dtype)
    if not fused_ba:
        if fused_beta_decay_output:
            beta, decay = fused_beta_decay_chunk(
                b,
                a,
                weights.dt_bias,
                weights.a_log,
            )
        else:
            beta = mx.sigmoid(b.astype(mx.float32))
            decay_log = -mx.exp(weights.a_log.astype(mx.float32))[None, :] * _softplus(
                a.astype(mx.float32) + weights.dt_bias.astype(mx.float32)[None, :]
            )
            decay = mx.exp(decay_log)
    z_heads = z.reshape(tokens, config.num_v_heads, config.head_v_dim)
    if fused_convolved_recurrence and tokens <= 8:
        recurrent, gated = fused_recurrence_convolved_core_gate_column_small_chunk(
            state.recurrent,
            convolved,
            beta,
            decay,
            z_heads,
            weights.norm,
        )
    else:
        query_end = config.key_dim
        key_end = query_end + config.key_dim
        query = convolved[:, :query_end].reshape(
            tokens,
            config.num_k_heads,
            config.head_k_dim,
        )
        key = convolved[:, query_end:key_end].reshape(
            tokens,
            config.num_k_heads,
            config.head_k_dim,
        )
        value = convolved[:, key_end:].reshape(
            tokens,
            config.num_v_heads,
            config.head_v_dim,
        )
        query = _l2norm(query)
        key = _l2norm(key)
        repeats = config.num_v_heads // config.num_k_heads
        if repeats > 1:
            query = mx.repeat(query, repeats, axis=1)
            key = mx.repeat(key, repeats, axis=1)
        query = query * (config.head_k_dim**-0.5)
        recurrent, gated = fused_recurrence_core_gate_column_chunk(
            state.recurrent,
            key,
            query,
            value.astype(mx.float32),
            beta,
            decay,
            z_heads,
            weights.norm,
        )
    output_input = gated.reshape(tokens, config.value_dim)
    output = (
        dense.token_tiled_matvec(weights.out_proj, output_input, token_tile=8)
        if token_tiled_projections
        else _linear_batch(weights.out_proj, output_input)
    )
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

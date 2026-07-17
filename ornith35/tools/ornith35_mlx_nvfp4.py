#!/usr/bin/env python3
"""MLX composition boundary for Ornith-35 packed ModelOpt NVFP4."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

from ornith35_nvfp4 import NVFP4Error, NVFP4Weight, SafetensorsFile, require


KERNEL_HEADER = r"""
constant float ornith35_e2m1_values[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

inline float ornith35_decode_e2m1(uchar nibble) {
    float value = ornith35_e2m1_values[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

inline float ornith35_decode_e4m3fn(uchar bits) {
    uint exponent = (bits >> 3) & 15u;
    uint mantissa = bits & 7u;
    float value;
    if (exponent == 0u) {
        value = float(mantissa) * 0.001953125f;
    } else if (exponent == 15u && mantissa == 7u) {
        value = NAN;
    } else {
        value = (1.0f + float(mantissa) * 0.125f)
            * exp2(float(int(exponent) - 7));
    }
    return (bits & 128u) ? -value : value;
}
"""

KERNEL_SOURCE = r"""
uint row = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (row >= ROWS) return;
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
float inverse_global = 1.0f / global_scale[0];
float sum = 0.0f;
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    float scale = ornith35_decode_e4m3fn(block_scale[row * blocks_per_row + block])
        * inverse_global;
    uint column_base = block << 4;
    uint packed_base = row * packed_columns + (column_base >> 1);
    for (uint pair = 0; pair < 8u; pair++) {
        uchar packed = packed_weight[packed_base + pair];
        uint column = column_base + (pair << 1);
        sum += ornith35_decode_e2m1(packed & 15u) * scale * input[column];
        sum += ornith35_decode_e2m1(packed >> 4) * scale * input[column + 1u];
    }
}
sum = simd_sum(sum);
if (thread_index_in_simdgroup == 0) output[row] = sum;
"""

_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_matvec_f32",
    input_names=["packed_weight", "block_scale", "global_scale", "input"],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=KERNEL_SOURCE,
)


SELECTED_KERNEL_SOURCE = r"""
uint work_item = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (work_item >= TOPK * ROWS) return;
uint slot = work_item / ROWS;
uint row = work_item - slot * ROWS;
uint expert = selected_experts[slot];
if (expert >= EXPERTS) {
    if (thread_index_in_simdgroup == 0) output[slot * ROWS + row] = NAN;
    return;
}
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
uint packed_base = (expert * ROWS + row) * packed_columns;
uint scale_base = (expert * ROWS + row) * blocks_per_row;
uint input_base = BATCHED_INPUT ? slot * COLUMNS : 0u;
float inverse_global = 1.0f / global_scale[expert];
float sum = 0.0f;
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    float scale = ornith35_decode_e4m3fn(block_scale[scale_base + block]) * inverse_global;
    uint column_base = block << 4;
    uint byte_base = packed_base + (column_base >> 1);
    for (uint pair = 0; pair < 8u; pair++) {
        uchar packed = packed_weight[byte_base + pair];
        uint column = column_base + (pair << 1);
        sum += ornith35_decode_e2m1(packed & 15u) * scale * input[input_base + column];
        sum += ornith35_decode_e2m1(packed >> 4) * scale * input[input_base + column + 1u];
    }
}
sum = simd_sum(sum);
if (thread_index_in_simdgroup == 0) output[slot * ROWS + row] = sum;
"""


_selected_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_selected_matvec_f32",
    input_names=[
        "packed_weight",
        "block_scale",
        "global_scale",
        "selected_experts",
        "input",
    ],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=SELECTED_KERNEL_SOURCE,
)


PAIRED_KERNEL_SOURCE = r"""
uint row = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (row >= ROWS) return;
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
float gate_inverse_global = 1.0f / gate_global_scale[0];
float up_inverse_global = 1.0f / up_global_scale[0];
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    uint scale_index = row * blocks_per_row + block;
    float gate_scale = ornith35_decode_e4m3fn(gate_block_scale[scale_index])
        * gate_inverse_global;
    float up_scale = ornith35_decode_e4m3fn(up_block_scale[scale_index])
        * up_inverse_global;
    uint column_base = block << 4;
    uint packed_base = row * packed_columns + (column_base >> 1);
    for (uint pair = 0; pair < 8u; pair++) {
        uchar gate_packed = gate_weight[packed_base + pair];
        uchar up_packed = up_weight[packed_base + pair];
        uint column = column_base + (pair << 1);
        float first = input[column];
        float second = input[column + 1u];
        gate_sum += ornith35_decode_e2m1(gate_packed & 15u) * gate_scale * first;
        gate_sum += ornith35_decode_e2m1(gate_packed >> 4) * gate_scale * second;
        up_sum += ornith35_decode_e2m1(up_packed & 15u) * up_scale * first;
        up_sum += ornith35_decode_e2m1(up_packed >> 4) * up_scale * second;
    }
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (thread_index_in_simdgroup == 0) {
    output[row] = gate_sum;
    output[ROWS + row] = up_sum;
}
"""


_paired_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_paired_matvec_f32",
    input_names=[
        "gate_weight",
        "gate_block_scale",
        "gate_global_scale",
        "up_weight",
        "up_block_scale",
        "up_global_scale",
        "input",
    ],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=PAIRED_KERNEL_SOURCE,
)


SELECTED_PAIRED_KERNEL_SOURCE = r"""
uint work_item = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (work_item >= TOPK * ROWS) return;
uint slot = work_item / ROWS;
uint row = work_item - slot * ROWS;
uint expert = selected_experts[slot];
if (expert >= EXPERTS) {
    if (thread_index_in_simdgroup == 0) {
        output[(slot * 2u) * ROWS + row] = NAN;
        output[(slot * 2u + 1u) * ROWS + row] = NAN;
    }
    return;
}
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
uint packed_base = (expert * ROWS + row) * packed_columns;
uint scale_base = (expert * ROWS + row) * blocks_per_row;
float gate_inverse_global = 1.0f / gate_global_scale[expert];
float up_inverse_global = 1.0f / up_global_scale[expert];
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    float gate_scale = ornith35_decode_e4m3fn(gate_block_scale[scale_base + block])
        * gate_inverse_global;
    float up_scale = ornith35_decode_e4m3fn(up_block_scale[scale_base + block])
        * up_inverse_global;
    uint column_base = block << 4;
    uint byte_base = packed_base + (column_base >> 1);
    for (uint pair = 0; pair < 8u; pair++) {
        uchar gate_packed = gate_weight[byte_base + pair];
        uchar up_packed = up_weight[byte_base + pair];
        uint column = column_base + (pair << 1);
        float first = input[column];
        float second = input[column + 1u];
        gate_sum += ornith35_decode_e2m1(gate_packed & 15u) * gate_scale * first;
        gate_sum += ornith35_decode_e2m1(gate_packed >> 4) * gate_scale * second;
        up_sum += ornith35_decode_e2m1(up_packed & 15u) * up_scale * first;
        up_sum += ornith35_decode_e2m1(up_packed >> 4) * up_scale * second;
    }
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (thread_index_in_simdgroup == 0) {
    output[(slot * 2u) * ROWS + row] = gate_sum;
    output[(slot * 2u + 1u) * ROWS + row] = up_sum;
}
"""


_selected_paired_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_selected_paired_matvec_f32",
    input_names=[
        "gate_weight",
        "gate_block_scale",
        "gate_global_scale",
        "up_weight",
        "up_block_scale",
        "up_global_scale",
        "selected_experts",
        "input",
    ],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=SELECTED_PAIRED_KERNEL_SOURCE,
)


SELECTED_WEIGHTED_KERNEL_SOURCE = r"""
uint row = threadgroup_position_in_grid.x;
if (row >= ROWS) return;
uint slot = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
threadgroup float partial[8];
if (slot < TOPK) {
    uint expert = selected_experts[slot];
    float sum = 0.0f;
    if (expert < EXPERTS) {
        uint packed_columns = COLUMNS >> 1;
        uint blocks_per_row = COLUMNS >> 4;
        uint packed_base = (expert * ROWS + row) * packed_columns;
        uint scale_base = (expert * ROWS + row) * blocks_per_row;
        uint input_base = slot * COLUMNS;
        float inverse_global = 1.0f / global_scale[expert];
        for (uint block = lane; block < blocks_per_row; block += 32u) {
            float scale = ornith35_decode_e4m3fn(block_scale[scale_base + block])
                * inverse_global;
            uint column_base = block << 4;
            uint byte_base = packed_base + (column_base >> 1);
            for (uint pair = 0; pair < 8u; pair++) {
                uchar packed = packed_weight[byte_base + pair];
                uint column = column_base + (pair << 1);
                sum += ornith35_decode_e2m1(packed & 15u)
                    * scale * input[input_base + column];
                sum += ornith35_decode_e2m1(packed >> 4)
                    * scale * input[input_base + column + 1u];
            }
        }
        sum = simd_sum(sum);
    } else {
        sum = NAN;
    }
    if (lane == 0u) {
        float rounded = ORNITH35_MODEL_ROUND(sum);
        partial[slot] = rounded * float(routing_weights[slot]);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (slot == 0u && lane == 0u) {
    float total = partial[0];
    for (uint index = 1u; index < TOPK; index++) total += partial[index];
    output[row] = ORNITH35_MODEL_OUTPUT(total);
}
"""


_selected_weighted_f32_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_selected_weighted_matvec_f32",
    input_names=[
        "packed_weight",
        "block_scale",
        "global_scale",
        "selected_experts",
        "input",
        "routing_weights",
    ],
    output_names=["output"],
    header=KERNEL_HEADER
    + r"""
#define ORNITH35_MODEL_ROUND(value) (value)
#define ORNITH35_MODEL_OUTPUT(value) (value)
""",
    source=SELECTED_WEIGHTED_KERNEL_SOURCE,
)


_selected_weighted_bf16_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_selected_weighted_matvec_bf16",
    input_names=[
        "packed_weight",
        "block_scale",
        "global_scale",
        "selected_experts",
        "input",
        "routing_weights",
    ],
    output_names=["output"],
    header=KERNEL_HEADER
    + r"""
#define ORNITH35_MODEL_ROUND(value) float(bfloat16_t(value))
#define ORNITH35_MODEL_OUTPUT(value) bfloat16_t(value)
""",
    source=SELECTED_WEIGHTED_KERNEL_SOURCE,
)


SELECTED_SHARED_WEIGHTED_ROWS4_KERNEL_SOURCE = r"""
uint row_base = threadgroup_position_in_grid.x * 4u;
uint slot = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
threadgroup float routed_partial[32];
threadgroup float shared_partial[4];
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
for (uint local_row = 0u; local_row < 4u; ++local_row) {
    uint row = row_base + local_row;
    if (slot < TOPK) {
        uint expert = selected_experts[slot];
        float sum = 0.0f;
        if (expert < EXPERTS && row < ROWS) {
            uint packed_base = (expert * ROWS + row) * packed_columns;
            uint scale_base = (expert * ROWS + row) * blocks_per_row;
            uint input_base = slot * COLUMNS;
            float inverse_global = 1.0f / global_scale[expert];
            for (uint block = lane; block < blocks_per_row; block += 32u) {
                float scale = ornith35_decode_e4m3fn(block_scale[scale_base + block])
                    * inverse_global;
                uint column_base = block << 4;
                uint byte_base = packed_base + (column_base >> 1);
                for (uint pair = 0u; pair < 8u; ++pair) {
                    uchar packed = packed_weight[byte_base + pair];
                    uint column = column_base + (pair << 1);
                    sum += ornith35_decode_e2m1(packed & 15u)
                        * scale * input[input_base + column];
                    sum += ornith35_decode_e2m1(packed >> 4)
                        * scale * input[input_base + column + 1u];
                }
            }
            sum = simd_sum(sum);
        } else {
            sum = NAN;
        }
        if (lane == 0u) {
            routed_partial[local_row * TOPK + slot] =
                float(bfloat16_t(sum)) * float(routing_weights[slot]);
        }
    } else if (slot == TOPK) {
        float sum = 0.0f;
        if (row < ROWS) {
            uint packed_base = row * packed_columns;
            uint scale_base = row * blocks_per_row;
            float inverse_global = 1.0f / shared_global_scale[0];
            for (uint block = lane; block < blocks_per_row; block += 32u) {
                float scale = ornith35_decode_e4m3fn(
                    shared_block_scale[scale_base + block]
                ) * inverse_global;
                uint column_base = block << 4;
                uint byte_base = packed_base + (column_base >> 1);
                for (uint pair = 0u; pair < 8u; ++pair) {
                    uchar packed = shared_weight[byte_base + pair];
                    uint column = column_base + (pair << 1);
                    sum += ornith35_decode_e2m1(packed & 15u)
                        * scale * shared_input[column];
                    sum += ornith35_decode_e2m1(packed >> 4)
                        * scale * shared_input[column + 1u];
                }
            }
            sum = simd_sum(sum);
        }
        if (lane == 0u) shared_partial[local_row] = float(bfloat16_t(sum));
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (slot == 0u && lane == 0u) {
    for (uint local_row = 0u; local_row < 4u; ++local_row) {
        uint row = row_base + local_row;
        if (row < ROWS) {
            float total = routed_partial[local_row * TOPK];
            for (uint index = 1u; index < TOPK; ++index) {
                total += routed_partial[local_row * TOPK + index];
            }
            bfloat16_t routed = bfloat16_t(total);
            bfloat16_t product = bfloat16_t(
                shared_partial[local_row] * float(shared_multiplier)
            );
            output[row] = bfloat16_t(float(routed) + float(product));
        }
    }
}
"""


_selected_shared_weighted_rows4_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_selected_shared_weighted_rows4_bf16",
    input_names=[
        "packed_weight",
        "block_scale",
        "global_scale",
        "shared_weight",
        "shared_block_scale",
        "shared_global_scale",
        "selected_experts",
        "input",
        "shared_input",
        "routing_weights",
        "shared_multiplier",
    ],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=SELECTED_SHARED_WEIGHTED_ROWS4_KERNEL_SOURCE,
)


BATCHED_KERNEL_SOURCE = r"""
uint work_item = threadgroup_position_in_grid.x * SIMDGROUPS_PER_THREADGROUP
    + simdgroup_index_in_threadgroup;
uint row_groups = ROWS / ROWS_PER_SIMDGROUP;
if (work_item >= TOKENS * row_groups) return;
uint token = work_item / row_groups;
uint row_group = work_item - token * row_groups;
uint row_base = row_group * ROWS_PER_SIMDGROUP;
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
float inverse_global = 1.0f / global_scale[0];
float sums[ROWS_PER_SIMDGROUP];
for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
    sums[local_row] = 0.0f;
}
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    uint column_base = block << 4;
    uint input_base = token * COLUMNS;
    float scales[ROWS_PER_SIMDGROUP];
    uint packed_bases[ROWS_PER_SIMDGROUP];
    for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
        uint row = row_base + local_row;
        scales[local_row] = ornith35_decode_e4m3fn(
            block_scale[row * blocks_per_row + block]
        ) * inverse_global;
        packed_bases[local_row] = row * packed_columns + (column_base >> 1);
    }
    for (uint pair = 0; pair < 8u; pair++) {
        uint column = column_base + (pair << 1);
        float first = input[input_base + column];
        float second = input[input_base + column + 1u];
        for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
            uchar packed = packed_weight[packed_bases[local_row] + pair];
            sums[local_row] += ornith35_decode_e2m1(packed & 15u)
                * scales[local_row] * first;
            sums[local_row] += ornith35_decode_e2m1(packed >> 4)
                * scales[local_row] * second;
        }
    }
}
for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
    sums[local_row] = simd_sum(sums[local_row]);
    if (thread_index_in_simdgroup == 0) {
        output[token * ROWS + row_base + local_row] = sums[local_row];
    }
}
"""


_batched_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_batched_matvec_f32",
    input_names=["packed_weight", "block_scale", "global_scale", "input"],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=BATCHED_KERNEL_SOURCE,
)


BATCHED_PAIRED_KERNEL_SOURCE = r"""
uint work_item = threadgroup_position_in_grid.x * SIMDGROUPS_PER_THREADGROUP
    + simdgroup_index_in_threadgroup;
uint row_groups = ROWS / ROWS_PER_SIMDGROUP;
if (work_item >= TOKENS * row_groups) return;
uint token = work_item / row_groups;
uint row_group = work_item - token * row_groups;
uint row_base = row_group * ROWS_PER_SIMDGROUP;
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
float gate_inverse_global = 1.0f / gate_global_scale[0];
float up_inverse_global = 1.0f / up_global_scale[0];
float gate_sums[ROWS_PER_SIMDGROUP];
float up_sums[ROWS_PER_SIMDGROUP];
for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
    gate_sums[local_row] = 0.0f;
    up_sums[local_row] = 0.0f;
}
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    uint column_base = block << 4;
    uint input_base = token * COLUMNS;
    float gate_scales[ROWS_PER_SIMDGROUP];
    float up_scales[ROWS_PER_SIMDGROUP];
    uint packed_bases[ROWS_PER_SIMDGROUP];
    for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
        uint row = row_base + local_row;
        uint scale_index = row * blocks_per_row + block;
        gate_scales[local_row] = ornith35_decode_e4m3fn(
            gate_block_scale[scale_index]
        ) * gate_inverse_global;
        up_scales[local_row] = ornith35_decode_e4m3fn(
            up_block_scale[scale_index]
        ) * up_inverse_global;
        packed_bases[local_row] = row * packed_columns + (column_base >> 1);
    }
    for (uint pair = 0; pair < 8u; pair++) {
        uint column = column_base + (pair << 1);
        float first = input[input_base + column];
        float second = input[input_base + column + 1u];
        for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
            uchar gate_packed = gate_weight[packed_bases[local_row] + pair];
            uchar up_packed = up_weight[packed_bases[local_row] + pair];
            gate_sums[local_row] += ornith35_decode_e2m1(gate_packed & 15u)
                * gate_scales[local_row] * first;
            gate_sums[local_row] += ornith35_decode_e2m1(gate_packed >> 4)
                * gate_scales[local_row] * second;
            up_sums[local_row] += ornith35_decode_e2m1(up_packed & 15u)
                * up_scales[local_row] * first;
            up_sums[local_row] += ornith35_decode_e2m1(up_packed >> 4)
                * up_scales[local_row] * second;
        }
    }
}
for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
    gate_sums[local_row] = simd_sum(gate_sums[local_row]);
    up_sums[local_row] = simd_sum(up_sums[local_row]);
    if (thread_index_in_simdgroup == 0) {
        uint row = row_base + local_row;
        uint output_base = token * 2u * ROWS;
        output[output_base + row] = gate_sums[local_row];
        output[output_base + ROWS + row] = up_sums[local_row];
    }
}
"""


_batched_paired_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_batched_paired_matvec_f32",
    input_names=[
        "gate_weight",
        "gate_block_scale",
        "gate_global_scale",
        "up_weight",
        "up_block_scale",
        "up_global_scale",
        "input",
    ],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=BATCHED_PAIRED_KERNEL_SOURCE,
)


BATCHED_SELECTED_PAIRED_KERNEL_SOURCE = r"""
uint work_item = threadgroup_position_in_grid.x * SIMDGROUPS_PER_THREADGROUP
    + simdgroup_index_in_threadgroup;
uint row_groups = ROWS / ROWS_PER_SIMDGROUP;
if (work_item >= TOKENS * TOPK * row_groups) return;
uint token_slot = work_item / row_groups;
uint row_group = work_item - token_slot * row_groups;
uint row_base = row_group * ROWS_PER_SIMDGROUP;
uint token = token_slot / TOPK;
uint slot = token_slot - token * TOPK;
uint expert = selected_experts[token_slot];
if (expert >= EXPERTS) {
    if (thread_index_in_simdgroup == 0) {
        uint output_base = token_slot * 2u * ROWS;
        for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
            uint row = row_base + local_row;
            output[output_base + row] = NAN;
            output[output_base + ROWS + row] = NAN;
        }
    }
    return;
}
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
float gate_inverse_global = 1.0f / gate_global_scale[expert];
float up_inverse_global = 1.0f / up_global_scale[expert];
float gate_sums[ROWS_PER_SIMDGROUP];
float up_sums[ROWS_PER_SIMDGROUP];
for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
    gate_sums[local_row] = 0.0f;
    up_sums[local_row] = 0.0f;
}
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    uint column_base = block << 4;
    uint input_base = token * COLUMNS;
    float gate_scales[ROWS_PER_SIMDGROUP];
    float up_scales[ROWS_PER_SIMDGROUP];
    uint byte_bases[ROWS_PER_SIMDGROUP];
    for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
        uint row = row_base + local_row;
        uint expert_row = expert * ROWS + row;
        uint scale_base = expert_row * blocks_per_row;
        gate_scales[local_row] = ornith35_decode_e4m3fn(
            gate_block_scale[scale_base + block]
        ) * gate_inverse_global;
        up_scales[local_row] = ornith35_decode_e4m3fn(
            up_block_scale[scale_base + block]
        ) * up_inverse_global;
        byte_bases[local_row] = expert_row * packed_columns
            + (column_base >> 1);
    }
    for (uint pair = 0; pair < 8u; pair++) {
        uint column = column_base + (pair << 1);
        float first = input[input_base + column];
        float second = input[input_base + column + 1u];
        for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
            uchar gate_packed = gate_weight[byte_bases[local_row] + pair];
            uchar up_packed = up_weight[byte_bases[local_row] + pair];
            gate_sums[local_row] += ornith35_decode_e2m1(gate_packed & 15u)
                * gate_scales[local_row] * first;
            gate_sums[local_row] += ornith35_decode_e2m1(gate_packed >> 4)
                * gate_scales[local_row] * second;
            up_sums[local_row] += ornith35_decode_e2m1(up_packed & 15u)
                * up_scales[local_row] * first;
            up_sums[local_row] += ornith35_decode_e2m1(up_packed >> 4)
                * up_scales[local_row] * second;
        }
    }
}
for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
    gate_sums[local_row] = simd_sum(gate_sums[local_row]);
    up_sums[local_row] = simd_sum(up_sums[local_row]);
    if (thread_index_in_simdgroup == 0) {
        uint row = row_base + local_row;
        uint output_base = token_slot * 2u * ROWS;
        output[output_base + row] = gate_sums[local_row];
        output[output_base + ROWS + row] = up_sums[local_row];
    }
}
"""


_batched_selected_paired_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_batched_selected_paired_matvec_f32",
    input_names=[
        "gate_weight",
        "gate_block_scale",
        "gate_global_scale",
        "up_weight",
        "up_block_scale",
        "up_global_scale",
        "selected_experts",
        "input",
    ],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=BATCHED_SELECTED_PAIRED_KERNEL_SOURCE,
)


BATCHED_SELECTED_WEIGHTED_KERNEL_SOURCE = r"""
uint rows_per_threadgroup = ROW_GROUPS_PER_THREADGROUP * ROWS_PER_SIMDGROUP;
uint row_groups = ROWS / rows_per_threadgroup;
uint token = threadgroup_position_in_grid.x / row_groups;
uint row_group = threadgroup_position_in_grid.x - token * row_groups;
uint local_row_group = simdgroup_index_in_threadgroup / TOPK;
uint slot = simdgroup_index_in_threadgroup - local_row_group * TOPK;
uint row_base = row_group * rows_per_threadgroup
    + local_row_group * ROWS_PER_SIMDGROUP;
uint lane = thread_index_in_simdgroup;
threadgroup float partial[
    ROW_GROUPS_PER_THREADGROUP * ROWS_PER_SIMDGROUP * 8
];
if (slot < TOPK) {
    uint token_slot = token * TOPK + slot;
    uint expert = selected_experts[token_slot];
    float sums[ROWS_PER_SIMDGROUP];
    for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
        sums[local_row] = 0.0f;
    }
    if (expert < EXPERTS) {
        uint packed_columns = COLUMNS >> 1;
        uint blocks_per_row = COLUMNS >> 4;
        uint input_base = token_slot * COLUMNS;
        float inverse_global = 1.0f / global_scale[expert];
        for (uint block = lane; block < blocks_per_row; block += 32u) {
            uint column_base = block << 4;
            float scales[ROWS_PER_SIMDGROUP];
            uint byte_bases[ROWS_PER_SIMDGROUP];
            for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
                uint expert_row = expert * ROWS + row_base + local_row;
                scales[local_row] = ornith35_decode_e4m3fn(
                    block_scale[expert_row * blocks_per_row + block]
                ) * inverse_global;
                byte_bases[local_row] = expert_row * packed_columns
                    + (column_base >> 1);
            }
            for (uint pair = 0; pair < 8u; pair++) {
                uint column = column_base + (pair << 1);
                float first = input[input_base + column];
                float second = input[input_base + column + 1u];
                for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
                    uchar packed = packed_weight[byte_bases[local_row] + pair];
                    sums[local_row] += ornith35_decode_e2m1(packed & 15u)
                        * scales[local_row] * first;
                    sums[local_row] += ornith35_decode_e2m1(packed >> 4)
                        * scales[local_row] * second;
                }
            }
        }
    } else {
        for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
            sums[local_row] = NAN;
        }
    }
    uint partial_row_base = local_row_group * ROWS_PER_SIMDGROUP;
    for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
        sums[local_row] = simd_sum(sums[local_row]);
        if (lane == 0u) {
            float rounded = ORNITH35_MODEL_ROUND(sums[local_row]);
            partial[(partial_row_base + local_row) * 8u + slot] =
                rounded * float(routing_weights[token_slot]);
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (slot == 0u && lane == 0u) {
    uint partial_row_base = local_row_group * ROWS_PER_SIMDGROUP;
    for (uint local_row = 0u; local_row < ROWS_PER_SIMDGROUP; local_row++) {
        uint partial_base = (partial_row_base + local_row) * 8u;
        float total = partial[partial_base];
        for (uint index = 1u; index < TOPK; index++) {
            total += partial[partial_base + index];
        }
        output[token * ROWS + row_base + local_row] =
            ORNITH35_MODEL_OUTPUT(total);
    }
}
"""


_batched_selected_weighted_f32_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_batched_selected_weighted_matvec_f32",
    input_names=[
        "packed_weight",
        "block_scale",
        "global_scale",
        "selected_experts",
        "input",
        "routing_weights",
    ],
    output_names=["output"],
    header=KERNEL_HEADER
    + r"""
#define ORNITH35_MODEL_ROUND(value) (value)
#define ORNITH35_MODEL_OUTPUT(value) (value)
""",
    source=BATCHED_SELECTED_WEIGHTED_KERNEL_SOURCE,
)


_batched_selected_weighted_bf16_kernel = mx.fast.metal_kernel(
    name="ornith35_nvfp4_batched_selected_weighted_matvec_bf16",
    input_names=[
        "packed_weight",
        "block_scale",
        "global_scale",
        "selected_experts",
        "input",
        "routing_weights",
    ],
    output_names=["output"],
    header=KERNEL_HEADER
    + r"""
#define ORNITH35_MODEL_ROUND(value) float(bfloat16_t(value))
#define ORNITH35_MODEL_OUTPUT(value) bfloat16_t(value)
""",
    source=BATCHED_SELECTED_WEIGHTED_KERNEL_SOURCE,
)


def nvfp4_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    vector: mx.array,
) -> mx.array:
    require(
        packed_weight.dtype == mx.uint8 and packed_weight.ndim == 2,
        "invalid MLX packed weight",
    )
    require(
        block_scale.dtype == mx.uint8 and block_scale.ndim == 2,
        "invalid MLX block scale",
    )
    require(
        global_scale.dtype == mx.float32 and global_scale.size == 1,
        "invalid MLX global scale",
    )
    require(vector.dtype == mx.float32 and vector.ndim == 1, "invalid MLX input vector")
    rows, packed_columns = packed_weight.shape
    columns = packed_columns * 2
    require(columns % 16 == 0 and vector.size == columns, "MLX NVFP4 input shape mismatch")
    require(
        block_scale.shape == (rows, columns // 16),
        "MLX NVFP4 scale shape mismatch",
    )
    outputs = _kernel(
        inputs=[packed_weight, block_scale, global_scale, vector],
        template=[("ROWS", rows), ("COLUMNS", columns)],
        grid=(((rows + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def nvfp4_paired_matvec(
    gate_weight: mx.array,
    gate_scale: mx.array,
    gate_global_scale: mx.array,
    up_weight: mx.array,
    up_scale: mx.array,
    up_global_scale: mx.array,
    vector: mx.array,
) -> mx.array:
    """Evaluate equal-shaped gate/up projections in one Metal dispatch."""
    require(gate_weight.dtype == mx.uint8 and gate_weight.ndim == 2, "invalid gate weight")
    require(up_weight.dtype == mx.uint8 and up_weight.shape == gate_weight.shape, "invalid up weight")
    require(gate_scale.dtype == mx.uint8 and gate_scale.ndim == 2, "invalid gate scale")
    require(up_scale.dtype == mx.uint8 and up_scale.shape == gate_scale.shape, "invalid up scale")
    require(
        gate_global_scale.dtype == mx.float32 and gate_global_scale.shape == (1,),
        "invalid gate global scale",
    )
    require(
        up_global_scale.dtype == mx.float32 and up_global_scale.shape == (1,),
        "invalid up global scale",
    )
    require(vector.dtype == mx.float32 and vector.ndim == 1, "invalid paired input")
    rows, packed_columns = gate_weight.shape
    columns = packed_columns * 2
    require(columns % 16 == 0 and vector.size == columns, "paired input shape mismatch")
    require(gate_scale.shape == (rows, columns // 16), "paired scale shape mismatch")
    return _paired_kernel(
        inputs=[
            gate_weight,
            gate_scale,
            gate_global_scale,
            up_weight,
            up_scale,
            up_global_scale,
            vector,
        ],
        template=[("ROWS", rows), ("COLUMNS", columns)],
        grid=(((rows + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(2, rows)],
        output_dtypes=[mx.float32],
    )[0]


def nvfp4_selected_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    selected_experts: mx.array,
    vectors: mx.array,
    *,
    batched_input: bool,
) -> mx.array:
    """Evaluate selected stacked weights without reading expert IDs on CPU."""
    require(
        packed_weight.dtype == mx.uint8 and packed_weight.ndim == 3,
        "invalid stacked MLX packed weight",
    )
    require(
        block_scale.dtype == mx.uint8 and block_scale.ndim == 3,
        "invalid stacked MLX block scale",
    )
    require(
        global_scale.dtype == mx.float32 and global_scale.ndim == 1,
        "invalid stacked MLX global scale",
    )
    require(
        selected_experts.dtype == mx.uint32 and selected_experts.ndim == 1,
        "selected experts must be a uint32 vector",
    )
    require(vectors.dtype == mx.float32, "selected NVFP4 inputs must be FP32")
    experts, rows, packed_columns = packed_weight.shape
    columns = packed_columns * 2
    top_k = selected_experts.size
    require(experts > 0 and rows > 0 and top_k > 0, "selected NVFP4 shape is empty")
    require(global_scale.shape == (experts,), "stacked global-scale shape mismatch")
    require(columns % 16 == 0, "stacked NVFP4 input is not block aligned")
    require(
        block_scale.shape == (experts, rows, columns // 16),
        "stacked NVFP4 scale shape mismatch",
    )
    expected_input = (top_k, columns) if batched_input else (columns,)
    require(vectors.shape == expected_input, "selected NVFP4 input shape mismatch")
    outputs = _selected_kernel(
        inputs=[packed_weight, block_scale, global_scale, selected_experts, vectors],
        template=[
            ("EXPERTS", experts),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("TOPK", top_k),
            ("BATCHED_INPUT", int(batched_input)),
        ],
        grid=((((top_k * rows) + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(top_k, rows)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def nvfp4_selected_paired_matvec(
    gate_weight: mx.array,
    gate_scale: mx.array,
    gate_global_scale: mx.array,
    up_weight: mx.array,
    up_scale: mx.array,
    up_global_scale: mx.array,
    selected_experts: mx.array,
    vector: mx.array,
) -> mx.array:
    """Evaluate selected gate/up stacks together without expert-ID readback."""
    require(gate_weight.dtype == mx.uint8 and gate_weight.ndim == 3, "invalid gate stack")
    require(up_weight.dtype == mx.uint8 and up_weight.shape == gate_weight.shape, "invalid up stack")
    require(gate_scale.dtype == mx.uint8 and gate_scale.ndim == 3, "invalid gate-scale stack")
    require(up_scale.dtype == mx.uint8 and up_scale.shape == gate_scale.shape, "invalid up-scale stack")
    require(gate_global_scale.dtype == mx.float32, "invalid gate-global stack")
    require(up_global_scale.dtype == mx.float32, "invalid up-global stack")
    require(
        selected_experts.dtype == mx.uint32 and selected_experts.ndim == 1,
        "selected experts must be a uint32 vector",
    )
    require(vector.dtype == mx.float32 and vector.ndim == 1, "invalid selected paired input")
    experts, rows, packed_columns = gate_weight.shape
    columns = packed_columns * 2
    top_k = selected_experts.size
    require(experts > 0 and rows > 0 and top_k > 0, "selected paired shape is empty")
    require(gate_global_scale.shape == (experts,), "gate-global stack shape mismatch")
    require(up_global_scale.shape == (experts,), "up-global stack shape mismatch")
    require(columns % 16 == 0 and vector.size == columns, "selected paired input mismatch")
    require(
        gate_scale.shape == (experts, rows, columns // 16),
        "selected paired scale mismatch",
    )
    return _selected_paired_kernel(
        inputs=[
            gate_weight,
            gate_scale,
            gate_global_scale,
            up_weight,
            up_scale,
            up_global_scale,
            selected_experts,
            vector,
        ],
        template=[
            ("EXPERTS", experts),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("TOPK", top_k),
        ],
        grid=((((top_k * rows) + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(top_k, 2, rows)],
        output_dtypes=[mx.float32],
    )[0]


def nvfp4_selected_weighted_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    selected_experts: mx.array,
    vectors: mx.array,
    routing_weights: mx.array,
) -> mx.array:
    """Evaluate selected projections and their ordered routing sum together."""
    require(
        packed_weight.dtype == mx.uint8 and packed_weight.ndim == 3,
        "invalid selected weighted stack",
    )
    require(
        block_scale.dtype == mx.uint8 and block_scale.ndim == 3,
        "invalid selected weighted scale",
    )
    require(
        global_scale.dtype == mx.float32 and global_scale.ndim == 1,
        "invalid selected weighted global scale",
    )
    require(
        selected_experts.dtype == mx.uint32 and selected_experts.ndim == 1,
        "selected experts must be a uint32 vector",
    )
    require(vectors.dtype == mx.float32 and vectors.ndim == 2, "invalid weighted inputs")
    require(
        routing_weights.dtype in (mx.bfloat16, mx.float32) and routing_weights.ndim == 1,
        "invalid routing weights",
    )
    experts, rows, packed_columns = packed_weight.shape
    columns = packed_columns * 2
    top_k = selected_experts.size
    require(experts > 0 and rows > 0 and 0 < top_k <= 8, "invalid selected weighted shape")
    require(global_scale.shape == (experts,), "weighted global-scale shape mismatch")
    require(columns % 16 == 0, "selected weighted input is not block aligned")
    require(
        block_scale.shape == (experts, rows, columns // 16),
        "selected weighted scale shape mismatch",
    )
    require(vectors.shape == (top_k, columns), "selected weighted input shape mismatch")
    require(routing_weights.shape == (top_k,), "routing weight shape mismatch")
    kernel = (
        _selected_weighted_bf16_kernel
        if routing_weights.dtype == mx.bfloat16
        else _selected_weighted_f32_kernel
    )
    return kernel(
        inputs=[
            packed_weight,
            block_scale,
            global_scale,
            selected_experts,
            vectors,
            routing_weights,
        ],
        template=[
            ("EXPERTS", experts),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("TOPK", top_k),
        ],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[routing_weights.dtype],
    )[0]


def nvfp4_batched_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    vectors: mx.array,
    *,
    simdgroups_per_threadgroup: int = 32,
    rows_per_simdgroup: int | None = None,
) -> mx.array:
    """Evaluate one packed projection for a nonempty token matrix."""
    require(packed_weight.dtype == mx.uint8 and packed_weight.ndim == 2, "invalid batched weight")
    require(block_scale.dtype == mx.uint8 and block_scale.ndim == 2, "invalid batched scale")
    require(
        global_scale.dtype == mx.float32 and global_scale.shape == (1,),
        "invalid batched global scale",
    )
    require(vectors.dtype == mx.float32 and vectors.ndim == 2, "invalid batched inputs")
    rows, packed_columns = packed_weight.shape
    tokens, columns = vectors.shape
    require(tokens > 0 and columns == packed_columns * 2, "batched input shape mismatch")
    require(columns % 16 == 0, "batched input is not block aligned")
    require(block_scale.shape == (rows, columns // 16), "batched scale shape mismatch")
    require(
        simdgroups_per_threadgroup in (8, 16, 32),
        "invalid batched SIMD-group count",
    )
    if rows_per_simdgroup is None:
        rows_per_simdgroup = 4 if rows % 4 == 0 else 2 if rows % 2 == 0 else 1
    require(rows_per_simdgroup in (1, 2, 4), "invalid batched SIMD-group rows")
    require(rows % rows_per_simdgroup == 0, "batched rows are not SIMD-group aligned")
    work_items = tokens * (rows // rows_per_simdgroup)
    threads = simdgroups_per_threadgroup * 32
    return _batched_kernel(
        inputs=[packed_weight, block_scale, global_scale, vectors],
        template=[
            ("TOKENS", tokens),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("SIMDGROUPS_PER_THREADGROUP", simdgroups_per_threadgroup),
            ("ROWS_PER_SIMDGROUP", rows_per_simdgroup),
        ],
        grid=(
            ((work_items + simdgroups_per_threadgroup - 1)
             // simdgroups_per_threadgroup) * threads,
            1,
            1,
        ),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tokens, rows)],
        output_dtypes=[mx.float32],
    )[0]


def nvfp4_batched_paired_matvec(
    gate_weight: mx.array,
    gate_scale: mx.array,
    gate_global_scale: mx.array,
    up_weight: mx.array,
    up_scale: mx.array,
    up_global_scale: mx.array,
    vectors: mx.array,
    *,
    simdgroups_per_threadgroup: int = 32,
    rows_per_simdgroup: int | None = None,
) -> mx.array:
    """Evaluate equal-shaped shared gate/up projections for many tokens."""
    require(gate_weight.dtype == mx.uint8 and gate_weight.ndim == 2, "invalid batched gate")
    require(up_weight.dtype == mx.uint8 and up_weight.shape == gate_weight.shape, "invalid batched up")
    require(gate_scale.dtype == mx.uint8 and gate_scale.ndim == 2, "invalid batched gate scale")
    require(up_scale.dtype == mx.uint8 and up_scale.shape == gate_scale.shape, "invalid batched up scale")
    require(gate_global_scale.dtype == mx.float32 and gate_global_scale.shape == (1,), "invalid gate global")
    require(up_global_scale.dtype == mx.float32 and up_global_scale.shape == (1,), "invalid up global")
    require(vectors.dtype == mx.float32 and vectors.ndim == 2, "invalid batched paired inputs")
    rows, packed_columns = gate_weight.shape
    tokens, columns = vectors.shape
    require(tokens > 0 and columns == packed_columns * 2, "batched paired input mismatch")
    require(columns % 16 == 0, "batched paired input is not block aligned")
    require(gate_scale.shape == (rows, columns // 16), "batched paired scale mismatch")
    if rows_per_simdgroup is None:
        rows_per_simdgroup = 2 if rows % 2 == 0 else 1
    require(
        simdgroups_per_threadgroup in (8, 16, 32),
        "invalid batched paired SIMD-group count",
    )
    require(
        rows_per_simdgroup in (1, 2, 4),
        "invalid batched paired SIMD-group rows",
    )
    require(
        rows % rows_per_simdgroup == 0,
        "batched paired rows are not SIMD-group aligned",
    )
    work_items = tokens * (rows // rows_per_simdgroup)
    threads = simdgroups_per_threadgroup * 32
    return _batched_paired_kernel(
        inputs=[
            gate_weight,
            gate_scale,
            gate_global_scale,
            up_weight,
            up_scale,
            up_global_scale,
            vectors,
        ],
        template=[
            ("TOKENS", tokens),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("SIMDGROUPS_PER_THREADGROUP", simdgroups_per_threadgroup),
            ("ROWS_PER_SIMDGROUP", rows_per_simdgroup),
        ],
        grid=(
            ((work_items + simdgroups_per_threadgroup - 1)
             // simdgroups_per_threadgroup) * threads,
            1,
            1,
        ),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tokens, 2, rows)],
        output_dtypes=[mx.float32],
    )[0]


def nvfp4_batched_selected_paired_matvec(
    gate_weight: mx.array,
    gate_scale: mx.array,
    gate_global_scale: mx.array,
    up_weight: mx.array,
    up_scale: mx.array,
    up_global_scale: mx.array,
    selected_experts: mx.array,
    vectors: mx.array,
    *,
    simdgroups_per_threadgroup: int = 32,
    rows_per_simdgroup: int | None = None,
) -> mx.array:
    """Evaluate each token's selected gate/up experts without CPU routing."""
    require(gate_weight.dtype == mx.uint8 and gate_weight.ndim == 3, "invalid batched gate stack")
    require(up_weight.dtype == mx.uint8 and up_weight.shape == gate_weight.shape, "invalid batched up stack")
    require(gate_scale.dtype == mx.uint8 and gate_scale.ndim == 3, "invalid batched gate scales")
    require(up_scale.dtype == mx.uint8 and up_scale.shape == gate_scale.shape, "invalid batched up scales")
    require(gate_global_scale.dtype == mx.float32 and gate_global_scale.ndim == 1, "invalid gate globals")
    require(up_global_scale.dtype == mx.float32 and up_global_scale.ndim == 1, "invalid up globals")
    require(
        selected_experts.dtype == mx.uint32 and selected_experts.ndim == 2,
        "batched selected experts must be a uint32 matrix",
    )
    require(vectors.dtype == mx.float32 and vectors.ndim == 2, "invalid selected batched inputs")
    experts, rows, packed_columns = gate_weight.shape
    tokens, top_k = selected_experts.shape
    columns = packed_columns * 2
    require(tokens > 0 and top_k > 0 and vectors.shape == (tokens, columns), "selected batched shape mismatch")
    require(gate_global_scale.shape == (experts,), "batched gate-global shape mismatch")
    require(up_global_scale.shape == (experts,), "batched up-global shape mismatch")
    require(columns % 16 == 0, "selected batched input is not block aligned")
    require(gate_scale.shape == (experts, rows, columns // 16), "selected batched scale mismatch")
    if rows_per_simdgroup is None:
        rows_per_simdgroup = 2 if rows % 2 == 0 else 1
    require(
        simdgroups_per_threadgroup in (8, 16, 32),
        "invalid selected batched SIMD-group count",
    )
    require(
        rows_per_simdgroup in (1, 2, 4),
        "invalid selected batched SIMD-group rows",
    )
    require(
        rows % rows_per_simdgroup == 0,
        "selected batched rows are not SIMD-group aligned",
    )
    work_items = tokens * top_k * (rows // rows_per_simdgroup)
    threads = simdgroups_per_threadgroup * 32
    return _batched_selected_paired_kernel(
        inputs=[
            gate_weight,
            gate_scale,
            gate_global_scale,
            up_weight,
            up_scale,
            up_global_scale,
            selected_experts,
            vectors,
        ],
        template=[
            ("TOKENS", tokens),
            ("EXPERTS", experts),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("TOPK", top_k),
            ("SIMDGROUPS_PER_THREADGROUP", simdgroups_per_threadgroup),
            ("ROWS_PER_SIMDGROUP", rows_per_simdgroup),
        ],
        grid=(
            ((work_items + simdgroups_per_threadgroup - 1)
             // simdgroups_per_threadgroup) * threads,
            1,
            1,
        ),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tokens, top_k, 2, rows)],
        output_dtypes=[mx.float32],
    )[0]


def nvfp4_batched_selected_weighted_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    selected_experts: mx.array,
    vectors: mx.array,
    routing_weights: mx.array,
    *,
    row_groups_per_threadgroup: int | None = None,
    rows_per_simdgroup: int | None = None,
) -> mx.array:
    """Evaluate and reduce each token's selected down projections."""
    require(packed_weight.dtype == mx.uint8 and packed_weight.ndim == 3, "invalid batched down stack")
    require(block_scale.dtype == mx.uint8 and block_scale.ndim == 3, "invalid batched down scales")
    require(global_scale.dtype == mx.float32 and global_scale.ndim == 1, "invalid batched globals")
    require(
        selected_experts.dtype == mx.uint32 and selected_experts.ndim == 2,
        "batched down experts must be a uint32 matrix",
    )
    require(vectors.dtype == mx.float32 and vectors.ndim == 3, "invalid batched down inputs")
    require(
        routing_weights.dtype in (mx.bfloat16, mx.float32) and routing_weights.ndim == 2,
        "invalid batched routing weights",
    )
    experts, rows, packed_columns = packed_weight.shape
    tokens, top_k = selected_experts.shape
    columns = packed_columns * 2
    require(tokens > 0 and 0 < top_k <= 8, "invalid batched down shape")
    require(global_scale.shape == (experts,), "batched down global shape mismatch")
    require(columns % 16 == 0, "batched down input is not block aligned")
    require(block_scale.shape == (experts, rows, columns // 16), "batched down scale mismatch")
    require(vectors.shape == (tokens, top_k, columns), "batched down input shape mismatch")
    require(routing_weights.shape == (tokens, top_k), "batched routing shape mismatch")
    if rows_per_simdgroup is None:
        rows_per_simdgroup = 4 if rows % 4 == 0 else 2 if rows % 2 == 0 else 1
    if row_groups_per_threadgroup is None:
        available_groups = rows // rows_per_simdgroup
        row_groups_per_threadgroup = (
            4 if available_groups % 4 == 0 else 2 if available_groups % 2 == 0 else 1
        )
    require(
        row_groups_per_threadgroup in (1, 2, 4),
        "invalid batched down row-group count",
    )
    require(
        rows_per_simdgroup in (1, 2, 4),
        "invalid batched down SIMD-group rows",
    )
    rows_per_threadgroup = row_groups_per_threadgroup * rows_per_simdgroup
    require(rows % rows_per_threadgroup == 0, "batched down rows are not group aligned")
    kernel = (
        _batched_selected_weighted_bf16_kernel
        if routing_weights.dtype == mx.bfloat16
        else _batched_selected_weighted_f32_kernel
    )
    return kernel(
        inputs=[
            packed_weight,
            block_scale,
            global_scale,
            selected_experts,
            vectors,
            routing_weights,
        ],
        template=[
            ("TOKENS", tokens),
            ("EXPERTS", experts),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("TOPK", top_k),
            ("ROW_GROUPS_PER_THREADGROUP", row_groups_per_threadgroup),
            ("ROWS_PER_SIMDGROUP", rows_per_simdgroup),
        ],
        grid=(
            tokens * (rows // rows_per_threadgroup)
            * row_groups_per_threadgroup * top_k * 32,
            1,
            1,
        ),
        threadgroup=(row_groups_per_threadgroup * top_k * 32, 1, 1),
        output_shapes=[(tokens, rows)],
        output_dtypes=[routing_weights.dtype],
    )[0]


def nvfp4_selected_weighted_rows4_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    selected_experts: mx.array,
    vectors: mx.array,
    routing_weights: mx.array,
) -> mx.array:
    """Evaluate one routed-down token with four output rows per SIMD group."""
    require(
        selected_experts.ndim == 1 and vectors.ndim == 2
        and routing_weights.ndim == 1,
        "invalid one-token routed-down input",
    )
    rows = packed_weight.shape[1] if packed_weight.ndim == 3 else 0
    require(rows > 0 and rows % 4 == 0, "routed-down rows are not four-way aligned")
    return nvfp4_batched_selected_weighted_matvec(
        packed_weight,
        block_scale,
        global_scale,
        selected_experts[None, :],
        vectors[None, :, :],
        routing_weights[None, :],
        row_groups_per_threadgroup=1,
        rows_per_simdgroup=4,
    )[0]


def nvfp4_selected_shared_weighted_rows4_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    shared_weight: mx.array,
    shared_block_scale: mx.array,
    shared_global_scale: mx.array,
    selected_experts: mx.array,
    vectors: mx.array,
    shared_vector: mx.array,
    routing_weights: mx.array,
    shared_multiplier: mx.array,
) -> mx.array:
    """Fuse one token's routed/shared down projections and gated merge."""
    require(packed_weight.dtype == mx.uint8 and packed_weight.ndim == 3, "invalid routed stack")
    require(block_scale.dtype == mx.uint8 and block_scale.ndim == 3, "invalid routed scales")
    require(global_scale.dtype == mx.float32 and global_scale.ndim == 1, "invalid routed globals")
    require(shared_weight.dtype == mx.uint8 and shared_weight.ndim == 2, "invalid shared weight")
    require(shared_block_scale.dtype == mx.uint8 and shared_block_scale.ndim == 2, "invalid shared scales")
    require(
        shared_global_scale.dtype == mx.float32 and shared_global_scale.shape == (1,),
        "invalid shared global",
    )
    require(selected_experts.dtype == mx.uint32 and selected_experts.ndim == 1, "invalid routed experts")
    require(vectors.dtype == mx.float32 and vectors.ndim == 2, "invalid routed inputs")
    require(shared_vector.dtype == mx.float32 and shared_vector.ndim == 1, "invalid shared input")
    require(
        routing_weights.dtype == mx.bfloat16 and routing_weights.ndim == 1,
        "fused routing weights must be BF16",
    )
    require(
        shared_multiplier.dtype == mx.bfloat16 and shared_multiplier.shape == (),
        "fused shared multiplier must be a BF16 scalar",
    )
    experts, rows, packed_columns = packed_weight.shape
    columns = packed_columns * 2
    top_k = selected_experts.size
    require(rows > 0 and rows % 4 == 0 and 0 < top_k <= 8, "invalid fused down shape")
    require(global_scale.shape == (experts,), "fused routed global shape mismatch")
    require(block_scale.shape == (experts, rows, columns // 16), "fused routed scale mismatch")
    require(shared_weight.shape == (rows, packed_columns), "fused shared weight mismatch")
    require(shared_block_scale.shape == (rows, columns // 16), "fused shared scale mismatch")
    require(vectors.shape == (top_k, columns), "fused routed input mismatch")
    require(shared_vector.shape == (columns,), "fused shared input mismatch")
    require(routing_weights.shape == (top_k,), "fused routing shape mismatch")
    simdgroups = top_k + 1
    threads = simdgroups * 32
    return _selected_shared_weighted_rows4_kernel(
        inputs=[
            packed_weight,
            block_scale,
            global_scale,
            shared_weight,
            shared_block_scale,
            shared_global_scale,
            selected_experts,
            vectors,
            shared_vector,
            routing_weights,
            shared_multiplier,
        ],
        template=[
            ("EXPERTS", experts),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("TOPK", top_k),
        ],
        grid=((rows // 4) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.bfloat16],
    )[0]


def load_weight(
    source_path: Path, prefix: str
) -> tuple[SafetensorsFile, NVFP4Weight, mx.array, mx.array, mx.array]:
    source = SafetensorsFile(source_path)
    try:
        reference = NVFP4Weight(source, prefix)
        packed = mx.array(
            memoryview(source.tensor_bytes(reference.weight_name)),
            dtype=mx.uint8,
        ).reshape(reference.rows, reference.packed_columns)
        scales = mx.array(
            memoryview(source.tensor_bytes(reference.scale_name)),
            dtype=mx.uint8,
        ).reshape(reference.rows, reference.blocks_per_row)
        global_scale = mx.array([reference.global_scale], dtype=mx.float32)
        mx.eval(packed, scales, global_scale)
        return source, reference, packed, scales, global_scale
    except Exception:
        source.close()
        raise


def benchmark(source_path: Path, prefix: str, repeats: int) -> dict[str, float]:
    source, reference, packed, scales, global_scale = load_weight(source_path, prefix)
    try:
        values = [
            math.sin(column * 0.013) + math.cos(column * 0.007) * 0.25
            for column in range(reference.columns)
        ]
        vector = mx.array(values, dtype=mx.float32)
        output = nvfp4_matvec(packed, scales, global_scale, vector)
        mx.eval(output)
        mx.synchronize()

        started = time.perf_counter()
        outputs = [
            nvfp4_matvec(packed, scales, global_scale, vector)
            for _ in range(repeats)
        ]
        mx.eval(*outputs)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        actual = outputs[-1].tolist()
        expected = [reference.matvec_row(row, values) for row in range(reference.rows)]
        error2 = math.fsum((left - right) ** 2 for left, right in zip(actual, expected))
        reference2 = math.fsum(value * value for value in expected)
        relative_l2 = math.sqrt(error2 / max(reference2, 1e-30))
        max_abs = max(abs(left - right) for left, right in zip(actual, expected))
        per_call_ms = elapsed * 1000 / repeats
        bytes_per_call = (
            len(source.tensor_bytes(reference.weight_name))
            + len(source.tensor_bytes(reference.scale_name))
            + (reference.rows + reference.columns) * 4
        )
        return {
            "ms": per_call_ms,
            "bandwidth_gbs": bytes_per_call / (per_call_ms * 1e6),
            "relative_l2": relative_l2,
            "max_abs": max_abs,
        }
    finally:
        source.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tensor-prefix", required=True)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        result = benchmark(args.source, args.tensor_prefix, args.repeats)
        print(
            f"mlx-nvfp4 prefix={args.tensor_prefix} ms={result['ms']:.6f} "
            f"bandwidth_gbs={result['bandwidth_gbs']:.2f} "
            f"relative_l2={result['relative_l2']:.9g} "
            f"max_abs={result['max_abs']:.9g}"
        )
        require(
            result["relative_l2"] <= 2e-5 and result["max_abs"] <= 2e-4,
            "MLX NVFP4 drift exceeds tolerance",
        )
        return 0
    except (NVFP4Error, OSError, ValueError) as exc:
        print(f"ornith35 MLX NVFP4 error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

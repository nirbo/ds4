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

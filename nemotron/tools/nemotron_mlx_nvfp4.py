#!/usr/bin/env python3
"""MLX composition wrapper for the packed Nemotron NVFP4 Metal kernel."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_nvfp4 import NVFP4Weight, SafetensorsFile, locate_prefix


KERNEL_HEADER = r"""
constant float nemotron_e2m1_values[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

inline float nemotron_decode_e2m1(uchar nibble) {
    float value = nemotron_e2m1_values[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

inline float nemotron_decode_e4m3fn(uchar bits) {
    uint exponent = (bits >> 3) & 15u;
    uint mantissa = bits & 7u;
    float value;
    if (exponent == 0u) {
        value = float(mantissa) * 0.001953125f;
    } else if (exponent == 15u && mantissa == 7u) {
        value = NAN;
    } else {
        value = (1.0f + float(mantissa) * 0.125f) * exp2(float(int(exponent) - 7));
    }
    return (bits & 128u) ? -value : value;
}
"""

KERNEL_SOURCE = r"""
uint row = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (row >= ROWS) return;
uint packed_columns = COLUMNS >> 1;
uint blocks_per_row = COLUMNS >> 4;
float sum = 0.0f;
for (uint block = thread_index_in_simdgroup; block < blocks_per_row; block += 32u) {
    float scale = nemotron_decode_e4m3fn(block_scale[row * blocks_per_row + block])
        * global_scale[0];
    uint column_base = block << 4;
    uint packed_base = row * packed_columns + (column_base >> 1);
    for (uint pair = 0; pair < 8u; pair++) {
        uchar packed = packed_weight[packed_base + pair];
        uint column = column_base + (pair << 1);
        sum += nemotron_decode_e2m1(packed & 15u) * scale * input[column];
        sum += nemotron_decode_e2m1(packed >> 4) * scale * input[column + 1u];
    }
}
sum = simd_sum(sum);
if (thread_index_in_simdgroup == 0) output[row] = sum;
"""

_kernel = mx.fast.metal_kernel(
    name="nemotron_nvfp4_matvec_f32",
    input_names=["packed_weight", "block_scale", "global_scale", "input"],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=KERNEL_SOURCE,
)


def nvfp4_matvec(
    packed_weight: mx.array,
    block_scale: mx.array,
    global_scale: mx.array,
    vector: mx.array,
) -> mx.array:
    require(packed_weight.dtype == mx.uint8 and packed_weight.ndim == 2, "invalid MLX packed weight")
    require(block_scale.dtype == mx.uint8 and block_scale.ndim == 2, "invalid MLX block scale")
    require(global_scale.dtype == mx.float32 and global_scale.size == 1, "invalid MLX global scale")
    require(vector.dtype == mx.float32 and vector.ndim == 1, "invalid MLX input vector")
    rows, packed_columns = packed_weight.shape
    columns = packed_columns * 2
    require(columns % 16 == 0 and vector.size == columns, "MLX NVFP4 input shape mismatch")
    require(block_scale.shape == (rows, columns // 16), "MLX NVFP4 scale shape mismatch")
    outputs = _kernel(
        inputs=[packed_weight, block_scale, global_scale, vector],
        template=[("ROWS", rows), ("COLUMNS", columns)],
        grid=(((rows + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def load_weight(source_dir: Path, prefix: str) -> tuple[SafetensorsFile, NVFP4Weight, mx.array, mx.array, mx.array]:
    index = load_json(source_dir / "model.safetensors.index.json")
    shard_path, _ = locate_prefix(source_dir, index, prefix)
    shard = SafetensorsFile(shard_path)
    reference = NVFP4Weight(shard, prefix)
    packed = mx.array(memoryview(shard.tensor_bytes(reference.weight_name)), dtype=mx.uint8).reshape(
        reference.rows, reference.packed_columns
    )
    scales = mx.array(memoryview(shard.tensor_bytes(reference.scale_name)), dtype=mx.uint8).reshape(
        reference.rows, reference.blocks_per_row
    )
    global_scale = mx.array([reference.global_scale], dtype=mx.float32)
    return shard, reference, packed, scales, global_scale


def benchmark(source_dir: Path, prefix: str, repeats: int) -> dict[str, float]:
    shard, reference, packed, scales, global_scale = load_weight(source_dir, prefix)
    try:
        values = [math.sin(column * 0.013) + math.cos(column * 0.007) * 0.25 for column in range(reference.columns)]
        vector = mx.array(values, dtype=mx.float32)
        output = nvfp4_matvec(packed, scales, global_scale, vector)
        mx.eval(output)
        mx.synchronize()

        start = time.perf_counter()
        outputs = [
            nvfp4_matvec(packed, scales, global_scale, vector)
            for _ in range(repeats)
        ]
        mx.eval(*outputs)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        output = outputs[-1]
        actual = output.tolist()
        expected = [reference.matvec_row(row, values) for row in range(reference.rows)]
        error2 = math.fsum((left - right) ** 2 for left, right in zip(actual, expected))
        reference2 = math.fsum(value * value for value in expected)
        relative_l2 = math.sqrt(error2 / max(reference2, 1e-30))
        max_abs = max(abs(left - right) for left, right in zip(actual, expected))
        per_call_ms = elapsed * 1000 / repeats
        bytes_per_call = len(shard.tensor_bytes(reference.weight_name)) + len(
            shard.tensor_bytes(reference.scale_name)
        ) + (reference.rows + reference.columns) * 4
        return {
            "ms": per_call_ms,
            "bandwidth_gbs": bytes_per_call / (per_call_ms * 1e6),
            "relative_l2": relative_l2,
            "max_abs": max_abs,
        }
    finally:
        shard.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--tensor-prefix", required=True)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        result = benchmark(args.source_dir, args.tensor_prefix, args.repeats)
        print(
            f"mlx nvfp4: prefix={args.tensor_prefix} ms={result['ms']:.6f} "
            f"bandwidth={result['bandwidth_gbs']:.2f}GB/s "
            f"relative_l2={result['relative_l2']:.9g} max_abs={result['max_abs']:.9g}"
        )
        require(result["relative_l2"] <= 2e-5 and result["max_abs"] <= 2e-4, "MLX NVFP4 drift exceeds tolerance")
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron MLX NVFP4 error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

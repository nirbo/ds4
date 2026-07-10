#!/usr/bin/env python3
"""Mixed-precision MLX linear primitives for the Nemotron ModelOpt checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_nvfp4 import SafetensorsFile, decode_e4m3fn


FP8_HEADER = r"""
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

FP8_SOURCE = r"""
uint row = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (row >= ROWS) return;
float sum = 0.0f;
uint base = row * COLUMNS;
for (uint column = thread_index_in_simdgroup; column < COLUMNS; column += 32u) {
    sum += nemotron_decode_e4m3fn(weight[base + column]) * input[column];
}
sum = simd_sum(sum) * global_scale[0];
if (thread_index_in_simdgroup == 0) output[row] = sum;
"""

_fp8_kernel = mx.fast.metal_kernel(
    name="nemotron_modelopt_fp8_matvec_f32",
    input_names=["weight", "global_scale", "input"],
    output_names=["output"],
    header=FP8_HEADER,
    source=FP8_SOURCE,
)


_unity_mxfp8_scales: dict[tuple[int, int], mx.array] = {}


def fp8_matvec_custom(weight: mx.array, global_scale: mx.array, vector: mx.array) -> mx.array:
    require(weight.dtype == mx.uint8 and weight.ndim == 2, "invalid ModelOpt FP8 weight")
    require(global_scale.dtype == mx.float32 and global_scale.size == 1, "invalid ModelOpt FP8 scale")
    require(vector.dtype == mx.float32 and vector.ndim == 1, "invalid ModelOpt FP8 input")
    rows, columns = weight.shape
    require(vector.size == columns, "ModelOpt FP8 input shape mismatch")
    return _fp8_kernel(
        inputs=[weight, global_scale.reshape(1), vector],
        template=[("ROWS", rows), ("COLUMNS", columns)],
        grid=(((rows + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )[0]


def fp8_matvec(weight: mx.array, global_scale: mx.array, vector: mx.array) -> mx.array:
    """Use MLX's optimized MXFP8 qmm with exact unity block scales."""

    require(weight.dtype == mx.uint8 and weight.ndim == 2, "invalid ModelOpt FP8 weight")
    require(global_scale.dtype == mx.float32 and global_scale.size == 1, "invalid ModelOpt FP8 scale")
    require(vector.dtype == mx.float32 and vector.ndim == 1, "invalid ModelOpt FP8 input")
    rows, columns = weight.shape
    require(columns % 32 == 0 and vector.size == columns, "ModelOpt FP8 input shape mismatch")
    scale_shape = (rows, columns // 32)
    unity_scales = _unity_mxfp8_scales.get(scale_shape)
    if unity_scales is None:
        # E8M0 exponent 127 represents exactly 1.0.
        unity_scales = mx.full(scale_shape, 127, dtype=mx.uint8)
        _unity_mxfp8_scales[scale_shape] = unity_scales
    return mx.quantized_matmul(
        (vector * global_scale.reshape(()))[None, :],
        weight.view(mx.uint32),
        unity_scales,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )[0]


def bf16_matvec(weight: mx.array, vector: mx.array) -> mx.array:
    require(weight.dtype == mx.bfloat16 and weight.ndim == 2, "invalid BF16 weight")
    require(vector.ndim == 1 and vector.size == weight.shape[1], "BF16 input shape mismatch")
    return weight @ vector


def locate_tensor(source_dir: Path, name: str) -> tuple[Path, dict]:
    index = load_json(source_dir / "model.safetensors.index.json")
    shard_name = index.get("weight_map", {}).get(name)
    require(isinstance(shard_name, str), f"tensor is absent from index: {name}")
    tensors = mx.load(str(source_dir / shard_name))
    require(name in tensors, f"tensor is absent from shard: {name}")
    return source_dir / shard_name, tensors


def benchmark(source_dir: Path, prefix: str, repeats: int) -> dict[str, float]:
    weight_name = f"{prefix}.weight"
    scale_name = f"{prefix}.weight_scale"
    shard_path, tensors = locate_tensor(source_dir, weight_name)
    require(scale_name in tensors, f"missing ModelOpt FP8 scale: {scale_name}")
    weight = tensors[weight_name]
    scale = tensors[scale_name].reshape(1).astype(mx.float32)
    require(weight.dtype == mx.uint8 and weight.ndim == 2, "real FP8 tensor did not load as U8")
    rows, columns = weight.shape
    values = [math.sin(column * 0.011) + math.cos(column * 0.003) * 0.2 for column in range(columns)]
    vector = mx.array(values, dtype=mx.float32)
    output = fp8_matvec(weight, scale, vector)
    mx.eval(output)
    mx.synchronize()

    shard = SafetensorsFile(shard_path)
    try:
        scalar_scale = shard.f32_scalar(scale_name)
        sampled_rows = sorted({0, rows // 3, rows // 2, rows - 1})
        actual = output[sampled_rows].tolist()
        expected = []
        for row in sampled_rows:
            encoded = shard.tensor_range(weight_name, row * columns, columns)
            expected.append(
                math.fsum(decode_e4m3fn(byte) * value for byte, value in zip(encoded, values))
                * scalar_scale
            )
        error2 = math.fsum((left - right) ** 2 for left, right in zip(actual, expected))
        reference2 = math.fsum(value * value for value in expected)
        relative_l2 = math.sqrt(error2 / max(reference2, 1e-30))
        max_abs = max(abs(left - right) for left, right in zip(actual, expected))
    finally:
        shard.close()

    started = time.perf_counter()
    outputs = [fp8_matvec(weight, scale, vector) for _ in range(repeats)]
    mx.eval(*outputs)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    per_call_ms = elapsed * 1000 / repeats
    bytes_per_call = weight.size + (rows + columns + 1) * 4
    return {
        "rows": rows,
        "columns": columns,
        "ms": per_call_ms,
        "bandwidth_gbs": bytes_per_call / (per_call_ms * 1e6),
        "relative_l2": relative_l2,
        "max_abs": max_abs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--tensor-prefix", default="backbone.layers.0.mixer.in_proj")
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        result = benchmark(args.source_dir, args.tensor_prefix, args.repeats)
        print(
            f"mlx fp8: prefix={args.tensor_prefix} shape={result['rows']}x{result['columns']} "
            f"ms={result['ms']:.6f} bandwidth={result['bandwidth_gbs']:.2f}GB/s "
            f"relative_l2={result['relative_l2']:.9g} max_abs={result['max_abs']:.9g}"
        )
        require(result["relative_l2"] <= 2e-5 and result["max_abs"] <= 2e-3, "FP8 drift exceeds tolerance")
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron MLX linear error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Checked MLX affine quantization boundary for Ornith-35 vocabulary matrices."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ornith35_moe_reference import require


@dataclass(frozen=True)
class MLXAffineQuantizedMatrix:
    packed: mx.array
    scales: mx.array
    biases: mx.array
    shape: tuple[int, int]
    group_size: int
    bits: int


def validate(matrix: MLXAffineQuantizedMatrix) -> None:
    require(isinstance(matrix, MLXAffineQuantizedMatrix), "invalid quantized matrix")
    require(len(matrix.shape) == 2, "quantized matrix shape mismatch")
    rows, columns = matrix.shape
    require(rows > 0 and columns > 0, "quantized matrix shape mismatch")
    require(matrix.group_size in (32, 64, 128), "invalid affine group size")
    require(matrix.bits in (2, 3, 4, 5, 6, 8), "invalid affine bit width")
    require(columns % matrix.group_size == 0, "quantized columns are not grouped")
    require(columns * matrix.bits % 32 == 0, "quantized columns are not packable")
    require(matrix.packed.dtype == mx.uint32, "quantized matrix payload must be uint32")
    require(
        matrix.packed.shape == (rows, columns * matrix.bits // 32),
        "quantized matrix payload shape mismatch",
    )
    scale_shape = (rows, columns // matrix.group_size)
    require(matrix.scales.shape == scale_shape, "quantized scale shape mismatch")
    require(matrix.biases.shape == scale_shape, "quantized bias shape mismatch")
    require(matrix.scales.dtype == mx.bfloat16, "quantized scales must be BF16")
    require(matrix.biases.dtype == mx.bfloat16, "quantized biases must be BF16")


def quantize_affine(
    weight: mx.array,
    *,
    bits: int = 8,
    group_size: int = 32,
) -> MLXAffineQuantizedMatrix:
    require(weight.ndim == 2, "affine quantization requires a matrix")
    require(weight.dtype == mx.bfloat16, "affine source matrix must be BF16")
    packed, scales, biases = mx.quantize(
        weight,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    matrix = MLXAffineQuantizedMatrix(
        packed=packed,
        scales=scales,
        biases=biases,
        shape=(weight.shape[0], weight.shape[1]),
        group_size=group_size,
        bits=bits,
    )
    validate(matrix)
    return matrix


def project(matrix: MLXAffineQuantizedMatrix, hidden: mx.array) -> mx.array:
    validate(matrix)
    require(hidden.shape[-1] == matrix.shape[1], "quantized projection shape mismatch")
    require(hidden.dtype == matrix.scales.dtype, "quantized projection dtype mismatch")
    return mx.quantized_matmul(
        hidden,
        matrix.packed,
        matrix.scales,
        matrix.biases,
        transpose=True,
        group_size=matrix.group_size,
        bits=matrix.bits,
        mode="affine",
    )


def dequantize_rows(
    matrix: MLXAffineQuantizedMatrix,
    indices: int | mx.array,
) -> mx.array:
    validate(matrix)
    scalar = isinstance(indices, int)
    selected = slice(indices, indices + 1) if scalar else indices
    result = mx.dequantize(
        matrix.packed[selected],
        scales=matrix.scales[selected],
        biases=matrix.biases[selected],
        group_size=matrix.group_size,
        bits=matrix.bits,
        mode="affine",
    )
    return result[0] if scalar else result


def stored_bytes(matrix: MLXAffineQuantizedMatrix) -> int:
    validate(matrix)
    return sum(
        value.size * value.itemsize
        for value in (matrix.packed, matrix.scales, matrix.biases)
    )

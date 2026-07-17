#!/usr/bin/env python3
"""Checked MLX affine quantization boundary for Ornith-35 vocabulary matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import mlx.core as mx

from ornith35_moe_reference import require
from ornith35_nvfp4 import SafetensorsFile


@dataclass(frozen=True)
class MLXAffineQuantizedMatrix:
    packed: mx.array
    scales: mx.array
    biases: mx.array
    shape: tuple[int, int]
    group_size: int
    bits: int


class MLXMappedBF16Matrix:
    """Read exact BF16 rows from a retained, verified safetensors mapping."""

    def __init__(self, path: Path, name: str, shape: tuple[int, int]):
        self._source = SafetensorsFile(path)
        try:
            entry = self._source.entry(name)
            require(entry.get("dtype") == "BF16", "mapped matrix must be BF16")
            require(entry.get("shape") == list(shape), "mapped matrix shape mismatch")
            expected_bytes = shape[0] * shape[1] * 2
            require(
                self._source.tensor_nbytes(name) == expected_bytes,
                "mapped matrix payload mismatch",
            )
            start, _ = entry["data_offsets"]
            self._offset = self._source.payload_offset + start
            self._row_bytes = shape[1] * 2
            self.shape = shape
            self.dtype = mx.bfloat16
        except Exception:
            self._source.close()
            raise

    def close(self) -> None:
        source = getattr(self, "_source", None)
        if source is not None and not source._map.closed:
            source.close()

    def __del__(self) -> None:
        self.close()

    def rows(self, token_ids: Sequence[int]) -> mx.array:
        values = tuple(token_ids)
        require(bool(values), "mapped row selection must not be empty")
        require(
            all(isinstance(token_id, int) and 0 <= token_id < self.shape[0] for token_id in values),
            "mapped row index is out of range",
        )
        payload = b"".join(
            self._source._map[
                self._offset + token_id * self._row_bytes :
                self._offset + (token_id + 1) * self._row_bytes
            ]
            for token_id in values
        )
        return (
            mx.array(memoryview(payload), dtype=mx.uint8)
            .view(mx.bfloat16)
            .reshape(len(values), self.shape[1])
        )

    def row(self, token_id: int) -> mx.array:
        require(
            isinstance(token_id, int) and 0 <= token_id < self.shape[0],
            "mapped row index is out of range",
        )
        payload = self._source._map[
            self._offset + token_id * self._row_bytes :
            self._offset + (token_id + 1) * self._row_bytes
        ]
        return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.bfloat16)


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

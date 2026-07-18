#!/usr/bin/env python3
"""Checked MLX affine quantization boundary for Ornith-35 vocabulary matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import mlx.core as mx

from ornith35_moe_reference import require
from ornith35_nvfp4 import SafetensorsFile


EXACT_BF16_ROWS_KERNEL_SOURCE = r"""
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x * 8u + group;
if (row >= row_count) return;
float sum = 0.0f;
for (uint column = lane * 4u; column < 2048u; column += 128u) {
    uint weight_base = row * 2048u + column;
    float input0 = float(hidden[column]);
    float input1 = float(hidden[column + 1u]);
    float input2 = float(hidden[column + 2u]);
    float input3 = float(hidden[column + 3u]);
    sum += float(weight[weight_base]) * input0;
    sum += float(weight[weight_base + 1u]) * input1;
    sum += float(weight[weight_base + 2u]) * input2;
    sum += float(weight[weight_base + 3u]) * input3;
}
for (ushort offset = 16; offset >= 1; offset >>= 1) {
    sum += simd_shuffle_down(sum, offset);
}
if (lane == 0u) output[row] = bfloat16_t(sum);
"""


_exact_bf16_rows_kernel = mx.fast.metal_kernel(
    name="ornith35_vocab_exact_bf16_rows",
    input_names=["weight", "hidden", "row_count"],
    output_names=["output"],
    source=EXACT_BF16_ROWS_KERNEL_SOURCE,
)


EXACT_BF16_BLOCK_KERNEL_SOURCE = r"""
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x * 8u + group;
if (row >= row_count) return;
float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
for (uint column = lane * 4u; column < 2048u; column += 128u) {
    uint weight_base = row * 2048u + column;
    float weight0 = float(weight[weight_base]);
    float weight1 = float(weight[weight_base + 1u]);
    float weight2 = float(weight[weight_base + 2u]);
    float weight3 = float(weight[weight_base + 3u]);
    for (uint token = 0u; token < TOKENS; ++token) {
        uint hidden_base = token * 2048u + column;
        sums[token] += weight0 * float(hidden[hidden_base]);
        sums[token] += weight1 * float(hidden[hidden_base + 1u]);
        sums[token] += weight2 * float(hidden[hidden_base + 2u]);
        sums[token] += weight3 * float(hidden[hidden_base + 3u]);
    }
}
for (ushort offset = 16; offset >= 1; offset >>= 1) {
    for (uint token = 0u; token < TOKENS; ++token) {
        sums[token] += simd_shuffle_down(sums[token], offset);
    }
}
if (lane == 0u) {
    for (uint token = 0u; token < TOKENS; ++token) {
        output[token * row_count + row] = bfloat16_t(sums[token]);
    }
}
"""


_exact_bf16_block_kernel = mx.fast.metal_kernel(
    name="ornith35_vocab_exact_bf16_block",
    input_names=["weight", "hidden", "row_count"],
    output_names=["output"],
    source=EXACT_BF16_BLOCK_KERNEL_SOURCE,
)


@dataclass(frozen=True)
class MLXAffineQuantizedMatrix:
    packed: mx.array
    scales: mx.array
    biases: mx.array
    shape: tuple[int, int]
    group_size: int
    bits: int
    reference: MLXMappedBF16Matrix | None = None


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
    if matrix.reference is not None:
        require(
            isinstance(matrix.reference, MLXMappedBF16Matrix)
            and matrix.reference.shape == matrix.shape
            and matrix.reference.dtype == mx.bfloat16,
            "quantized matrix reference mismatch",
        )


def quantize_affine(
    weight: mx.array,
    *,
    bits: int = 8,
    group_size: int = 32,
    reference: MLXMappedBF16Matrix | None = None,
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
        reference=reference,
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


def project_bf16_rows_exact(weight: mx.array, hidden: mx.array) -> mx.array:
    """Reproduce the production full-head BF16 GEMV for selected rows."""
    require(
        weight.dtype == mx.bfloat16
        and weight.ndim == 2
        and 0 < weight.shape[0] <= 256
        and weight.shape[1] == 2048,
        "exact vocabulary-row weight mismatch",
    )
    require(
        hidden.dtype == mx.bfloat16 and hidden.shape == (2048,),
        "exact vocabulary-row hidden mismatch",
    )
    rows = weight.shape[0]
    return _exact_bf16_rows_kernel(
        inputs=[weight, hidden, mx.array(rows, dtype=mx.uint32)],
        grid=(((rows + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.bfloat16],
    )[0]


def project_bf16_block_exact(weight: mx.array, hidden: mx.array) -> mx.array:
    """Project up to eight hidden rows while reading each BF16 weight once."""
    require(
        weight.dtype == mx.bfloat16
        and weight.ndim == 2
        and weight.shape[0] > 0
        and weight.shape[1] == 2048,
        "exact block vocabulary weight mismatch",
    )
    require(
        hidden.dtype == mx.bfloat16
        and hidden.ndim == 2
        and 1 <= hidden.shape[0] <= 8
        and hidden.shape[1] == 2048,
        "exact block vocabulary hidden mismatch",
    )
    rows = weight.shape[0]
    tokens = hidden.shape[0]
    return _exact_bf16_block_kernel(
        inputs=[weight, hidden, mx.array(rows, dtype=mx.uint32)],
        template=[("TOKENS", tokens)],
        grid=(((rows + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, rows)],
        output_dtypes=[mx.bfloat16],
    )[0]


def exact_candidate_scores(
    matrix: MLXAffineQuantizedMatrix,
    approximate_logits: mx.array,
    hidden: mx.array,
    *,
    candidate_count: int = 64,
) -> tuple[list[int], list[float]]:
    """Rescore a Q8 candidate pool with exact mapped BF16 source rows."""
    validate(matrix)
    require(matrix.reference is not None, "quantized matrix has no exact reference")
    require(
        approximate_logits.dtype == mx.bfloat16
        and approximate_logits.shape == (matrix.shape[0],),
        "candidate logits mismatch",
    )
    require(
        hidden.dtype == mx.bfloat16 and hidden.shape == (matrix.shape[1],),
        "candidate hidden mismatch",
    )
    require(
        1 <= candidate_count <= min(256, matrix.shape[0]),
        "candidate count is outside the exact rerank limit",
    )
    indices = mx.argpartition(
        approximate_logits,
        approximate_logits.size - candidate_count,
    )[-candidate_count:]
    mx.eval(indices)
    token_ids = [int(value) for value in indices.tolist()]
    rows = matrix.reference.rows(token_ids)
    scores = project_bf16_rows_exact(rows, hidden)
    mx.eval(scores)
    return token_ids, [float(value) for value in scores.tolist()]


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

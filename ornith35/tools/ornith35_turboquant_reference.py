#!/usr/bin/env python3
"""Dependency-free PolarQuant and QJL numerical authority for Ornith-35."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Sequence


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]
HEAD_DIM = 256


class TurboQuantError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TurboQuantError(message)


# Generated from the exact spherical-coordinate Beta density in arXiv
# 2504.19874v1 with converged Lloyd-Max iteration. The d=128 values also match
# the independently pinned 0xSero reference files to displayed precision.
_CENTROIDS: dict[tuple[int, int], tuple[float, ...]] = {
    (128, 1): (-0.07066157273809426, 0.07066157273809426),
    (128, 2): (
        -0.13304019825336852,
        -0.03999094521535639,
        0.039990945215356344,
        0.13304019825336846,
    ),
    (128, 3): (
        -0.18839061380207797,
        -0.11813298369899358,
        -0.06658059531595682,
        -0.021602468667239208,
        0.021602468667239208,
        0.06658059531595682,
        0.11813298369899358,
        0.18839061380207797,
    ),
    (128, 4): (
        -0.2376271867309536,
        -0.18079372947217434,
        -0.14176165429673318,
        -0.11024706538276369,
        -0.08279256667309583,
        -0.057744535605257115,
        -0.03413402823112088,
        -0.011296498142743925,
        0.011296498142743847,
        0.03413402823112081,
        0.05774453560525704,
        0.08279256667309573,
        0.11024706538276359,
        0.14176165429673304,
        0.18079372947217423,
        0.23762718673095337,
    ),
    (256, 1): (-0.049916507721605746, 0.049916507721605746),
    (256, 2): (
        -0.09423680066889677,
        -0.02828811999467261,
        0.028288119994672584,
        0.09423680066889675,
    ),
    (256, 3): (
        -0.1338479440323176,
        -0.08375896775388561,
        -0.04716193453734492,
        -0.015295735736674062,
        0.015295735736674062,
        0.04716193453734492,
        0.08375896775388561,
        0.1338479440323176,
    ),
    (256, 4): (
        -0.1693834023763135,
        -0.1285573496586392,
        -0.10066636577797276,
        -0.07821939632405821,
        -0.05870616746679093,
        -0.04092915969184447,
        -0.024188228109530426,
        -0.008004050632916985,
        0.00800405063291698,
        0.024188228109530422,
        0.04092915969184447,
        0.05870616746679093,
        0.07821939632405821,
        0.10066636577797276,
        0.1285573496586392,
        0.1693834023763135,
    ),
}

_MSE_TOTAL: dict[tuple[int, int], float] = {
    (128, 1): 0.36088859368690346,
    (128, 2): 0.11600007032012696,
    (128, 3): 0.0339659261608477,
    (128, 4): 0.009314791876879424,
    (256, 1): 0.362135617760949,
    (256, 2): 0.11674003883805695,
    (256, 3): 0.0342560042687041,
    (256, 4): 0.009407533529022872,
}


@dataclass(frozen=True)
class Codebook:
    dimension: int
    bits: int
    centroids: tuple[float, ...]
    boundaries: tuple[float, ...]
    expected_total_mse: float


@dataclass(frozen=True)
class MSEEncoding:
    indices: tuple[int, ...]
    norm: float
    bits: int


@dataclass(frozen=True)
class ProductEncoding:
    mse: MSEEncoding
    residual_signs: tuple[int, ...]
    residual_norm: float
    total_bits: int


class SplitMixGaussian:
    """Version-stable SplitMix64 plus Box-Muller Gaussian stream."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        require(isinstance(seed, int), "Gaussian seed must be an integer")
        self._state = seed & self._MASK
        self._spare: float | None = None

    def _next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & self._MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return value ^ (value >> 31)

    def _uniform_open(self) -> float:
        return ((self._next_u64() >> 11) + 0.5) / float(1 << 53)

    def next(self) -> float:
        if self._spare is not None:
            value = self._spare
            self._spare = None
            return value
        radius = math.sqrt(-2.0 * math.log(self._uniform_open()))
        angle = 2.0 * math.pi * self._uniform_open()
        self._spare = radius * math.sin(angle)
        return radius * math.cos(angle)


def gaussian_matrix(rows: int, columns: int, seed: int) -> tuple[tuple[float, ...], ...]:
    require(rows > 0 and columns > 0, "Gaussian matrix dimensions must be positive")
    stream = SplitMixGaussian(seed)
    return tuple(tuple(stream.next() for _ in range(columns)) for _ in range(rows))


def codebook(dimension: int, bits: int) -> Codebook:
    centroids = _CENTROIDS.get((dimension, bits))
    require(centroids is not None, "unsupported TurboQuant codebook")
    boundaries = (-1.0,) + tuple(
        (left + right) * 0.5 for left, right in zip(centroids, centroids[1:])
    ) + (1.0,)
    return Codebook(
        dimension=dimension,
        bits=bits,
        centroids=centroids,
        boundaries=boundaries,
        expected_total_mse=_MSE_TOTAL[(dimension, bits)],
    )


def _validate_vector(vector: Vector, dimension: int, name: str) -> None:
    require(len(vector) == dimension, f"{name} dimension mismatch")
    require(all(math.isfinite(value) for value in vector), f"{name} is not finite")


def _validate_matrix(matrix: Matrix, dimension: int, name: str) -> None:
    require(len(matrix) == dimension, f"{name} row count mismatch")
    require(
        all(len(row) == dimension for row in matrix),
        f"{name} column count mismatch",
    )
    require(
        all(math.isfinite(value) for row in matrix for value in row),
        f"{name} is not finite",
    )


def matvec(matrix: Matrix, vector: Vector) -> tuple[float, ...]:
    dimension = len(vector)
    _validate_vector(vector, dimension, "matrix input")
    _validate_matrix(matrix, dimension, "matrix")
    return tuple(math.fsum(value * item for value, item in zip(row, vector)) for row in matrix)


def transpose_matvec(matrix: Matrix, vector: Vector) -> tuple[float, ...]:
    dimension = len(vector)
    _validate_vector(vector, dimension, "transpose input")
    _validate_matrix(matrix, dimension, "matrix")
    return tuple(
        math.fsum(matrix[row][column] * vector[row] for row in range(dimension))
        for column in range(dimension)
    )


def dot(left: Vector, right: Vector) -> float:
    require(len(left) == len(right), "dot-product dimension mismatch")
    return math.fsum(a * b for a, b in zip(left, right))


def l2_norm(vector: Vector) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def quantize_mse(vector: Vector, bits: int, rotation: Matrix) -> MSEEncoding:
    dimension = len(vector)
    selected = codebook(dimension, bits)
    _validate_vector(vector, dimension, "MSE input")
    _validate_matrix(rotation, dimension, "rotation")
    norm = l2_norm(vector)
    require(norm > 0.0, "TurboQuant cannot encode a zero vector")
    rotated = matvec(rotation, tuple(value / norm for value in vector))
    interior = selected.boundaries[1:-1]
    indices = tuple(bisect_right(interior, value) for value in rotated)
    return MSEEncoding(indices=indices, norm=norm, bits=bits)


def dequantize_mse(encoding: MSEEncoding, rotation: Matrix) -> tuple[float, ...]:
    selected = codebook(len(encoding.indices), encoding.bits)
    _validate_matrix(rotation, selected.dimension, "rotation")
    require(
        math.isfinite(encoding.norm) and encoding.norm >= 0.0,
        "MSE norm is invalid",
    )
    require(
        all(0 <= index < len(selected.centroids) for index in encoding.indices),
        "MSE codebook index is invalid",
    )
    rotated = tuple(selected.centroids[index] for index in encoding.indices)
    return tuple(value * encoding.norm for value in transpose_matvec(rotation, rotated))


def quantize_product(
    vector: Vector,
    total_bits: int,
    rotation: Matrix,
    projection: Matrix,
) -> ProductEncoding:
    require(2 <= total_bits <= 5, "product bit width must be in [2, 5]")
    mse = quantize_mse(vector, total_bits - 1, rotation)
    reconstructed = dequantize_mse(mse, rotation)
    residual = tuple(value - estimate for value, estimate in zip(vector, reconstructed))
    residual_norm = l2_norm(residual)
    _validate_matrix(projection, len(vector), "QJL projection")
    projected = matvec(projection, residual)
    signs = tuple(1 if value >= 0.0 else -1 for value in projected)
    return ProductEncoding(
        mse=mse,
        residual_signs=signs,
        residual_norm=residual_norm,
        total_bits=total_bits,
    )


def dequantize_product(
    encoding: ProductEncoding,
    rotation: Matrix,
    projection: Matrix,
) -> tuple[float, ...]:
    dimension = len(encoding.mse.indices)
    require(encoding.mse.bits + 1 == encoding.total_bits, "product bit width mismatch")
    require(
        len(encoding.residual_signs) == dimension
        and all(value in (-1, 1) for value in encoding.residual_signs),
        "QJL sign payload is invalid",
    )
    require(
        math.isfinite(encoding.residual_norm) and encoding.residual_norm >= 0.0,
        "QJL residual norm is invalid",
    )
    _validate_matrix(projection, dimension, "QJL projection")
    base = dequantize_mse(encoding.mse, rotation)
    scale = encoding.residual_norm * math.sqrt(math.pi / 2.0) / dimension
    correction = transpose_matvec(projection, encoding.residual_signs)
    return tuple(value + scale * residual for value, residual in zip(base, correction))


def product_inner_product(
    query: Vector,
    encoding: ProductEncoding,
    rotation: Matrix,
    projection: Matrix,
) -> float:
    dimension = len(query)
    require(len(encoding.mse.indices) == dimension, "product query dimension mismatch")
    base = dequantize_mse(encoding.mse, rotation)
    projected_query = matvec(projection, query)
    correction = (
        encoding.residual_norm
        * math.sqrt(math.pi / 2.0)
        / dimension
        * dot(projected_query, encoding.residual_signs)
    )
    return dot(query, base) + correction


def packed_bits_bytes(items: int, bits: int, alignment: int = 1) -> int:
    require(items > 0 and bits > 0, "packed dimensions must be positive")
    require(alignment > 0, "packed alignment must be positive")
    raw = (items * bits + 7) // 8
    return ((raw + alignment - 1) // alignment) * alignment


def mse_vector_bytes(
    dimension: int,
    bits: int,
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    codebook(dimension, bits)
    require(scalar_bytes in (2, 4), "norm scalar must be BF16/FP16 or FP32")
    return packed_bits_bytes(dimension, bits, alignment) + scalar_bytes


def product_vector_bytes(
    dimension: int,
    total_bits: int,
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    require(2 <= total_bits <= 5, "product bit width must be in [2, 5]")
    codebook(dimension, total_bits - 1)
    require(scalar_bytes in (2, 4), "norm scalar must be BF16/FP16 or FP32")
    indices = packed_bits_bytes(dimension, total_bits - 1, alignment)
    signs = packed_bits_bytes(dimension, 1, alignment)
    return indices + signs + 2 * scalar_bytes


def split_mse_vector_bytes(
    dimensions: Iterable[int],
    bits: Iterable[int],
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    groups = tuple(zip(dimensions, bits))
    require(groups, "mixed MSE profile must have at least one group")
    return sum(
        mse_vector_bytes(size, width, scalar_bytes=scalar_bytes, alignment=alignment)
        for size, width in groups
    )


def split_product_vector_bytes(
    dimensions: Iterable[int],
    total_bits: Iterable[int],
    *,
    scalar_bytes: int = 2,
    alignment: int = 4,
) -> int:
    groups = tuple(zip(dimensions, total_bits))
    require(groups, "mixed product profile must have at least one group")
    return sum(
        product_vector_bytes(size, width, scalar_bytes=scalar_bytes, alignment=alignment)
        for size, width in groups
    )


def cache_payload_bytes(
    tokens: int,
    layers: int,
    kv_heads: int,
    key_bytes: int,
    value_bytes: int,
) -> int:
    require(tokens >= 0 and layers > 0 and kv_heads > 0, "invalid cache geometry")
    require(key_bytes > 0 and value_bytes > 0, "invalid cache vector size")
    return tokens * layers * kv_heads * (key_bytes + value_bytes)

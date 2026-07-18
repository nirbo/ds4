#!/usr/bin/env python3
"""MLX composition for evaluating Ornith-35 PolarQuant and QJL candidates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import struct

import mlx.core as mx

import ornith35_turboquant_reference as reference


@dataclass(frozen=True)
class MLXTransform:
    matrix: mx.array
    sha256: str
    seed: int
    kind: str


@dataclass(frozen=True)
class MLXMSEEncoding:
    indices: mx.array
    norms: mx.array
    bits: int
    dimension: int


@dataclass(frozen=True)
class MLXProductEncoding:
    mse: MLXMSEEncoding
    residual_signs: mx.array
    residual_norms: mx.array
    total_bits: int


@dataclass(frozen=True)
class MLXChannelSplit:
    dimension: int
    high_channels: tuple[int, ...]
    low_channels: tuple[int, ...]
    inverse_order: tuple[int, ...]


@dataclass(frozen=True)
class MLXSplitMSEEncoding:
    split: MLXChannelSplit
    high: MLXMSEEncoding
    low: MLXMSEEncoding


@dataclass(frozen=True)
class MLXSplitProductEncoding:
    split: MLXChannelSplit
    high: MLXProductEncoding
    low: MLXProductEncoding


def require(condition: bool, message: str) -> None:
    if not condition:
        raise reference.TurboQuantError(message)


def _matrix_sha256(matrix: mx.array) -> str:
    require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "transform must be square")
    mx.eval(matrix)
    digest = hashlib.sha256()
    for row in matrix.tolist():
        for value in row:
            digest.update(struct.pack("<f", value))
    return digest.hexdigest()


@lru_cache(maxsize=16)
def haar_rotation(dimension: int, seed: int) -> MLXTransform:
    """Generate the paper's Gaussian-QR rotation on MLX's CPU backend."""
    reference.codebook(dimension, 1)
    gaussian = mx.array(
        reference.gaussian_matrix(dimension, dimension, seed),
        dtype=mx.float32,
    )
    q, r = mx.linalg.qr(gaussian, stream=mx.cpu)
    mx.eval(q, r)
    diagonal = [float(r[index, index].item()) for index in range(dimension)]
    signs = mx.array(
        [1.0 if value >= 0.0 else -1.0 for value in diagonal],
        dtype=mx.float32,
    )
    matrix = q * signs[None, :]
    mx.eval(matrix)
    return MLXTransform(
        matrix=matrix,
        sha256=_matrix_sha256(matrix),
        seed=seed,
        kind="gaussian-qr",
    )


@lru_cache(maxsize=16)
def qjl_projection(dimension: int, seed: int) -> MLXTransform:
    reference.codebook(dimension, 1)
    matrix = mx.array(
        reference.gaussian_matrix(dimension, dimension, seed),
        dtype=mx.float32,
    )
    mx.eval(matrix)
    return MLXTransform(
        matrix=matrix,
        sha256=_matrix_sha256(matrix),
        seed=seed,
        kind="gaussian-projection",
    )


def _validate_transform(transform: MLXTransform, dimension: int, kind: str) -> None:
    require(isinstance(transform, MLXTransform), f"invalid {kind} transform")
    require(
        transform.matrix.shape == (dimension, dimension)
        and transform.matrix.dtype == mx.float32,
        f"{kind} transform geometry mismatch",
    )


def _codebook_arrays(dimension: int, bits: int) -> tuple[mx.array, mx.array]:
    selected = reference.codebook(dimension, bits)
    return (
        mx.array(selected.centroids, dtype=mx.float32),
        mx.array(selected.boundaries[1:-1], dtype=mx.float32),
    )


def channel_split(dimension: int, high_channels: tuple[int, ...]) -> MLXChannelSplit:
    require(dimension > 1, "split dimension must exceed one")
    require(
        high_channels
        and len(high_channels) < dimension
        and len(set(high_channels)) == len(high_channels)
        and all(isinstance(channel, int) and 0 <= channel < dimension for channel in high_channels),
        "invalid high-precision channel set",
    )
    high = tuple(sorted(high_channels))
    selected = frozenset(high)
    low = tuple(channel for channel in range(dimension) if channel not in selected)
    order = high + low
    positions = {channel: index for index, channel in enumerate(order)}
    return MLXChannelSplit(
        dimension=dimension,
        high_channels=high,
        low_channels=low,
        inverse_order=tuple(positions[channel] for channel in range(dimension)),
    )


def _split_vectors(
    vectors: mx.array,
    split: MLXChannelSplit,
) -> tuple[mx.array, mx.array]:
    require(
        vectors.ndim >= 1 and vectors.shape[-1] == split.dimension,
        "channel-split input geometry mismatch",
    )
    high = mx.array(split.high_channels, dtype=mx.uint32)
    low = mx.array(split.low_channels, dtype=mx.uint32)
    return vectors[..., high], vectors[..., low]


def _restore_split(
    high: mx.array,
    low: mx.array,
    split: MLXChannelSplit,
) -> mx.array:
    require(
        high.shape[:-1] == low.shape[:-1]
        and high.shape[-1] == len(split.high_channels)
        and low.shape[-1] == len(split.low_channels),
        "channel-split reconstruction geometry mismatch",
    )
    ordered = mx.concatenate((high, low), axis=-1)
    inverse = mx.array(split.inverse_order, dtype=mx.uint32)
    return ordered[..., inverse]


def quantize_mse(
    vectors: mx.array,
    bits: int,
    rotation: MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXMSEEncoding:
    require(vectors.ndim >= 1, "MSE input rank mismatch")
    dimension = vectors.shape[-1]
    _validate_transform(rotation, dimension, "rotation")
    centroids, boundaries = _codebook_arrays(dimension, bits)
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported norm dtype")
    source = vectors.astype(mx.float32)
    norms = mx.sqrt(mx.sum(source * source, axis=-1, keepdims=True))
    minimum = mx.min(norms)
    mx.eval(minimum)
    require(float(minimum.item()) > 0.0, "TurboQuant cannot encode a zero vector")
    rotated = (source / norms) @ mx.swapaxes(rotation.matrix, -2, -1)
    indices = mx.zeros(rotated.shape, dtype=mx.uint8)
    for boundary in boundaries:
        indices = indices + (rotated >= boundary).astype(mx.uint8)
    stored_norms = norms.astype(norm_dtype)
    mx.eval(indices, stored_norms, centroids)
    return MLXMSEEncoding(
        indices=indices,
        norms=stored_norms,
        bits=bits,
        dimension=dimension,
    )


def dequantize_mse(
    encoding: MLXMSEEncoding,
    rotation: MLXTransform,
) -> mx.array:
    _validate_transform(rotation, encoding.dimension, "rotation")
    require(
        encoding.indices.shape[-1] == encoding.dimension
        and encoding.indices.dtype == mx.uint8,
        "MSE index payload mismatch",
    )
    require(
        encoding.norms.shape == (*encoding.indices.shape[:-1], 1)
        and encoding.norms.dtype in (mx.bfloat16, mx.float32),
        "MSE norm payload mismatch",
    )
    centroids, _ = _codebook_arrays(encoding.dimension, encoding.bits)
    rotated = centroids[encoding.indices]
    return (rotated @ rotation.matrix) * encoding.norms.astype(mx.float32)


def quantize_product(
    vectors: mx.array,
    total_bits: int,
    rotation: MLXTransform,
    projection: MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXProductEncoding:
    require(2 <= total_bits <= 5, "product bit width must be in [2, 5]")
    dimension = vectors.shape[-1]
    _validate_transform(projection, dimension, "QJL projection")
    mse = quantize_mse(
        vectors,
        total_bits - 1,
        rotation,
        norm_dtype=norm_dtype,
    )
    source = vectors.astype(mx.float32)
    residual = source - dequantize_mse(mse, rotation)
    residual_norms = mx.sqrt(mx.sum(residual * residual, axis=-1, keepdims=True))
    signs = (residual @ mx.swapaxes(projection.matrix, -2, -1)) >= 0.0
    stored_residual_norms = residual_norms.astype(norm_dtype)
    mx.eval(signs, stored_residual_norms)
    return MLXProductEncoding(
        mse=mse,
        residual_signs=signs,
        residual_norms=stored_residual_norms,
        total_bits=total_bits,
    )


def dequantize_product(
    encoding: MLXProductEncoding,
    rotation: MLXTransform,
    projection: MLXTransform,
) -> mx.array:
    dimension = encoding.mse.dimension
    require(encoding.mse.bits + 1 == encoding.total_bits, "product bit width mismatch")
    _validate_transform(projection, dimension, "QJL projection")
    require(
        encoding.residual_signs.shape == encoding.mse.indices.shape
        and encoding.residual_signs.dtype == mx.bool_,
        "QJL sign payload mismatch",
    )
    require(
        encoding.residual_norms.shape == encoding.mse.norms.shape
        and encoding.residual_norms.dtype == encoding.mse.norms.dtype,
        "QJL residual norm payload mismatch",
    )
    signs = mx.where(encoding.residual_signs, 1.0, -1.0)
    correction = signs @ projection.matrix
    scale = math.sqrt(math.pi / 2.0) / dimension
    return (
        dequantize_mse(encoding.mse, rotation)
        + correction
        * encoding.residual_norms.astype(mx.float32)
        * scale
    )


def product_inner_products(
    queries: mx.array,
    encoding: MLXProductEncoding,
    rotation: MLXTransform,
    projection: MLXTransform,
) -> mx.array:
    """Return query/key scores without materializing QJL-corrected keys."""
    dimension = encoding.mse.dimension
    require(
        queries.ndim == 2 and queries.shape[-1] == dimension,
        "product query geometry mismatch",
    )
    keys = dequantize_mse(encoding.mse, rotation)
    require(keys.ndim == 2, "product key geometry mismatch")
    base = queries.astype(mx.float32) @ mx.swapaxes(keys, -2, -1)
    projected_queries = queries.astype(mx.float32) @ mx.swapaxes(
        projection.matrix,
        -2,
        -1,
    )
    signs = mx.where(encoding.residual_signs, 1.0, -1.0)
    correction = projected_queries @ mx.swapaxes(signs, -2, -1)
    scale = math.sqrt(math.pi / 2.0) / dimension
    return (
        base
        + correction
        * mx.swapaxes(encoding.residual_norms.astype(mx.float32), -2, -1)
        * scale
    )


def mse_inner_products(
    queries: mx.array,
    encoding: MLXMSEEncoding,
    rotation: MLXTransform,
) -> mx.array:
    keys = dequantize_mse(encoding, rotation)
    require(
        queries.ndim == 2 and keys.ndim == 2 and queries.shape[-1] == keys.shape[-1],
        "MSE inner-product geometry mismatch",
    )
    return queries.astype(mx.float32) @ mx.swapaxes(keys, -2, -1)


def quantize_split_mse(
    vectors: mx.array,
    split: MLXChannelSplit,
    high_bits: int,
    low_bits: int,
    high_rotation: MLXTransform,
    low_rotation: MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXSplitMSEEncoding:
    high, low = _split_vectors(vectors, split)
    return MLXSplitMSEEncoding(
        split=split,
        high=quantize_mse(high, high_bits, high_rotation, norm_dtype=norm_dtype),
        low=quantize_mse(low, low_bits, low_rotation, norm_dtype=norm_dtype),
    )


def dequantize_split_mse(
    encoding: MLXSplitMSEEncoding,
    high_rotation: MLXTransform,
    low_rotation: MLXTransform,
) -> mx.array:
    return _restore_split(
        dequantize_mse(encoding.high, high_rotation),
        dequantize_mse(encoding.low, low_rotation),
        encoding.split,
    )


def split_mse_inner_products(
    queries: mx.array,
    encoding: MLXSplitMSEEncoding,
    high_rotation: MLXTransform,
    low_rotation: MLXTransform,
) -> mx.array:
    keys = dequantize_split_mse(encoding, high_rotation, low_rotation)
    require(
        queries.ndim == 2 and keys.ndim == 2 and queries.shape[-1] == keys.shape[-1],
        "split MSE inner-product geometry mismatch",
    )
    return queries.astype(mx.float32) @ mx.swapaxes(keys, -2, -1)


def quantize_split_product(
    vectors: mx.array,
    split: MLXChannelSplit,
    high_bits: int,
    low_bits: int,
    high_rotation: MLXTransform,
    low_rotation: MLXTransform,
    high_projection: MLXTransform,
    low_projection: MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXSplitProductEncoding:
    high, low = _split_vectors(vectors, split)
    return MLXSplitProductEncoding(
        split=split,
        high=quantize_product(
            high,
            high_bits,
            high_rotation,
            high_projection,
            norm_dtype=norm_dtype,
        ),
        low=quantize_product(
            low,
            low_bits,
            low_rotation,
            low_projection,
            norm_dtype=norm_dtype,
        ),
    )


def dequantize_split_product(
    encoding: MLXSplitProductEncoding,
    high_rotation: MLXTransform,
    low_rotation: MLXTransform,
    high_projection: MLXTransform,
    low_projection: MLXTransform,
) -> mx.array:
    return _restore_split(
        dequantize_product(encoding.high, high_rotation, high_projection),
        dequantize_product(encoding.low, low_rotation, low_projection),
        encoding.split,
    )


def split_product_inner_products(
    queries: mx.array,
    encoding: MLXSplitProductEncoding,
    high_rotation: MLXTransform,
    low_rotation: MLXTransform,
    high_projection: MLXTransform,
    low_projection: MLXTransform,
) -> mx.array:
    high, low = _split_vectors(queries, encoding.split)
    return product_inner_products(
        high,
        encoding.high,
        high_rotation,
        high_projection,
    ) + product_inner_products(
        low,
        encoding.low,
        low_rotation,
        low_projection,
    )

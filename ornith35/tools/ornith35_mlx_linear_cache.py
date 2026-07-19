#!/usr/bin/env python3
"""Checked access to the Ornith-35 append-only MLX cache extension."""

from __future__ import annotations

from pathlib import Path
import sys

import mlx.core as mx

from ornith35_moe_reference import MoEError, require


EXTENSION_ROOT = Path(__file__).resolve().parents[1] / "extensions" / "kv_cache"
_APPEND_BF16 = None
_APPEND_KV_BF16 = None
_APPEND_KV_TRANSPOSED_BF16 = None
_APPEND_PACKED_MSE8 = None


def _load_append():
    global _APPEND_BF16, _APPEND_KV_BF16, _APPEND_KV_TRANSPOSED_BF16
    global _APPEND_PACKED_MSE8
    if _APPEND_BF16 is not None:
        return (
            _APPEND_BF16,
            _APPEND_KV_BF16,
            _APPEND_KV_TRANSPOSED_BF16,
            _APPEND_PACKED_MSE8,
        )
    extension_root = str(EXTENSION_ROOT)
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    try:
        from ornith35_mlx_kv_cache import (
            append_bf16,
            append_kv_bf16,
            append_kv_transposed_bf16,
            append_packed_mse8,
        )
    except ImportError as exc:
        raise MoEError(
            "Ornith-35 linear K/V extension is not built; "
            "run ornith35/build_extensions.sh"
        ) from exc
    _APPEND_BF16 = append_bf16
    _APPEND_KV_BF16 = append_kv_bf16
    _APPEND_KV_TRANSPOSED_BF16 = append_kv_transposed_bf16
    _APPEND_PACKED_MSE8 = append_packed_mse8
    return (
        _APPEND_BF16,
        _APPEND_KV_BF16,
        _APPEND_KV_TRANSPOSED_BF16,
        _APPEND_PACKED_MSE8,
    )


def append_bf16(cache: mx.array, update: mx.array, position: int) -> mx.array:
    """Alias `cache` and overwrite one contiguous, previously unused range."""
    require(isinstance(position, int), "linear cache position must be an integer")
    append, _, _, _ = _load_append()
    return append(cache, update, position)


def append_kv_bf16(
    keys: mx.array,
    values: mx.array,
    key_update: mx.array,
    value_update: mx.array,
    position: int,
) -> tuple[mx.array, mx.array]:
    """Alias paired K/V buffers and update both in one Metal dispatch."""
    require(isinstance(position, int), "linear cache position must be an integer")
    _, append_kv, _, _ = _load_append()
    return tuple(append_kv(keys, values, key_update, value_update, position))


def append_kv_transposed_bf16(
    keys: mx.array,
    values: mx.array,
    key_update: mx.array,
    value_update: mx.array,
    position: int,
) -> tuple[mx.array, mx.array]:
    """Append contiguous `[tokens, heads, width]` K/V projections directly."""
    require(isinstance(position, int), "linear cache position must be an integer")
    _, _, append_kv, _ = _load_append()
    return tuple(append_kv(keys, values, key_update, value_update, position))


def append_packed_mse8(
    packed_keys: mx.array,
    key_norms: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    packed_key_update: mx.array,
    key_norm_update: mx.array,
    packed_value_update: mx.array,
    value_norm_update: mx.array,
    position: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Alias and update paired packed K8 payloads and their BF16 norms."""
    require(isinstance(position, int), "linear cache position must be an integer")
    _, _, _, append_packed = _load_append()
    return tuple(
        append_packed(
            packed_keys,
            key_norms,
            packed_values,
            value_norms,
            packed_key_update,
            key_norm_update,
            packed_value_update,
            value_norm_update,
            position,
        )
    )

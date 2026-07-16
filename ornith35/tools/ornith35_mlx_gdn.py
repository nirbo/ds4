#!/usr/bin/env python3
"""One-token MLX GatedDeltaNet composition for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ornith35_gdn_reference import GDNConfig, require
from ornith35_nvfp4 import SafetensorsFile


PRODUCTION_CONFIG = GDNConfig(
    hidden_size=2048,
    num_k_heads=16,
    num_v_heads=32,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel_size=4,
    rms_norm_eps=1e-6,
)


@dataclass(frozen=True)
class MLXGDNWeights:
    in_proj_qkv: mx.array
    in_proj_z: mx.array
    in_proj_b: mx.array
    in_proj_a: mx.array
    conv1d: mx.array
    dt_bias: mx.array
    a_log: mx.array
    norm: mx.array
    out_proj: mx.array


@dataclass(frozen=True)
class MLXGDNState:
    conv: mx.array
    recurrent: mx.array


def validate_weights(weights: MLXGDNWeights, config: GDNConfig) -> None:
    expected = {
        "in_proj_qkv": (config.conv_dim, config.hidden_size),
        "in_proj_z": (config.value_dim, config.hidden_size),
        "in_proj_b": (config.num_v_heads, config.hidden_size),
        "in_proj_a": (config.num_v_heads, config.hidden_size),
        "conv1d": (config.conv_dim, config.conv_kernel_size),
        "dt_bias": (config.num_v_heads,),
        "a_log": (config.num_v_heads,),
        "norm": (config.head_v_dim,),
        "out_proj": (config.hidden_size, config.value_dim),
    }
    for name, shape in expected.items():
        require(getattr(weights, name).shape == shape, f"{name} shape mismatch")
    model_dtype = weights.in_proj_qkv.dtype
    require(
        model_dtype in (mx.bfloat16, mx.float32),
        "GDN model dtype must be BF16 or reference FP32",
    )
    require(
        all(array.dtype == model_dtype for array in weights.__dict__.values()),
        "GDN weight dtype mismatch",
    )


def validate_state(state: MLXGDNState, config: GDNConfig) -> None:
    require(
        state.conv.shape == (config.conv_dim, config.conv_kernel_size),
        "convolution state shape mismatch",
    )
    require(
        state.recurrent.shape
        == (config.num_v_heads, config.head_k_dim, config.head_v_dim),
        "recurrent state shape mismatch",
    )
    require(state.recurrent.dtype == mx.float32, "recurrent state must be FP32")


def zeros_state(config: GDNConfig, conv_dtype: mx.Dtype = mx.bfloat16) -> MLXGDNState:
    return MLXGDNState(
        conv=mx.zeros((config.conv_dim, config.conv_kernel_size), dtype=conv_dtype),
        recurrent=mx.zeros(
            (config.num_v_heads, config.head_k_dim, config.head_v_dim),
            dtype=mx.float32,
        ),
    )


def _linear(weight: mx.array, vector: mx.array) -> mx.array:
    return mx.matmul(weight, vector)


def _softplus(value: mx.array) -> mx.array:
    return mx.maximum(value, 0.0) + mx.log1p(mx.exp(-mx.abs(value)))


def _silu(value: mx.array) -> mx.array:
    return value * mx.sigmoid(value)


def _l2norm(value: mx.array) -> mx.array:
    value32 = value.astype(mx.float32)
    return value32 * mx.rsqrt(mx.sum(value32 * value32, axis=-1, keepdims=True) + 1e-6)


def decode_step(
    hidden: mx.array,
    state: MLXGDNState,
    weights: MLXGDNWeights,
    config: GDNConfig = PRODUCTION_CONFIG,
) -> tuple[mx.array, MLXGDNState]:
    """Append one token without mutating the caller's rollback state."""
    require(hidden.ndim == 1 and hidden.shape == (config.hidden_size,), "hidden-state shape mismatch")
    validate_state(state, config)
    validate_weights(weights, config)

    model_dtype = weights.in_proj_qkv.dtype
    require(state.conv.dtype == model_dtype, "convolution state dtype mismatch")
    hidden = hidden.astype(model_dtype)
    mixed = _linear(weights.in_proj_qkv, hidden)
    z = _linear(weights.in_proj_z, hidden)
    b = _linear(weights.in_proj_b, hidden)
    a = _linear(weights.in_proj_a, hidden)

    next_conv = mx.concatenate([state.conv[:, 1:], mixed[:, None]], axis=1)
    convolved32 = mx.sum(
        next_conv.astype(mx.float32) * weights.conv1d.astype(mx.float32),
        axis=1,
    )
    convolved = _silu(convolved32).astype(model_dtype)

    query_end = config.key_dim
    key_end = query_end + config.key_dim
    query = convolved[:query_end].reshape(config.num_k_heads, config.head_k_dim)
    key = convolved[query_end:key_end].reshape(config.num_k_heads, config.head_k_dim)
    value = convolved[key_end:].reshape(config.num_v_heads, config.head_v_dim)
    query = _l2norm(query)
    key = _l2norm(key)
    repeats = config.num_v_heads // config.num_k_heads
    if repeats > 1:
        query = mx.repeat(query, repeats, axis=0)
        key = mx.repeat(key, repeats, axis=0)

    beta = mx.sigmoid(b.astype(mx.float32))[:, None]
    decay_log = -mx.exp(weights.a_log.astype(mx.float32)) * _softplus(
        a.astype(mx.float32) + weights.dt_bias.astype(mx.float32)
    )
    decayed = state.recurrent * mx.exp(decay_log)[:, None, None]
    memory = mx.sum(decayed * key[:, :, None], axis=1)
    delta = (value.astype(mx.float32) - memory) * beta
    recurrent = decayed + key[:, :, None] * delta[:, None, :]
    query = query * (config.head_k_dim**-0.5)
    core = mx.sum(recurrent * query[:, :, None], axis=1)

    variance = mx.mean(core * core, axis=-1, keepdims=True)
    normalized = core * mx.rsqrt(variance + config.rms_norm_eps)
    weighted = (
        normalized.astype(model_dtype) * weights.norm.astype(model_dtype)
    ).astype(model_dtype)
    gated = (weighted.astype(mx.float32) * _silu(z.astype(mx.float32).reshape(
        config.num_v_heads, config.head_v_dim
    ))).astype(model_dtype)
    output = _linear(weights.out_proj, gated.reshape(config.value_dim))
    return output, MLXGDNState(conv=next_conv, recurrent=recurrent)


def _load_bf16(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "BF16", f"expected BF16 tensor: {name}")
    require(entry.get("shape") == list(shape), f"tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    expected_bytes = 2
    for size in shape:
        expected_bytes *= size
    require(len(payload) == expected_bytes, f"tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.bfloat16).reshape(shape)


def load_layer(source_path: Path, layer: int) -> MLXGDNWeights:
    require(layer >= 0 and layer % 4 != 3 and layer < 40, "layer is not an Ornith GDN layer")
    prefix = f"model.language_model.layers.{layer}.linear_attn"
    with SafetensorsFile(source_path) as source:
        weights = MLXGDNWeights(
            in_proj_qkv=_load_bf16(source, f"{prefix}.in_proj_qkv.weight", (8192, 2048)),
            in_proj_z=_load_bf16(source, f"{prefix}.in_proj_z.weight", (4096, 2048)),
            in_proj_b=_load_bf16(source, f"{prefix}.in_proj_b.weight", (32, 2048)),
            in_proj_a=_load_bf16(source, f"{prefix}.in_proj_a.weight", (32, 2048)),
            conv1d=_load_bf16(source, f"{prefix}.conv1d.weight", (8192, 1, 4)).reshape(8192, 4),
            dt_bias=_load_bf16(source, f"{prefix}.dt_bias", (32,)),
            a_log=_load_bf16(source, f"{prefix}.A_log", (32,)),
            norm=_load_bf16(source, f"{prefix}.norm.weight", (128,)),
            out_proj=_load_bf16(source, f"{prefix}.out_proj.weight", (2048, 4096)),
        )
        mx.eval(*weights.__dict__.values())
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights

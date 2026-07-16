#!/usr/bin/env python3
"""One-token MLX full-attention composition for text-only Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ornith35_attention_reference import AttentionConfig, require
from ornith35_nvfp4 import SafetensorsFile


PRODUCTION_CONFIG = AttentionConfig(
    hidden_size=2048,
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=256,
    rotary_dim=64,
    rope_theta=10_000_000.0,
    rms_norm_eps=1e-6,
)


@dataclass(frozen=True)
class MLXAttentionWeights:
    q_proj: mx.array
    k_proj: mx.array
    v_proj: mx.array
    o_proj: mx.array
    q_norm: mx.array
    k_norm: mx.array


@dataclass(frozen=True)
class MLXAttentionState:
    keys: mx.array
    values: mx.array


def validate_weights(weights: MLXAttentionWeights, config: AttentionConfig) -> None:
    expected = {
        "q_proj": (config.query_dim * 2, config.hidden_size),
        "k_proj": (config.kv_dim, config.hidden_size),
        "v_proj": (config.kv_dim, config.hidden_size),
        "o_proj": (config.hidden_size, config.query_dim),
        "q_norm": (config.head_dim,),
        "k_norm": (config.head_dim,),
    }
    for name, shape in expected.items():
        require(getattr(weights, name).shape == shape, f"{name} shape mismatch")
    model_dtype = weights.q_proj.dtype
    require(
        model_dtype in (mx.bfloat16, mx.float32),
        "attention model dtype must be BF16 or reference FP32",
    )
    require(
        all(array.dtype == model_dtype for array in weights.__dict__.values()),
        "attention weight dtype mismatch",
    )


def state_length(state: MLXAttentionState, config: AttentionConfig) -> int:
    require(state.keys.ndim == 3, "key state rank mismatch")
    require(state.values.ndim == 3, "value state rank mismatch")
    require(state.keys.shape[0] == config.num_kv_heads, "key state head mismatch")
    require(state.values.shape[0] == config.num_kv_heads, "value state head mismatch")
    require(state.keys.shape[2] == config.head_dim, "key state width mismatch")
    require(state.values.shape[2] == config.head_dim, "value state width mismatch")
    require(state.keys.shape[1] == state.values.shape[1], "KV state length mismatch")
    require(state.keys.dtype == state.values.dtype, "KV state dtype mismatch")
    return state.keys.shape[1]


def zeros_state(
    config: AttentionConfig,
    dtype: mx.Dtype = mx.bfloat16,
) -> MLXAttentionState:
    shape = (config.num_kv_heads, 0, config.head_dim)
    return MLXAttentionState(keys=mx.zeros(shape, dtype=dtype), values=mx.zeros(shape, dtype=dtype))


def _rms_norm(value: mx.array, weight: mx.array, eps: float, dtype: mx.Dtype) -> mx.array:
    value32 = value.astype(mx.float32)
    normalized = value32 * mx.rsqrt(mx.mean(value32 * value32, axis=-1, keepdims=True) + eps)
    return (normalized * (1.0 + weight.astype(mx.float32))).astype(dtype)


def _apply_text_rope(
    value: mx.array,
    position: int,
    config: AttentionConfig,
    dtype: mx.Dtype,
) -> mx.array:
    half = config.rotary_dim // 2
    indices = mx.arange(0, config.rotary_dim, 2, dtype=mx.float32)
    inverse_frequencies = mx.power(config.rope_theta, -indices / config.rotary_dim)
    frequencies = inverse_frequencies * position
    angles = mx.concatenate([frequencies, frequencies])
    cosine = mx.cos(angles).astype(dtype)
    sine = mx.sin(angles).astype(dtype)
    rotary = value[:, : config.rotary_dim]
    rotated = mx.concatenate([-rotary[:, half:], rotary[:, :half]], axis=-1)
    embedded = rotary * cosine + rotated * sine
    return mx.concatenate([embedded, value[:, config.rotary_dim :]], axis=-1)


def decode_step(
    hidden: mx.array,
    state: MLXAttentionState,
    weights: MLXAttentionWeights,
    config: AttentionConfig = PRODUCTION_CONFIG,
) -> tuple[mx.array, MLXAttentionState]:
    """Append one causal text token without mutating the rollback state."""
    require(hidden.ndim == 1 and hidden.shape == (config.hidden_size,), "hidden-state shape mismatch")
    position = state_length(state, config)
    validate_weights(weights, config)
    model_dtype = weights.q_proj.dtype
    require(state.keys.dtype == model_dtype, "KV state dtype mismatch")
    hidden = hidden.astype(model_dtype)

    query_gate = mx.matmul(weights.q_proj, hidden).reshape(
        config.num_q_heads, config.head_dim * 2
    )
    query = query_gate[:, : config.head_dim]
    gate = query_gate[:, config.head_dim :]
    key = mx.matmul(weights.k_proj, hidden).reshape(config.num_kv_heads, config.head_dim)
    value = mx.matmul(weights.v_proj, hidden).reshape(config.num_kv_heads, config.head_dim)
    query = _rms_norm(query, weights.q_norm, config.rms_norm_eps, model_dtype)
    key = _rms_norm(key, weights.k_norm, config.rms_norm_eps, model_dtype)
    query = _apply_text_rope(query, position, config, model_dtype)
    key = _apply_text_rope(key, position, config, model_dtype)

    next_keys = mx.concatenate([state.keys, key[:, None, :]], axis=1)
    next_values = mx.concatenate([state.values, value[:, None, :]], axis=1)
    groups = config.num_q_heads // config.num_kv_heads
    repeated_keys = mx.repeat(next_keys, groups, axis=0)
    repeated_values = mx.repeat(next_values, groups, axis=0)
    scores = mx.matmul(query[:, None, :], mx.swapaxes(repeated_keys, 1, 2)).reshape(
        config.num_q_heads, position + 1
    )
    scores = scores * (config.head_dim**-0.5)
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(model_dtype)
    attended = mx.matmul(probabilities[:, None, :], repeated_values).reshape(
        config.num_q_heads, config.head_dim
    )
    attended = attended * mx.sigmoid(gate)
    output = mx.matmul(weights.o_proj, attended.reshape(config.query_dim))
    return output, MLXAttentionState(keys=next_keys, values=next_values)


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


def load_layer(source_path: Path, layer: int) -> MLXAttentionWeights:
    require(layer >= 0 and layer % 4 == 3 and layer < 40, "layer is not an Ornith attention layer")
    prefix = f"model.language_model.layers.{layer}.self_attn"
    with SafetensorsFile(source_path) as source:
        weights = MLXAttentionWeights(
            q_proj=_load_bf16(source, f"{prefix}.q_proj.weight", (8192, 2048)),
            k_proj=_load_bf16(source, f"{prefix}.k_proj.weight", (512, 2048)),
            v_proj=_load_bf16(source, f"{prefix}.v_proj.weight", (512, 2048)),
            o_proj=_load_bf16(source, f"{prefix}.o_proj.weight", (2048, 4096)),
            q_norm=_load_bf16(source, f"{prefix}.q_norm.weight", (256,)),
            k_norm=_load_bf16(source, f"{prefix}.k_norm.weight", (256,)),
        )
        mx.eval(*weights.__dict__.values())
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights

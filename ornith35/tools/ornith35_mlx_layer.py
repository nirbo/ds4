#!/usr/bin/env python3
"""Complete one-token decoder-layer composition for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_moe as moe
from ornith35_moe_reference import require
from ornith35_nvfp4 import SafetensorsFile


@dataclass(frozen=True)
class LayerNorms:
    input_layernorm: mx.array
    post_attention_layernorm: mx.array


@dataclass(frozen=True)
class GDNLayerWeights:
    token_mixer: gdn.MLXGDNWeights
    moe: moe.MLXMoEWeights
    norms: LayerNorms


@dataclass(frozen=True)
class AttentionLayerWeights:
    token_mixer: attention.MLXAttentionWeights
    moe: moe.MLXMoEWeights
    norms: LayerNorms


@dataclass(frozen=True)
class LayerResult:
    output: mx.array
    state: gdn.MLXGDNState | attention.MLXAttentionState
    selected_experts: mx.array
    routing_weights: mx.array


def qwen_rms_norm(
    hidden: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Apply Qwen3.5's centered `(1 + weight)` RMSNorm."""
    require(hidden.ndim == 1 and weight.shape == hidden.shape, "RMSNorm shape mismatch")
    require(hidden.dtype == weight.dtype, "RMSNorm dtype mismatch")
    hidden32 = hidden.astype(mx.float32)
    normalized = hidden32 * mx.rsqrt(mx.mean(hidden32 * hidden32) + eps)
    return (normalized * (1.0 + weight.astype(mx.float32))).astype(hidden.dtype)


def _validate_norms(norms: LayerNorms, hidden_size: int, dtype: mx.Dtype) -> None:
    require(norms.input_layernorm.shape == (hidden_size,), "input RMSNorm shape mismatch")
    require(
        norms.post_attention_layernorm.shape == (hidden_size,),
        "post-attention RMSNorm shape mismatch",
    )
    require(norms.input_layernorm.dtype == dtype, "input RMSNorm dtype mismatch")
    require(
        norms.post_attention_layernorm.dtype == dtype,
        "post-attention RMSNorm dtype mismatch",
    )


def forward_gdn(
    hidden: mx.array,
    state: gdn.MLXGDNState,
    weights: GDNLayerWeights,
    gdn_config: gdn.GDNConfig = gdn.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
) -> LayerResult:
    require(gdn_config.hidden_size == moe_config.hidden_size, "layer hidden-size mismatch")
    dtype = weights.token_mixer.in_proj_qkv.dtype
    require(weights.moe.router.dtype == dtype, "GDN/MoE dtype mismatch")
    _validate_norms(weights.norms, gdn_config.hidden_size, dtype)
    hidden = hidden.astype(dtype)
    mixed_input = qwen_rms_norm(hidden, weights.norms.input_layernorm, gdn_config.rms_norm_eps)
    mixed, next_state = gdn.decode_step(mixed_input, state, weights.token_mixer, gdn_config)
    hidden = (hidden + mixed).astype(dtype)
    moe_input = qwen_rms_norm(hidden, weights.norms.post_attention_layernorm, gdn_config.rms_norm_eps)
    moe_result = moe.forward(moe_input, weights.moe, moe_config)
    return LayerResult(
        output=(hidden + moe_result.output).astype(dtype),
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
    )


def forward_attention(
    hidden: mx.array,
    state: attention.MLXAttentionState,
    weights: AttentionLayerWeights,
    attention_config: attention.AttentionConfig = attention.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
) -> LayerResult:
    require(
        attention_config.hidden_size == moe_config.hidden_size,
        "layer hidden-size mismatch",
    )
    dtype = weights.token_mixer.q_proj.dtype
    require(weights.moe.router.dtype == dtype, "attention/MoE dtype mismatch")
    _validate_norms(weights.norms, attention_config.hidden_size, dtype)
    hidden = hidden.astype(dtype)
    mixed_input = qwen_rms_norm(
        hidden,
        weights.norms.input_layernorm,
        attention_config.rms_norm_eps,
    )
    mixed, next_state = attention.decode_step(
        mixed_input,
        state,
        weights.token_mixer,
        attention_config,
    )
    hidden = (hidden + mixed).astype(dtype)
    moe_input = qwen_rms_norm(
        hidden,
        weights.norms.post_attention_layernorm,
        attention_config.rms_norm_eps,
    )
    moe_result = moe.forward(moe_input, weights.moe, moe_config)
    return LayerResult(
        output=(hidden + moe_result.output).astype(dtype),
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
    )


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


def _load_norms(source_path: Path, layer: int) -> LayerNorms:
    prefix = f"model.language_model.layers.{layer}"
    with SafetensorsFile(source_path) as source:
        norms = LayerNorms(
            input_layernorm=_load_bf16(source, f"{prefix}.input_layernorm.weight", (2048,)),
            post_attention_layernorm=_load_bf16(
                source,
                f"{prefix}.post_attention_layernorm.weight",
                (2048,),
            ),
        )
        mx.eval(norms.input_layernorm, norms.post_attention_layernorm)
    return norms


def load_layer(source_path: Path, layer: int) -> GDNLayerWeights | AttentionLayerWeights:
    require(0 <= layer < 40, "layer is outside the Ornith text model")
    norms = _load_norms(source_path, layer)
    moe_weights = moe.load_layer(source_path, layer)
    if layer % 4 == 3:
        return AttentionLayerWeights(
            token_mixer=attention.load_layer(source_path, layer),
            moe=moe_weights,
            norms=norms,
        )
    return GDNLayerWeights(
        token_mixer=gdn.load_layer(source_path, layer),
        moe=moe_weights,
        norms=norms,
    )

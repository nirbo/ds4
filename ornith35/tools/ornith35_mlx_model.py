#!/usr/bin/env python3
"""Text-only resident one-token model boundary for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_moe as moe
from ornith35_moe_reference import require
from ornith35_nvfp4 import SafetensorsFile, require_verified_source


LAYER_GDN = "gdn"
LAYER_ATTENTION = "attention"


@dataclass(frozen=True)
class TextModelConfig:
    vocab_size: int
    hidden_size: int
    layer_types: tuple[str, ...]
    gdn: gdn.GDNConfig
    attention: attention.AttentionConfig
    moe: moe.MoEConfig
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        require(self.vocab_size > 0 and self.hidden_size > 0, "model dimensions must be positive")
        require(len(self.layer_types) > 0, "model must contain decoder layers")
        require(
            all(kind in (LAYER_GDN, LAYER_ATTENTION) for kind in self.layer_types),
            "invalid decoder-layer type",
        )
        require(
            self.gdn.hidden_size == self.hidden_size
            and self.attention.hidden_size == self.hidden_size
            and self.moe.hidden_size == self.hidden_size,
            "model component hidden-size mismatch",
        )
        require(self.rms_norm_eps > 0.0, "model RMS epsilon must be positive")


PRODUCTION_CONFIG = TextModelConfig(
    vocab_size=248_320,
    hidden_size=2048,
    layer_types=tuple(
        LAYER_ATTENTION if index % 4 == 3 else LAYER_GDN
        for index in range(40)
    ),
    gdn=gdn.PRODUCTION_CONFIG,
    attention=attention.PRODUCTION_CONFIG,
    moe=moe.PRODUCTION_CONFIG,
    rms_norm_eps=1e-6,
)


LayerWeights = layer.GDNLayerWeights | layer.AttentionLayerWeights
LayerState = gdn.MLXGDNState | attention.MLXAttentionState


@dataclass(frozen=True)
class TextModelWeights:
    embedding: mx.array
    layers: tuple[LayerWeights, ...]
    final_norm: mx.array
    lm_head: mx.array


@dataclass(frozen=True)
class TextModelState:
    position: int
    layers: tuple[LayerState, ...]


@dataclass(frozen=True)
class TextModelResult:
    logits: mx.array
    hidden: mx.array
    state: TextModelState
    selected_experts: tuple[mx.array, ...]
    routing_weights: tuple[mx.array, ...]


def validate_weights(weights: TextModelWeights, config: TextModelConfig) -> None:
    require(
        weights.embedding.shape == (config.vocab_size, config.hidden_size),
        "embedding shape mismatch",
    )
    require(weights.lm_head.shape == (config.vocab_size, config.hidden_size), "LM-head shape mismatch")
    require(weights.final_norm.shape == (config.hidden_size,), "final RMSNorm shape mismatch")
    require(len(weights.layers) == len(config.layer_types), "decoder-layer count mismatch")
    dtype = weights.embedding.dtype
    require(dtype in (mx.bfloat16, mx.float32), "invalid text-model dtype")
    require(weights.lm_head.dtype == dtype, "LM-head dtype mismatch")
    require(weights.final_norm.dtype == dtype, "final RMSNorm dtype mismatch")
    for index, (kind, layer_weights) in enumerate(zip(config.layer_types, weights.layers)):
        expected = layer.GDNLayerWeights if kind == LAYER_GDN else layer.AttentionLayerWeights
        require(isinstance(layer_weights, expected), f"decoder-layer type mismatch at {index}")
        token_dtype = (
            layer_weights.token_mixer.in_proj_qkv.dtype
            if kind == LAYER_GDN
            else layer_weights.token_mixer.q_proj.dtype
        )
        require(token_dtype == dtype, f"decoder-layer dtype mismatch at {index}")


def initial_state(weights: TextModelWeights, config: TextModelConfig) -> TextModelState:
    validate_weights(weights, config)
    dtype = weights.embedding.dtype
    states = tuple(
        gdn.zeros_state(config.gdn, conv_dtype=dtype)
        if kind == LAYER_GDN
        else attention.zeros_state(config.attention, dtype=dtype)
        for kind in config.layer_types
    )
    return TextModelState(position=0, layers=states)


def validate_state(state: TextModelState, config: TextModelConfig) -> None:
    require(state.position >= 0, "model position must be nonnegative")
    require(len(state.layers) == len(config.layer_types), "model-state layer count mismatch")
    for index, (kind, layer_state) in enumerate(zip(config.layer_types, state.layers)):
        if kind == LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            gdn.validate_state(layer_state, config.gdn)
        else:
            require(
                isinstance(layer_state, attention.MLXAttentionState),
                f"attention state mismatch at {index}",
            )
            length = attention.state_length(layer_state, config.attention)
            require(length == state.position, f"attention position mismatch at {index}")


def forward_token(
    token_id: int,
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
) -> TextModelResult:
    """Evaluate one token and return full-vocabulary target logits lazily."""
    require(isinstance(token_id, int) and 0 <= token_id < config.vocab_size, "token ID is out of range")
    validate_weights(weights, config)
    validate_state(state, config)
    hidden = weights.embedding[token_id]
    next_states = []
    selected_experts = []
    routing_weights = []
    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(config.layer_types, weights.layers, state.layers)
    ):
        if kind == LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), f"GDN weights mismatch at {index}")
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            result = layer.forward_gdn(
                hidden,
                layer_state,
                layer_weights,
                config.gdn,
                config.moe,
            )
        else:
            require(
                isinstance(layer_weights, layer.AttentionLayerWeights),
                f"attention weights mismatch at {index}",
            )
            require(
                isinstance(layer_state, attention.MLXAttentionState),
                f"attention state mismatch at {index}",
            )
            result = layer.forward_attention(
                hidden,
                layer_state,
                layer_weights,
                config.attention,
                config.moe,
            )
        hidden = result.output
        next_states.append(result.state)
        selected_experts.append(result.selected_experts)
        routing_weights.append(result.routing_weights)

    hidden = layer.qwen_rms_norm(hidden, weights.final_norm, config.rms_norm_eps)
    logits = mx.matmul(weights.lm_head, hidden)
    return TextModelResult(
        logits=logits,
        hidden=hidden,
        state=TextModelState(position=state.position + 1, layers=tuple(next_states)),
        selected_experts=tuple(selected_experts),
        routing_weights=tuple(routing_weights),
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


def load_text_model(root: Path) -> TextModelWeights:
    """Load only explicitly cataloged text tensors from a verified source."""
    source_path = require_verified_source(root)
    with SafetensorsFile(source_path) as source:
        embedding = _load_bf16(
            source,
            "model.language_model.embed_tokens.weight",
            (248_320, 2048),
        )
        final_norm = _load_bf16(source, "model.language_model.norm.weight", (2048,))
        lm_head = _load_bf16(source, "lm_head.weight", (248_320, 2048))
        mx.eval(embedding, final_norm, lm_head)
    layers = tuple(layer.load_layer(source_path, index) for index in range(40))
    weights = TextModelWeights(
        embedding=embedding,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights

#!/usr/bin/env python3
"""Text-only resident one-token model boundary for Ornith-35."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import statistics
import sys
from threading import Lock
import time
from weakref import ReferenceType, ref

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_moe as moe
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT, SafetensorsFile, require_verified_source


LAYER_GDN = "gdn"
LAYER_ATTENTION = "attention"
_DECODE_SESSION_SEAL = object()
_LINEAR_DECODE_SESSION_SEAL = object()


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
LayerState = (
    gdn.MLXGDNState
    | attention.MLXAttentionState
    | attention.MLXLinearAttentionState
)


@dataclass(frozen=True)
class TextModelWeights:
    embedding: mx.array | vocab.MLXAffineQuantizedMatrix | vocab.MLXMappedBF16Matrix
    layers: tuple[LayerWeights, ...]
    final_norm: mx.array
    lm_head: mx.array | vocab.MLXAffineQuantizedMatrix


@dataclass(frozen=True)
class TextModelState:
    position: int
    layers: tuple[LayerState, ...]


@dataclass(frozen=True)
class TextDecodeSession:
    """Weights and rollback state accepted once for unchecked nested decode."""

    weights: TextModelWeights
    state: TextModelState
    config: TextModelConfig
    _seal: object = field(repr=False, compare=False)


@dataclass
class _LinearDecodeOwner:
    session: ReferenceType[object] | None = None
    lock: Lock = field(default_factory=Lock, repr=False)


@dataclass
class TextLinearDecodeSession:
    """Single-owner, append-only decode state with no rollback contract."""

    weights: TextModelWeights
    state: TextModelState
    config: TextModelConfig
    capacity: int
    _owner: _LinearDecodeOwner = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class TextModelTransition:
    hidden: mx.array
    state: TextModelState
    selected_experts: tuple[mx.array, ...]
    routing_weights: tuple[mx.array, ...]


@dataclass(frozen=True)
class TextModelResult(TextModelTransition):
    logits: mx.array


@dataclass(frozen=True)
class TextModelChunkTransition:
    hidden: mx.array
    state: TextModelState
    selected_experts: tuple[mx.array, ...]
    routing_weights: tuple[mx.array, ...]


@dataclass(frozen=True)
class TextModelChunkResult(TextModelChunkTransition):
    logits: mx.array


def _matrix_shape(
    matrix: mx.array | vocab.MLXAffineQuantizedMatrix | vocab.MLXMappedBF16Matrix,
) -> tuple[int, ...]:
    return matrix.shape


def matrix_dtype(
    matrix: mx.array | vocab.MLXAffineQuantizedMatrix | vocab.MLXMappedBF16Matrix,
) -> mx.Dtype:
    if isinstance(matrix, vocab.MLXAffineQuantizedMatrix):
        vocab.validate(matrix)
        return matrix.scales.dtype
    if isinstance(matrix, vocab.MLXMappedBF16Matrix):
        return matrix.dtype
    return matrix.dtype


def embed_token(
    embedding: mx.array | vocab.MLXAffineQuantizedMatrix | vocab.MLXMappedBF16Matrix,
    token_id: int,
) -> mx.array:
    if isinstance(embedding, vocab.MLXAffineQuantizedMatrix):
        return vocab.dequantize_rows(embedding, token_id)
    if isinstance(embedding, vocab.MLXMappedBF16Matrix):
        return embedding.row(token_id)
    return embedding[token_id]


def embed_tokens(
    embedding: mx.array | vocab.MLXAffineQuantizedMatrix | vocab.MLXMappedBF16Matrix,
    token_ids: Sequence[int],
) -> mx.array:
    if isinstance(embedding, vocab.MLXMappedBF16Matrix):
        return embedding.rows(token_ids)
    indices = mx.array(token_ids, dtype=mx.uint32)
    if isinstance(embedding, vocab.MLXAffineQuantizedMatrix):
        return vocab.dequantize_rows(embedding, indices)
    return mx.take(embedding, indices, axis=0)


def validate_weights(weights: TextModelWeights, config: TextModelConfig) -> None:
    require(
        _matrix_shape(weights.embedding) == (config.vocab_size, config.hidden_size),
        "embedding shape mismatch",
    )
    if isinstance(weights.lm_head, vocab.MLXAffineQuantizedMatrix):
        vocab.validate(weights.lm_head)
        lm_head_shape = weights.lm_head.shape
        lm_head_dtype = weights.lm_head.scales.dtype
    else:
        require(isinstance(weights.lm_head, mx.array), "invalid LM-head weights")
        lm_head_shape = weights.lm_head.shape
        lm_head_dtype = weights.lm_head.dtype
    require(lm_head_shape == (config.vocab_size, config.hidden_size), "LM-head shape mismatch")
    require(weights.final_norm.shape == (config.hidden_size,), "final RMSNorm shape mismatch")
    require(len(weights.layers) == len(config.layer_types), "decoder-layer count mismatch")
    dtype = matrix_dtype(weights.embedding)
    require(dtype in (mx.bfloat16, mx.float32), "invalid text-model dtype")
    require(lm_head_dtype == dtype, "LM-head dtype mismatch")
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


def project_lm_head(
    lm_head: mx.array | vocab.MLXAffineQuantizedMatrix,
    hidden: mx.array,
) -> mx.array:
    """Apply the authoritative BF16 or explicitly quantized vocabulary head."""
    if isinstance(lm_head, vocab.MLXAffineQuantizedMatrix):
        return vocab.project(lm_head, hidden)
    return mx.matmul(lm_head, hidden)


def initial_state(weights: TextModelWeights, config: TextModelConfig) -> TextModelState:
    validate_weights(weights, config)
    dtype = matrix_dtype(weights.embedding)
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


def _validate_decode_session(
    weights: TextModelWeights,
    state: TextModelState,
    config: TextModelConfig,
) -> None:
    validate_weights(weights, config)
    validate_state(state, config)
    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(config.layer_types, weights.layers, state.layers)
    ):
        if kind == LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), f"GDN weights mismatch at {index}")
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            gdn.validate_weights(layer_weights.token_mixer, config.gdn)
            require(
                layer_state.conv.dtype == layer_weights.token_mixer.in_proj_qkv.dtype,
                f"GDN state dtype mismatch at {index}",
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
            attention.validate_weights(layer_weights.token_mixer, config.attention)
            require(
                layer_state.keys.dtype == layer_weights.token_mixer.q_proj.dtype,
                f"attention state dtype mismatch at {index}",
            )
        layer._validate_norms(layer_weights.norms, config.hidden_size, matrix_dtype(weights.embedding))
        moe.validate_weights(layer_weights.moe, config.moe)


def start_decode_session(
    weights: TextModelWeights,
    state: TextModelState,
    config: TextModelConfig = PRODUCTION_CONFIG,
) -> TextDecodeSession:
    """Deeply validate immutable decode inputs and bind them into a session."""
    _validate_decode_session(weights, state, config)
    return TextDecodeSession(
        weights=weights,
        state=state,
        config=config,
        _seal=_DECODE_SESSION_SEAL,
    )


def start_linear_decode_session(
    weights: TextModelWeights,
    state: TextModelState,
    capacity: int,
    config: TextModelConfig = PRODUCTION_CONFIG,
) -> TextLinearDecodeSession:
    """Move an immutable prefix into fixed-capacity, single-owner K/V buffers."""
    _validate_decode_session(weights, state, config)
    require(matrix_dtype(weights.embedding) == mx.bfloat16, "linear decode requires BF16 weights")
    require(capacity >= state.position, "linear decode capacity is shorter than the prefix")
    next_states: list[LayerState] = []
    arrays: list[mx.array] = []
    for kind, layer_state in zip(config.layer_types, state.layers):
        if kind == LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), "invalid GDN state")
            next_states.append(layer_state)
        else:
            require(
                isinstance(layer_state, attention.MLXAttentionState),
                "linear decode source must have immutable attention state",
            )
            linear_state = attention.linearize_state(
                layer_state,
                capacity,
                config.attention,
            )
            next_states.append(linear_state)
            arrays.extend((linear_state.keys, linear_state.values))
    if arrays:
        mx.eval(*arrays)
        mx.synchronize()
    owner = _LinearDecodeOwner()
    session = TextLinearDecodeSession(
        weights=weights,
        state=TextModelState(position=state.position, layers=tuple(next_states)),
        config=config,
        capacity=capacity,
        _owner=owner,
        _seal=_LINEAR_DECODE_SESSION_SEAL,
    )
    owner.session = ref(session)
    return session


def _forward_hidden_token(
    token_id: int,
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
    _validated: bool = False,
) -> TextModelTransition:
    """Evaluate one token through the final norm without projecting logits."""
    require(isinstance(token_id, int) and 0 <= token_id < config.vocab_size, "token ID is out of range")
    if not _validated:
        validate_weights(weights, config)
        validate_state(state, config)
    hidden = embed_token(weights.embedding, token_id)
    normalized_input = None
    next_states = []
    selected_experts = []
    routing_weights = []
    attention_rope = (
        attention.make_text_rope(
            state.position,
            1,
            config.attention,
            matrix_dtype(weights.embedding),
        )
        if fused_attention_qk_norm_rope
        else None
    )
    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(config.layer_types, weights.layers, state.layers)
    ):
        next_input_norm = (
            weights.layers[index + 1].norms.input_layernorm
            if index + 1 < len(weights.layers)
            else weights.final_norm
        )
        if kind == LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), f"GDN weights mismatch at {index}")
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            result = layer.forward_gdn(
                hidden,
                layer_state,
                layer_weights,
                config.gdn,
                config.moe,
                normalized_input=normalized_input,
                next_input_norm=next_input_norm,
                fused_residual_mean_square=fused_residual_mean_square,
                fused_residual_rmsnorm=fused_residual_rmsnorm,
                fused_gdn_convolution=fused_gdn_convolution,
                fused_gdn_recurrence=fused_gdn_recurrence,
                fused_gdn_core_gate=fused_gdn_core_gate,
                paired_moe_gate_up=paired_moe_gate_up,
                fused_moe_shared_gate=fused_moe_shared_gate,
                fused_moe_routed_down=fused_moe_routed_down,
                _validated=_validated,
            )
        else:
            require(
                isinstance(layer_weights, layer.AttentionLayerWeights),
                f"attention weights mismatch at {index}",
            )
            require(
                isinstance(
                    layer_state,
                    (attention.MLXAttentionState, attention.MLXLinearAttentionState),
                ),
                f"attention state mismatch at {index}",
            )
            result = layer.forward_attention(
                hidden,
                layer_state,
                layer_weights,
                config.attention,
                config.moe,
                normalized_input=normalized_input,
                next_input_norm=next_input_norm,
                attention_rope=attention_rope,
                fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
                grouped_attention_gqa=grouped_attention_gqa,
                fused_residual_mean_square=fused_residual_mean_square,
                fused_residual_rmsnorm=fused_residual_rmsnorm,
                paired_moe_gate_up=paired_moe_gate_up,
                fused_moe_shared_gate=fused_moe_shared_gate,
                fused_moe_routed_down=fused_moe_routed_down,
                _validated=_validated,
            )
        hidden = result.output
        normalized_input = result.normalized_output
        next_states.append(result.state)
        selected_experts.append(result.selected_experts)
        routing_weights.append(result.routing_weights)

    require(normalized_input is not None, "final normalized output is missing")
    hidden = normalized_input
    return TextModelTransition(
        hidden=hidden,
        state=TextModelState(position=state.position + 1, layers=tuple(next_states)),
        selected_experts=tuple(selected_experts),
        routing_weights=tuple(routing_weights),
    )


def forward_hidden_token(
    token_id: int,
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
) -> TextModelTransition:
    """Evaluate one checked token transition without projecting logits."""
    return _forward_hidden_token(
        token_id,
        state,
        weights,
        config,
        fused_residual_mean_square=fused_residual_mean_square,
        fused_residual_rmsnorm=fused_residual_rmsnorm,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_shared_gate=fused_moe_shared_gate,
        fused_moe_routed_down=fused_moe_routed_down,
    )


def forward_token(
    token_id: int,
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
) -> TextModelResult:
    """Evaluate one token and return full-vocabulary target logits lazily."""
    transition = forward_hidden_token(
        token_id,
        state,
        weights,
        config,
        fused_residual_mean_square=fused_residual_mean_square,
        fused_residual_rmsnorm=fused_residual_rmsnorm,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_shared_gate=fused_moe_shared_gate,
        fused_moe_routed_down=fused_moe_routed_down,
    )
    return TextModelResult(
        hidden=transition.hidden,
        state=transition.state,
        selected_experts=transition.selected_experts,
        routing_weights=transition.routing_weights,
        logits=project_lm_head(weights.lm_head, transition.hidden),
    )


def forward_session_token(
    token_id: int,
    session: TextDecodeSession,
    *,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
) -> tuple[TextModelResult, TextDecodeSession]:
    """Advance a deeply validated immutable decode session by one token."""
    require(
        isinstance(session, TextDecodeSession)
        and session._seal is _DECODE_SESSION_SEAL,
        "invalid decode session",
    )
    transition = _forward_hidden_token(
        token_id,
        session.state,
        session.weights,
        session.config,
        fused_residual_mean_square=fused_residual_mean_square,
        fused_residual_rmsnorm=fused_residual_rmsnorm,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_shared_gate=fused_moe_shared_gate,
        fused_moe_routed_down=fused_moe_routed_down,
        _validated=True,
    )
    result = TextModelResult(
        hidden=transition.hidden,
        state=transition.state,
        selected_experts=transition.selected_experts,
        routing_weights=transition.routing_weights,
        logits=project_lm_head(session.weights.lm_head, transition.hidden),
    )
    return result, TextDecodeSession(
        weights=session.weights,
        state=result.state,
        config=session.config,
        _seal=_DECODE_SESSION_SEAL,
    )


def _require_linear_session(session: TextLinearDecodeSession) -> None:
    require(
        isinstance(session, TextLinearDecodeSession)
        and session._seal is _LINEAR_DECODE_SESSION_SEAL
        and session._owner.session is not None
        and session._owner.session() is session,
        "invalid linear decode session",
    )


def _forward_linear_session_token(
    token_id: int,
    session: TextLinearDecodeSession,
    *,
    project_logits: bool,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
) -> TextModelTransition | TextModelResult:
    _require_linear_session(session)
    with session._owner.lock:
        require(session.state.position < session.capacity, "linear decode capacity exhausted")
        transition = _forward_hidden_token(
            token_id,
            session.state,
            session.weights,
            session.config,
            fused_residual_mean_square=fused_residual_mean_square,
            fused_residual_rmsnorm=fused_residual_rmsnorm,
            fused_gdn_convolution=fused_gdn_convolution,
            fused_gdn_recurrence=fused_gdn_recurrence,
            fused_gdn_core_gate=fused_gdn_core_gate,
            fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
            grouped_attention_gqa=grouped_attention_gqa,
            paired_moe_gate_up=paired_moe_gate_up,
            fused_moe_shared_gate=fused_moe_shared_gate,
            fused_moe_routed_down=fused_moe_routed_down,
            _validated=True,
        )
        if project_logits:
            result = TextModelResult(
                hidden=transition.hidden,
                state=transition.state,
                selected_experts=transition.selected_experts,
                routing_weights=transition.routing_weights,
                logits=project_lm_head(session.weights.lm_head, transition.hidden),
            )
            evaluate_result(result)
        else:
            result = transition
            evaluate_transition(result)
        session.state = result.state
        return result


def forward_linear_session_token(
    token_id: int,
    session: TextLinearDecodeSession,
    *,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
) -> TextModelResult:
    """Advance and commit one token to a single-owner linear decode session."""
    result = _forward_linear_session_token(
        token_id,
        session,
        project_logits=True,
        fused_residual_mean_square=fused_residual_mean_square,
        fused_residual_rmsnorm=fused_residual_rmsnorm,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_shared_gate=fused_moe_shared_gate,
        fused_moe_routed_down=fused_moe_routed_down,
    )
    require(isinstance(result, TextModelResult), "linear decode result mismatch")
    return result


def forward_linear_session_hidden_token(
    token_id: int,
    session: TextLinearDecodeSession,
) -> TextModelTransition:
    """Advance one prefill-tail token without projecting unused logits."""
    result = _forward_linear_session_token(
        token_id,
        session,
        project_logits=False,
    )
    require(type(result) is TextModelTransition, "linear hidden result mismatch")
    return result


def prefill_hidden_chunk(
    token_ids: Sequence[int],
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    _validated: bool = False,
) -> TextModelChunkTransition:
    """Evaluate a nonempty prompt chunk through the final centered norm."""
    tokens = tuple(token_ids)
    require(tokens, "prefill chunk must contain at least one token")
    require(
        all(isinstance(token, int) and 0 <= token < config.vocab_size for token in tokens),
        "prefill chunk token ID is out of range",
    )
    if not _validated:
        validate_weights(weights, config)
        validate_state(state, config)
    hidden = embed_tokens(weights.embedding, tokens)
    normalized_input = None
    next_states = []
    selected_experts = []
    routing_weights = []
    attention_rope = (
        attention.make_text_rope(
            state.position,
            len(tokens),
            config.attention,
            matrix_dtype(weights.embedding),
        )
        if shared_attention_rope
        else None
    )
    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(config.layer_types, weights.layers, state.layers)
    ):
        next_input_norm = (
            weights.layers[index + 1].norms.input_layernorm
            if index + 1 < len(weights.layers)
            else weights.final_norm
        )
        if kind == LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), f"GDN weights mismatch at {index}")
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            result = layer.prefill_gdn(
                hidden,
                layer_state,
                layer_weights,
                config.gdn,
                config.moe,
                normalized_input=normalized_input,
                next_input_norm=next_input_norm,
                fused_moe_shared_gate=fused_moe_shared_gate,
            )
        else:
            require(
                isinstance(layer_weights, layer.AttentionLayerWeights),
                f"attention weights mismatch at {index}",
            )
            require(
                isinstance(
                    layer_state,
                    (attention.MLXAttentionState, attention.MLXLinearAttentionState),
                ),
                f"attention state mismatch at {index}",
            )
            result = layer.prefill_attention(
                hidden,
                layer_state,
                layer_weights,
                config.attention,
                config.moe,
                normalized_input=normalized_input,
                next_input_norm=next_input_norm,
                use_steel=use_steel,
                attention_rope=attention_rope,
                grouped_attention_gqa=grouped_attention_gqa,
                fused_moe_shared_gate=fused_moe_shared_gate,
                exact_long_attention=exact_long_attention,
            )
        hidden = result.output
        normalized_input = result.normalized_output
        next_states.append(result.state)
        selected_experts.append(result.selected_experts)
        routing_weights.append(result.routing_weights)

    require(normalized_input is not None, "final chunk normalized output is missing")
    return TextModelChunkTransition(
        hidden=normalized_input,
        state=TextModelState(
            position=state.position + len(tokens),
            layers=tuple(next_states),
        ),
        selected_experts=tuple(selected_experts),
        routing_weights=tuple(routing_weights),
    )


def prefill_chunk(
    token_ids: Sequence[int],
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
) -> TextModelChunkResult:
    """Evaluate one prompt chunk and project only its final hidden state."""
    transition = prefill_hidden_chunk(
        token_ids,
        state,
        weights,
        config,
        use_steel=use_steel,
        shared_attention_rope=shared_attention_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        fused_moe_shared_gate=fused_moe_shared_gate,
        exact_long_attention=exact_long_attention,
    )
    return TextModelChunkResult(
        hidden=transition.hidden,
        state=transition.state,
        selected_experts=transition.selected_experts,
        routing_weights=transition.routing_weights,
        logits=project_lm_head(weights.lm_head, transition.hidden[-1]),
    )


def prefill_linear_session_chunk(
    token_ids: Sequence[int],
    session: TextLinearDecodeSession,
    *,
    project_logits: bool,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
) -> TextModelChunkTransition | TextModelChunkResult:
    """Advance and eagerly commit one chunk to a single-owner linear session."""
    tokens = tuple(token_ids)
    _require_linear_session(session)
    with session._owner.lock:
        require(tokens, "prefill chunk must contain at least one token")
        require(
            session.state.position + len(tokens) <= session.capacity,
            "linear decode capacity exhausted",
        )
        transition = prefill_hidden_chunk(
            tokens,
            session.state,
            session.weights,
            session.config,
            use_steel=use_steel,
            shared_attention_rope=shared_attention_rope,
            grouped_attention_gqa=grouped_attention_gqa,
            fused_moe_shared_gate=fused_moe_shared_gate,
            exact_long_attention=exact_long_attention,
            _validated=True,
        )
        if project_logits:
            result = TextModelChunkResult(
                hidden=transition.hidden,
                state=transition.state,
                selected_experts=transition.selected_experts,
                routing_weights=transition.routing_weights,
                logits=project_lm_head(session.weights.lm_head, transition.hidden[-1]),
            )
            evaluate_chunk_result(result)
        else:
            result = transition
            evaluate_chunk_transition(result)
        session.state = result.state
        return result


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


def load_text_model(
    root: Path,
    *,
    map_embedding: bool = False,
    quantize_embedding: bool = False,
    quantize_lm_head: bool = False,
    embedding_bits: int = 8,
    embedding_group_size: int = 32,
    lm_head_bits: int = 8,
    lm_head_group_size: int = 32,
) -> TextModelWeights:
    """Load only explicitly cataloged text tensors from a verified source."""
    require(
        not (map_embedding and quantize_embedding),
        "embedding cannot be both mapped and quantized",
    )
    source_path = require_verified_source(root)
    mapped_embedding = (
        vocab.MLXMappedBF16Matrix(
            source_path,
            "model.language_model.embed_tokens.weight",
            (248_320, 2048),
        )
        if map_embedding
        else None
    )
    with SafetensorsFile(source_path) as source:
        if mapped_embedding is not None:
            embedding = mapped_embedding
        else:
            source_embedding = _load_bf16(
                source,
                "model.language_model.embed_tokens.weight",
                (248_320, 2048),
            )
            embedding = (
                vocab.quantize_affine(
                    source_embedding,
                    bits=embedding_bits,
                    group_size=embedding_group_size,
                )
                if quantize_embedding
                else source_embedding
            )
        final_norm = _load_bf16(source, "model.language_model.norm.weight", (2048,))
        source_lm_head = _load_bf16(source, "lm_head.weight", (248_320, 2048))
        if quantize_lm_head:
            lm_head = vocab.quantize_affine(
                source_lm_head,
                bits=lm_head_bits,
                group_size=lm_head_group_size,
            )
            head_arrays = (lm_head.packed, lm_head.scales, lm_head.biases)
        else:
            lm_head = source_lm_head
            head_arrays = (lm_head,)
        embedding_arrays = (
            (embedding.packed, embedding.scales, embedding.biases)
            if isinstance(embedding, vocab.MLXAffineQuantizedMatrix)
            else (() if isinstance(embedding, vocab.MLXMappedBF16Matrix) else (embedding,))
        )
        mx.eval(*embedding_arrays, final_norm, *head_arrays)
    if quantize_embedding:
        del source_embedding
    if quantize_lm_head:
        del source_lm_head
    if quantize_embedding or quantize_lm_head:
        mx.clear_cache()
    layers = tuple(layer.load_layer(source_path, index) for index in range(40))
    weights = TextModelWeights(
        embedding=embedding,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights


def parse_token_ids(
    text: str,
    vocab_size: int = PRODUCTION_CONFIG.vocab_size,
) -> tuple[int, ...]:
    try:
        values = tuple(
            int(value.strip()) for value in text.split(",") if value.strip()
        )
    except ValueError as exc:
        raise MoEError("tokens must be comma-separated integers") from exc
    require(values, "at least one token is required")
    require(
        all(0 <= value < vocab_size for value in values),
        "token ID is out of range",
    )
    return values


def _state_arrays(state: TextModelState) -> list[mx.array]:
    arrays: list[mx.array] = []
    for layer_state in state.layers:
        if isinstance(layer_state, gdn.MLXGDNState):
            arrays.extend((layer_state.conv, layer_state.recurrent))
        else:
            require(
                isinstance(
                    layer_state,
                    (attention.MLXAttentionState, attention.MLXLinearAttentionState),
                ),
                "invalid layer state",
            )
            arrays.extend((layer_state.keys, layer_state.values))
    return arrays


def evaluate_transition(result: TextModelTransition) -> None:
    """Materialize a hidden transition and its complete rollback state."""
    mx.eval(
        result.hidden,
        *_state_arrays(result.state),
    )
    mx.synchronize()


def evaluate_result(result: TextModelResult) -> None:
    """Materialize a target result without exposing cache internals to frontends."""
    mx.eval(
        result.logits,
        result.hidden,
        *result.selected_experts,
        *result.routing_weights,
        *_state_arrays(result.state),
    )
    mx.synchronize()


def evaluate_chunk_transition(
    result: TextModelChunkTransition,
    *,
    diagnostics: bool = False,
) -> None:
    """Materialize a chunk and complete rollback state on one synchronization."""
    arrays = [result.hidden, *_state_arrays(result.state)]
    if diagnostics:
        arrays.extend((*result.selected_experts, *result.routing_weights))
    mx.eval(*arrays)
    mx.synchronize()


def evaluate_chunk_result(
    result: TextModelChunkResult,
    *,
    diagnostics: bool = False,
) -> None:
    """Materialize final chunk logits and state on one synchronization."""
    arrays = [result.logits, result.hidden, *_state_arrays(result.state)]
    if diagnostics:
        arrays.extend((*result.selected_experts, *result.routing_weights))
    mx.eval(*arrays)
    mx.synchronize()


def run_source_smoke(root: Path, token_ids: tuple[int, ...]) -> None:
    """Run a bounded resident source smoke; this is not a quality acceptance."""
    started = time.perf_counter()
    weights = load_text_model(root)
    loaded = time.perf_counter()
    print(
        "real-model-load "
        f"layers={len(weights.layers)} elapsed_s={loaded - started:.3f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    state = initial_state(weights, PRODUCTION_CONFIG)
    timings = []
    for step, token_id in enumerate(token_ids, start=1):
        before = time.perf_counter()
        result = forward_token(token_id, state, weights, PRODUCTION_CONFIG)
        evaluate_result(result)
        elapsed = time.perf_counter() - before
        timings.append(elapsed)
        require(
            bool(mx.all(mx.isfinite(result.logits)).item()),
            "non-finite target logits",
        )
        require(
            all(
                abs(float(mx.sum(route.astype(mx.float32)).item()) - 1.0) < 2e-3
                for route in result.routing_weights
            ),
            "routing weights are not normalized",
        )
        top_id = int(mx.argmax(result.logits).item())
        top_logit = float(result.logits[top_id].item())
        hidden_l2 = float(
            mx.sqrt(mx.sum(result.hidden.astype(mx.float32) ** 2)).item()
        )
        print(
            "real-model-token "
            f"step={step} input={token_id} output={top_id} "
            f"top_logit={top_logit:.9g} hidden_l2={hidden_l2:.9g} "
            f"elapsed_s={elapsed:.3f}",
            flush=True,
        )
        state = result.state
    if len(timings) > 2:
        measured = timings[2:]
        mean = sum(measured) / len(measured)
        print(
            "real-model-post-warmup "
            f"tokens={len(measured)} mean_ms={mean * 1000:.3f} "
            f"median_ms={statistics.median(measured) * 1000:.3f} "
            f"tokens_s={1.0 / mean:.3f}",
            flush=True,
        )
    print(
        "real-model-done "
        f"position={state.position} active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--tokens",
        default="248044,9707",
        help="comma-separated token IDs for a bounded sequential smoke",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_source_smoke(args.root, parse_token_ids(args.tokens))
    except (MoEError, OSError, ValueError) as exc:
        print(f"ornith35 real model smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_compiled as compiled
import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_moe as moe
import ornith35_mlx_turboquant_cache as turboquant_cache
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT, SafetensorsFile, require_verified_source


LAYER_GDN = "gdn"
LAYER_ATTENTION = "attention"
_DECODE_SESSION_SEAL = object()
_LINEAR_DECODE_SESSION_SEAL = object()
_LINEAR_DECODE_CHECKPOINT_SEAL = object()
_TURBOQUANT_DECODE_SESSION_SEAL = object()


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
    | attention.MLXTurboQuantImmutableAttentionState
    | attention.MLXTurboQuantAttentionState
)
CompiledGDNLayers = tuple[compiled.CompiledGDNLayer | None, ...]
CompiledAttentionTails = tuple[compiled.CompiledAttentionTail | None, ...]
CompiledPrefillTails = tuple[compiled.CompiledPrefillTail, ...]


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
    context_profile: str = context.NATIVE_PROFILE_ID


@dataclass(frozen=True)
class TextDecodeSession:
    """Weights and rollback state accepted once for unchecked nested decode."""

    weights: TextModelWeights
    state: TextModelState
    config: TextModelConfig
    _compiled_gdn_layers: CompiledGDNLayers | None = field(repr=False, compare=False)
    _compiled_attention_tails: CompiledAttentionTails | None = field(
        repr=False,
        compare=False,
    )
    _seal: object = field(repr=False, compare=False)


@dataclass
class _LinearDecodeOwner:
    session: ReferenceType[object] | None = None
    lock: Lock = field(default_factory=Lock, repr=False)


@dataclass
class TextLinearDecodeSession:
    """Single-owner fixed cache with checked journal and prefix rollback."""

    weights: TextModelWeights
    state: TextModelState
    config: TextModelConfig
    capacity: int
    _compiled_gdn_layers: CompiledGDNLayers | None = field(repr=False, compare=False)
    _compiled_attention_tails: CompiledAttentionTails | None = field(
        repr=False,
        compare=False,
    )
    _owner: _LinearDecodeOwner = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class TextLinearDecodeCheckpoint:
    """Owner-bound logical position and recurrent state for exact replay."""

    state: TextModelState
    _owner: _LinearDecodeOwner = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


@dataclass
class TextTurboQuantDecodeSession:
    """Single-owner lossy K4-MSE K/V decode after authoritative BF16 prefill."""

    weights: TextModelWeights
    state: TextModelState
    config: TextModelConfig
    capacity: int
    _compiled_gdn_layers: CompiledGDNLayers | None = field(repr=False, compare=False)
    _compiled_attention_tails: CompiledAttentionTails | None = field(
        repr=False,
        compare=False,
    )
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


@dataclass(frozen=True)
class TextModelAuxTransition(TextModelTransition):
    """Target transition plus vLLM-indexed, pre-final-norm layer outputs."""

    auxiliary_hidden_state_indices: tuple[int, ...]
    auxiliary_hidden_states: tuple[mx.array, ...]


@dataclass(frozen=True)
class TextModelAuxChunkTransition(TextModelChunkTransition):
    """Target chunk plus vLLM-indexed, pre-final-norm layer outputs."""

    auxiliary_hidden_state_indices: tuple[int, ...]
    auxiliary_hidden_states: tuple[mx.array, ...]


@dataclass(frozen=True)
class TextModelAuxResult(TextModelResult):
    auxiliary_hidden_state_indices: tuple[int, ...]
    auxiliary_hidden_states: tuple[mx.array, ...]


@dataclass(frozen=True)
class TextModelAuxChunkResult(TextModelChunkResult):
    auxiliary_hidden_state_indices: tuple[int, ...]
    auxiliary_hidden_states: tuple[mx.array, ...]


@dataclass(frozen=True)
class TextModelAttentionInputChunkTransition(TextModelChunkTransition):
    """Target chunk plus exact normalized inputs to full-attention mixers."""

    attention_layer_indices: tuple[int, ...]
    attention_inputs: tuple[mx.array, ...]


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


def validate_aux_hidden_state_indices(
    indices: Sequence[int],
    config: TextModelConfig,
) -> tuple[int, ...]:
    """Validate vLLM indexing: 0 is embedding, N is output of layer N-1."""
    values = tuple(indices)
    require(values, "at least one auxiliary hidden-state index is required")
    require(
        all(isinstance(index, int) for index in values),
        "auxiliary hidden-state indices must be integers",
    )
    require(
        values == tuple(sorted(set(values))),
        "auxiliary hidden-state indices must be unique and increasing",
    )
    require(
        values[0] >= 0 and values[-1] <= len(config.layer_types),
        "auxiliary hidden-state index is out of range",
    )
    return values


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


def initial_state(
    weights: TextModelWeights,
    config: TextModelConfig,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> TextModelState:
    validate_weights(weights, config)
    context.validate_range(context_profile, 0)
    dtype = matrix_dtype(weights.embedding)
    states = tuple(
        gdn.zeros_state(config.gdn, conv_dtype=dtype)
        if kind == LAYER_GDN
        else attention.zeros_state(
            config.attention,
            dtype=dtype,
            context_profile=context_profile,
        )
        for kind in config.layer_types
    )
    return TextModelState(
        position=0,
        layers=states,
        context_profile=context_profile,
    )


def validate_state(state: TextModelState, config: TextModelConfig) -> None:
    profile = context.validate_range(state.context_profile, state.position)
    require(len(state.layers) == len(config.layer_types), "model-state layer count mismatch")
    for index, (kind, layer_state) in enumerate(zip(config.layer_types, state.layers)):
        if kind == LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            gdn.validate_state(layer_state, config.gdn)
        else:
            require(
                isinstance(
                    layer_state,
                    (
                        attention.MLXAttentionState,
                        attention.MLXLinearAttentionState,
                        attention.MLXTurboQuantImmutableAttentionState,
                        attention.MLXTurboQuantAttentionState,
                    ),
                ),
                f"attention state mismatch at {index}",
            )
            require(
                layer_state.context_profile == profile.profile_id,
                f"attention context profile mismatch at {index}",
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


def _build_compiled_gdn_layers(
    weights: TextModelWeights,
    state: TextModelState,
    config: TextModelConfig,
    *,
    enabled: bool,
) -> CompiledGDNLayers | None:
    """Bind and warm only the production fixed-shape GatedDeltaNet layers."""
    if (
        not enabled
        or config != PRODUCTION_CONFIG
        or matrix_dtype(weights.embedding) != mx.bfloat16
    ):
        return None
    compiled_layers: list[compiled.CompiledGDNLayer | None] = []
    for index, (kind, layer_weights) in enumerate(zip(config.layer_types, weights.layers)):
        if kind == LAYER_ATTENTION:
            compiled_layers.append(None)
            continue
        require(
            isinstance(layer_weights, layer.GDNLayerWeights),
            f"compiled GDN weights mismatch at {index}",
        )
        next_input_norm = (
            weights.layers[index + 1].norms.input_layernorm
            if index + 1 < len(weights.layers)
            else weights.final_norm
        )
        compiled_layers.append(
            compiled.compile_gdn_layer(
                index,
                layer_weights,
                next_input_norm,
                config.gdn,
                config.moe,
            )
        )
    result = tuple(compiled_layers)
    compiled.warm_compiled_gdn_layers(
        result,
        state.layers,
        matrix_dtype(weights.embedding),
    )
    return result


def _build_compiled_attention_tails(
    weights: TextModelWeights,
    config: TextModelConfig,
    *,
    enabled: bool,
) -> CompiledAttentionTails | None:
    """Bind and warm the fixed work after each production attention mixer."""
    if (
        not enabled
        or config != PRODUCTION_CONFIG
        or matrix_dtype(weights.embedding) != mx.bfloat16
    ):
        return None
    compiled_tails: list[compiled.CompiledAttentionTail | None] = []
    for index, (kind, layer_weights) in enumerate(
        zip(config.layer_types, weights.layers)
    ):
        if kind == LAYER_GDN:
            compiled_tails.append(None)
            continue
        require(
            isinstance(layer_weights, layer.AttentionLayerWeights),
            f"compiled attention weights mismatch at {index}",
        )
        next_input_norm = (
            weights.layers[index + 1].norms.input_layernorm
            if index + 1 < len(weights.layers)
            else weights.final_norm
        )
        compiled_tails.append(
            compiled.compile_attention_tail(
                index,
                layer_weights,
                next_input_norm,
                config.moe,
            )
        )
    result = tuple(compiled_tails)
    compiled.warm_compiled_attention_tails(
        result,
        matrix_dtype(weights.embedding),
    )
    return result


def build_compiled_prefill_tails(
    weights: TextModelWeights,
    config: TextModelConfig,
    tokens: int,
    *,
    enabled: bool,
) -> CompiledPrefillTails | None:
    """Bind and warm exact fixed-token residual/MoE batch tails."""
    if (
        not enabled
        or config != PRODUCTION_CONFIG
        or matrix_dtype(weights.embedding) != mx.bfloat16
    ):
        return None
    require(1 <= tokens <= 8, "compiled prefill token count is invalid")
    compiled_tails = []
    for index, layer_weights in enumerate(weights.layers):
        next_input_norm = (
            weights.layers[index + 1].norms.input_layernorm
            if index + 1 < len(weights.layers)
            else weights.final_norm
        )
        compiled_tails.append(
            compiled.compile_prefill_tail(
                index,
                tokens,
                layer_weights,
                next_input_norm,
                config.moe,
            )
        )
    result = tuple(compiled_tails)
    compiled.warm_compiled_prefill_tails(
        result,
        matrix_dtype(weights.embedding),
    )
    return result


def start_decode_session(
    weights: TextModelWeights,
    state: TextModelState,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    compile_gdn_layers: bool = True,
    compile_attention_tails: bool = True,
) -> TextDecodeSession:
    """Deeply validate immutable decode inputs and bind them into a session."""
    _validate_decode_session(weights, state, config)
    compiled_gdn_layers = _build_compiled_gdn_layers(
        weights,
        state,
        config,
        enabled=compile_gdn_layers,
    )
    compiled_attention_tails = _build_compiled_attention_tails(
        weights,
        config,
        enabled=compile_attention_tails,
    )
    return TextDecodeSession(
        weights=weights,
        state=state,
        config=config,
        _compiled_gdn_layers=compiled_gdn_layers,
        _compiled_attention_tails=compiled_attention_tails,
        _seal=_DECODE_SESSION_SEAL,
    )


def start_linear_decode_session(
    weights: TextModelWeights,
    state: TextModelState,
    capacity: int,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    compile_gdn_layers: bool = True,
    compile_attention_tails: bool = True,
) -> TextLinearDecodeSession:
    """Move an immutable prefix into fixed-capacity, single-owner K/V buffers."""
    _validate_decode_session(weights, state, config)
    require(matrix_dtype(weights.embedding) == mx.bfloat16, "linear decode requires BF16 weights")
    require(capacity >= state.position, "linear decode capacity is shorter than the prefix")
    context.validate_range(state.context_profile, 0, capacity)
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
    linear_state = TextModelState(
        position=state.position,
        layers=tuple(next_states),
        context_profile=state.context_profile,
    )
    compiled_gdn_layers = _build_compiled_gdn_layers(
        weights,
        linear_state,
        config,
        enabled=compile_gdn_layers,
    )
    compiled_attention_tails = _build_compiled_attention_tails(
        weights,
        config,
        enabled=compile_attention_tails,
    )
    owner = _LinearDecodeOwner()
    session = TextLinearDecodeSession(
        weights=weights,
        state=linear_state,
        config=config,
        capacity=capacity,
        _compiled_gdn_layers=compiled_gdn_layers,
        _compiled_attention_tails=compiled_attention_tails,
        _owner=owner,
        _seal=_LINEAR_DECODE_SESSION_SEAL,
    )
    owner.session = ref(session)
    return session


def start_turboquant_decode_session(
    weights: TextModelWeights,
    state: TextModelState,
    capacity: int,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    compile_gdn_layers: bool = True,
    compile_attention_tails: bool = True,
) -> TextTurboQuantDecodeSession:
    """Compress a validated immutable or active linear BF16 prefix into K4-MSE K/V."""
    require(config == PRODUCTION_CONFIG, "TurboQuant requires production model geometry")
    require(matrix_dtype(weights.embedding) == mx.bfloat16, "TurboQuant requires BF16 weights")
    require(capacity >= state.position, "TurboQuant capacity is shorter than the prefix")
    context.validate_range(state.context_profile, 0, capacity)
    attention_states = tuple(
        layer_state
        for kind, layer_state in zip(config.layer_types, state.layers)
        if kind == LAYER_ATTENTION
    )
    source_is_bf16 = all(
        isinstance(
            layer_state,
            (attention.MLXAttentionState, attention.MLXLinearAttentionState),
        )
        for layer_state in attention_states
    )
    source_is_packed = all(
        isinstance(layer_state, attention.MLXTurboQuantImmutableAttentionState)
        for layer_state in attention_states
    )
    require(source_is_bf16 or source_is_packed, "TurboQuant source state types are mixed")
    validate_weights(weights, config)
    validate_state(state, config)
    next_states: list[LayerState] = []
    arrays: list[mx.array] = []
    for kind, layer_state in zip(config.layer_types, state.layers):
        if kind == LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), "invalid GDN state")
            next_states.append(layer_state)
            continue
        if isinstance(
            layer_state,
            (attention.MLXAttentionState, attention.MLXLinearAttentionState),
        ):
            if isinstance(layer_state, attention.MLXLinearAttentionState):
                require(
                    layer_state.position == state.position,
                    "TurboQuant linear source position mismatch",
                )
                keys = layer_state.keys[:, : state.position]
                values = layer_state.values[:, : state.position]
            else:
                keys = layer_state.keys
                values = layer_state.values
            packed = turboquant_cache.linearize_bf16_kv(
                keys,
                values,
                capacity,
                context_profile=layer_state.context_profile,
            )
        else:
            require(
                isinstance(layer_state, attention.MLXTurboQuantImmutableAttentionState),
                "TurboQuant source must be uniformly immutable",
            )
            packed = turboquant_cache.linearize_state(layer_state, capacity)
        next_states.append(packed)
        arrays.extend(
            (
                packed.packed_keys,
                packed.key_norms,
                packed.packed_values,
                packed.value_norms,
                packed.exact_keys,
                packed.exact_values,
            )
        )
    if arrays:
        mx.eval(*arrays)
        mx.synchronize()
    packed_state = TextModelState(
        position=state.position,
        layers=tuple(next_states),
        context_profile=state.context_profile,
    )
    compiled_gdn_layers = _build_compiled_gdn_layers(
        weights,
        packed_state,
        config,
        enabled=compile_gdn_layers,
    )
    compiled_attention_tails = _build_compiled_attention_tails(
        weights,
        config,
        enabled=compile_attention_tails,
    )
    owner = _LinearDecodeOwner()
    session = TextTurboQuantDecodeSession(
        weights=weights,
        state=packed_state,
        config=config,
        capacity=capacity,
        _compiled_gdn_layers=compiled_gdn_layers,
        _compiled_attention_tails=compiled_attention_tails,
        _owner=owner,
        _seal=_TURBOQUANT_DECODE_SESSION_SEAL,
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
    fused_postnorm_router: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_gdn_recurrence_inputs: bool = True,
    fused_gdn_beta_decay: bool = True,
    fused_gdn_input_transition: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
    _compiled_gdn_layers: CompiledGDNLayers | None = None,
    _compiled_attention_tails: CompiledAttentionTails | None = None,
    _validated: bool = False,
    _aux_hidden_state_indices: tuple[int, ...] = (),
    _aux_hidden_states: list[mx.array] | None = None,
) -> TextModelTransition:
    """Evaluate one token through the final norm without projecting logits."""
    require(isinstance(token_id, int) and 0 <= token_id < config.vocab_size, "token ID is out of range")
    if not _validated:
        validate_weights(weights, config)
        validate_state(state, config)
    if _aux_hidden_states is None:
        require(
            not _aux_hidden_state_indices,
            "auxiliary hidden-state sink is missing",
        )
        capture_indices = None
    else:
        require(not _aux_hidden_states, "auxiliary hidden-state sink is not empty")
        capture_indices = frozenset(
            validate_aux_hidden_state_indices(_aux_hidden_state_indices, config)
        )
    hidden = embed_token(weights.embedding, token_id)
    if capture_indices is not None and 0 in capture_indices:
        _aux_hidden_states.append(hidden)
    if _compiled_gdn_layers is not None:
        require(
            len(_compiled_gdn_layers) == len(config.layer_types),
            "compiled GDN layer count mismatch",
        )
    if _compiled_attention_tails is not None:
        require(
            len(_compiled_attention_tails) == len(config.layer_types),
            "compiled attention-tail count mismatch",
        )
    use_compiled_gdn = _compiled_gdn_layers is not None and all(
        (
            fused_residual_mean_square,
            fused_residual_rmsnorm,
            fused_postnorm_router,
            fused_gdn_convolution,
            fused_gdn_recurrence,
            fused_gdn_core_gate,
            fused_gdn_recurrence_inputs,
            fused_gdn_beta_decay,
            fused_gdn_input_transition,
            paired_moe_gate_up,
            fused_moe_shared_gate,
            fused_moe_routed_down,
        )
    )
    use_compiled_attention_tail = _compiled_attention_tails is not None and all(
        (
            fused_residual_mean_square,
            fused_residual_rmsnorm,
            fused_postnorm_router,
            paired_moe_gate_up,
            fused_moe_shared_gate,
            fused_moe_routed_down,
        )
    )
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
            state.context_profile,
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
            compiled_layer = (
                _compiled_gdn_layers[index]
                if use_compiled_gdn and _compiled_gdn_layers is not None
                else None
            )
            if compiled_layer is not None:
                result = compiled_layer(hidden, layer_state, normalized_input)
            else:
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
                    fused_postnorm_router=fused_postnorm_router,
                    fused_gdn_convolution=fused_gdn_convolution,
                    fused_gdn_recurrence=fused_gdn_recurrence,
                    fused_gdn_core_gate=fused_gdn_core_gate,
                    fused_gdn_recurrence_inputs=fused_gdn_recurrence_inputs,
                    fused_gdn_beta_decay=fused_gdn_beta_decay,
                    fused_gdn_input_transition=fused_gdn_input_transition,
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
                    (
                        attention.MLXAttentionState,
                        attention.MLXLinearAttentionState,
                        attention.MLXTurboQuantImmutableAttentionState,
                        attention.MLXTurboQuantAttentionState,
                    ),
                ),
                f"attention state mismatch at {index}",
            )
            compiled_tail = (
                _compiled_attention_tails[index]
                if use_compiled_attention_tail
                and _compiled_attention_tails is not None
                else None
            )
            if compiled_tail is not None:
                require(
                    normalized_input is not None,
                    "compiled attention layer is missing its normalized input",
                )
                mixed, next_state = attention.decode_step(
                    normalized_input,
                    layer_state,
                    layer_weights.token_mixer,
                    config.attention,
                    rope=attention_rope,
                    fused_qk_norm_rope=fused_attention_qk_norm_rope,
                    grouped_gqa=grouped_attention_gqa,
                    _validated=_validated,
                )
                tail_result = compiled_tail(hidden, mixed)
                result = layer.LayerResult(
                    output=tail_result.output,
                    state=next_state,
                    selected_experts=tail_result.selected_experts,
                    routing_weights=tail_result.routing_weights,
                    normalized_output=tail_result.normalized_output,
                )
            else:
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
                    fused_postnorm_router=fused_postnorm_router,
                    paired_moe_gate_up=paired_moe_gate_up,
                    fused_moe_shared_gate=fused_moe_shared_gate,
                    fused_moe_routed_down=fused_moe_routed_down,
                    _validated=_validated,
                )
        hidden = result.output
        if capture_indices is not None and index + 1 in capture_indices:
            _aux_hidden_states.append(hidden)
        normalized_input = result.normalized_output
        next_states.append(result.state)
        selected_experts.append(result.selected_experts)
        routing_weights.append(result.routing_weights)

    require(normalized_input is not None, "final normalized output is missing")
    if capture_indices is not None:
        require(
            len(_aux_hidden_states) == len(capture_indices),
            "auxiliary hidden-state capture is incomplete",
        )
    hidden = normalized_input
    return TextModelTransition(
        hidden=hidden,
        state=TextModelState(
            position=state.position + 1,
            layers=tuple(next_states),
            context_profile=state.context_profile,
        ),
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
    fused_postnorm_router: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_gdn_recurrence_inputs: bool = True,
    fused_gdn_beta_decay: bool = True,
    fused_gdn_input_transition: bool = True,
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
        fused_postnorm_router=fused_postnorm_router,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_gdn_recurrence_inputs=fused_gdn_recurrence_inputs,
        fused_gdn_beta_decay=fused_gdn_beta_decay,
        fused_gdn_input_transition=fused_gdn_input_transition,
        fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_shared_gate=fused_moe_shared_gate,
        fused_moe_routed_down=fused_moe_routed_down,
    )


def forward_hidden_token_with_aux(
    token_id: int,
    state: TextModelState,
    weights: TextModelWeights,
    auxiliary_hidden_state_indices: Sequence[int],
    config: TextModelConfig = PRODUCTION_CONFIG,
) -> TextModelAuxTransition:
    """Evaluate one token and retain only explicitly selected target states."""
    indices = validate_aux_hidden_state_indices(
        auxiliary_hidden_state_indices,
        config,
    )
    captured: list[mx.array] = []
    transition = _forward_hidden_token(
        token_id,
        state,
        weights,
        config,
        _aux_hidden_state_indices=indices,
        _aux_hidden_states=captured,
    )
    return TextModelAuxTransition(
        hidden=transition.hidden,
        state=transition.state,
        selected_experts=transition.selected_experts,
        routing_weights=transition.routing_weights,
        auxiliary_hidden_state_indices=indices,
        auxiliary_hidden_states=tuple(captured),
    )


def forward_token(
    token_id: int,
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_postnorm_router: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_gdn_recurrence_inputs: bool = True,
    fused_gdn_beta_decay: bool = True,
    fused_gdn_input_transition: bool = True,
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
        fused_postnorm_router=fused_postnorm_router,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_gdn_recurrence_inputs=fused_gdn_recurrence_inputs,
        fused_gdn_beta_decay=fused_gdn_beta_decay,
        fused_gdn_input_transition=fused_gdn_input_transition,
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
    fused_postnorm_router: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_gdn_recurrence_inputs: bool = True,
    fused_gdn_beta_decay: bool = True,
    fused_gdn_input_transition: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
    compiled_gdn_layers: bool = True,
    compiled_attention_tails: bool = True,
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
        fused_postnorm_router=fused_postnorm_router,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_gdn_recurrence_inputs=fused_gdn_recurrence_inputs,
        fused_gdn_beta_decay=fused_gdn_beta_decay,
        fused_gdn_input_transition=fused_gdn_input_transition,
        fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_shared_gate=fused_moe_shared_gate,
        fused_moe_routed_down=fused_moe_routed_down,
        _compiled_gdn_layers=(
            session._compiled_gdn_layers if compiled_gdn_layers else None
        ),
        _compiled_attention_tails=(
            session._compiled_attention_tails if compiled_attention_tails else None
        ),
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
        _compiled_gdn_layers=session._compiled_gdn_layers,
        _compiled_attention_tails=session._compiled_attention_tails,
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


def validate_linear_decode_session(session: TextLinearDecodeSession) -> None:
    """Validate the single owner and every fixed-capacity attention state."""
    _require_linear_session(session)
    validate_state(session.state, session.config)
    require(session.capacity >= session.state.position, "linear session capacity mismatch")
    for kind, layer_state in zip(session.config.layer_types, session.state.layers):
        if kind == LAYER_ATTENTION:
            require(
                isinstance(layer_state, attention.MLXLinearAttentionState)
                and layer_state.capacity == session.capacity,
                "linear session attention ownership mismatch",
            )


def _forward_linear_session_token(
    token_id: int,
    session: TextLinearDecodeSession,
    *,
    project_logits: bool,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_postnorm_router: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_gdn_recurrence_inputs: bool = True,
    fused_gdn_beta_decay: bool = True,
    fused_gdn_input_transition: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
    compiled_gdn_layers: bool = True,
    compiled_attention_tails: bool = True,
    _aux_hidden_state_indices: tuple[int, ...] = (),
    _aux_hidden_states: list[mx.array] | None = None,
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
            fused_postnorm_router=fused_postnorm_router,
            fused_gdn_convolution=fused_gdn_convolution,
            fused_gdn_recurrence=fused_gdn_recurrence,
            fused_gdn_core_gate=fused_gdn_core_gate,
            fused_gdn_recurrence_inputs=fused_gdn_recurrence_inputs,
            fused_gdn_beta_decay=fused_gdn_beta_decay,
            fused_gdn_input_transition=fused_gdn_input_transition,
            fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
            grouped_attention_gqa=grouped_attention_gqa,
            paired_moe_gate_up=paired_moe_gate_up,
            fused_moe_shared_gate=fused_moe_shared_gate,
            fused_moe_routed_down=fused_moe_routed_down,
            _compiled_gdn_layers=(
                session._compiled_gdn_layers if compiled_gdn_layers else None
            ),
            _compiled_attention_tails=(
                session._compiled_attention_tails
                if compiled_attention_tails
                else None
            ),
            _validated=True,
            _aux_hidden_state_indices=_aux_hidden_state_indices,
            _aux_hidden_states=_aux_hidden_states,
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
    fused_postnorm_router: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_gdn_recurrence_inputs: bool = True,
    fused_gdn_beta_decay: bool = True,
    fused_gdn_input_transition: bool = True,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
    compiled_gdn_layers: bool = True,
    compiled_attention_tails: bool = True,
) -> TextModelResult:
    """Advance and commit one token to a single-owner linear decode session."""
    result = _forward_linear_session_token(
        token_id,
        session,
        project_logits=True,
        fused_residual_mean_square=fused_residual_mean_square,
        fused_residual_rmsnorm=fused_residual_rmsnorm,
        fused_postnorm_router=fused_postnorm_router,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        fused_gdn_recurrence_inputs=fused_gdn_recurrence_inputs,
        fused_gdn_beta_decay=fused_gdn_beta_decay,
        fused_gdn_input_transition=fused_gdn_input_transition,
        fused_attention_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_shared_gate=fused_moe_shared_gate,
        fused_moe_routed_down=fused_moe_routed_down,
        compiled_gdn_layers=compiled_gdn_layers,
        compiled_attention_tails=compiled_attention_tails,
    )
    require(isinstance(result, TextModelResult), "linear decode result mismatch")
    return result


def forward_linear_session_token_with_aux(
    token_id: int,
    session: TextLinearDecodeSession,
    auxiliary_hidden_state_indices: Sequence[int],
) -> TextModelAuxResult:
    """Commit one optimized token and retain selected target layer outputs."""
    indices = validate_aux_hidden_state_indices(
        auxiliary_hidden_state_indices,
        session.config,
    )
    captured: list[mx.array] = []
    result = _forward_linear_session_token(
        token_id,
        session,
        project_logits=True,
        _aux_hidden_state_indices=indices,
        _aux_hidden_states=captured,
    )
    require(isinstance(result, TextModelResult), "linear auxiliary result mismatch")
    mx.eval(*captured)
    return TextModelAuxResult(
        hidden=result.hidden,
        state=result.state,
        selected_experts=result.selected_experts,
        routing_weights=result.routing_weights,
        logits=result.logits,
        auxiliary_hidden_state_indices=indices,
        auxiliary_hidden_states=tuple(captured),
    )


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


def _require_turboquant_session(session: TextTurboQuantDecodeSession) -> None:
    require(
        isinstance(session, TextTurboQuantDecodeSession)
        and session._seal is _TURBOQUANT_DECODE_SESSION_SEAL
        and session._owner.session is not None
        and session._owner.session() is session,
        "invalid TurboQuant decode session",
    )


def validate_turboquant_decode_session(session: TextTurboQuantDecodeSession) -> None:
    """Validate packed ownership without reconstructing historical K/V."""
    _require_turboquant_session(session)
    validate_state(session.state, session.config)
    require(session.capacity >= session.state.position, "TurboQuant session capacity mismatch")
    for kind, layer_state in zip(session.config.layer_types, session.state.layers):
        if kind == LAYER_ATTENTION:
            require(
                isinstance(layer_state, attention.MLXTurboQuantAttentionState)
                and layer_state.capacity == session.capacity,
                "TurboQuant attention ownership mismatch",
            )


def _forward_turboquant_session_token(
    token_id: int,
    session: TextTurboQuantDecodeSession,
    *,
    project_logits: bool,
) -> TextModelTransition | TextModelResult:
    _require_turboquant_session(session)
    with session._owner.lock:
        require(session.state.position < session.capacity, "TurboQuant capacity exhausted")
        transition = _forward_hidden_token(
            token_id,
            session.state,
            session.weights,
            session.config,
            _compiled_gdn_layers=session._compiled_gdn_layers,
            _compiled_attention_tails=session._compiled_attention_tails,
            _validated=True,
        )
        if project_logits:
            result: TextModelTransition | TextModelResult = TextModelResult(
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


def forward_turboquant_session_token(
    token_id: int,
    session: TextTurboQuantDecodeSession,
) -> TextModelResult:
    """Advance one lossy packed-K/V target token and commit its state."""
    result = _forward_turboquant_session_token(token_id, session, project_logits=True)
    require(isinstance(result, TextModelResult), "TurboQuant decode result mismatch")
    return result


def forward_turboquant_session_hidden_token(
    token_id: int,
    session: TextTurboQuantDecodeSession,
) -> TextModelTransition:
    """Advance one packed-K/V token without projecting unused logits."""
    result = _forward_turboquant_session_token(token_id, session, project_logits=False)
    require(type(result) is TextModelTransition, "TurboQuant hidden result mismatch")
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
    fused_long_attention: bool | None = None,
    _validated: bool = False,
    _gdn_rollback_inputs: list[mx.array] | None = None,
    _compiled_prefill_tails: CompiledPrefillTails | None = None,
    _aux_hidden_state_indices: tuple[int, ...] = (),
    _aux_hidden_states: list[mx.array] | None = None,
    _attention_inputs: list[mx.array] | None = None,
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
    if _aux_hidden_states is None:
        require(
            not _aux_hidden_state_indices,
            "auxiliary hidden-state sink is missing",
        )
        capture_indices = None
    else:
        require(not _aux_hidden_states, "auxiliary hidden-state sink is not empty")
        capture_indices = frozenset(
            validate_aux_hidden_state_indices(_aux_hidden_state_indices, config)
        )
    if _attention_inputs is not None:
        require(not _attention_inputs, "attention-input sink is not empty")
    if _compiled_prefill_tails is not None:
        require(
            len(_compiled_prefill_tails) == len(config.layer_types)
            and all(tail.tokens == len(tokens) for tail in _compiled_prefill_tails),
            "compiled prefill-tail contract mismatch",
        )
    hidden = embed_tokens(weights.embedding, tokens)
    if capture_indices is not None and 0 in capture_indices:
        _aux_hidden_states.append(hidden)
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
            state.context_profile,
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
        compiled_tail = (
            _compiled_prefill_tails[index]
            if _compiled_prefill_tails is not None
            else None
        )
        mixed_input = None
        if compiled_tail is not None:
            dtype = (
                layer_weights.token_mixer.in_proj_qkv.dtype
                if kind == LAYER_GDN
                else layer_weights.token_mixer.q_proj.dtype
            )
            hidden = hidden.astype(dtype)
            mixed_input = normalized_input
            if mixed_input is None:
                mixed_input = layer.qwen_rms_norm_batch(
                    hidden,
                    layer_weights.norms.input_layernorm,
                    config.rms_norm_eps,
                )
        if kind == LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), f"GDN weights mismatch at {index}")
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            if _gdn_rollback_inputs is not None:
                rollback_input = mixed_input if compiled_tail is not None else normalized_input
                if rollback_input is None:
                    rollback_input = layer.qwen_rms_norm_batch(
                        hidden,
                        layer_weights.norms.input_layernorm,
                        config.rms_norm_eps,
                    )
                _gdn_rollback_inputs.append(rollback_input)
            if compiled_tail is not None:
                require(mixed_input is not None, "compiled GDN prefill input is missing")
                mixed, next_state = gdn.prefill_chunk(
                    mixed_input,
                    layer_state,
                    layer_weights.token_mixer,
                    config.gdn,
                )
                tail_result = compiled_tail(hidden, mixed)
                result = layer.LayerResult(
                    output=tail_result.output,
                    state=next_state,
                    selected_experts=tail_result.selected_experts,
                    routing_weights=tail_result.routing_weights,
                    normalized_output=tail_result.normalized_output,
                )
            else:
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
            if _attention_inputs is not None:
                captured_input = mixed_input if compiled_tail is not None else normalized_input
                require(captured_input is not None, "normalized attention input is missing")
                _attention_inputs.append(captured_input)
            if compiled_tail is not None:
                require(mixed_input is not None, "compiled attention prefill input is missing")
                mixed, next_state = attention.prefill_chunk(
                    mixed_input,
                    layer_state,
                    layer_weights.token_mixer,
                    config.attention,
                    use_steel=use_steel,
                    rope=attention_rope,
                    grouped_gqa=grouped_attention_gqa,
                    exact_long_prefill=exact_long_attention,
                    fused_long_softmax_value=fused_long_attention,
                )
                tail_result = compiled_tail(hidden, mixed)
                result = layer.LayerResult(
                    output=tail_result.output,
                    state=next_state,
                    selected_experts=tail_result.selected_experts,
                    routing_weights=tail_result.routing_weights,
                    normalized_output=tail_result.normalized_output,
                )
            else:
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
                    fused_long_attention=fused_long_attention,
                )
        hidden = result.output
        if capture_indices is not None and index + 1 in capture_indices:
            _aux_hidden_states.append(hidden)
        normalized_input = result.normalized_output
        next_states.append(result.state)
        selected_experts.append(result.selected_experts)
        routing_weights.append(result.routing_weights)

    require(normalized_input is not None, "final chunk normalized output is missing")
    if capture_indices is not None:
        require(
            len(_aux_hidden_states) == len(capture_indices),
            "auxiliary hidden-state capture is incomplete",
        )
    if _attention_inputs is not None:
        require(
            len(_attention_inputs) == sum(kind == LAYER_ATTENTION for kind in config.layer_types),
            "attention-input capture is incomplete",
        )
    return TextModelChunkTransition(
        hidden=normalized_input,
        state=TextModelState(
            position=state.position + len(tokens),
            layers=tuple(next_states),
            context_profile=state.context_profile,
        ),
        selected_experts=tuple(selected_experts),
        routing_weights=tuple(routing_weights),
    )


def prefill_hidden_chunk_with_aux(
    token_ids: Sequence[int],
    state: TextModelState,
    weights: TextModelWeights,
    auxiliary_hidden_state_indices: Sequence[int],
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
) -> TextModelAuxChunkTransition:
    """Prefill a chunk and retain only explicitly selected target states."""
    indices = validate_aux_hidden_state_indices(
        auxiliary_hidden_state_indices,
        config,
    )
    captured: list[mx.array] = []
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
        fused_long_attention=fused_long_attention,
        _aux_hidden_state_indices=indices,
        _aux_hidden_states=captured,
    )
    return TextModelAuxChunkTransition(
        hidden=transition.hidden,
        state=transition.state,
        selected_experts=transition.selected_experts,
        routing_weights=transition.routing_weights,
        auxiliary_hidden_state_indices=indices,
        auxiliary_hidden_states=tuple(captured),
    )


def prefill_hidden_chunk_with_attention_inputs(
    token_ids: Sequence[int],
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    use_steel: bool = False,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
) -> TextModelAttentionInputChunkTransition:
    """Prefill while retaining only exact full-attention mixer inputs."""
    captured: list[mx.array] = []
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
        fused_long_attention=fused_long_attention,
        _attention_inputs=captured,
    )
    return TextModelAttentionInputChunkTransition(
        hidden=transition.hidden,
        state=transition.state,
        selected_experts=transition.selected_experts,
        routing_weights=transition.routing_weights,
        attention_layer_indices=tuple(
            index for index, kind in enumerate(config.layer_types) if kind == LAYER_ATTENTION
        ),
        attention_inputs=tuple(captured),
    )


def prefill_hidden_chunk_with_gdn_rollback(
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
    fused_long_attention: bool | None = None,
    compiled_prefill_tails: CompiledPrefillTails | None = None,
    _validated: bool = False,
) -> tuple[TextModelChunkTransition, tuple[mx.array, ...]]:
    """Prefill a target block and retain compact GDN rollback inputs."""
    rollback_inputs: list[mx.array] = []
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
        fused_long_attention=fused_long_attention,
        _validated=_validated,
        _gdn_rollback_inputs=rollback_inputs,
        _compiled_prefill_tails=compiled_prefill_tails,
    )
    require(
        len(rollback_inputs) == config.layer_types.count(LAYER_GDN),
        "GDN rollback-input count mismatch",
    )
    return transition, tuple(rollback_inputs)


def prefill_hidden_chunk_with_gdn_rollback_and_aux(
    token_ids: Sequence[int],
    state: TextModelState,
    weights: TextModelWeights,
    auxiliary_hidden_state_indices: Sequence[int],
    config: TextModelConfig = PRODUCTION_CONFIG,
    *,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
    compiled_prefill_tails: CompiledPrefillTails | None = None,
    _validated: bool = False,
) -> tuple[TextModelAuxChunkTransition, tuple[mx.array, ...]]:
    """Prefill a verifier block while retaining rollback and selected target states."""
    indices = validate_aux_hidden_state_indices(
        auxiliary_hidden_state_indices,
        config,
    )
    rollback_inputs: list[mx.array] = []
    captured: list[mx.array] = []
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
        fused_long_attention=fused_long_attention,
        _validated=_validated,
        _gdn_rollback_inputs=rollback_inputs,
        _compiled_prefill_tails=compiled_prefill_tails,
        _aux_hidden_state_indices=indices,
        _aux_hidden_states=captured,
    )
    require(
        len(rollback_inputs) == config.layer_types.count(LAYER_GDN),
        "GDN rollback-input count mismatch",
    )
    return (
        TextModelAuxChunkTransition(
            hidden=transition.hidden,
            state=transition.state,
            selected_experts=transition.selected_experts,
            routing_weights=transition.routing_weights,
            auxiliary_hidden_state_indices=indices,
            auxiliary_hidden_states=tuple(captured),
        ),
        tuple(rollback_inputs),
    )


def prefill_state_chunk(
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
    fused_long_attention: bool | None = None,
    _validated: bool = False,
) -> TextModelState:
    """Advance all persistent state without computing unobserved final-layer output."""
    tokens = tuple(token_ids)
    require(tokens, "prefill chunk must contain at least one token")
    require(
        all(isinstance(token, int) and 0 <= token < config.vocab_size for token in tokens),
        "prefill chunk token ID is out of range",
    )
    require(
        config.layer_types[-1] == LAYER_ATTENTION
        and isinstance(weights.layers[-1], layer.AttentionLayerWeights),
        "state-only prefill requires a final full-attention layer",
    )
    if not _validated:
        validate_weights(weights, config)
        validate_state(state, config)
    hidden = embed_tokens(weights.embedding, tokens)
    normalized_input = None
    next_states = []
    attention_rope = (
        attention.make_text_rope(
            state.position,
            len(tokens),
            config.attention,
            matrix_dtype(weights.embedding),
            state.context_profile,
        )
        if shared_attention_rope
        else None
    )
    final_index = len(config.layer_types) - 1
    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(
            config.layer_types[:final_index],
            weights.layers[:final_index],
            state.layers[:final_index],
        )
    ):
        next_input_norm = weights.layers[index + 1].norms.input_layernorm
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
                fused_long_attention=fused_long_attention,
            )
        hidden = result.output
        normalized_input = result.normalized_output
        next_states.append(result.state)

    require(normalized_input is not None, "final-layer normalized input is missing")
    final_weights = weights.layers[final_index]
    final_state = state.layers[final_index]
    require(
        isinstance(final_weights, layer.AttentionLayerWeights),
        "final attention weights mismatch",
    )
    require(
        isinstance(
            final_state,
            (attention.MLXAttentionState, attention.MLXLinearAttentionState),
        ),
        "final attention state mismatch",
    )
    next_states.append(
        attention.prefill_kv_chunk(
            normalized_input,
            final_state,
            final_weights.token_mixer,
            config.attention,
            rope=attention_rope,
        )
    )
    return TextModelState(
        position=state.position + len(tokens),
        layers=tuple(next_states),
        context_profile=state.context_profile,
    )


def prefill_final_chunk(
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
    fused_long_attention: bool | None = None,
    _validated: bool = False,
) -> TextModelResult:
    """Advance a prompt chunk and evaluate only its final observable token."""
    tokens = tuple(token_ids)
    require(tokens, "prefill chunk must contain at least one token")
    require(
        all(isinstance(token, int) and 0 <= token < config.vocab_size for token in tokens),
        "prefill chunk token ID is out of range",
    )
    require(
        config.layer_types[-1] == LAYER_ATTENTION
        and isinstance(weights.layers[-1], layer.AttentionLayerWeights),
        "final-token prefill requires a final full-attention layer",
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
            state.context_profile,
        )
        if shared_attention_rope
        else None
    )
    final_index = len(config.layer_types) - 1
    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(
            config.layer_types[:final_index],
            weights.layers[:final_index],
            state.layers[:final_index],
        )
    ):
        next_input_norm = weights.layers[index + 1].norms.input_layernorm
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
                fused_long_attention=fused_long_attention,
            )
        hidden = result.output
        normalized_input = result.normalized_output
        next_states.append(result.state)
        selected_experts.append(result.selected_experts[-1])
        routing_weights.append(result.routing_weights[-1])

    require(normalized_input is not None, "final-layer normalized input is missing")
    final_weights = weights.layers[final_index]
    final_state = state.layers[final_index]
    require(
        isinstance(final_weights, layer.AttentionLayerWeights),
        "final attention weights mismatch",
    )
    require(
        isinstance(
            final_state,
            (attention.MLXAttentionState, attention.MLXLinearAttentionState),
        ),
        "final attention state mismatch",
    )
    final_result = layer.prefill_attention_last(
        hidden,
        final_state,
        final_weights,
        config.attention,
        config.moe,
        normalized_input=normalized_input,
        next_input_norm=weights.final_norm,
        attention_rope=attention_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        fused_moe_shared_gate=fused_moe_shared_gate,
        exact_long_attention=exact_long_attention,
        fused_long_attention=fused_long_attention,
    )
    require(
        final_result.normalized_output is not None,
        "final prompt normalized output is missing",
    )
    final_hidden = final_result.normalized_output[0]
    next_states.append(final_result.state)
    selected_experts.append(final_result.selected_experts[0])
    routing_weights.append(final_result.routing_weights[0])
    return TextModelResult(
        hidden=final_hidden,
        state=TextModelState(
            position=state.position + len(tokens),
            layers=tuple(next_states),
            context_profile=state.context_profile,
        ),
        selected_experts=tuple(selected_experts),
        routing_weights=tuple(routing_weights),
        logits=project_lm_head(weights.lm_head, final_hidden),
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
    fused_long_attention: bool | None = None,
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
        fused_long_attention=fused_long_attention,
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
    fused_long_attention: bool | None = None,
    _gdn_rollback_inputs: list[mx.array] | None = None,
    _compiled_prefill_tails: CompiledPrefillTails | None = None,
    _aux_hidden_state_indices: tuple[int, ...] = (),
    _aux_hidden_states: list[mx.array] | None = None,
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
            fused_long_attention=fused_long_attention,
            _validated=True,
            _gdn_rollback_inputs=_gdn_rollback_inputs,
            _compiled_prefill_tails=_compiled_prefill_tails,
            _aux_hidden_state_indices=_aux_hidden_state_indices,
            _aux_hidden_states=_aux_hidden_states,
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


def prefill_linear_session_chunk_with_gdn_rollback_and_aux(
    token_ids: Sequence[int],
    session: TextLinearDecodeSession,
    auxiliary_hidden_state_indices: Sequence[int],
    *,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
    compiled_prefill_tails: CompiledPrefillTails | None = None,
) -> tuple[TextModelAuxChunkTransition, tuple[mx.array, ...]]:
    """Advance one owned verifier block and retain rollback plus target states."""
    indices = validate_aux_hidden_state_indices(
        auxiliary_hidden_state_indices,
        session.config,
    )
    rollback_inputs: list[mx.array] = []
    captured: list[mx.array] = []
    transition = prefill_linear_session_chunk(
        token_ids,
        session,
        project_logits=False,
        use_steel=use_steel,
        shared_attention_rope=shared_attention_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        fused_moe_shared_gate=fused_moe_shared_gate,
        exact_long_attention=exact_long_attention,
        fused_long_attention=fused_long_attention,
        _gdn_rollback_inputs=rollback_inputs,
        _compiled_prefill_tails=compiled_prefill_tails,
        _aux_hidden_state_indices=indices,
        _aux_hidden_states=captured,
    )
    require(type(transition) is TextModelChunkTransition, "linear verifier result mismatch")
    require(
        len(rollback_inputs) == session.config.layer_types.count(LAYER_GDN),
        "GDN rollback-input count mismatch",
    )
    return (
        TextModelAuxChunkTransition(
            hidden=transition.hidden,
            state=transition.state,
            selected_experts=transition.selected_experts,
            routing_weights=transition.routing_weights,
            auxiliary_hidden_state_indices=indices,
            auxiliary_hidden_states=tuple(captured),
        ),
        tuple(rollback_inputs),
    )


def restore_linear_session_state(
    session: TextLinearDecodeSession,
    state: TextModelState,
) -> None:
    """Rollback logical position/recurrent state while retaining owned K/V buffers."""
    _require_linear_session(session)
    with session._owner.lock:
        validate_state(state, session.config)
        require(
            state.position <= session.state.position,
            "linear rollback cannot advance the session",
        )
        for kind, current, restored in zip(
            session.config.layer_types,
            session.state.layers,
            state.layers,
        ):
            if kind == LAYER_ATTENTION:
                require(
                    isinstance(current, attention.MLXLinearAttentionState)
                    and isinstance(restored, attention.MLXLinearAttentionState)
                    and restored.keys is current.keys
                    and restored.values is current.values
                    and restored.capacity == session.capacity,
                    "linear rollback changed K/V ownership",
                )
        session.state = state


def checkpoint_linear_session_state(
    session: TextLinearDecodeSession,
) -> TextLinearDecodeCheckpoint:
    """Capture an owner-bound prefix without copying fixed-capacity K/V buffers."""
    _require_linear_session(session)
    with session._owner.lock:
        validate_linear_decode_session(session)
        return TextLinearDecodeCheckpoint(
            state=session.state,
            _owner=session._owner,
            _seal=_LINEAR_DECODE_CHECKPOINT_SEAL,
        )


def restore_linear_session_checkpoint(
    session: TextLinearDecodeSession,
    checkpoint: TextLinearDecodeCheckpoint,
) -> TextModelState:
    """Restore a captured logical prefix while retaining current K/V aliases."""
    _require_linear_session(session)
    require(
        isinstance(checkpoint, TextLinearDecodeCheckpoint)
        and checkpoint._seal is _LINEAR_DECODE_CHECKPOINT_SEAL
        and checkpoint._owner is session._owner,
        "linear checkpoint does not belong to this session",
    )
    with session._owner.lock:
        saved = checkpoint.state
        validate_state(saved, session.config)
        require(
            saved.position <= session.state.position,
            "linear checkpoint cannot advance the session",
        )
        layers: list[LayerState] = []
        for kind, current, restored in zip(
            session.config.layer_types,
            session.state.layers,
            saved.layers,
        ):
            if kind == LAYER_GDN:
                require(
                    isinstance(current, gdn.MLXGDNState)
                    and isinstance(restored, gdn.MLXGDNState),
                    "linear checkpoint GDN state mismatch",
                )
                layers.append(restored)
                continue
            require(
                isinstance(current, attention.MLXLinearAttentionState)
                and isinstance(restored, attention.MLXLinearAttentionState)
                and current.capacity == session.capacity
                and restored.capacity == session.capacity
                and current.context_profile == restored.context_profile,
                "linear checkpoint K/V ownership mismatch",
            )
            layers.append(
                attention.MLXLinearAttentionState(
                    keys=current.keys,
                    values=current.values,
                    position=saved.position,
                    capacity=session.capacity,
                    context_profile=current.context_profile,
                )
            )
        restored_state = TextModelState(
            position=saved.position,
            layers=tuple(layers),
            context_profile=saved.context_profile,
        )
        validate_state(restored_state, session.config)
        session.state = restored_state
        return restored_state


def prefill_linear_session_chunk_with_aux(
    token_ids: Sequence[int],
    session: TextLinearDecodeSession,
    auxiliary_hidden_state_indices: Sequence[int],
    *,
    project_logits: bool,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
) -> TextModelAuxChunkTransition | TextModelAuxChunkResult:
    """Commit one optimized prompt chunk and retain selected target states."""
    indices = validate_aux_hidden_state_indices(
        auxiliary_hidden_state_indices,
        session.config,
    )
    captured: list[mx.array] = []
    result = prefill_linear_session_chunk(
        token_ids,
        session,
        project_logits=project_logits,
        use_steel=use_steel,
        shared_attention_rope=shared_attention_rope,
        grouped_attention_gqa=grouped_attention_gqa,
        fused_moe_shared_gate=fused_moe_shared_gate,
        exact_long_attention=exact_long_attention,
        fused_long_attention=fused_long_attention,
        _aux_hidden_state_indices=indices,
        _aux_hidden_states=captured,
    )
    mx.eval(*captured)
    common = {
        "hidden": result.hidden,
        "state": result.state,
        "selected_experts": result.selected_experts,
        "routing_weights": result.routing_weights,
        "auxiliary_hidden_state_indices": indices,
        "auxiliary_hidden_states": tuple(captured),
    }
    if isinstance(result, TextModelChunkResult):
        return TextModelAuxChunkResult(logits=result.logits, **common)
    require(type(result) is TextModelChunkTransition, "linear auxiliary chunk mismatch")
    return TextModelAuxChunkTransition(**common)


def prefill_linear_session_state_chunk(
    token_ids: Sequence[int],
    session: TextLinearDecodeSession,
    *,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
) -> TextModelState:
    """Advance and commit a chunk whose final-layer hidden output is unobserved."""
    tokens = tuple(token_ids)
    _require_linear_session(session)
    with session._owner.lock:
        require(tokens, "prefill chunk must contain at least one token")
        require(
            session.state.position + len(tokens) <= session.capacity,
            "linear decode capacity exhausted",
        )
        next_state = prefill_state_chunk(
            tokens,
            session.state,
            session.weights,
            session.config,
            use_steel=use_steel,
            shared_attention_rope=shared_attention_rope,
            grouped_attention_gqa=grouped_attention_gqa,
            fused_moe_shared_gate=fused_moe_shared_gate,
            exact_long_attention=exact_long_attention,
            fused_long_attention=fused_long_attention,
            _validated=True,
        )
        evaluate_state(next_state)
        session.state = next_state
        return next_state


def prefill_linear_session_final_chunk(
    token_ids: Sequence[int],
    session: TextLinearDecodeSession,
    *,
    use_steel: bool = True,
    shared_attention_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
) -> TextModelResult:
    """Advance and commit a final prompt chunk with one observable token."""
    tokens = tuple(token_ids)
    _require_linear_session(session)
    with session._owner.lock:
        require(tokens, "prefill chunk must contain at least one token")
        require(
            session.state.position + len(tokens) <= session.capacity,
            "linear decode capacity exhausted",
        )
        result = prefill_final_chunk(
            tokens,
            session.state,
            session.weights,
            session.config,
            use_steel=use_steel,
            shared_attention_rope=shared_attention_rope,
            grouped_attention_gqa=grouped_attention_gqa,
            fused_moe_shared_gate=fused_moe_shared_gate,
            exact_long_attention=exact_long_attention,
            fused_long_attention=fused_long_attention,
            _validated=True,
        )
        evaluate_result(result)
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
            head_reference = vocab.MLXMappedBF16Matrix(
                source_path,
                "lm_head.weight",
                (248_320, 2048),
            )
            lm_head = vocab.quantize_affine(
                source_lm_head,
                bits=lm_head_bits,
                group_size=lm_head_group_size,
                reference=head_reference,
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


def load_exact_block_lm_head(root: Path) -> mx.array:
    """Load the retained source BF16 head for exact multi-token verification."""
    source_path = require_verified_source(root)
    with SafetensorsFile(source_path) as source:
        lm_head = _load_bf16(source, "lm_head.weight", (248_320, 2048))
        mx.eval(lm_head)
    return lm_head


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
        elif isinstance(
            layer_state,
            (
                attention.MLXTurboQuantImmutableAttentionState,
                attention.MLXTurboQuantAttentionState,
            ),
        ):
            arrays.extend(
                (
                    layer_state.packed_keys,
                    layer_state.key_norms,
                    layer_state.packed_values,
                    layer_state.value_norms,
                    layer_state.exact_keys,
                    layer_state.exact_values,
                )
            )
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


def evaluate_state(state: TextModelState) -> None:
    """Materialize a complete persistent state without hidden diagnostics."""
    mx.eval(*_state_arrays(state))
    mx.synchronize()


def evaluate_transition(
    result: TextModelTransition,
    *,
    additional_arrays: Sequence[mx.array] = (),
) -> None:
    """Materialize a hidden transition, state, and dependent outputs together."""
    mx.eval(
        result.hidden,
        *_state_arrays(result.state),
        *additional_arrays,
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
    additional_arrays: Sequence[mx.array] = (),
) -> None:
    """Materialize a chunk, rollback state, and dependent outputs together."""
    arrays = [result.hidden, *_state_arrays(result.state)]
    if diagnostics:
        arrays.extend((*result.selected_experts, *result.routing_weights))
    arrays.extend(additional_arrays)
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

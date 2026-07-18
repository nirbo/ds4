#!/usr/bin/env python3
"""MLX composition and strict loader for the Ornith-35 DSpark draft."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Sequence

import mlx.core as mx

import ornith35_mlx_linear_cache as linear_cache
from ornith35_dspark_reference import DSparkConfig, PRODUCTION_CONFIG
from ornith35_nvfp4 import SafetensorsFile


class MLXDSparkError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MLXDSparkError(message)


_LINEAR_CONTEXT_SEAL = object()


@dataclass
class _LinearContextOwner:
    generation: int = 0
    position: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)


@dataclass(frozen=True)
class MLXDraftAttentionWeights:
    q_proj: mx.array
    k_proj: mx.array
    v_proj: mx.array
    o_proj: mx.array
    q_norm: mx.array
    k_norm: mx.array


@dataclass(frozen=True)
class MLXDraftLayerWeights:
    attention: MLXDraftAttentionWeights
    input_norm: mx.array
    post_attention_norm: mx.array
    gate_proj: mx.array
    up_proj: mx.array
    down_proj: mx.array


@dataclass(frozen=True)
class MLXDSparkWeights:
    d2t: mx.array
    t2d: mx.array
    embedding: mx.array
    fc: mx.array
    hidden_norm: mx.array
    layers: tuple[MLXDraftLayerWeights, ...]
    norm: mx.array
    lm_head: mx.array
    markov_w1: mx.array
    markov_w2: mx.array
    confidence_weight: mx.array
    confidence_bias: mx.array


@dataclass(frozen=True)
class MLXDSparkContextState:
    position: int
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]


@dataclass(frozen=True)
class MLXDSparkLinearContextState:
    """Versioned view over one lock-protected set of fixed-capacity buffers."""

    position: int
    capacity: int
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    _generation: int = field(repr=False, compare=False)
    _owner: _LinearContextOwner = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class MLXDSparkProposal:
    target_token_ids: mx.array
    draft_token_ids: mx.array
    confidence: mx.array
    hidden_states: mx.array
    base_logits: mx.array
    corrected_logits: mx.array


def _weight_arrays(weights: MLXDSparkWeights) -> list[mx.array]:
    arrays = [
        weights.embedding,
        weights.fc,
        weights.hidden_norm,
        weights.norm,
        weights.lm_head,
        weights.markov_w1,
        weights.markov_w2,
        weights.confidence_weight,
        weights.confidence_bias,
    ]
    for layer in weights.layers:
        arrays.extend(
            (
                layer.input_norm,
                layer.post_attention_norm,
                layer.attention.q_proj,
                layer.attention.k_proj,
                layer.attention.v_proj,
                layer.attention.o_proj,
                layer.attention.q_norm,
                layer.attention.k_norm,
                layer.gate_proj,
                layer.up_proj,
                layer.down_proj,
            )
        )
    return arrays


def expected_tensor_specs(
    config: DSparkConfig,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    specs: dict[str, tuple[str, tuple[int, ...]]] = {
        "d2t": ("I64", (config.draft_vocab_size,)),
        "embed_tokens.weight": (
            "BF16",
            (config.target_vocab_size, config.hidden_size),
        ),
        "fc.weight": ("BF16", (config.hidden_size, config.aux_width)),
        "hidden_norm.weight": ("BF16", (config.hidden_size,)),
    }
    for index in range(config.num_layers):
        prefix = f"layers.{index}"
        specs.update(
            {
                f"{prefix}.input_layernorm.weight": (
                    "BF16",
                    (config.hidden_size,),
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    "BF16",
                    (config.hidden_size, config.intermediate_size),
                ),
                f"{prefix}.mlp.gate_proj.weight": (
                    "BF16",
                    (config.intermediate_size, config.hidden_size),
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    "BF16",
                    (config.intermediate_size, config.hidden_size),
                ),
                f"{prefix}.post_attention_layernorm.weight": (
                    "BF16",
                    (config.hidden_size,),
                ),
                f"{prefix}.self_attn.k_norm.weight": (
                    "BF16",
                    (config.head_dim,),
                ),
                f"{prefix}.self_attn.k_proj.weight": (
                    "BF16",
                    (config.kv_width, config.hidden_size),
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    "BF16",
                    (config.hidden_size, config.query_width),
                ),
                f"{prefix}.self_attn.q_norm.weight": (
                    "BF16",
                    (config.head_dim,),
                ),
                f"{prefix}.self_attn.q_proj.weight": (
                    "BF16",
                    (config.query_width, config.hidden_size),
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    "BF16",
                    (config.kv_width, config.hidden_size),
                ),
            }
        )
    specs.update(
        {
            "lm_head.weight": (
                "BF16",
                (config.draft_vocab_size, config.hidden_size),
            ),
            "markov_head.markov_w1.weight": (
                "BF16",
                (config.target_vocab_size, config.markov_rank),
            ),
            "markov_head.markov_w2.weight": (
                "BF16",
                (config.draft_vocab_size, config.markov_rank),
            ),
            "norm.weight": ("BF16", (config.hidden_size,)),
            "confidence_head.proj.bias": ("BF16", (1,)),
            "confidence_head.proj.weight": (
                "BF16",
                (1, config.hidden_size + config.markov_rank),
            ),
            "t2d": ("BOOL", (config.target_vocab_size,)),
        }
    )
    return specs


def validate_weights(weights: MLXDSparkWeights, config: DSparkConfig) -> None:
    require(weights.d2t.dtype == mx.int64, "DSpark d2t must be I64")
    require(weights.d2t.shape == (config.draft_vocab_size,), "DSpark d2t shape mismatch")
    require(weights.t2d.dtype == mx.bool_, "DSpark t2d must be boolean")
    require(weights.t2d.shape == (config.target_vocab_size,), "DSpark t2d shape mismatch")
    mapping_checks = (
        mx.sum(weights.t2d.astype(mx.int32)) == config.draft_vocab_size,
        mx.all(weights.d2t >= 0),
        mx.all(weights.d2t < config.target_vocab_size),
        mx.all(weights.d2t[1:] > weights.d2t[:-1]),
        mx.all(mx.take(weights.t2d, weights.d2t, axis=0)),
    )
    mx.eval(*mapping_checks)
    require(bool(mapping_checks[0].item()), "DSpark t2d population mismatch")
    require(
        all(bool(value.item()) for value in mapping_checks[1:]),
        "DSpark mappings disagree",
    )

    arrays = _weight_arrays(weights)
    dtype = weights.embedding.dtype
    require(dtype in (mx.bfloat16, mx.float32), "unsupported DSpark model dtype")
    require(all(value.dtype == dtype for value in arrays), "mixed DSpark weight dtypes")
    expected = {
        "embedding": (weights.embedding, (config.target_vocab_size, config.hidden_size)),
        "fc": (weights.fc, (config.hidden_size, config.aux_width)),
        "hidden norm": (weights.hidden_norm, (config.hidden_size,)),
        "final norm": (weights.norm, (config.hidden_size,)),
        "LM head": (weights.lm_head, (config.draft_vocab_size, config.hidden_size)),
        "Markov W1": (weights.markov_w1, (config.target_vocab_size, config.markov_rank)),
        "Markov W2": (weights.markov_w2, (config.draft_vocab_size, config.markov_rank)),
        "confidence weight": (
            weights.confidence_weight,
            (config.hidden_size + config.markov_rank,),
        ),
        "confidence bias": (weights.confidence_bias, (1,)),
    }
    for name, (value, shape) in expected.items():
        require(value.shape == shape, f"DSpark {name} shape mismatch")
    require(len(weights.layers) == config.num_layers, "DSpark layer count mismatch")
    for index, layer in enumerate(weights.layers):
        expected_layer = {
            "input norm": (layer.input_norm, (config.hidden_size,)),
            "post-attention norm": (layer.post_attention_norm, (config.hidden_size,)),
            "q projection": (
                layer.attention.q_proj,
                (config.query_width, config.hidden_size),
            ),
            "k projection": (
                layer.attention.k_proj,
                (config.kv_width, config.hidden_size),
            ),
            "v projection": (
                layer.attention.v_proj,
                (config.kv_width, config.hidden_size),
            ),
            "output projection": (
                layer.attention.o_proj,
                (config.hidden_size, config.query_width),
            ),
            "q norm": (layer.attention.q_norm, (config.head_dim,)),
            "k norm": (layer.attention.k_norm, (config.head_dim,)),
            "gate projection": (
                layer.gate_proj,
                (config.intermediate_size, config.hidden_size),
            ),
            "up projection": (
                layer.up_proj,
                (config.intermediate_size, config.hidden_size),
            ),
            "down projection": (
                layer.down_proj,
                (config.hidden_size, config.intermediate_size),
            ),
        }
        for name, (value, shape) in expected_layer.items():
            require(value.shape == shape, f"DSpark layer {index} {name} shape mismatch")


def qwen3_rms_norm(hidden: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Standard Qwen3 RMSNorm; draft weights are not centered Qwen3.5 weights."""
    require(hidden.shape[-1] == weight.shape[0], "DSpark RMSNorm shape mismatch")
    dtype = hidden.dtype
    values = hidden.astype(mx.float32)
    variance = mx.mean(values * values, axis=-1, keepdims=True)
    normalized = values * mx.rsqrt(variance + eps)
    return weight * normalized.astype(dtype)


def _linear(hidden: mx.array, weight: mx.array) -> mx.array:
    return mx.matmul(hidden, mx.transpose(weight))


def _rope_heads(hidden: mx.array, positions: mx.array, config: DSparkConfig) -> mx.array:
    require(
        hidden.ndim == 3
        and hidden.shape[0] == positions.shape[0]
        and hidden.shape[2] == config.head_dim,
        "DSpark RoPE input mismatch",
    )
    half = config.rotary_dim // 2
    frequencies = config.rope_theta ** (
        -mx.arange(0, config.rotary_dim, 2, dtype=mx.float32) / config.rotary_dim
    )
    angles = positions.astype(mx.float32)[:, None] * frequencies[None, :]
    cosine = mx.cos(angles).astype(hidden.dtype)[:, None, :]
    sine = mx.sin(angles).astype(hidden.dtype)[:, None, :]
    left = hidden[:, :, :half]
    right = hidden[:, :, half : config.rotary_dim]
    rotated = mx.concatenate(
        (left * cosine - right * sine, right * cosine + left * sine),
        axis=-1,
    )
    if config.rotary_dim == config.head_dim:
        return rotated
    return mx.concatenate((rotated, hidden[:, :, config.rotary_dim :]), axis=-1)


def _normalize_heads(hidden: mx.array, weight: mx.array, config: DSparkConfig) -> mx.array:
    require(hidden.shape[-1] == config.head_dim, "DSpark head width mismatch")
    return qwen3_rms_norm(hidden, weight, config.rms_norm_eps)


def initial_context(config: DSparkConfig, dtype: mx.Dtype) -> MLXDSparkContextState:
    require(dtype in (mx.bfloat16, mx.float32), "unsupported DSpark context dtype")
    keys = tuple(
        mx.zeros((0, config.num_kv_heads, config.head_dim), dtype=dtype)
        for _ in range(config.num_layers)
    )
    values = tuple(
        mx.zeros((0, config.num_kv_heads, config.head_dim), dtype=dtype)
        for _ in range(config.num_layers)
    )
    return MLXDSparkContextState(position=0, keys=keys, values=values)


def initial_linear_context(
    config: DSparkConfig,
    capacity: int,
) -> MLXDSparkLinearContextState:
    require(
        isinstance(capacity, int)
        and config.block_size <= capacity <= config.max_position_embeddings,
        "invalid DSpark linear-context capacity",
    )
    shape = (config.num_kv_heads, capacity, config.head_dim)
    owner = _LinearContextOwner()
    return MLXDSparkLinearContextState(
        position=0,
        capacity=capacity,
        keys=tuple(mx.zeros(shape, dtype=mx.bfloat16) for _ in range(config.num_layers)),
        values=tuple(mx.zeros(shape, dtype=mx.bfloat16) for _ in range(config.num_layers)),
        _generation=0,
        _owner=owner,
        _seal=_LINEAR_CONTEXT_SEAL,
    )


def validate_context(
    state: MLXDSparkContextState,
    config: DSparkConfig,
    dtype: mx.Dtype,
) -> None:
    require(state.position >= 0, "DSpark context position must be nonnegative")
    require(
        len(state.keys) == config.num_layers and len(state.values) == config.num_layers,
        "DSpark context layer count mismatch",
    )
    expected_shape = (state.position, config.num_kv_heads, config.head_dim)
    for index, (keys, values) in enumerate(zip(state.keys, state.values)):
        require(keys.shape == expected_shape, f"DSpark key shape mismatch at layer {index}")
        require(values.shape == expected_shape, f"DSpark value shape mismatch at layer {index}")
        require(
            keys.dtype == dtype and values.dtype == dtype,
            f"DSpark context dtype mismatch at layer {index}",
        )


def validate_linear_context(
    state: MLXDSparkLinearContextState,
    config: DSparkConfig,
    dtype: mx.Dtype,
) -> None:
    require(
        isinstance(state, MLXDSparkLinearContextState)
        and state._seal is _LINEAR_CONTEXT_SEAL
        and state._generation == state._owner.generation
        and state.position == state._owner.position,
        "invalid or stale DSpark linear context",
    )
    require(dtype == mx.bfloat16, "DSpark linear context requires BF16")
    require(
        isinstance(state.capacity, int)
        and 0 <= state.position <= state.capacity <= config.max_position_embeddings,
        "invalid DSpark linear-context position or capacity",
    )
    require(
        len(state.keys) == config.num_layers and len(state.values) == config.num_layers,
        "DSpark linear-context layer count mismatch",
    )
    expected_shape = (config.num_kv_heads, state.capacity, config.head_dim)
    for index, (keys, values) in enumerate(zip(state.keys, state.values)):
        require(
            keys.shape == expected_shape and values.shape == expected_shape,
            f"DSpark linear-context shape mismatch at layer {index}",
        )
        require(
            keys.dtype == dtype and values.dtype == dtype,
            f"DSpark linear-context dtype mismatch at layer {index}",
        )


def project_auxiliary_hidden_states(
    auxiliary_hidden_states: Sequence[mx.array],
    weights: MLXDSparkWeights,
    config: DSparkConfig,
) -> mx.array:
    require(
        len(auxiliary_hidden_states) == len(config.aux_hidden_state_indices),
        "DSpark auxiliary hidden-state count mismatch",
    )
    first = auxiliary_hidden_states[0]
    require(
        first.ndim == 2 and first.shape[0] > 0 and first.shape[1] == config.hidden_size,
        "DSpark auxiliary hidden-state shape mismatch",
    )
    require(
        all(value.shape == first.shape and value.dtype == first.dtype for value in auxiliary_hidden_states),
        "DSpark auxiliary hidden states disagree",
    )
    require(first.dtype == weights.fc.dtype, "target/DSpark auxiliary dtype mismatch")
    concatenated = mx.concatenate(tuple(auxiliary_hidden_states), axis=-1)
    projected = _linear(concatenated, weights.fc)
    return qwen3_rms_norm(projected, weights.hidden_norm, config.rms_norm_eps)


def append_context(
    state: MLXDSparkContextState | MLXDSparkLinearContextState,
    auxiliary_hidden_states: Sequence[mx.array],
    weights: MLXDSparkWeights,
    config: DSparkConfig = PRODUCTION_CONFIG,
    *,
    _validated: bool = False,
) -> MLXDSparkContextState | MLXDSparkLinearContextState:
    """Append target features while serializing the fixed-capacity owner."""
    if isinstance(state, MLXDSparkLinearContextState):
        with state._owner.lock:
            return _append_context(
                state,
                auxiliary_hidden_states,
                weights,
                config,
                _validated=_validated,
            )
    return _append_context(
        state,
        auxiliary_hidden_states,
        weights,
        config,
        _validated=_validated,
    )


def _append_context(
    state: MLXDSparkContextState | MLXDSparkLinearContextState,
    auxiliary_hidden_states: Sequence[mx.array],
    weights: MLXDSparkWeights,
    config: DSparkConfig = PRODUCTION_CONFIG,
    *,
    _validated: bool = False,
) -> MLXDSparkContextState | MLXDSparkLinearContextState:
    if not _validated:
        validate_weights(weights, config)
    dtype = weights.embedding.dtype
    linear = isinstance(state, MLXDSparkLinearContextState)
    if linear:
        validate_linear_context(state, config, dtype)
    else:
        require(isinstance(state, MLXDSparkContextState), "invalid DSpark context state")
        validate_context(state, config, dtype)
    projected = project_auxiliary_hidden_states(auxiliary_hidden_states, weights, config)
    tokens = projected.shape[0]
    require(
        state.position + tokens <= config.max_position_embeddings,
        "DSpark context exceeds the configured position limit",
    )
    if linear:
        require(
            state.position + tokens <= state.capacity,
            "DSpark linear-context capacity exhausted",
        )
    positions = mx.arange(state.position, state.position + tokens, dtype=mx.int32)
    next_keys = []
    next_values = []
    for index, layer in enumerate(weights.layers):
        keys = _linear(projected, layer.attention.k_proj).reshape(
            tokens,
            config.num_kv_heads,
            config.head_dim,
        )
        keys = _normalize_heads(keys, layer.attention.k_norm, config)
        keys = _rope_heads(keys, positions, config)
        values = _linear(projected, layer.attention.v_proj).reshape(
            tokens,
            config.num_kv_heads,
            config.head_dim,
        )
        if linear:
            appended_keys, appended_values = linear_cache.append_kv_transposed_bf16(
                state.keys[index],
                state.values[index],
                keys,
                values,
                state.position,
            )
            next_keys.append(appended_keys)
            next_values.append(appended_values)
        else:
            next_keys.append(mx.concatenate((state.keys[index], keys), axis=0))
            next_values.append(mx.concatenate((state.values[index], values), axis=0))
    if linear:
        result = MLXDSparkLinearContextState(
            position=state.position + tokens,
            capacity=state.capacity,
            keys=tuple(next_keys),
            values=tuple(next_values),
            _generation=state._generation + 1,
            _owner=state._owner,
            _seal=_LINEAR_CONTEXT_SEAL,
        )
        state._owner.generation = result._generation
        state._owner.position = result.position
        validate_linear_context(result, config, dtype)
    else:
        result = MLXDSparkContextState(
            position=state.position + tokens,
            keys=tuple(next_keys),
            values=tuple(next_values),
        )
        validate_context(result, config, dtype)
    return result


def _attention(
    hidden: mx.array,
    context_keys: mx.array,
    context_values: mx.array,
    positions: mx.array,
    weights: MLXDraftAttentionWeights,
    config: DSparkConfig,
    *,
    context_transposed: bool = False,
) -> mx.array:
    tokens = hidden.shape[0]
    query = _linear(hidden, weights.q_proj).reshape(
        tokens,
        config.num_q_heads,
        config.head_dim,
    )
    query = _rope_heads(_normalize_heads(query, weights.q_norm, config), positions, config)
    noise_keys = _linear(hidden, weights.k_proj).reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    noise_keys = _rope_heads(
        _normalize_heads(noise_keys, weights.k_norm, config),
        positions,
        config,
    )
    noise_values = _linear(hidden, weights.v_proj).reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    if context_transposed:
        keys = mx.concatenate((context_keys, mx.transpose(noise_keys, (1, 0, 2))), axis=1)
        values = mx.concatenate(
            (context_values, mx.transpose(noise_values, (1, 0, 2))),
            axis=1,
        )
    else:
        keys = mx.transpose(mx.concatenate((context_keys, noise_keys), axis=0), (1, 0, 2))
        values = mx.transpose(
            mx.concatenate((context_values, noise_values), axis=0),
            (1, 0, 2),
        )
    repeats = config.num_q_heads // config.num_kv_heads
    if repeats > 1:
        keys = mx.repeat(keys, repeats, axis=0)
        values = mx.repeat(values, repeats, axis=0)
    query = mx.transpose(query, (1, 0, 2))
    scores = mx.matmul(query, mx.transpose(keys, (0, 2, 1))) * (config.head_dim**-0.5)
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(hidden.dtype)
    attended = mx.matmul(probabilities, values)
    attended = mx.transpose(attended, (1, 0, 2)).reshape(tokens, config.query_width)
    return _linear(attended, weights.o_proj)


def _decoder_layer(
    hidden: mx.array,
    context_keys: mx.array,
    context_values: mx.array,
    positions: mx.array,
    weights: MLXDraftLayerWeights,
    config: DSparkConfig,
    *,
    context_transposed: bool = False,
) -> mx.array:
    dtype = hidden.dtype
    normalized = qwen3_rms_norm(hidden, weights.input_norm, config.rms_norm_eps)
    mixed = _attention(
        normalized,
        context_keys,
        context_values,
        positions,
        weights.attention,
        config,
        context_transposed=context_transposed,
    )
    hidden = (hidden + mixed).astype(dtype)
    normalized = qwen3_rms_norm(hidden, weights.post_attention_norm, config.rms_norm_eps)
    gate = _linear(normalized, weights.gate_proj)
    up = _linear(normalized, weights.up_proj)
    intermediate = (mx.sigmoid(gate) * gate) * up
    return (hidden + _linear(intermediate, weights.down_proj)).astype(dtype)


def propose(
    anchor_token_id: int | mx.array,
    state: MLXDSparkContextState | MLXDSparkLinearContextState,
    weights: MLXDSparkWeights,
    config: DSparkConfig = PRODUCTION_CONFIG,
    *,
    _validated: bool = False,
) -> MLXDSparkProposal:
    """Build one proposal while holding the fixed-capacity context owner."""
    if isinstance(state, MLXDSparkLinearContextState):
        with state._owner.lock:
            return _propose(
                anchor_token_id,
                state,
                weights,
                config,
                _validated=_validated,
            )
    return _propose(
        anchor_token_id,
        state,
        weights,
        config,
        _validated=_validated,
    )


def _propose(
    anchor_token_id: int | mx.array,
    state: MLXDSparkContextState | MLXDSparkLinearContextState,
    weights: MLXDSparkWeights,
    config: DSparkConfig = PRODUCTION_CONFIG,
    *,
    _validated: bool = False,
) -> MLXDSparkProposal:
    if not _validated:
        validate_weights(weights, config)
    linear = isinstance(state, MLXDSparkLinearContextState)
    if linear:
        validate_linear_context(state, config, weights.embedding.dtype)
        require(
            state.position + config.block_size <= state.capacity,
            "DSpark linear-context capacity cannot hold a complete proposal",
        )
    else:
        require(isinstance(state, MLXDSparkContextState), "invalid DSpark context state")
        validate_context(state, config, weights.embedding.dtype)
    require(
        state.position + config.block_size <= config.max_position_embeddings,
        "DSpark proposal exceeds the configured position limit",
    )
    if isinstance(anchor_token_id, int):
        require(
            0 <= anchor_token_id < config.target_vocab_size,
            "DSpark anchor token is out of range",
        )
        anchor = mx.array(anchor_token_id, dtype=mx.int64)
    else:
        require(
            isinstance(anchor_token_id, mx.array)
            and anchor_token_id.shape == ()
            and anchor_token_id.dtype in (mx.int32, mx.int64),
            "DSpark anchor token must be an integer scalar",
        )
        anchor = anchor_token_id.astype(mx.int64)
    token_ids = mx.concatenate(
        (
            anchor.astype(mx.int32).reshape(1),
            mx.full((config.block_size - 1,), config.mask_token_id, dtype=mx.int32),
        )
    )
    hidden = mx.take(weights.embedding, token_ids, axis=0)
    positions = mx.arange(state.position, state.position + config.block_size, dtype=mx.int32)
    for index, layer in enumerate(weights.layers):
        context_keys = (
            state.keys[index][:, : state.position]
            if linear
            else state.keys[index]
        )
        context_values = (
            state.values[index][:, : state.position]
            if linear
            else state.values[index]
        )
        hidden = _decoder_layer(
            hidden,
            context_keys,
            context_values,
            positions,
            layer,
            config,
            context_transposed=linear,
        )
    hidden = qwen3_rms_norm(hidden, weights.norm, config.rms_norm_eps)
    base_logits = _linear(hidden[1:], weights.lm_head)

    target_tokens = []
    draft_tokens = []
    confidences = []
    corrected_rows = []
    previous = anchor
    confidence_hidden = weights.confidence_weight[: config.hidden_size]
    confidence_markov = weights.confidence_weight[config.hidden_size :]
    for slot in range(config.block_size - 1):
        previous_embedding = mx.take(weights.markov_w1, previous, axis=0)
        corrected = base_logits[slot] + _linear(previous_embedding, weights.markov_w2)
        draft_token = mx.argmax(corrected).astype(mx.int64)
        target_token = mx.take(weights.d2t, draft_token, axis=0)
        confidence_logit = (
            mx.sum(hidden[slot + 1] * confidence_hidden)
            + mx.sum(previous_embedding * confidence_markov)
            + weights.confidence_bias[0]
        )
        target_tokens.append(target_token)
        draft_tokens.append(draft_token)
        confidences.append(mx.sigmoid(confidence_logit))
        corrected_rows.append(corrected)
        previous = target_token
    return MLXDSparkProposal(
        target_token_ids=mx.stack(target_tokens),
        draft_token_ids=mx.stack(draft_tokens),
        confidence=mx.stack(confidences),
        hidden_states=hidden,
        base_logits=base_logits,
        corrected_logits=mx.stack(corrected_rows),
    )


def _load_bf16(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "BF16", f"expected BF16 DSpark tensor: {name}")
    require(entry.get("shape") == list(shape), f"DSpark tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    require(len(payload) == 2 * _product(shape), f"DSpark tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.bfloat16).reshape(shape)


def _load_i64(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "I64", f"expected I64 DSpark tensor: {name}")
    require(entry.get("shape") == list(shape), f"DSpark tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    require(len(payload) == 8 * _product(shape), f"DSpark tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.int64).reshape(shape)


def _load_bool(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "BOOL", f"expected BOOL DSpark tensor: {name}")
    require(entry.get("shape") == list(shape), f"DSpark tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    require(len(payload) == _product(shape), f"DSpark tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).astype(mx.bool_).reshape(shape)


def _product(shape: tuple[int, ...]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value


def load_weights(
    source_path: Path,
    config: DSparkConfig = PRODUCTION_CONFIG,
) -> MLXDSparkWeights:
    """Load only the exact released DSpark tensor schema from a local file."""
    with SafetensorsFile(source_path) as source:
        expected_names = set(expected_tensor_specs(config))
        require(set(source.tensors) == expected_names, "DSpark tensor schema mismatch")
        layers = []
        for index in range(config.num_layers):
            prefix = f"layers.{index}"
            attention = MLXDraftAttentionWeights(
                q_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.q_proj.weight",
                    (config.query_width, config.hidden_size),
                ),
                k_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.k_proj.weight",
                    (config.kv_width, config.hidden_size),
                ),
                v_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.v_proj.weight",
                    (config.kv_width, config.hidden_size),
                ),
                o_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.o_proj.weight",
                    (config.hidden_size, config.query_width),
                ),
                q_norm=_load_bf16(
                    source,
                    f"{prefix}.self_attn.q_norm.weight",
                    (config.head_dim,),
                ),
                k_norm=_load_bf16(
                    source,
                    f"{prefix}.self_attn.k_norm.weight",
                    (config.head_dim,),
                ),
            )
            layers.append(
                MLXDraftLayerWeights(
                    attention=attention,
                    input_norm=_load_bf16(
                        source,
                        f"{prefix}.input_layernorm.weight",
                        (config.hidden_size,),
                    ),
                    post_attention_norm=_load_bf16(
                        source,
                        f"{prefix}.post_attention_layernorm.weight",
                        (config.hidden_size,),
                    ),
                    gate_proj=_load_bf16(
                        source,
                        f"{prefix}.mlp.gate_proj.weight",
                        (config.intermediate_size, config.hidden_size),
                    ),
                    up_proj=_load_bf16(
                        source,
                        f"{prefix}.mlp.up_proj.weight",
                        (config.intermediate_size, config.hidden_size),
                    ),
                    down_proj=_load_bf16(
                        source,
                        f"{prefix}.mlp.down_proj.weight",
                        (config.hidden_size, config.intermediate_size),
                    ),
                )
            )
        weights = MLXDSparkWeights(
            d2t=_load_i64(source, "d2t", (config.draft_vocab_size,)),
            t2d=_load_bool(source, "t2d", (config.target_vocab_size,)),
            embedding=_load_bf16(
                source,
                "embed_tokens.weight",
                (config.target_vocab_size, config.hidden_size),
            ),
            fc=_load_bf16(source, "fc.weight", (config.hidden_size, config.aux_width)),
            hidden_norm=_load_bf16(source, "hidden_norm.weight", (config.hidden_size,)),
            layers=tuple(layers),
            norm=_load_bf16(source, "norm.weight", (config.hidden_size,)),
            lm_head=_load_bf16(
                source,
                "lm_head.weight",
                (config.draft_vocab_size, config.hidden_size),
            ),
            markov_w1=_load_bf16(
                source,
                "markov_head.markov_w1.weight",
                (config.target_vocab_size, config.markov_rank),
            ),
            markov_w2=_load_bf16(
                source,
                "markov_head.markov_w2.weight",
                (config.draft_vocab_size, config.markov_rank),
            ),
            confidence_weight=_load_bf16(
                source,
                "confidence_head.proj.weight",
                (1, config.hidden_size + config.markov_rank),
            ).reshape(config.hidden_size + config.markov_rank),
            confidence_bias=_load_bf16(source, "confidence_head.proj.bias", (1,)),
        )
        mx.eval(weights.d2t, weights.t2d, *_weight_arrays(weights))
    validate_weights(weights, config)
    return weights

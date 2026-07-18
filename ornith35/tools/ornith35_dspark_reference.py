#!/usr/bin/env python3
"""Dependency-free scalar oracle for the released Ornith-35 DSpark draft."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


Vector = Sequence[float]
Matrix = Sequence[Vector]


class DSparkReferenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DSparkReferenceError(message)


@dataclass(frozen=True)
class DSparkConfig:
    target_vocab_size: int
    draft_vocab_size: int
    hidden_size: int
    aux_hidden_state_indices: tuple[int, ...]
    block_size: int
    mask_token_id: int
    num_layers: int
    intermediate_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    max_position_embeddings: int
    rms_norm_eps: float
    markov_rank: int

    def __post_init__(self) -> None:
        require(self.target_vocab_size > 0, "target vocabulary must be positive")
        require(0 < self.draft_vocab_size <= self.target_vocab_size, "invalid draft vocabulary")
        require(self.hidden_size > 0 and self.intermediate_size > 0, "invalid draft widths")
        require(self.aux_hidden_state_indices, "DSpark requires target auxiliary states")
        require(
            self.aux_hidden_state_indices
            == tuple(sorted(set(self.aux_hidden_state_indices))),
            "auxiliary hidden-state indices must be unique and increasing",
        )
        require(self.block_size >= 2, "DSpark block must include anchor and draft slots")
        require(0 <= self.mask_token_id < self.target_vocab_size, "invalid mask token")
        require(self.num_layers > 0, "DSpark requires decoder layers")
        require(
            self.num_q_heads > 0
            and self.num_kv_heads > 0
            and self.num_q_heads % self.num_kv_heads == 0,
            "invalid grouped-query head counts",
        )
        require(self.head_dim > 0, "invalid attention head dimension")
        require(
            0 < self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0,
            "invalid rotary dimension",
        )
        require(
            self.rope_theta > 0.0
            and self.max_position_embeddings >= self.block_size
            and self.rms_norm_eps > 0.0,
            "invalid numeric config",
        )
        require(self.markov_rank > 0, "DSpark requires a positive Markov rank")

    @property
    def query_width(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def aux_width(self) -> int:
        return len(self.aux_hidden_state_indices) * self.hidden_size


PRODUCTION_CONFIG = DSparkConfig(
    target_vocab_size=248_320,
    draft_vocab_size=32_000,
    hidden_size=2_048,
    aux_hidden_state_indices=(9, 19, 29),
    block_size=8,
    mask_token_id=248_077,
    num_layers=3,
    intermediate_size=6_144,
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=256,
    # The training Qwen3 rotary class ignored the inherited factor-0.25 field.
    rotary_dim=256,
    rope_theta=10_000_000.0,
    max_position_embeddings=262_144,
    rms_norm_eps=1e-6,
    markov_rank=256,
)


@dataclass(frozen=True)
class DraftAttentionWeights:
    q_proj: Matrix
    k_proj: Matrix
    v_proj: Matrix
    o_proj: Matrix
    q_norm: Vector
    k_norm: Vector


@dataclass(frozen=True)
class DraftLayerWeights:
    attention: DraftAttentionWeights
    input_norm: Vector
    post_attention_norm: Vector
    gate_proj: Matrix
    up_proj: Matrix
    down_proj: Matrix


@dataclass(frozen=True)
class DSparkWeights:
    d2t: Sequence[int]
    t2d: Sequence[bool]
    embedding: Matrix
    fc: Matrix
    hidden_norm: Vector
    layers: tuple[DraftLayerWeights, ...]
    norm: Vector
    lm_head: Matrix
    markov_w1: Matrix
    markov_w2: Matrix
    confidence_weight: Vector
    confidence_bias: float


@dataclass(frozen=True)
class DSparkContextState:
    position: int
    keys: tuple[tuple[tuple[float, ...], ...], ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]


@dataclass(frozen=True)
class DSparkProposal:
    target_token_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    confidence: tuple[float, ...]
    hidden_states: tuple[tuple[float, ...], ...]
    base_logits: tuple[tuple[float, ...], ...]
    corrected_logits: tuple[tuple[float, ...], ...]


def _validate_vector(name: str, value: Vector, size: int) -> None:
    require(len(value) == size, f"{name} shape mismatch")


def _validate_matrix(name: str, value: Matrix, rows: int, columns: int) -> None:
    require(len(value) == rows, f"{name} row count mismatch")
    require(all(len(row) == columns for row in value), f"{name} column count mismatch")


def validate_weights(weights: DSparkWeights, config: DSparkConfig) -> None:
    require(len(weights.d2t) == config.draft_vocab_size, "d2t shape mismatch")
    require(len(weights.t2d) == config.target_vocab_size, "t2d shape mismatch")
    selected = tuple(index for index, enabled in enumerate(weights.t2d) if enabled)
    require(len(selected) == config.draft_vocab_size, "t2d population mismatch")
    require(tuple(weights.d2t) == selected, "d2t and t2d mappings disagree")
    _validate_matrix(
        "embedding",
        weights.embedding,
        config.target_vocab_size,
        config.hidden_size,
    )
    _validate_matrix("fc", weights.fc, config.hidden_size, config.aux_width)
    _validate_vector("hidden norm", weights.hidden_norm, config.hidden_size)
    require(len(weights.layers) == config.num_layers, "draft layer count mismatch")
    for index, layer in enumerate(weights.layers):
        prefix = f"draft layer {index}"
        _validate_vector(f"{prefix} input norm", layer.input_norm, config.hidden_size)
        _validate_vector(
            f"{prefix} post-attention norm",
            layer.post_attention_norm,
            config.hidden_size,
        )
        _validate_matrix(
            f"{prefix} q projection",
            layer.attention.q_proj,
            config.query_width,
            config.hidden_size,
        )
        _validate_matrix(
            f"{prefix} k projection",
            layer.attention.k_proj,
            config.kv_width,
            config.hidden_size,
        )
        _validate_matrix(
            f"{prefix} v projection",
            layer.attention.v_proj,
            config.kv_width,
            config.hidden_size,
        )
        _validate_matrix(
            f"{prefix} output projection",
            layer.attention.o_proj,
            config.hidden_size,
            config.query_width,
        )
        _validate_vector(f"{prefix} q norm", layer.attention.q_norm, config.head_dim)
        _validate_vector(f"{prefix} k norm", layer.attention.k_norm, config.head_dim)
        _validate_matrix(
            f"{prefix} gate projection",
            layer.gate_proj,
            config.intermediate_size,
            config.hidden_size,
        )
        _validate_matrix(
            f"{prefix} up projection",
            layer.up_proj,
            config.intermediate_size,
            config.hidden_size,
        )
        _validate_matrix(
            f"{prefix} down projection",
            layer.down_proj,
            config.hidden_size,
            config.intermediate_size,
        )
    _validate_vector("draft final norm", weights.norm, config.hidden_size)
    _validate_matrix(
        "draft LM head",
        weights.lm_head,
        config.draft_vocab_size,
        config.hidden_size,
    )
    _validate_matrix(
        "Markov W1",
        weights.markov_w1,
        config.target_vocab_size,
        config.markov_rank,
    )
    _validate_matrix(
        "Markov W2",
        weights.markov_w2,
        config.draft_vocab_size,
        config.markov_rank,
    )
    _validate_vector(
        "confidence weight",
        weights.confidence_weight,
        config.hidden_size + config.markov_rank,
    )


def matvec(matrix: Matrix, vector: Vector) -> list[float]:
    return [math.fsum(left * right for left, right in zip(row, vector)) for row in matrix]


def rms_norm(hidden: Vector, weight: Vector, eps: float) -> list[float]:
    inverse = 1.0 / math.sqrt(math.fsum(value * value for value in hidden) / len(hidden) + eps)
    return [value * inverse * scale for value, scale in zip(hidden, weight)]


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def apply_rope(
    vector: Vector,
    position: int,
    config: DSparkConfig,
) -> list[float]:
    require(position >= 0, "RoPE position must be nonnegative")
    output = list(vector)
    half = config.rotary_dim // 2
    for index in range(half):
        frequency = config.rope_theta ** (-(2.0 * index) / config.rotary_dim)
        angle = position * frequency
        cosine = math.cos(angle)
        sine = math.sin(angle)
        left = vector[index]
        right = vector[index + half]
        output[index] = left * cosine - right * sine
        output[index + half] = right * cosine + left * sine
    return output


def initial_context(config: DSparkConfig) -> DSparkContextState:
    empty = tuple(tuple() for _ in range(config.num_layers))
    return DSparkContextState(position=0, keys=empty, values=empty)


def validate_context(state: DSparkContextState, config: DSparkConfig) -> None:
    require(state.position >= 0, "DSpark context position must be nonnegative")
    require(
        len(state.keys) == config.num_layers and len(state.values) == config.num_layers,
        "DSpark context layer count mismatch",
    )
    for index, (keys, values) in enumerate(zip(state.keys, state.values)):
        require(
            len(keys) == state.position and len(values) == state.position,
            f"DSpark context length mismatch at layer {index}",
        )
        require(
            all(len(row) == config.kv_width for row in keys),
            f"DSpark key width mismatch at layer {index}",
        )
        require(
            all(len(row) == config.kv_width for row in values),
            f"DSpark value width mismatch at layer {index}",
        )


def project_auxiliary_hidden_states(
    auxiliary_hidden_states: Sequence[Sequence[Vector]],
    weights: DSparkWeights,
    config: DSparkConfig,
) -> list[list[float]]:
    require(
        len(auxiliary_hidden_states) == len(config.aux_hidden_state_indices),
        "auxiliary hidden-state count mismatch",
    )
    tokens = len(auxiliary_hidden_states[0])
    require(tokens > 0, "auxiliary hidden-state chunk is empty")
    require(
        all(len(value) == tokens for value in auxiliary_hidden_states),
        "auxiliary hidden-state token counts disagree",
    )
    output = []
    for token in range(tokens):
        concatenated: list[float] = []
        for value in auxiliary_hidden_states:
            row = value[token]
            require(len(row) == config.hidden_size, "auxiliary hidden-state width mismatch")
            concatenated.extend(row)
        output.append(
            rms_norm(
                matvec(weights.fc, concatenated),
                weights.hidden_norm,
                config.rms_norm_eps,
            )
        )
    return output


def _normalize_heads(flattened: Vector, norm: Vector, config: DSparkConfig) -> list[float]:
    heads = len(flattened) // config.head_dim
    output: list[float] = []
    for head in range(heads):
        start = head * config.head_dim
        output.extend(
            rms_norm(
                flattened[start : start + config.head_dim],
                norm,
                config.rms_norm_eps,
            )
        )
    return output


def _rope_heads(flattened: Vector, position: int, config: DSparkConfig) -> list[float]:
    heads = len(flattened) // config.head_dim
    output: list[float] = []
    for head in range(heads):
        start = head * config.head_dim
        output.extend(
            apply_rope(flattened[start : start + config.head_dim], position, config)
        )
    return output


def append_context(
    state: DSparkContextState,
    auxiliary_hidden_states: Sequence[Sequence[Vector]],
    weights: DSparkWeights,
    config: DSparkConfig,
) -> DSparkContextState:
    validate_weights(weights, config)
    validate_context(state, config)
    projected = project_auxiliary_hidden_states(auxiliary_hidden_states, weights, config)
    require(
        state.position + len(projected) <= config.max_position_embeddings,
        "DSpark context exceeds the configured position limit",
    )
    next_keys = [list(layer_keys) for layer_keys in state.keys]
    next_values = [list(layer_values) for layer_values in state.values]
    for layer_index, layer in enumerate(weights.layers):
        for offset, hidden in enumerate(projected):
            position = state.position + offset
            key = _rope_heads(
                _normalize_heads(matvec(layer.attention.k_proj, hidden), layer.attention.k_norm, config),
                position,
                config,
            )
            value = matvec(layer.attention.v_proj, hidden)
            next_keys[layer_index].append(tuple(key))
            next_values[layer_index].append(tuple(value))
    result = DSparkContextState(
        position=state.position + len(projected),
        keys=tuple(tuple(rows) for rows in next_keys),
        values=tuple(tuple(rows) for rows in next_values),
    )
    validate_context(result, config)
    return result


def _softmax(values: Vector) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def _attention(
    hidden: Sequence[Vector],
    context_keys: Sequence[Vector],
    context_values: Sequence[Vector],
    positions: Sequence[int],
    weights: DraftAttentionWeights,
    config: DSparkConfig,
) -> list[list[float]]:
    normalized_queries = []
    noise_keys = []
    noise_values = []
    for row, position in zip(hidden, positions):
        normalized_queries.append(
            _rope_heads(
                _normalize_heads(matvec(weights.q_proj, row), weights.q_norm, config),
                position,
                config,
            )
        )
        noise_keys.append(
            _rope_heads(
                _normalize_heads(matvec(weights.k_proj, row), weights.k_norm, config),
                position,
                config,
            )
        )
        noise_values.append(matvec(weights.v_proj, row))

    all_keys = [*context_keys, *noise_keys]
    all_values = [*context_values, *noise_values]
    groups = config.num_q_heads // config.num_kv_heads
    scaling = config.head_dim**-0.5
    outputs = []
    for query in normalized_queries:
        attended: list[float] = []
        for query_head in range(config.num_q_heads):
            kv_head = query_head // groups
            q_start = query_head * config.head_dim
            kv_start = kv_head * config.head_dim
            query_row = query[q_start : q_start + config.head_dim]
            scores = [
                math.fsum(
                    query_row[column] * key[kv_start + column]
                    for column in range(config.head_dim)
                )
                * scaling
                for key in all_keys
            ]
            probabilities = _softmax(scores)
            attended.extend(
                math.fsum(
                    probability * value[kv_start + column]
                    for probability, value in zip(probabilities, all_values)
                )
                for column in range(config.head_dim)
            )
        outputs.append(matvec(weights.o_proj, attended))
    return outputs


def _decoder_layer(
    hidden: Sequence[Vector],
    context_keys: Sequence[Vector],
    context_values: Sequence[Vector],
    positions: Sequence[int],
    weights: DraftLayerWeights,
    config: DSparkConfig,
) -> list[list[float]]:
    normalized = [rms_norm(row, weights.input_norm, config.rms_norm_eps) for row in hidden]
    mixed = _attention(
        normalized,
        context_keys,
        context_values,
        positions,
        weights.attention,
        config,
    )
    first_residual = [
        [left + right for left, right in zip(row, delta)]
        for row, delta in zip(hidden, mixed)
    ]
    output = []
    for row in first_residual:
        mlp_input = rms_norm(row, weights.post_attention_norm, config.rms_norm_eps)
        gate = matvec(weights.gate_proj, mlp_input)
        up = matvec(weights.up_proj, mlp_input)
        intermediate = [silu(left) * right for left, right in zip(gate, up)]
        delta = matvec(weights.down_proj, intermediate)
        output.append([left + right for left, right in zip(row, delta)])
    return output


def _argmax_lowest(values: Vector) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def propose(
    anchor_token_id: int,
    state: DSparkContextState,
    weights: DSparkWeights,
    config: DSparkConfig,
) -> DSparkProposal:
    validate_weights(weights, config)
    validate_context(state, config)
    require(0 <= anchor_token_id < config.target_vocab_size, "anchor token is out of range")
    require(
        state.position + config.block_size <= config.max_position_embeddings,
        "DSpark proposal exceeds the configured position limit",
    )
    token_ids = [anchor_token_id] + [config.mask_token_id] * (config.block_size - 1)
    hidden = [list(weights.embedding[token_id]) for token_id in token_ids]
    positions = [state.position + offset for offset in range(config.block_size)]
    for index, layer in enumerate(weights.layers):
        hidden = _decoder_layer(
            hidden,
            state.keys[index],
            state.values[index],
            positions,
            layer,
            config,
        )
    hidden = [rms_norm(row, weights.norm, config.rms_norm_eps) for row in hidden]

    target_tokens = []
    draft_tokens = []
    confidences = []
    base_logits = []
    corrected_logits = []
    previous = anchor_token_id
    for slot in range(1, config.block_size):
        slot_hidden = hidden[slot]
        base = matvec(weights.lm_head, slot_hidden)
        previous_embedding = weights.markov_w1[previous]
        bias = matvec(weights.markov_w2, previous_embedding)
        corrected = [left + right for left, right in zip(base, bias)]
        draft_token = _argmax_lowest(corrected)
        target_token = int(weights.d2t[draft_token])
        confidence_logit = (
            math.fsum(
                left * right
                for left, right in zip(
                    weights.confidence_weight,
                    [*slot_hidden, *previous_embedding],
                )
            )
            + weights.confidence_bias
        )
        confidence = 1.0 / (1.0 + math.exp(-confidence_logit))
        base_logits.append(tuple(base))
        corrected_logits.append(tuple(corrected))
        draft_tokens.append(draft_token)
        target_tokens.append(target_token)
        confidences.append(confidence)
        previous = target_token
    return DSparkProposal(
        target_token_ids=tuple(target_tokens),
        draft_token_ids=tuple(draft_tokens),
        confidence=tuple(confidences),
        hidden_states=tuple(tuple(row) for row in hidden),
        base_logits=tuple(base_logits),
        corrected_logits=tuple(corrected_logits),
    )

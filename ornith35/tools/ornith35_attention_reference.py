#!/usr/bin/env python3
"""Dependency-free scalar oracle for Ornith-35 full-attention decode."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import ornith35_context as context


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


class AttentionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttentionError(message)


@dataclass(frozen=True)
class AttentionConfig:
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        require(self.hidden_size > 0, "hidden size must be positive")
        require(self.num_q_heads > 0 and self.num_kv_heads > 0, "head counts must be positive")
        require(
            self.num_q_heads % self.num_kv_heads == 0,
            "query-head count must be divisible by KV-head count",
        )
        require(self.head_dim > 0, "head dimension must be positive")
        require(
            0 < self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0,
            "rotary dimension must be positive, even, and no larger than the head",
        )
        require(self.rope_theta > 0.0, "RoPE theta must be positive")
        require(self.rms_norm_eps > 0.0, "RMS epsilon must be positive")

    @property
    def query_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class AttentionWeights:
    q_proj: Matrix
    k_proj: Matrix
    v_proj: Matrix
    o_proj: Matrix
    q_norm: Vector
    k_norm: Vector


@dataclass(frozen=True)
class AttentionState:
    keys: tuple[tuple[tuple[float, ...], ...], ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]
    context_profile: str = context.NATIVE_PROFILE_ID


def zeros_state(
    config: AttentionConfig,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> AttentionState:
    context.validate_range(context_profile, 0)
    empty = tuple(tuple() for _ in range(config.num_kv_heads))
    return AttentionState(
        keys=empty,
        values=empty,
        context_profile=context_profile,
    )


def _validate_matrix(matrix: Matrix, rows: int, columns: int, name: str) -> None:
    require(len(matrix) == rows, f"{name} row mismatch")
    require(all(len(row) == columns for row in matrix), f"{name} column mismatch")


def validate_weights(weights: AttentionWeights, config: AttentionConfig) -> None:
    _validate_matrix(
        weights.q_proj,
        config.query_dim * 2,
        config.hidden_size,
        "q_proj",
    )
    _validate_matrix(weights.k_proj, config.kv_dim, config.hidden_size, "k_proj")
    _validate_matrix(weights.v_proj, config.kv_dim, config.hidden_size, "v_proj")
    _validate_matrix(weights.o_proj, config.hidden_size, config.query_dim, "o_proj")
    require(len(weights.q_norm) == config.head_dim, "q_norm shape mismatch")
    require(len(weights.k_norm) == config.head_dim, "k_norm shape mismatch")


def state_length(state: AttentionState, config: AttentionConfig) -> int:
    profile = context.resolve_profile(state.context_profile)
    require(len(state.keys) == config.num_kv_heads, "key state head mismatch")
    require(len(state.values) == config.num_kv_heads, "value state head mismatch")
    lengths = {len(head) for head in (*state.keys, *state.values)}
    require(len(lengths) == 1, "KV state length mismatch")
    length = lengths.pop()
    require(
        all(
            len(vector) == config.head_dim
            for history in (*state.keys, *state.values)
            for vector in history
        ),
        "KV state head-width mismatch",
    )
    require(
        length <= profile.max_position_embeddings,
        "KV state exceeds its context profile",
    )
    return length


def _linear(weight: Matrix, vector: Vector) -> list[float]:
    return [math.fsum(left * right for left, right in zip(row, vector)) for row in weight]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _rms_norm(vector: Vector, weight: Vector, eps: float) -> list[float]:
    variance = math.fsum(value * value for value in vector) / len(vector)
    inverse = 1.0 / math.sqrt(variance + eps)
    return [value * inverse * (1.0 + scale) for value, scale in zip(vector, weight)]


def _apply_rope(
    vector: Vector,
    position: int,
    config: AttentionConfig,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> list[float]:
    context.validate_range(context_profile, position, 1)
    half = config.rotary_dim // 2
    inverse_frequencies, attention_factor = context.rope_parameters(
        context_profile,
        config.rotary_dim,
        config.rope_theta,
    )
    frequencies = [position * inverse for inverse in inverse_frequencies]
    angles = [*frequencies, *frequencies]
    rotary = vector[: config.rotary_dim]
    rotated = [-value for value in rotary[half:]] + list(rotary[:half])
    embedded = [
        value * (math.cos(angle) * attention_factor)
        + rotated_value * (math.sin(angle) * attention_factor)
        for value, rotated_value, angle in zip(rotary, rotated, angles)
    ]
    return [*embedded, *vector[config.rotary_dim :]]


def _softmax(values: Vector) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


def decode_step(
    hidden: Vector,
    state: AttentionState,
    weights: AttentionWeights,
    config: AttentionConfig,
) -> tuple[list[float], AttentionState]:
    """Evaluate one exact causal text position with Qwen3.5 GQA."""
    require(len(hidden) == config.hidden_size, "hidden-state width mismatch")
    validate_weights(weights, config)
    position = state_length(state, config)

    query_gate = _linear(weights.q_proj, hidden)
    queries = []
    gates = []
    stride = config.head_dim * 2
    for head in range(config.num_q_heads):
        start = head * stride
        queries.append(query_gate[start : start + config.head_dim])
        gates.append(query_gate[start + config.head_dim : start + stride])
    keys_flat = _linear(weights.k_proj, hidden)
    values_flat = _linear(weights.v_proj, hidden)
    keys = [
        keys_flat[index : index + config.head_dim]
        for index in range(0, config.kv_dim, config.head_dim)
    ]
    values = [
        values_flat[index : index + config.head_dim]
        for index in range(0, config.kv_dim, config.head_dim)
    ]
    queries = [
        _apply_rope(
            _rms_norm(query, weights.q_norm, config.rms_norm_eps),
            position,
            config,
            state.context_profile,
        )
        for query in queries
    ]
    keys = [
        _apply_rope(
            _rms_norm(key, weights.k_norm, config.rms_norm_eps),
            position,
            config,
            state.context_profile,
        )
        for key in keys
    ]
    next_keys = tuple((*state.keys[head], tuple(keys[head])) for head in range(config.num_kv_heads))
    next_values = tuple(
        (*state.values[head], tuple(values[head])) for head in range(config.num_kv_heads)
    )

    groups = config.num_q_heads // config.num_kv_heads
    scaling = config.head_dim**-0.5
    attended = []
    for head in range(config.num_q_heads):
        kv_head = head // groups
        scores = [
            math.fsum(left * right for left, right in zip(queries[head], key)) * scaling
            for key in next_keys[kv_head]
        ]
        probabilities = _softmax(scores)
        attended.extend(
            math.fsum(
                probability * value[index]
                for probability, value in zip(probabilities, next_values[kv_head])
            )
            * _sigmoid(gates[head][index])
            for index in range(config.head_dim)
        )

    return _linear(weights.o_proj, attended), AttentionState(
        keys=next_keys,
        values=next_values,
        context_profile=state.context_profile,
    )

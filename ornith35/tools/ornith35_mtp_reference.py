#!/usr/bin/env python3
"""Dependency-free scalar oracle for the Ornith-35 Qwen3.5 MTP sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import ornith35_attention_reference as attention
from ornith35_moe_reference import MoEConfig


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


class MTPError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MTPError(message)


@dataclass(frozen=True)
class MTPConfig:
    hidden_size: int
    attention: attention.AttentionConfig
    moe: MoEConfig
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        require(self.hidden_size > 0, "MTP hidden size must be positive")
        require(
            self.attention.hidden_size == self.hidden_size
            and self.moe.hidden_size == self.hidden_size,
            "MTP component hidden-size mismatch",
        )
        require(self.rms_norm_eps > 0.0, "MTP RMS epsilon must be positive")


PRODUCTION_CONFIG = MTPConfig(
    hidden_size=2048,
    attention=attention.AttentionConfig(
        hidden_size=2048,
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        rope_theta=10_000_000.0,
    ),
    moe=MoEConfig(
        hidden_size=2048,
        intermediate_size=512,
        num_experts=256,
        top_k=8,
    ),
)


@dataclass(frozen=True)
class DenseExpertWeights:
    gate: Matrix
    up: Matrix
    down: Matrix


@dataclass(frozen=True)
class DenseMoEWeights:
    router: Matrix
    experts: tuple[DenseExpertWeights, ...]
    shared_expert: DenseExpertWeights
    shared_gate: Vector


@dataclass(frozen=True)
class MTPWeights:
    fc: Matrix
    pre_fc_norm_embedding: Vector
    pre_fc_norm_hidden: Vector
    input_layernorm: Vector
    attention: attention.AttentionWeights
    moe: DenseMoEWeights
    post_attention_layernorm: Vector
    norm: Vector


@dataclass(frozen=True)
class DenseMoEResult:
    output: tuple[float, ...]
    selected_experts: tuple[int, ...]
    routing_weights: tuple[float, ...]


@dataclass(frozen=True)
class MTPResult:
    hidden: tuple[float, ...]
    state: attention.AttentionState
    selected_experts: tuple[int, ...]
    routing_weights: tuple[float, ...]


def _validate_matrix(
    matrix: Matrix,
    rows: int,
    columns: int,
    name: str,
) -> None:
    require(len(matrix) == rows, f"{name} row mismatch")
    require(all(len(row) == columns for row in matrix), f"{name} column mismatch")


def _validate_expert(
    expert: DenseExpertWeights,
    config: MoEConfig,
    name: str,
) -> None:
    _validate_matrix(
        expert.gate,
        config.intermediate_size,
        config.hidden_size,
        f"{name} gate",
    )
    _validate_matrix(
        expert.up,
        config.intermediate_size,
        config.hidden_size,
        f"{name} up",
    )
    _validate_matrix(
        expert.down,
        config.hidden_size,
        config.intermediate_size,
        f"{name} down",
    )


def validate_moe_weights(weights: DenseMoEWeights, config: MoEConfig) -> None:
    _validate_matrix(
        weights.router,
        config.num_experts,
        config.hidden_size,
        "MTP router",
    )
    require(len(weights.experts) == config.num_experts, "MTP expert count mismatch")
    require(len(weights.shared_gate) == config.hidden_size, "MTP shared-gate mismatch")
    for index, expert in enumerate(weights.experts):
        _validate_expert(expert, config, f"MTP expert {index}")
    _validate_expert(weights.shared_expert, config, "MTP shared expert")


def validate_weights(weights: MTPWeights, config: MTPConfig) -> None:
    hidden = config.hidden_size
    _validate_matrix(weights.fc, hidden, hidden * 2, "MTP fc")
    for name in (
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ):
        require(len(getattr(weights, name)) == hidden, f"MTP {name} mismatch")
    attention.validate_weights(weights.attention, config.attention)
    validate_moe_weights(weights.moe, config.moe)


def _linear(weight: Matrix, vector: Vector) -> list[float]:
    return [math.fsum(left * right for left, right in zip(row, vector)) for row in weight]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _rms_norm(vector: Vector, weight: Vector, eps: float) -> list[float]:
    require(len(vector) == len(weight), "MTP RMSNorm width mismatch")
    mean_square = math.fsum(value * value for value in vector) / len(vector)
    inverse = 1.0 / math.sqrt(mean_square + eps)
    return [
        value * inverse * (1.0 + centered_weight)
        for value, centered_weight in zip(vector, weight)
    ]


def _expert(expert: DenseExpertWeights, hidden: Vector) -> list[float]:
    gate = _linear(expert.gate, hidden)
    up = _linear(expert.up, hidden)
    intermediate = [
        gate_value * _sigmoid(gate_value) * up_value
        for gate_value, up_value in zip(gate, up)
    ]
    return _linear(expert.down, intermediate)


def forward_moe(
    hidden: Vector,
    weights: DenseMoEWeights,
    config: MoEConfig,
) -> DenseMoEResult:
    require(len(hidden) == config.hidden_size, "MTP MoE input mismatch")
    validate_moe_weights(weights, config)
    logits = _linear(weights.router, hidden)
    maximum = max(logits)
    probabilities = [math.exp(value - maximum) for value in logits]
    denominator = math.fsum(probabilities)
    probabilities = [value / denominator for value in probabilities]
    selected = tuple(
        sorted(
            range(config.num_experts),
            key=lambda index: probabilities[index],
            reverse=True,
        )[: config.top_k]
    )
    selected_mass = math.fsum(probabilities[index] for index in selected)
    routing = tuple(probabilities[index] / selected_mass for index in selected)
    expert_outputs = [_expert(weights.experts[index], hidden) for index in selected]
    routed = [
        math.fsum(
            routing[slot] * expert_outputs[slot][column]
            for slot in range(config.top_k)
        )
        for column in range(config.hidden_size)
    ]
    shared = _expert(weights.shared_expert, hidden)
    shared_multiplier = _sigmoid(
        math.fsum(left * right for left, right in zip(weights.shared_gate, hidden))
    )
    return DenseMoEResult(
        output=tuple(
            left + right * shared_multiplier for left, right in zip(routed, shared)
        ),
        selected_experts=selected,
        routing_weights=routing,
    )


def forward_step(
    next_token_embedding: Vector,
    target_hidden: Vector,
    state: attention.AttentionState,
    weights: MTPWeights,
    config: MTPConfig,
) -> MTPResult:
    """Predict after pairing one target hidden row with the following token."""
    require(
        len(next_token_embedding) == config.hidden_size,
        "MTP token embedding mismatch",
    )
    require(len(target_hidden) == config.hidden_size, "MTP target hidden mismatch")
    validate_weights(weights, config)

    normalized_embedding = _rms_norm(
        next_token_embedding,
        weights.pre_fc_norm_embedding,
        config.rms_norm_eps,
    )
    normalized_hidden = _rms_norm(
        target_hidden,
        weights.pre_fc_norm_hidden,
        config.rms_norm_eps,
    )
    hidden = _linear(weights.fc, (*normalized_embedding, *normalized_hidden))

    attention_input = _rms_norm(
        hidden,
        weights.input_layernorm,
        config.rms_norm_eps,
    )
    mixed, next_state = attention.decode_step(
        attention_input,
        state,
        weights.attention,
        config.attention,
    )
    hidden = [left + right for left, right in zip(hidden, mixed)]
    moe_input = _rms_norm(
        hidden,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
    )
    moe_result = forward_moe(moe_input, weights.moe, config.moe)
    hidden = [left + right for left, right in zip(hidden, moe_result.output)]
    hidden = _rms_norm(hidden, weights.norm, config.rms_norm_eps)
    return MTPResult(
        hidden=tuple(hidden),
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
    )

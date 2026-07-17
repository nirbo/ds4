#!/usr/bin/env python3
"""Dependency-free packed-NVFP4 oracle for the Ornith-35 MoE block."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ornith35_nvfp4 import decode_e2m1, decode_e4m3fn


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


class MoEError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MoEError(message)


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int

    def __post_init__(self) -> None:
        require(self.hidden_size > 0 and self.hidden_size % 16 == 0, "hidden size must be block aligned")
        require(
            self.intermediate_size > 0 and self.intermediate_size % 16 == 0,
            "intermediate size must be block aligned",
        )
        require(self.num_experts > 0, "expert count must be positive")
        require(0 < self.top_k <= self.num_experts, "top-k is outside expert count")


@dataclass(frozen=True)
class PackedWeight:
    packed: tuple[tuple[int, ...], ...]
    scales: tuple[tuple[int, ...], ...]
    global_scale: float

    @property
    def rows(self) -> int:
        return len(self.packed)

    @property
    def columns(self) -> int:
        return len(self.packed[0]) * 2 if self.packed else 0

    def validate(self, rows: int, columns: int, name: str) -> None:
        require(self.rows == rows, f"{name} row mismatch")
        require(columns > 0 and columns % 16 == 0, f"{name} column alignment mismatch")
        require(
            all(len(row) == columns // 2 for row in self.packed),
            f"{name} packed-column mismatch",
        )
        require(len(self.scales) == rows, f"{name} scale-row mismatch")
        require(
            all(len(row) == columns // 16 for row in self.scales),
            f"{name} scale-column mismatch",
        )
        require(math.isfinite(self.global_scale) and self.global_scale > 0.0, f"{name} global scale")

    def matvec(self, vector: Vector) -> list[float]:
        require(len(vector) == self.columns, "packed matvec width mismatch")
        output = []
        for row in range(self.rows):
            output.append(
                math.fsum(
                    decode_e2m1(
                        self.packed[row][column // 2] >> 4
                        if column & 1
                        else self.packed[row][column // 2] & 0xF
                    )
                    * decode_e4m3fn(self.scales[row][column // 16])
                    / self.global_scale
                    * vector[column]
                    for column in range(self.columns)
                )
            )
        return output


@dataclass(frozen=True)
class ExpertWeights:
    gate: PackedWeight
    up: PackedWeight
    down: PackedWeight


@dataclass(frozen=True)
class MoEWeights:
    router: Matrix
    experts: tuple[ExpertWeights, ...]
    shared_expert: ExpertWeights
    shared_gate: Vector


@dataclass(frozen=True)
class MoEResult:
    output: tuple[float, ...]
    selected_experts: tuple[int, ...]
    routing_weights: tuple[float, ...]


def validate_weights(weights: MoEWeights, config: MoEConfig) -> None:
    require(len(weights.router) == config.num_experts, "router row mismatch")
    require(all(len(row) == config.hidden_size for row in weights.router), "router width mismatch")
    require(len(weights.experts) == config.num_experts, "expert count mismatch")
    require(len(weights.shared_gate) == config.hidden_size, "shared gate width mismatch")
    for index, expert in enumerate((*weights.experts, weights.shared_expert)):
        label = "shared" if index == config.num_experts else f"expert {index}"
        expert.gate.validate(config.intermediate_size, config.hidden_size, f"{label} gate")
        expert.up.validate(config.intermediate_size, config.hidden_size, f"{label} up")
        expert.down.validate(config.hidden_size, config.intermediate_size, f"{label} down")


def _linear(weight: Matrix, vector: Vector) -> list[float]:
    return [math.fsum(left * right for left, right in zip(row, vector)) for row in weight]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def _expert(expert: ExpertWeights, hidden: Vector) -> list[float]:
    gate = expert.gate.matvec(hidden)
    up = expert.up.matvec(hidden)
    intermediate = [_silu(left) * right for left, right in zip(gate, up)]
    return expert.down.matvec(intermediate)


def forward(hidden: Vector, weights: MoEWeights, config: MoEConfig) -> MoEResult:
    require(len(hidden) == config.hidden_size, "hidden-state width mismatch")
    validate_weights(weights, config)
    logits = _linear(weights.router, hidden)
    maximum = max(logits)
    probabilities = [math.exp(value - maximum) for value in logits]
    denominator = math.fsum(probabilities)
    probabilities = [value / denominator for value in probabilities]
    selected = tuple(
        sorted(range(config.num_experts), key=lambda index: probabilities[index], reverse=True)[
            : config.top_k
        ]
    )
    selected_mass = math.fsum(probabilities[index] for index in selected)
    routing = tuple(probabilities[index] / selected_mass for index in selected)

    expert_outputs = [_expert(weights.experts[index], hidden) for index in selected]
    routed = [
        math.fsum(routing[slot] * expert_outputs[slot][column] for slot in range(config.top_k))
        for column in range(config.hidden_size)
    ]
    shared = _expert(weights.shared_expert, hidden)
    shared_multiplier = _sigmoid(math.fsum(left * right for left, right in zip(weights.shared_gate, hidden)))
    output = tuple(left + right * shared_multiplier for left, right in zip(routed, shared))
    return MoEResult(output=output, selected_experts=selected, routing_weights=routing)

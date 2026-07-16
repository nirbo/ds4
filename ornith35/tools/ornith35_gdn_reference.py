#!/usr/bin/env python3
"""Dependency-free scalar oracle for Ornith-35 GatedDeltaNet decode."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


class GDNError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GDNError(message)


@dataclass(frozen=True)
class GDNConfig:
    hidden_size: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        require(self.hidden_size > 0, "hidden size must be positive")
        require(self.num_k_heads > 0, "key-head count must be positive")
        require(self.num_v_heads > 0, "value-head count must be positive")
        require(self.head_k_dim > 0 and self.head_v_dim > 0, "head dimensions must be positive")
        require(self.conv_kernel_size > 0, "convolution width must be positive")
        require(
            self.num_v_heads % self.num_k_heads == 0,
            "value-head count must be divisible by key-head count",
        )
        require(self.rms_norm_eps > 0.0, "RMS epsilon must be positive")

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim


@dataclass(frozen=True)
class GDNWeights:
    in_proj_qkv: Matrix
    in_proj_z: Matrix
    in_proj_b: Matrix
    in_proj_a: Matrix
    conv1d: Matrix
    dt_bias: Vector
    a_log: Vector
    norm: Vector
    out_proj: Matrix


@dataclass(frozen=True)
class GDNState:
    conv: tuple[tuple[float, ...], ...]
    recurrent: tuple[tuple[tuple[float, ...], ...], ...]


def zeros_state(config: GDNConfig) -> GDNState:
    return GDNState(
        conv=tuple(
            tuple(0.0 for _ in range(config.conv_kernel_size))
            for _ in range(config.conv_dim)
        ),
        recurrent=tuple(
            tuple(
                tuple(0.0 for _ in range(config.head_v_dim))
                for _ in range(config.head_k_dim)
            )
            for _ in range(config.num_v_heads)
        ),
    )


def _validate_matrix(matrix: Matrix, rows: int, columns: int, name: str) -> None:
    require(len(matrix) == rows, f"{name} row mismatch")
    require(all(len(row) == columns for row in matrix), f"{name} column mismatch")


def validate_weights(weights: GDNWeights, config: GDNConfig) -> None:
    _validate_matrix(weights.in_proj_qkv, config.conv_dim, config.hidden_size, "in_proj_qkv")
    _validate_matrix(weights.in_proj_z, config.value_dim, config.hidden_size, "in_proj_z")
    _validate_matrix(weights.in_proj_b, config.num_v_heads, config.hidden_size, "in_proj_b")
    _validate_matrix(weights.in_proj_a, config.num_v_heads, config.hidden_size, "in_proj_a")
    _validate_matrix(weights.conv1d, config.conv_dim, config.conv_kernel_size, "conv1d")
    require(len(weights.dt_bias) == config.num_v_heads, "dt_bias shape mismatch")
    require(len(weights.a_log) == config.num_v_heads, "A_log shape mismatch")
    require(len(weights.norm) == config.head_v_dim, "gated norm shape mismatch")
    _validate_matrix(weights.out_proj, config.hidden_size, config.value_dim, "out_proj")


def validate_state(state: GDNState, config: GDNConfig) -> None:
    require(len(state.conv) == config.conv_dim, "convolution state channel mismatch")
    require(
        all(len(row) == config.conv_kernel_size for row in state.conv),
        "convolution state width mismatch",
    )
    require(len(state.recurrent) == config.num_v_heads, "recurrent state head mismatch")
    require(
        all(len(head) == config.head_k_dim for head in state.recurrent),
        "recurrent state key width mismatch",
    )
    require(
        all(
            len(key_row) == config.head_v_dim
            for head in state.recurrent
            for key_row in head
        ),
        "recurrent state value width mismatch",
    )


def _linear(weight: Matrix, vector: Vector) -> list[float]:
    return [math.fsum(left * right for left, right in zip(row, vector)) for row in weight]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def _l2norm(vector: Vector, eps: float = 1e-6) -> list[float]:
    inverse = 1.0 / math.sqrt(math.fsum(value * value for value in vector) + eps)
    return [value * inverse for value in vector]


def decode_step(
    hidden: Vector,
    state: GDNState,
    weights: GDNWeights,
    config: GDNConfig,
) -> tuple[list[float], GDNState]:
    """Evaluate the exact one-token Transformers fallback equations."""
    require(len(hidden) == config.hidden_size, "hidden-state width mismatch")
    validate_state(state, config)
    validate_weights(weights, config)

    mixed = _linear(weights.in_proj_qkv, hidden)
    z = _linear(weights.in_proj_z, hidden)
    b = _linear(weights.in_proj_b, hidden)
    a = _linear(weights.in_proj_a, hidden)

    next_conv = []
    convolved = []
    for channel in range(config.conv_dim):
        row = tuple((*state.conv[channel][1:], mixed[channel]))
        next_conv.append(row)
        convolved.append(
            _silu(math.fsum(value * weight for value, weight in zip(row, weights.conv1d[channel])))
        )

    query_flat = convolved[: config.key_dim]
    key_flat = convolved[config.key_dim : config.key_dim * 2]
    value_flat = convolved[config.key_dim * 2 :]
    query_base = [
        _l2norm(query_flat[index : index + config.head_k_dim])
        for index in range(0, config.key_dim, config.head_k_dim)
    ]
    key_base = [
        _l2norm(key_flat[index : index + config.head_k_dim])
        for index in range(0, config.key_dim, config.head_k_dim)
    ]
    values = [
        value_flat[index : index + config.head_v_dim]
        for index in range(0, config.value_dim, config.head_v_dim)
    ]
    repeats = config.num_v_heads // config.num_k_heads
    queries = [query_base[head // repeats] for head in range(config.num_v_heads)]
    keys = [key_base[head // repeats] for head in range(config.num_v_heads)]
    query_scale = 1.0 / math.sqrt(config.head_k_dim)

    recurrent = []
    core_output = []
    for head in range(config.num_v_heads):
        decay = math.exp(
            -math.exp(weights.a_log[head]) * _softplus(a[head] + weights.dt_bias[head])
        )
        beta = _sigmoid(b[head])
        decayed = [
            [state.recurrent[head][key][value] * decay for value in range(config.head_v_dim)]
            for key in range(config.head_k_dim)
        ]
        memory = [
            math.fsum(decayed[key][value] * keys[head][key] for key in range(config.head_k_dim))
            for value in range(config.head_v_dim)
        ]
        delta = [(values[head][value] - memory[value]) * beta for value in range(config.head_v_dim)]
        updated = [
            [decayed[key][value] + keys[head][key] * delta[value] for value in range(config.head_v_dim)]
            for key in range(config.head_k_dim)
        ]
        recurrent.append(tuple(tuple(row) for row in updated))
        core_output.append(
            [
                math.fsum(updated[key][value] * queries[head][key] * query_scale for key in range(config.head_k_dim))
                for value in range(config.head_v_dim)
            ]
        )

    gated = []
    for head in range(config.num_v_heads):
        variance = math.fsum(value * value for value in core_output[head]) / config.head_v_dim
        inverse = 1.0 / math.sqrt(variance + config.rms_norm_eps)
        z_head = z[head * config.head_v_dim : (head + 1) * config.head_v_dim]
        gated.extend(
            core_output[head][index] * inverse * weights.norm[index] * _silu(z_head[index])
            for index in range(config.head_v_dim)
        )

    return _linear(weights.out_proj, gated), GDNState(
        conv=tuple(next_conv),
        recurrent=tuple(recurrent),
    )

#!/usr/bin/env python3
"""MLX port of Gefen's block-shared optimizer-state representation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from nemotron_metadata import require


UPSTREAM_REVISION = "704034f0d62871cc651a5ebae7b5547c55e0fc37"


def divisors(value: int, max_divisor: int | None = None) -> list[int]:
    small = []
    large = []
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate:
            continue
        if candidate != value and (max_divisor is None or candidate <= max_divisor):
            small.append(candidate)
        paired = value // candidate
        if candidate != paired and (max_divisor is None or paired <= max_divisor):
            large.append(paired)
    return small + large[::-1]


def largest_divisor_at_most(value: int, limit: int) -> int:
    require(value > 0 and limit > 0, "Gefen divisor arguments must be positive")
    if value <= limit:
        return value
    candidates = divisors(value, limit)
    require(candidates, "Gefen could not find a magnitude divisor")
    return candidates[-1]


def find_period(squared_gradient: np.ndarray, max_period: int | None = None) -> int:
    """Port upstream automatic block-period selection from its CPU backend."""

    values = np.asarray(squared_gradient, dtype=np.float64).reshape(-1)
    require(values.size > 0, "Gefen period input is empty")
    candidates = []
    for period in divisors(values.size, max_period):
        if values.size // period < 4:
            continue
        logs = np.log(values + 1e-12).reshape(-1, period)
        means = logs.mean(axis=1)
        score = float(np.abs(means[1:] - means[:-1]).mean())
        candidates.append((period, score))
    if not candidates:
        return 1
    local_maxima = []
    last_period = candidates[-1][0]
    for index in range(1, len(candidates) - 1):
        period, score = candidates[index]
        if (
            period > 8
            and period != last_period
            and score > candidates[index - 1][1]
            and score > candidates[index + 1][1]
        ):
            local_maxima.append((score, period))
    period = max(local_maxima)[1] if local_maxima else 1
    if period > 1024:
        period = min(
            candidate
            for candidate in divisors(values.size)
            if 1024 < candidate <= period and period % candidate == 0
        )
    return period


def normalized_histogram(
    gradients: dict[str, mx.array], periods: dict[str, int], bins: int = 4096
) -> np.ndarray:
    counts = np.zeros((bins,), dtype=np.int64)
    for name, gradient in gradients.items():
        flat = np.asarray(gradient, dtype=np.float32).reshape(-1)
        period = periods[name]
        blocks = flat.reshape(-1, period)
        scale = np.max(np.abs(blocks), axis=1, keepdims=True)
        normalized = np.divide(blocks, scale, out=np.zeros_like(blocks), where=scale > 0)
        indices = np.floor((normalized.reshape(-1) + 1.0) * (bins / 2)).astype(np.int64)
        np.clip(indices, 0, bins - 1, out=indices)
        counts += np.bincount(indices, minlength=bins)
    return counts


def weighted_lloyd_codebook(
    counts: np.ndarray, levels: int = 256, iterations: int = 32
) -> np.ndarray:
    """Solve the upstream histogram objective efficiently on Apple CPUs."""

    counts = np.asarray(counts, dtype=np.float64)
    require(counts.ndim == 1 and counts.sum() > 0, "Gefen histogram is empty")
    centers = -1.0 + (np.arange(counts.size, dtype=np.float64) + 0.5) * (2 / counts.size)
    cumulative = np.cumsum(counts)
    targets = (np.arange(levels, dtype=np.float64) + 0.5) * cumulative[-1] / levels
    codebook = centers[np.searchsorted(cumulative, targets)].copy()
    codebook[0] = -1.0
    codebook[-1] = 1.0
    for _ in range(iterations):
        boundaries = (codebook[:-1] + codebook[1:]) * 0.5
        assignments = np.searchsorted(boundaries, centers)
        weighted = np.bincount(assignments, weights=counts * centers, minlength=levels)
        weight = np.bincount(assignments, weights=counts, minlength=levels)
        updated = codebook.copy()
        np.divide(weighted, weight, out=updated, where=weight > 0)
        updated[0] = -1.0
        updated[-1] = 1.0
        updated = np.maximum.accumulate(updated)
        if np.max(np.abs(updated - codebook)) < 1e-7:
            codebook = updated
            break
        codebook = updated
    return codebook.astype(np.float32)


def nearest_codebook_indices(codebook: mx.array, values: mx.array) -> mx.array:
    """Eight-step parallel binary search for a sorted 256-entry codebook."""

    require(codebook.shape == (256,), "Gefen codebook must have 256 entries")
    low = mx.zeros(values.shape, dtype=mx.int32)
    high = mx.full(values.shape, 255, dtype=mx.int32)
    for _ in range(8):
        middle = (low + high) // 2
        move_right = values > codebook[middle]
        low = mx.where(move_right, mx.minimum(middle + 1, 255), low)
        high = mx.where(move_right, high, middle)
    right = low
    left = mx.maximum(right - 1, 0)
    choose_left = mx.abs(values - codebook[left]) <= mx.abs(values - codebook[right])
    return mx.where(choose_left, left, right).astype(mx.uint8)


@dataclass
class GefenState:
    period: int
    magnitude_period: int
    momentum_indices: mx.array
    momentum_magnitude: mx.array
    vmean: mx.array


class GefenMLX:
    """Unfused MLX Gefen with upstream update equations and Apple codebook solve."""

    def __init__(
        self,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        period_cap_by_shape: bool = False,
    ):
        require(learning_rate > 0 and eps > 0 and weight_decay >= 0, "invalid Gefen settings")
        require(all(0 <= beta < 1 for beta in betas), "invalid Gefen betas")
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.period_cap_by_shape = period_cap_by_shape
        self.step_count = 0
        self.codebook: mx.array | None = None
        self.states: dict[str, GefenState] = {}

    def initialize(self, parameters: dict[str, mx.array], gradients: dict[str, mx.array]) -> None:
        require(set(parameters) == set(gradients) and parameters, "Gefen parameter tree mismatch")
        periods = {}
        for name, gradient in gradients.items():
            max_period = max(gradient.shape) if self.period_cap_by_shape and gradient.ndim else None
            periods[name] = (
                1
                if gradient.size == 1
                else find_period(np.square(np.asarray(gradient, dtype=np.float32)), max_period)
            )
            require(gradient.size % periods[name] == 0, f"Gefen period does not divide {name}")
        self.codebook = mx.array(weighted_lloyd_codebook(normalized_histogram(gradients, periods)))
        for name, gradient in gradients.items():
            period = periods[name]
            magnitude_period = largest_divisor_at_most(gradient.size, 256) if period == 1 else period
            self.states[name] = GefenState(
                period=period,
                magnitude_period=magnitude_period,
                momentum_indices=mx.zeros(gradient.shape, dtype=mx.uint8),
                momentum_magnitude=mx.zeros((gradient.size // magnitude_period, 1)),
                vmean=mx.zeros((gradient.size // period, 1)),
            )
        mx.eval(self.codebook, *self.state_arrays())

    def state_arrays(self) -> list[mx.array]:
        return [
            value
            for state in self.states.values()
            for value in (state.momentum_indices, state.momentum_magnitude, state.vmean)
        ]

    def state_bytes(self) -> int:
        return sum(array.nbytes for array in self.state_arrays()) + (
            self.codebook.nbytes if self.codebook is not None else 0
        )

    def update(
        self, parameters: dict[str, mx.array], gradients: dict[str, mx.array]
    ) -> dict[str, mx.array]:
        if self.codebook is None:
            self.initialize(parameters, gradients)
        require(set(parameters) == set(self.states) == set(gradients), "Gefen update tree mismatch")
        self.step_count += 1
        correction1 = 1 - self.beta1**self.step_count
        correction2 = 1 - self.beta2**self.step_count
        updated = {}
        for name, parameter in parameters.items():
            gradient = gradients[name].astype(mx.float32)
            state = self.states[name]
            blocks = gradient.reshape(-1, state.period)
            state.vmean = self.beta2 * state.vmean + (1 - self.beta2) * mx.mean(
                mx.square(blocks), axis=1, keepdims=True
            )
            magnitude_repeat = state.magnitude_period
            magnitude = mx.repeat(state.momentum_magnitude, magnitude_repeat, axis=0).reshape(
                gradient.shape
            )
            momentum = self.codebook[state.momentum_indices.astype(mx.int32)] * magnitude
            momentum = self.beta1 * momentum + (1 - self.beta1) * gradient
            magnitude_blocks = momentum.reshape(-1, state.magnitude_period)
            state.momentum_magnitude = mx.max(mx.abs(magnitude_blocks), axis=1, keepdims=True)
            expanded_magnitude = mx.repeat(
                state.momentum_magnitude, state.magnitude_period, axis=0
            ).reshape(gradient.shape)
            normalized = mx.where(expanded_magnitude > 0, momentum / expanded_magnitude, 0.0)
            state.momentum_indices = nearest_codebook_indices(self.codebook, normalized)
            quantized_momentum = (
                self.codebook[state.momentum_indices.astype(mx.int32)] * expanded_magnitude
            )
            denominator = mx.repeat(
                mx.sqrt(state.vmean / correction2) + self.eps,
                state.period,
                axis=1,
            ).reshape(gradient.shape)
            decayed = parameter * (1 - self.learning_rate * self.weight_decay)
            updated[name] = decayed - self.learning_rate * (
                quantized_momentum / correction1
            ) / denominator
        mx.eval(*updated.values(), *self.state_arrays())
        return updated

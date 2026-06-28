#!/usr/bin/env python3
"""Reference 1-bit block quantization for Ornith expert experiments.

This is a correctness harness, not a production storage format.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class IQ1Vector:
    n: int
    block_size: int
    scales: tuple[float, ...]
    signs: bytes


def block_scale(values: list[float], importance: list[float] | None = None) -> float:
    if not values:
        return 0.0
    if importance is None:
        return sum(abs(v) for v in values) / len(values)
    if len(importance) != len(values):
        raise ValueError("importance length does not match values")
    weight_sum = sum(importance)
    if weight_sum <= 0.0:
        return sum(abs(v) for v in values) / len(values)
    return sum(w * abs(v) for v, w in zip(values, importance)) / weight_sum


def pack_signs(values: list[float]) -> bytes:
    out = bytearray((len(values) + 7) // 8)
    for i, value in enumerate(values):
        if value >= 0.0:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def sign_at(signs: bytes, i: int) -> float:
    return 1.0 if signs[i // 8] & (1 << (i % 8)) else -1.0


def quantize_iq1(
    values: list[float],
    block_size: int = 256,
    importance: list[float] | None = None,
) -> IQ1Vector:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if importance is not None and len(importance) != len(values):
        raise ValueError("importance length does not match values")
    if importance is not None and any(w < 0.0 for w in importance):
        raise ValueError("importance weights must be non-negative")
    scales = []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        weights = importance[start:start + block_size] if importance is not None else None
        scales.append(block_scale(block, weights))
    return IQ1Vector(
        n=len(values),
        block_size=block_size,
        scales=tuple(scales),
        signs=pack_signs(values),
    )


def dequantize_iq1(q: IQ1Vector) -> list[float]:
    out = []
    for i in range(q.n):
        out.append(q.scales[i // q.block_size] * sign_at(q.signs, i))
    return out


def dot_iq1(q: IQ1Vector, x: list[float]) -> float:
    if len(x) != q.n:
        raise ValueError("dot input length does not match quantized vector")
    total = 0.0
    for i, xi in enumerate(x):
        total += q.scales[i // q.block_size] * sign_at(q.signs, i) * xi
    return total


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dot input lengths differ")
    return sum(x * y for x, y in zip(a, b))


def mse(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("mse input lengths differ")
    if not a:
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def weighted_mse(a: list[float], b: list[float], importance: list[float]) -> float:
    if len(a) != len(b) or len(a) != len(importance):
        raise ValueError("weighted_mse input lengths differ")
    weight_sum = sum(importance)
    if weight_sum <= 0.0:
        return mse(a, b)
    return sum(w * (x - y) ** 2 for x, y, w in zip(a, b, importance)) / weight_sum


def demo(seed: int, n: int, block_size: int) -> dict[str, float]:
    rng = random.Random(seed)
    weights = [rng.gauss(0.0, 0.7) + 0.05 * math.sin(i) for i in range(n)]
    activations = [rng.gauss(0.0, 1.0) for _ in range(n)]
    importance = [x * x + 1e-6 for x in activations]
    q = quantize_iq1(weights, block_size)
    qw = quantize_iq1(weights, block_size, importance)
    restored = dequantize_iq1(q)
    restored_w = dequantize_iq1(qw)
    packed_dot = dot_iq1(q, activations)
    restored_dot = dot(restored, activations)
    return {
        "mse": mse(weights, restored),
        "weighted_mse": weighted_mse(weights, restored, importance),
        "weighted_scale_mse": weighted_mse(weights, restored_w, importance),
        "source_dot": dot(weights, activations),
        "quant_dot": packed_dot,
        "weighted_quant_dot": dot_iq1(qw, activations),
        "packed_vs_restored_dot_abs": abs(packed_dot - restored_dot),
        "bits_per_weight_without_scales": 1.0,
        "scale_count": float(len(q.scales)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stats = demo(args.seed, args.n, args.block_size)
    for name, value in stats.items():
        print(f"{name}: {value:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

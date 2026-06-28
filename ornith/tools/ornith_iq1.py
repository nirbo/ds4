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


def pack_signs(values: list[float]) -> bytes:
    out = bytearray((len(values) + 7) // 8)
    for i, value in enumerate(values):
        if value >= 0.0:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def sign_at(signs: bytes, i: int) -> float:
    return 1.0 if signs[i // 8] & (1 << (i % 8)) else -1.0


def quantize_iq1(values: list[float], block_size: int = 256) -> IQ1Vector:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    scales = []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        scale = sum(abs(v) for v in block) / len(block) if block else 0.0
        scales.append(scale)
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


def demo(seed: int, n: int, block_size: int) -> dict[str, float]:
    rng = random.Random(seed)
    weights = [rng.gauss(0.0, 0.7) + 0.05 * math.sin(i) for i in range(n)]
    activations = [rng.gauss(0.0, 1.0) for _ in range(n)]
    q = quantize_iq1(weights, block_size)
    restored = dequantize_iq1(q)
    packed_dot = dot_iq1(q, activations)
    restored_dot = dot(restored, activations)
    return {
        "mse": mse(weights, restored),
        "source_dot": dot(weights, activations),
        "quant_dot": packed_dot,
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

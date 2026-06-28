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


@dataclass(frozen=True)
class IQ1Matrix:
    rows: int
    cols: int
    row_vectors: tuple[IQ1Vector, ...]


@dataclass(frozen=True)
class TernaryVector:
    n: int
    block_size: int
    scales: tuple[float, ...]
    codes: bytes


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


def pack_codes(codes: list[int]) -> bytes:
    out = bytearray((len(codes) + 3) // 4)
    for i, code in enumerate(codes):
        out[i // 4] |= (code & 3) << ((i % 4) * 2)
    return bytes(out)


def code_at(codes: bytes, i: int) -> int:
    return (codes[i // 4] >> ((i % 4) * 2)) & 3


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


def quantize_ternary(
    values: list[float],
    block_size: int = 256,
    keep_fraction: float = 0.5,
    importance: list[float] | None = None,
) -> TernaryVector:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if importance is not None and len(importance) != len(values):
        raise ValueError("importance length does not match values")
    codes: list[int] = []
    scales = []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        weights = importance[start:start + block_size] if importance is not None else [1.0] * len(block)
        keep = max(1, round(len(block) * keep_fraction))
        order = sorted(range(len(block)), key=lambda i: abs(block[i]) * weights[i], reverse=True)
        kept = set(order[:keep])
        weight_sum = sum(weights[i] for i in kept)
        scale = sum(weights[i] * abs(block[i]) for i in kept) / weight_sum if weight_sum > 0 else block_scale(block)
        scales.append(scale)
        for i, value in enumerate(block):
            if i not in kept:
                codes.append(0)
            elif value >= 0.0:
                codes.append(1)
            else:
                codes.append(2)
    return TernaryVector(len(values), block_size, tuple(scales), pack_codes(codes))


def quantize_iq1_rows(
    rows: list[list[float]],
    block_size: int = 256,
    importance: list[float] | None = None,
) -> IQ1Matrix:
    if not rows:
        return IQ1Matrix(rows=0, cols=0, row_vectors=())
    cols = len(rows[0])
    if any(len(row) != cols for row in rows):
        raise ValueError("matrix rows must have the same length")
    return IQ1Matrix(
        rows=len(rows),
        cols=cols,
        row_vectors=tuple(quantize_iq1(row, block_size, importance) for row in rows),
    )


def dequantize_iq1(q: IQ1Vector) -> list[float]:
    out = []
    for i in range(q.n):
        out.append(q.scales[i // q.block_size] * sign_at(q.signs, i))
    return out


def dequantize_ternary(q: TernaryVector) -> list[float]:
    out = []
    for i in range(q.n):
        code = code_at(q.codes, i)
        scale = q.scales[i // q.block_size]
        out.append(scale if code == 1 else -scale if code == 2 else 0.0)
    return out


def matvec_iq1(q: IQ1Matrix, x: list[float]) -> list[float]:
    if len(x) != q.cols:
        raise ValueError("matvec input length does not match matrix columns")
    return [dot_iq1(row, x) for row in q.row_vectors]


def dot_iq1(q: IQ1Vector, x: list[float]) -> float:
    if len(x) != q.n:
        raise ValueError("dot input length does not match quantized vector")
    total = 0.0
    for i, xi in enumerate(x):
        total += q.scales[i // q.block_size] * sign_at(q.signs, i) * xi
    return total


def dot_ternary(q: TernaryVector, x: list[float]) -> float:
    if len(x) != q.n:
        raise ValueError("dot input length does not match quantized vector")
    total = 0.0
    for i, xi in enumerate(x):
        code = code_at(q.codes, i)
        if code:
            total += q.scales[i // q.block_size] * (1.0 if code == 1 else -1.0) * xi
    return total


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dot input lengths differ")
    return sum(x * y for x, y in zip(a, b))


def matvec(rows: list[list[float]], x: list[float]) -> list[float]:
    return [dot(row, x) for row in rows]


def bits_per_weight(q: IQ1Vector, scale_bits: int = 16) -> float:
    if q.n == 0:
        return 0.0
    return (q.n + len(q.scales) * scale_bits) / q.n


def ternary_entropy_bits_per_weight(q: TernaryVector, scale_bits: int = 16) -> float:
    if q.n == 0:
        return 0.0
    return math.log2(3.0) + len(q.scales) * scale_bits / q.n


def ternary_packed_bits_per_weight(q: TernaryVector, scale_bits: int = 16) -> float:
    if q.n == 0:
        return 0.0
    return 2.0 + len(q.scales) * scale_bits / q.n


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
    qt = quantize_ternary(weights, block_size, 0.5, importance)
    matrix = [weights[i:i + block_size] for i in range(0, min(n, block_size * 4), block_size)]
    qm = quantize_iq1_rows(matrix, block_size, importance[:block_size])
    mat_x = activations[:block_size]
    qy = matvec_iq1(qm, mat_x)
    dy = matvec([dequantize_iq1(row) for row in qm.row_vectors], mat_x)
    restored = dequantize_iq1(q)
    restored_w = dequantize_iq1(qw)
    restored_t = dequantize_ternary(qt)
    packed_dot = dot_iq1(q, activations)
    restored_dot = dot(restored, activations)
    ternary_dot = dot_ternary(qt, activations)
    return {
        "mse": mse(weights, restored),
        "weighted_mse": weighted_mse(weights, restored, importance),
        "weighted_scale_mse": weighted_mse(weights, restored_w, importance),
        "ternary_weighted_mse": weighted_mse(weights, restored_t, importance),
        "source_dot": dot(weights, activations),
        "quant_dot": packed_dot,
        "weighted_quant_dot": dot_iq1(qw, activations),
        "ternary_quant_dot": ternary_dot,
        "packed_vs_restored_dot_abs": abs(packed_dot - restored_dot),
        "ternary_packed_vs_restored_dot_abs": abs(ternary_dot - dot(restored_t, activations)),
        "matvec_packed_vs_restored_max_abs": max((abs(a - b) for a, b in zip(qy, dy)), default=0.0),
        "bits_per_weight_without_scales": 1.0,
        "bits_per_weight_f16_scales": bits_per_weight(q, 16),
        "bits_per_weight_f32_scales": bits_per_weight(q, 32),
        "ternary_entropy_bits_f16_scales": ternary_entropy_bits_per_weight(qt, 16),
        "ternary_packed_bits_f16_scales": ternary_packed_bits_per_weight(qt, 16),
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

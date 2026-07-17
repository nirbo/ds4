#!/usr/bin/env python3
"""Paired real-layer benchmark for exact batched MoE optimizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_moe as moe
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT, require_verified_source


def evaluate(result: moe.MLXMoEResult) -> None:
    mx.eval(result.output, result.selected_experts, result.routing_weights)
    mx.synchronize()


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.mean(retained)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    weights = moe.load_layer(require_verified_source(args.root), args.layer)
    mx.random.seed(args.seed)
    hidden = mx.random.normal(
        (args.tokens, moe.PRODUCTION_CONFIG.hidden_size),
        dtype=mx.float32,
    ).astype(mx.bfloat16)
    mx.eval(hidden)
    print(
        "moe-dense-bench-ready "
        f"feature={args.feature} layer={args.layer} tokens={args.tokens} "
        f"setup_s={time.perf_counter() - started:.3f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f}",
        flush=True,
    )

    baseline_times: list[float] = []
    candidate_times: list[float] = []
    baseline = None
    candidate = None
    mx.reset_peak_memory()
    for step in range(args.warmup + args.rounds):
        order = (False, True) if step % 2 == 0 else (True, False)
        elapsed: dict[bool, float] = {}
        for candidate_enabled in order:
            token_tiled = (
                candidate_enabled if args.feature == "token-tiled" else True
            )
            direct_bf16 = (
                candidate_enabled if args.feature == "direct-bf16" else False
            )
            begin = time.perf_counter()
            result = moe.forward_batch(
                hidden,
                weights,
                token_tiled_shared=token_tiled,
                direct_bf16_inputs=direct_bf16,
            )
            evaluate(result)
            elapsed[candidate_enabled] = time.perf_counter() - begin
            if candidate_enabled:
                candidate = result
            else:
                baseline = result
        if step >= args.warmup:
            baseline_times.append(elapsed[False])
            candidate_times.append(elapsed[True])

    require(baseline is not None and candidate is not None, "benchmark produced no result")
    checks = (
        mx.array_equal(baseline.output, candidate.output),
        mx.array_equal(baseline.selected_experts, candidate.selected_experts),
        mx.array_equal(baseline.routing_weights, candidate.routing_weights),
    )
    mx.eval(*checks)
    require(all(bool(check.item()) for check in checks), "batched MoE parity failed")
    baseline_mean = trimmed_mean(baseline_times)
    candidate_mean = trimmed_mean(candidate_times)
    print(
        "moe-dense-bench-result "
        f"feature={args.feature} exact_tensors=3 "
        f"baseline_ms={baseline_mean * 1000:.3f} "
        f"candidate_ms={candidate_mean * 1000:.3f} "
        f"speedup={baseline_mean / candidate_mean:.4f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--feature",
        choices=("token-tiled", "direct-bf16"),
        default="token-tiled",
    )
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.layer in range(40), "invalid MoE layer")
        require(args.tokens in (8, 16, 32, 64, 128), "invalid benchmark token count")
        require(args.warmup >= 1, "benchmark warmup must be positive")
        require(args.rounds >= 4, "benchmark rounds must be at least four")
        run(args)
    except (MoEError, OSError, ValueError) as exc:
        print(f"MoE dense benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

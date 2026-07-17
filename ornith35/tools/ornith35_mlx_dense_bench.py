#!/usr/bin/env python3
"""Paired real-layer benchmark for exact token-tiled GDN projections."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_gdn as gdn
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT, require_verified_source


def evaluate(result: tuple[mx.array, gdn.MLXGDNState]) -> None:
    output, state = result
    mx.eval(output, state.conv, state.recurrent)
    mx.synchronize()


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.mean(retained)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    weights = gdn.load_layer(require_verified_source(args.root), args.layer)
    mx.random.seed(args.seed)
    hidden = mx.random.normal(
        (args.tokens, gdn.PRODUCTION_CONFIG.hidden_size),
        dtype=mx.float32,
    ).astype(mx.bfloat16)
    state = gdn.zeros_state(gdn.PRODUCTION_CONFIG)
    mx.eval(hidden, state.conv, state.recurrent)
    print(
        "dense-bench-ready "
        f"layer={args.layer} tokens={args.tokens} "
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
        for tiled in order:
            begin = time.perf_counter()
            result = gdn.prefill_chunk(
                hidden,
                state,
                weights,
                token_tiled_projections=tiled,
            )
            evaluate(result)
            elapsed[tiled] = time.perf_counter() - begin
            if tiled:
                candidate = result
            else:
                baseline = result
        if step >= args.warmup:
            baseline_times.append(elapsed[False])
            candidate_times.append(elapsed[True])

    require(baseline is not None and candidate is not None, "benchmark produced no result")
    checks = (
        mx.array_equal(baseline[0], candidate[0]),
        mx.array_equal(baseline[1].conv, candidate[1].conv),
        mx.array_equal(baseline[1].recurrent, candidate[1].recurrent),
    )
    mx.eval(*checks)
    require(all(bool(check.item()) for check in checks), "token-tiled parity failed")
    baseline_mean = trimmed_mean(baseline_times)
    candidate_mean = trimmed_mean(candidate_times)
    print(
        "dense-bench-result "
        "exact_tensors=3 "
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
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.layer in range(40) and args.layer % 4 != 3, "invalid GDN layer")
        require(args.tokens in (8, 16, 32, 64, 128), "invalid benchmark token count")
        require(args.warmup >= 1, "benchmark warmup must be positive")
        require(args.rounds >= 4, "benchmark rounds must be at least four")
        run(args)
    except (MoEError, OSError, ValueError) as exc:
        print(f"dense benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

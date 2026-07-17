#!/usr/bin/env python3
"""Paired real-layer benchmark for exact token-tiled attention projections."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_attention as attention
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT, require_verified_source


def evaluate(
    result: tuple[
        mx.array,
        attention.MLXAttentionState | attention.MLXLinearAttentionState,
    ],
) -> None:
    output, state = result
    mx.eval(output, state.keys, state.values)
    mx.synchronize()


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.mean(retained)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    weights = attention.load_layer(require_verified_source(args.root), args.layer)
    mx.random.seed(args.seed)
    hidden = mx.random.normal(
        (args.tokens, attention.PRODUCTION_CONFIG.hidden_size),
        dtype=mx.float32,
    ).astype(mx.bfloat16)
    state = attention.MLXAttentionState(
        keys=mx.random.normal(
            (
                attention.PRODUCTION_CONFIG.num_kv_heads,
                args.prefix,
                attention.PRODUCTION_CONFIG.head_dim,
            ),
            dtype=mx.float32,
        ).astype(mx.bfloat16),
        values=mx.random.normal(
            (
                attention.PRODUCTION_CONFIG.num_kv_heads,
                args.prefix,
                attention.PRODUCTION_CONFIG.head_dim,
            ),
            dtype=mx.float32,
        ).astype(mx.bfloat16),
    )
    rope = attention.make_text_rope(
        args.prefix,
        args.tokens,
        attention.PRODUCTION_CONFIG,
        mx.bfloat16,
    )
    mx.eval(hidden, state.keys, state.values, rope.cosine, rope.sine)
    print(
        "attention-dense-bench-ready "
        f"layer={args.layer} prefix={args.prefix} tokens={args.tokens} "
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
            result = attention.prefill_chunk(
                hidden,
                state,
                weights,
                use_steel=False,
                rope=rope,
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
        mx.array_equal(baseline[1].keys, candidate[1].keys),
        mx.array_equal(baseline[1].values, candidate[1].values),
    )
    mx.eval(*checks)
    require(all(bool(check.item()) for check in checks), "attention tiling parity failed")
    baseline_mean = trimmed_mean(baseline_times)
    candidate_mean = trimmed_mean(candidate_times)
    print(
        "attention-dense-bench-result "
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
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.layer in range(40) and args.layer % 4 == 3, "invalid attention layer")
        require(args.prefix >= 0, "benchmark prefix must be nonnegative")
        require(args.tokens in (8, 16, 32, 64, 128), "invalid benchmark token count")
        require(args.warmup >= 1, "benchmark warmup must be positive")
        require(args.rounds >= 4, "benchmark rounds must be at least four")
        run(args)
    except (MoEError, OSError, ValueError) as exc:
        print(f"attention dense benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired exactness and speed benchmark for final-token prompt chunks."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import mlx.core as mx

import ornith35_mlx_model as model
from ornith35_mlx_state_prefill_bench import (
    parse_prefixes,
    state_arrays,
    synthetic_state,
    trimmed_mean,
)
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT


def require_exact(
    expected: model.TextModelChunkResult,
    actual: model.TextModelResult,
) -> None:
    checks = [
        mx.array_equal(expected.hidden[-1], actual.hidden),
        mx.array_equal(expected.logits, actual.logits),
    ]
    checks.extend(
        mx.array_equal(left[-1], right)
        for left, right in zip(expected.selected_experts, actual.selected_experts)
    )
    checks.extend(
        mx.array_equal(left[-1], right)
        for left, right in zip(expected.routing_weights, actual.routing_weights)
    )
    expected_state = state_arrays(expected.state)
    actual_state = state_arrays(actual.state)
    require(len(expected_state) == 80, "unexpected persistent tensor count")
    checks.extend(
        mx.array_equal(left, right)
        for left, right in zip(expected_state, actual_state)
    )
    mx.eval(*checks)
    mismatches = [index for index, check in enumerate(checks) if not bool(check.item())]
    require(not mismatches, f"final-token parity mismatch at tensors {mismatches}")


def run_prefix(
    weights: model.TextModelWeights,
    prefix: int,
    chunk: int,
    warmup: int,
    rounds: int,
) -> None:
    source = synthetic_state(weights, prefix)
    capacity = prefix + chunk * (warmup + rounds)
    full = model.start_linear_decode_session(
        weights,
        source,
        capacity,
        model.PRODUCTION_CONFIG,
    )
    final = model.start_linear_decode_session(
        weights,
        source,
        capacity,
        model.PRODUCTION_CONFIG,
    )
    tokens = tuple(9707 + index % 17 for index in range(chunk))
    full_times: list[float] = []
    final_times: list[float] = []
    expected = None
    actual = None
    mx.reset_peak_memory()
    for step in range(warmup + rounds):
        final_first = step % 2 == 1
        if final_first:
            started = time.perf_counter()
            actual = model.prefill_linear_session_final_chunk(
                tokens,
                final,
                use_steel=False,
            )
            final_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            expected = model.prefill_linear_session_chunk(
                tokens,
                full,
                project_logits=True,
                use_steel=False,
            )
            full_elapsed = time.perf_counter() - started
        else:
            started = time.perf_counter()
            expected = model.prefill_linear_session_chunk(
                tokens,
                full,
                project_logits=True,
                use_steel=False,
            )
            full_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            actual = model.prefill_linear_session_final_chunk(
                tokens,
                final,
                use_steel=False,
            )
            final_elapsed = time.perf_counter() - started
        require(
            isinstance(expected, model.TextModelChunkResult),
            "full final benchmark result mismatch",
        )
        if step >= warmup:
            full_times.append(full_elapsed)
            final_times.append(final_elapsed)
    require(expected is not None and actual is not None, "benchmark produced no result")
    require_exact(expected, actual)
    full_mean = trimmed_mean(full_times)
    final_mean = trimmed_mean(final_times)
    print(
        "final-prefill-bench "
        f"prefix={prefix} chunk={chunk} rounds={rounds} exact_checks=162 "
        f"full_ms={full_mean * 1000:.3f} "
        f"final_ms={final_mean * 1000:.3f} "
        f"full_tok_s={chunk / full_mean:.3f} "
        f"final_tok_s={chunk / final_mean:.3f} "
        f"speedup={full_mean / final_mean:.4f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    del expected, actual, full, final, source
    gc.collect()
    mx.clear_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prefixes", default="0,4096,65536,131072")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument(
        "--mapped-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.chunk in (8, 16, 32, 64, 128), "invalid benchmark chunk")
        require(args.warmup >= 1, "benchmark warmup must be positive")
        require(args.rounds >= 4, "benchmark rounds must be at least four")
        started = time.perf_counter()
        weights = model.load_text_model(
            args.root,
            map_embedding=args.mapped_embedding,
        )
        print(
            "final-prefill-model-ready "
            f"load_s={time.perf_counter() - started:.3f} "
            f"mapped_embedding={str(args.mapped_embedding).lower()} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        for prefix in parse_prefixes(args.prefixes):
            run_prefix(weights, prefix, args.chunk, args.warmup, args.rounds)
    except (MoEError, OSError, ValueError) as exc:
        print(f"final-token prefill benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

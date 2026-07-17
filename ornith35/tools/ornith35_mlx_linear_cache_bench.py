#!/usr/bin/env python3
"""Paired exactness and decode benchmark for immutable versus linear K/V."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT


def synthetic_state(
    weights: model.TextModelWeights,
    prefix: int,
) -> model.TextModelState:
    """Create a deterministic cache with production shapes but no prefill cost."""
    require(prefix >= 0, "benchmark prefix must be nonnegative")
    base = model.initial_state(weights, model.PRODUCTION_CONFIG)
    states: list[model.LayerState] = []
    arrays: list[mx.array] = []
    for kind, state in zip(model.PRODUCTION_CONFIG.layer_types, base.layers):
        if kind == model.LAYER_GDN:
            require(isinstance(state, gdn.MLXGDNState), "invalid GDN state")
            states.append(state)
        else:
            cache = attention.MLXAttentionState(
                keys=mx.zeros((2, prefix, 256), dtype=mx.bfloat16),
                values=mx.zeros((2, prefix, 256), dtype=mx.bfloat16),
            )
            states.append(cache)
            arrays.extend((cache.keys, cache.values))
    if arrays:
        mx.eval(*arrays)
        mx.synchronize()
    return model.TextModelState(position=prefix, layers=tuple(states))


def result_arrays(
    result: model.TextModelResult,
) -> tuple[mx.array, ...]:
    arrays = [result.logits, result.hidden]
    arrays.extend(result.selected_experts)
    arrays.extend(result.routing_weights)
    for state in result.state.layers:
        if isinstance(state, gdn.MLXGDNState):
            arrays.extend((state.conv, state.recurrent))
        elif isinstance(state, attention.MLXAttentionState):
            arrays.extend((state.keys, state.values))
        else:
            require(
                isinstance(state, attention.MLXLinearAttentionState),
                "invalid benchmark state",
            )
            arrays.extend(
                (
                    state.keys[:, : state.position, :],
                    state.values[:, : state.position, :],
                )
            )
    return tuple(arrays)


def require_exact(
    expected: model.TextModelResult,
    actual: model.TextModelResult,
) -> None:
    expected_arrays = result_arrays(expected)
    actual_arrays = result_arrays(actual)
    require(len(expected_arrays) == 162, "unexpected benchmark tensor count")
    checks = [
        mx.array_equal(left, right)
        for left, right in zip(expected_arrays, actual_arrays)
    ]
    mx.eval(*checks)
    mismatches = [index for index, check in enumerate(checks) if not bool(check.item())]
    require(not mismatches, f"linear K/V parity mismatch at tensors {mismatches}")


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.mean(retained)


def run_prefix(
    weights: model.TextModelWeights,
    prefix: int,
    token_id: int,
    warmup: int,
    rounds: int,
) -> None:
    state = synthetic_state(weights, prefix)
    immutable = model.start_decode_session(weights, state, model.PRODUCTION_CONFIG)
    linear = model.start_linear_decode_session(
        weights,
        state,
        prefix + warmup + rounds,
        model.PRODUCTION_CONFIG,
    )
    immutable_times: list[float] = []
    linear_times: list[float] = []
    expected = None
    actual = None
    mx.reset_peak_memory()
    for step in range(warmup + rounds):
        linear_first = step % 2 == 1
        if linear_first:
            started = time.perf_counter()
            actual = model.forward_linear_session_token(token_id, linear)
            linear_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            expected, immutable = model.forward_session_token(token_id, immutable)
            model.evaluate_result(expected)
            immutable_elapsed = time.perf_counter() - started
        else:
            started = time.perf_counter()
            expected, immutable = model.forward_session_token(token_id, immutable)
            model.evaluate_result(expected)
            immutable_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            actual = model.forward_linear_session_token(token_id, linear)
            linear_elapsed = time.perf_counter() - started
        if step >= warmup:
            immutable_times.append(immutable_elapsed)
            linear_times.append(linear_elapsed)
    require(expected is not None and actual is not None, "benchmark produced no result")
    require_exact(expected, actual)
    immutable_mean = trimmed_mean(immutable_times)
    linear_mean = trimmed_mean(linear_times)
    print(
        "linear-kv-bench "
        f"prefix={prefix} rounds={rounds} exact_tensors=162 "
        f"immutable_ms={immutable_mean * 1000:.3f} "
        f"linear_ms={linear_mean * 1000:.3f} "
        f"immutable_tok_s={1.0 / immutable_mean:.3f} "
        f"linear_tok_s={1.0 / linear_mean:.3f} "
        f"speedup={immutable_mean / linear_mean:.4f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    del expected, actual, immutable, linear, state
    gc.collect()
    mx.clear_cache()


def parse_prefixes(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise MoEError("prefixes must be comma-separated integers") from exc
    require(values and all(value >= 0 for value in values), "invalid benchmark prefixes")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prefixes", default="0,4096,16384")
    parser.add_argument("--token", type=int, default=9707)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.warmup >= 2, "benchmark warmup must be at least two")
        require(args.rounds >= 10, "benchmark rounds must be at least ten")
        require(
            0 <= args.token < model.PRODUCTION_CONFIG.vocab_size,
            "benchmark token is out of range",
        )
        started = time.perf_counter()
        weights = model.load_text_model(args.root)
        print(
            "linear-kv-model-ready "
            f"load_s={time.perf_counter() - started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        for prefix in parse_prefixes(args.prefixes):
            run_prefix(weights, prefix, args.token, args.warmup, args.rounds)
    except (MoEError, OSError, ValueError) as exc:
        print(f"linear K/V benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

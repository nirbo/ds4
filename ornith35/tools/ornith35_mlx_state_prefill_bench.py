#!/usr/bin/env python3
"""Paired exactness and speed benchmark for state-only prompt chunks."""

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
    require(prefix >= 0, "benchmark prefix must be nonnegative")
    base = model.initial_state(weights, model.PRODUCTION_CONFIG)
    states: list[model.LayerState] = []
    arrays: list[mx.array] = []
    for kind, state in zip(model.PRODUCTION_CONFIG.layer_types, base.layers):
        if kind == model.LAYER_GDN:
            require(isinstance(state, gdn.MLXGDNState), "invalid GDN state")
            states.append(state)
            continue
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


def state_arrays(state: model.TextModelState) -> tuple[mx.array, ...]:
    arrays: list[mx.array] = []
    for layer_state in state.layers:
        if isinstance(layer_state, gdn.MLXGDNState):
            arrays.extend((layer_state.conv, layer_state.recurrent))
            continue
        require(
            isinstance(layer_state, attention.MLXLinearAttentionState),
            "benchmark requires linear attention state",
        )
        arrays.extend(
            (
                layer_state.keys[:, : layer_state.position, :],
                layer_state.values[:, : layer_state.position, :],
            )
        )
    return tuple(arrays)


def require_exact(
    expected: model.TextModelState,
    actual: model.TextModelState,
) -> None:
    require(expected.position == actual.position, "state-only position mismatch")
    expected_arrays = state_arrays(expected)
    actual_arrays = state_arrays(actual)
    require(len(expected_arrays) == 80, "unexpected persistent tensor count")
    checks = [
        mx.array_equal(left, right)
        for left, right in zip(expected_arrays, actual_arrays)
    ]
    mx.eval(*checks)
    mismatches = [index for index, check in enumerate(checks) if not bool(check.item())]
    require(not mismatches, f"state-only parity mismatch at tensors {mismatches}")


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.mean(retained)


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
    state_only = model.start_linear_decode_session(
        weights,
        source,
        capacity,
        model.PRODUCTION_CONFIG,
    )
    tokens = tuple(9707 + index % 17 for index in range(chunk))
    full_times: list[float] = []
    state_times: list[float] = []
    mx.reset_peak_memory()
    for step in range(warmup + rounds):
        state_first = step % 2 == 1
        if state_first:
            started = time.perf_counter()
            model.prefill_linear_session_state_chunk(
                tokens,
                state_only,
                use_steel=False,
            )
            state_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            model.prefill_linear_session_chunk(
                tokens,
                full,
                project_logits=False,
                use_steel=False,
            )
            full_elapsed = time.perf_counter() - started
        else:
            started = time.perf_counter()
            model.prefill_linear_session_chunk(
                tokens,
                full,
                project_logits=False,
                use_steel=False,
            )
            full_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            model.prefill_linear_session_state_chunk(
                tokens,
                state_only,
                use_steel=False,
            )
            state_elapsed = time.perf_counter() - started
        if step >= warmup:
            full_times.append(full_elapsed)
            state_times.append(state_elapsed)
    require_exact(full.state, state_only.state)
    full_mean = trimmed_mean(full_times)
    state_mean = trimmed_mean(state_times)
    print(
        "state-prefill-bench "
        f"prefix={prefix} chunk={chunk} rounds={rounds} exact_tensors=80 "
        f"full_ms={full_mean * 1000:.3f} "
        f"state_ms={state_mean * 1000:.3f} "
        f"full_tok_s={chunk / full_mean:.3f} "
        f"state_tok_s={chunk / state_mean:.3f} "
        f"speedup={full_mean / state_mean:.4f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    del full, state_only, source
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
            "state-prefill-model-ready "
            f"load_s={time.perf_counter() - started:.3f} "
            f"mapped_embedding={str(args.mapped_embedding).lower()} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        for prefix in parse_prefixes(args.prefixes):
            run_prefix(weights, prefix, args.chunk, args.warmup, args.rounds)
    except (MoEError, OSError, ValueError) as exc:
        print(f"state-only prefill benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

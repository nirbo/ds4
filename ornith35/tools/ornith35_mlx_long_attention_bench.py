#!/usr/bin/env python3
"""Verify and time exact long-prefix Ornith-35 attention batching."""

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


def deterministic(shape: tuple[int, ...], phase: float) -> mx.array:
    size = 1
    for dimension in shape:
        size *= dimension
    values = mx.sin(mx.arange(size, dtype=mx.float32) * 0.013 + phase) * 0.125
    return values.reshape(shape).astype(mx.bfloat16)


def make_state(prefix: int, nonzero: bool) -> attention.MLXAttentionState:
    if not nonzero:
        keys = mx.zeros((2, prefix, 256), dtype=mx.bfloat16)
        values = mx.zeros((2, prefix, 256), dtype=mx.bfloat16)
    else:
        positions = mx.arange(prefix, dtype=mx.float32)[None, :, None]
        dimensions = mx.arange(256, dtype=mx.float32)[None, None, :]
        heads = mx.arange(2, dtype=mx.float32)[:, None, None]
        keys = (
            mx.sin(positions * 0.0013 + dimensions * 0.017 + heads * 0.31)
            * 0.125
        ).astype(mx.bfloat16)
        values = (
            mx.cos(positions * 0.0017 + dimensions * 0.011 + heads * 0.23)
            * 0.125
        ).astype(mx.bfloat16)
    mx.eval(keys, values)
    return attention.MLXAttentionState(keys=keys, values=values)


def timed(operation) -> tuple[float, mx.array, attention.MLXAttentionState]:
    started = time.perf_counter()
    output, state = operation()
    mx.eval(output, state.keys, state.values)
    mx.synchronize()
    return time.perf_counter() - started, output, state


def run_prefix(
    hidden: mx.array,
    state: attention.MLXAttentionState,
    weights: attention.MLXAttentionWeights,
    prefix: int,
    warmup: int,
    rounds: int,
    feature: str,
) -> None:
    source_times = []
    candidate_times = []
    source_output = None
    candidate_output = None
    source_state = None
    candidate_state = None
    for index in range(warmup + rounds):
        source_exact = feature in ("fused-softmax-value", "key-tiled-scores")
        operations = (
            (
                "source",
                lambda: attention.prefill_chunk(
                    hidden,
                    state,
                    weights,
                    use_steel=False,
                    exact_long_prefill=source_exact,
                    fused_long_softmax_value=False,
                    key_tiled_long_scores=False,
                ),
            ),
            (
                "candidate",
                lambda: attention.prefill_chunk(
                    hidden,
                    state,
                    weights,
                    use_steel=False,
                    exact_long_prefill=True,
                    fused_long_softmax_value=feature == "fused-softmax-value",
                    key_tiled_long_scores=feature in (
                        "key-tiled-scores",
                        "key-tiled-vs-standard",
                    ),
                ),
            ),
        )
        if index % 2:
            operations = tuple(reversed(operations))
        samples = {}
        for name, operation in operations:
            samples[name] = timed(operation)
        if index >= warmup:
            source_times.append(samples["source"][0])
            candidate_times.append(samples["candidate"][0])
        _, source_output, source_state = samples["source"]
        _, candidate_output, candidate_state = samples["candidate"]

    require(source_output is not None and candidate_output is not None, "missing outputs")
    require(source_state is not None and candidate_state is not None, "missing states")
    output_differences = mx.sum(source_output != candidate_output)
    max_abs = mx.max(
        mx.abs(source_output.astype(mx.float32) - candidate_output.astype(mx.float32))
    )
    key_equal = mx.array_equal(source_state.keys, candidate_state.keys)
    value_equal = mx.array_equal(source_state.values, candidate_state.values)
    mx.eval(output_differences, max_abs, key_equal, value_equal)
    source_median = statistics.median(source_times)
    candidate_median = statistics.median(candidate_times)
    print(
        "long-attention "
        f"feature={feature} prefix={prefix} "
        f"chunk={hidden.shape[0]} rounds={rounds} "
        f"different={int(output_differences.item())}/{source_output.size} "
        f"max_abs={float(max_abs.item()):.9g} "
        f"kv_exact={str(bool(key_equal.item()) and bool(value_equal.item())).lower()} "
        f"source_ms={source_median * 1000:.3f} "
        f"candidate_ms={candidate_median * 1000:.3f} "
        f"speedup={source_median / candidate_median:.4f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )


def parse_prefixes(text: str, minimum: int) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as exc:
        raise MoEError("prefixes must be comma-separated integers") from exc
    require(
        values
        and all(value >= minimum for value in values),
        "prefixes must select the exact long-attention path",
    )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--feature",
        choices=(
            "exact-batching",
            "fused-softmax-value",
            "key-tiled-scores",
            "key-tiled-vs-standard",
        ),
        default="exact-batching",
    )
    parser.add_argument("--layer", type=int, default=39)
    parser.add_argument("--prefixes", default="106496,131072,262016")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--nonzero-cache", action="store_true")
    parser.add_argument(
        "--force-exact",
        action="store_true",
        help="benchmark exact kernels below the production crossover",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.layer in range(3, 40, 4), "layer must use full attention")
        require(args.chunk in (8, 16, 32, 64, 128), "invalid chunk")
        require(args.warmup >= 1 and args.rounds >= 3, "insufficient timing rounds")
        minimum = 0 if args.force_exact else attention.EXACT_LONG_PREFILL_MIN_PREFIX
        prefixes = parse_prefixes(args.prefixes, minimum)
        if args.force_exact and args.feature != "key-tiled-vs-standard":
            attention.EXACT_LONG_PREFILL_MIN_PREFIX = 0
        weights = attention.load_layer(require_verified_source(args.root), args.layer)
        hidden = deterministic((args.chunk, 2048), args.layer * 0.17)
        mx.eval(hidden)
        for prefix in prefixes:
            run_prefix(
                hidden,
                make_state(prefix, args.nonzero_cache),
                weights,
                prefix,
                args.warmup,
                args.rounds,
                args.feature,
            )
            mx.clear_cache()
    except (MoEError, OSError, ValueError) as exc:
        print(f"long attention benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired synthetic crossover benchmark for BF16 and packed K9-MSE K/V."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_linear_cache as linear_cache
import ornith35_mlx_turboquant_cache as packed_cache
from ornith35_moe_reference import MoEError, require


def bf16_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
) -> mx.array:
    history = keys.shape[1]
    grouped_queries = queries.reshape(2, 8, 1, 256)
    grouped_keys = mx.swapaxes(keys, 1, 2)[:, None, :, :]
    scores = mx.matmul(grouped_queries, grouped_keys).reshape(16, history)
    probabilities = mx.softmax(scores.astype(mx.float32) * (256**-0.5), axis=-1)
    grouped_probabilities = probabilities.astype(mx.bfloat16).reshape(2, 8, 1, history)
    return mx.matmul(grouped_probabilities, values[:, None, :, :]).reshape(16, 256)


def make_inputs(length: int) -> tuple[mx.array, mx.array, mx.array, packed_cache.MLXPackedMSEState]:
    require(length > 0, "benchmark length must be positive")
    queries = mx.full((16, 256), 0.03125, dtype=mx.bfloat16)
    keys = mx.full((2, length, 256), 0.015625, dtype=mx.bfloat16)
    values = mx.full((2, length, 256), 0.0625, dtype=mx.bfloat16)
    head = min(length, packed_cache.PRODUCTION_EXACT_HEAD_TOKENS)
    remaining = length - head
    tail = min(remaining, packed_cache.PRODUCTION_EXACT_TAIL_TOKENS)
    history = remaining - tail
    packed_shape = (2, history, packed_cache.PACKED_DIM)
    norm_shape = (2, history, 1)
    state = packed_cache.MLXPackedMSEState(
        packed_keys=mx.full(packed_shape, 0x87, dtype=mx.uint8),
        key_norms=mx.ones(norm_shape, dtype=packed_cache.PRODUCTION_NORM_DTYPE),
        packed_values=mx.full(packed_shape, 0x78, dtype=mx.uint8),
        value_norms=mx.ones(norm_shape, dtype=packed_cache.PRODUCTION_NORM_DTYPE),
        exact_head_keys=keys[:, :head],
        exact_head_values=values[:, :head],
        exact_keys=keys[:, -tail:] if tail else keys[:, :0],
        exact_values=values[:, -tail:] if tail else values[:, :0],
    )
    packed_cache.validate_state(state)
    mx.eval(
        queries,
        keys,
        values,
        state.packed_keys,
        state.key_norms,
        state.packed_values,
        state.value_norms,
        state.exact_head_keys,
        state.exact_head_values,
    )
    return queries, keys, values, state


def timed(operation) -> tuple[float, mx.array]:
    started = time.perf_counter()
    output = operation()
    mx.eval(output)
    mx.synchronize()
    return time.perf_counter() - started, output


def measure_append_costs(warmup: int, rounds: int) -> tuple[float, float]:
    capacity = warmup + rounds + 1
    key_update = mx.full((2, 1, 256), 0.0234375, dtype=mx.bfloat16)
    value_update = mx.full((2, 1, 256), 0.0703125, dtype=mx.bfloat16)
    bf16_keys = mx.zeros((2, capacity, 256), dtype=mx.bfloat16)
    bf16_values = mx.zeros((2, capacity, 256), dtype=mx.bfloat16)
    empty = mx.zeros((2, 0, 256), dtype=mx.bfloat16)
    packed = packed_cache.linearize_bf16_kv(empty, empty, capacity)
    mx.eval(bf16_keys, bf16_values, key_update, value_update)
    mx.synchronize()
    bf16_times: list[float] = []
    packed_times: list[float] = []
    for index in range(warmup + rounds):
        operations = (
            (
                "bf16",
                lambda: linear_cache.append_kv_bf16(
                    bf16_keys,
                    bf16_values,
                    key_update,
                    value_update,
                    index,
                ),
            ),
            (
                "packed",
                lambda: packed_cache.advance_linear_state(
                    packed,
                    key_update,
                    value_update,
                ),
            ),
        )
        if index % 2:
            operations = tuple(reversed(operations))
        elapsed = {}
        next_packed = None
        for name, operation in operations:
            started = time.perf_counter()
            result = operation()
            if name == "bf16":
                bf16_keys, bf16_values = result
                mx.eval(bf16_keys, bf16_values)
            else:
                next_packed = result
                mx.eval(
                    next_packed.packed_keys,
                    next_packed.key_norms,
                    next_packed.packed_values,
                    next_packed.value_norms,
                    next_packed.exact_head_keys,
                    next_packed.exact_head_values,
                    next_packed.exact_keys,
                    next_packed.exact_values,
                )
            mx.synchronize()
            elapsed[name] = time.perf_counter() - started
        require(next_packed is not None, "packed append produced no state")
        packed = next_packed
        if index >= warmup:
            bf16_times.append(elapsed["bf16"])
            packed_times.append(elapsed["packed"])
    return statistics.median(bf16_times), statistics.median(packed_times)


def run_length(
    length: int,
    warmup: int,
    rounds: int,
    bf16_append: float,
    packed_append: float,
) -> None:
    queries, keys, values, state = make_inputs(length)
    bf16_times: list[float] = []
    packed_times: list[float] = []
    bf16_output = None
    packed_output = None
    mx.reset_peak_memory()
    for index in range(warmup + rounds):
        operations = (
            ("bf16", lambda: bf16_attention(queries, keys, values)),
            ("packed", lambda: packed_cache.packed_attention(queries, state)[0]),
        )
        if index % 2:
            operations = tuple(reversed(operations))
        samples = {}
        for name, operation in operations:
            samples[name] = timed(operation)
        if index >= warmup:
            bf16_times.append(samples["bf16"][0])
            packed_times.append(samples["packed"][0])
        bf16_output = samples["bf16"][1]
        packed_output = samples["packed"][1]

    require(bf16_output is not None and packed_output is not None, "missing benchmark output")
    finite = mx.all(mx.isfinite(bf16_output)) & mx.all(mx.isfinite(packed_output))
    mx.eval(finite)
    require(bool(finite.item()), "benchmark produced non-finite output")
    bf16_median = statistics.median(bf16_times)
    packed_median = statistics.median(packed_times)
    bf16_total = bf16_median + bf16_append
    packed_total = packed_median + packed_append
    bf16_bytes = keys.nbytes + values.nbytes
    packed_bytes = packed_cache.stored_bytes(state)
    print(
        "turboquant-cache-bench "
        f"tokens={length} rounds={rounds} "
        f"bf16_ms={bf16_median * 1000:.4f} "
        f"packed_ms={packed_median * 1000:.4f} "
        f"bf16_append_ms={bf16_append * 1000:.4f} "
        f"packed_encode_append_ms={packed_append * 1000:.4f} "
        f"attention_speedup={bf16_median / packed_median:.4f} "
        f"full_speedup={bf16_total / packed_total:.4f} "
        f"bf16_mib={bf16_bytes / 2**20:.6f} "
        f"packed_mib={packed_bytes / 2**20:.6f} "
        f"compression={bf16_bytes / packed_bytes:.4f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    del queries, keys, values, state, bf16_output, packed_output
    gc.collect()
    mx.clear_cache()


def parse_lengths(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as exc:
        raise MoEError("tokens must be comma-separated integers") from exc
    require(values and all(value > 0 for value in values), "invalid benchmark token lengths")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="16,64,256,1024,4096,16384,65536")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.warmup >= 1, "benchmark warmup must be positive")
        require(args.rounds >= 4, "benchmark rounds must be at least four")
        bf16_append, packed_append = measure_append_costs(args.warmup, args.rounds)
        print(
            "turboquant-cache-maintenance "
            f"rounds={args.rounds} "
            f"bf16_append_ms={bf16_append * 1000:.4f} "
            f"packed_encode_append_ms={packed_append * 1000:.4f}",
            flush=True,
        )
        for length in parse_lengths(args.tokens):
            run_length(
                length,
                args.warmup,
                args.rounds,
                bf16_append,
                packed_append,
            )
    except (MoEError, OSError, ValueError) as exc:
        print(f"TurboQuant cache benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

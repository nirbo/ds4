#!/usr/bin/env python3
"""Benchmark exact resident Nemotron multi-token verification blocks."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, require
from nemotron_mlx_resident import ResidentModel, preflight


def parse_block_sizes(value: str) -> list[int]:
    try:
        sizes = sorted({int(item) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block sizes must be comma-separated integers") from exc
    if not sizes or sizes[0] < 2:
        raise argparse.ArgumentTypeError("verification block sizes must be at least 2")
    return sizes


def compare_logits(actual: mx.array, reference: mx.array) -> dict[str, float | bool]:
    require(actual.shape == reference.shape, "verification/reference logit shape mismatch")
    difference = actual.astype(mx.float32) - reference.astype(mx.float32)
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(reference.astype(mx.float32))))
    return {
        "relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "max_abs": float(mx.max(mx.abs(difference))),
        "top1_equal": (
            mx.argmax(actual, axis=-1).tolist()
            == mx.argmax(reference, axis=-1).tolist()
        ),
    }


def sequential_logits(model: ResidentModel, token_ids: list[int]) -> mx.array:
    return mx.stack([model.logits(token_id) for token_id in token_ids])


def timed_call(callable_) -> tuple[mx.array, float]:
    started = time.perf_counter()
    result = callable_()
    if isinstance(result, tuple):
        arrays = []
        for value in result:
            if isinstance(value, mx.array):
                arrays.append(value)
            elif isinstance(value, dict):
                snapshots = value.values()
                if value and all(isinstance(key, int) for key in value):
                    first = next(iter(value.values()))
                    if isinstance(first, dict):
                        snapshots = (
                            snapshot
                            for prefix in value.values()
                            for snapshot in prefix.values()
                        )
                for snapshot in snapshots:
                    if snapshot[0] == "arrays":
                        arrays.extend(snapshot[1])
                    else:
                        arrays.extend(snapshot[1:3])
        mx.eval(*arrays)
    else:
        mx.eval(result)
    mx.synchronize()
    return result, time.perf_counter() - started


def benchmark_block(
    model: ResidentModel,
    initial_snapshot: dict[int, tuple],
    token_ids: list[int],
    repeats: int,
) -> dict[str, float | bool]:
    model.restore(initial_snapshot)
    reference = sequential_logits(model, token_ids)
    mx.eval(reference)
    model.restore(initial_snapshot)

    batched = model.logits_sequence(token_ids)
    mx.eval(batched)
    comparison = compare_logits(batched, reference)
    model.restore(initial_snapshot)

    rollback = model.logits(token_ids[0])
    mx.eval(rollback)
    rollback_error = float(mx.max(mx.abs(rollback - reference[0])))
    model.restore(initial_snapshot)

    _, _, accepted_snapshot = model.verify_sequence(token_ids, 0)
    model.restore(accepted_snapshot)
    captured_continuation = model.logits(token_ids[1])
    model.restore(initial_snapshot)
    model.logits(token_ids[0])
    incremental_continuation = model.logits(token_ids[1])
    mx.eval(captured_continuation, incremental_continuation)
    captured_cache_error = float(
        mx.max(mx.abs(captured_continuation - incremental_continuation))
    )
    model.restore(initial_snapshot)

    prefix_cache_error = 0.0
    if len(token_ids) > 2:
        prefix_indices = tuple(range(len(token_ids) - 1))
        _, _, prefix_snapshots = model.verify_sequence_prefixes(
            token_ids,
            prefix_indices,
        )
        for prefix_index in prefix_indices:
            model.restore(prefix_snapshots[prefix_index])
            captured_continuation = model.logits(token_ids[prefix_index + 1])
            model.restore(initial_snapshot)
            incremental_continuation = None
            for token_id in token_ids[: prefix_index + 2]:
                incremental_continuation = model.logits(token_id)
            require(incremental_continuation is not None, "empty prefix verification")
            mx.eval(captured_continuation, incremental_continuation)
            prefix_cache_error = max(
                prefix_cache_error,
                float(mx.max(mx.abs(captured_continuation - incremental_continuation))),
            )
        model.restore(initial_snapshot)

    # Compile and page in this exact block shape before timed samples.
    model.logits_sequence(token_ids)
    model.restore(initial_snapshot)

    sequential_seconds = []
    batched_seconds = []
    captured_seconds = []
    for _ in range(repeats):
        model.restore(initial_snapshot)
        _, elapsed = timed_call(lambda: sequential_logits(model, token_ids))
        sequential_seconds.append(elapsed)
        model.restore(initial_snapshot)
        _, elapsed = timed_call(lambda: model.logits_sequence(token_ids))
        batched_seconds.append(elapsed)
        model.restore(initial_snapshot)
        _, elapsed = timed_call(lambda: model.verify_sequence(token_ids, 0))
        captured_seconds.append(elapsed)
    model.restore(initial_snapshot)

    sequential_median = statistics.median(sequential_seconds)
    batched_median = statistics.median(batched_seconds)
    return {
        **comparison,
        "rollback_max_abs": rollback_error,
        "captured_cache_max_abs": captured_cache_error,
        "prefix_cache_max_abs": prefix_cache_error,
        "sequential_ms": sequential_median * 1000,
        "batched_ms": batched_median * 1000,
        "speedup": sequential_median / batched_median,
        "captured_ms": statistics.median(captured_seconds) * 1000,
        "verified_tokens_per_second": len(token_ids) / batched_median,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="2+2=")
    parser.add_argument(
        "--block-sizes",
        type=parse_block_sizes,
        default=parse_block_sizes("2,4,8"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--margin-gib", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        result = preflight(args.model_dir, args.margin_gib)
        print(
            f"verify-preflight safe={result['safe_to_attempt']} "
            f"required_gib={result['required_gib']:.3f}"
        )
        require(result["safe_to_attempt"], "Metal wired cap is too low for resident verification")
        previous_limit = mx.set_wired_limit(result["required_bytes"])
        mx.set_cache_limit(256 * 2**20)
        try:
            started = time.perf_counter()
            model = ResidentModel(args.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            require(prompt_ids, "prompt encoded to no tokens")
            logits = None
            for token_id in prompt_ids:
                logits = model.logits(token_id)
            require(logits is not None, "prompt prefill produced no logits")
            initial_snapshot = model.snapshot()

            generated = []
            for position in range(max(args.block_sizes)):
                token_id = int(mx.argmax(logits))
                generated.append(token_id)
                if position + 1 < max(args.block_sizes):
                    logits = model.logits(token_id)
            model.restore(initial_snapshot)
            print(
                f"verify-ready prompt_tokens={len(prompt_ids)} "
                f"setup_seconds={time.perf_counter() - started:.3f} "
                f"draft_ids={','.join(str(token_id) for token_id in generated)}"
            )

            for block_size in args.block_sizes:
                metrics = benchmark_block(
                    model,
                    initial_snapshot,
                    generated[:block_size],
                    args.repeats,
                )
                print(
                    f"verify-block size={block_size} sequential_ms={metrics['sequential_ms']:.3f} "
                    f"batched_ms={metrics['batched_ms']:.3f} speedup={metrics['speedup']:.3f} "
                    f"captured_ms={metrics['captured_ms']:.3f} "
                    f"verified_tok_s={metrics['verified_tokens_per_second']:.3f} "
                    f"relative_l2={metrics['relative_l2']:.9g} max_abs={metrics['max_abs']:.9g} "
                    f"rollback_max_abs={metrics['rollback_max_abs']:.9g} "
                    f"captured_cache_max_abs={metrics['captured_cache_max_abs']:.9g} "
                    f"prefix_cache_max_abs={metrics['prefix_cache_max_abs']:.9g} "
                    f"top1_equal={metrics['top1_equal']}"
                )
                require(metrics["top1_equal"], f"block-{block_size} top-1 mismatch")
                require(
                    metrics["relative_l2"] <= 2e-5,
                    f"block-{block_size} logit drift exceeds tolerance",
                )
                require(
                    metrics["rollback_max_abs"] <= 2e-4,
                    f"block-{block_size} rollback drift exceeds tolerance",
                )
                require(
                    metrics["captured_cache_max_abs"] <= 2e-4,
                    f"block-{block_size} accepted-cache capture drift exceeds tolerance",
                )
                require(
                    metrics["prefix_cache_max_abs"] <= 2e-4,
                    f"block-{block_size} prefix-cache capture drift exceeds tolerance",
                )
            print(
                f"verify-done active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f}"
            )
        finally:
            mx.set_wired_limit(previous_limit)
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron verification benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

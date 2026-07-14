#!/usr/bin/env python3
"""Profile the resident Nemotron target path at layer boundaries."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_mamba import mamba_sequence_exact
from nemotron_mlx_resident import ResidentModel, preflight


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = sorted({int(item) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block sizes must be comma-separated integers") from exc
    if not sizes or sizes[0] < 1 or sizes[-1] > 8:
        raise argparse.ArgumentTypeError("block sizes must be between 1 and 8")
    return sizes


def evaluate(array: mx.array) -> float:
    started = time.perf_counter()
    mx.eval(array)
    mx.synchronize()
    return time.perf_counter() - started


def profile_once(
    model: ResidentModel,
    snapshot: dict[int, tuple],
    token_ids: list[int],
) -> dict:
    model.restore(snapshot)
    tokens = mx.array(token_ids, dtype=mx.int32)
    started = time.perf_counter()
    x = model.embeddings[tokens].astype(mx.float32).reshape(
        1, len(token_ids), model.hidden_size
    )
    embedding_seconds = evaluate(x)
    layers = []
    for layer, (kind, block) in enumerate(zip(model.pattern, model.blocks)):
        started_layer = time.perf_counter()
        if kind == "M":
            cache = model.caches[layer]
            mask = create_ssm_mask(x, cache)
            x = (
                mamba_sequence_exact(block, x, cache, mask)
                if len(token_ids) > 1
                else block(x, mask=mask, cache=cache)
            )
        elif kind == "*":
            cache = model.caches[layer]
            x = block(x, mask=create_attention_mask(x, cache), cache=cache)
        else:
            x = block(x)
        mx.eval(x)
        mx.synchronize()
        layers.append((layer, kind, time.perf_counter() - started_layer))

    normalized = mx.fast.rms_norm(
        x,
        model.final_norm,
        model.config["layer_norm_epsilon"],
    )
    norm_seconds = evaluate(normalized)
    logits = model.lm_head(normalized).reshape(len(token_ids), -1)
    head_seconds = evaluate(logits)
    return {
        "wall_seconds": time.perf_counter() - started,
        "embedding_seconds": embedding_seconds,
        "layers": layers,
        "norm_seconds": norm_seconds,
        "head_seconds": head_seconds,
    }


def median_profile(samples: list[dict]) -> dict:
    require(samples, "profile has no measured samples")
    layer_count = len(samples[0]["layers"])
    require(
        all(len(sample["layers"]) == layer_count for sample in samples),
        "profile layer count changed between samples",
    )
    layers = []
    for position in range(layer_count):
        identities = {(sample["layers"][position][0], sample["layers"][position][1]) for sample in samples}
        require(len(identities) == 1, "profile layer identity changed between samples")
        layer, kind = identities.pop()
        layers.append(
            (
                layer,
                kind,
                statistics.median(sample["layers"][position][2] for sample in samples),
            )
        )
    return {
        "wall_seconds": statistics.median(sample["wall_seconds"] for sample in samples),
        "embedding_seconds": statistics.median(sample["embedding_seconds"] for sample in samples),
        "layers": layers,
        "norm_seconds": statistics.median(sample["norm_seconds"] for sample in samples),
        "head_seconds": statistics.median(sample["head_seconds"] for sample in samples),
    }


def timed_sequence(
    model: ResidentModel,
    snapshot: dict[int, tuple],
    token_ids: list[int],
) -> float:
    model.restore(snapshot)
    started = time.perf_counter()
    logits = model.logits_sequence(token_ids)
    mx.eval(logits)
    mx.synchronize()
    return time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="Complete this Python function:\n\ndef binary_search(values, target):\n")
    parser.add_argument("--block-sizes", type=parse_sizes, default=parse_sizes("1,2,3"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--margin-gib", type=float, default=0.5)
    parser.add_argument("--top-layers", type=int, default=12)
    parser.add_argument(
        "--trace-repeats",
        type=int,
        default=0,
        help="run only a sleep-delimited unsynchronized window for Metal tracing",
    )
    parser.add_argument("--paged-embeddings", action="store_true")
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    parser.add_argument("--compile-mamba", action="store_true")
    parser.add_argument("--expert-top-k", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        require(args.top_layers > 0, "top layer count must be positive")
        require(args.trace_repeats >= 0, "trace repeats cannot be negative")
        native_top_k = load_json(args.model_dir / "config.json")["num_experts_per_tok"]
        require(
            args.expert_top_k is None or 1 <= args.expert_top_k <= native_top_k,
            f"expert top-k must be between 1 and {native_top_k}",
        )
        require(
            args.trace_repeats == 0 or len(args.block_sizes) == 1,
            "trace mode requires exactly one block size",
        )
        memory = preflight(
            args.model_dir,
            args.margin_gib,
            paged_embeddings=args.paged_embeddings,
        )
        print(
            f"profile-preflight safe={memory['safe_to_attempt']} "
            f"required_gib={memory['required_gib']:.3f} cap_gib={memory['effective_cap_gib']:.3f}",
            flush=True,
        )
        require(memory["safe_to_attempt"], "Metal wired cap is too low for resident profiling")
        previous_limit = mx.set_wired_limit(memory["effective_cap_bytes"])
        mx.set_cache_limit(256 * 2**20)
        try:
            load_started = time.perf_counter()
            model = ResidentModel(
                args.model_dir,
                paged_embeddings=args.paged_embeddings,
                embedding_cache_rows=args.embedding_cache_rows,
                compile_mamba=args.compile_mamba,
            )
            if args.expert_top_k is not None:
                model.set_expert_top_k(args.expert_top_k)
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            require(prompt_ids, "prompt encoded to no tokens")
            logits = None
            for token_id in prompt_ids:
                logits = model.logits(token_id)
            require(logits is not None, "prompt prefill produced no logits")
            snapshot = model.snapshot()
            generated = []
            for position in range(max(args.block_sizes)):
                token_id = int(mx.argmax(logits))
                generated.append(token_id)
                if position + 1 < max(args.block_sizes):
                    logits = model.logits(token_id)
            model.restore(snapshot)
            print(
                f"profile-ready prompt_tokens={len(prompt_ids)} setup_seconds={time.perf_counter() - load_started:.3f} "
                f"active_gib={mx.get_active_memory() / 2**30:.3f} peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                flush=True,
            )

            if args.trace_repeats:
                block_size = args.block_sizes[0]
                token_ids = generated[:block_size]
                timed_sequence(model, snapshot, token_ids)
                print(
                    f"profile-trace-idle phase=before block={block_size} repeats={args.trace_repeats}",
                    flush=True,
                )
                time.sleep(1.0)
                trace_started = time.perf_counter()
                samples = [
                    timed_sequence(model, snapshot, token_ids)
                    for _ in range(args.trace_repeats)
                ]
                print(
                    f"profile-trace-window block={block_size} repeats={args.trace_repeats} "
                    f"elapsed_ms={(time.perf_counter() - trace_started) * 1000:.3f} "
                    f"median_ms={statistics.median(samples) * 1000:.3f}",
                    flush=True,
                )
                time.sleep(1.0)
                return 0

            for block_size in args.block_sizes:
                token_ids = generated[:block_size]
                timed_sequence(model, snapshot, token_ids)
                sequence_samples = [
                    timed_sequence(model, snapshot, token_ids)
                    for _ in range(args.repeats)
                ]
                profile_once(model, snapshot, token_ids)
                profile = median_profile(
                    [profile_once(model, snapshot, token_ids) for _ in range(args.repeats)]
                )
                by_kind = defaultdict(float)
                for _, kind, seconds in profile["layers"]:
                    by_kind[kind] += seconds
                sequence_ms = statistics.median(sequence_samples) * 1000
                profile_ms = profile["wall_seconds"] * 1000
                print(
                    f"profile-block size={block_size} sequence_ms={sequence_ms:.3f} "
                    f"synchronized_ms={profile_ms:.3f} sync_inflation={profile_ms / sequence_ms:.3f} "
                    f"embedding_ms={profile['embedding_seconds'] * 1000:.3f} "
                    f"mamba_ms={by_kind['M'] * 1000:.3f} moe_ms={by_kind['E'] * 1000:.3f} "
                    f"attention_ms={by_kind['*'] * 1000:.3f} norm_ms={profile['norm_seconds'] * 1000:.3f} "
                    f"head_ms={profile['head_seconds'] * 1000:.3f}",
                    flush=True,
                )
                top = sorted(profile["layers"], key=lambda item: item[2], reverse=True)[: args.top_layers]
                print(
                    "profile-top size=" + str(block_size) + " "
                    + ",".join(f"{layer}:{kind}:{seconds * 1000:.3f}" for layer, kind, seconds in top),
                    flush=True,
                )
            print(
                f"profile-done active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                flush=True,
            )
        finally:
            mx.set_wired_limit(previous_limit)
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron runtime profile error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

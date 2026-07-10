#!/usr/bin/env python3
"""Resumable activation-aware expert calibration for official Nemotron NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_stream_forward import StreamingForward
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state


FORMAT = "nemotron-mlx-calibration-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_samples(path: Path) -> list[tuple[str, str]]:
    corpus = load_json(path)
    require(isinstance(corpus, dict) and corpus, "calibration corpus must be a non-empty object")
    categories = []
    for category, samples in corpus.items():
        require(isinstance(category, str) and isinstance(samples, list), "invalid calibration corpus category")
        categories.append((category, [sample for sample in samples if isinstance(sample, str) and sample]))
    result = []
    position = 0
    while True:
        added = False
        for category, samples in categories:
            if position < len(samples):
                result.append((category, samples[position]))
                added = True
        if not added:
            break
        position += 1
    require(result, "calibration corpus has no text samples")
    return result


def build_batches(
    tokenizer, samples: list[tuple[str, str]], batch_tokens: int, max_sample_tokens: int
) -> list[dict]:
    batches = []
    for category, sample in samples:
        token_ids = tokenizer.encode(sample, add_special_tokens=False)[:max_sample_tokens]
        for offset in range(0, len(token_ids), batch_tokens):
            chunk = token_ids[offset : offset + batch_tokens]
            if chunk:
                batches.append(
                    {
                        "category": category,
                        "sample_sha256": hashlib.sha256(sample.encode("utf-8")).hexdigest(),
                        "offset": offset,
                        "token_ids": chunk,
                    }
                )
    require(batches, "calibration corpus encoded to no tokens")
    return batches


def empty_layers(config: dict) -> dict[str, dict]:
    experts = config["n_routed_experts"]
    return {
        str(layer): {
            "counts": [0] * experts,
            "score_sum": [0.0] * experts,
            "weighted_output_norm_sum": [0.0] * experts,
            "output_norm_sum": [0.0] * experts,
            "max_score": [0.0] * experts,
            "max_output_norm": [0.0] * experts,
        }
        for layer, kind in enumerate(config["hybrid_override_pattern"])
        if kind == "E"
    }


def merge_routing(layers: dict[str, dict], routing: dict[int, dict]) -> None:
    for layer, observation in routing.items():
        aggregate = layers[str(layer)]
        indices = observation["indices"]
        scores = observation["scores"]
        norms = observation["output_norms"]
        require(len(indices) == len(scores) == len(norms), f"routing observation length mismatch: layer {layer}")
        for expert, score, norm in zip(indices, scores, norms):
            aggregate["counts"][expert] += 1
            aggregate["score_sum"][expert] += score
            aggregate["weighted_output_norm_sum"][expert] += score * norm
            aggregate["output_norm_sum"][expert] += norm
            aggregate["max_score"][expert] = max(aggregate["max_score"][expert], score)
            aggregate["max_output_norm"][expert] = max(aggregate["max_output_norm"][expert], norm)


def coverage_summary(layers: dict[str, dict]) -> dict:
    by_layer = {
        layer: sum(count > 0 for count in values["counts"])
        for layer, values in layers.items()
    }
    total = sum(by_layer.values())
    slots = sum(len(values["counts"]) for values in layers.values())
    return {
        "observed_slots": total,
        "total_slots": slots,
        "coverage": total / slots,
        "min_observed": min(by_layer.values()),
        "max_observed": max(by_layer.values()),
        "by_layer": by_layer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--max-sample-tokens", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.batch_tokens > 0, "batch-tokens must be positive")
        max_sample_tokens = args.max_sample_tokens or args.batch_tokens
        require(max_sample_tokens > 0, "max-sample-tokens must be positive")
        require(args.max_batches is None or args.max_batches > 0, "max-batches must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        batches = build_batches(tokenizer, corpus_samples(args.corpus), args.batch_tokens, max_sample_tokens)
        identity = {
            "format": FORMAT,
            "source_revision": source_state["revision"],
            "corpus_sha256": sha256_file(args.corpus),
            "batch_tokens": args.batch_tokens,
            "max_sample_tokens": max_sample_tokens,
            "total_batches": len(batches),
        }
        if args.output.exists():
            state = load_json(args.output)
            for key, value in identity.items():
                require(state.get(key) == value, f"calibration state identity mismatch: {key}")
        else:
            state = {
                **identity,
                "status": "running",
                "completed_batches": [],
                "total_tokens": 0,
                "categories": {},
                "layers": empty_layers(config),
            }
            atomic_json(args.output, state)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        operation_log.write(
            f"run-start source_revision={source_state['revision']} completed={len(state['completed_batches'])} "
            f"total_batches={len(batches)} batch_tokens={args.batch_tokens}"
        )
        completed = set(state["completed_batches"])
        processed = 0
        for batch_index, batch in enumerate(batches):
            if batch_index in completed:
                continue
            if args.max_batches is not None and processed >= args.max_batches:
                break
            operation_log.write(
                f"batch-start index={batch_index} category={batch['category']} tokens={len(batch['token_ids'])}"
            )
            started = time.perf_counter()
            runner = StreamingForward(args.source_dir)
            runner.forward_sequence(batch["token_ids"], score_head=False)
            merge_routing(state["layers"], runner.routing)
            state["completed_batches"].append(batch_index)
            state["total_tokens"] += len(batch["token_ids"])
            state["categories"][batch["category"]] = state["categories"].get(batch["category"], 0) + len(
                batch["token_ids"]
            )
            state["coverage"] = coverage_summary(state["layers"])
            atomic_json(args.output, state)
            processed += 1
            operation_log.write(
                f"batch-done index={batch_index} elapsed={time.perf_counter() - started:.2f}s "
                f"tokens_total={state['total_tokens']} coverage={state['coverage']['coverage']:.3%} "
                f"min_observed={state['coverage']['min_observed']}"
            )
            del runner
        if len(state["completed_batches"]) == len(batches):
            state["status"] = "complete"
            atomic_json(args.output, state)
        operation_log.write(
            f"run-stop status={state['status']} processed={processed} completed={len(state['completed_batches'])} "
            f"coverage={state.get('coverage', {}).get('coverage', 0):.3%}"
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        if operation_log:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron MLX calibration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

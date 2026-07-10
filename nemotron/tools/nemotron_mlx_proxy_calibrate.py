#!/usr/bin/env python3
"""Capture routing co-occurrence and same-input expert-output similarity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import build_batches, corpus_samples, sha256_file
from nemotron_mlx_stream_forward import StreamingForward
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state


FORMAT = "nemotron-proxy-calibration-v1"


def initialize_arrays(config: dict, categories: list[str]) -> dict[str, np.ndarray]:
    experts = config["n_routed_experts"]
    arrays = {}
    for layer, kind in enumerate(config["hybrid_override_pattern"]):
        if kind != "E":
            continue
        prefix = f"layer_{layer:03d}"
        arrays[f"{prefix}_pair_counts"] = np.zeros((experts, experts), dtype=np.uint32)
        arrays[f"{prefix}_cosine_sums"] = np.zeros((experts, experts), dtype=np.float32)
        arrays[f"{prefix}_score_product_sums"] = np.zeros((experts, experts), dtype=np.float32)
        arrays[f"{prefix}_category_counts"] = np.zeros(
            (len(categories), experts),
            dtype=np.uint32,
        )
    return arrays


def merge_observations(
    arrays: dict[str, np.ndarray],
    routing: dict[int, dict],
    category_index: int,
    top_k: int,
) -> None:
    for layer, observation in routing.items():
        require("pair_cosines" in observation, f"layer {layer} has no pair-cosine observation")
        indices = np.asarray(observation["indices"], dtype=np.int64).reshape(-1, top_k)
        scores = np.asarray(observation["scores"], dtype=np.float32).reshape(-1, top_k)
        cosines = np.asarray(observation["pair_cosines"], dtype=np.float32).reshape(
            -1,
            top_k,
            top_k,
        )
        require(
            indices.shape[0] == scores.shape[0] == cosines.shape[0],
            f"layer {layer} proxy observation token mismatch",
        )
        prefix = f"layer_{layer:03d}"
        counts = arrays[f"{prefix}_pair_counts"]
        cosine_sums = arrays[f"{prefix}_cosine_sums"]
        score_sums = arrays[f"{prefix}_score_product_sums"]
        category_counts = arrays[f"{prefix}_category_counts"]
        for token_indices, token_scores, token_cosines in zip(indices, scores, cosines):
            rows = np.broadcast_to(token_indices[:, None], (top_k, top_k))
            columns = np.broadcast_to(token_indices[None, :], (top_k, top_k))
            np.add.at(counts, (rows, columns), 1)
            np.add.at(cosine_sums, (rows, columns), token_cosines)
            np.add.at(
                score_sums,
                (rows, columns),
                token_scores[:, None] * token_scores[None, :],
            )
            np.add.at(category_counts[category_index], token_indices, 1)


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_arrays(path: Path, expected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        require(set(loaded.files) == set(expected), "proxy calibration array catalog mismatch")
        result = {name: loaded[name] for name in loaded.files}
    for name, reference in expected.items():
        require(
            result[name].shape == reference.shape and result[name].dtype == reference.dtype,
            f"proxy calibration array mismatch: {name}",
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--max-sample-tokens", type=int, default=32)
    parser.add_argument("--total-batches", type=int, default=16)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.batch_tokens > 0, "batch tokens must be positive")
        require(args.max_sample_tokens > 0, "max sample tokens must be positive")
        require(args.total_batches > 0, "total batches must be positive")
        require(args.max_batches is None or args.max_batches > 0, "max batches must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        category_names = sorted({category for category, _ in corpus_samples(args.corpus)})
        category_to_index = {category: index for index, category in enumerate(category_names)}
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        batches = build_batches(
            tokenizer,
            corpus_samples(args.corpus),
            args.batch_tokens,
            args.max_sample_tokens,
        )[: args.total_batches]
        require(len(batches) == args.total_batches, "corpus has fewer batches than requested")
        identity = {
            "format": FORMAT,
            "source_revision": source_state["revision"],
            "corpus_sha256": sha256_file(args.corpus),
            "batch_tokens": args.batch_tokens,
            "max_sample_tokens": args.max_sample_tokens,
            "total_batches": len(batches),
            "categories": category_names,
            "experts": config["n_routed_experts"],
            "top_k": config["num_experts_per_tok"],
            "layers": [
                layer
                for layer, kind in enumerate(config["hybrid_override_pattern"])
                if kind == "E"
            ],
        }
        empty = initialize_arrays(config, category_names)
        if args.state.exists():
            state = load_json(args.state)
            for key, value in identity.items():
                require(state.get(key) == value, f"proxy calibration identity mismatch: {key}")
            require(args.arrays.is_file(), "proxy calibration state has no array file")
            require(state.get("arrays_sha256") == sha256_file(args.arrays), "proxy arrays hash mismatch")
            arrays = load_arrays(args.arrays, empty)
        else:
            args.state.parent.mkdir(parents=True, exist_ok=True)
            require(not args.arrays.exists(), "proxy arrays exist without state")
            state = {
                **identity,
                "status": "running",
                "completed_batches": [],
                "total_tokens": 0,
                "category_tokens": {},
            }
            arrays = empty
            save_arrays(args.arrays, arrays)
            state["arrays_sha256"] = sha256_file(args.arrays)
            atomic_json(args.state, state)
        operation_log = OperationLog(args.state.with_suffix(".log"))
        operation_log.write(
            f"proxy-run-start revision={source_state['revision']} "
            f"completed={len(state['completed_batches'])} total={len(batches)}"
        )
        completed = set(state["completed_batches"])
        processed = 0
        for batch_index, batch in enumerate(batches):
            if batch_index in completed:
                continue
            if args.max_batches is not None and processed >= args.max_batches:
                break
            operation_log.write(
                f"proxy-batch-start index={batch_index} category={batch['category']} "
                f"tokens={len(batch['token_ids'])}"
            )
            started = time.perf_counter()
            runner = StreamingForward(args.source_dir)
            runner.forward_sequence(
                batch["token_ids"],
                score_head=False,
                capture_pair_cosines=True,
            )
            merge_observations(
                arrays,
                runner.routing,
                category_to_index[batch["category"]],
                identity["top_k"],
            )
            save_arrays(args.arrays, arrays)
            state["completed_batches"].append(batch_index)
            state["total_tokens"] += len(batch["token_ids"])
            state["category_tokens"][batch["category"]] = (
                state["category_tokens"].get(batch["category"], 0)
                + len(batch["token_ids"])
            )
            state["arrays_sha256"] = sha256_file(args.arrays)
            if len(state["completed_batches"]) == len(batches):
                state["status"] = "complete"
            atomic_json(args.state, state)
            processed += 1
            operation_log.write(
                f"proxy-batch-done index={batch_index} elapsed={time.perf_counter() - started:.2f}s "
                f"tokens_total={state['total_tokens']}"
            )
            del runner
        operation_log.write(
            f"proxy-run-stop status={state['status']} processed={processed} "
            f"completed={len(state['completed_batches'])}"
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"proxy-run-failed error={exc}")
        print(f"nemotron proxy calibration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

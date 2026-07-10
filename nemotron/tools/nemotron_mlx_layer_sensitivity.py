#!/usr/bin/env python3
"""Measure per-layer hard-pruning sensitivity on independent hidden states."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import build_batches, corpus_samples, sha256_file
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_proxy_compare import error_metrics
from nemotron_mlx_stream_forward import StreamingForward
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-layer-sensitivity-v1"


def parse_plan(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    require(separator == "=" and label and path, f"invalid plan specification: {value}")
    return label, Path(path)


def pruned_routed_output(block, x: mx.array, retained: list[int]) -> mx.array:
    hidden = block.norm(x)
    original_indices, scores = block.route_retained(hidden, retained)
    latent = block.fc1_latent(hidden)
    selected = expert_outputs(latent, block.experts, original_indices)
    return block.fc2_latent((selected * scores[..., None]).sum(axis=-2))


def baseline_components(block, x: mx.array) -> tuple[mx.array, mx.array]:
    hidden = block.norm(x)
    indices, scores = block.route(hidden)
    latent = block.fc1_latent(hidden)
    selected = expert_outputs(latent, block.experts, indices)
    routed = block.fc2_latent((selected * scores[..., None]).sum(axis=-2))
    shared_hidden = mx.square(mx.maximum(block.shared_up(hidden), mx.array(0.0, dtype=hidden.dtype)))
    return routed, block.shared_down(shared_hidden)


def summarize(results: list[dict], labels: list[str], layers: list[int]) -> dict:
    summary = {}
    for label in labels:
        summary[label] = {}
        for layer in layers:
            rows = [row for row in results if row["budget"] == label and row["layer"] == layer]
            values = [row["update"]["relative_l2"] for row in rows]
            summary[label][str(layer)] = {
                "mean_update_relative_l2": float(np.mean(values)),
                "max_update_relative_l2": max(values),
                "mean_routed_relative_l2": float(
                    np.mean([row["routed"]["relative_l2"] for row in rows])
                ),
                "mean_output_relative_l2": float(
                    np.mean([row["output"]["relative_l2"] for row in rows])
                ),
                "max_output_relative_l2": max(row["output"]["relative_l2"] for row in rows),
                "categories": {row["category"]: row["output"]["relative_l2"] for row in rows},
            }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--max-sample-tokens", type=int, default=32)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.batch_tokens > 0 and args.max_sample_tokens > 0, "token limits must be positive")
        require(args.max_batches is None or args.max_batches > 0, "max batches must be positive")
        config = load_json(args.source_dir / "config.json")
        layers = [index for index, kind in enumerate(config["hybrid_override_pattern"]) if kind == "E"]
        plan_specs = [parse_plan(value) for value in args.plan]
        require(len({label for label, _ in plan_specs}) == len(plan_specs), "duplicate plan label")
        plans = {}
        plan_hashes = {}
        for label, path in plan_specs:
            plan = load_json(path)
            require(plan.get("source_revision") is not None, f"plan {label} has no source revision")
            require(plan["model_moe_layers"] == layers, f"plan {label} layer catalog mismatch")
            plans[label] = plan
            plan_hashes[label] = sha256_file(path)
        revisions = {plan["source_revision"] for plan in plans.values()}
        require(len(revisions) == 1, "plan source revisions differ")

        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        batches = build_batches(
            tokenizer,
            corpus_samples(args.corpus),
            args.batch_tokens,
            args.max_sample_tokens,
        )
        if args.max_batches is not None:
            batches = batches[: args.max_batches]
        require(batches, "no sensitivity batches selected")

        captured = []
        for batch_index, batch in enumerate(batches):
            print(
                f"sensitivity-capture-start batch={batch_index} category={batch['category']} "
                f"tokens={len(batch['token_ids'])}",
                flush=True,
            )
            started = time.perf_counter()
            runner = StreamingForward(args.source_dir)
            runner.forward_sequence(
                batch["token_ids"],
                score_head=False,
                capture_layer_inputs=set(layers),
            )
            require(set(runner.layer_inputs) == set(layers), "layer input capture is incomplete")
            captured.append((batch, runner.layer_inputs))
            print(
                f"sensitivity-capture-done batch={batch_index} elapsed={time.perf_counter() - started:.2f}s",
                flush=True,
            )
            del runner
            gc.collect()
            mx.clear_cache()

        results = []
        for layer in layers:
            print(f"sensitivity-layer-start layer={layer}", flush=True)
            started = time.perf_counter()
            block = load_moe_layer(args.source_dir, layer)
            for batch_index, (batch, inputs) in enumerate(captured):
                x = mx.array(inputs[layer])
                baseline_routed, shared = baseline_components(block, x)
                mx.eval(baseline_routed, shared)
                baseline_routed_np = np.asarray(baseline_routed, dtype=np.float32)
                shared_np = np.asarray(shared, dtype=np.float32)
                baseline_update_np = baseline_routed_np + shared_np
                x_np = inputs[layer].astype(np.float32)
                baseline_output_np = x_np + baseline_update_np
                for label, _ in plan_specs:
                    candidate = pruned_routed_output(
                        block,
                        x,
                        plans[label]["kept_by_layer"][str(layer)],
                    )
                    mx.eval(candidate)
                    candidate_np = np.asarray(candidate, dtype=np.float32)
                    results.append(
                        {
                            "layer": layer,
                            "batch": batch_index,
                            "category": batch["category"],
                            "tokens": len(batch["token_ids"]),
                            "budget": label,
                            "retained_experts": plans[label]["new_num_experts"],
                            "routed": error_metrics(candidate_np, baseline_routed_np),
                            "update": error_metrics(candidate_np + shared_np, baseline_update_np),
                            "output": error_metrics(
                                x_np + candidate_np + shared_np,
                                baseline_output_np,
                            ),
                        }
                    )
            del block
            gc.collect()
            mx.clear_cache()
            print(
                f"sensitivity-layer-done layer={layer} elapsed={time.perf_counter() - started:.2f}s",
                flush=True,
            )
        labels = [label for label, _ in plan_specs]
        report = {
            "format": FORMAT,
            "source_revision": next(iter(revisions)),
            "corpus_sha256": sha256_file(args.corpus),
            "plan_sha256": plan_hashes,
            "batch_tokens": args.batch_tokens,
            "max_sample_tokens": args.max_sample_tokens,
            "batches": [
                {
                    "category": batch["category"],
                    "sample_sha256": batch["sample_sha256"],
                    "offset": batch["offset"],
                    "tokens": len(batch["token_ids"]),
                }
                for batch, _ in captured
            ],
            "layers": layers,
            "results": results,
            "summary": summarize(results, labels, layers),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        print(f"sensitivity-report path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        print(f"nemotron layer sensitivity error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

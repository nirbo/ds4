#!/usr/bin/env python3
"""Compare hard pruning and original-router proxy experts on real hidden states."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import build_batches, corpus_samples, sha256_file
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import NemotronLatentMoELayer, load_moe_layer
from nemotron_mlx_stream_forward import StreamingForward
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-proxy-comparison-v1"


def routed_output(
    block: NemotronLatentMoELayer,
    x: mx.array,
    prototype_map: mx.array | None = None,
) -> mx.array:
    hidden = block.norm(x)
    indices, scores = block.route(hidden)
    if prototype_map is not None:
        indices = prototype_map[indices]
    latent = block.fc1_latent(hidden)
    selected = expert_outputs(latent, block.experts, indices)
    return block.fc2_latent((selected * scores[..., None]).sum(axis=-2))


def shared_output(block: NemotronLatentMoELayer, x: mx.array) -> mx.array:
    hidden = block.norm(x)
    activated = mx.square(mx.maximum(block.shared_up(hidden), mx.array(0.0, dtype=hidden.dtype)))
    return block.shared_down(activated)


def error_metrics(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    candidate = candidate.astype(np.float64)
    baseline = baseline.astype(np.float64)
    difference = candidate - baseline
    denominator = max(float(np.linalg.norm(baseline)), 1e-30)
    return {
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(
            np.vdot(candidate.reshape(-1), baseline.reshape(-1))
            / max(float(np.linalg.norm(candidate) * np.linalg.norm(baseline)), 1e-30)
        ),
    }


def unique_mapping_metrics(mapped_indices: np.ndarray) -> dict[str, float]:
    require(mapped_indices.ndim == 3, "mapped indices must be batch/sequence/top-k")
    counts = [len(set(int(value) for value in row)) for row in mapped_indices.reshape(-1, mapped_indices.shape[-1])]
    return {
        "mean_unique_prototypes": float(np.mean(counts)),
        "min_unique_prototypes": min(counts),
        "max_unique_prototypes": max(counts),
        "mean_duplicate_fraction": float(1.0 - np.mean(counts) / mapped_indices.shape[-1]),
    }


def compare_layer(
    source_dir: Path,
    candidate_dir: Path,
    layer: int,
    token_ids: list[int],
    mapping: list[int],
) -> dict:
    started = time.perf_counter()
    runner = StreamingForward(source_dir)
    x = runner.forward_sequence(token_ids, max_layers=layer, score_head=False)
    mx.eval(x)
    del runner
    gc.collect()
    mx.clear_cache()

    source = load_moe_layer(source_dir, layer)
    require(len(mapping) == source.experts.up.experts, f"layer {layer} proxy map length mismatch")
    proxy_map = mx.array(mapping, dtype=mx.uint32)
    original_indices, _ = source.route(source.norm(x))
    mapped_indices = proxy_map[original_indices]
    baseline_routed = routed_output(source, x)
    proxy_routed = routed_output(source, x, proxy_map)
    shared = shared_output(source, x)
    mx.eval(mapped_indices, baseline_routed, proxy_routed, shared)
    uniqueness = unique_mapping_metrics(np.asarray(mapped_indices, dtype=np.uint32))
    baseline_routed_np = np.asarray(baseline_routed, dtype=np.float32)
    proxy_routed_np = np.asarray(proxy_routed, dtype=np.float32)
    shared_np = np.asarray(shared, dtype=np.float32)
    x_np = np.asarray(x, dtype=np.float32)
    del source, proxy_map, baseline_routed, proxy_routed, shared
    gc.collect()
    mx.clear_cache()

    hard = load_moe_layer(candidate_dir, layer)
    hard_routed = routed_output(hard, x)
    mx.eval(hard_routed)
    hard_routed_np = np.asarray(hard_routed, dtype=np.float32)
    del hard, hard_routed, x
    gc.collect()
    mx.clear_cache()

    baseline_update = baseline_routed_np + shared_np
    proxy_update = proxy_routed_np + shared_np
    hard_update = hard_routed_np + shared_np
    baseline_output = x_np + baseline_update
    return {
        "layer": layer,
        "tokens": len(token_ids),
        "proxy": {
            "routed": error_metrics(proxy_routed_np, baseline_routed_np),
            "update": error_metrics(proxy_update, baseline_update),
            "output": error_metrics(x_np + proxy_update, baseline_output),
        },
        "hard": {
            "routed": error_metrics(hard_routed_np, baseline_routed_np),
            "update": error_metrics(hard_update, baseline_update),
            "output": error_metrics(x_np + hard_update, baseline_output),
        },
        "proxy_routing": uniqueness,
        "elapsed_seconds": time.perf_counter() - started,
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(max(value, 1e-30)) for value in values) / len(values))


def parse_layers(value: str) -> list[int]:
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise MetadataError(f"invalid layer list: {value}") from exc
    require(result and len(result) == len(set(result)), "layer list is empty or duplicated")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--proxy-plan", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--layers", default="1,34,87")
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--max-sample-tokens", type=int, default=32)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.batch_tokens > 0 and args.max_sample_tokens > 0, "token limits must be positive")
        require(args.batches > 0, "batch count must be positive")
        source_config = load_json(args.source_dir / "config.json")
        candidate_config = load_json(args.candidate_dir / "config.json")
        plan = load_json(args.proxy_plan)
        require(plan.get("format") == "nemotron-proxy-plan-v1", "invalid proxy plan")
        require(
            source_config["n_routed_experts"] == plan["old_num_experts"]
            and candidate_config["n_routed_experts"] == plan["physical_experts"],
            "proxy plan/model expert counts do not match",
        )
        layers = parse_layers(args.layers)
        for layer in layers:
            require(
                source_config["hybrid_override_pattern"][layer] == "E",
                f"layer {layer} is not LatentMoE",
            )
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        batches = build_batches(
            tokenizer,
            corpus_samples(args.corpus),
            args.batch_tokens,
            args.max_sample_tokens,
        )[: args.batches]
        require(len(batches) == args.batches, "corpus has fewer batches than requested")
        results = []
        for batch_index, batch in enumerate(batches):
            for layer in layers:
                print(
                    f"proxy-compare-start batch={batch_index} category={batch['category']} layer={layer}",
                    flush=True,
                )
                result = compare_layer(
                    args.source_dir,
                    args.candidate_dir,
                    layer,
                    batch["token_ids"],
                    plan["prototype_by_original_layer"][str(layer)],
                )
                result.update({"batch": batch_index, "category": batch["category"]})
                results.append(result)
                print(
                    f"proxy-compare-done batch={batch_index} layer={layer} "
                    f"proxy_routed_l2={result['proxy']['routed']['relative_l2']:.6g} "
                    f"hard_routed_l2={result['hard']['routed']['relative_l2']:.6g} "
                    f"elapsed={result['elapsed_seconds']:.2f}s",
                    flush=True,
                )
        summary = {}
        for method in ("proxy", "hard"):
            for scope in ("routed", "update", "output"):
                values = [item[method][scope]["relative_l2"] for item in results]
                summary[f"{method}_{scope}_relative_l2_geomean"] = geometric_mean(values)
                summary[f"{method}_{scope}_relative_l2_max"] = max(values)
        summary["proxy_to_hard_routed_geomean_ratio"] = (
            summary["proxy_routed_relative_l2_geomean"]
            / max(summary["hard_routed_relative_l2_geomean"], 1e-30)
        )
        summary["proxy_mean_unique_prototypes"] = float(
            np.mean([item["proxy_routing"]["mean_unique_prototypes"] for item in results])
        )
        summary["proxy_mean_duplicate_fraction"] = float(
            np.mean([item["proxy_routing"]["mean_duplicate_fraction"] for item in results])
        )
        report = {
            "format": FORMAT,
            "source_revision": plan["source_revision"],
            "source_dir": str(args.source_dir.resolve()),
            "candidate_dir": str(args.candidate_dir.resolve()),
            "proxy_plan_sha256": sha256_file(args.proxy_plan),
            "corpus_sha256": sha256_file(args.corpus),
            "layers": layers,
            "batch_tokens": args.batch_tokens,
            "batches": args.batches,
            "results": results,
            "summary": summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        print(f"proxy-compare-report path={args.output} sha256={sha256_file(args.output)}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        print(f"nemotron proxy comparison error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

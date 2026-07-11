#!/usr/bin/env python3
"""Fit and validate dense functional expert proxies for one Nemotron MoE layer."""

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
from nemotron_mlx_calibrate import sha256_file
from nemotron_mlx_layer_distill import capture_inputs
from nemotron_mlx_layer_sensitivity import baseline_components, pruned_routed_output
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_proxy_compare import error_metrics, routed_output
from nemotron_mlx_stream_forward import validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state


FORMAT = "nemotron-dense-proxy-v1"


def fit_dense_mapping(
    projected: np.ndarray,
    route_indices: np.ndarray,
    route_scores: np.ndarray,
    retained: list[int],
) -> tuple[list[int], list[dict]]:
    """Map removed experts by output error on contexts where each was selected."""

    require(projected.ndim == 3, "projected outputs must be token/expert/feature")
    tokens, experts, _ = projected.shape
    require(route_indices.shape == route_scores.shape, "route arrays do not match")
    require(route_indices.ndim == 2 and route_indices.shape[0] == tokens, "route token mismatch")
    require(retained == sorted(set(retained)) and retained, "invalid retained expert set")
    retained_array = np.asarray(retained, dtype=np.int64)
    mapping = list(range(experts))
    rows = []
    for expert in range(experts):
        if expert in retained:
            continue
        selected = route_indices == expert
        weights = np.sum(np.where(selected, route_scores, 0.0), axis=1).astype(np.float64)
        observed = weights > 0
        if not np.any(observed):
            observed = np.ones(tokens, dtype=bool)
            weights = np.ones(tokens, dtype=np.float64)
        target = projected[observed, expert].astype(np.float64)
        candidates = projected[observed][:, retained_array].astype(np.float64)
        squared = np.sum(np.square(candidates - target[:, None, :]), axis=-1)
        distances = np.average(squared, axis=0, weights=weights[observed])
        position = int(np.argmin(distances))
        mapping[expert] = int(retained_array[position])
        rows.append(
            {
                "expert": expert,
                "prototype": mapping[expert],
                "selected_tokens": int(np.count_nonzero(selected)),
                "projected_mse": float(distances[position]),
            }
        )
    return mapping, rows


def project_all_experts(block, inputs: list[np.ndarray], projection: mx.array):
    projected_rows = []
    route_rows = []
    score_rows = []
    experts = block.experts.up.experts
    for x_np in inputs:
        x = mx.array(x_np)
        hidden = block.norm(x)
        latent = block.fc1_latent(hidden)
        indices, scores = block.route(hidden)
        all_indices = mx.broadcast_to(
            mx.arange(experts, dtype=mx.uint32)[None, None, :],
            (*latent.shape[:2], experts),
        )
        outputs = expert_outputs(latent, block.experts, all_indices)
        projected = outputs.astype(mx.float32) @ projection
        mx.eval(indices, scores, projected)
        projected_rows.append(np.asarray(projected, dtype=np.float32).reshape(-1, experts, projection.shape[1]))
        route_rows.append(np.asarray(indices, dtype=np.int64).reshape(-1, indices.shape[-1]))
        score_rows.append(np.asarray(scores, dtype=np.float32).reshape(-1, scores.shape[-1]))
        del x, hidden, latent, indices, scores, all_indices, outputs, projected
        mx.clear_cache()
    return np.concatenate(projected_rows), np.concatenate(route_rows), np.concatenate(score_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--train-corpus", required=True, type=Path)
    parser.add_argument("--validation-corpus", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--projection-dims", type=int, default=32)
    parser.add_argument("--max-sample-tokens", type=int, default=16)
    parser.add_argument("--train-cases", type=int, default=2)
    parser.add_argument("--validation-cases", type=int, default=2)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.projection_dims > 0, "projection dimensions must be positive")
        require(args.train_cases > 0 and args.validation_cases > 0, "case counts must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        require(config["hybrid_override_pattern"][args.layer] == "E", "selected layer is not MoE")
        plan = load_json(args.plan)
        retained_by_layer = validate_virtual_plan(plan, config, source_state["revision"])
        retained = retained_by_layer[str(args.layer)]
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        train = capture_inputs(
            args.source_dir,
            tokenizer,
            args.train_corpus,
            [args.layer],
            args.max_sample_tokens,
            args.train_cases,
            "train",
        )
        validation = capture_inputs(
            args.source_dir,
            tokenizer,
            args.validation_corpus,
            [args.layer],
            args.max_sample_tokens,
            args.validation_cases,
            "validation",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "fit.log")
        operation_log.write(
            f"dense-proxy-start layer={args.layer} retained={len(retained)} "
            f"projection_dims={args.projection_dims}"
        )
        block = load_moe_layer(args.source_dir, args.layer)
        rng = np.random.default_rng(0x4E56465034 + args.layer)
        projection_np = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(block.experts.down.output_dims, args.projection_dims),
        ) / np.sqrt(args.projection_dims)
        projected, route_indices, route_scores = project_all_experts(
            block,
            [inputs[args.layer] for _, inputs in train],
            mx.array(projection_np),
        )
        mapping, fit_rows = fit_dense_mapping(projected, route_indices, route_scores, retained)
        proxy_map = mx.array(mapping, dtype=mx.uint32)
        results = []
        for case, (batch, inputs) in enumerate(validation):
            x_np = inputs[args.layer].astype(np.float32)
            x = mx.array(x_np)
            baseline_routed, shared = baseline_components(block, x)
            proxy_routed = routed_output(block, x, proxy_map)
            hard_routed = pruned_routed_output(block, x, retained)
            mx.eval(baseline_routed, shared, proxy_routed, hard_routed)
            baseline_np = np.asarray(baseline_routed, dtype=np.float32)
            shared_np = np.asarray(shared, dtype=np.float32)
            baseline_output = x_np + baseline_np + shared_np
            row = {"case": case, "category": batch["category"], "tokens": len(batch["token_ids"])}
            for label, value in (("dense_proxy", proxy_routed), ("hard", hard_routed)):
                routed_np = np.asarray(value, dtype=np.float32)
                row[label] = {
                    "routed": error_metrics(routed_np, baseline_np),
                    "output": error_metrics(x_np + routed_np + shared_np, baseline_output),
                }
            results.append(row)
        summary = {}
        for label in ("dense_proxy", "hard"):
            for scope in ("routed", "output"):
                values = [row[label][scope]["relative_l2"] for row in results]
                summary[f"{label}_{scope}_mean_relative_l2"] = float(np.mean(values))
                summary[f"{label}_{scope}_max_relative_l2"] = max(values)
        summary["proxy_to_hard_output_mean_ratio"] = (
            summary["dense_proxy_output_mean_relative_l2"]
            / max(summary["hard_output_mean_relative_l2"], 1e-30)
        )
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "plan_sha256": sha256_file(args.plan),
            "train_corpus_sha256": sha256_file(args.train_corpus),
            "validation_corpus_sha256": sha256_file(args.validation_corpus),
            "layer": args.layer,
            "retained_experts": len(retained),
            "projection_dims": args.projection_dims,
            "train_cases": args.train_cases,
            "validation_cases": args.validation_cases,
            "max_sample_tokens": args.max_sample_tokens,
            "mapping": mapping,
            "fit": fit_rows,
            "results": results,
            "summary": summary,
        }
        atomic_json(args.output_dir / "report.json", report)
        operation_log.write(
            f"dense-proxy-done ratio={summary['proxy_to_hard_output_mean_ratio']:.6g}"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(
            f"dense-proxy-report path={args.output_dir / 'report.json'} "
            f"sha256={sha256_file(args.output_dir / 'report.json')}"
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        if operation_log is not None:
            operation_log.write(f"dense-proxy-failed error={exc}")
        print(f"nemotron dense proxy error: {exc}", file=sys.stderr)
        return 1
    finally:
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())

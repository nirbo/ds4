#!/usr/bin/env python3
"""Evaluate exact-block routed-expert width pruning on one Nemotron MoE layer."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import sha256_file
from nemotron_mlx_layer_distill import capture_inputs
from nemotron_mlx_layer_sensitivity import baseline_components, pruned_routed_output
from nemotron_mlx_moe import (
    NVFP4ExpertMLP,
    expert_outputs,
    slice_expert_blocks,
    switch_matmul,
)
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_proxy_compare import error_metrics
from nemotron_mlx_stream_forward import validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state


FORMAT = "nemotron-width-prune-v1"
BLOCK = 16


def select_blocks(importance: np.ndarray, keep_blocks: int) -> np.ndarray:
    require(importance.ndim == 2, "importance must be expert/block")
    require(0 < keep_blocks <= importance.shape[1], "invalid retained block count")
    order = np.argsort(-importance, axis=1, kind="stable")[:, :keep_blocks]
    return np.sort(order, axis=1).astype(np.int32)


def down_column_energy(experts: NVFP4ExpertMLP, operation_log: OperationLog) -> np.ndarray:
    result = np.empty((experts.down.experts, experts.down.input_dims), dtype=np.float32)
    for expert in range(experts.down.experts):
        weight = mx.dequantize(
            experts.down.weight[expert].view(mx.uint32),
            experts.down.scales[expert],
            None,
            BLOCK,
            4,
            "nvfp4",
            dtype=mx.float32,
        ) * experts.down.global_scales[expert]
        energy = mx.sum(mx.square(weight), axis=0)
        mx.eval(energy)
        result[expert] = np.asarray(energy, dtype=np.float32)
        del weight, energy
        if (expert + 1) % 64 == 0:
            operation_log.write(f"down-energy-progress experts={expert + 1}/{experts.down.experts}")
            mx.clear_cache()
    return result


def activation_importance(block, train, layer: int, down_energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    experts = block.experts.up.experts
    width = block.experts.up.output_dims
    importance = np.zeros((experts, width), dtype=np.float64)
    selections = np.zeros(experts, dtype=np.uint32)
    for _, inputs in train:
        x = mx.array(inputs[layer])
        hidden = block.norm(x)
        latent = block.fc1_latent(hidden)
        indices, scores = block.route(hidden)
        up = switch_matmul(latent, block.experts.up, indices).squeeze(-2)
        activated = mx.square(mx.maximum(up, mx.array(0.0, dtype=up.dtype)))
        mx.eval(indices, scores, activated)
        indices_np = np.asarray(indices, dtype=np.int64).reshape(-1, indices.shape[-1])
        scores_np = np.asarray(scores, dtype=np.float32).reshape(-1, scores.shape[-1])
        activated_np = np.asarray(activated, dtype=np.float32).reshape(-1, indices.shape[-1], width)
        for token_indices, token_scores, token_activations in zip(
            indices_np, scores_np, activated_np
        ):
            for expert, score, values in zip(token_indices, token_scores, token_activations):
                importance[expert] += float(score) ** 2 * np.square(values) * down_energy[expert]
                selections[expert] += 1
        del x, hidden, latent, indices, scores, up, activated
        mx.clear_cache()
    return importance.reshape(experts, width // BLOCK, BLOCK).sum(axis=-1), selections


def exact_block_importance(
    block,
    train,
    layer: int,
    operation_log: OperationLog,
) -> tuple[np.ndarray, np.ndarray]:
    experts = block.experts.up.experts
    width = block.experts.up.output_dims
    blocks = width // BLOCK
    down = mx.dequantize(
        block.experts.down.weight.view(mx.uint32),
        block.experts.down.scales,
        None,
        BLOCK,
        4,
        "nvfp4",
        dtype=mx.bfloat16,
    ) * block.experts.down.global_scales[:, None, None]
    mx.eval(down)
    importance = np.zeros((experts, blocks), dtype=np.float64)
    selections = np.zeros(experts, dtype=np.uint32)
    for case, (_, inputs) in enumerate(train):
        x = mx.array(inputs[layer])
        hidden = block.norm(x)
        latent = block.fc1_latent(hidden)
        indices, scores = block.route(hidden)
        up = switch_matmul(latent, block.experts.up, indices).squeeze(-2)
        activated = mx.square(mx.maximum(up, mx.array(0.0, dtype=up.dtype)))
        mx.eval(indices, scores, activated)
        indices_np = np.asarray(indices, dtype=np.int64).reshape(-1, indices.shape[-1])
        scores_np = np.asarray(scores, dtype=np.float32).reshape(-1, scores.shape[-1])
        activated = activated.reshape(-1, indices.shape[-1], blocks, BLOCK)
        for token, (token_indices, token_scores) in enumerate(zip(indices_np, scores_np)):
            selected_down = down[mx.array(token_indices, dtype=mx.uint32)].reshape(
                indices.shape[-1], block.experts.down.output_dims, blocks, BLOCK
            )
            contributions = mx.einsum(
                "kobi,kbi->kbo",
                selected_down,
                activated[token],
            )
            energy = mx.sum(mx.square(contributions.astype(mx.float32)), axis=-1)
            mx.eval(energy)
            energy_np = np.asarray(energy, dtype=np.float32)
            for expert, score, values in zip(token_indices, token_scores, energy_np):
                importance[expert] += float(score) ** 2 * values
                selections[expert] += 1
            del selected_down, contributions, energy
        operation_log.write(f"exact-importance-progress cases={case + 1}/{len(train)}")
        del x, hidden, latent, indices, scores, up, activated
        mx.clear_cache()
    del down
    mx.clear_cache()
    return importance, selections


def evaluate_layer(
    source_dir: Path,
    layer: int,
    retained: list[int],
    train,
    validation,
    keep_ratio: float,
    importance_method: str,
    operation_log: OperationLog,
) -> dict:
    block = load_moe_layer(source_dir, layer)
    source_blocks = block.experts.up.output_dims // BLOCK
    keep_blocks = round(source_blocks * keep_ratio)
    require(keep_blocks > 0, "keep ratio removes every block")
    operation_log.write(
        f"width-prune-layer-start layer={layer} blocks={source_blocks}->{keep_blocks} "
        f"width={block.experts.up.output_dims}->{keep_blocks * BLOCK}"
    )
    down_energy = None
    if importance_method == "exact":
        importance, selections = exact_block_importance(block, train, layer, operation_log)
    else:
        down_energy = down_column_energy(block.experts, operation_log)
        importance, selections = activation_importance(block, train, layer, down_energy)
    unobserved = selections == 0
    if np.any(unobserved):
        if down_energy is None:
            down_energy = down_column_energy(block.experts, operation_log)
        fallback = down_energy.reshape(down_energy.shape[0], source_blocks, BLOCK).sum(axis=-1)
        importance[unobserved] = fallback[unobserved]
    kept_blocks = select_blocks(importance, keep_blocks)
    narrowed = slice_expert_blocks(block.experts, kept_blocks, BLOCK)
    mx.eval(
        narrowed.up.weight, narrowed.up.scales,
        narrowed.down.weight, narrowed.down.scales,
    )
    results = []
    for case, (batch, inputs) in enumerate(validation):
        x_np = inputs[layer].astype(np.float32)
        x = mx.array(x_np)
        hidden = block.norm(x)
        indices, scores = block.route(hidden)
        latent = block.fc1_latent(hidden)
        baseline_routed, shared = baseline_components(block, x)
        width_values = expert_outputs(latent, narrowed, indices)
        width_routed = block.fc2_latent((width_values * scores[..., None]).sum(axis=-2))
        hard_routed = pruned_routed_output(block, x, retained)
        mx.eval(baseline_routed, shared, width_routed, hard_routed)
        baseline_np = np.asarray(baseline_routed, dtype=np.float32)
        shared_np = np.asarray(shared, dtype=np.float32)
        baseline_output = x_np + baseline_np + shared_np
        row = {"case": case, "category": batch["category"], "tokens": len(batch["token_ids"])}
        for label, value in (("width", width_routed), ("hard", hard_routed)):
            routed_np = np.asarray(value, dtype=np.float32)
            row[label] = {
                "routed": error_metrics(routed_np, baseline_np),
                "output": error_metrics(x_np + routed_np + shared_np, baseline_output),
            }
        results.append(row)
    summary = {
        "unobserved_experts": int(np.count_nonzero(unobserved)),
        "selection_min": int(selections.min()),
        "selection_median": float(np.median(selections)),
        "selection_max": int(selections.max()),
    }
    for label in ("width", "hard"):
        for scope in ("routed", "output"):
            values = [row[label][scope]["relative_l2"] for row in results]
            summary[f"{label}_{scope}_mean_relative_l2"] = float(np.mean(values))
            summary[f"{label}_{scope}_max_relative_l2"] = max(values)
    summary["width_to_hard_output_mean_ratio"] = (
        summary["width_output_mean_relative_l2"]
        / max(summary["hard_output_mean_relative_l2"], 1e-30)
    )
    operation_log.write(
        f"width-prune-layer-done layer={layer} "
        f"ratio={summary['width_to_hard_output_mean_ratio']:.6g}"
    )
    del block, narrowed
    gc.collect()
    mx.clear_cache()
    return {
        "source_width": source_blocks * BLOCK,
        "retained_width": keep_blocks * BLOCK,
        "source_blocks": source_blocks,
        "retained_blocks": keep_blocks,
        "keep_ratio": keep_blocks / source_blocks,
        "retained_experts_hard_baseline": len(retained),
        "kept_blocks": kept_blocks.tolist(),
        "summary": summary,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--train-corpus", required=True, type=Path)
    parser.add_argument("--validation-corpus", required=True, type=Path)
    parser.add_argument("--layer", required=True, help="MoE layer number, comma list, or 'all'")
    parser.add_argument("--keep-ratio", type=float, default=0.75)
    parser.add_argument("--importance", choices=("exact", "diagonal"), default="exact")
    parser.add_argument("--max-sample-tokens", type=int, default=16)
    parser.add_argument("--train-cases", type=int, default=8)
    parser.add_argument("--validation-cases", type=int, default=8)
    parser.add_argument("--concatenate-train", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(0 < args.keep_ratio <= 1, "keep ratio must be in (0, 1]")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        model_layers = [
            layer for layer, kind in enumerate(config["hybrid_override_pattern"]) if kind == "E"
        ]
        if args.layer == "all":
            layers = model_layers
        else:
            try:
                layers = [int(value) for value in args.layer.split(",")]
            except ValueError as exc:
                raise MetadataError(f"invalid layer specification: {args.layer}") from exc
            require(layers and len(layers) == len(set(layers)), "layer list is empty or duplicated")
            require(all(layer in model_layers for layer in layers), "selected layer is not MoE")
        plan = load_json(args.plan)
        retained_by_layer = validate_virtual_plan(plan, config, source_state["revision"])
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        train = capture_inputs(
            args.source_dir, tokenizer, args.train_corpus, layers,
            args.max_sample_tokens, args.train_cases, "train", args.concatenate_train,
        )
        validation = capture_inputs(
            args.source_dir, tokenizer, args.validation_corpus, layers,
            args.max_sample_tokens, args.validation_cases, "validation",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "run.log")
        operation_log.write(
            f"width-prune-start layers={','.join(map(str, layers))} "
            f"importance={args.importance} keep_ratio={args.keep_ratio}"
        )
        layer_results = {
            str(layer): evaluate_layer(
                args.source_dir,
                layer,
                retained_by_layer[str(layer)],
                train,
                validation,
                args.keep_ratio,
                args.importance,
                operation_log,
            )
            for layer in layers
        }
        ratios = [value["summary"]["width_to_hard_output_mean_ratio"] for value in layer_results.values()]
        summary = {
            "layers": len(layers),
            "width_wins": sum(ratio < 1.0 for ratio in ratios),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_median": float(np.median(ratios)),
            "ratio_min": min(ratios),
            "ratio_max": max(ratios),
        }
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "plan_sha256": sha256_file(args.plan),
            "train_corpus_sha256": sha256_file(args.train_corpus),
            "validation_corpus_sha256": sha256_file(args.validation_corpus),
            "layers": layers,
            "requested_keep_ratio": args.keep_ratio,
            "importance": args.importance,
            "train_cases": args.train_cases,
            "concatenate_train": args.concatenate_train,
            "validation_cases": args.validation_cases,
            "max_sample_tokens": args.max_sample_tokens,
            "summary": summary,
            "layer_results": layer_results,
        }
        atomic_json(args.output_dir / "report.json", report)
        operation_log.write(f"width-prune-done width_wins={summary['width_wins']}/{len(layers)}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(
            f"width-prune-report path={args.output_dir / 'report.json'} "
            f"sha256={sha256_file(args.output_dir / 'report.json')}"
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        if operation_log is not None:
            operation_log.write(f"width-prune-failed error={exc}")
        print(f"nemotron width prune error: {exc}", file=sys.stderr)
        return 1
    finally:
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())

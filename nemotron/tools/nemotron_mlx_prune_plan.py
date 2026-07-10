#!/usr/bin/env python3
"""Build a guarded Nemotron expert-pruning plan from MLX calibration evidence."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import PLAN_FORMAT, atomic_json


WEIGHTS = {
    "weighted_output_norm_sum": 0.55,
    "score_sum": 0.20,
    "counts": 0.15,
    "max_output_norm": 0.10,
}


def rank_fraction(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    denominator = max(1, len(values) - 1)
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def layer_importance(observation: dict) -> list[float]:
    experts = len(observation["counts"])
    result = [0.0] * experts
    for metric, weight in WEIGHTS.items():
        ranks = rank_fraction([float(value) for value in observation[metric]])
        result = [left + weight * right for left, right in zip(result, ranks)]
    # Calibration misses are unknown, not unimportant.
    for expert, count in enumerate(observation["counts"]):
        if count == 0:
            result[expert] = math.inf
    return result


def minimum_coverage(prune_ratio: float) -> float:
    if prune_ratio <= 0.10:
        return 0.85
    if prune_ratio <= 0.15:
        return 0.90
    if prune_ratio <= 0.20:
        return 0.93
    return 0.96


def build_plan(calibration: dict, prune_ratio: float) -> dict:
    require(calibration.get("format") == "nemotron-mlx-calibration-v1", "unsupported calibration format")
    require(0.0 < prune_ratio < 0.5, "prune ratio must be between zero and 0.5")
    coverage = calibration.get("coverage", {})
    required_coverage = minimum_coverage(prune_ratio)
    require(coverage.get("coverage", 0.0) >= required_coverage, "calibration coverage is below pruning guard")
    layers = calibration.get("layers")
    require(isinstance(layers, dict) and layers, "calibration has no layer observations")
    old_experts = len(next(iter(layers.values()))["counts"])
    new_experts = math.ceil(old_experts * (1.0 - prune_ratio))
    drop_count = old_experts - new_experts
    kept_by_layer = {}
    dropped_by_layer = {}
    scores_by_layer = {}
    for layer in sorted(layers, key=int):
        observation = layers[layer]
        require(len(observation["counts"]) == old_experts, f"expert count mismatch in layer {layer}")
        observed = sum(count > 0 for count in observation["counts"])
        require(observed >= drop_count, f"layer {layer} has too few observed experts for guarded pruning")
        importance = layer_importance(observation)
        dropped = sorted(range(old_experts), key=lambda expert: (importance[expert], expert))[:drop_count]
        require(all(observation["counts"][expert] > 0 for expert in dropped), f"plan would prune unobserved layer {layer} expert")
        kept = sorted(set(range(old_experts)) - set(dropped))
        kept_by_layer[layer] = kept
        dropped_by_layer[layer] = sorted(dropped)
        scores_by_layer[layer] = {
            str(expert): importance[expert]
            for expert in dropped
        }
    model_layers = [int(layer) for layer in sorted(layers, key=int)]
    return {
        "format": PLAN_FORMAT,
        "source_revision": calibration["source_revision"],
        "old_num_experts": old_experts,
        "new_num_experts": new_experts,
        "prune_ratio_requested": prune_ratio,
        "prune_ratio_actual": drop_count / old_experts,
        "model_moe_layers": model_layers,
        "kept_by_layer": kept_by_layer,
        "dropped_by_layer": dropped_by_layer,
        "old_to_new_by_layer": {
            layer: {str(old): new for new, old in enumerate(kept)}
            for layer, kept in kept_by_layer.items()
        },
        "selection": {
            "strategy": "guarded-hybrid-route-weighted-output",
            "metric_weights": WEIGHTS,
            "unobserved_experts": "protected",
            "required_coverage": required_coverage,
            "calibration_coverage": coverage,
            "calibration_tokens": calibration.get("total_tokens"),
            "calibration_corpus_sha256": calibration.get("corpus_sha256"),
            "dropped_importance_by_layer": scores_by_layer,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--prune-ratio", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(load_json(args.calibration), args.prune_ratio)
        atomic_json(args.output, plan)
        coverage = plan["selection"]["calibration_coverage"]
        print(
            f"nemotron prune plan: experts={plan['old_num_experts']}->{plan['new_num_experts']} "
            f"ratio={plan['prune_ratio_actual']:.3%} coverage={coverage['coverage']:.3%} "
            f"min_observed={coverage['min_observed']} output={args.output}"
        )
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron prune plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

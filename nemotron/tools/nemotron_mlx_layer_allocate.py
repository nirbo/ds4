#!/usr/bin/env python3
"""Allocate nonuniform per-layer expert budgets from held-out sensitivity curves."""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import sha256_file
from nemotron_mlx_layer_sensitivity import parse_plan
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-nonuniform-prune-plan-v1"


def layer_cost(summary: dict, robust_weight: float) -> float:
    mean = float(summary["mean_output_relative_l2"])
    maximum = float(summary["max_output_relative_l2"])
    return mean * mean + robust_weight * maximum * maximum


def monotonic_costs(
    sensitivity: dict,
    labels: list[str],
    layers: list[int],
    robust_weight: float,
) -> dict[int, dict[str, float]]:
    result = {}
    for layer in layers:
        running = 0.0
        result[layer] = {"r0": 0.0}
        for label in labels:
            running = max(
                running,
                layer_cost(sensitivity["summary"][label][str(layer)], robust_weight),
            )
            result[layer][label] = running
    return result


def allocate_exact(
    layers: list[int],
    retained_by_label: dict[str, int],
    costs: dict[int, dict[str, float]],
    target_total: int,
) -> tuple[dict[int, str], float]:
    states: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    labels = list(retained_by_label)
    for position, layer in enumerate(layers):
        remaining = len(layers) - position - 1
        minimum = min(retained_by_label.values()) * remaining
        maximum = max(retained_by_label.values()) * remaining
        next_states = {}
        for total, (cost, choices) in states.items():
            for label in labels:
                candidate_total = total + retained_by_label[label]
                if candidate_total + minimum > target_total or candidate_total + maximum < target_total:
                    continue
                candidate = (cost + costs[layer][label], choices + (label,))
                previous = next_states.get(candidate_total)
                if previous is None or candidate[0] < previous[0]:
                    next_states[candidate_total] = candidate
        require(next_states, f"target expert total {target_total} is unreachable")
        states = next_states
    require(target_total in states, f"target expert total {target_total} is unreachable")
    cost, labels_by_layer = states[target_total]
    return dict(zip(layers, labels_by_layer)), cost


def category_proxy(summary: dict, allocation: dict[int, str]) -> dict[str, float]:
    squared = collections.defaultdict(float)
    for layer, label in allocation.items():
        if label == "r0":
            continue
        for category, error in summary[label][str(layer)]["categories"].items():
            squared[category] += float(error) ** 2
    return {category: math.sqrt(value) for category, value in sorted(squared.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", required=True, type=Path)
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--robust-weight", type=float, default=0.25)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.robust_weight >= 0.0, "robust weight must be nonnegative")
        sensitivity = load_json(args.sensitivity)
        require(sensitivity.get("format") == "nemotron-layer-sensitivity-v1", "invalid sensitivity report")
        layers = sensitivity["layers"]
        specifications = [parse_plan(value) for value in args.plan]
        plans = {label: load_json(path) for label, path in specifications}
        require(len(plans) == len(specifications), "duplicate plan label")
        labels = sorted(plans, key=lambda label: plans[label]["new_num_experts"], reverse=True)
        previous = None
        for label in labels:
            plan = plans[label]
            require(
                sensitivity["plan_sha256"].get(label) == sha256_file(dict(specifications)[label]),
                f"sensitivity/plan hash mismatch: {label}",
            )
            require(plan["model_moe_layers"] == layers, f"plan layer mismatch: {label}")
            retained = plan["new_num_experts"]
            require(previous is None or retained < previous, "plan expert budgets are not unique")
            previous = retained
        require(all(target in plans for target in args.target), "target is not one of the supplied plans")
        old_experts = next(iter(plans.values()))["old_num_experts"]
        retained_by_label = {"r0": old_experts, **{label: plans[label]["new_num_experts"] for label in labels}}
        costs = monotonic_costs(sensitivity, labels, layers, args.robust_weight)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for target in args.target:
            target_total = plans[target]["new_num_experts"] * len(layers)
            allocation, objective = allocate_exact(layers, retained_by_label, costs, target_total)
            uniform = {layer: target for layer in layers}
            uniform_objective = sum(costs[layer][target] for layer in layers)
            kept_by_layer = {}
            dropped_by_layer = {}
            for layer, label in allocation.items():
                if label == "r0":
                    kept = list(range(old_experts))
                    dropped = []
                else:
                    kept = plans[label]["kept_by_layer"][str(layer)]
                    dropped = plans[label]["dropped_by_layer"][str(layer)]
                kept_by_layer[str(layer)] = kept
                dropped_by_layer[str(layer)] = dropped
            label_counts = collections.Counter(allocation.values())
            output = {
                "format": FORMAT,
                "source_revision": sensitivity["source_revision"],
                "sensitivity_sha256": sha256_file(args.sensitivity),
                "source_plan_sha256": sensitivity["plan_sha256"],
                "old_num_experts": old_experts,
                "model_moe_layers": layers,
                "target_uniform_label": target,
                "target_total_experts": target_total,
                "average_retained_experts": target_total / len(layers),
                "new_num_experts_by_layer": {
                    str(layer): len(kept_by_layer[str(layer)]) for layer in layers
                },
                "budget_label_by_layer": {str(layer): allocation[layer] for layer in layers},
                "budget_label_counts": dict(sorted(label_counts.items())),
                "kept_by_layer": kept_by_layer,
                "dropped_by_layer": dropped_by_layer,
                "old_to_new_by_layer": {
                    str(layer): {str(old): new for new, old in enumerate(kept_by_layer[str(layer)])}
                    for layer in layers
                },
                "allocation": {
                    "strategy": "heldout-monotonic-robust-output-error-dp",
                    "robust_weight": args.robust_weight,
                    "objective": objective,
                    "uniform_objective": uniform_objective,
                    "objective_improvement": 1.0 - objective / max(uniform_objective, 1e-30),
                    "category_error_proxy": category_proxy(sensitivity["summary"], allocation),
                    "uniform_category_error_proxy": category_proxy(sensitivity["summary"], uniform),
                    "unobserved_experts": "protected by every source plan",
                },
            }
            path = args.output_dir / f"plan-{target}-nonuniform.json"
            atomic_json(path, output)
            print(
                f"layer-allocation target={target} objective={objective:.6g} "
                f"uniform={uniform_objective:.6g} improvement={output['allocation']['objective_improvement']:.2%} "
                f"budgets={dict(sorted(label_counts.items()))} path={path} sha256={sha256_file(path)}"
            )
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        print(f"nemotron layer allocation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

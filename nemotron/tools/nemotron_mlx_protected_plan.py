#!/usr/bin/env python3
"""Add specialist expert protections to a nonuniform plan without changing its size."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import sha256_file
from nemotron_mlx_prune_plan import layer_importance
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-nonuniform-prune-plan-v1"


def specialist_importance(observation: dict, minimum_events: int) -> list[float]:
    scores = layer_importance(observation)
    return [
        score if int(observation["counts"][expert]) >= minimum_events else -math.inf
        for expert, score in enumerate(scores)
    ]


def protected_layer(
    template_kept: list[int],
    baseline_observation: dict,
    specialist_observation: dict,
    protection_fraction: float,
    base_core_fraction: float,
    max_swap_fraction: float,
    minimum_events: int,
    specialist_weight: float,
) -> tuple[list[int], dict]:
    expert_count = len(baseline_observation["counts"])
    require(len(specialist_observation["counts"]) == expert_count, "calibration expert-count mismatch")
    require(len(template_kept) == len(set(template_kept)), "template kept map contains duplicates")
    require(all(0 <= expert < expert_count for expert in template_kept), "template expert index out of range")
    retained_count = len(template_kept)
    if retained_count == expert_count:
        return sorted(template_kept), {
            "retained": retained_count,
            "swaps": 0,
            "added": [],
            "removed": [],
            "specialist_priority": [],
            "specialist_protected": [],
            "specialist_unretained": [],
            "base_core_protected": list(range(expert_count)),
            "positive_joint_gain": 0.0,
        }

    baseline_scores = layer_importance(baseline_observation)
    specialist_scores = specialist_importance(specialist_observation, minimum_events)
    baseline_unobserved = {
        expert for expert, count in enumerate(baseline_observation["counts"]) if int(count) == 0
    }
    require(baseline_unobserved <= set(template_kept), "template prunes baseline-unobserved experts")

    core_count = min(retained_count, math.ceil(retained_count * base_core_fraction))
    base_core = set(
        sorted(range(expert_count), key=lambda expert: (baseline_scores[expert], -expert), reverse=True)[
            :core_count
        ]
    ) | baseline_unobserved
    specialist_count = min(
        retained_count,
        math.ceil(retained_count * protection_fraction),
        sum(math.isfinite(score) for score in specialist_scores),
    )
    specialist_core = set(
        sorted(
            (expert for expert, score in enumerate(specialist_scores) if math.isfinite(score)),
            key=lambda expert: (specialist_scores[expert], -expert),
            reverse=True,
        )[:specialist_count]
    )

    current = set(template_kept)

    def joint_score(expert: int) -> float:
        specialist = specialist_scores[expert]
        return baseline_scores[expert] + specialist_weight * (
            specialist if math.isfinite(specialist) else 0.0
        )

    missing = sorted(
        specialist_core - current,
        key=lambda expert: (joint_score(expert), specialist_scores[expert], -expert),
        reverse=True,
    )
    evictable = sorted(
        current - base_core - specialist_core,
        key=lambda expert: (joint_score(expert), baseline_scores[expert], expert),
    )
    swap_limit = math.floor(retained_count * max_swap_fraction)
    added = []
    removed = []
    positive_joint_gain = 0.0
    for candidate, victim in zip(missing, evictable):
        if len(added) >= swap_limit:
            break
        gain = joint_score(candidate) - joint_score(victim)
        if gain <= 0.0:
            break
        added.append(candidate)
        removed.append(victim)
        positive_joint_gain += gain
    swaps = len(added)
    current.difference_update(removed)
    current.update(added)
    require(len(current) == retained_count, "protected plan changed retained expert count")
    require(baseline_unobserved <= current, "protected plan dropped baseline-unobserved expert")
    require(base_core <= current, "protected plan dropped a baseline core expert")
    return sorted(current), {
        "retained": retained_count,
        "swaps": swaps,
        "added": added,
        "removed": removed,
        "specialist_priority": sorted(specialist_core),
        "specialist_protected": sorted(specialist_core & current),
        "specialist_unretained": sorted(specialist_core - current),
        "base_core_protected": sorted(base_core),
        "positive_joint_gain": positive_joint_gain,
    }


def build_protected_plan(
    template: dict,
    baseline: dict,
    specialist: dict,
    protection_fraction: float = 0.25,
    base_core_fraction: float = 0.50,
    max_swap_fraction: float = 0.02,
    minimum_events: int = 2,
    specialist_weight: float = 0.25,
) -> dict:
    require(template.get("format") == FORMAT, "unsupported template plan format")
    require(baseline.get("format") == "nemotron-mlx-calibration-v1", "invalid baseline calibration")
    require(specialist.get("format") == "nemotron-mlx-calibration-v1", "invalid specialist calibration")
    require(0.0 < protection_fraction <= 1.0, "protection fraction must be in (0, 1]")
    require(0.0 < base_core_fraction <= 1.0, "base core fraction must be in (0, 1]")
    require(0.0 <= max_swap_fraction <= 1.0, "max swap fraction must be in [0, 1]")
    require(minimum_events > 0, "minimum events must be positive")
    require(specialist_weight >= 0.0, "specialist weight must be nonnegative")
    revisions = {template.get("source_revision"), baseline.get("source_revision"), specialist.get("source_revision")}
    require(len(revisions) == 1 and None not in revisions, "source revision mismatch")
    layers = [str(layer) for layer in template["model_moe_layers"]]
    require(set(layers) == set(baseline["layers"]) == set(specialist["layers"]), "calibration layer mismatch")

    kept_by_layer = {}
    reports = {}
    for layer in layers:
        kept, report = protected_layer(
            template["kept_by_layer"][layer],
            baseline["layers"][layer],
            specialist["layers"][layer],
            protection_fraction,
            base_core_fraction,
            max_swap_fraction,
            minimum_events,
            specialist_weight,
        )
        kept_by_layer[layer] = kept
        reports[layer] = report
    old_experts = int(template["old_num_experts"])
    dropped_by_layer = {
        layer: sorted(set(range(old_experts)) - set(kept)) for layer, kept in kept_by_layer.items()
    }
    output = dict(template)
    output["kept_by_layer"] = kept_by_layer
    output["dropped_by_layer"] = dropped_by_layer
    output["old_to_new_by_layer"] = {
        layer: {str(old): new for new, old in enumerate(kept)} for layer, kept in kept_by_layer.items()
    }
    output["protection"] = {
        "strategy": "fixed-budget-specialist-reservation-v1",
        "protection_fraction": protection_fraction,
        "base_core_fraction": base_core_fraction,
        "max_swap_fraction": max_swap_fraction,
        "minimum_specialist_events": minimum_events,
        "specialist_weight": specialist_weight,
        "baseline_calibration_sha256": baseline.get("_sha256"),
        "specialist_calibration_sha256": specialist.get("_sha256"),
        "total_swaps": sum(report["swaps"] for report in reports.values()),
        "layers_changed": sum(report["swaps"] > 0 for report in reports.values()),
        "by_layer": reports,
        "size_invariant": "retained expert count is unchanged in every layer",
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-plan", required=True, type=Path)
    parser.add_argument("--baseline-calibration", required=True, type=Path)
    parser.add_argument("--specialist-calibration", required=True, type=Path)
    parser.add_argument("--protection-fraction", type=float, default=0.25)
    parser.add_argument("--base-core-fraction", type=float, default=0.50)
    parser.add_argument("--max-swap-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-events", type=int, default=2)
    parser.add_argument("--specialist-weight", type=float, default=0.25)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        template = load_json(args.template_plan)
        baseline = load_json(args.baseline_calibration)
        specialist = load_json(args.specialist_calibration)
        baseline["_sha256"] = sha256_file(args.baseline_calibration)
        specialist["_sha256"] = sha256_file(args.specialist_calibration)
        plan = build_protected_plan(
            template,
            baseline,
            specialist,
            args.protection_fraction,
            args.base_core_fraction,
            args.max_swap_fraction,
            args.minimum_events,
            args.specialist_weight,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, plan)
        protection = plan["protection"]
        print(
            f"nemotron protected plan: swaps={protection['total_swaps']} "
            f"layers_changed={protection['layers_changed']} total_experts={plan['target_total_experts']} "
            f"output={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        print(f"nemotron protected plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Swap success-attributed experts into a template without changing its size."""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_prune_plan import layer_importance
from nemotron_prune_materialize import atomic_json, sha256_file


ATTRIBUTION_FORMAT = "nemotron-trajectory-attribution-v1"
PLAN_FORMAT = "nemotron-nonuniform-prune-plan-v1"


def tied_rank_fraction(values: list[float]) -> list[float]:
    """Return average percentile ranks while assigning equal values equal ranks."""
    require(values, "cannot rank an empty value list")
    order = sorted(range(len(values)), key=lambda index: values[index])
    denominator = max(1, len(values) - 1)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + end - 1) / 2) / denominator
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def aggregate_trajectory_scores(
    reports: list[dict],
    label: str,
    template_hash: str,
    expert_count: int,
) -> dict[int, list[float]]:
    require(reports, "no trajectory reports supplied")
    trajectories = set()
    totals: dict[int, list[float]] = collections.defaultdict(lambda: [0.0] * expert_count)
    expected_layers = None
    for report in reports:
        require(report.get("format") == ATTRIBUTION_FORMAT, "invalid trajectory report format")
        require(report.get("status") == "complete", "trajectory report is incomplete")
        require(report.get("plan_sha256", {}).get(label) == template_hash, "trajectory/template hash mismatch")
        trajectory = report.get("capture", {}).get("identity", {}).get("trajectory", {})
        identity = (
            str(trajectory.get("trajectory_format", "livecodebench")),
            str(trajectory.get("task_id")),
            int(trajectory.get("repeat", -1)),
        )
        require(identity not in trajectories, f"duplicate trajectory report: {identity}")
        trajectories.add(identity)
        layers = [int(row["layer"]) for row in report.get("layers", [])]
        require(layers and len(layers) == len(set(layers)), "trajectory layers are missing or duplicated")
        require(expected_layers is None or layers == expected_layers, "trajectory layer catalogs differ")
        expected_layers = layers
        for row in report["layers"]:
            layer = int(row["layer"])
            result = row.get("plans", {}).get(label)
            require(isinstance(result, dict), f"trajectory has no plan result: {label}")
            importance = {
                int(expert): float(score)
                for expert, score in result.get("selected_expert_importance", {}).items()
                if float(score) > 0.0
            }
            require(all(0 <= expert < expert_count for expert in importance), "trajectory expert index out of range")
            total = sum(importance.values())
            base_error = float(result["curves"][0]["output"]["relative_l2"])
            if total <= 0.0 or base_error <= 0.0:
                continue
            for expert, value in importance.items():
                totals[layer][expert] += base_error * value / total
    return {
        layer: [value / len(reports) for value in values]
        for layer, values in totals.items()
    }


def top_fraction(experts: list[int], scores: list[float], fraction: float) -> set[int]:
    count = min(len(experts), math.ceil(len(experts) * fraction))
    return set(sorted(experts, key=lambda expert: (scores[expert], -expert), reverse=True)[:count])


def build_swap_plan(
    template: dict,
    addback: dict,
    baseline: dict,
    recovery_reports: list[dict],
    guard_reports: list[dict],
    recovery_hashes: list[str],
    guard_hashes: list[str],
    template_hash: str,
    addback_hash: str,
    baseline_hash: str,
    label: str = "r25",
    base_core_fraction: float = 0.50,
    trajectory_core_fraction: float = 0.10,
    baseline_weight: float = 0.50,
    recovery_weight: float = 0.25,
    guard_weight: float = 0.25,
    specialist: dict | None = None,
    specialist_hash: str | None = None,
    specialist_core_fraction: float = 0.10,
    specialist_weight: float = 0.0,
    guard_core_fraction: float | None = None,
) -> dict:
    require(template.get("format") == PLAN_FORMAT, "unsupported template plan format")
    require(addback.get("format") == PLAN_FORMAT, "unsupported addback plan format")
    require(baseline.get("format") == "nemotron-mlx-calibration-v1", "invalid baseline calibration")
    require(0.0 <= base_core_fraction <= 1.0, "invalid base core fraction")
    require(0.0 <= trajectory_core_fraction <= 1.0, "invalid trajectory core fraction")
    guard_core_fraction = (
        trajectory_core_fraction if guard_core_fraction is None else guard_core_fraction
    )
    require(0.0 <= guard_core_fraction <= 1.0, "invalid guard core fraction")
    require(0.0 <= specialist_core_fraction <= 1.0, "invalid specialist core fraction")
    require(specialist is not None or specialist_weight == 0.0, "specialist weight requires calibration")
    weights = [baseline_weight, recovery_weight, guard_weight, specialist_weight]
    require(all(weight >= 0.0 for weight in weights) and sum(weights) > 0.0, "invalid evidence weights")
    revisions = {template.get("source_revision"), addback.get("source_revision"), baseline.get("source_revision")}
    if specialist is not None:
        require(specialist.get("format") == "nemotron-mlx-calibration-v1", "invalid specialist calibration")
        revisions.add(specialist.get("source_revision"))
    require(len(revisions) == 1 and None not in revisions, "source revision mismatch")
    require(template["model_moe_layers"] == addback["model_moe_layers"], "plan layer mismatch")
    layers = [int(layer) for layer in template["model_moe_layers"]]
    require(set(map(str, layers)) == set(baseline["layers"]), "baseline layer mismatch")
    if specialist is not None:
        require(set(map(str, layers)) == set(specialist["layers"]), "specialist layer mismatch")
    expert_count = int(template["old_num_experts"])
    require(int(addback["old_num_experts"]) == expert_count, "plan expert-count mismatch")

    recovery = aggregate_trajectory_scores(
        recovery_reports, label, template_hash, expert_count
    )
    guards = aggregate_trajectory_scores(guard_reports, label, template_hash, expert_count)
    kept_by_layer = {}
    reports = {}
    for layer in layers:
        key = str(layer)
        template_kept = set(template["kept_by_layer"][key])
        addback_kept = set(addback["kept_by_layer"][key])
        require(template_kept <= addback_kept, f"addback plan removes template expert in layer {layer}")
        additions = sorted(addback_kept - template_kept)
        if not additions:
            kept_by_layer[key] = sorted(template_kept)
            reports[key] = {
                "added": [],
                "removed": [],
                "base_core_protected": [],
                "trajectory_core_protected": [],
                "positive_joint_swaps": 0,
            }
            continue

        observation = baseline["layers"][key]
        require(len(observation["counts"]) == expert_count, f"baseline expert-count mismatch in layer {layer}")
        baseline_scores = layer_importance(observation)
        recovery_scores = tied_rank_fraction(recovery.get(layer, [0.0] * expert_count))
        guard_scores = tied_rank_fraction(guards.get(layer, [0.0] * expert_count))
        baseline_ranks = tied_rank_fraction([
            score if math.isfinite(score) else 1.0 for score in baseline_scores
        ])
        if specialist is None:
            specialist_ranks = [0.5] * expert_count
        else:
            specialist_observation = specialist["layers"][key]
            require(
                len(specialist_observation["counts"]) == expert_count,
                f"specialist expert-count mismatch in layer {layer}",
            )
            raw_specialist = layer_importance(specialist_observation)
            observed_values = [
                score for score in raw_specialist if math.isfinite(score)
            ]
            neutral = sum(observed_values) / max(1, len(observed_values))
            specialist_ranks = tied_rank_fraction([
                score if math.isfinite(score) else neutral for score in raw_specialist
            ])
        template_list = sorted(template_kept)
        base_core = top_fraction(template_list, baseline_ranks, base_core_fraction)
        trajectory_core = (
            top_fraction(template_list, recovery_scores, trajectory_core_fraction)
            | top_fraction(template_list, guard_scores, guard_core_fraction)
        )
        specialist_core = (
            set()
            if specialist is None
            else top_fraction(template_list, specialist_ranks, specialist_core_fraction)
        )
        unobserved = {
            expert for expert, count in enumerate(observation["counts"]) if int(count) == 0
        }
        require(unobserved <= template_kept, f"template prunes baseline-unobserved expert in layer {layer}")
        protected = base_core | trajectory_core | specialist_core | unobserved

        def joint_score(expert: int) -> float:
            return (
                baseline_weight * baseline_ranks[expert]
                + recovery_weight * recovery_scores[expert]
                + guard_weight * guard_scores[expert]
                + specialist_weight * specialist_ranks[expert]
            ) / sum(weights)

        evictable = sorted(
            template_kept - protected,
            key=lambda expert: (joint_score(expert), baseline_ranks[expert], expert),
        )
        require(len(evictable) >= len(additions), f"insufficient safe eviction capacity in layer {layer}")
        removed = evictable[: len(additions)]
        kept = (template_kept - set(removed)) | set(additions)
        require(len(kept) == len(template_kept), f"swap changed layer {layer} expert count")
        require(protected <= kept, f"swap removed protected expert in layer {layer}")
        kept_by_layer[key] = sorted(kept)
        reports[key] = {
            "added": additions,
            "removed": removed,
            "base_core_protected": sorted(base_core),
            "trajectory_core_protected": sorted(trajectory_core),
            "specialist_core_protected": sorted(specialist_core),
            "unobserved_protected": sorted(unobserved),
            "positive_joint_swaps": sum(
                joint_score(added) > joint_score(removed_expert)
                for added, removed_expert in zip(
                    sorted(additions, key=lambda expert: joint_score(expert), reverse=True),
                    removed,
                )
            ),
            "added_joint_score": {str(expert): joint_score(expert) for expert in additions},
            "removed_joint_score": {str(expert): joint_score(expert) for expert in removed},
        }

    output = dict(template)
    output["kept_by_layer"] = kept_by_layer
    output["dropped_by_layer"] = {
        key: sorted(set(range(expert_count)) - set(kept))
        for key, kept in kept_by_layer.items()
    }
    output["old_to_new_by_layer"] = {
        key: {str(old): new for new, old in enumerate(kept)}
        for key, kept in kept_by_layer.items()
    }
    output["new_num_experts_by_layer"] = {
        key: len(kept) for key, kept in kept_by_layer.items()
    }
    output["target_total_experts"] = sum(len(kept) for kept in kept_by_layer.values())
    output["average_retained_experts"] = output["target_total_experts"] / len(layers)
    output["trajectory_swap"] = {
        "strategy": "fixed-budget-success-trajectory-swap-v1",
        "tool_sha256": sha256_file(Path(__file__)),
        "template_plan_sha256": template_hash,
        "addback_plan_sha256": addback_hash,
        "baseline_calibration_sha256": baseline_hash,
        "specialist_calibration_sha256": specialist_hash,
        "recovery_report_sha256": recovery_hashes,
        "guard_report_sha256": guard_hashes,
        "plan_label": label,
        "base_core_fraction": base_core_fraction,
        "trajectory_core_fraction": trajectory_core_fraction,
        "guard_core_fraction": guard_core_fraction,
        "specialist_core_fraction": specialist_core_fraction,
        "weights": {
            "baseline": baseline_weight,
            "recovery": recovery_weight,
            "guard": guard_weight,
            "specialist": specialist_weight,
        },
        "total_swaps": sum(len(report["added"]) for report in reports.values()),
        "layers_changed": sum(bool(report["added"]) for report in reports.values()),
        "positive_joint_swaps": sum(report["positive_joint_swaps"] for report in reports.values()),
        "by_layer": reports,
        "size_policy": "exact template expert count in every layer",
    }
    require(output["target_total_experts"] == int(template["target_total_experts"]), "swap changed total expert count")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-plan", required=True, type=Path)
    parser.add_argument("--addback-plan", required=True, type=Path)
    parser.add_argument("--baseline-calibration", required=True, type=Path)
    parser.add_argument("--specialist-calibration", type=Path)
    parser.add_argument("--recovery-report", action="append", required=True, type=Path)
    parser.add_argument("--guard-report", action="append", required=True, type=Path)
    parser.add_argument("--plan-label", default="r25")
    parser.add_argument("--base-core-fraction", type=float, default=0.50)
    parser.add_argument("--trajectory-core-fraction", type=float, default=0.10)
    parser.add_argument("--guard-core-fraction", type=float)
    parser.add_argument("--specialist-core-fraction", type=float, default=0.10)
    parser.add_argument("--baseline-weight", type=float, default=0.50)
    parser.add_argument("--recovery-weight", type=float, default=0.25)
    parser.add_argument("--guard-weight", type=float, default=0.25)
    parser.add_argument("--specialist-weight", type=float, default=0.25)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        template = load_json(args.template_plan)
        addback = load_json(args.addback_plan)
        baseline = load_json(args.baseline_calibration)
        specialist = (
            None if args.specialist_calibration is None else load_json(args.specialist_calibration)
        )
        recovery_reports = [load_json(path) for path in args.recovery_report]
        guard_reports = [load_json(path) for path in args.guard_report]
        plan = build_swap_plan(
            template,
            addback,
            baseline,
            recovery_reports,
            guard_reports,
            [sha256_file(path) for path in args.recovery_report],
            [sha256_file(path) for path in args.guard_report],
            sha256_file(args.template_plan),
            sha256_file(args.addback_plan),
            sha256_file(args.baseline_calibration),
            args.plan_label,
            args.base_core_fraction,
            args.trajectory_core_fraction,
            args.baseline_weight,
            args.recovery_weight,
            args.guard_weight,
            specialist=specialist,
            specialist_hash=(
                None
                if args.specialist_calibration is None
                else sha256_file(args.specialist_calibration)
            ),
            specialist_core_fraction=args.specialist_core_fraction,
            specialist_weight=(
                0.0 if args.specialist_calibration is None else args.specialist_weight
            ),
            guard_core_fraction=args.guard_core_fraction,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, plan)
        swap = plan["trajectory_swap"]
        print(
            f"trajectory-swap swaps={swap['total_swaps']} layers={swap['layers_changed']} "
            f"positive_joint={swap['positive_joint_swaps']} average={plan['average_retained_experts']:.3f} "
            f"output={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron trajectory swap plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Thin a protected prune plan without changing any surviving expert identity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_prune_plan import layer_importance
from nemotron_mlx_trajectory_swap_plan import tied_rank_fraction
from nemotron_prune_materialize import atomic_json, sha256_file


PLAN_FORMAT = "nemotron-nonuniform-prune-plan-v1"


def aggregate_trajectory_importance(
    reports: list[dict],
    layers: list[int],
    expert_count: int,
    expected_plan_hash: str,
) -> dict[str, list[float]]:
    scores = {str(layer): [0.0] * expert_count for layer in layers}
    for report in reports:
        require(
            report.get("format") == "nemotron-trajectory-attribution-v1",
            "invalid trajectory report",
        )
        require(report.get("status") == "complete", "incomplete trajectory report")
        require(
            expected_plan_hash in report.get("plan_sha256", {}).values(),
            "trajectory/plan hash mismatch",
        )
        require(
            [int(row["layer"]) for row in report["layers"]] == layers,
            "trajectory layer mismatch",
        )
        for row in report["layers"]:
            plan_rows = row.get("plans", {})
            require(plan_rows, "trajectory report has no plan evidence")
            evidence = next(iter(plan_rows.values()))["selected_expert_importance"]
            layer_scores = scores[str(row["layer"])]
            for expert, value in evidence.items():
                layer_scores[int(expert)] += float(value)
    return scores


def build_nested_plan(
    template: dict,
    baseline: dict,
    specialist: dict,
    remove_experts: int,
    maximum_per_layer: int,
    template_hash: str,
    baseline_hash: str,
    specialist_hash: str,
    baseline_weight: float = 0.65,
    specialist_weight: float = 0.35,
    trajectory_reports: list[dict] | None = None,
    trajectory_hashes: list[str] | None = None,
    trajectory_weight: float = 0.0,
) -> dict:
    require(template.get("format") == PLAN_FORMAT, "unsupported template plan format")
    require(baseline.get("format") == "nemotron-mlx-calibration-v1", "invalid baseline calibration")
    require(specialist.get("format") == "nemotron-mlx-calibration-v1", "invalid specialist calibration")
    require(remove_experts > 0 and maximum_per_layer > 0, "invalid thinning budget")
    require(
        baseline_weight >= 0 and specialist_weight >= 0 and trajectory_weight >= 0,
        "invalid evidence weight",
    )
    require(
        baseline_weight + specialist_weight + trajectory_weight > 0,
        "evidence weights sum to zero",
    )
    trajectory_reports = trajectory_reports or []
    trajectory_hashes = trajectory_hashes or []
    require(len(trajectory_reports) == len(trajectory_hashes), "trajectory hash mismatch")
    require(trajectory_weight == 0 or trajectory_reports, "trajectory weight requires reports")
    revisions = {
        template.get("source_revision"),
        baseline.get("source_revision"),
        specialist.get("source_revision"),
    }
    require(len(revisions) == 1 and None not in revisions, "source revision mismatch")
    layers = [int(layer) for layer in template["model_moe_layers"]]
    require(set(map(str, layers)) == set(baseline["layers"]), "baseline layer mismatch")
    require(set(map(str, layers)) == set(specialist["layers"]), "specialist layer mismatch")
    protection = template.get("trajectory_swap", {}).get("by_layer", {})
    require(set(protection) == set(map(str, layers)), "template has no complete protection catalog")

    trajectory_scores = aggregate_trajectory_importance(
        trajectory_reports,
        layers,
        int(template["old_num_experts"]),
        template_hash,
    )

    candidates = []
    protected_by_layer = {}
    for layer in layers:
        key = str(layer)
        kept = set(template["kept_by_layer"][key])
        row = protection[key]
        protected = set()
        for field in (
            "base_core_protected",
            "trajectory_core_protected",
            "specialist_core_protected",
            "unobserved_protected",
        ):
            protected.update(map(int, row.get(field, [])))
        if template.get("budget_label_by_layer", {}).get(key) == "r0":
            protected = set(kept)
        protected &= kept
        protected_by_layer[key] = sorted(protected)

        broad_scores = layer_importance(baseline["layers"][key])
        specialist_scores = layer_importance(specialist["layers"][key])
        broad_ranks = tied_rank_fraction(broad_scores)
        specialist_ranks = tied_rank_fraction(specialist_scores)
        trajectory_ranks = tied_rank_fraction(trajectory_scores[key])
        weight_total = baseline_weight + specialist_weight + trajectory_weight
        ranked = []
        for expert in kept - protected:
            score = (
                baseline_weight * broad_ranks[expert]
                + specialist_weight * specialist_ranks[expert]
                + trajectory_weight * trajectory_ranks[expert]
            ) / weight_total
            ranked.append(
                (
                    score,
                    broad_ranks[expert],
                    specialist_ranks[expert],
                    trajectory_ranks[expert],
                    expert,
                )
            )
        ranked.sort()
        for score, broad, special, trajectory, expert in ranked[:maximum_per_layer]:
            candidates.append((score, broad, special, trajectory, layer, expert))

    require(len(candidates) >= remove_experts, "insufficient protected thinning capacity")
    selected = sorted(candidates)[:remove_experts]
    removed_by_layer = {str(layer): [] for layer in layers}
    for _, _, _, _, layer, expert in selected:
        removed_by_layer[str(layer)].append(expert)
    kept_by_layer = {
        str(layer): sorted(
            set(template["kept_by_layer"][str(layer)]) - set(removed_by_layer[str(layer)])
        )
        for layer in layers
    }
    expert_count = int(template["old_num_experts"])
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
    output["target_total_experts"] = sum(map(len, kept_by_layer.values()))
    output["average_retained_experts"] = output["target_total_experts"] / len(layers)
    output["nested_thinning"] = {
        "strategy": "protected-nested-thinning-v1",
        "tool_sha256": sha256_file(Path(__file__)),
        "template_plan_sha256": template_hash,
        "baseline_calibration_sha256": baseline_hash,
        "specialist_calibration_sha256": specialist_hash,
        "trajectory_report_sha256": trajectory_hashes,
        "removed_experts": remove_experts,
        "maximum_removed_per_layer": maximum_per_layer,
        "weights": {
            "baseline": baseline_weight,
            "specialist": specialist_weight,
            "trajectory": trajectory_weight,
        },
        "removed_by_layer": {key: sorted(values) for key, values in removed_by_layer.items()},
        "protected_by_layer": protected_by_layer,
        "size_policy": "strict subset of template survivors; r0 layers untouched",
    }
    require(
        int(template["target_total_experts"]) - output["target_total_experts"] == remove_experts,
        "thinning changed the wrong number of experts",
    )
    return output


def build_repaired_plan(
    parent: dict,
    candidate: dict,
    baseline: dict,
    specialist: dict,
    trajectory_reports: list[dict],
    trajectory_hashes: list[str],
    swaps: int,
    parent_hash: str,
    candidate_hash: str,
    baseline_hash: str,
    specialist_hash: str,
) -> dict:
    require(parent.get("format") == candidate.get("format") == PLAN_FORMAT, "invalid repair plan")
    require(parent.get("source_revision") == candidate.get("source_revision"), "repair revision mismatch")
    require(swaps > 0 and trajectory_reports, "repair requires swaps and trajectory reports")
    require(len(trajectory_reports) == len(trajectory_hashes), "repair trajectory hash mismatch")
    require(
        baseline.get("format") == specialist.get("format") == "nemotron-mlx-calibration-v1",
        "invalid repair calibration",
    )
    require(
        parent.get("source_revision")
        == baseline.get("source_revision")
        == specialist.get("source_revision"),
        "repair calibration revision mismatch",
    )
    layers = [int(layer) for layer in parent["model_moe_layers"]]
    require(candidate["model_moe_layers"] == parent["model_moe_layers"], "repair layer mismatch")
    expert_count = int(parent["old_num_experts"])
    require(int(candidate["old_num_experts"]) == expert_count, "repair expert-count mismatch")
    require(set(map(str, layers)) == set(baseline["layers"]), "repair baseline layer mismatch")
    require(set(map(str, layers)) == set(specialist["layers"]), "repair specialist layer mismatch")
    trajectory = aggregate_trajectory_importance(
        trajectory_reports, layers, expert_count, candidate_hash
    )
    protected_catalog = candidate.get("nested_thinning", {}).get("protected_by_layer", {})
    require(set(protected_catalog) == set(map(str, layers)), "repair candidate lacks protection catalog")

    pairs = []
    for layer in layers:
        key = str(layer)
        parent_kept = set(parent["kept_by_layer"][key])
        candidate_kept = set(candidate["kept_by_layer"][key])
        require(candidate_kept <= parent_kept, f"repair candidate is not nested in layer {layer}")
        additions = parent_kept - candidate_kept
        if not additions:
            continue
        protected = set(protected_catalog[key])
        broad = tied_rank_fraction(layer_importance(baseline["layers"][key]))
        special = tied_rank_fraction(layer_importance(specialist["layers"][key]))
        trajectory_rank = tied_rank_fraction(trajectory[key])
        additions_ranked = sorted(additions, key=lambda expert: (-trajectory_rank[expert], expert))
        evictions_ranked = sorted(
            candidate_kept - protected,
            key=lambda expert: (
                0.45 * broad[expert] + 0.25 * special[expert] + 0.30 * trajectory_rank[expert],
                expert,
            ),
        )
        for added, removed in zip(additions_ranked, evictions_ranked):
            gain = trajectory_rank[added] - trajectory_rank[removed]
            if gain > 0:
                pairs.append((gain, layer, added, removed))
    require(len(pairs) >= swaps, "insufficient positive repair swaps")
    selected = sorted(pairs, key=lambda item: (-item[0], item[1], item[2], item[3]))[:swaps]
    kept_by_layer = {key: list(values) for key, values in candidate["kept_by_layer"].items()}
    by_layer = {str(layer): {"added": [], "removed": []} for layer in layers}
    for _, layer, added, removed in selected:
        key = str(layer)
        kept = set(kept_by_layer[key])
        require(removed in kept and added not in kept, "repair swap collision")
        kept.remove(removed)
        kept.add(added)
        kept_by_layer[key] = sorted(kept)
        by_layer[key]["added"].append(added)
        by_layer[key]["removed"].append(removed)

    output = dict(candidate)
    output["kept_by_layer"] = kept_by_layer
    output["dropped_by_layer"] = {
        key: sorted(set(range(expert_count)) - set(kept)) for key, kept in kept_by_layer.items()
    }
    output["old_to_new_by_layer"] = {
        key: {str(old): new for new, old in enumerate(kept)} for key, kept in kept_by_layer.items()
    }
    output["new_num_experts_by_layer"] = {key: len(kept) for key, kept in kept_by_layer.items()}
    output["nested_repair"] = {
        "strategy": "bounded-trajectory-repair-v1",
        "tool_sha256": sha256_file(Path(__file__)),
        "parent_plan_sha256": parent_hash,
        "candidate_plan_sha256": candidate_hash,
        "baseline_calibration_sha256": baseline_hash,
        "specialist_calibration_sha256": specialist_hash,
        "trajectory_report_sha256": trajectory_hashes,
        "swaps": swaps,
        "layers_changed": sum(bool(row["added"]) for row in by_layer.values()),
        "by_layer": by_layer,
        "size_policy": "same-layer swaps preserve candidate expert counts",
    }
    require(output["target_total_experts"] == candidate["target_total_experts"], "repair changed size")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-plan", required=True, type=Path)
    parser.add_argument("--baseline-calibration", required=True, type=Path)
    parser.add_argument("--specialist-calibration", required=True, type=Path)
    parser.add_argument("--remove-experts", type=int, default=0)
    parser.add_argument("--maximum-per-layer", type=int, default=25)
    parser.add_argument("--baseline-weight", type=float, default=0.65)
    parser.add_argument("--specialist-weight", type=float, default=0.35)
    parser.add_argument("--trajectory-report", action="append", type=Path, default=[])
    parser.add_argument("--trajectory-weight", type=float, default=0.0)
    parser.add_argument("--repair-plan", type=Path)
    parser.add_argument("--repair-swaps", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        template = load_json(args.template_plan)
        baseline = load_json(args.baseline_calibration)
        specialist = load_json(args.specialist_calibration)
        trajectory_reports = [load_json(path) for path in args.trajectory_report]
        trajectory_hashes = [sha256_file(path) for path in args.trajectory_report]
        if args.repair_plan is not None:
            require(args.repair_swaps > 0, "repair plan requires --repair-swaps")
            plan = build_repaired_plan(
                template,
                load_json(args.repair_plan),
                baseline,
                specialist,
                trajectory_reports,
                trajectory_hashes,
                args.repair_swaps,
                sha256_file(args.template_plan),
                sha256_file(args.repair_plan),
                sha256_file(args.baseline_calibration),
                sha256_file(args.specialist_calibration),
            )
        else:
            require(args.remove_experts > 0, "nested thinning requires --remove-experts")
            plan = build_nested_plan(
                template,
                baseline,
                specialist,
                args.remove_experts,
                args.maximum_per_layer,
                sha256_file(args.template_plan),
                sha256_file(args.baseline_calibration),
                sha256_file(args.specialist_calibration),
                args.baseline_weight,
                args.specialist_weight,
                trajectory_reports,
                trajectory_hashes,
                args.trajectory_weight,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, plan)
        print(
            f"nested-thin removed={args.remove_experts} repaired={args.repair_swaps} "
            f"average={plan['average_retained_experts']:.3f} "
            f"output={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron nested thinning error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

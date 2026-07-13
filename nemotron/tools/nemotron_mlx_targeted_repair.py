#!/usr/bin/env python3
"""Apply bounded recovery swaps to an exact-size nested expert plan."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_prune_plan import layer_importance
from nemotron_mlx_trajectory_swap_plan import tied_rank_fraction
from nemotron_prune_materialize import atomic_json, sha256_file


PLAN_FORMAT = "nemotron-nonuniform-prune-plan-v1"
REPORT_FORMAT = "nemotron-trajectory-attribution-v1"


def aggregate_reports(
    reports: list[dict], layers: list[int], expert_count: int, plan_hash: str
) -> dict[str, list[float]]:
    require(reports, "no trajectory reports supplied")
    totals = {str(layer): [0.0] * expert_count for layer in layers}
    identities = set()
    for report in reports:
        require(report.get("format") == REPORT_FORMAT, "invalid trajectory report")
        require(report.get("status") == "complete", "incomplete trajectory report")
        labels = [label for label, digest in report.get("plan_sha256", {}).items() if digest == plan_hash]
        require(len(labels) == 1, "trajectory/plan hash mismatch")
        label = labels[0]
        trajectory = report["capture"]["identity"]["trajectory"]
        identity = (
            trajectory["trajectory_format"],
            trajectory["task_id"],
            int(trajectory["repeat"]),
        )
        require(identity not in identities, f"duplicate trajectory: {identity}")
        identities.add(identity)
        require([int(row["layer"]) for row in report["layers"]] == layers, "layer mismatch")
        for row in report["layers"]:
            scores = totals[str(row["layer"])]
            evidence = row["plans"][label]["selected_expert_importance"]
            for expert, value in evidence.items():
                scores[int(expert)] += float(value)
    return totals


def top_experts(experts: set[int], ranks: list[float], fraction: float) -> set[int]:
    count = min(len(experts), math.ceil(len(experts) * fraction))
    return set(sorted(experts, key=lambda expert: (-ranks[expert], expert))[:count])


def build_targeted_repair(
    parent: dict,
    candidate: dict,
    baseline: dict,
    specialist: dict,
    recovery_reports: list[dict],
    guard_reports: list[dict],
    recovery_hashes: list[str],
    guard_hashes: list[str],
    swaps: int,
    parent_hash: str,
    candidate_hash: str,
    baseline_hash: str,
    specialist_hash: str,
    guard_core_fraction: float = 0.20,
    guard_tolerance: float = 0.10,
) -> dict:
    require(parent.get("format") == candidate.get("format") == PLAN_FORMAT, "invalid plan")
    require(swaps > 0, "repair swaps must be positive")
    require(0 <= guard_core_fraction <= 1 and guard_tolerance >= 0, "invalid guard policy")
    revision = parent.get("source_revision")
    require(
        revision
        == candidate.get("source_revision")
        == baseline.get("source_revision")
        == specialist.get("source_revision"),
        "source revision mismatch",
    )
    require(
        baseline.get("format") == specialist.get("format") == "nemotron-mlx-calibration-v1",
        "invalid calibration",
    )
    require(len(recovery_reports) == len(recovery_hashes), "recovery hash mismatch")
    require(len(guard_reports) == len(guard_hashes), "guard hash mismatch")
    layers = [int(layer) for layer in parent["model_moe_layers"]]
    require(candidate["model_moe_layers"] == parent["model_moe_layers"], "plan layer mismatch")
    require(set(map(str, layers)) == set(baseline["layers"]), "baseline layer mismatch")
    require(set(map(str, layers)) == set(specialist["layers"]), "specialist layer mismatch")
    expert_count = int(parent["old_num_experts"])
    require(int(candidate["old_num_experts"]) == expert_count, "expert-count mismatch")
    recovery = aggregate_reports(recovery_reports, layers, expert_count, candidate_hash)
    guards = aggregate_reports(guard_reports, layers, expert_count, candidate_hash)
    protected_catalog = candidate.get("nested_thinning", {}).get("protected_by_layer", {})
    require(set(protected_catalog) == set(map(str, layers)), "missing protection catalog")

    pairs = []
    layer_evidence = {}
    for layer in layers:
        key = str(layer)
        parent_kept = set(parent["kept_by_layer"][key])
        candidate_kept = set(candidate["kept_by_layer"][key])
        require(candidate_kept <= parent_kept, f"candidate is not nested in layer {layer}")
        additions = parent_kept - candidate_kept
        broad = tied_rank_fraction(layer_importance(baseline["layers"][key]))
        special = tied_rank_fraction(layer_importance(specialist["layers"][key]))
        recover = tied_rank_fraction(recovery[key])
        guard = tied_rank_fraction(guards[key])
        protected = set(protected_catalog[key]) | top_experts(
            candidate_kept, guard, guard_core_fraction
        )

        def score(expert: int) -> float:
            return 0.50 * recover[expert] + 0.20 * guard[expert] + 0.20 * broad[expert] + 0.10 * special[expert]

        add_ranked = sorted(additions, key=lambda expert: (-score(expert), expert))
        remove_ranked = sorted(candidate_kept - protected, key=lambda expert: (score(expert), expert))
        accepted = 0
        for added, removed in zip(add_ranked, remove_ranked):
            recovery_gain = recover[added] - recover[removed]
            guard_delta = guard[added] - guard[removed]
            joint_gain = score(added) - score(removed)
            if recovery_gain <= 0 or joint_gain <= 0 or guard_delta < -guard_tolerance:
                continue
            pairs.append((recovery_gain + 0.25 * joint_gain, layer, added, removed))
            accepted += 1
        layer_evidence[key] = {
            "available_additions": len(additions),
            "protected_survivors": len(protected),
            "positive_pairs": accepted,
        }

    require(len(pairs) >= swaps, "insufficient guarded positive swaps")
    selected = sorted(pairs, key=lambda item: (-item[0], item[1], item[2], item[3]))[:swaps]
    kept_by_layer = {key: list(values) for key, values in candidate["kept_by_layer"].items()}
    by_layer = {str(layer): {"added": [], "removed": []} for layer in layers}
    for _, layer, added, removed in selected:
        key = str(layer)
        kept = set(kept_by_layer[key])
        require(added not in kept and removed in kept, "repair swap collision")
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
    output["targeted_repair"] = {
        "strategy": "guarded-targeted-repair-v1",
        "tool_sha256": sha256_file(Path(__file__)),
        "parent_plan_sha256": parent_hash,
        "candidate_plan_sha256": candidate_hash,
        "baseline_calibration_sha256": baseline_hash,
        "specialist_calibration_sha256": specialist_hash,
        "recovery_report_sha256": recovery_hashes,
        "guard_report_sha256": guard_hashes,
        "swaps": swaps,
        "layers_changed": sum(bool(row["added"]) for row in by_layer.values()),
        "guard_core_fraction": guard_core_fraction,
        "guard_tolerance": guard_tolerance,
        "by_layer": by_layer,
        "layer_evidence": layer_evidence,
        "size_policy": "same-layer swaps preserve candidate expert counts",
    }
    require(output["target_total_experts"] == candidate["target_total_experts"], "repair changed size")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--candidate-plan", required=True, type=Path)
    parser.add_argument("--baseline-calibration", required=True, type=Path)
    parser.add_argument("--specialist-calibration", required=True, type=Path)
    parser.add_argument("--recovery-report", action="append", required=True, type=Path)
    parser.add_argument("--guard-report", action="append", required=True, type=Path)
    parser.add_argument("--swaps", required=True, type=int)
    parser.add_argument("--guard-core-fraction", type=float, default=0.20)
    parser.add_argument("--guard-tolerance", type=float, default=0.10)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        recovery = [load_json(path) for path in args.recovery_report]
        guards = [load_json(path) for path in args.guard_report]
        plan = build_targeted_repair(
            load_json(args.parent_plan),
            load_json(args.candidate_plan),
            load_json(args.baseline_calibration),
            load_json(args.specialist_calibration),
            recovery,
            guards,
            [sha256_file(path) for path in args.recovery_report],
            [sha256_file(path) for path in args.guard_report],
            args.swaps,
            sha256_file(args.parent_plan),
            sha256_file(args.candidate_plan),
            sha256_file(args.baseline_calibration),
            sha256_file(args.specialist_calibration),
            args.guard_core_fraction,
            args.guard_tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, plan)
        repair = plan["targeted_repair"]
        print(
            f"targeted-repair swaps={repair['swaps']} layers={repair['layers_changed']} "
            f"output={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron targeted repair error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

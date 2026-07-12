#!/usr/bin/env python3
"""Build an add-only expert plan from source-attributed reasoning trajectories."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


ATTRIBUTION_FORMAT = "nemotron-trajectory-attribution-v1"
PLAN_FORMAT = "nemotron-nonuniform-prune-plan-v1"
EXPERT_SLOT_BYTES = 3_104_788


def aggregate_importance(reports: list[dict], label: str, template_hash: str) -> dict[int, dict[int, dict]]:
    require(reports, "no trajectory reports supplied")
    trajectories = set()
    aggregate: dict[int, dict[int, dict]] = collections.defaultdict(dict)
    expected_layers = None
    for report in reports:
        require(report.get("format") == ATTRIBUTION_FORMAT, "invalid trajectory report format")
        require(report.get("status") == "complete", "trajectory report is incomplete")
        require(report.get("plan_sha256", {}).get(label) == template_hash, "trajectory/template hash mismatch")
        trajectory = report.get("capture", {}).get("identity", {}).get("trajectory", {})
        identity = (str(trajectory.get("task_id")), int(trajectory.get("repeat", -1)))
        require(identity not in trajectories, f"duplicate trajectory report: {identity}")
        trajectories.add(identity)
        layer_rows = report.get("layers", [])
        layers = [int(row["layer"]) for row in layer_rows]
        require(layers and len(layers) == len(set(layers)), "trajectory layers are missing or duplicated")
        require(expected_layers is None or layers == expected_layers, "trajectory layer catalogs differ")
        expected_layers = layers
        for row in layer_rows:
            layer = int(row["layer"])
            result = row.get("plans", {}).get(label)
            require(isinstance(result, dict), f"trajectory has no plan result: {label}")
            importance = {
                int(expert): float(score)
                for expert, score in result.get("removed_expert_importance", {}).items()
                if float(score) > 0.0
            }
            total = sum(importance.values())
            base_error = float(result["curves"][0]["output"]["relative_l2"])
            if total <= 0.0 or base_error <= 0.0:
                continue
            for expert, value in importance.items():
                item = aggregate[layer].setdefault(expert, {"score": 0.0, "trajectories": 0})
                item["score"] += base_error * value / total
                item["trajectories"] += 1
    for experts in aggregate.values():
        for item in experts.values():
            item["score"] /= len(reports)
    return aggregate


def allocate_addbacks(
    template: dict,
    aggregate: dict[int, dict[int, dict]],
    target: int,
    minimum_per_layer: int,
    maximum_per_layer: int,
) -> dict[int, list[int]]:
    require(0 <= minimum_per_layer <= maximum_per_layer, "invalid per-layer addback bounds")
    layers = [int(layer) for layer in template["model_moe_layers"]]
    old_experts = int(template["old_num_experts"])
    ranked = {}
    additions = {}
    for layer in layers:
        retained = set(template["kept_by_layer"][str(layer)])
        candidates = [
            expert
            for expert in aggregate.get(layer, {})
            if expert not in retained
        ]
        candidates.sort(
            key=lambda expert: (
                aggregate[layer][expert]["score"],
                aggregate[layer][expert]["trajectories"],
                -expert,
            ),
            reverse=True,
        )
        limit = min(maximum_per_layer, old_experts - len(retained), len(candidates))
        require(
            limit >= min(minimum_per_layer, old_experts - len(retained)),
            f"insufficient evidence for layer {layer}",
        )
        ranked[layer] = candidates[:limit]
        additions[layer] = candidates[: min(minimum_per_layer, limit)]

    selected = sum(len(values) for values in additions.values())
    capacity = sum(len(values) for values in ranked.values())
    require(selected <= target <= capacity, f"target addback {target} is outside [{selected}, {capacity}]")
    remaining = []
    for layer in layers:
        already = set(additions[layer])
        for expert in ranked[layer]:
            if expert not in already:
                item = aggregate[layer][expert]
                remaining.append((item["score"], item["trajectories"], -layer, -expert, layer, expert))
    remaining.sort(reverse=True)
    for _, _, _, _, layer, expert in remaining[: target - selected]:
        additions[layer].append(expert)
    require(sum(len(values) for values in additions.values()) == target, "addback allocation is incomplete")
    return {layer: sorted(values) for layer, values in additions.items()}


def build_plan(
    template: dict,
    reports: list[dict],
    report_hashes: list[str],
    label: str,
    target: int,
    minimum_per_layer: int,
    maximum_per_layer: int,
    template_hash: str,
    template_payload_bytes: int | None,
) -> dict:
    require(template.get("format") == PLAN_FORMAT, "unsupported template plan format")
    aggregate = aggregate_importance(reports, label, template_hash)
    additions = allocate_addbacks(template, aggregate, target, minimum_per_layer, maximum_per_layer)
    old_experts = int(template["old_num_experts"])
    layers = [int(layer) for layer in template["model_moe_layers"]]
    kept_by_layer = {
        str(layer): sorted(set(template["kept_by_layer"][str(layer)]).union(additions[layer]))
        for layer in layers
    }
    output = dict(template)
    output["kept_by_layer"] = kept_by_layer
    output["dropped_by_layer"] = {
        layer: sorted(set(range(old_experts)) - set(kept)) for layer, kept in kept_by_layer.items()
    }
    output["old_to_new_by_layer"] = {
        layer: {str(old): new for new, old in enumerate(kept)} for layer, kept in kept_by_layer.items()
    }
    output["new_num_experts_by_layer"] = {layer: len(kept) for layer, kept in kept_by_layer.items()}
    output["target_total_experts"] = sum(len(kept) for kept in kept_by_layer.values())
    output["average_retained_experts"] = output["target_total_experts"] / len(layers)
    output["budget_label_by_layer"] = {
        str(layer): f"trajectory-add-{len(additions[layer])}" for layer in layers
    }
    output["budget_label_counts"] = dict(
        sorted(collections.Counter(output["budget_label_by_layer"].values()).items())
    )
    output["trajectory_protection"] = {
        "strategy": "source-teacher-reasoning-addback-v1",
        "tool_sha256": sha256_file(Path(__file__)),
        "template_plan_sha256": template_hash,
        "trajectory_report_sha256": report_hashes,
        "report_count": len(reports),
        "plan_label": label,
        "target_addback_experts": target,
        "minimum_addback_per_pruned_layer": minimum_per_layer,
        "maximum_addback_per_layer": maximum_per_layer,
        "expert_slot_bytes": EXPERT_SLOT_BYTES,
        "added_payload_bytes": target * EXPERT_SLOT_BYTES,
        "projected_payload_bytes": (
            template_payload_bytes + target * EXPERT_SLOT_BYTES
            if template_payload_bytes is not None
            else None
        ),
        "added_by_layer": {str(layer): additions[layer] for layer in layers},
        "evidence_by_layer": {
            str(layer): {
                str(expert): aggregate[layer][expert]
                for expert in additions[layer]
            }
            for layer in layers
        },
        "size_policy": "add-only; no retained template expert is removed",
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-plan", required=True, type=Path)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--plan-label", default="r25")
    parser.add_argument("--target-addback", required=True, type=int)
    parser.add_argument("--minimum-per-pruned-layer", type=int, default=4)
    parser.add_argument("--maximum-per-layer", type=int, default=16)
    parser.add_argument("--template-payload-bytes", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.target_addback > 0, "target addback must be positive")
        require(args.template_payload_bytes is None or args.template_payload_bytes > 0, "invalid payload size")
        template = load_json(args.template_plan)
        reports = [load_json(path) for path in args.report]
        report_hashes = [sha256_file(path) for path in args.report]
        plan = build_plan(
            template,
            reports,
            report_hashes,
            args.plan_label,
            args.target_addback,
            args.minimum_per_pruned_layer,
            args.maximum_per_layer,
            sha256_file(args.template_plan),
            args.template_payload_bytes,
        )
        atomic_json(args.output, plan)
        protection = plan["trajectory_protection"]
        projected = protection["projected_payload_bytes"]
        projected_text = "unknown" if projected is None else f"{projected / 2**30:.4f}GiB"
        print(
            f"trajectory-plan reports={len(reports)} addback={args.target_addback} "
            f"average_experts={plan['average_retained_experts']:.3f} "
            f"projected={projected_text} output={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron trajectory plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

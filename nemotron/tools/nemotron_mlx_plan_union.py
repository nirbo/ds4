#!/usr/bin/env python3
"""Build a provenance-bound union candidate pool from compatible prune plans."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


PLAN_FORMAT = "nemotron-nonuniform-prune-plan-v1"


def build_union(
    template: dict,
    additions: dict,
    template_hash: str,
    additions_hash: str,
    trajectory_additions_only: bool = False,
    max_trajectory_additions: int | None = None,
) -> dict:
    require(template.get("format") == additions.get("format") == PLAN_FORMAT, "invalid plan format")
    require(template.get("source_revision") == additions.get("source_revision"), "revision mismatch")
    require(template.get("old_num_experts") == additions.get("old_num_experts"), "expert mismatch")
    require(template.get("model_moe_layers") == additions.get("model_moe_layers"), "layer mismatch")
    layers = [int(layer) for layer in template["model_moe_layers"]]
    expert_count = int(template["old_num_experts"])
    trajectory_by_layer = additions.get("trajectory_swap", {}).get("by_layer", {})
    if trajectory_additions_only:
        require(
            set(trajectory_by_layer) == set(map(str, layers)),
            "trajectory additions do not cover the model layers",
        )
    require(
        max_trajectory_additions is None or trajectory_additions_only,
        "trajectory candidate limit requires trajectory-only mode",
    )
    selected_trajectory = None
    if max_trajectory_additions is not None:
        require(max_trajectory_additions > 0, "trajectory candidate limit must be positive")
        ranked = []
        for layer in layers:
            row = trajectory_by_layer[str(layer)]
            scores = row.get("added_joint_score", {})
            template_kept = set(template["kept_by_layer"][str(layer)])
            for expert in row.get("added", []):
                require(str(expert) in scores, f"missing trajectory score for layer {layer} expert {expert}")
                if int(expert) not in template_kept:
                    ranked.append((float(scores[str(expert)]), layer, int(expert)))
        require(max_trajectory_additions <= len(ranked), "trajectory candidate limit exceeds pool")
        selected_trajectory = {
            (layer, expert)
            for _, layer, expert in sorted(ranked, key=lambda item: (-item[0], item[1], item[2]))[
                :max_trajectory_additions
            ]
        }
    kept_by_layer = {}
    for layer in layers:
        key = str(layer)
        candidates = (
            trajectory_by_layer[key].get("added", [])
            if trajectory_additions_only
            else additions["kept_by_layer"][key]
        )
        if selected_trajectory is not None:
            candidates = [expert for expert in candidates if (layer, int(expert)) in selected_trajectory]
        kept = sorted(set(template["kept_by_layer"][key]) | set(candidates))
        require(kept and kept[0] >= 0 and kept[-1] < expert_count, f"invalid expert in layer {layer}")
        kept_by_layer[key] = kept
    output = dict(template)
    output["kept_by_layer"] = kept_by_layer
    output["dropped_by_layer"] = {
        key: sorted(set(range(expert_count)) - set(kept)) for key, kept in kept_by_layer.items()
    }
    output["old_to_new_by_layer"] = {
        key: {str(old): new for new, old in enumerate(kept)} for key, kept in kept_by_layer.items()
    }
    output["new_num_experts_by_layer"] = {key: len(kept) for key, kept in kept_by_layer.items()}
    output["target_total_experts"] = sum(len(kept) for kept in kept_by_layer.values())
    output["average_retained_experts"] = output["target_total_experts"] / len(layers)
    output["union_candidate_pool"] = {
        "strategy": "plan-union-v1",
        "tool_sha256": sha256_file(Path(__file__)),
        "template_plan_sha256": template_hash,
        "additions_plan_sha256": additions_hash,
        "added_candidates": output["target_total_experts"]
        - int(template["target_total_experts"]),
        "candidate_source": (
            "trajectory_swap.by_layer.added"
            if trajectory_additions_only
            else "kept_by_layer"
        ),
        "candidate_limit": max_trajectory_additions,
        "size_policy": "candidate pool only; not a materialization plan",
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--additions", required=True, type=Path)
    parser.add_argument("--trajectory-additions-only", action="store_true")
    parser.add_argument("--max-trajectory-additions", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_union(
            load_json(args.template),
            load_json(args.additions),
            sha256_file(args.template),
            sha256_file(args.additions),
            args.trajectory_additions_only,
            args.max_trajectory_additions,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, result)
        print(
            f"plan-union experts={result['target_total_experts']} "
            f"added={result['union_candidate_pool']['added_candidates']} "
            f"output={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron plan union error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Blend an established MTP expert plan with candidate-specific routing evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import sha256_file


PLAN_FORMAT = "nemotron-mtp-expert-plan-v1"
REPORT_FORMAT = "nemotron-mtp-acceptance-v1"


def normalized_score_mass(report: dict) -> dict[int, float]:
    require(report.get("format") == REPORT_FORMAT, "unsupported MTP acceptance report")
    raw = report.get("scored_expert_score_mass")
    require(isinstance(raw, dict) and raw, "MTP report has no scored expert mass")
    scores = {int(expert): float(score) for expert, score in raw.items()}
    require(
        len(scores) == len(raw)
        and all(0 <= expert < 512 and score >= 0.0 for expert, score in scores.items()),
        "invalid MTP scored expert mass",
    )
    total = sum(scores.values())
    require(total > 0.0, "MTP scored expert mass is empty")
    return {expert: score / total for expert, score in scores.items()}


def blend_experts(
    base: list[int],
    base_scores: dict[int, float],
    adaptation_scores: dict[int, float],
    swaps: int,
    adaptation_weight: float,
) -> tuple[list[int], list[int], list[int]]:
    require(
        base
        and len(set(base)) == len(base)
        and all(0 <= expert < 512 for expert in base),
        "invalid base MTP expert set",
    )
    require(0 <= swaps <= len(base), "invalid MTP blend swap count")
    require(0.0 <= adaptation_weight <= 1.0, "invalid MTP adaptation weight")
    universe = set(base_scores) | set(adaptation_scores)
    joint = {
        expert: (1.0 - adaptation_weight) * base_scores.get(expert, 0.0)
        + adaptation_weight * adaptation_scores.get(expert, 0.0)
        for expert in universe
    }
    base_set = set(base)
    additions = sorted(
        universe - base_set,
        key=lambda expert: (joint[expert], adaptation_scores.get(expert, 0.0), -expert),
        reverse=True,
    )[:swaps]
    removals = sorted(
        base,
        key=lambda expert: (joint.get(expert, 0.0), base_scores.get(expert, 0.0), expert),
    )[:swaps]
    require(len(additions) == swaps and len(removals) == swaps, "insufficient MTP blend candidates")
    removed = set(removals)
    result = [expert for expert in base if expert not in removed] + additions
    require(len(result) == len(base) and len(set(result)) == len(base), "invalid blended MTP plan")
    return result, removals, additions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--base-report", required=True, type=Path)
    parser.add_argument("--adaptation-report", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--swaps", required=True, type=int)
    parser.add_argument("--adaptation-weight", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_plan = load_json(args.base_plan)
        require(base_plan.get("format") == PLAN_FORMAT, "unsupported base MTP plan")
        base = base_plan.get("budgets", {}).get(str(args.budget))
        require(isinstance(base, list) and len(base) == args.budget, "base MTP plan budget mismatch")
        base_report = load_json(args.base_report)
        adaptation_report = load_json(args.adaptation_report)
        experts, removed, added = blend_experts(
            base,
            normalized_score_mass(base_report),
            normalized_score_mass(adaptation_report),
            args.swaps,
            args.adaptation_weight,
        )
        result = {
            "format": PLAN_FORMAT,
            "ranking": "fixed-swap-normalized-joint-score-mass",
            "base_plan": str(args.base_plan.resolve()),
            "base_plan_sha256": sha256_file(args.base_plan),
            "base_report": str(args.base_report.resolve()),
            "base_report_sha256": sha256_file(args.base_report),
            "adaptation_report": str(args.adaptation_report.resolve()),
            "adaptation_report_sha256": sha256_file(args.adaptation_report),
            "adaptation_weight": args.adaptation_weight,
            "swaps": args.swaps,
            "removed_experts": removed,
            "added_experts": added,
            "budgets": {str(args.budget): experts},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".part")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
        print(
            f"mtp-blend-plan-done output={args.output} budget={args.budget} "
            f"swaps={args.swaps} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron MTP blend plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

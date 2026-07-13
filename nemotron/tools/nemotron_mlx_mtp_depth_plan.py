#!/usr/bin/env python3
"""Build fixed-budget MTP expert plans from recursive-depth routing evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require, sha256_file


REPORT_FORMAT = "nemotron-mtp-recursive-acceptance-v1"
PLAN_FORMAT = "nemotron-mtp-expert-plan-v1"


def parse_weights(value: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depth weights must be comma-separated numbers") from exc
    if not weights or not all(math.isfinite(weight) and weight >= 0 for weight in weights):
        raise argparse.ArgumentTypeError("depth weights must be finite and nonnegative")
    if not any(weights):
        raise argparse.ArgumentTypeError("at least one depth weight must be positive")
    return weights


def rank_experts(
    score_mass_by_depth: dict[str, dict[str, float]],
    counts_by_depth: dict[str, dict[str, int]],
    weights: tuple[float, ...],
) -> tuple[list[int], dict[int, float]]:
    require(len(weights) <= len(score_mass_by_depth), "weights exceed captured MTP depths")
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for depth, weight in enumerate(weights, start=1):
        masses = score_mass_by_depth.get(str(depth))
        depth_counts = counts_by_depth.get(str(depth))
        require(isinstance(masses, dict) and masses, f"depth {depth} has no route mass")
        require(isinstance(depth_counts, dict) and depth_counts, f"depth {depth} has no route counts")
        total_mass = sum(float(value) for value in masses.values())
        require(total_mass > 0, f"depth {depth} has no positive route mass")
        for expert_text, mass in masses.items():
            expert = int(expert_text)
            scores[expert] = scores.get(expert, 0.0) + weight * float(mass) / total_mass
            counts[expert] = counts.get(expert, 0) + int(depth_counts[expert_text])
    ordered = sorted(
        scores,
        key=lambda expert: (scores[expert], counts[expert], -expert),
        reverse=True,
    )
    return ordered, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--depth-weights", type=parse_weights, default=parse_weights("1,1,1"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(22 <= args.budget <= 512, "MTP expert budget must be between 22 and 512")
        report = load_json(args.report)
        require(report.get("format") == REPORT_FORMAT, "unsupported recursive MTP report")
        require(isinstance(report.get("source_revision"), str), "recursive report has no revision")
        ordered, scores = rank_experts(
            report.get("expert_score_mass_by_depth", {}),
            report.get("expert_counts_by_depth", {}),
            args.depth_weights,
        )
        require(len(ordered) >= args.budget, "routing evidence does not cover requested budget")
        retained = ordered[: args.budget]
        plan = {
            "format": PLAN_FORMAT,
            "source_report": str(args.report.resolve()),
            "source_report_sha256": sha256_file(args.report),
            "source_revision": report.get("source_revision"),
            "depth_weights": list(args.depth_weights),
            "ranking": retained,
            "ranking_scores": {str(expert): scores[expert] for expert in retained},
            "budgets": {str(args.budget): retained},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".part")
        temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(
            f"mtp-depth-plan-done path={args.output} budget={args.budget} "
            f"weights={','.join(str(value) for value in args.depth_weights)} "
            f"sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError) as exc:
        print(f"nemotron MTP depth plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build nested sensitive-expert promotion plans for a fitted one-bit MTP head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-lowbit-expert-plan-v1"
FIT_FORMAT = "nemotron-mtp-binary-fit-v1"
SIDECAR_FORMAT = "nemotron-mlx-mtp-sidecar-v1"


def rank_experts(
    report: dict,
    expert_count: int,
    acceptance_pairs: list[tuple[dict, dict]] | None = None,
) -> list[dict]:
    validation = report.get("binary_fit_validation", {})
    metrics = validation.get("expert_metrics")
    require(isinstance(metrics, list), "binary-fit report has no expert metrics")
    by_expert = {}
    for row in metrics:
        expert = row.get("expert")
        require(
            isinstance(expert, int) and 0 <= expert < expert_count,
            "binary-fit report has an invalid expert ID",
        )
        require(expert not in by_expert, "binary-fit report repeats an expert")
        residual = float(row.get("validation_after_error2", 0.0))
        reference = float(row.get("validation_reference2", 0.0))
        route_mass = float(row.get("route_score_mass", 0.0))
        require(
            residual >= 0.0 and reference >= 0.0 and route_mass >= 0.0,
            "binary-fit report has a negative sensitivity metric",
        )
        by_expert[expert] = {
            "expert": expert,
            "validation_residual_error2": residual,
            "validation_reference2": reference,
            "validation_relative_error2": residual / max(reference, 1e-30),
            "calibration_route_score_mass": route_mass,
            "validation_samples": int(row.get("validation_samples", 0)),
            "task_recovery_score": 0.0,
            "teacher_recovery_route_mass": 0.0,
            "teacher_regression_route_mass": 0.0,
        }
    ranked = []
    for expert in range(expert_count):
        ranked.append(
            by_expert.get(
                expert,
                {
                    "expert": expert,
                    "validation_residual_error2": 0.0,
                    "validation_reference2": 0.0,
                    "validation_relative_error2": 0.0,
                    "calibration_route_score_mass": 0.0,
                    "validation_samples": 0,
                    "task_recovery_score": 0.0,
                    "teacher_recovery_route_mass": 0.0,
                    "teacher_regression_route_mass": 0.0,
                },
            )
        )
    if acceptance_pairs:
        rows_by_expert = {row["expert"]: row for row in ranked}
        for baseline, teacher in acceptance_pairs:
            baseline_rows = baseline.get("scored_row_results")
            teacher_rows = teacher.get("scored_row_results")
            require(
                isinstance(baseline_rows, list)
                and isinstance(teacher_rows, list)
                and len(baseline_rows) == len(teacher_rows),
                "acceptance reports have incompatible row evidence",
            )
            for baseline_row, teacher_row in zip(baseline_rows, teacher_rows, strict=True):
                require(
                    baseline_row["row"] == teacher_row["row"]
                    and baseline_row["expected_token_id"] == teacher_row["expected_token_id"],
                    "acceptance report rows are not aligned",
                )
                expected = baseline_row["expected_token_id"]
                baseline_top1 = baseline_row["predicted_token_id"] == expected
                teacher_top1 = teacher_row["predicted_token_id"] == expected
                baseline_top5 = expected in baseline_row["top5_token_ids"]
                teacher_top5 = expected in teacher_row["top5_token_ids"]
                top1_delta = int(teacher_top1) - int(baseline_top1)
                top5_delta = int(teacher_top5) - int(baseline_top5)
                task_delta = top1_delta + 0.25 * top5_delta
                expert_ids = baseline_row["routed_expert_ids"]
                route_scores = baseline_row["route_scores"]
                require(
                    len(expert_ids) == len(route_scores)
                    and expert_ids == teacher_row["routed_expert_ids"],
                    "acceptance reports have different routed experts",
                )
                for expert, route_score in zip(expert_ids, route_scores, strict=True):
                    require(expert in rows_by_expert, "acceptance report routes an unknown expert")
                    row = rows_by_expert[expert]
                    weighted = float(route_score)
                    row["task_recovery_score"] += weighted * task_delta
                    if top1_delta > 0:
                        row["teacher_recovery_route_mass"] += weighted
                    elif top1_delta < 0:
                        row["teacher_regression_route_mass"] += weighted
    return sorted(
        ranked,
        key=lambda row: (
            -row["task_recovery_score"],
            -row["teacher_recovery_route_mass"],
            -row["validation_residual_error2"],
            -row["calibration_route_score_mass"],
            row["expert"],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", required=True, action="append", type=int)
    parser.add_argument("--baseline-acceptance-report", action="append", type=Path, default=[])
    parser.add_argument("--teacher-acceptance-report", action="append", type=Path, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = load_json(args.fit_report)
        require(
            report.get("format") == SIDECAR_FORMAT and report.get("status") == "complete",
            "binary-fit sidecar report is incomplete",
        )
        fit = report.get("binary_fit", {})
        require(fit.get("format") == FIT_FORMAT, "report is not activation-fitted binary MTP")
        expert_count = report.get("budget")
        require(isinstance(expert_count, int) and expert_count > 0, "invalid MTP expert count")
        budgets = sorted(set(args.budget))
        require(
            len(budgets) == len(args.budget)
            and all(0 < budget <= expert_count for budget in budgets),
            "promotion budgets must be unique and within the expert count",
        )
        require(
            len(args.baseline_acceptance_report) == len(args.teacher_acceptance_report),
            "baseline and teacher acceptance reports must be paired",
        )
        acceptance_pairs = []
        acceptance_provenance = []
        for baseline_path, teacher_path in zip(
            args.baseline_acceptance_report,
            args.teacher_acceptance_report,
            strict=True,
        ):
            baseline = load_json(baseline_path)
            teacher = load_json(teacher_path)
            require(
                baseline.get("format") == teacher.get("format") == "nemotron-mtp-acceptance-v1",
                "unsupported MTP acceptance report",
            )
            require(
                baseline.get("trace_sha256") == teacher.get("trace_sha256"),
                "paired acceptance reports use different traces",
            )
            require(
                baseline.get("sidecar_sha256") == report["artifact_sha256"],
                "baseline acceptance report does not describe the fitted binary artifact",
            )
            acceptance_pairs.append((baseline, teacher))
            acceptance_provenance.append(
                {
                    "baseline": str(baseline_path.resolve()),
                    "baseline_sha256": sha256_file(baseline_path),
                    "teacher": str(teacher_path.resolve()),
                    "teacher_sha256": sha256_file(teacher_path),
                    "trace_sha256": baseline["trace_sha256"],
                }
            )
        ranking = rank_experts(report, expert_count, acceptance_pairs)
        plan = {
            "format": FORMAT,
            "source_revision": report["source_revision"],
            "fit_report": str(args.fit_report.resolve()),
            "fit_report_sha256": sha256_file(args.fit_report),
            "fit_artifact_sha256": report["artifact_sha256"],
            "expert_count": expert_count,
            "score": (
                "teacher-recovery-route-mass-then-heldout-output-residual"
                if acceptance_pairs
                else "heldout-route-weighted-expert-output-residual"
            ),
            "acceptance_reports": acceptance_provenance,
            "ranked_experts": ranking,
            "budgets": {
                str(budget): sorted(row["expert"] for row in ranking[:budget])
                for budget in budgets
            },
        }
        atomic_json(args.output, plan)
        print(
            f"mtp-lowbit-plan output={args.output} experts={expert_count} "
            f"budgets={','.join(str(value) for value in budgets)}",
            flush=True,
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError) as exc:
        print(f"nemotron MTP low-bit plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

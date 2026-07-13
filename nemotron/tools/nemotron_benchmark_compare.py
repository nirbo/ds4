#!/usr/bin/env python3
"""Compare paired Nemotron MBPP or HumanEval reports with strict identity."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-benchmark-paired-comparison-v1"
SUPPORTED_FORMATS = {
    "nemotron-mbpp-eval-v1": (),
    "nemotron-humaneval-v1": ("code_eval_helper_sha256",),
}
COMMON_IDENTITY_KEYS = (
    "evaluator_sha256",
    "resident_runtime_sha256",
    "dataset_sha256",
    "sandbox_python",
    "sandbox_python_version",
    "sample_size",
    "sample_offset",
    "task_ids",
    "max_new_tokens",
)


def matrix(rows: list[tuple[bool, bool]]) -> dict[str, int]:
    counts = collections.Counter(rows)
    return {
        "both_pass": counts[(True, True)],
        "baseline_only": counts[(True, False)],
        "candidate_only": counts[(False, True)],
        "both_fail": counts[(False, False)],
    }


def result_catalog(report: dict, label: str) -> dict[str, dict]:
    rows = {str(row["task_id"]): row for row in report["results"]}
    require(len(rows) == len(report["results"]), f"{label} has duplicate tasks")
    require(list(rows) == [str(task) for task in report["task_ids"]], f"{label} task order mismatch")
    require(len(rows) == int(report["sample_size"]), f"{label} task count mismatch")
    require(
        sum(bool(row["passed"]) for row in rows.values())
        == int(report["summary"]["passed"]),
        f"{label} summary pass count mismatch",
    )
    return rows


def compare_reports(baseline: dict, candidate: dict) -> dict:
    require(
        baseline.get("status") == candidate.get("status") == "complete",
        "reports must be complete",
    )
    report_format = baseline.get("format")
    require(report_format == candidate.get("format"), "report protocol mismatch: format")
    require(report_format in SUPPORTED_FORMATS, "unsupported report format")
    for key in (*COMMON_IDENTITY_KEYS, *SUPPORTED_FORMATS[report_format]):
        require(baseline.get(key) == candidate.get(key), f"report protocol mismatch: {key}")

    baseline_rows = result_catalog(baseline, "baseline")
    candidate_rows = result_catalog(candidate, "candidate")
    require(set(baseline_rows) == set(candidate_rows), "paired task catalogs differ")
    task_ids = [str(task) for task in baseline["task_ids"]]
    outcomes = [
        (bool(baseline_rows[task]["passed"]), bool(candidate_rows[task]["passed"]))
        for task in task_ids
    ]
    flips = [
        {
            "task_id": task,
            "baseline_passed": bool(baseline_rows[task]["passed"]),
            "candidate_passed": bool(candidate_rows[task]["passed"]),
            "baseline_generated_tokens": int(baseline_rows[task]["generated_tokens"]),
            "candidate_generated_tokens": int(candidate_rows[task]["generated_tokens"]),
            "candidate_error": str(candidate_rows[task].get("error", ""))[:500],
        }
        for task in task_ids
        if bool(baseline_rows[task]["passed"]) != bool(candidate_rows[task]["passed"])
    ]
    baseline_passed = sum(before for before, _ in outcomes)
    candidate_passed = sum(after for _, after in outcomes)
    return {
        "format": FORMAT,
        "benchmark_format": report_format,
        "tasks": len(task_ids),
        "matrix": matrix(outcomes),
        "pass_delta": candidate_passed - baseline_passed,
        "baseline_passed": baseline_passed,
        "candidate_passed": candidate_passed,
        "baseline_only": [row["task_id"] for row in flips if row["baseline_passed"]],
        "candidate_only": [row["task_id"] for row in flips if row["candidate_passed"]],
        "flips": flips,
        "exact_response_matches": sum(
            baseline_rows[task].get("response") == candidate_rows[task].get("response")
            for task in task_ids
        ),
        "exact_code_matches": sum(
            baseline_rows[task].get("code") == candidate_rows[task].get("code")
            for task in task_ids
        ),
        "baseline_summary": baseline["summary"],
        "candidate_summary": candidate["summary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = compare_reports(load_json(args.baseline), load_json(args.candidate))
        report.update(
            {
                "baseline_report": str(args.baseline.resolve()),
                "baseline_sha256": sha256_file(args.baseline),
                "candidate_report": str(args.candidate.resolve()),
                "candidate_sha256": sha256_file(args.candidate),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        print(
            f"benchmark-paired format={report['benchmark_format']} tasks={report['tasks']} "
            f"delta={report['pass_delta']:+d} output={args.output} "
            f"sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron benchmark compare error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

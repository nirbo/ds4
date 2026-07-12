#!/usr/bin/env python3
"""Compare paired Nemotron LiveCodeBench reports with strict protocol identity."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-livecodebench-paired-comparison-v1"
RAW_FORMAT = "nemotron-livecodebench-v1"
RESCORE_FORMAT = "nemotron-livecodebench-rescore-v1"
COMMON_IDENTITY_KEYS = (
    "evaluator_sha256",
    "dataset_sha256",
    "dataset_state_sha256",
    "private_state_sha256",
    "private_index_sha256",
    "sandbox_python",
    "sandbox_python_version",
    "task_ids",
    "execution_timeout",
)
RAW_IDENTITY_KEYS = COMMON_IDENTITY_KEYS + (
    "generation_helper_sha256",
    "sampling",
    "max_new_tokens",
    "max_public_cases",
    "generation",
    "nvidia_reference_protocol",
)
RESCORE_IDENTITY_KEYS = COMMON_IDENTITY_KEYS + ("rescorer_sha256",)


def outcome_class(row: dict) -> str:
    if row["passed"]:
        return "pass"
    if row["truncated"]:
        return "truncated"
    error = str(row.get("error", "")).lower()
    if "timed out" in error:
        return "timeout"
    if "traceback" in error or 'file "' in error:
        return "runtime_or_syntax"
    return "wrong_answer"


def matrix(rows: list[tuple[bool, bool]]) -> dict[str, int]:
    counts = collections.Counter(rows)
    return {
        "both_pass": counts[(True, True)],
        "baseline_only": counts[(True, False)],
        "candidate_only": counts[(False, True)],
        "both_fail": counts[(False, False)],
    }


def compare_reports(baseline: dict, candidate: dict) -> dict:
    require(baseline.get("status") == candidate.get("status") == "complete", "reports must be complete")
    report_format = baseline.get("format")
    require(report_format == candidate.get("format"), "report protocol mismatch: format")
    require(report_format in (RAW_FORMAT, RESCORE_FORMAT), "unsupported report format")
    identity_keys = RAW_IDENTITY_KEYS if report_format == RAW_FORMAT else RESCORE_IDENTITY_KEYS
    for key in identity_keys:
        require(baseline.get(key) == candidate.get(key), f"report protocol mismatch: {key}")
    baseline_rows = {(str(row["task_id"]), int(row.get("repeat", 0))): row for row in baseline["results"]}
    candidate_rows = {(str(row["task_id"]), int(row.get("repeat", 0))): row for row in candidate["results"]}
    require(len(baseline_rows) == len(baseline["results"]), "baseline has duplicate samples")
    require(len(candidate_rows) == len(candidate["results"]), "candidate has duplicate samples")
    require(set(baseline_rows) == set(candidate_rows), "paired sample catalogs differ")
    keys = sorted(baseline_rows)
    for key in keys:
        require(
            baseline_rows[key]["seed"] == candidate_rows[key]["seed"]
            and baseline_rows[key]["difficulty"] == candidate_rows[key]["difficulty"],
            f"paired sample identity mismatch: {key}",
        )

    difficulties = sorted({str(baseline_rows[key]["difficulty"]) for key in keys})
    flips = []
    for key in keys:
        before = baseline_rows[key]
        after = candidate_rows[key]
        if bool(before["passed"]) != bool(after["passed"]):
            flips.append(
                {
                    "task_id": key[0],
                    "repeat": key[1],
                    "seed": before["seed"],
                    "difficulty": before["difficulty"],
                    "baseline_passed": before["passed"],
                    "candidate_passed": after["passed"],
                    "baseline_class": outcome_class(before),
                    "candidate_class": outcome_class(after),
                    "candidate_error": str(after.get("error", ""))[:500],
                }
            )
    by_difficulty = {}
    for difficulty in difficulties:
        selected = [key for key in keys if str(baseline_rows[key]["difficulty"]) == difficulty]
        by_difficulty[difficulty] = {
            "baseline_passed": sum(bool(baseline_rows[key]["passed"]) for key in selected),
            "candidate_passed": sum(bool(candidate_rows[key]["passed"]) for key in selected),
            "matrix": matrix(
                [(bool(baseline_rows[key]["passed"]), bool(candidate_rows[key]["passed"])) for key in selected]
            ),
        }

    tasks = list(baseline["task_ids"])
    baseline_task = {
        task: any(row["passed"] for key, row in baseline_rows.items() if key[0] == task) for task in tasks
    }
    candidate_task = {
        task: any(row["passed"] for key, row in candidate_rows.items() if key[0] == task) for task in tasks
    }
    task_flips = [
        {
            "task_id": task,
            "baseline_passed": baseline_task[task],
            "candidate_passed": candidate_task[task],
        }
        for task in tasks
        if baseline_task[task] != candidate_task[task]
    ]
    return {
        "format": FORMAT,
        "samples": len(keys),
        "tasks": len(tasks),
        "sample_matrix": matrix(
            [(bool(baseline_rows[key]["passed"]), bool(candidate_rows[key]["passed"])) for key in keys]
        ),
        "by_difficulty": by_difficulty,
        "failure_classes": {
            "baseline": dict(
                sorted(collections.Counter(outcome_class(row) for row in baseline_rows.values()).items())
            ),
            "candidate": dict(
                sorted(collections.Counter(outcome_class(row) for row in candidate_rows.values()).items())
            ),
        },
        "truncation_matrix": matrix(
            [(bool(baseline_rows[key]["truncated"]), bool(candidate_rows[key]["truncated"])) for key in keys]
        ),
        "sample_flips": flips,
        "task_matrix": matrix([(baseline_task[task], candidate_task[task]) for task in tasks]),
        "task_flips": task_flips,
        "baseline_summary": baseline["summary"],
        "candidate_summary": candidate["summary"],
        "sample_pass_delta": candidate["summary"]["passed_samples"] - baseline["summary"]["passed_samples"],
        "task_pass_delta": candidate["summary"]["tasks_with_pass"] - baseline["summary"]["tasks_with_pass"],
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
            f"livecodebench-paired samples={report['samples']} sample_delta={report['sample_pass_delta']} "
            f"task_delta={report['task_pass_delta']} output={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron LiveCodeBench compare error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

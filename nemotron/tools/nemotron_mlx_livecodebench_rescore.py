#!/usr/bin/env python3
"""Rescore stored LiveCodeBench code with the current dated public harness."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_livecodebench import (
    DATED_FORMAT,
    attach_private_tests,
    check_cases,
    deterministic_items,
    stratified_items,
    summarize_results,
)
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-livecodebench-rescore-v1"


def selected_items(dataset: Path, sampling: dict) -> list[dict]:
    if sampling.get("mode") == "stratified":
        return stratified_items(
            dataset,
            int(sampling["samples_per_difficulty"]),
            int(sampling["offset_per_difficulty"]),
        )
    require(sampling.get("mode") == "global", "unknown source sampling mode")
    return deterministic_items(
        dataset, int(sampling["sample_size"]), int(sampling["sample_offset"])
    )


def rescore_rows(
    source_rows: list[dict],
    items: list[dict],
    sandbox_root: Path,
    python: Path,
    timeout: int,
) -> list[dict]:
    items_by_id = {str(item["question_id"]): item for item in items}
    rows = []
    for source in source_rows:
        task_id = str(source["task_id"])
        require(task_id in items_by_id, f"source task is absent from dataset: {task_id}")
        item = items_by_id[task_id]
        passed, error, cases_run = check_cases(
            source["code"], item, sandbox_root, python, 0, timeout
        )
        rows.append(
            {
                **source,
                "passed": passed,
                "source_passed": bool(source["passed"]),
                "cases_run": cases_run,
                "error": error,
                "test_type": item["public_test_cases"][0]["testtype"],
                "difficulty": item.get("difficulty", ""),
                "contest_date": item.get("contest_date"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-state", required=True, type=Path)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--private-state", type=Path)
    parser.add_argument("--private-index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execution-timeout", type=int, default=6)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.execution_timeout > 0, "invalid execution timeout")
        require(shutil.which("sandbox-exec") is not None, "sandbox-exec is required")
        require(args.python.is_file(), "sandbox Python interpreter is unavailable")
        source = load_json(args.source_report)
        require(source.get("status") == "complete", "source report is incomplete")
        require(
            source.get("dataset_sha256") == sha256_file(args.dataset),
            "source report dataset hash mismatch",
        )
        state = load_json(args.dataset_state)
        require(state.get("format") == DATED_FORMAT, "unsupported dated dataset state")
        require(state.get("status") == "complete", "dated dataset is incomplete")
        require(
            state.get("output_sha256") == sha256_file(args.dataset),
            "dated dataset hash mismatch",
        )
        items = selected_items(args.dataset, source["sampling"])
        private_paths = (args.private_dir, args.private_state, args.private_index)
        require(
            all(path is None for path in private_paths)
            or all(path is not None for path in private_paths),
            "private-dir, private-state, and private-index must be supplied together",
        )
        if args.private_dir is not None:
            attach_private_tests(
                items,
                args.private_dir,
                args.private_state,
                args.private_index,
                args.dataset_state,
            )
        task_ids = [str(item["question_id"]) for item in items]
        require(source.get("task_ids") == task_ids, "source task order mismatch")
        evaluator_path = Path(__file__).with_name("nemotron_mlx_livecodebench.py")
        identity = {
            "format": FORMAT,
            "rescorer_sha256": sha256_file(Path(__file__)),
            "evaluator_sha256": sha256_file(evaluator_path),
            "source_report": str(args.source_report.resolve()),
            "source_report_sha256": sha256_file(args.source_report),
            "dataset_sha256": sha256_file(args.dataset),
            "dataset_state_sha256": sha256_file(args.dataset_state),
            "private_state_sha256": (
                sha256_file(args.private_state) if args.private_state is not None else None
            ),
            "private_index_sha256": (
                sha256_file(args.private_index) if args.private_index is not None else None
            ),
            "sandbox_python": str(args.python.resolve()),
            "sandbox_python_version": subprocess.check_output(
                [str(args.python.resolve()), "--version"], text=True
            ).strip(),
            "execution_timeout": args.execution_timeout,
            "task_ids": task_ids,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {**identity, "status": "running", "results": []}
        atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        sandbox_root = args.output.parent / ".livecodebench-rescore-sandbox"
        sandbox_root.mkdir(exist_ok=True)
        operation_log.write(f"rescore-start samples={len(source['results'])}")
        report["results"] = rescore_rows(
            source["results"],
            items,
            sandbox_root,
            args.python.resolve(),
            args.execution_timeout,
        )
        report["summary"] = summarize_results(report["results"], len(items))
        report["summary"]["changed_samples"] = sum(
            row["passed"] != row["source_passed"] for row in report["results"]
        )
        report["status"] = "complete"
        atomic_json(args.output, report)
        operation_log.write(
            f"rescore-complete passed={report['summary']['passed_samples']}/"
            f"{report['summary']['completed_samples']}"
        )
        print(json.dumps(report["summary"], sort_keys=True))
        print(f"livecodebench-rescore path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if operation_log is not None:
            operation_log.write(f"rescore-failed error={exc}")
        print(f"nemotron LiveCodeBench rescore error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

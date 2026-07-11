#!/usr/bin/env python3
"""Rescore stored HumanEval responses with the current sandbox harness."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_humaneval import deterministic_items, extract_completion, human_eval_tests
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-humaneval-rescore-v1"


def rescore_items(
    source_rows: dict[str, dict],
    items: list[dict],
    sandbox_root: Path,
    python: Path,
) -> list[dict]:
    rows = []
    for item in items:
        task_id = str(item["task_id"])
        source_row = source_rows[task_id]
        code = extract_completion(
            source_row["response"], item["prompt"], item["entry_point"]
        )
        passed, error = human_eval_tests(code, item, sandbox_root, python)
        rows.append(
            {
                "task_id": task_id,
                "passed": passed,
                "source_passed": bool(source_row["passed"]),
                "code": code,
                "error": error,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(shutil.which("sandbox-exec") is not None, "sandbox-exec is required")
        require(args.python.is_file(), "sandbox Python interpreter is unavailable")
        source = load_json(args.source_report)
        require(source.get("status") == "complete", "source HumanEval report is incomplete")
        items = deterministic_items(
            args.dataset, int(source["sample_size"]), int(source["sample_offset"])
        )
        source_rows = {str(row["task_id"]): row for row in source["results"]}
        task_ids = [str(item["task_id"]) for item in items]
        require(list(source_rows) == task_ids, "source HumanEval task order mismatch")
        require(source.get("dataset_sha256") == sha256_file(args.dataset), "dataset hash mismatch")
        helper_path = Path(__file__).with_name("nemotron_mlx_mbpp.py")
        evaluator_path = Path(__file__).with_name("nemotron_mlx_humaneval.py")
        identity = {
            "format": FORMAT,
            "rescorer_sha256": sha256_file(Path(__file__)),
            "evaluator_sha256": sha256_file(evaluator_path),
            "code_eval_helper_sha256": sha256_file(helper_path),
            "source_report": str(args.source_report.resolve()),
            "source_report_sha256": sha256_file(args.source_report),
            "dataset_sha256": sha256_file(args.dataset),
            "sandbox_python": str(args.python.resolve()),
            "sandbox_python_version": subprocess.check_output(
                [str(args.python.resolve()), "--version"], text=True
            ).strip(),
            "task_ids": task_ids,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {**identity, "status": "running", "results": []}
        atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        sandbox_root = args.output.parent / ".humaneval-rescore-sandbox"
        sandbox_root.mkdir(exist_ok=True)
        operation_log.write(f"rescore-start total={len(items)}")
        report["results"] = rescore_items(
            source_rows, items, sandbox_root, args.python.resolve()
        )
        passed_count = sum(row["passed"] for row in report["results"])
        changed = [
            row["task_id"]
            for row in report["results"]
            if row["passed"] != row["source_passed"]
        ]
        report["summary"] = {
            "completed": len(report["results"]),
            "passed": passed_count,
            "pass_at_1": passed_count / len(report["results"]),
            "changed_task_ids": changed,
        }
        report["status"] = "complete"
        atomic_json(args.output, report)
        operation_log.write(
            f"rescore-complete passed={passed_count}/{len(items)} changed={changed}"
        )
        print(json.dumps(report["summary"], sort_keys=True))
        print(f"humaneval-rescore path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if operation_log is not None:
            operation_log.write(f"rescore-failed error={exc}")
        print(f"nemotron HumanEval rescore error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a provenance-bound LiveCodeBench calibration corpus outside an eval window."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import sha256_file
from nemotron_mlx_livecodebench import load_items, prompt_for
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-livecodebench-calibration-v1"


def parse_date(value: str) -> dt.date:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise MetadataError(f"invalid contest date: {value}") from exc


def report_task_ids(paths: list[Path], dataset_sha256: str | None = None) -> tuple[set[str], list[dict]]:
    excluded: set[str] = set()
    reports = []
    for path in paths:
        report = load_json(path)
        task_ids = report.get("task_ids")
        require(isinstance(task_ids, list), f"evaluation report has no task_ids: {path}")
        if dataset_sha256 is not None and report.get("dataset_sha256") is not None:
            require(report["dataset_sha256"] == dataset_sha256, f"evaluation report dataset mismatch: {path}")
        excluded.update(str(task_id) for task_id in task_ids)
        reports.append({"path": str(path.resolve()), "sha256": sha256_file(path), "tasks": len(task_ids)})
    return excluded, reports


def build_corpus(
    dataset: Path,
    start_date: dt.date,
    end_date: dt.date,
    excluded_ids: set[str],
) -> tuple[dict[str, list[str]], list[dict]]:
    require(start_date <= end_date, "start date must not follow end date")
    selected = []
    for item in load_items(dataset):
        task_id = str(item["question_id"])
        contest_date = parse_date(str(item["contest_date"]))
        if start_date <= contest_date <= end_date and task_id not in excluded_ids:
            selected.append(item)
    selected.sort(key=lambda item: (str(item["difficulty"]), str(item["contest_date"]), str(item["question_id"])))
    require(selected, "date window and exclusions selected no calibration tasks")

    corpus: dict[str, list[str]] = {}
    rows = []
    for item in selected:
        difficulty = str(item.get("difficulty", "unknown")).lower()
        category = f"livecodebench_{difficulty}"
        prompt = prompt_for(item)
        corpus.setdefault(category, []).append(prompt)
        rows.append(
            {
                "task_id": str(item["question_id"]),
                "difficulty": difficulty,
                "contest_date": str(item["contest_date"]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return corpus, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--start-date", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--exclude-report", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset_sha256 = sha256_file(args.dataset)
        excluded_ids, reports = report_task_ids(args.exclude_report, dataset_sha256)
        corpus, rows = build_corpus(args.dataset, args.start_date, args.end_date, excluded_ids)
        selected_ids = {row["task_id"] for row in rows}
        require(not selected_ids & excluded_ids, "calibration/evaluation task overlap")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, corpus)
        atomic_json(
            args.state,
            {
                "format": FORMAT,
                "dataset": str(args.dataset.resolve()),
                "dataset_sha256": dataset_sha256,
                "date_window": {"start": args.start_date.isoformat(), "end": args.end_date.isoformat()},
                "excluded_reports": reports,
                "excluded_task_count": len(excluded_ids),
                "corpus": str(args.output.resolve()),
                "corpus_sha256": sha256_file(args.output),
                "task_count": len(rows),
                "category_counts": {
                    category: len(samples) for category, samples in sorted(corpus.items())
                },
                "tasks": rows,
            },
        )
        print(
            f"nemotron LiveCodeBench calibration: tasks={len(rows)} categories="
            f"{ {key: len(value) for key, value in sorted(corpus.items())} } output={args.output}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"nemotron LiveCodeBench calibration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

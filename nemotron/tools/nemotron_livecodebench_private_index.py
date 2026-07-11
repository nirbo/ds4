#!/usr/bin/env python3
"""Build a provenance-bound random-access index for private LiveCodeBench rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from nemotron_metadata import MetadataError, load_json, require, sha256_file
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-livecodebench-private-index-v1"
PRIVATE_FORMAT = "nemotron-livecodebench-private-v1"


def index_file(path: Path) -> dict[str, dict[str, Any]]:
    entries = {}
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MetadataError(f"invalid private JSONL row at {path}:{offset}: {exc}") from exc
            question_id = str(row.get("question_id", ""))
            require(question_id, f"private row has no question ID at {path}:{offset}")
            require(question_id not in entries, f"duplicate private question ID: {question_id}")
            cases = row.get("private_test_cases")
            require(isinstance(cases, list) and cases, f"private row is empty: {question_id}")
            entries[question_id] = {
                "offset": offset,
                "length": len(line),
                "tests": len(cases),
            }
    return entries


def read_indexed_row(
    private_dir: Path, index: dict[str, Any], question_id: str
) -> dict[str, Any]:
    entry = index["entries"].get(question_id)
    require(isinstance(entry, dict), f"private index has no task: {question_id}")
    path = private_dir / entry["file"]
    with path.open("rb") as handle:
        handle.seek(entry["offset"])
        line = handle.read(entry["length"])
    require(len(line) == entry["length"], f"short private row read: {question_id}")
    try:
        row = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataError(f"invalid indexed private row {question_id}: {exc}") from exc
    require(str(row.get("question_id")) == question_id, f"private index identity mismatch: {question_id}")
    cases = row.get("private_test_cases")
    require(
        isinstance(cases, list) and len(cases) == entry["tests"],
        f"private index test-count mismatch: {question_id}",
    )
    return row


def validate_index_files(private_dir: Path, index: dict[str, Any]) -> None:
    for record in index["files"]:
        path = private_dir / record["file"]
        require(path.is_file(), f"private indexed file is missing: {path}")
        require(path.stat().st_size == record["bytes"], f"private indexed file size mismatch: {path}")
        require(sha256_file(path) == record["sha256"], f"private indexed file hash mismatch: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--private-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        state = load_json(args.private_state)
        require(state.get("format") == PRIVATE_FORMAT, "unsupported private state")
        require(state.get("status") == "complete", "private materialization is incomplete")
        entries = {}
        files = []
        for record in state["completed"]:
            path = args.private_dir / record["file"]
            require(path.is_file(), f"private output is missing: {path}")
            require(path.stat().st_size == record["output_bytes"], f"private output size mismatch: {path}")
            require(sha256_file(path) == record["sha256"], f"private output hash mismatch: {path}")
            file_entries = index_file(path)
            for question_id, entry in file_entries.items():
                require(question_id not in entries, f"duplicate private task: {question_id}")
                entries[question_id] = {"file": record["file"], **entry}
            files.append(
                {
                    "file": record["file"],
                    "bytes": record["output_bytes"],
                    "sha256": record["sha256"],
                    "tasks": len(file_entries),
                }
            )
            print(f"private-index-file path={path} tasks={len(file_entries)}")
        require(len(entries) == state["summary"]["tasks"], "private index task count mismatch")
        output = {
            "format": FORMAT,
            "status": "complete",
            "indexer_sha256": sha256_file(Path(__file__)),
            "private_state_sha256": sha256_file(args.private_state),
            "private_dir": str(args.private_dir.resolve()),
            "tasks": len(entries),
            "tests": sum(entry["tests"] for entry in entries.values()),
            "files": files,
            "entries": entries,
        }
        require(output["tests"] == state["summary"]["tests"], "private index test count mismatch")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, output)
        print(
            f"private-index-complete tasks={output['tasks']} tests={output['tests']} "
            f"output={args.output}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"nemotron LiveCodeBench private index error: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

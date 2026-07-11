#!/usr/bin/env python3
"""Materialize official LiveCodeBench private tests from bounded HTTP ranges."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pickle
import time
import zlib
from pathlib import Path
from typing import Any

from nemotron_livecodebench_dataset import (
    HTTPRangeReader,
    MANIFEST_FORMAT,
    load_jsonl,
    source_url,
    validate_manifest,
)
from nemotron_metadata import MetadataError, load_json, require, sha256_file
from nemotron_prune_materialize import OperationLog, atomic_json


FORMAT = "nemotron-livecodebench-private-v1"
MAX_PRIVATE_ROW_BYTES = 512 * 2**20


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"global object is forbidden: {module}.{name}")


def decode_private_cases(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        cases = value
    else:
        require(isinstance(value, str), "private tests are not a string or list")
        try:
            cases = json.loads(value)
        except json.JSONDecodeError:
            try:
                compressed = base64.b64decode(value, validate=True)
                decompressor = zlib.decompressobj()
                payload = decompressor.decompress(compressed, MAX_PRIVATE_ROW_BYTES + 1)
                require(
                    len(payload) <= MAX_PRIVATE_ROW_BYTES and decompressor.eof,
                    "compressed private-test row exceeds safety limit",
                )
                decoded = RestrictedUnpickler(io.BytesIO(payload)).load()
                if isinstance(decoded, bytes):
                    decoded = decoded.decode("utf-8")
                cases = json.loads(decoded) if isinstance(decoded, str) else decoded
            except (ValueError, zlib.error, pickle.UnpicklingError, json.JSONDecodeError) as exc:
                raise MetadataError(f"cannot decode private tests: {exc}") from exc
    require(isinstance(cases, list) and cases, "private-test list is empty")
    for case in cases:
        require(isinstance(case, dict), "private test is not an object")
        require(case.get("testtype") in ("stdin", "functional"), "unsupported private test type")
        require("input" in case and "output" in case, "private test is incomplete")
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_private_shard(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    selected: list[dict[str, Any]],
    token: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise MetadataError("pyarrow is required for private-test materialization") from exc

    reader = HTTPRangeReader(
        source_url(manifest, entry), entry["size"], entry["xet_hash"], token
    )
    buffered = io.BufferedReader(reader, buffer_size=64 * 1024)
    parquet_file = parquet.ParquetFile(buffered)
    table = parquet_file.read(columns=["question_id", "private_test_cases"])
    source_rows = table.to_pylist()
    selected_by_row = {int(row["official_row"]): row for row in selected}
    require(len(selected_by_row) == len(selected), "duplicate selected source row")
    output = []
    for row_index, public in sorted(selected_by_row.items()):
        require(0 <= row_index < len(source_rows), "selected source row is out of range")
        source = source_rows[row_index]
        question_id = str(source["question_id"])
        require(
            question_id == str(public["question_id"]),
            f"private row identity mismatch at {row_index}",
        )
        cases = decode_private_cases(source["private_test_cases"])
        output.append({"question_id": question_id, "private_test_cases": cases})
    return output, {"requests": reader.requests, "bytes": reader.bytes_read}


def identity_for(
    manifest_path: Path,
    manifest: dict[str, Any],
    public_catalog: Path,
    public_state_path: Path,
    public_state: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "manifest_sha256": sha256_file(manifest_path),
        "public_catalog_sha256": sha256_file(public_catalog),
        "public_state_sha256": sha256_file(public_state_path),
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "config": manifest.get("config"),
        "start_date": public_state["start_date"],
        "end_date": public_state["end_date"],
        "task_ids": [str(row["question_id"]) for row in selected],
    }


def validate_completed(output_dir: Path, completed: list[dict[str, Any]]) -> None:
    for record in completed:
        path = output_dir / record["file"]
        require(path.is_file(), f"completed private output is missing: {path}")
        require(path.stat().st_size == record["output_bytes"], f"private output size mismatch: {path}")
        require(sha256_file(path) == record["sha256"], f"private output hash mismatch: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--public-catalog", required=True, type=Path)
    parser.add_argument("--public-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    operation_log = None
    try:
        args = parse_args()
        require(args.max_shards is None or args.max_shards > 0, "invalid max-shards")
        manifest = load_json(args.manifest)
        require(manifest.get("format") == MANIFEST_FORMAT, "unsupported dataset manifest")
        files = validate_manifest(manifest)
        public_state = load_json(args.public_state)
        require(public_state.get("status") == "complete", "public catalog is incomplete")
        require(
            public_state.get("manifest_sha256") == sha256_file(args.manifest),
            "public catalog manifest mismatch",
        )
        require(
            public_state.get("output_sha256") == sha256_file(args.public_catalog),
            "public catalog hash mismatch",
        )
        selected = load_jsonl(args.public_catalog)
        identity = identity_for(
            args.manifest,
            manifest,
            args.public_catalog,
            args.public_state,
            public_state,
            selected,
        )
        selected_by_shard: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            selected_by_shard.setdefault(row["official_shard"], []).append(row)
        required_entries = [entry for entry in files if entry["path"] in selected_by_shard]
        expected_transfer = int(public_state["hidden_test_transfer_bytes"])
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "tasks": len(selected),
                        "shards": len(required_entries),
                        "expected_transfer_bytes": expected_transfer,
                        "expected_transfer_gib": expected_transfer / 2**30,
                    },
                    sort_keys=True,
                )
            )
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.state.with_suffix(".log"))
        if args.state.exists():
            state = load_json(args.state)
            for key, value in identity.items():
                require(state.get(key) == value, f"private state identity mismatch: {key}")
        else:
            state = {**identity, "status": "running", "completed": []}
            atomic_json(args.state, state)
        validate_completed(args.output_dir, state["completed"])
        completed_paths = {record["source"] for record in state["completed"]}
        operation_log.write(
            f"private-run-start completed={len(completed_paths)} "
            f"required={len(required_entries)} output_dir={args.output_dir}"
        )
        processed = 0
        token = os.environ.get("HF_TOKEN")
        for entry in required_entries:
            if entry["path"] in completed_paths:
                continue
            if args.max_shards is not None and processed >= args.max_shards:
                break
            started = time.perf_counter()
            operation_log.write(
                f"private-shard-start path={entry['path']} "
                f"tasks={len(selected_by_shard[entry['path']])}"
            )
            rows, transfer = read_private_shard(
                manifest, entry, selected_by_shard[entry["path"]], token
            )
            output_name = Path(entry["path"]).stem + ".private.jsonl"
            output_path = args.output_dir / output_name
            write_jsonl(output_path, rows)
            test_count = sum(len(row["private_test_cases"]) for row in rows)
            elapsed = time.perf_counter() - started
            record = {
                "source": entry["path"],
                "file": output_name,
                "tasks": len(rows),
                "tests": test_count,
                "transfer_requests": transfer["requests"],
                "transfer_bytes": transfer["bytes"],
                "output_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "elapsed_seconds": elapsed,
            }
            state["completed"].append(record)
            atomic_json(args.state, state)
            processed += 1
            operation_log.write(
                f"private-shard-done path={entry['path']} tests={test_count} "
                f"transfer={transfer['bytes']} output={record['output_bytes']} "
                f"elapsed={elapsed:.2f}s"
            )
        completed_paths = {record["source"] for record in state["completed"]}
        if completed_paths == {entry["path"] for entry in required_entries}:
            state["status"] = "complete"
            state["summary"] = {
                "tasks": sum(record["tasks"] for record in state["completed"]),
                "tests": sum(record["tests"] for record in state["completed"]),
                "transfer_bytes": sum(
                    record["transfer_bytes"] for record in state["completed"]
                ),
                "output_bytes": sum(
                    record["output_bytes"] for record in state["completed"]
                ),
            }
            atomic_json(args.state, state)
            operation_log.write(
                f"private-complete {json.dumps(state['summary'], sort_keys=True)}"
            )
        else:
            operation_log.write(
                f"private-paused completed={len(completed_paths)}/"
                f"{len(required_entries)}"
            )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if operation_log is not None:
            operation_log.write(f"private-failed error={exc}")
        print(f"nemotron LiveCodeBench private error: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

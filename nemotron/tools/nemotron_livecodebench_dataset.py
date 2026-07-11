#!/usr/bin/env python3
"""Range-read and validate a dated official LiveCodeBench public-test catalog."""

from __future__ import annotations

import argparse
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from nemotron_metadata import MetadataError, load_json, require, sha256_file
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-livecodebench-dated-v1"
MANIFEST_FORMAT = "nemotron-livecodebench-parquet-manifest-v1"
READ_COLUMNS = (
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "metadata",
)


class HTTPRangeReader(io.RawIOBase):
    def __init__(
        self, url: str, size: int, xet_hash: str, token: str | None = None
    ) -> None:
        self.url = url
        self.size = size
        self.position = 0
        self.requests = 0
        self.bytes_read = 0
        self.token = token
        self.xet_hash = xet_hash

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if not 0 <= position <= self.size:
            raise ValueError(f"seek outside source: {position}")
        self.position = position
        return position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        length = min(len(buffer), self.size - self.position)
        end = self.position + length - 1
        headers = {"Range": f"bytes={self.position}-{end}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                content_range = response.headers.get("Content-Range", "")
                etag = response.headers.get("ETag", "").strip('"')
        except (OSError, urllib.error.HTTPError) as exc:
            raise MetadataError(
                f"HTTP range request failed at {self.position}-{end}: {exc}"
            ) from exc
        require(
            len(payload) == length,
            f"range response length mismatch: expected {length}, got {len(payload)}",
        )
        require(
            content_range.startswith(f"bytes {self.position}-{end}/"),
            f"invalid Content-Range: {content_range!r}",
        )
        require(etag == self.xet_hash, f"range response object mismatch: {etag!r}")
        buffer[:length] = payload
        self.position += length
        self.requests += 1
        self.bytes_read += length
        return length


def parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise MetadataError(f"invalid contest date: {value!r}") from exc


def parse_json_field(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise MetadataError(f"invalid JSON in {name}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL row {line_number} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    require(manifest.get("format") == MANIFEST_FORMAT, "unsupported dataset manifest")
    require(
        isinstance(manifest.get("repository"), str) and manifest["repository"],
        "manifest repository is missing",
    )
    revision = manifest.get("revision")
    require(isinstance(revision, str) and len(revision) == 40, "invalid dataset revision")
    files = manifest.get("files")
    require(isinstance(files, list) and files, "dataset manifest has no files")
    for entry in files:
        require(isinstance(entry, dict), "invalid manifest file entry")
        require(
            isinstance(entry.get("path"), str) and entry["path"].endswith(".parquet"),
            "invalid Parquet path",
        )
        require(isinstance(entry.get("size"), int) and entry["size"] > 8, "invalid file size")
        digest = entry.get("lfs_sha256")
        require(isinstance(digest, str) and len(digest) == 64, "invalid LFS SHA-256")
        xet_hash = entry.get("xet_hash")
        require(isinstance(xet_hash, str) and len(xet_hash) == 64, "invalid Xet hash")
    return files


def source_url(manifest: dict[str, Any], entry: dict[str, Any]) -> str:
    repository = urllib.parse.quote(manifest["repository"], safe="/")
    revision = urllib.parse.quote(manifest["revision"], safe="")
    path = urllib.parse.quote(entry["path"], safe="/")
    return f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"


def inspect_shard(
    manifest: dict[str, Any], entry: dict[str, Any], token: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, int]], dict[str, int]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise MetadataError("pyarrow is required for ranged Parquet metadata reads") from exc

    reader = HTTPRangeReader(
        source_url(manifest, entry), entry["size"], entry["xet_hash"], token
    )
    buffered = io.BufferedReader(reader, buffer_size=64 * 1024)
    parquet_file = parquet.ParquetFile(buffered)
    require(parquet_file.schema.names == list(READ_COLUMNS)[:9] + ["private_test_cases", "metadata"], "unexpected LiveCodeBench schema")
    private_chunks = []
    first_row = 0
    private_index = parquet_file.schema.names.index("private_test_cases")
    for group_index in range(parquet_file.metadata.num_row_groups):
        group = parquet_file.metadata.row_group(group_index)
        column = group.column(private_index)
        start = column.dictionary_page_offset
        if start is None or start < 0:
            start = column.data_page_offset
        private_chunks.append(
            {
                "row_group": group_index,
                "first_row": first_row,
                "rows": group.num_rows,
                "offset": start,
                "compressed_bytes": column.total_compressed_size,
            }
        )
        first_row += group.num_rows
    table = parquet_file.read(columns=list(READ_COLUMNS))
    rows = table.to_pylist()
    require(len(rows) == parquet_file.metadata.num_rows, "Parquet row count mismatch")
    return rows, private_chunks, {
        "requests": reader.requests,
        "bytes": reader.bytes_read,
    }


def join_official_rows(
    official: list[dict[str, Any]], local: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    official_by_id = {str(row["question_id"]): row for row in official}
    local_by_id = {str(row["question_id"]): row for row in local}
    require(len(official_by_id) == len(official), "duplicate official question ID")
    require(len(local_by_id) == len(local), "duplicate local question ID")
    missing = official_by_id.keys() - local_by_id.keys()
    require(not missing, f"local dataset is missing {len(missing)} official question IDs")
    joined = []
    for question_id, source in official_by_id.items():
        candidate = local_by_id[question_id]
        for field in ("question_title", "question_content", "starter_code", "difficulty"):
            require(
                candidate.get(field) == source.get(field),
                f"local dataset mismatch for {question_id}: {field}",
            )
        official_tests = parse_json_field(source.get("public_test_cases"), "public_test_cases")
        local_tests = parse_json_field(
            candidate.get("public_test_cases"), "local public_test_cases"
        )
        require(
            local_tests == official_tests,
            f"local dataset mismatch for {question_id}: public_test_cases",
        )
        joined.append(
            {
                **candidate,
                "public_test_cases": local_tests,
                "contest_date": source["contest_date"],
                "platform": source["platform"],
                "contest_id": source["contest_id"],
                "metadata": parse_json_field(source.get("metadata"), "metadata"),
                "official_shard": source["_official_shard"],
                "official_row": source["_official_row"],
                "official_row_group": source["_official_row_group"],
            }
        )
    return joined


def dated_rows(rows: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    selected = [row for row in rows if start <= parse_date(row["contest_date"]) <= end]
    return sorted(selected, key=lambda row: (row["contest_date"], str(row["question_id"])))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--local-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        manifest = load_json(args.manifest)
        files = validate_manifest(manifest)
        start = parse_date(args.start_date)
        end = parse_date(args.end_date)
        require(start <= end, "start date follows end date")
        official = []
        shard_records = []
        private_chunks_by_shard: dict[str, list[dict[str, int]]] = {}
        token = os.environ.get("HF_TOKEN")
        for entry in files:
            rows, private_chunks, transfer = inspect_shard(manifest, entry, token)
            for row_index, row in enumerate(rows):
                row["_official_shard"] = entry["path"]
                row["_official_row"] = row_index
                for chunk in private_chunks:
                    if chunk["first_row"] <= row_index < chunk["first_row"] + chunk["rows"]:
                        row["_official_row_group"] = chunk["row_group"]
                        break
            official.extend(rows)
            private_chunks_by_shard[entry["path"]] = private_chunks
            shard_records.append({**entry, "metadata_transfer": transfer})
            print(
                f"catalog-shard path={entry['path']} rows={len(rows)} "
                f"requests={transfer['requests']} bytes={transfer['bytes']}"
            )
        local = load_jsonl(args.local_dataset)
        joined = join_official_rows(official, local)
        selected = dated_rows(joined, start, end)
        require(selected, "dated split is empty")
        required_groups = {
            (row["official_shard"], row["official_row_group"]) for row in selected
        }
        hidden_chunks = []
        for shard, group_index in sorted(required_groups):
            chunk = private_chunks_by_shard[shard][group_index]
            hidden_chunks.append({"shard": shard, **chunk})
        write_jsonl(args.output, selected)
        state = {
            "format": FORMAT,
            "status": "complete",
            "manifest_sha256": sha256_file(args.manifest),
            "local_dataset_sha256": sha256_file(args.local_dataset),
            "output_sha256": sha256_file(args.output),
            "repository": manifest["repository"],
            "revision": manifest["revision"],
            "config": manifest.get("config"),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "official_rows": len(official),
            "local_rows": len(local),
            "local_only_rows": len(local) - len(official),
            "selected_rows": len(selected),
            "difficulty_counts": {
                difficulty: sum(row["difficulty"] == difficulty for row in selected)
                for difficulty in ("easy", "medium", "hard")
            },
            "metadata_transfer_bytes": sum(
                record["metadata_transfer"]["bytes"] for record in shard_records
            ),
            "metadata_requests": sum(
                record["metadata_transfer"]["requests"] for record in shard_records
            ),
            "hidden_test_transfer_bytes": sum(
                chunk["compressed_bytes"] for chunk in hidden_chunks
            ),
            "hidden_test_chunks": hidden_chunks,
            "shards": shard_records,
        }
        atomic_json(args.state, state)
        print(
            f"catalog-done rows={len(selected)} output={args.output} "
            f"metadata_bytes={state['metadata_transfer_bytes']} "
            f"hidden_bytes={state['hidden_test_transfer_bytes']}"
        )
        return 0
    except MetadataError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

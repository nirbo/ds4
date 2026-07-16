#!/usr/bin/env python3
"""Verify the complete pinned Ornith-35 source and write durable state."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import struct
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
)
VERIFIED_STATE_FORMAT = "ornith35-source-verified-v1"
SOURCE_STATE_FORMAT = "ornith35-source-metadata-v1"
EXPECTED_REPOSITORY = "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
EXPECTED_REVISION = "85ffd2d0629ae5fa4f860dda356ec33161806c9b"
EXPECTED_WEIGHT_BYTES = 23_741_821_016
EXPECTED_WEIGHT_SHA256 = "68a4b2b8605076825302be20132cf69342b44a0385c19e6de741af5ec3114ca0"
READ_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 1024 * 1024 * 1024


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path, *, progress_bytes: int = 0) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    processed = 0
    next_progress = progress_bytes
    started = time.monotonic()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_BYTES):
            digest.update(chunk)
            processed += len(chunk)
            if progress_bytes and processed >= next_progress:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    "verify-hash-progress "
                    f"bytes={processed}/{total} "
                    f"percent={processed * 100 / total:.2f} "
                    f"rate_mib_s={processed / 2**20 / elapsed:.1f}",
                    flush=True,
                )
                while next_progress <= processed:
                    next_progress += progress_bytes
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    rendered = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with temporary.open("wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_metadata_state(
    metadata_state: dict[str, Any], metadata_dir: Path, *, strict_target: bool
) -> dict[str, Any]:
    require(metadata_state.get("format") == SOURCE_STATE_FORMAT, "unsupported metadata state")
    require(isinstance(metadata_state.get("metadata_files"), dict), "missing metadata hashes")
    weight = metadata_state.get("weight")
    require(isinstance(weight, dict), "missing source weight metadata")
    if strict_target:
        require(metadata_state.get("repository") == EXPECTED_REPOSITORY, "repository mismatch")
        require(metadata_state.get("revision") == EXPECTED_REVISION, "revision mismatch")
        require(weight.get("name") == "model.safetensors", "weight filename mismatch")
        require(weight.get("file_bytes") == EXPECTED_WEIGHT_BYTES, "weight size identity mismatch")
        require(weight.get("sha256") == EXPECTED_WEIGHT_SHA256, "weight hash identity mismatch")

    for name in ("config.json", "model.safetensors.header.json"):
        entry = metadata_state["metadata_files"].get(name)
        path = metadata_dir / name
        require(isinstance(entry, dict), f"metadata state does not cover {name}")
        require(path.is_file(), f"missing metadata file: {path}")
        require(entry.get("bytes") == path.stat().st_size, f"metadata size mismatch: {name}")
        require(entry.get("sha256") == sha256_file(path), f"metadata hash mismatch: {name}")
    return weight


def verify_weight_file(
    source_path: Path,
    header_path: Path,
    weight: dict[str, Any],
    *,
    progress_bytes: int = PROGRESS_BYTES,
) -> dict[str, Any]:
    require(not source_path.is_symlink(), "source weight must not be a symbolic link")
    try:
        source_stat = source_path.stat()
    except OSError as exc:
        raise VerificationError(f"cannot stat source weight {source_path}: {exc}") from exc
    require(stat.S_ISREG(source_stat.st_mode), "source weight is not a regular file")
    require(source_stat.st_size == weight.get("file_bytes"), "source weight size mismatch")

    try:
        expected_header = header_path.read_bytes()
        with source_path.open("rb") as handle:
            prefix = handle.read(8)
            require(len(prefix) == 8, "truncated safetensors prefix")
            header_bytes = struct.unpack("<Q", prefix)[0]
            require(header_bytes == weight.get("header_bytes"), "safetensors header size mismatch")
            actual_header = handle.read(header_bytes)
    except OSError as exc:
        raise VerificationError(f"cannot inspect source weight {source_path}: {exc}") from exc
    require(
        actual_header == expected_header,
        "source safetensors header differs from pinned header",
    )
    require(
        source_stat.st_size - 8 - header_bytes == weight.get("payload_bytes"),
        "source payload size mismatch",
    )

    print(
        f"verify-hash-start path={source_path} bytes={source_stat.st_size}",
        flush=True,
    )
    started = time.monotonic()
    digest = sha256_file(source_path, progress_bytes=progress_bytes)
    elapsed = max(time.monotonic() - started, 1e-9)
    source_stat_after = source_path.stat()
    require(
        (
            source_stat_after.st_ino,
            source_stat_after.st_size,
            source_stat_after.st_mtime_ns,
        )
        == (source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns),
        "source weight changed during verification",
    )
    require(digest == weight.get("sha256"), "source weight SHA-256 mismatch")
    print(
        "verify-hash-done "
        f"sha256={digest} elapsed_s={elapsed:.2f} "
        f"rate_mib_s={source_stat.st_size / 2**20 / elapsed:.1f}",
        flush=True,
    )
    return {
        "name": source_path.name,
        "bytes": source_stat.st_size,
        "sha256": digest,
        "header_bytes": header_bytes,
        "header_sha256": hashlib.sha256(expected_header).hexdigest(),
        "payload_bytes": source_stat.st_size - 8 - header_bytes,
    }


def verify_source(
    root: Path,
    *,
    source_path: Path | None = None,
    output_path: Path | None = None,
    strict_target: bool = True,
    progress_bytes: int = PROGRESS_BYTES,
) -> dict[str, Any]:
    metadata_dir = root / "metadata"
    metadata_state_path = metadata_dir / "source-state.json"
    header_path = metadata_dir / "model.safetensors.header.json"
    source_path = source_path or root / "source-nvfp4" / "model.safetensors"
    output_path = output_path or root / "source-nvfp4-state.json"

    metadata_state = load_json(metadata_state_path)
    weight = validate_metadata_state(metadata_state, metadata_dir, strict_target=strict_target)
    verified_weight = verify_weight_file(
        source_path,
        header_path,
        weight,
        progress_bytes=progress_bytes,
    )
    verifier_path = Path(__file__).resolve()
    result = {
        "format": VERIFIED_STATE_FORMAT,
        "repository": metadata_state["repository"],
        "revision": metadata_state["revision"],
        "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "weight": verified_weight,
        "metadata": {
            "state_file": str(metadata_state_path),
            "state_sha256": sha256_file(metadata_state_path),
            "config_sha256": sha256_file(metadata_dir / "config.json"),
            "header_sha256": sha256_file(header_path),
        },
        "verifier": {
            "path": str(verifier_path),
            "sha256": sha256_file(verifier_path),
        },
    }
    atomic_write_json(output_path, result)
    print(f"verify-source-done state={output_path}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        verify_source(args.root, source_path=args.source, output_path=args.out)
    except (VerificationError, OSError) as exc:
        print(f"ornith35 source verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

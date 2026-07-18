#!/usr/bin/env python3
"""Extract the pinned Qwen3.5 MTP tensors one verified shard at a time."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import time
from typing import Any, BinaryIO


DEFAULT_ROOT = Path(
    "/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
)
STATE_FORMAT = "ornith35-mtp-extract-v1"
METADATA_FORMAT = "ornith35-companion-metadata-v1"
READ_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 1024 * 1024 * 1024


class MTPExtractionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MTPExtractionError(message)


@dataclass(frozen=True)
class PinnedShard:
    name: str
    file_bytes: int
    sha256: str
    header_bytes: int
    header_sha256: str
    payload_bytes: int
    mtp_tensor_count: int
    mtp_payload_bytes: int


@dataclass(frozen=True)
class ExtractionProfile:
    key: str
    repository: str
    revision: str
    tensor_count: int
    payload_bytes: int
    shards: tuple[PinnedShard, ...]


PRODUCTION_PROFILE = ExtractionProfile(
    key="mtp-source",
    repository="Qwen/Qwen3.5-35B-A3B",
    revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    tensor_count=785,
    payload_bytes=1_689_281_536,
    shards=(
        PinnedShard(
            name="model.safetensors-00013-of-00014.safetensors",
            file_bytes=5_367_839_544,
            sha256="da8cc27a2a99eeba5674bfef03c80b3c4f08edfd839ad319a7702fcaedfd874f",
            header_bytes=23_344,
            header_sha256="5d7fb10af28ff02acfa88a1e623ccb41b360e442ff2bddeefa131a71235da416",
            payload_bytes=5_367_816_192,
            mtp_tensor_count=3,
            mtp_payload_bytes=67_108_864,
        ),
        PinnedShard(
            name="model.safetensors-00014-of-00014.safetensors",
            file_bytes=2_224_764_664,
            sha256="d5e08a7dd670d7ef8da38e7af29cb8cdf42ecded5dfb2d9fcaf06cae79dfaa41",
            header_bytes=189_072,
            header_sha256="091eaab7a7dd4059a782c2fed3e4c3942546ef4803f9be95ff14bf63bc33c59d",
            payload_bytes=2_224_575_584,
            mtp_tensor_count=782,
            mtp_payload_bytes=1_622_172_672,
        ),
    ),
)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    source_start: int
    source_end: int
    output_start: int
    output_end: int

    @property
    def nbytes(self) -> int:
        return self.source_end - self.source_start


@dataclass(frozen=True)
class ExtractionCatalog:
    tensors: tuple[TensorSpec, ...]
    by_shard: dict[str, tuple[TensorSpec, ...]]
    output_header: bytes
    output_bytes: int
    metadata_state_sha256: str


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def file_identity(value: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MTPExtractionError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object in {path}")
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
                    "mtp-hash-progress "
                    f"bytes={processed}/{total} "
                    f"percent={processed * 100.0 / total:.2f} "
                    f"rate_mib_s={processed / 2**20 / elapsed:.1f}",
                    flush=True,
                )
                while next_progress <= processed:
                    next_progress += progress_bytes
    return digest.hexdigest()


def sha256_range(handle: BinaryIO, start: int, length: int) -> str:
    digest = hashlib.sha256()
    handle.seek(start)
    remaining = length
    while remaining:
        chunk = handle.read(min(READ_BYTES, remaining))
        require(chunk, "truncated tensor payload")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def repository_revision(tool_path: Path) -> str:
    repo_root = tool_path.resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MTPExtractionError(f"cannot resolve repository revision: {exc}") from exc
    revision = result.stdout.strip()
    require(
        len(revision) == 40 and all(value in "0123456789abcdef" for value in revision),
        "invalid repository revision",
    )
    return revision


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
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safetensors_header_bytes(header: dict[str, Any]) -> bytes:
    rendered = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return rendered + b" " * (-len(rendered) % 8)


def _tensor_nbytes(dtype: str, shape: tuple[int, ...]) -> int:
    require(dtype == "BF16", "MTP tensor is not BF16")
    elements = 1
    for dimension in shape:
        require(isinstance(dimension, int) and dimension > 0, "invalid MTP shape")
        elements *= dimension
    return elements * 2


def _validate_metadata_files(metadata_dir: Path, state: dict[str, Any]) -> None:
    entries = state.get("metadata_files")
    require(isinstance(entries, dict), "MTP metadata hashes are absent")
    for name, expected in entries.items():
        require(isinstance(name, str) and isinstance(expected, dict), "invalid metadata entry")
        path = metadata_dir / name
        require(path.is_file() and not path.is_symlink(), f"missing MTP metadata: {name}")
        require(path.stat().st_size == expected.get("bytes"), f"metadata size drift: {name}")
        require(sha256_file(path) == expected.get("sha256"), f"metadata hash drift: {name}")


def build_catalog(
    root: Path,
    profile: ExtractionProfile = PRODUCTION_PROFILE,
) -> ExtractionCatalog:
    metadata_dir = root / "metadata-mtp-source"
    state_path = metadata_dir / "source-state.json"
    state = load_json(state_path)
    require(state.get("format") == METADATA_FORMAT, "unsupported MTP metadata state")
    require(state.get("profile") == profile.key, "MTP metadata profile mismatch")
    require(state.get("repository") == profile.repository, "MTP repository mismatch")
    require(state.get("revision") == profile.revision, "MTP revision mismatch")
    _validate_metadata_files(metadata_dir, state)

    mtp = state.get("mtp")
    require(isinstance(mtp, dict), "MTP inventory is absent")
    require(mtp.get("tensor_count") == profile.tensor_count, "MTP tensor count drift")
    require(mtp.get("payload_bytes") == profile.payload_bytes, "MTP payload size drift")
    state_shards = mtp.get("shards")
    require(isinstance(state_shards, list), "MTP shard inventory is absent")
    state_by_name = {
        value.get("name"): value for value in state_shards if isinstance(value, dict)
    }
    require(len(state_by_name) == len(profile.shards), "MTP shard count drift")

    index = load_json(metadata_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict), "MTP source index has no weight map")
    mtp_names = sorted(
        name for name in weight_map if isinstance(name, str) and name.startswith("mtp.")
    )
    require(len(mtp_names) == profile.tensor_count, "MTP index tensor count drift")

    source_entries: dict[str, tuple[str, dict[str, Any]]] = {}
    profile_names = {shard.name for shard in profile.shards}
    for shard in profile.shards:
        inventory = state_by_name.get(shard.name)
        require(isinstance(inventory, dict), f"MTP inventory omits {shard.name}")
        expected_inventory = {
            "file_bytes": shard.file_bytes,
            "sha256": shard.sha256,
            "header_bytes": shard.header_bytes,
            "header_sha256": shard.header_sha256,
            "payload_bytes": shard.payload_bytes,
        }
        require(
            all(inventory.get(key) == value for key, value in expected_inventory.items()),
            f"pinned MTP shard identity drift: {shard.name}",
        )
        header_name = inventory.get("header_file")
        require(isinstance(header_name, str), f"MTP header name missing: {shard.name}")
        header_path = metadata_dir / header_name
        header_bytes = header_path.read_bytes()
        require(len(header_bytes) == shard.header_bytes, f"MTP header size drift: {shard.name}")
        require(
            hashlib.sha256(header_bytes).hexdigest() == shard.header_sha256,
            f"MTP header hash drift: {shard.name}",
        )
        try:
            header = json.loads(header_bytes)
        except json.JSONDecodeError as exc:
            raise MTPExtractionError(f"invalid MTP header {header_name}: {exc}") from exc
        require(isinstance(header, dict), f"invalid MTP header object: {header_name}")
        header_mtp_names = {
            name for name in header if isinstance(name, str) and name.startswith("mtp.")
        }
        indexed_shard_names = {
            name for name in mtp_names if weight_map.get(name) == shard.name
        }
        require(
            header_mtp_names == indexed_shard_names,
            f"MTP index/header coverage drift: {shard.name}",
        )
        ranges = []
        shard_payload = 0
        for name in sorted(header_mtp_names):
            entry = header.get(name)
            require(isinstance(entry, dict), f"invalid MTP tensor entry: {name}")
            dtype = entry.get("dtype")
            shape_value = entry.get("shape")
            offsets = entry.get("data_offsets")
            require(isinstance(shape_value, list), f"invalid MTP shape: {name}")
            require(
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(value, int) for value in offsets),
                f"invalid MTP offsets: {name}",
            )
            shape = tuple(shape_value)
            start, end = offsets
            require(0 <= start < end <= shard.payload_bytes, f"MTP range invalid: {name}")
            require(end - start == _tensor_nbytes(dtype, shape), f"MTP size mismatch: {name}")
            ranges.append((start, end, name))
            shard_payload += end - start
            source_entries[name] = (shard.name, entry)
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            require(previous[1] <= current[0], f"overlapping MTP ranges: {shard.name}")
        require(len(ranges) == shard.mtp_tensor_count, f"MTP shard tensor drift: {shard.name}")
        require(shard_payload == shard.mtp_payload_bytes, f"MTP shard payload drift: {shard.name}")

    require(
        all(weight_map.get(name) in profile_names for name in mtp_names),
        "MTP index points outside pinned shards",
    )
    require(set(source_entries) == set(mtp_names), "MTP tensor coverage is incomplete")

    output_entries: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    output_offset = 0
    provisional = []
    for name in mtp_names:
        shard_name, entry = source_entries[name]
        source_start, source_end = entry["data_offsets"]
        nbytes = source_end - source_start
        output_entries[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [output_offset, output_offset + nbytes],
        }
        provisional.append(
            TensorSpec(
                name=name,
                dtype=entry["dtype"],
                shape=tuple(entry["shape"]),
                shard=shard_name,
                source_start=source_start,
                source_end=source_end,
                output_start=output_offset,
                output_end=output_offset + nbytes,
            )
        )
        output_offset += nbytes
    require(output_offset == profile.payload_bytes, "MTP output payload drift")
    output_header = _safetensors_header_bytes(output_entries)
    by_shard = {
        shard.name: tuple(value for value in provisional if value.shard == shard.name)
        for shard in profile.shards
    }
    return ExtractionCatalog(
        tensors=tuple(provisional),
        by_shard=by_shard,
        output_header=output_header,
        output_bytes=8 + len(output_header) + profile.payload_bytes,
        metadata_state_sha256=sha256_file(state_path),
    )


def _expected_state(
    profile: ExtractionProfile,
    catalog: ExtractionCatalog,
    tool_sha256: str,
    runtime_revision: str,
    output_name: str,
) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "status": "partial",
        "profile": profile.key,
        "repository": profile.repository,
        "revision": profile.revision,
        "runtime_revision": runtime_revision,
        "tool_sha256": tool_sha256,
        "metadata_state_sha256": catalog.metadata_state_sha256,
        "output": {
            "name": output_name,
            "part_name": output_name + ".part",
            "bytes": catalog.output_bytes,
            "header_bytes": len(catalog.output_header),
            "header_sha256": hashlib.sha256(catalog.output_header).hexdigest(),
            "payload_bytes": profile.payload_bytes,
            "tensor_count": profile.tensor_count,
        },
        "completed_shards": {},
    }


def _validate_state_identity(
    state: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key in (
        "format",
        "profile",
        "repository",
        "revision",
        "runtime_revision",
        "tool_sha256",
        "metadata_state_sha256",
        "output",
    ):
        require(state.get(key) == expected.get(key), f"MTP extraction state drift: {key}")
    require(state.get("status") in ("partial", "complete"), "invalid MTP extraction status")
    require(isinstance(state.get("completed_shards"), dict), "invalid completed shard state")


def _validate_completed_state(
    completed_shards: dict[str, Any],
    profile: ExtractionProfile,
    catalog: ExtractionCatalog,
) -> None:
    pinned_by_name = {shard.name: shard for shard in profile.shards}
    require(
        set(completed_shards).issubset(pinned_by_name),
        "MTP state contains an unknown completed shard",
    )
    for shard_name, completed in completed_shards.items():
        require(isinstance(completed, dict), f"invalid completed state: {shard_name}")
        shard = pinned_by_name[shard_name]
        specs = catalog.by_shard[shard_name]
        require(
            completed.get("source_sha256") == shard.sha256,
            f"MTP completed source hash drift: {shard_name}",
        )
        require(
            completed.get("tensor_count") == len(specs),
            f"MTP completed tensor count drift: {shard_name}",
        )
        require(
            completed.get("payload_bytes") == shard.mtp_payload_bytes,
            f"MTP completed payload drift: {shard_name}",
        )


def _validate_output_container(
    path: Path,
    catalog: ExtractionCatalog,
) -> None:
    require(path.is_file() and not path.is_symlink(), f"MTP output is absent: {path}")
    require(path.stat().st_size == catalog.output_bytes, "MTP output size drift")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, "truncated MTP output prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        require(header_bytes == len(catalog.output_header), "MTP output header size drift")
        require(handle.read(header_bytes) == catalog.output_header, "MTP output header drift")


def _verify_completed_ranges(
    output_path: Path,
    catalog: ExtractionCatalog,
    completed_shards: dict[str, Any],
) -> None:
    payload_base = 8 + len(catalog.output_header)
    with output_path.open("rb") as handle:
        for shard_name, completed in completed_shards.items():
            require(isinstance(completed, dict), f"invalid completed state: {shard_name}")
            hashes = completed.get("tensor_sha256")
            require(isinstance(hashes, dict), f"MTP tensor hashes absent: {shard_name}")
            specs = catalog.by_shard.get(shard_name)
            require(specs is not None, f"state names unknown MTP shard: {shard_name}")
            require(set(hashes) == {spec.name for spec in specs}, "MTP hash coverage drift")
            for spec in specs:
                digest = sha256_range(
                    handle,
                    payload_base + spec.output_start,
                    spec.nbytes,
                )
                require(digest == hashes[spec.name], f"MTP output tensor drift: {spec.name}")


def _initialize_output(
    part_path: Path,
    catalog: ExtractionCatalog,
) -> None:
    require(not part_path.exists(), f"orphan MTP partial output exists: {part_path}")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    with part_path.open("xb") as handle:
        handle.write(struct.pack("<Q", len(catalog.output_header)))
        handle.write(catalog.output_header)
        handle.truncate(catalog.output_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(part_path.parent)


def _verify_source_shard(path: Path, shard: PinnedShard) -> FileIdentity:
    require(path.is_file() and not path.is_symlink(), f"MTP source shard is absent: {path}")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "MTP source shard is not regular")
    require(before.st_size == shard.file_bytes, f"MTP source size drift: {shard.name}")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, f"truncated MTP source prefix: {shard.name}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        require(header_bytes == shard.header_bytes, f"MTP source header size drift: {shard.name}")
        header = handle.read(header_bytes)
    require(
        hashlib.sha256(header).hexdigest() == shard.header_sha256,
        f"MTP source header drift: {shard.name}",
    )
    print(f"mtp-source-hash-start shard={shard.name} bytes={shard.file_bytes}", flush=True)
    started = time.monotonic()
    digest = sha256_file(path, progress_bytes=PROGRESS_BYTES)
    elapsed = max(time.monotonic() - started, 1e-9)
    after = file_identity(path.stat())
    require(
        after == file_identity(before),
        f"MTP source changed during verification: {shard.name}",
    )
    require(digest == shard.sha256, f"MTP source SHA-256 drift: {shard.name}")
    print(
        "mtp-source-hash-done "
        f"shard={shard.name} sha256={digest} elapsed_s={elapsed:.2f} "
        f"rate_mib_s={shard.file_bytes / 2**20 / elapsed:.1f}",
        flush=True,
    )
    return after


def _copy_shard_tensors(
    source_path: Path,
    output_path: Path,
    shard: PinnedShard,
    specs: tuple[TensorSpec, ...],
    catalog: ExtractionCatalog,
    source_identity: FileIdentity,
) -> dict[str, str]:
    source_payload_base = 8 + shard.header_bytes
    output_payload_base = 8 + len(catalog.output_header)
    hashes: dict[str, str] = {}
    started = time.monotonic()
    copied = 0
    with source_path.open("rb") as source, output_path.open("r+b") as output:
        require(
            file_identity(os.fstat(source.fileno())) == source_identity,
            f"MTP source changed before extraction: {shard.name}",
        )
        for index, spec in enumerate(specs, start=1):
            source.seek(source_payload_base + spec.source_start)
            output.seek(output_payload_base + spec.output_start)
            digest = hashlib.sha256()
            remaining = spec.nbytes
            while remaining:
                chunk = source.read(min(READ_BYTES, remaining))
                require(chunk, f"truncated MTP tensor: {spec.name}")
                output.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                remaining -= len(chunk)
            hashes[spec.name] = digest.hexdigest()
            if index % 64 == 0 or index == len(specs):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    "mtp-extract-progress "
                    f"shard={shard.name} tensors={index}/{len(specs)} "
                    f"bytes={copied}/{shard.mtp_payload_bytes} "
                    f"rate_mib_s={copied / 2**20 / elapsed:.1f}",
                    flush=True,
                )
        output.flush()
        os.fsync(output.fileno())
        require(
            file_identity(os.fstat(source.fileno())) == source_identity,
            f"MTP source changed during extraction: {shard.name}",
        )
    with output_path.open("rb") as output:
        for spec in specs:
            digest = sha256_range(
                output,
                output_payload_base + spec.output_start,
                spec.nbytes,
            )
            require(digest == hashes[spec.name], f"MTP copy verification failed: {spec.name}")
    return hashes


def _delete_verified_source(
    source_path: Path,
    raw_dir: Path,
    shard: PinnedShard,
    source_identity: FileIdentity | None = None,
) -> None:
    identity = source_identity or _verify_source_shard(source_path, shard)
    require(
        file_identity(source_path.stat()) == identity,
        f"MTP source changed before deletion: {shard.name}",
    )
    source_path.unlink()
    fsync_directory(raw_dir)
    print(f"mtp-source-deleted shard={shard.name}", flush=True)


def _delete_completed_sources(
    raw_dir: Path,
    profile: ExtractionProfile,
    completed_shards: dict[str, Any],
    shard_name: str | None,
) -> None:
    for shard in profile.shards:
        if shard.name not in completed_shards:
            continue
        if shard_name is not None and shard.name != shard_name:
            continue
        source_path = raw_dir / shard.name
        if source_path.exists():
            _delete_verified_source(source_path, raw_dir, shard)


def print_extraction_plan(
    root: Path,
    profile: ExtractionProfile = PRODUCTION_PROFILE,
) -> None:
    catalog = build_catalog(root, profile)
    source_bytes = sum(shard.file_bytes for shard in profile.shards)
    conservative_peak = max(
        shard.file_bytes + catalog.output_bytes for shard in profile.shards
    )
    free_bytes = shutil.disk_usage(root).free
    print(
        "mtp-plan "
        f"repository={profile.repository} revision={profile.revision} "
        f"shards={len(profile.shards)} source_bytes={source_bytes} "
        f"output_bytes={catalog.output_bytes} "
        f"conservative_peak_bytes={conservative_peak} free_bytes={free_bytes}"
    )
    for shard in profile.shards:
        print(
            "mtp-plan-shard "
            f"name={shard.name} file_bytes={shard.file_bytes} "
            f"sha256={shard.sha256} mtp_tensors={shard.mtp_tensor_count} "
            f"mtp_payload_bytes={shard.mtp_payload_bytes}"
        )


def _complete_output(
    part_path: Path,
    final_path: Path,
    state_path: Path,
    state: dict[str, Any],
    catalog: ExtractionCatalog,
) -> dict[str, Any]:
    _validate_output_container(part_path, catalog)
    _verify_completed_ranges(part_path, catalog, state["completed_shards"])
    print(f"mtp-output-hash-start path={part_path} bytes={catalog.output_bytes}", flush=True)
    digest = sha256_file(part_path, progress_bytes=PROGRESS_BYTES)
    require(not final_path.exists(), f"MTP final output already exists: {final_path}")
    part_path.replace(final_path)
    fsync_directory(final_path.parent)
    completed = dict(state)
    completed["status"] = "complete"
    completed["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    completed["output_sha256"] = digest
    atomic_write_json(state_path, completed)
    print(f"mtp-extract-complete output={final_path} sha256={digest}", flush=True)
    return completed


def extract_available_shards(
    root: Path,
    *,
    profile: ExtractionProfile = PRODUCTION_PROFILE,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    state_path: Path | None = None,
    output_name: str = "mtp.safetensors",
    shard_name: str | None = None,
    max_shards: int = 1,
    delete_source: bool = False,
) -> dict[str, Any]:
    require(max_shards > 0, "MTP max-shards must be positive")
    require(
        output_name != "" and Path(output_name).name == output_name,
        "MTP output name must be a basename",
    )
    raw_dir = raw_dir or root / "source-mtp-raw"
    output_dir = output_dir or root / "source-mtp"
    state_path = state_path or root / "source-mtp-state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    part_path = output_dir / (output_name + ".part")
    final_path = output_dir / output_name
    catalog = build_catalog(root, profile)
    tool_path = Path(__file__).resolve()
    tool_sha256 = sha256_file(tool_path)
    expected = _expected_state(
        profile,
        catalog,
        tool_sha256,
        repository_revision(tool_path),
        output_name,
    )
    pinned_by_name = {shard.name: shard for shard in profile.shards}
    if shard_name is not None:
        require(shard_name in pinned_by_name, f"unknown pinned MTP shard: {shard_name}")

    if state_path.exists():
        state = load_json(state_path)
        _validate_state_identity(state, expected)
    else:
        require(not final_path.exists(), f"untracked MTP final output exists: {final_path}")
        if part_path.exists():
            _validate_output_container(part_path, catalog)
            print(f"mtp-extract-initial-output-recovered path={part_path}", flush=True)
        else:
            _initialize_output(part_path, catalog)
        state = expected
        atomic_write_json(state_path, state)
        print(f"mtp-extract-state-created state={state_path}", flush=True)

    if state["status"] == "complete":
        require(not part_path.exists(), "complete MTP state still has partial output")
        _validate_completed_state(state["completed_shards"], profile, catalog)
        require(
            set(state["completed_shards"]) == set(pinned_by_name),
            "complete MTP state has incomplete shard coverage",
        )
        _validate_output_container(final_path, catalog)
        require(
            sha256_file(final_path) == state.get("output_sha256"),
            "complete MTP output hash drift",
        )
        if delete_source:
            _delete_completed_sources(
                raw_dir,
                profile,
                state["completed_shards"],
                shard_name,
            )
        print(f"mtp-extract-already-complete output={final_path}", flush=True)
        return state

    if final_path.exists():
        require(not part_path.exists(), "partial and final MTP outputs both exist")
        require(
            len(state["completed_shards"]) == len(profile.shards),
            "incomplete MTP state conflicts with final output",
        )
        _validate_completed_state(state["completed_shards"], profile, catalog)
        _validate_output_container(final_path, catalog)
        _verify_completed_ranges(final_path, catalog, state["completed_shards"])
        recovered = dict(state)
        recovered["status"] = "complete"
        recovered["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        )
        recovered["output_sha256"] = sha256_file(
            final_path,
            progress_bytes=PROGRESS_BYTES,
        )
        atomic_write_json(state_path, recovered)
        if delete_source:
            _delete_completed_sources(
                raw_dir,
                profile,
                recovered["completed_shards"],
                shard_name,
            )
        print(f"mtp-extract-recovered output={final_path}", flush=True)
        return recovered
    _validate_output_container(part_path, catalog)
    completed_shards = dict(state["completed_shards"])
    _validate_completed_state(completed_shards, profile, catalog)
    _verify_completed_ranges(part_path, catalog, completed_shards)

    if shard_name is not None:
        if shard_name in completed_shards:
            source_path = raw_dir / shard_name
            if delete_source and source_path.exists():
                _delete_verified_source(
                    source_path,
                    raw_dir,
                    pinned_by_name[shard_name],
                )
            if len(completed_shards) == len(profile.shards):
                return _complete_output(
                    part_path,
                    final_path,
                    state_path,
                    state,
                    catalog,
                )
            print(f"mtp-extract-shard-already-complete shard={shard_name}", flush=True)
            return state
        candidates = [pinned_by_name[shard_name]]
    else:
        candidates = list(profile.shards)
    processed = 0
    for shard in candidates:
        if shard.name in completed_shards:
            continue
        source_path = raw_dir / shard.name
        if not source_path.exists():
            if shard_name is not None:
                raise MTPExtractionError(f"requested MTP source is absent: {source_path}")
            continue
        print(f"mtp-extract-shard-start shard={shard.name} source={source_path}", flush=True)
        source_identity = _verify_source_shard(source_path, shard)
        hashes = _copy_shard_tensors(
            source_path,
            part_path,
            shard,
            catalog.by_shard[shard.name],
            catalog,
            source_identity,
        )
        completed_shards[shard.name] = {
            "source_sha256": shard.sha256,
            "tensor_count": len(catalog.by_shard[shard.name]),
            "payload_bytes": sum(spec.nbytes for spec in catalog.by_shard[shard.name]),
            "tensor_sha256": hashes,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        state = dict(state)
        state["completed_shards"] = completed_shards
        atomic_write_json(state_path, state)
        print(
            f"mtp-extract-shard-verified shard={shard.name} state={state_path}",
            flush=True,
        )
        if delete_source:
            _delete_verified_source(
                source_path,
                raw_dir,
                shard,
                source_identity,
            )
        processed += 1
        if processed >= max_shards:
            break

    if len(completed_shards) == len(profile.shards):
        return _complete_output(part_path, final_path, state_path, state, catalog)
    if processed == 0:
        pending = [
            shard.name for shard in profile.shards if shard.name not in completed_shards
        ]
        raise MTPExtractionError(f"no pending MTP source shard is available: {pending}")
    print(
        "mtp-extract-paused "
        f"completed={len(completed_shards)}/{len(profile.shards)} state={state_path}",
        flush=True,
    )
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output-name", default="mtp.safetensors")
    parser.add_argument("--shard")
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="validate pinned metadata and print the no-download extraction plan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.plan:
            print_extraction_plan(args.root)
            return 0
        extract_available_shards(
            args.root,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            state_path=args.state,
            output_name=args.output_name,
            shard_name=args.shard,
            max_shards=args.max_shards,
            delete_source=args.delete_source,
        )
        return 0
    except (MTPExtractionError, OSError, ValueError) as exc:
        print(f"mtp-extract-error: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

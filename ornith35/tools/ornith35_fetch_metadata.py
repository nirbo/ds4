#!/usr/bin/env python3
"""Fetch pinned Ornith-35 metadata and safetensors headers without weights."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


TARGET_STATE_FORMAT = "ornith35-source-metadata-v1"
COMPANION_STATE_FORMAT = "ornith35-companion-metadata-v1"
DEFAULT_ROOT = Path(
    "/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
)
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
USER_AGENT = "ornith35-metadata/1"

PROFILES: dict[str, dict[str, Any]] = {
    "target": {
        "repository": "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4",
        "revision": "85ffd2d0629ae5fa4f860dda356ec33161806c9b",
        "directory": "metadata",
        "metadata": [
            "README.md",
            "QUICKSTART_DGX_SPARK.md",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "recipe.yaml",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "weight": "model.safetensors",
        "weight_bytes": 23_741_821_016,
        "weight_sha256": "68a4b2b8605076825302be20132cf69342b44a0385c19e6de741af5ec3114ca0",
        "state_format": TARGET_STATE_FORMAT,
    },
    "dspark": {
        "repository": "pablogrant/ORNITH-1.0_35B_AEON_PABLOG-OPTIMIZED_UNCENSORED_DSPARK-DRAFT_NVFP4",
        "revision": "9383b3c33ddf982114a4f72e07c890bfd6c35df2",
        "directory": "metadata-dspark",
        "metadata": ["README.md", "config.json", "config.py", "val_metrics.json"],
        "weight": "model.safetensors",
        "weight_bytes": 1_657_168_394,
        "weight_sha256": "7ab36d46959066cbb68925239e069498f2847cd0ef4be87b08a995222ee4d06b",
        "state_format": COMPANION_STATE_FORMAT,
    },
    "mtp-source": {
        "repository": "Qwen/Qwen3.5-35B-A3B",
        "revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "directory": "metadata-mtp-source",
        "metadata": ["README.md", "config.json", "model.safetensors.index.json"],
        "state_format": COMPANION_STATE_FORMAT,
    },
}


class FetchError(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        suffix = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"{timestamp} {event}" + (f" {suffix}" if suffix else "")
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def request_bytes(
    url: str,
    *,
    max_bytes: int,
    byte_range: tuple[int, int] | None = None,
) -> tuple[bytes, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = response.getcode()
            if byte_range is not None and status != 206:
                raise FetchError(f"server ignored bounded range for {url}: status={status}")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise FetchError(f"response exceeds byte ceiling for {url}: {declared}")
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise FetchError(f"response exceeded byte ceiling while reading {url}")
            return data, response.headers
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise FetchError(f"request failed for {url}: {exc}") from exc


def api_url(repository: str, revision: str) -> str:
    repo = urllib.parse.quote(repository, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"


def resolve_url(repository: str, revision: str, name: str) -> str:
    repo = urllib.parse.quote(repository, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    path = urllib.parse.quote(name, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def parse_api_manifest(data: bytes, repository: str, revision: str) -> dict[str, Any]:
    try:
        manifest = json.loads(data)
    except json.JSONDecodeError as exc:
        raise FetchError(f"invalid Hugging Face API JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FetchError("Hugging Face API response is not an object")
    if manifest.get("id") != repository or manifest.get("sha") != revision:
        raise FetchError("Hugging Face API repository or revision mismatch")
    siblings = manifest.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise FetchError("Hugging Face API response has no files")
    return manifest


def sibling_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in manifest["siblings"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("rfilename"), str):
            raise FetchError("invalid Hugging Face sibling entry")
        result[entry["rfilename"]] = entry
    return result


def fetch_metadata_file(
    repository: str,
    revision: str,
    name: str,
    expected: dict[str, Any],
    output: Path,
    logger: Logger,
) -> dict[str, Any]:
    size = expected.get("size")
    if not isinstance(size, int) or size < 0:
        raise FetchError(f"API has no exact size for metadata file {name}")
    if size > MAX_METADATA_BYTES:
        raise FetchError(f"metadata allowlist file is too large: {name} bytes={size}")
    logger.write("metadata-download-start", file=name, bytes=size)
    data, _ = request_bytes(
        resolve_url(repository, revision, name),
        max_bytes=MAX_METADATA_BYTES,
    )
    if len(data) != size:
        raise FetchError(f"metadata size mismatch for {name}: expected={size} actual={len(data)}")
    lfs = expected.get("lfs")
    digest = sha256_bytes(data)
    if isinstance(lfs, dict) and lfs.get("sha256") != digest:
        raise FetchError(f"metadata LFS hash mismatch for {name}")
    destination = output / name
    atomic_write(destination, data)
    logger.write("metadata-download-done", file=name, bytes=len(data), sha256=digest)
    return {"bytes": len(data), "sha256": digest}


def fetch_safetensors_header(
    repository: str,
    revision: str,
    name: str,
    file_info: dict[str, Any],
    output: Path,
    logger: Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    file_bytes = file_info.get("size")
    lfs = file_info.get("lfs")
    if not isinstance(file_bytes, int) or file_bytes <= 8 or not isinstance(lfs, dict):
        raise FetchError(f"invalid safetensors API metadata for {name}")
    url = resolve_url(repository, revision, name)
    logger.write("header-prefix-start", file=name, range="0-7")
    prefix, _ = request_bytes(url, max_bytes=8, byte_range=(0, 7))
    if len(prefix) != 8:
        raise FetchError(f"truncated safetensors prefix for {name}")
    header_bytes = struct.unpack("<Q", prefix)[0]
    if not 0 < header_bytes <= MAX_HEADER_BYTES:
        raise FetchError(f"unsafe safetensors header size for {name}: {header_bytes}")
    logger.write("header-json-start", file=name, bytes=header_bytes)
    raw_header, _ = request_bytes(
        url,
        max_bytes=header_bytes,
        byte_range=(8, 8 + header_bytes - 1),
    )
    if len(raw_header) != header_bytes:
        raise FetchError(f"truncated safetensors header for {name}")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise FetchError(f"invalid safetensors header JSON for {name}: {exc}") from exc
    if not isinstance(header, dict) or not header:
        raise FetchError(f"empty safetensors header for {name}")

    header_name = f"{name}.header.json"
    atomic_write(output / header_name, raw_header)
    digest = sha256_bytes(raw_header)
    payload_bytes = file_bytes - 8 - header_bytes
    if payload_bytes <= 0:
        raise FetchError(f"invalid safetensors payload size for {name}")
    logger.write(
        "header-json-done",
        file=name,
        header_bytes=header_bytes,
        payload_bytes=payload_bytes,
        tensors=len(header) - (1 if "__metadata__" in header else 0),
        sha256=digest,
    )
    return header, {
        "name": name,
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "sha256": lfs.get("sha256"),
        "header_file": header_name,
        "header_sha256": digest,
    }


def analyze_mtp_index(index: dict[str, Any]) -> dict[str, Any]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise FetchError("MTP source index has no weight_map")
    names = sorted(name for name in weight_map if isinstance(name, str) and name.startswith("mtp."))
    if not names:
        raise FetchError("MTP source index has no mtp.* tensors")
    tensor_shards: dict[str, str] = {}
    for name in names:
        shard = weight_map[name]
        if not isinstance(shard, str) or not shard:
            raise FetchError("MTP source index contains an invalid shard name")
        tensor_shards[name] = shard
    return {
        "tensor_names": names,
        "tensor_count": len(names),
        "tensor_shards": tensor_shards,
        "shards": sorted(set(tensor_shards.values())),
    }


def mtp_payload_from_headers(
    analysis: dict[str, Any], headers: dict[str, dict[str, Any]]
) -> int:
    total = 0
    tensor_shards = analysis["tensor_shards"]
    for name, expected_shard in tensor_shards.items():
        if expected_shard not in headers or name not in headers[expected_shard]:
            raise FetchError(f"MTP tensor header coverage mismatch: {name}")
        duplicate_shards = [
            shard
            for shard, header in headers.items()
            if shard != expected_shard and name in header
        ]
        if duplicate_shards:
            raise FetchError(f"MTP tensor appears in multiple shard headers: {name}")
        entry = headers[expected_shard][name]
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise FetchError(f"invalid MTP tensor offsets: {name}")
        total += offsets[1] - offsets[0]
    return total


def run_profile(profile_name: str, root: Path) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    repository = profile["repository"]
    revision = profile["revision"]
    output = root / profile["directory"]
    output.mkdir(parents=True, exist_ok=True)
    logger = Logger(output / "fetch.log")
    logger.write("fetch-start", profile=profile_name, repository=repository, revision=revision)

    api_data, _ = request_bytes(api_url(repository, revision), max_bytes=4 * 1024 * 1024)
    manifest = parse_api_manifest(api_data, repository, revision)
    files = sibling_map(manifest)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(output / "repo-api.json", manifest_bytes)

    metadata_files: dict[str, dict[str, Any]] = {
        "repo-api.json": {
            "bytes": len(manifest_bytes),
            "sha256": sha256_bytes(manifest_bytes),
        }
    }
    for name in profile["metadata"]:
        if name not in files:
            raise FetchError(f"pinned repository is missing metadata file: {name}")
        metadata_files[name] = fetch_metadata_file(
            repository, revision, name, files[name], output, logger
        )

    state: dict[str, Any] = {
        "format": profile["state_format"],
        "profile": profile_name,
        "repository": repository,
        "revision": revision,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "metadata_files": metadata_files,
    }

    weight_name = profile.get("weight")
    if isinstance(weight_name, str):
        if weight_name not in files:
            raise FetchError(f"pinned repository is missing weight file: {weight_name}")
        if files[weight_name].get("size") != profile["weight_bytes"]:
            raise FetchError(f"unexpected weight size for {profile_name}")
        lfs = files[weight_name].get("lfs")
        if not isinstance(lfs, dict) or lfs.get("sha256") != profile["weight_sha256"]:
            raise FetchError(f"unexpected weight hash for {profile_name}")
        _, weight = fetch_safetensors_header(
            repository, revision, weight_name, files[weight_name], output, logger
        )
        state["weight"] = weight
        header_name = weight["header_file"]
        header_path = output / header_name
        metadata_files[header_name] = {
            "bytes": header_path.stat().st_size,
            "sha256": weight["header_sha256"],
        }

    if profile_name == "mtp-source":
        index_path = output / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FetchError(f"cannot read MTP source index: {exc}") from exc
        analysis = analyze_mtp_index(index)
        shard_headers: dict[str, dict[str, Any]] = {}
        shard_state = []
        for shard in analysis["shards"]:
            if shard not in files:
                raise FetchError(f"MTP source API is missing shard: {shard}")
            header, weight = fetch_safetensors_header(
                repository, revision, shard, files[shard], output, logger
            )
            shard_headers[shard] = header
            shard_state.append(weight)
            header_name = weight["header_file"]
            header_path = output / header_name
            metadata_files[header_name] = {
                "bytes": header_path.stat().st_size,
                "sha256": weight["header_sha256"],
            }
        state["mtp"] = {
            "tensor_count": analysis["tensor_count"],
            "payload_bytes": mtp_payload_from_headers(analysis, shard_headers),
            "shards": shard_state,
        }

    state_bytes = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(output / "source-state.json", state_bytes)
    logger.write("fetch-done", profile=profile_name, state=output / "source-state.json")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=[*PROFILES, "all"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = list(PROFILES) if args.profile == "all" else [args.profile]
    try:
        for profile in profiles:
            run_profile(profile, args.root)
    except (FetchError, OSError) as exc:
        print(f"ornith35 metadata fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

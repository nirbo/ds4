#!/usr/bin/env python3
"""Refresh the small immutable shard-identity snapshot for Nemotron BF16."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemotron_bf16_source import EXPECTED_REPOSITORY, EXPECTED_REVISION, STATE_FORMAT
from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


def merge_remote_shards(state: dict, index: dict, remote_files: list[dict]) -> dict:
    require(state.get("format") == STATE_FORMAT, "unsupported BF16 metadata state")
    expected = sorted(set(index.get("weight_map", {}).values()))
    by_name = {}
    for row in remote_files:
        name = row.get("name")
        require(isinstance(name, str) and name not in by_name, "remote BF16 file name is invalid")
        by_name[name] = row
    require(set(expected) <= set(by_name), "remote BF16 metadata is missing indexed shards")
    shards = {}
    for name in expected:
        row = by_name[name]
        size = row.get("bytes")
        digest = row.get("sha256")
        blob_id = row.get("blob_id")
        require(isinstance(size, int) and size > 0, f"remote BF16 size is invalid: {name}")
        require(isinstance(digest, str) and len(digest) == 64, f"remote BF16 SHA-256 is invalid: {name}")
        require(isinstance(blob_id, str) and len(blob_id) == 40, f"remote BF16 blob ID is invalid: {name}")
        shards[name] = {"bytes": size, "sha256": digest, "blob_id": blob_id}
    result = dict(state)
    result["shards"] = shards
    result["indexed_shard_file_bytes"] = sum(entry["bytes"] for entry in shards.values())
    result["remote_snapshot"] = {
        "repository": EXPECTED_REPOSITORY,
        "revision": EXPECTED_REVISION,
        "files_metadata": True,
        "indexed_shards": len(shards),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise MetadataError("huggingface_hub is required only for metadata refresh") from exc
        state = load_json(args.state)
        require(state.get("repository") == EXPECTED_REPOSITORY, "unexpected BF16 metadata repository")
        require(state.get("revision") == EXPECTED_REVISION, "unexpected BF16 metadata revision")
        index_path = args.metadata_dir / "model.safetensors.index.json"
        require(
            sha256_file(index_path) == state.get("files", {}).get(index_path.name, {}).get("sha256"),
            "BF16 index no longer matches metadata state",
        )
        index = load_json(index_path)
        info = HfApi().model_info(
            EXPECTED_REPOSITORY,
            revision=EXPECTED_REVISION,
            files_metadata=True,
        )
        remote = []
        for sibling in info.siblings:
            lfs = getattr(sibling, "lfs", None)
            if sibling.rfilename.endswith(".safetensors"):
                remote.append(
                    {
                        "name": sibling.rfilename,
                        "bytes": sibling.size,
                        "sha256": None if lfs is None else lfs.sha256,
                        "blob_id": sibling.blob_id,
                    }
                )
        snapshot = merge_remote_shards(state, index, remote)
        atomic_json(args.state, snapshot)
        print(
            f"nemotron BF16 snapshot: shards={len(snapshot['shards'])} "
            f"file_gib={snapshot['indexed_shard_file_bytes'] / 2**30:.6f} "
            f"state={args.state} sha256={sha256_file(args.state)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError) as exc:
        print(f"nemotron BF16 snapshot error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

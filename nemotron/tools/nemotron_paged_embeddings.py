#!/usr/bin/env python3
"""Exact mmap-backed BF16 embedding rows for resident Nemotron."""

from __future__ import annotations

import hashlib
import json
import mmap
import sys
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import sha256_file
from nemotron_safetensors_inventory import read_safetensors_header


EMBEDDING_NAME = "backbone.embeddings.weight"
CATALOG_NAME = "nemotron_paged_embedding_catalog.json"
CATALOG_FORMAT = "nemotron-paged-embedding-v1"


def embedding_layout(model_dir: Path) -> tuple[Path, int, tuple[int, int], int]:
    index = load_json(model_dir / "model.safetensors.index.json")
    shard_name = index.get("weight_map", {}).get(EMBEDDING_NAME)
    require(
        isinstance(shard_name, str) and Path(shard_name).name == shard_name,
        "packed runtime has no valid embedding shard",
    )
    shard = model_dir / shard_name
    tensors, header_size, _ = read_safetensors_header(shard)
    entry = tensors.get(EMBEDDING_NAME)
    require(isinstance(entry, dict), "embedding tensor is absent from packed shard")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    require(
        entry.get("dtype") == "BF16"
        and isinstance(shape, list)
        and len(shape) == 2
        and all(isinstance(value, int) and value > 0 for value in shape)
        and isinstance(offsets, list)
        and len(offsets) == 2,
        "packed embedding layout is invalid",
    )
    tensor_bytes = shape[0] * shape[1] * 2
    require(offsets[1] - offsets[0] == tensor_bytes, "packed embedding byte size mismatch")
    payload_offset = 8 + header_size + offsets[0]
    return shard, payload_offset, (shape[0], shape[1]), tensor_bytes


def payload_sha256(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            chunk = handle.read(min(8 * 2**20, remaining))
            require(chunk, "truncated embedding payload while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def build_catalog(model_dir: Path) -> dict:
    shard, offset, shape, tensor_bytes = embedding_layout(model_dir)
    report_path = model_dir / "nemotron_mlx_pack_report.json"
    report = load_json(report_path)
    require(
        report.get("format") == "nemotron-mlx-runtime-v1"
        and report.get("status") == "complete"
        and isinstance(report.get("source_revision"), str),
        "packed runtime report is invalid",
    )
    return {
        "format": CATALOG_FORMAT,
        "source_revision": report["source_revision"],
        "index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
        "pack_report_sha256": sha256_file(report_path),
        "shard": shard.name,
        "shard_bytes": shard.stat().st_size,
        "tensor": EMBEDDING_NAME,
        "dtype": "BF16",
        "shape": list(shape),
        "payload_offset": offset,
        "payload_bytes": tensor_bytes,
        "payload_sha256": payload_sha256(shard, offset, tensor_bytes),
    }


def write_catalog(model_dir: Path) -> Path:
    catalog = build_catalog(model_dir)
    path = model_dir / CATALOG_NAME
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def validate_catalog(model_dir: Path, verify_payload: bool = True) -> dict:
    path = model_dir / CATALOG_NAME
    catalog = load_json(path)
    shard, offset, shape, tensor_bytes = embedding_layout(model_dir)
    report_path = model_dir / "nemotron_mlx_pack_report.json"
    report = load_json(report_path)
    require(
        catalog.get("format") == CATALOG_FORMAT
        and catalog.get("source_revision") == report.get("source_revision")
        and catalog.get("index_sha256")
        == sha256_file(model_dir / "model.safetensors.index.json")
        and catalog.get("pack_report_sha256") == sha256_file(report_path)
        and catalog.get("shard") == shard.name
        and catalog.get("shard_bytes") == shard.stat().st_size
        and catalog.get("tensor") == EMBEDDING_NAME
        and catalog.get("dtype") == "BF16"
        and catalog.get("shape") == list(shape)
        and catalog.get("payload_offset") == offset
        and catalog.get("payload_bytes") == tensor_bytes,
        "paged embedding catalog does not match the packed runtime",
    )
    if verify_payload:
        require(
            catalog.get("payload_sha256")
            == payload_sha256(shard, offset, tensor_bytes),
            "paged embedding payload hash mismatch",
        )
    return catalog


class PagedBF16Embedding:
    """Copy exact BF16 rows from mmap while keeping the full table off Metal."""

    dtype = mx.bfloat16

    def __init__(
        self,
        model_dir: Path,
        cache_rows: int = 256,
        verify_payload: bool = True,
    ):
        require(cache_rows >= 0, "embedding row cache cannot be negative")
        validate_catalog(model_dir, verify_payload)
        self.path, self.payload_offset, self.shape, self.nbytes = embedding_layout(model_dir)
        self.row_bytes = self.shape[1] * 2
        self.cache_rows = cache_rows
        self._file = self.path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._cache: OrderedDict[int, mx.array] = OrderedDict()
        self.lookups = 0
        self.cache_hits = 0
        self.staging_seconds = 0.0

    def close(self) -> None:
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.clear()
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except BufferError:
            pass

    def _row(self, token_id: int) -> mx.array:
        require(isinstance(token_id, int) and 0 <= token_id < self.shape[0], "invalid embedding token ID")
        self.lookups += 1
        row = self._cache.pop(token_id, None)
        if row is not None:
            self.cache_hits += 1
            self._cache[token_id] = row
            return row
        offset = self.payload_offset + token_id * self.row_bytes
        view = np.frombuffer(
            self._map,
            dtype="<u2",
            count=self.shape[1],
            offset=offset,
        )
        row = mx.array(view).view(mx.bfloat16)
        del view
        if self.cache_rows:
            self._cache[token_id] = row
            while len(self._cache) > self.cache_rows:
                self._cache.popitem(last=False)
        return row

    def rows(self, token_ids: list[int] | tuple[int, ...]) -> mx.array:
        require(token_ids, "embedding lookup requires at least one token")
        started = time.perf_counter()
        rows = [self._row(token_id) for token_id in token_ids]
        result = rows[0][None] if len(rows) == 1 else mx.stack(rows)
        self.staging_seconds += time.perf_counter() - started
        return result

    def __getitem__(self, key: int | mx.array) -> mx.array:
        if isinstance(key, int):
            return self.rows([key])[0]
        require(isinstance(key, mx.array) and key.ndim <= 1, "invalid embedding row selector")
        return self.rows([int(token_id) for token_id in key.tolist()])


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} MODEL_DIR", file=sys.stderr)
        return 2
    try:
        path = write_catalog(Path(sys.argv[1]))
        print(f"paged-embedding-catalog path={path} sha256={sha256_file(path)}")
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron paged embedding error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

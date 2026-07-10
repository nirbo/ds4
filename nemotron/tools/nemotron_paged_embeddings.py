#!/usr/bin/env python3
"""Exact mmap-backed BF16 embedding rows for resident Nemotron."""

from __future__ import annotations

import mmap
import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import load_json, require
from nemotron_safetensors_inventory import read_safetensors_header


EMBEDDING_NAME = "backbone.embeddings.weight"


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


class PagedBF16Embedding:
    """Copy exact BF16 rows from mmap while keeping the full table off Metal."""

    dtype = mx.bfloat16

    def __init__(self, model_dir: Path, cache_rows: int = 256):
        require(cache_rows >= 0, "embedding row cache cannot be negative")
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
        self._cache.clear()
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

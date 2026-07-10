#!/usr/bin/env python3
"""Tests for exact mmap-backed Nemotron embedding rows."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_paged_embeddings import EMBEDDING_NAME, PagedBF16Embedding  # noqa: E402


class PagedEmbeddingTest(unittest.TestCase):
    def test_rows_preserve_bf16_bits_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "global.safetensors"
            values = np.array(
                [
                    [0x3F80, 0x4000, 0x4040, 0x4080],
                    [0xBF80, 0x0000, 0x3F00, 0x4100],
                    [0x3E80, 0x3FC0, 0x4020, 0x40A0],
                ],
                dtype=np.uint16,
            )
            header = {
                EMBEDDING_NAME: {
                    "dtype": "BF16",
                    "shape": [3, 4],
                    "data_offsets": [0, values.nbytes],
                }
            }
            encoded = json.dumps(header, separators=(",", ":")).encode()
            shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + values.tobytes())
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {EMBEDDING_NAME: shard.name}})
            )

            embedding = PagedBF16Embedding(root, cache_rows=2)
            actual = embedding.rows([2, 0, 2])
            mx.eval(actual)
            self.assertEqual(actual.view(mx.uint16).tolist(), values[[2, 0, 2]].tolist())
            self.assertEqual(embedding.lookups, 3)
            self.assertEqual(embedding.cache_hits, 1)
            embedding.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

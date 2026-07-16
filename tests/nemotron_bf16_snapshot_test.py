#!/usr/bin/env python3
"""Tests for immutable Nemotron BF16 remote shard snapshots."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_bf16_snapshot import merge_remote_shards  # noqa: E402


class BF16SnapshotTest(unittest.TestCase):
    def test_snapshot_requires_every_indexed_shard(self) -> None:
        state = {"format": "nemotron-bf16-metadata-state-v1"}
        index = {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        }
        remote = [
            {
                "name": f"model-{ordinal:05d}-of-00002.safetensors",
                "bytes": ordinal * 100,
                "sha256": str(ordinal) * 64,
                "blob_id": str(ordinal) * 40,
            }
            for ordinal in (1, 2)
        ]
        result = merge_remote_shards(state, index, remote)
        self.assertEqual(result["indexed_shard_file_bytes"], 300)
        self.assertEqual(result["remote_snapshot"]["indexed_shards"], 2)
        with self.assertRaisesRegex(Exception, "missing indexed shards"):
            merge_remote_shards(state, index, remote[:1])


if __name__ == "__main__":
    unittest.main(verbosity=2)

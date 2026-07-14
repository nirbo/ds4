#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_mtp_recursive_features import (  # noqa: E402
    FORMAT,
    validate_feature_shard,
)
from nemotron_prune_materialize import sha256_file  # noqa: E402


class RecursiveFeaturesTest(unittest.TestCase):
    def test_validates_exact_feature_shape_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.safetensors"
            mx.save_safetensors(
                str(path),
                {
                    "mtp_hidden": mx.zeros((2, 4096), dtype=mx.bfloat16),
                    "mtp_token_ids": mx.array([1, 2], dtype=mx.int32),
                },
                metadata={"format": FORMAT, "prompt_index": "3"},
            )
            entry = {
                "rows": 2,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            validate_feature_shard(path, entry, 3)


if __name__ == "__main__":
    unittest.main()

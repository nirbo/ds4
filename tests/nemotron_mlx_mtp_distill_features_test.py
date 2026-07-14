#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_mtp_distill_features import (  # noqa: E402
    FORMAT,
    recursive_teacher_rows,
    validate_feature_shard,
)
from nemotron_prune_materialize import sha256_file  # noqa: E402


class FakeMTP:
    draft_token_ids = None

    def draft_step(self, hidden, token):
        next_hidden = hidden.astype(mx.float32) + token
        logits = mx.array([0.0, float(token), float(token + 1)])
        return logits, next_hidden, mx.zeros((0,)), mx.zeros((0,))

    def argmax_token(self, logits):
        return int(mx.argmax(logits))


class MTPDistillFeaturesTest(unittest.TestCase):
    def test_recursion_uses_each_teacher_proposal_and_hidden(self):
        arrays = recursive_teacher_rows(
            FakeMTP(), mx.zeros((2, 4096)), [1, 2], max_depth=3, top_k=2
        )
        self.assertEqual(arrays["teacher_hidden"].shape, (2, 3, 4096))
        self.assertEqual(arrays["teacher_token_ids"].tolist(), [[2, 2, 2], [2, 2, 2]])
        self.assertEqual(arrays["teacher_hidden"][0, :, 0].tolist(), [1.0, 3.0, 5.0])

    def test_validates_exact_feature_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.safetensors"
            arrays = {
                "teacher_hidden": mx.zeros((2, 3, 4096), dtype=mx.float32),
                "teacher_token_ids": mx.zeros((2, 3), dtype=mx.int32),
                "teacher_top_indices": mx.zeros((2, 3, 4), dtype=mx.int32),
                "teacher_top_logits": mx.zeros((2, 3, 4), dtype=mx.float32),
            }
            mx.save_safetensors(
                str(path),
                arrays,
                metadata={
                    "format": FORMAT,
                    "prompt_index": "7",
                    "max_depth": "3",
                    "top_k": "4",
                },
            )
            entry = {
                "rows": 2,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            validate_feature_shard(path, entry, 7, 3, 4)


if __name__ == "__main__":
    unittest.main()

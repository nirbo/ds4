#!/usr/bin/env python3
"""Tests for Nemotron MLX calibration batching and aggregation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_calibrate import build_batches, corpus_samples, merge_routing  # noqa: E402


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class MLXCalibrateTest(unittest.TestCase):
    def test_corpus_interleaves_categories_and_caps_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.json"
            path.write_text(json.dumps({"code": ["abcdef", "xy"], "reasoning": ["123456"]}))
            samples = corpus_samples(path)
            self.assertEqual([category for category, _ in samples], ["code", "reasoning", "code"])
            batches = build_batches(FakeTokenizer(), samples, batch_tokens=4, max_sample_tokens=4)
            self.assertEqual([batch["category"] for batch in batches], ["code", "reasoning", "code"])
            self.assertEqual(batches[0]["token_ids"], [97, 98, 99, 100])

    def test_merge_accumulates_route_weighted_norm(self) -> None:
        layers = {
            "1": {
                "counts": [0, 0],
                "score_sum": [0.0, 0.0],
                "weighted_output_norm_sum": [0.0, 0.0],
                "output_norm_sum": [0.0, 0.0],
                "max_score": [0.0, 0.0],
                "max_output_norm": [0.0, 0.0],
            }
        }
        merge_routing(
            layers,
            {1: {"indices": [1, 1, 0], "scores": [0.2, 0.3, 0.5], "output_norms": [2.0, 4.0, 3.0]}},
        )
        self.assertEqual(layers["1"]["counts"], [1, 2])
        self.assertAlmostEqual(layers["1"]["weighted_output_norm_sum"][1], 1.6)
        self.assertEqual(layers["1"]["max_output_norm"], [3.0, 4.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

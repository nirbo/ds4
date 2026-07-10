#!/usr/bin/env python3
"""Tests for guarded activation-aware Nemotron expert planning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_prune_plan import build_plan  # noqa: E402
from nemotron_metadata import MetadataError  # noqa: E402


def calibration(coverage: float = 1.0):
    counts = [1, 2, 0, 4, 5]
    if coverage == 1.0:
        counts[2] = 3
    layer = {
        "counts": counts,
        "score_sum": [1.0, 2.0, 0.0, 4.0, 5.0],
        "weighted_output_norm_sum": [1.0, 2.0, 0.0, 4.0, 5.0],
        "output_norm_sum": [1.0, 2.0, 0.0, 4.0, 5.0],
        "max_score": [1.0, 2.0, 0.0, 4.0, 5.0],
        "max_output_norm": [1.0, 2.0, 0.0, 4.0, 5.0],
    }
    return {
        "format": "nemotron-mlx-calibration-v1",
        "source_revision": "revision",
        "corpus_sha256": "corpus",
        "total_tokens": 10,
        "coverage": {"coverage": coverage, "min_observed": sum(value > 0 for value in counts)},
        "layers": {"1": layer},
    }


class MLXPrunePlanTest(unittest.TestCase):
    def test_unobserved_expert_is_protected(self) -> None:
        plan = build_plan(calibration(coverage=0.95), 0.20)
        self.assertIn(2, plan["kept_by_layer"]["1"])
        self.assertEqual(plan["dropped_by_layer"]["1"], [0])

    def test_rejects_insufficient_coverage(self) -> None:
        with self.assertRaises(MetadataError):
            build_plan(calibration(coverage=0.80), 0.20)


if __name__ == "__main__":
    unittest.main(verbosity=2)

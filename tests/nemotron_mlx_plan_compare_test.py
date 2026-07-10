#!/usr/bin/env python3
"""Tests for plan-comparison aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_plan_compare import summarize  # noqa: E402


class PlanCompareTest(unittest.TestCase):
    def test_summary_counts_top1_and_aggregates(self) -> None:
        metric = {
            "baseline_top1": 1,
            "candidate_top1": 1,
            "top_k_overlap": 4,
            "relative_l2": 0.1,
            "centered_relative_l2": 0.2,
            "kl_baseline_candidate": 0.3,
            "cosine": 0.9,
        }
        result = summarize([{"candidate": metric}], "candidate")
        self.assertEqual(result["top1_matches"], 1)
        self.assertEqual(result["mean_top_k_overlap"], 4.0)
        self.assertEqual(result["mean_kl_baseline_candidate"], 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

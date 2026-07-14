#!/usr/bin/env python3
"""Tests for routed-expert top-k sensitivity helpers."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_topk_sweep import aggregate, parse_top_ks  # noqa: E402


class MLXTopKSweepTest(unittest.TestCase):
    def test_parses_ordered_unique_values(self) -> None:
        self.assertEqual(parse_top_ks("20,18,16"), [20, 18, 16])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_top_ks("16,16")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_top_ks("0,16")

    def test_aggregates_quality_metrics(self) -> None:
        summary = aggregate(
            [
                {
                    "baseline_top1": 1,
                    "candidate_top1": 1,
                    "kl_baseline_candidate": 0.1,
                    "centered_relative_l2": 0.2,
                    "candidate_rank_of_baseline_top1": 1,
                    "top_k_overlap": 60,
                },
                {
                    "baseline_top1": 2,
                    "candidate_top1": 3,
                    "kl_baseline_candidate": 0.3,
                    "centered_relative_l2": 0.4,
                    "candidate_rank_of_baseline_top1": 4,
                    "top_k_overlap": 50,
                },
            ]
        )
        self.assertEqual(summary["positions"], 2)
        self.assertEqual(summary["top1_equal"], 1)
        self.assertAlmostEqual(summary["mean_kl"], 0.2)
        self.assertEqual(summary["worst_baseline_top1_rank"], 4)
        self.assertEqual(summary["mean_top64_overlap"], 55)


if __name__ == "__main__":
    unittest.main(verbosity=2)

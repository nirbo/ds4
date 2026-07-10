#!/usr/bin/env python3
"""Tests for Nemotron full-logit comparison metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_compare_logits import compare  # noqa: E402


class MLXCompareLogitsTest(unittest.TestCase):
    def test_identical_logits(self) -> None:
        values = np.array([1.0, 3.0, -2.0, 0.5])
        report = compare(values, values.copy(), 2)
        self.assertEqual(report["relative_l2"], 0.0)
        self.assertEqual(report["kl_baseline_candidate"], 0.0)
        self.assertEqual(report["top_k_overlap"], 2)

    def test_constant_shift_is_centered_identity(self) -> None:
        baseline = np.array([1.0, 3.0, -2.0, 0.5])
        report = compare(baseline, baseline + 7.0, 4)
        self.assertLess(report["centered_relative_l2"], 1e-15)
        self.assertLess(abs(report["kl_baseline_candidate"]), 1e-15)
        self.assertEqual(report["baseline_top1"], report["candidate_top1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

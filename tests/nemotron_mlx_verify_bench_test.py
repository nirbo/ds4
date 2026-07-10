#!/usr/bin/env python3
"""Tests for Nemotron resident verification benchmarking helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_verify_bench import compare_logits, parse_block_sizes  # noqa: E402


class MLXVerifyBenchTest(unittest.TestCase):
    def test_block_sizes_are_sorted_and_unique(self) -> None:
        self.assertEqual(parse_block_sizes("8,2,4,2"), [2, 4, 8])

    def test_logit_comparison_detects_top1_identity(self) -> None:
        reference = mx.array([[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]], dtype=mx.float32)
        actual = reference + mx.array([[0.01, 0.0, -0.01], [0.0, 0.01, -0.01]])
        result = compare_logits(actual, reference)
        self.assertTrue(result["top1_equal"])
        self.assertGreater(result["relative_l2"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

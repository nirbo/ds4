#!/usr/bin/env python3
"""Tests for conservative NVFP4 target-head certificate accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_head_certificate import evaluate_certificates, parse_counts  # noqa: E402


class HeadCertificateTest(unittest.TestCase):
    def test_certificate_requires_exact_winner_above_every_bound(self) -> None:
        quant = np.array([[4.0, 3.0, 2.0], [4.0, 3.9, 1.0]], dtype=np.float32)
        exact = np.array([[4.1, 3.1, 2.0], [3.8, 4.2, 1.0]], dtype=np.float32)
        bounds = np.array([[0.05, 0.05, 0.05], [0.5, 0.5, 0.1]], dtype=np.float32)
        result = evaluate_certificates(quant, exact, bounds, [1, 2], 0.0)
        self.assertEqual(result["1"]["recalled"], 1)
        self.assertEqual(result["1"]["certified"], 1)
        self.assertEqual(result["2"]["recalled"], 2)
        self.assertEqual(result["2"]["certified"], 2)

    def test_candidate_count_parser_is_sorted_and_unique(self) -> None:
        self.assertEqual(parse_counts("8,2,8,4"), [2, 4, 8])


if __name__ == "__main__":
    unittest.main(verbosity=2)

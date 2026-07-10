#!/usr/bin/env python3
"""Tests for proxy-expert comparison metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_proxy_compare import (  # noqa: E402
    error_metrics,
    geometric_mean,
    parse_layers,
    unique_mapping_metrics,
)


class ProxyCompareTest(unittest.TestCase):
    def test_error_metrics(self) -> None:
        baseline = np.array([1.0, 2.0, -1.0], dtype=np.float32)
        metrics = error_metrics(baseline.copy(), baseline)
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertEqual(metrics["max_abs"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)

    def test_helpers(self) -> None:
        self.assertAlmostEqual(geometric_mean([1.0, 4.0]), 2.0)
        self.assertEqual(parse_layers("1,34,87"), [1, 34, 87])

    def test_unique_mapping_metrics(self) -> None:
        indices = np.array([[[1, 1, 2, 3], [4, 4, 4, 5]]], dtype=np.uint32)
        metrics = unique_mapping_metrics(indices)
        self.assertEqual(metrics["mean_unique_prototypes"], 2.5)
        self.assertEqual(metrics["min_unique_prototypes"], 2)
        self.assertEqual(metrics["max_unique_prototypes"], 3)
        self.assertEqual(metrics["mean_duplicate_fraction"], 0.375)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Focused tests for multi-sample Router KD gradient rebalancing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_router_kd_rebalance import aggregate_layer, global_gradient_norm


class RouterKdRebalanceTest(unittest.TestCase):
    def test_global_norm_combines_layer_norms(self) -> None:
        report = {"gradient_rows": {"1": {"norm": 3.0}, "3": {"norm": 4.0}}}
        self.assertEqual(global_gradient_norm(report), 5.0)

    def test_sample_global_unit_equalizes_sample_scale(self) -> None:
        gradients = [
            np.array([[10.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 2.0]], dtype=np.float32),
        ]
        result = aggregate_layer(gradients, "sample-global-unit", [10.0, 2.0])
        np.testing.assert_allclose(result, [[0.5, 0.5]])

    def test_layer_unit_equalizes_each_layer(self) -> None:
        gradients = [
            np.array([[3.0, 4.0]], dtype=np.float32),
            np.array([[0.0, 7.0]], dtype=np.float32),
        ]
        result = aggregate_layer(gradients, "layer-unit", [1.0, 1.0])
        np.testing.assert_allclose(result, [[0.3, 0.9]], rtol=1e-6)

    def test_minimum_row_support_freezes_single_sample_rows(self) -> None:
        gradients = [
            np.array([[2.0, 1.0], [3.0, 0.0]], dtype=np.float32),
            np.array([[4.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        ]
        result = aggregate_layer(gradients, "mean", [1.0, 1.0], min_row_support=2)
        np.testing.assert_allclose(result, [[3.0, 1.0], [0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()

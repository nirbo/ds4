#!/usr/bin/env python3
"""Tests for nested binary/native backbone precision planning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_plan import (  # noqa: E402
    aggregate_binary_residual,
    causal_error,
    nested_assignments,
    rank_binary_residuals,
)


class BackbonePlanTest(unittest.TestCase):
    def test_ranking_forces_regressions_native(self) -> None:
        state = {
            "format": "nemotron-backbone-lowbit-fit-state-v1",
            "skipped": [{"expert": 3}],
            "completed": [
                {
                    "expert": 0,
                    "validation_routes": 5,
                    "validation_weighted_residual_error2": 2.0,
                    "metrics": {"validation": {"initial_error2": 10.0, "fitted_error2": 8.0}},
                },
                {
                    "expert": 1,
                    "validation_routes": 7,
                    "validation_weighted_residual_error2": 4.0,
                    "metrics": {"validation": {"initial_error2": 10.0, "fitted_error2": 7.0}},
                },
                {
                    "expert": 2,
                    "validation_routes": 9,
                    "validation_weighted_residual_error2": 6.0,
                    "metrics": {"validation": {"initial_error2": 10.0, "fitted_error2": 11.0}},
                },
            ],
        }
        ranking, forced = rank_binary_residuals(state)
        self.assertEqual([row["expert"] for row in ranking], [1, 0])
        self.assertEqual(forced, [2, 3])
        plans = nested_assignments(4, ranking, forced, [2, 3, 4])
        self.assertEqual(plans["2"]["native_nvfp4_experts"], [2, 3])
        self.assertEqual(plans["3"]["native_nvfp4_experts"], [1, 2, 3])
        self.assertEqual(plans["4"]["binary_experts"], [])

    def test_sparse_residuals_add_on_shared_rows(self) -> None:
        evidence = {
            0: (
                np.array([0, 2], dtype=np.int32),
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ),
            1: (
                np.array([1, 2], dtype=np.int32),
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            ),
        }
        result = aggregate_binary_residual(evidence, [0, 1], 3, 2)
        np.testing.assert_array_equal(
            result,
            np.array([[1.0, 2.0], [5.0, 6.0], [10.0, 12.0]], dtype=np.float32),
        )

    def test_causal_error_reports_routed_and_full_denominators(self) -> None:
        class Identity:
            def __call__(self, value):
                return value

        residual = np.array([[3.0, 4.0]], dtype=np.float32)
        metrics = causal_error(
            residual,
            Identity(),
            {
                "full_layer_output2": 100.0,
                "routed_output2": 25.0,
                "routed_latent2": 25.0,
            },
        )
        self.assertAlmostEqual(metrics["full_layer_relative_l2"], 0.5)
        self.assertAlmostEqual(metrics["routed_relative_l2"], 1.0)
        self.assertAlmostEqual(metrics["routed_latent_relative_l2"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

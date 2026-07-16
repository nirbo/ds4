#!/usr/bin/env python3
"""Tests for mixed affine-tier backbone screening."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_tier_screen import (  # noqa: E402
    PROJECTION_COST_QUANTUM,
    aggregate_tier_residual,
    expanded_projected_gib,
    independent_tier_plans,
    projection_option_catalog,
    projected_model_bytes,
)


class BackboneTierScreenTest(unittest.TestCase):
    def test_sparse_tier_residuals_follow_assignment(self) -> None:
        rows = [
            np.array([0, 2], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
        ]
        residuals = {
            "q1": [
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            ],
            "q2": [
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            ],
        }
        result = aggregate_tier_residual(
            rows,
            residuals,
            ["q2", "q1"],
            3,
            2,
        )
        np.testing.assert_allclose(
            result,
            np.array([[0.1, 0.2], [5.0, 6.0], [7.3, 8.4]], dtype=np.float32),
        )

    def test_independent_knapsack_selects_best_expert_tiers(self) -> None:
        losses = np.array(
            [
                [10.0, 2.0, 0.0],
                [5.0, 4.0, 0.0],
            ],
            dtype=np.float64,
        )
        payloads = np.array([1, 2, 3], dtype=np.int64)
        plans = independent_tier_plans(losses, payloads, [2, 3, 4, 6], cost_quantum=1)
        self.assertEqual(plans[0]["assignment_indices"], [0, 0])
        self.assertEqual(plans[1]["assignment_indices"], [1, 0])
        self.assertEqual(plans[2]["assignment_indices"], [2, 0])
        self.assertEqual(plans[3]["assignment_indices"], [2, 2])
        self.assertEqual([plan["layer_payload_bytes"] for plan in plans], [2, 3, 4, 6])

    def test_rounded_native_cost_is_repaired_to_exact_budget(self) -> None:
        losses = np.array([[4.0, 1.0, 0.0], [4.0, 1.0, 0.0]], dtype=np.float64)
        payloads = np.array([10, 20, 31], dtype=np.int64)
        plan = independent_tier_plans(
            losses,
            payloads,
            [61],
            cost_quantum=10,
        )[0]
        self.assertLessEqual(plan["layer_payload_bytes"], 61)

    def test_knapsack_supports_nonordered_duplicate_projection_costs(self) -> None:
        losses = np.array(
            [
                [8.0, 1.0, 2.0],
                [8.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
        payloads = np.array([10, 20, 20], dtype=np.int64)
        plan = independent_tier_plans(
            losses,
            payloads,
            [40],
            cost_quantum=1,
            labels=("q1/q1", "q1/native", "native/q1"),
        )[0]
        self.assertEqual(plan["assignment_indices"], [1, 2])
        self.assertEqual(
            plan["tier_counts"],
            {"q1/q1": 0, "q1/native": 1, "native/q1": 1},
        )

    def test_dense_projected_range_includes_both_endpoints(self) -> None:
        self.assertEqual(
            expanded_projected_gib(None, [40.0, 41.0, 0.25]),
            [40.0, 40.25, 40.5, 40.75, 41.0],
        )

    def test_projection_catalog_splits_each_payload_exactly(self) -> None:
        labels, payloads = projection_option_catalog(
            np.array([10, 20, 30, 40, 50], dtype=np.int64)
        )
        self.assertEqual(labels[5], "q1/native_nvfp4")
        self.assertEqual(labels[9], "native_nvfp4/q1")
        self.assertEqual(payloads[5], 30)
        self.assertEqual(payloads[9], 30)

    def test_nemotron_projection_costs_fit_half_tier_quantum(self) -> None:
        _, payloads = projection_option_catalog(
            np.array([860160, 1548288, 2236416, 2924544, 3096584], dtype=np.int64)
        )
        units = np.rint(payloads / PROJECTION_COST_QUANTUM).astype(np.int64)
        np.testing.assert_array_less(
            np.abs(payloads - units * PROJECTION_COST_QUANTUM),
            np.full(payloads.shape, 9, dtype=np.int64),
        )

    def test_full_projection_keeps_fixed_payload_once(self) -> None:
        self.assertEqual(projected_model_bytes(100, 25, 4), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)

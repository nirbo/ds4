#!/usr/bin/env python3
"""Tests for fixed-budget specialist expert protection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_protected_plan import build_protected_plan, protected_layer  # noqa: E402


def observation(values: list[float], counts: list[int] | None = None) -> dict:
    counts = counts or [1] * len(values)
    return {
        "counts": counts,
        "score_sum": values,
        "weighted_output_norm_sum": values,
        "output_norm_sum": values,
        "max_score": values,
        "max_output_norm": values,
    }


class ProtectedPlanTest(unittest.TestCase):
    def test_swaps_specialist_without_changing_size_or_base_core(self) -> None:
        kept, report = protected_layer(
            [2, 3, 4, 5, 6, 7],
            observation([0, 1, 2, 3, 4, 5, 6, 7]),
            observation([7, 6, 0, 1, 2, 3, 4, 5], counts=[3] * 8),
            protection_fraction=0.5,
            base_core_fraction=0.5,
            max_swap_fraction=0.2,
            minimum_events=2,
            specialist_weight=0.5,
        )
        self.assertEqual(len(kept), 6)
        self.assertIn(1, kept)
        self.assertNotIn(2, kept)
        self.assertTrue({5, 6, 7} <= set(kept))
        self.assertEqual(report["swaps"], 1)

    def test_unobserved_baseline_expert_cannot_be_evicted(self) -> None:
        kept, _ = protected_layer(
            [0, 2, 3, 4],
            observation([0, 1, 2, 3, 4], counts=[0, 1, 1, 1, 1]),
            observation([0, 10, 1, 2, 3], counts=[3] * 5),
            protection_fraction=0.5,
            base_core_fraction=0.25,
            max_swap_fraction=0.5,
            minimum_events=2,
            specialist_weight=0.5,
        )
        self.assertIn(0, kept)

    def test_full_layer_remains_unchanged(self) -> None:
        kept, report = protected_layer(
            [0, 1, 2],
            observation([0, 1, 2]),
            observation([2, 1, 0]),
            0.5,
            0.5,
            0.5,
            1,
            0.5,
        )
        self.assertEqual(kept, [0, 1, 2])
        self.assertEqual(report["swaps"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

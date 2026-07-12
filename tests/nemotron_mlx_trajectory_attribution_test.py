#!/usr/bin/env python3
"""Tests for long-trajectory pruning-attribution helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_trajectory_attribution import (  # noqa: E402
    rank_removed_experts,
    sampled_positions,
)


class TrajectoryAttributionTest(unittest.TestCase):
    def test_positions_cover_first_and_last_generated_state(self) -> None:
        self.assertEqual(sampled_positions(4, 10, 3, 10), [3, 6, 9])
        positions = sampled_positions(4, 100, 3, 4)
        self.assertEqual(positions[0], 3)
        self.assertEqual(positions[-1], 99)
        self.assertLessEqual(len(positions), 4)

    def test_removed_experts_rank_by_route_weighted_output(self) -> None:
        indices = np.array([[[0, 1], [2, 1]]], dtype=np.int32)
        scores = np.array([[[0.5, 0.5], [0.25, 0.75]]], dtype=np.float32)
        norms = np.array([[[2.0, 1.0], [8.0, 1.0]]], dtype=np.float32)
        ranking, importance = rank_removed_experts(indices, scores, norms, [0])
        self.assertEqual(ranking, [2, 1])
        self.assertAlmostEqual(importance[2], 2.0)
        self.assertAlmostEqual(importance[1], 1.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for reasoning-trajectory expert addback planning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_trajectory_plan import allocate_addbacks  # noqa: E402


class TrajectoryPlanTest(unittest.TestCase):
    def test_allocation_honors_floor_cap_and_global_priority(self) -> None:
        template = {
            "old_num_experts": 5,
            "model_moe_layers": [1, 3],
            "kept_by_layer": {"1": [0, 1], "3": [0, 1]},
        }
        aggregate = {
            1: {
                2: {"score": 0.9, "trajectories": 2},
                3: {"score": 0.8, "trajectories": 2},
                4: {"score": 0.1, "trajectories": 1},
            },
            3: {
                2: {"score": 0.7, "trajectories": 2},
                3: {"score": 0.6, "trajectories": 2},
                4: {"score": 0.5, "trajectories": 2},
            },
        }
        result = allocate_addbacks(template, aggregate, 4, 1, 2)
        self.assertEqual(result, {1: [2, 3], 3: [2, 3]})


if __name__ == "__main__":
    unittest.main(verbosity=2)

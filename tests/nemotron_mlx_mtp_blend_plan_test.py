#!/usr/bin/env python3
"""Focused tests for fixed-size MTP expert-plan blending."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_mtp_blend_plan import blend_experts, normalized_score_mass  # noqa: E402


class MLXMTPBlendPlanTest(unittest.TestCase):
    def test_normalizes_scored_mass(self) -> None:
        scores = normalized_score_mass(
            {
                "format": "nemotron-mtp-acceptance-v1",
                "scored_expert_score_mass": {"1": 1.0, "2": 3.0},
            }
        )
        self.assertEqual(scores, {1: 0.25, 2: 0.75})

    def test_blend_preserves_budget_and_replaces_weakest_joint_expert(self) -> None:
        experts, removed, added = blend_experts(
            [0, 1, 2],
            {0: 0.5, 1: 0.3, 2: 0.2, 3: 0.0},
            {0: 0.1, 1: 0.2, 2: 0.1, 3: 0.6},
            swaps=1,
            adaptation_weight=0.5,
        )
        self.assertEqual(removed, [2])
        self.assertEqual(added, [3])
        self.assertEqual(experts, [0, 1, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)

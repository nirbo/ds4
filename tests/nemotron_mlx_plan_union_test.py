#!/usr/bin/env python3
"""Tests for provenance-bound prune-plan union pools."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_plan_union import build_union  # noqa: E402


class PlanUnionTest(unittest.TestCase):
    def test_unions_layers_and_rebuilds_maps(self) -> None:
        template = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 6,
            "model_moe_layers": [1],
            "kept_by_layer": {"1": [0, 2, 4]},
            "target_total_experts": 3,
        }
        additions = {**template, "kept_by_layer": {"1": [1, 2, 5]}}
        result = build_union(template, additions, "template", "additions")
        self.assertEqual(result["kept_by_layer"]["1"], [0, 1, 2, 4, 5])
        self.assertEqual(result["dropped_by_layer"]["1"], [3])
        self.assertEqual(result["old_to_new_by_layer"]["1"]["4"], 3)
        self.assertEqual(result["union_candidate_pool"]["added_candidates"], 2)

    def test_can_limit_pool_to_recorded_trajectory_additions(self) -> None:
        template = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 6,
            "model_moe_layers": [1],
            "kept_by_layer": {"1": [0, 2, 4]},
            "target_total_experts": 3,
        }
        additions = {
            **template,
            "kept_by_layer": {"1": [1, 2, 3, 5]},
            "trajectory_swap": {"by_layer": {"1": {"added": [5]}}},
        }
        result = build_union(
            template,
            additions,
            "template",
            "additions",
            trajectory_additions_only=True,
        )
        self.assertEqual(result["kept_by_layer"]["1"], [0, 2, 4, 5])
        self.assertEqual(result["union_candidate_pool"]["added_candidates"], 1)
        self.assertEqual(
            result["union_candidate_pool"]["candidate_source"],
            "trajectory_swap.by_layer.added",
        )

    def test_can_rank_and_bound_trajectory_candidates(self) -> None:
        template = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 6,
            "model_moe_layers": [1],
            "kept_by_layer": {"1": [0, 2, 4]},
            "target_total_experts": 3,
        }
        additions = {
            **template,
            "trajectory_swap": {
                "by_layer": {
                    "1": {
                        "added": [1, 2, 3, 5],
                        "added_joint_score": {"1": 0.2, "2": 1.0, "3": 0.9, "5": 0.5},
                    }
                }
            },
        }
        result = build_union(
            template,
            additions,
            "template",
            "additions",
            trajectory_additions_only=True,
            max_trajectory_additions=2,
        )
        self.assertEqual(result["kept_by_layer"]["1"], [0, 2, 3, 4, 5])
        self.assertEqual(result["union_candidate_pool"]["candidate_limit"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

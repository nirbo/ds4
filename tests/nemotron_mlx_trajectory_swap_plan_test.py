#!/usr/bin/env python3
"""Tests for fixed-budget success-trajectory expert swapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_trajectory_swap_plan import build_swap_plan, tied_rank_fraction  # noqa: E402


def observation(counts: list[int]) -> dict:
    values = list(range(len(counts)))
    return {
        "counts": counts,
        "score_sum": values,
        "weighted_output_norm_sum": values,
        "output_norm_sum": values,
        "max_score": values,
        "max_output_norm": values,
    }


def report(selected: dict[int, float], template_hash: str, task: str) -> dict:
    return {
        "format": "nemotron-trajectory-attribution-v1",
        "status": "complete",
        "plan_sha256": {"r25": template_hash},
        "capture": {"identity": {"trajectory": {"task_id": task, "repeat": 0}}},
        "layers": [
            {
                "layer": 1,
                "plans": {
                    "r25": {
                        "selected_expert_importance": {
                            str(expert): score for expert, score in selected.items()
                        },
                        "curves": [{"output": {"relative_l2": 0.1}}],
                    }
                },
            }
        ],
    }


class TrajectorySwapPlanTest(unittest.TestCase):
    def test_tied_ranks_are_equal(self) -> None:
        self.assertEqual(tied_rank_fraction([0.0, 0.0, 1.0]), [0.25, 0.25, 1.0])

    def test_preserves_size_addback_and_unobserved_expert(self) -> None:
        template = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 8,
            "model_moe_layers": [1],
            "kept_by_layer": {"1": [0, 2, 3, 4, 5, 6]},
            "target_total_experts": 6,
        }
        addback = {
            **template,
            "kept_by_layer": {"1": [0, 1, 2, 3, 4, 5, 6, 7]},
        }
        baseline = {
            "format": "nemotron-mlx-calibration-v1",
            "source_revision": "revision",
            "layers": {"1": observation([0, 1, 1, 1, 1, 1, 1, 1])},
        }
        recovery = report({1: 8.0, 2: 1.0, 3: 1.0}, "template", "recovery")
        guard = report({0: 8.0, 6: 4.0}, "template", "guard")
        plan = build_swap_plan(
            template,
            addback,
            baseline,
            [recovery],
            [guard],
            ["recovery-hash"],
            ["guard-hash"],
            "template",
            "addback",
            "baseline",
            base_core_fraction=0.20,
            trajectory_core_fraction=0.20,
        )
        kept = plan["kept_by_layer"]["1"]
        self.assertEqual(len(kept), 6)
        self.assertTrue({0, 1, 7} <= set(kept))
        self.assertEqual(plan["target_total_experts"], 6)
        self.assertEqual(plan["trajectory_swap"]["total_swaps"], 2)

    def test_guard_core_fraction_is_independent(self) -> None:
        template = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 8,
            "model_moe_layers": [1],
            "kept_by_layer": {"1": [0, 2, 3, 4, 5, 6]},
            "target_total_experts": 6,
        }
        addback = {**template, "kept_by_layer": {"1": list(range(8))}}
        baseline = {
            "format": "nemotron-mlx-calibration-v1",
            "source_revision": "revision",
            "layers": {"1": observation([1] * 8)},
        }
        recovery = report({2: 10.0}, "template", "recovery")
        guard = report({3: 10.0, 4: 9.0, 5: 8.0}, "template", "guard")
        plan = build_swap_plan(
            template,
            addback,
            baseline,
            [recovery],
            [guard],
            ["recovery-hash"],
            ["guard-hash"],
            "template",
            "addback",
            "baseline",
            base_core_fraction=0.0,
            trajectory_core_fraction=0.0,
            guard_core_fraction=0.5,
        )
        protected = set(plan["trajectory_swap"]["by_layer"]["1"]["trajectory_core_protected"])
        self.assertEqual(protected, {3, 4, 5})
        self.assertTrue(protected <= set(plan["kept_by_layer"]["1"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for guarded targeted nested-plan repair."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_targeted_repair import build_targeted_repair  # noqa: E402


def observation(values: list[float]) -> dict:
    return {
        "counts": [1] * len(values),
        "score_sum": values,
        "weighted_output_norm_sum": values,
        "max_output_norm": values,
    }


def report(task: str, scores: dict[str, float]) -> dict:
    return {
        "format": "nemotron-trajectory-attribution-v1",
        "status": "complete",
        "plan_sha256": {"current": "candidate"},
        "capture": {
            "identity": {
                "trajectory": {"trajectory_format": "mbpp", "task_id": task, "repeat": 0}
            }
        },
        "layers": [
            {
                "layer": 1,
                "plans": {"current": {"selected_expert_importance": scores}},
            }
        ],
    }


class TargetedRepairTest(unittest.TestCase):
    def test_swaps_recovery_expert_without_changing_size(self) -> None:
        parent = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 4,
            "model_moe_layers": [1],
            "target_total_experts": 4,
            "kept_by_layer": {"1": [0, 1, 2, 3]},
        }
        candidate = dict(parent)
        candidate["target_total_experts"] = 3
        candidate["kept_by_layer"] = {"1": [0, 1, 2]}
        candidate["nested_thinning"] = {"protected_by_layer": {"1": [0]}}
        calibration = {
            "format": "nemotron-mlx-calibration-v1",
            "source_revision": "revision",
            "layers": {"1": observation([4.0, 1.0, 2.0, 3.0])},
        }
        result = build_targeted_repair(
            parent,
            candidate,
            calibration,
            calibration,
            [report("recovery", {"2": 1.0, "3": 10.0})],
            [report("guard", {"0": 10.0, "2": 2.0, "3": 1.0})],
            ["recovery"],
            ["guard"],
            1,
            "parent",
            "candidate",
            "baseline",
            "specialist",
            guard_core_fraction=0.34,
        )
        self.assertEqual(result["kept_by_layer"]["1"], [0, 2, 3])
        self.assertEqual(result["target_total_experts"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

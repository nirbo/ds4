#!/usr/bin/env python3
"""Tests for protected nested expert thinning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_nested_thin import build_nested_plan, build_repaired_plan  # noqa: E402


def observation(values: list[float]) -> dict:
    return {
        "counts": [1] * len(values),
        "score_sum": values,
        "weighted_output_norm_sum": values,
        "max_output_norm": values,
    }


class NestedThinTest(unittest.TestCase):
    def test_preserves_protected_and_r0_experts(self) -> None:
        template = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 4,
            "model_moe_layers": [1, 3],
            "target_total_experts": 8,
            "kept_by_layer": {"1": [0, 1, 2, 3], "3": [0, 1, 2, 3]},
            "budget_label_by_layer": {"1": "r0", "3": "r25"},
            "trajectory_swap": {
                "by_layer": {
                    "1": {},
                    "3": {"base_core_protected": [0]},
                }
            },
        }
        calibration = {
            "format": "nemotron-mlx-calibration-v1",
            "source_revision": "revision",
            "layers": {
                "1": observation([1.0, 2.0, 3.0, 4.0]),
                "3": observation([4.0, 1.0, 2.0, 3.0]),
            },
        }
        result = build_nested_plan(
            template, calibration, calibration, 1, 2, "template", "broad", "specialist"
        )
        self.assertEqual(result["kept_by_layer"]["1"], [0, 1, 2, 3])
        self.assertEqual(result["kept_by_layer"]["3"], [0, 2, 3])
        self.assertEqual(result["target_total_experts"], 7)

    def test_trajectory_evidence_changes_thinning_choice(self) -> None:
        template = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 3,
            "model_moe_layers": [1],
            "target_total_experts": 3,
            "kept_by_layer": {"1": [0, 1, 2]},
            "budget_label_by_layer": {"1": "r25"},
            "trajectory_swap": {"by_layer": {"1": {}}},
        }
        calibration = {
            "format": "nemotron-mlx-calibration-v1",
            "source_revision": "revision",
            "layers": {"1": observation([1.0, 2.0, 3.0])},
        }
        trajectory = {
            "format": "nemotron-trajectory-attribution-v1",
            "status": "complete",
            "plan_sha256": {"base": "template"},
            "layers": [
                {
                    "layer": 1,
                    "plans": {
                        "base": {
                            "selected_expert_importance": {"0": 100.0, "1": 1.0, "2": 0.0}
                        }
                    },
                }
            ],
        }
        result = build_nested_plan(
            template,
            calibration,
            calibration,
            1,
            2,
            "template",
            "broad",
            "specialist",
            baseline_weight=0.1,
            specialist_weight=0.1,
            trajectory_reports=[trajectory],
            trajectory_hashes=["trajectory"],
            trajectory_weight=0.8,
        )
        self.assertEqual(result["kept_by_layer"]["1"], [0, 1])
        self.assertEqual(result["nested_thinning"]["trajectory_report_sha256"], ["trajectory"])

    def test_bounded_repair_preserves_layer_size(self) -> None:
        parent = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 4,
            "model_moe_layers": [1],
            "target_total_experts": 4,
            "average_retained_experts": 4.0,
            "kept_by_layer": {"1": [0, 1, 2, 3]},
        }
        candidate = dict(parent)
        candidate["target_total_experts"] = 3
        candidate["average_retained_experts"] = 3.0
        candidate["kept_by_layer"] = {"1": [0, 1, 2]}
        candidate["nested_thinning"] = {"protected_by_layer": {"1": [0]}}
        calibration = {
            "format": "nemotron-mlx-calibration-v1",
            "source_revision": "revision",
            "layers": {"1": observation([4.0, 1.0, 2.0, 3.0])},
        }
        trajectory = {
            "format": "nemotron-trajectory-attribution-v1",
            "status": "complete",
            "plan_sha256": {"base": "candidate"},
            "layers": [
                {
                    "layer": 1,
                    "plans": {
                        "base": {"selected_expert_importance": {"1": 0.0, "2": 1.0, "3": 9.0}}
                    },
                }
            ],
        }
        result = build_repaired_plan(
            parent,
            candidate,
            calibration,
            calibration,
            [trajectory],
            ["trajectory"],
            1,
            "parent",
            "candidate",
            "broad",
            "specialist",
        )
        self.assertEqual(result["kept_by_layer"]["1"], [0, 2, 3])
        self.assertEqual(result["target_total_experts"], 3)
        self.assertEqual(result["nested_repair"]["swaps"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

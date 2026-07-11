#!/usr/bin/env python3
"""Tests for expert/width hybrid plan selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_hybrid_plan import build_plan  # noqa: E402


class HybridPlanTest(unittest.TestCase):
    def test_selects_only_width_evidence_passing_mean_and_worst_gates(self) -> None:
        base = {
            "source_revision": "rev",
            "old_num_experts": 4,
            "model_moe_layers": [1, 3],
            "kept_by_layer": {"1": [0, 1, 2], "3": [0, 1, 2]},
        }
        common = {
            "source_blocks": 2,
            "kept_blocks": [[0], [0], [0], [0]],
        }
        report = {
            "format": "nemotron-width-prune-v1",
            "source_revision": "rev",
            "layer_results": {
                "1": {**common, "summary": {
                    "width_to_hard_output_mean_ratio": 0.8,
                    "width_output_max_relative_l2": 0.9,
                    "hard_output_max_relative_l2": 1.0,
                }},
                "3": {**common, "summary": {
                    "width_to_hard_output_mean_ratio": 0.7,
                    "width_output_max_relative_l2": 1.1,
                    "hard_output_max_relative_l2": 1.0,
                }},
            },
        }
        plan = build_plan(base, report, 0.98, 1.0)
        self.assertEqual(plan["width_layers"], [1])
        self.assertEqual(plan["layers"]["1"]["mode"], "width")
        self.assertEqual(plan["layers"]["3"]["mode"], "experts")
        self.assertEqual(plan["expert_equivalent_total"], 5.0)

    def test_explicit_allowlist_can_reject_other_passing_layers(self) -> None:
        base = {
            "source_revision": "rev",
            "old_num_experts": 4,
            "model_moe_layers": [1],
            "kept_by_layer": {"1": [0, 1, 2]},
        }
        report = {
            "format": "nemotron-width-prune-v1",
            "source_revision": "rev",
            "layer_results": {"1": {
                "source_blocks": 2,
                "kept_blocks": [[0], [0], [0], [0]],
                "summary": {
                    "width_to_hard_output_mean_ratio": 0.5,
                    "width_output_max_relative_l2": 0.5,
                    "hard_output_max_relative_l2": 1.0,
                },
            }},
        }
        plan = build_plan(base, report, 0.98, 1.0, set())
        self.assertEqual(plan["width_layers"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

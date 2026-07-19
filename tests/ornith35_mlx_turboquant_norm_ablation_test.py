#!/usr/bin/env python3
"""Focused policy and aggregation tests for TurboQuant norm ablation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ornith35" / "tools"))

import ornith35_mlx_turboquant_norm_ablation as ablation


class MLXTurboQuantNormAblationTest(unittest.TestCase):
    def test_policies_cover_each_attention_layer_once(self) -> None:
        policies = ablation.norm_policies()
        self.assertEqual(policies[0].name, "fp32-all")
        self.assertEqual(policies[0].bf16_layers, frozenset())
        self.assertEqual(policies[1].bf16_layers, frozenset(ablation.ATTENTION_LAYERS))
        exact = [policy for policy in policies if policy.name.startswith("exact-layer-")]
        self.assertEqual(
            tuple(next(iter(policy.exact_layers)) for policy in exact),
            ablation.ATTENTION_LAYERS,
        )
        singles = [policy for policy in policies if policy.name.startswith("bf16-layer-")]
        self.assertEqual(
            tuple(next(iter(policy.bf16_layers)) for policy in singles),
            ablation.ATTENTION_LAYERS,
        )
        selected = ablation.select_policies(["exact-layer-7", "fp32-all"])
        self.assertEqual(tuple(policy.name for policy in selected), ("exact-layer-7", "fp32-all"))
        with self.assertRaisesRegex(Exception, "must be unique"):
            ablation.select_policies(["fp32-all", "fp32-all"])

    def test_case_parser_and_weighted_aggregation(self) -> None:
        self.assertEqual(
            ablation.parse_case("fixture:seed-17"),
            ablation.AblationCase("fixture", "seed-17", 17),
        )
        aggregate = ablation.aggregate_policy_cases(
            [
                {
                    "summary": {
                        "steps": 2,
                        "top1": 1,
                        "top8_recall_mean": 0.5,
                        "kl_mean": 0.2,
                        "kl_max": 0.3,
                    },
                    "material_mismatches": 1,
                },
                {
                    "summary": {
                        "steps": 6,
                        "top1": 6,
                        "top8_recall_mean": 1.0,
                        "kl_mean": 0.0,
                        "kl_max": 0.1,
                    },
                    "material_mismatches": 0,
                },
            ]
        )
        self.assertEqual(aggregate["steps"], 8)
        self.assertEqual(aggregate["top1"], 7)
        self.assertEqual(aggregate["top8_recall_mean"], 0.875)
        self.assertAlmostEqual(aggregate["kl_mean"], 0.05)
        self.assertEqual(aggregate["kl_max"], 0.3)
        self.assertEqual(aggregate["material_mismatches"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for nonuniform layer-sensitivity helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_layer_sensitivity import parse_plan, summarize  # noqa: E402


class LayerSensitivityTest(unittest.TestCase):
    def test_parse_plan(self) -> None:
        self.assertEqual(parse_plan("r20=/tmp/plan.json"), ("r20", Path("/tmp/plan.json")))

    def test_summary_keeps_layer_and_category_sensitivity(self) -> None:
        metric = {"relative_l2": 0.25, "max_abs": 1.0, "cosine": 0.9}
        rows = [
            {
                "budget": "r20",
                "layer": 1,
                "category": "code",
                "update": metric,
                "routed": metric,
                "output": metric,
            },
            {
                "budget": "r20",
                "layer": 1,
                "category": "reasoning",
                "update": {**metric, "relative_l2": 0.5},
                "routed": metric,
                "output": {**metric, "relative_l2": 0.125},
            },
        ]
        result = summarize(rows, ["r20"], [1])["r20"]["1"]
        self.assertEqual(result["mean_update_relative_l2"], 0.375)
        self.assertEqual(result["max_update_relative_l2"], 0.5)
        self.assertEqual(result["mean_output_relative_l2"], 0.1875)
        self.assertEqual(set(result["categories"]), {"code", "reasoning"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

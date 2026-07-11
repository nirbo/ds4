#!/usr/bin/env python3
"""Tests for incremental hybrid runtime metadata."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_hybrid_materialize import transformed_config  # noqa: E402


class HybridMaterializeTest(unittest.TestCase):
    def test_config_preserves_runtime_format_and_records_width(self) -> None:
        source = {
            "n_routed_experts": 512,
            "nemotron_runtime": {
                "format": "nemotron-mlx-runtime-v1",
                "experts_by_layer": {"54": 308},
                "nonuniform_experts": True,
            },
        }
        result = transformed_config(source, 54, 2016, "plan-hash")
        self.assertEqual(result["nemotron_runtime"]["format"], "nemotron-mlx-runtime-v1")
        self.assertEqual(result["nemotron_runtime"]["experts_by_layer"]["54"], 512)
        self.assertEqual(result["nemotron_runtime"]["expert_width_by_layer"], {"54": 2016})
        self.assertEqual(result["nemotron_runtime"]["hybrid_plan_sha256"], "plan-hash")
        self.assertEqual(source["nemotron_runtime"]["experts_by_layer"]["54"], 308)


if __name__ == "__main__":
    unittest.main(verbosity=2)

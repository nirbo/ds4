#!/usr/bin/env python3
"""Tests for strict virtual-pruning plan validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_stream_forward import validate_virtual_plan  # noqa: E402


class StreamForwardTest(unittest.TestCase):
    def test_virtual_plan_validates_identity_and_layers(self) -> None:
        config = {
            "hybrid_override_pattern": "ME",
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
        }
        plan = {
            "source_revision": "revision",
            "old_num_experts": 4,
            "model_moe_layers": [1],
            "kept_by_layer": {"1": [0, 2, 3]},
        }
        self.assertEqual(validate_virtual_plan(plan, config, "revision"), plan["kept_by_layer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

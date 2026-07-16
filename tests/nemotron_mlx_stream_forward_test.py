#!/usr/bin/env python3
"""Tests for strict virtual-pruning plan validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_stream_forward import (  # noqa: E402
    validate_mixed_backbone_identity,
    validate_virtual_plan,
)


class StreamForwardTest(unittest.TestCase):
    def test_mixed_backbone_identity_binds_source_and_pack(self) -> None:
        config = {
            "hybrid_override_pattern": "ME",
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
        }
        metadata = {
            "format": "nemotron-backbone-lowbit-mixed-layer-v1",
            "layer": "1",
            "native_budget": "3",
            "native_source_revision": "revision",
            "contract_sha256": "contract",
            "fit_state_sha256": "fit",
            "plan_sha256": "plan",
            "fit_strategy": "native-target-rtn",
        }
        report = {
            "format": "nemotron-backbone-lowbit-pack-report-v1",
            "status": "complete",
            "layer": 1,
            "native_budget": 3,
            "native_source_revision": "revision",
            "file_sha256": "payload",
            "contract_sha256": "contract",
            "fit_state_sha256": "fit",
            "plan_sha256": "plan",
            "fit_strategy": "native-target-rtn",
        }
        self.assertEqual(
            validate_mixed_backbone_identity(metadata, report, config, "revision", "payload"),
            1,
        )
        report["plan_sha256"] = "wrong"
        with self.assertRaisesRegex(Exception, "plan_sha256"):
            validate_mixed_backbone_identity(metadata, report, config, "revision", "payload")

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

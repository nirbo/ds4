#!/usr/bin/env python3
"""Unit tests for physical mixed-backbone verification gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_verify import gate_metrics, parity_metrics  # noqa: E402


class BackboneVerifyTest(unittest.TestCase):
    def test_matching_physical_layer_passes(self) -> None:
        metrics = parity_metrics(
            full_reference2=100.0,
            virtual_error2=4.0,
            physical_error2=4.0004,
            physical_virtual_error2=1e-6,
            physical_max_abs=0.2,
            physical_virtual_max_abs=1e-3,
        )
        gate = gate_metrics(
            metrics,
            {"full_layer_reference2": 100.0, "error2": 4.0, "full_layer_relative_l2": 0.2},
            plan_relative_tolerance=1e-6,
            physical_plan_tolerance=5e-4,
            physical_virtual_tolerance=5e-4,
            physical_virtual_max_abs_tolerance=1e-2,
        )
        self.assertEqual(gate["result"], "passed")
        self.assertTrue(all(gate["checks"].values()))

    def test_kernel_drift_fails_independently_of_plan_replay(self) -> None:
        metrics = parity_metrics(
            full_reference2=100.0,
            virtual_error2=4.0,
            physical_error2=4.0,
            physical_virtual_error2=0.25,
            physical_max_abs=0.2,
            physical_virtual_max_abs=0.1,
        )
        gate = gate_metrics(
            metrics,
            {"full_layer_reference2": 100.0, "error2": 4.0, "full_layer_relative_l2": 0.2},
            plan_relative_tolerance=1e-6,
            physical_plan_tolerance=5e-4,
            physical_virtual_tolerance=5e-4,
            physical_virtual_max_abs_tolerance=1e-2,
        )
        self.assertEqual(gate["result"], "failed")
        self.assertFalse(gate["checks"]["physical_virtual_full_parity"])
        self.assertFalse(gate["checks"]["physical_virtual_max_abs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

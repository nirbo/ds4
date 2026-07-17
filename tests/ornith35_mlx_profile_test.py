#!/usr/bin/env python3
"""Unit checks for the Ornith-35 decode profiler's durable statistics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_profile as profile


def component(value: float) -> profile.ComponentProfile:
    return profile.ComponentProfile(
        embedding=value,
        layers=(
            profile.LayerTiming(
                index=7,
                kind="gdn",
                input_norm=value + 1.0,
                mixer=value + 2.0,
                post_norm=value + 3.0,
                moe=value + 4.0,
                residual=value + 5.0,
            ),
        ),
        final_norm=value + 6.0,
        lm_head=value + 7.0,
        logits=mx.array([value], dtype=mx.float32),
    )


class MLXProfileTest(unittest.TestCase):
    def test_component_profile_uses_per_boundary_medians(self) -> None:
        result = profile.median_profile([component(3.0), component(1.0), component(2.0)])
        self.assertEqual(result.embedding, 2.0)
        self.assertEqual(result.layers[0].mixer, 4.0)
        self.assertEqual(result.layers[0].moe, 6.0)
        self.assertEqual(result.final_norm, 8.0)
        self.assertEqual(result.lm_head, 9.0)
        self.assertEqual(result.logits.item(), 2.0)

    def test_optimized_hotpath_flags_default_on_and_can_be_disabled(self) -> None:
        with mock.patch.object(sys, "argv", ["profile"]):
            defaults = profile.parse_args()
        self.assertTrue(defaults.paired_moe_gate_up)
        self.assertTrue(defaults.fused_moe_shared_gate)
        self.assertTrue(defaults.fused_moe_routed_down)
        self.assertTrue(defaults.fused_residual_mean_square)
        self.assertTrue(defaults.fused_residual_rmsnorm)
        self.assertTrue(defaults.fused_gdn_convolution)
        self.assertTrue(defaults.fused_gdn_recurrence)
        self.assertTrue(defaults.fused_gdn_core_gate)
        self.assertTrue(defaults.fused_attention_qk_norm_rope)
        self.assertTrue(defaults.grouped_attention_gqa)
        self.assertFalse(defaults.quantized_embedding)
        self.assertFalse(defaults.quantized_lm_head)
        with mock.patch.object(
            sys,
            "argv",
            [
                "profile",
                "--no-fused-residual-mean-square",
                "--no-fused-residual-rmsnorm",
                "--no-fused-gdn-convolution",
                "--no-fused-gdn-recurrence",
                "--no-fused-gdn-core-gate",
                "--no-fused-attention-qk-norm-rope",
                "--no-grouped-attention-gqa",
                "--no-paired-moe-gate-up",
                "--no-fused-moe-shared-gate",
                "--no-fused-moe-routed-down",
            ],
        ):
            fallback = profile.parse_args()
        self.assertFalse(fallback.paired_moe_gate_up)
        self.assertFalse(fallback.fused_moe_shared_gate)
        self.assertFalse(fallback.fused_moe_routed_down)
        self.assertFalse(fallback.fused_residual_mean_square)
        self.assertFalse(fallback.fused_residual_rmsnorm)
        self.assertFalse(fallback.fused_gdn_convolution)
        self.assertFalse(fallback.fused_gdn_recurrence)
        self.assertFalse(fallback.fused_gdn_core_gate)
        self.assertFalse(fallback.fused_attention_qk_norm_rope)
        self.assertFalse(fallback.grouped_attention_gqa)


if __name__ == "__main__":
    unittest.main(verbosity=2)

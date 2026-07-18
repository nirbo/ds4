#!/usr/bin/env python3
"""Unit checks for Ornith-35 prefill-profile aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_prefill_profile as profile
import ornith35_mlx_attention as attention
import ornith35_mlx_model as model


def component(value: float) -> profile.ComponentProfile:
    return profile.ComponentProfile(
        embedding=value,
        rope=value + 1.0,
        layers=(
            profile.LayerTiming(
                index=18,
                kind="gdn",
                input_norm=value + 2.0,
                mixer=value + 3.0,
                post_norm=value + 4.0,
                moe=value + 5.0,
                residual=value + 6.0,
            ),
        ),
        final_kv=value + 7.0,
        state=None,  # type: ignore[arg-type]
    )


class MLXPrefillProfileTest(unittest.TestCase):
    def test_component_profile_uses_per_boundary_medians(self) -> None:
        result = profile.median_profile(
            [component(3.0), component(1.0), component(2.0)]
        )
        self.assertEqual(result.embedding, 2.0)
        self.assertEqual(result.rope, 3.0)
        self.assertEqual(result.layers[0].mixer, 5.0)
        self.assertEqual(result.layers[0].moe, 7.0)
        self.assertEqual(result.final_kv, 9.0)
        self.assertIsNone(result.state)

    def test_defaults_are_bounded_and_repeatable(self) -> None:
        with mock.patch.object(sys, "argv", ["prefill-profile"]):
            args = profile.parse_args()
        self.assertEqual(args.chunk, 128)
        self.assertEqual(args.prefix, 0)
        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.top_layers, 10)

    def test_synthetic_prefix_materializes_only_attention_history(self) -> None:
        initial_attention = attention.MLXAttentionState(
            keys=mx.zeros((2, 0, 4), dtype=mx.bfloat16),
            values=mx.zeros((2, 0, 4), dtype=mx.bfloat16),
        )
        initial = model.TextModelState(position=0, layers=(initial_attention,))
        config = SimpleNamespace(
            layer_types=(model.LAYER_ATTENTION,),
            attention=SimpleNamespace(num_kv_heads=2, head_dim=4),
        )

        result = profile.synthetic_prefix_state(initial, 7, config)  # type: ignore[arg-type]

        self.assertEqual(result.position, 7)
        self.assertEqual(result.layers[0].keys.shape, (2, 7, 4))
        self.assertEqual(result.layers[0].values.shape, (2, 7, 4))
        self.assertEqual(initial.layers[0].keys.shape, (2, 0, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)

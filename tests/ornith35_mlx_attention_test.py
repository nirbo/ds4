#!/usr/bin/env python3
"""Independent scalar parity checks for Ornith-35 MLX full attention."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_attention_reference as reference
import ornith35_mlx_attention as mlx_attention


def matrix(rows: int, columns: int, phase: float) -> list[list[float]]:
    return [
        [math.sin((row * columns + column + 1) * phase) * 0.13 for column in range(columns)]
        for row in range(rows)
    ]


def make_fixture() -> tuple[reference.AttentionConfig, reference.AttentionWeights]:
    config = reference.AttentionConfig(
        hidden_size=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        rotary_dim=2,
        rope_theta=10_000.0,
    )
    weights = reference.AttentionWeights(
        q_proj=matrix(config.query_dim * 2, config.hidden_size, 0.11),
        k_proj=matrix(config.kv_dim, config.hidden_size, 0.17),
        v_proj=matrix(config.kv_dim, config.hidden_size, 0.19),
        o_proj=matrix(config.hidden_size, config.query_dim, 0.23),
        q_norm=[0.1, -0.2, 0.05, 0.15],
        k_norm=[-0.1, 0.2, 0.07, -0.04],
    )
    return config, weights


def mlx_weights(weights: reference.AttentionWeights) -> mlx_attention.MLXAttentionWeights:
    return mlx_attention.MLXAttentionWeights(
        q_proj=mx.array(weights.q_proj, dtype=mx.float32),
        k_proj=mx.array(weights.k_proj, dtype=mx.float32),
        v_proj=mx.array(weights.v_proj, dtype=mx.float32),
        o_proj=mx.array(weights.o_proj, dtype=mx.float32),
        q_norm=mx.array(weights.q_norm, dtype=mx.float32),
        k_norm=mx.array(weights.k_norm, dtype=mx.float32),
    )


def flatten(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [value]


class MLXAttentionTest(unittest.TestCase):
    def test_production_contract(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        self.assertEqual(config.query_dim, 4096)
        self.assertEqual(config.kv_dim, 512)
        self.assertEqual(config.rotary_dim, 64)
        state = mlx_attention.zeros_state(config)
        self.assertEqual(state.keys.shape, (2, 0, 256))
        self.assertEqual(state.values.shape, (2, 0, 256))

    def test_multistep_scalar_parity_and_rollback(self) -> None:
        config, scalar_weights = make_fixture()
        gpu_weights = mlx_weights(scalar_weights)
        scalar_state = reference.zeros_state(config)
        gpu_state = mlx_attention.zeros_state(config, dtype=mx.float32)
        original_keys = gpu_state.keys
        original_values = gpu_state.values

        inputs = (
            [0.25, -0.5, 0.75, 0.1],
            [-0.2, 0.4, 0.3, -0.7],
            [0.9, 0.05, -0.6, 0.2],
        )
        for hidden in inputs:
            expected, scalar_state = reference.decode_step(
                hidden, scalar_state, scalar_weights, config
            )
            actual, gpu_state = mlx_attention.decode_step(
                mx.array(hidden, dtype=mx.float32), gpu_state, gpu_weights, config
            )
            mx.eval(actual, gpu_state.keys, gpu_state.values)
            for left, right in zip(actual.tolist(), expected):
                self.assertAlmostEqual(left, right, delta=2e-6)

        for left, right in zip(flatten(gpu_state.keys.tolist()), flatten(scalar_state.keys)):
            self.assertAlmostEqual(left, right, delta=2e-6)
        for left, right in zip(flatten(gpu_state.values.tolist()), flatten(scalar_state.values)):
            self.assertAlmostEqual(left, right, delta=2e-6)
        self.assertEqual(original_keys.shape, (1, 0, 4))
        self.assertEqual(original_values.shape, (1, 0, 4))

    def test_rope_keeps_high_position_angles_in_fp32(self) -> None:
        config = reference.AttentionConfig(
            hidden_size=4,
            num_q_heads=1,
            num_kv_heads=1,
            head_dim=8,
            rotary_dim=8,
            rope_theta=10_000_000.0,
        )
        values = [0.25, -0.5, 0.75, 0.1, -0.2, 0.4, 0.3, -0.7]
        expected = reference._apply_rope(values, 262_143, config)
        actual = mlx_attention._apply_text_rope(
            mx.array([values], dtype=mx.bfloat16),
            262_143,
            config,
            mx.bfloat16,
        )
        mx.eval(actual)
        for left, right in zip(actual[0].tolist(), expected):
            self.assertAlmostEqual(left, right, delta=8e-3)

    def test_rejects_non_attention_layer(self) -> None:
        with self.assertRaisesRegex(reference.AttentionError, "not an Ornith attention layer"):
            mlx_attention.load_layer(Path("unused.safetensors"), 2)

    def test_rejects_mixed_weight_dtypes(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        weights = mlx_attention.MLXAttentionWeights(
            **{**weights.__dict__, "q_norm": weights.q_norm.astype(mx.bfloat16)}
        )
        with self.assertRaisesRegex(reference.AttentionError, "weight dtype mismatch"):
            mlx_attention.validate_weights(weights, config)


if __name__ == "__main__":
    unittest.main(verbosity=2)

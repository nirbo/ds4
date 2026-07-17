#!/usr/bin/env python3
"""Full decoder-layer parity checks for Ornith-35 MLX composition."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_attention_reference as attention_reference
import ornith35_gdn_reference as gdn_reference
import ornith35_mlx_attention as mlx_attention
import ornith35_mlx_attention_test as attention_fixture
import ornith35_mlx_gdn as mlx_gdn
import ornith35_mlx_gdn_test as gdn_fixture
import ornith35_mlx_layer as layer
import ornith35_mlx_moe_test as moe_fixture
import ornith35_moe_reference as moe_reference


def scalar_rms(hidden, weight, eps=1e-6):
    inverse = 1.0 / math.sqrt(math.fsum(value * value for value in hidden) / len(hidden) + eps)
    return [value * inverse * (1.0 + scale) for value, scale in zip(hidden, weight)]


def make_gdn_fixture():
    config = gdn_reference.GDNConfig(
        hidden_size=16,
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=2,
        head_v_dim=2,
        conv_kernel_size=3,
    )
    weights = gdn_reference.GDNWeights(
        in_proj_qkv=gdn_fixture.matrix(config.conv_dim, config.hidden_size, 0.031),
        in_proj_z=gdn_fixture.matrix(config.value_dim, config.hidden_size, 0.037),
        in_proj_b=gdn_fixture.matrix(config.num_v_heads, config.hidden_size, 0.041),
        in_proj_a=gdn_fixture.matrix(config.num_v_heads, config.hidden_size, 0.043),
        conv1d=gdn_fixture.matrix(config.conv_dim, config.conv_kernel_size, 0.047),
        dt_bias=[-0.3, 0.2],
        a_log=[math.log(0.4), math.log(1.3)],
        norm=[0.8, 1.2],
        out_proj=gdn_fixture.matrix(config.hidden_size, config.value_dim, 0.053),
    )
    return config, weights


def make_attention_fixture():
    config = attention_reference.AttentionConfig(
        hidden_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        rotary_dim=2,
        rope_theta=10_000.0,
    )
    weights = attention_reference.AttentionWeights(
        q_proj=attention_fixture.matrix(config.query_dim * 2, config.hidden_size, 0.031),
        k_proj=attention_fixture.matrix(config.kv_dim, config.hidden_size, 0.037),
        v_proj=attention_fixture.matrix(config.kv_dim, config.hidden_size, 0.041),
        o_proj=attention_fixture.matrix(config.hidden_size, config.query_dim, 0.043),
        q_norm=[0.1, -0.2, 0.05, 0.15],
        k_norm=[-0.1, 0.2, 0.07, -0.04],
    )
    return config, weights


class MLXLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.moe_config, self.scalar_moe = moe_fixture.make_fixture()
        self.gpu_moe = moe_fixture.mlx_weights(self.scalar_moe)
        self.input_norm = [math.sin((index + 1) * 0.07) * 0.1 for index in range(16)]
        self.post_norm = [math.cos((index + 1) * 0.09) * 0.1 for index in range(16)]
        self.norms = layer.LayerNorms(
            input_layernorm=mx.array(self.input_norm, dtype=mx.float32),
            post_attention_layernorm=mx.array(self.post_norm, dtype=mx.float32),
        )

    def test_fused_production_residual_and_mean_match_materialized_graph(self) -> None:
        hidden = mx.array(
            [math.sin((index + 1) * 0.007) * 0.9 for index in range(2048)],
            dtype=mx.bfloat16,
        )
        delta = mx.array(
            [math.cos((index + 1) * 0.011) * 0.7 for index in range(2048)],
            dtype=mx.bfloat16,
        )
        weight = mx.array(
            [math.sin((index + 1) * 0.013) * 0.2 for index in range(2048)],
            dtype=mx.bfloat16,
        )
        expected_hidden = (hidden + delta).astype(mx.bfloat16)
        expected_mean = mx.mean(
            expected_hidden.astype(mx.float32) * expected_hidden.astype(mx.float32)
        ).reshape(1)
        expected_norm = layer.qwen_rms_norm(expected_hidden, weight)
        actual_hidden, actual_mean = layer.fused_residual_mean_square(hidden, delta)
        actual_norm = layer.qwen_rms_norm(
            actual_hidden,
            weight,
            mean_square=actual_mean,
        )
        fused_hidden, fused_norm = layer.fused_residual_rms_norm(
            hidden,
            delta,
            weight,
        )
        mx.eval(
            expected_hidden,
            expected_mean,
            expected_norm,
            actual_hidden,
            actual_mean,
            actual_norm,
            fused_hidden,
            fused_norm,
        )
        self.assertTrue(bool(mx.array_equal(actual_hidden, expected_hidden).item()))
        self.assertTrue(bool(mx.array_equal(actual_mean, expected_mean).item()))
        self.assertTrue(bool(mx.array_equal(actual_norm, expected_norm).item()))
        self.assertTrue(bool(mx.array_equal(fused_hidden, expected_hidden).item()))
        self.assertTrue(bool(mx.array_equal(fused_norm, expected_norm).item()))

    def scalar_layer(self, hidden, state, mixer_weights, mixer_config, mixer_forward):
        mixed_input = scalar_rms(hidden, self.input_norm)
        mixed, state = mixer_forward(mixed_input, state, mixer_weights, mixer_config)
        hidden = [left + right for left, right in zip(hidden, mixed)]
        moe_input = scalar_rms(hidden, self.post_norm)
        moe_result = moe_reference.forward(moe_input, self.scalar_moe, self.moe_config)
        output = [left + right for left, right in zip(hidden, moe_result.output)]
        return output, state, moe_result

    def test_gdn_decoder_layer_matches_scalar_composition(self) -> None:
        mixer_config, scalar_mixer = make_gdn_fixture()
        weights = layer.GDNLayerWeights(
            token_mixer=gdn_fixture.mlx_weights(scalar_mixer),
            moe=self.gpu_moe,
            norms=self.norms,
        )
        scalar_state = gdn_reference.zeros_state(mixer_config)
        gpu_state = mlx_gdn.zeros_state(mixer_config, conv_dtype=mx.float32)
        for step in range(2):
            hidden = [math.sin((index + 1) * (0.11 + step * 0.03)) * 0.4 for index in range(16)]
            expected, scalar_state, scalar_moe = self.scalar_layer(
                hidden,
                scalar_state,
                scalar_mixer,
                mixer_config,
                gdn_reference.decode_step,
            )
            actual = layer.forward_gdn(
                mx.array(hidden, dtype=mx.float32),
                gpu_state,
                weights,
                mixer_config,
                self.moe_config,
            )
            gpu_state = actual.state
            mx.eval(actual.output, gpu_state.conv, gpu_state.recurrent)
            self.assertEqual(tuple(actual.selected_experts.tolist()), scalar_moe.selected_experts)
            for left, right in zip(actual.output.tolist(), expected):
                self.assertAlmostEqual(left, right, delta=3e-5)

    def test_attention_decoder_layer_matches_scalar_composition(self) -> None:
        mixer_config, scalar_mixer = make_attention_fixture()
        weights = layer.AttentionLayerWeights(
            token_mixer=attention_fixture.mlx_weights(scalar_mixer),
            moe=self.gpu_moe,
            norms=self.norms,
        )
        scalar_state = attention_reference.zeros_state(mixer_config)
        gpu_state = mlx_attention.zeros_state(mixer_config, dtype=mx.float32)
        for step in range(2):
            hidden = [math.cos((index + 1) * (0.13 + step * 0.02)) * 0.35 for index in range(16)]
            expected, scalar_state, scalar_moe = self.scalar_layer(
                hidden,
                scalar_state,
                scalar_mixer,
                mixer_config,
                attention_reference.decode_step,
            )
            actual = layer.forward_attention(
                mx.array(hidden, dtype=mx.float32),
                gpu_state,
                weights,
                mixer_config,
                self.moe_config,
            )
            gpu_state = actual.state
            mx.eval(actual.output, gpu_state.keys, gpu_state.values)
            self.assertEqual(tuple(actual.selected_experts.tolist()), scalar_moe.selected_experts)
            for left, right in zip(actual.output.tolist(), expected):
                self.assertAlmostEqual(left, right, delta=3e-5)

    def test_loader_rejects_layer_outside_text_model(self) -> None:
        with self.assertRaisesRegex(moe_reference.MoEError, "outside the Ornith text model"):
            layer.load_layer(Path("unused.safetensors"), -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

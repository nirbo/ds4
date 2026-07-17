#!/usr/bin/env python3
"""Independent scalar parity checks for Ornith-35 MLX GatedDeltaNet."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_gdn_reference as reference
import ornith35_mlx_gdn as mlx_gdn


def matrix(rows: int, columns: int, phase: float) -> list[list[float]]:
    return [
        [math.sin((row * columns + column + 1) * phase) * 0.17 for column in range(columns)]
        for row in range(rows)
    ]


def make_fixture() -> tuple[reference.GDNConfig, reference.GDNWeights]:
    config = reference.GDNConfig(
        hidden_size=4,
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=2,
        head_v_dim=2,
        conv_kernel_size=3,
    )
    weights = reference.GDNWeights(
        in_proj_qkv=matrix(config.conv_dim, config.hidden_size, 0.11),
        in_proj_z=matrix(config.value_dim, config.hidden_size, 0.13),
        in_proj_b=matrix(config.num_v_heads, config.hidden_size, 0.17),
        in_proj_a=matrix(config.num_v_heads, config.hidden_size, 0.19),
        conv1d=matrix(config.conv_dim, config.conv_kernel_size, 0.23),
        dt_bias=[-0.3, 0.2],
        a_log=[math.log(0.4), math.log(1.3)],
        norm=[0.8, 1.2],
        out_proj=matrix(config.hidden_size, config.value_dim, 0.29),
    )
    return config, weights


def mlx_weights(weights: reference.GDNWeights) -> mlx_gdn.MLXGDNWeights:
    return mlx_gdn.MLXGDNWeights(
        in_proj_qkv=mx.array(weights.in_proj_qkv, dtype=mx.float32),
        in_proj_z=mx.array(weights.in_proj_z, dtype=mx.float32),
        in_proj_b=mx.array(weights.in_proj_b, dtype=mx.float32),
        in_proj_a=mx.array(weights.in_proj_a, dtype=mx.float32),
        conv1d=mx.array(weights.conv1d, dtype=mx.float32),
        dt_bias=mx.array(weights.dt_bias, dtype=mx.float32),
        a_log=mx.array(weights.a_log, dtype=mx.float32),
        norm=mx.array(weights.norm, dtype=mx.float32),
        out_proj=mx.array(weights.out_proj, dtype=mx.float32),
    )


def flatten(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [value]


class MLXGDNTest(unittest.TestCase):
    def test_fused_beta_decay_matches_mlx_formulas(self) -> None:
        b = mx.linspace(-12.0, 12.0, 32).astype(mx.bfloat16)
        a = mx.linspace(9.0, -9.0, 32).astype(mx.bfloat16)
        dt_bias = mx.linspace(-2.0, 1.0, 32).astype(mx.bfloat16)
        a_log = mx.linspace(-3.0, 2.0, 32).astype(mx.bfloat16)
        expected_beta = mx.sigmoid(b.astype(mx.float32))
        decay_log = -mx.exp(a_log.astype(mx.float32)) * mlx_gdn._softplus(
            a.astype(mx.float32) + dt_bias.astype(mx.float32)
        )
        expected_decay = mx.exp(decay_log)
        actual_beta, actual_decay = mlx_gdn.fused_beta_decay(
            b,
            a,
            dt_bias,
            a_log,
        )
        mx.eval(expected_beta, expected_decay, actual_beta, actual_decay)
        self.assertTrue(bool(mx.array_equal(actual_beta, expected_beta).item()))
        self.assertTrue(bool(mx.array_equal(actual_decay, expected_decay).item()))

    def test_fused_production_convolution_chunk_matches_token_steps(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        state = mx.array(
            [math.sin((index + 1) * 0.007) * 0.1 for index in range(config.conv_dim * 4)],
            dtype=mx.float32,
        ).reshape(config.conv_dim, 4).astype(mx.bfloat16)
        mixed = mx.array(
            [math.cos((index + 1) * 0.011) * 0.2 for index in range(3 * config.conv_dim)],
            dtype=mx.float32,
        ).reshape(3, config.conv_dim).astype(mx.bfloat16)
        weight = mx.array(
            [math.sin((index + 1) * 0.013) * 0.3 for index in range(config.conv_dim * 4)],
            dtype=mx.float32,
        ).reshape(config.conv_dim, 4).astype(mx.bfloat16)
        expected = []
        expected_state = state
        for token in mixed:
            expected_state, convolved = mlx_gdn.fused_conv_step(expected_state, token, weight)
            expected.append(convolved)
        expected_convolved = mx.stack(expected)
        actual_state, actual_convolved = mlx_gdn.fused_conv_chunk(state, mixed, weight)
        mx.eval(expected_state, expected_convolved, actual_state, actual_convolved)
        self.assertTrue(bool(mx.array_equal(actual_state, expected_state).item()))
        self.assertTrue(bool(mx.array_equal(actual_convolved, expected_convolved).item()))

    def test_fused_production_recurrence_chunk_matches_token_steps(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        mx.random.seed(13)
        recurrent = mx.random.uniform(
            -0.05,
            0.05,
            shape=(config.num_v_heads, config.head_k_dim, config.head_v_dim),
        ).astype(mx.float32)
        key = mx.random.uniform(-0.2, 0.2, shape=(3, 32, 128)).astype(mx.float32)
        query = mx.random.uniform(-0.02, 0.02, shape=(3, 32, 128)).astype(mx.float32)
        value = mx.random.uniform(-0.2, 0.2, shape=(3, 32, 128)).astype(mx.float32)
        beta = mx.random.uniform(0.1, 0.9, shape=(3, 32)).astype(mx.float32)
        decay = mx.random.uniform(0.8, 1.0, shape=(3, 32)).astype(mx.float32)
        z = mx.random.uniform(-0.5, 0.5, shape=(3, 32, 128)).astype(mx.bfloat16)
        norm = mx.random.uniform(0.7, 1.3, shape=(128,)).astype(mx.bfloat16)

        expected = []
        expected_recurrent = recurrent
        for token in range(3):
            expected_recurrent, gated = mlx_gdn.fused_recurrence_core_gate_step(
                expected_recurrent,
                key[token],
                query[token],
                value[token],
                beta[token],
                decay[token],
                z[token],
                norm,
            )
            expected.append(gated)
        expected_gated = mx.stack(expected)
        actual_recurrent, actual_gated = mlx_gdn.fused_recurrence_core_gate_chunk(
            recurrent,
            key,
            query,
            value,
            beta,
            decay,
            z,
            norm,
        )
        minimal_recurrent, minimal_gated = mlx_gdn.fused_recurrence_core_gate_chunk(
            recurrent,
            key,
            query,
            value,
            beta,
            decay,
            z,
            norm,
            simdgroups=8,
        )
        column_recurrent, column_gated = (
            mlx_gdn.fused_recurrence_core_gate_column_chunk(
                recurrent,
                key,
                query,
                value,
                beta,
                decay,
                z,
                norm,
            )
        )
        mx.eval(
            expected_recurrent,
            expected_gated,
            actual_recurrent,
            actual_gated,
            minimal_recurrent,
            minimal_gated,
            column_recurrent,
            column_gated,
        )
        self.assertTrue(bool(mx.array_equal(actual_recurrent, expected_recurrent).item()))
        self.assertTrue(bool(mx.array_equal(actual_gated, expected_gated).item()))
        self.assertTrue(bool(mx.array_equal(actual_recurrent, minimal_recurrent).item()))
        self.assertTrue(bool(mx.array_equal(actual_gated, minimal_gated).item()))
        self.assertTrue(bool(mx.array_equal(actual_recurrent, column_recurrent).item()))
        self.assertTrue(bool(mx.array_equal(actual_gated, column_gated).item()))

    def test_fused_production_convolution_matches_materialized_operations(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        conv_state = mx.array(
            [math.sin((index + 1) * 0.007) * 0.1 for index in range(config.conv_dim * 4)],
            dtype=mx.float32,
        ).reshape(config.conv_dim, 4).astype(mx.bfloat16)
        mixed = mx.array(
            [math.cos((index + 1) * 0.011) * 0.2 for index in range(config.conv_dim)],
            dtype=mx.bfloat16,
        )
        weight = mx.array(
            [math.sin((index + 1) * 0.013) * 0.3 for index in range(config.conv_dim * 4)],
            dtype=mx.float32,
        ).reshape(config.conv_dim, 4).astype(mx.bfloat16)
        expected_state = mx.concatenate([conv_state[:, 1:], mixed[:, None]], axis=1)
        expected_convolved = mx.sum(
            expected_state.astype(mx.float32) * weight.astype(mx.float32),
            axis=1,
        )
        actual_state, actual_convolved = mlx_gdn.fused_conv_step(conv_state, mixed, weight)
        mx.eval(expected_state, expected_convolved, actual_state, actual_convolved)
        self.assertTrue(bool(mx.array_equal(actual_state, expected_state).item()))
        self.assertTrue(bool(mx.array_equal(actual_convolved, expected_convolved).item()))

    def test_fused_production_qkv_convolution_silu_matches_split_path(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        mx.random.seed(20260717)
        hidden = mx.random.uniform(-0.2, 0.2, shape=(config.hidden_size,)).astype(
            mx.bfloat16
        )
        conv_state = mx.random.uniform(
            -0.1,
            0.1,
            shape=(config.conv_dim, config.conv_kernel_size),
        ).astype(mx.bfloat16)
        projection = mx.random.uniform(
            -0.03,
            0.03,
            shape=(config.conv_dim, config.hidden_size),
        ).astype(mx.bfloat16)
        conv_weight = mx.random.uniform(
            -0.2,
            0.2,
            shape=(config.conv_dim, config.conv_kernel_size),
        ).astype(mx.bfloat16)
        mixed = mx.matmul(projection, hidden)
        expected_state, convolved = mlx_gdn.fused_conv_step(
            conv_state,
            mixed,
            conv_weight,
        )
        expected = mlx_gdn._silu(convolved).astype(mx.bfloat16)
        actual_state, actual = mlx_gdn.fused_qkv_conv_silu_step(
            hidden,
            conv_state,
            projection,
            conv_weight,
        )
        mx.eval(expected_state, expected, actual_state, actual)
        self.assertTrue(bool(mx.array_equal(actual_state, expected_state).item()))
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_fused_qkv_transition_projections_match_separate_dispatches(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        mx.random.seed(20260720)
        hidden = mx.random.uniform(-0.2, 0.2, shape=(config.hidden_size,)).astype(
            mx.bfloat16
        )
        conv_state = mx.random.uniform(
            -0.1,
            0.1,
            shape=(config.conv_dim, config.conv_kernel_size),
        ).astype(mx.bfloat16)
        projection = mx.random.uniform(
            -0.03,
            0.03,
            shape=(config.conv_dim, config.hidden_size),
        ).astype(mx.bfloat16)
        conv_weight = mx.random.uniform(
            -0.2,
            0.2,
            shape=(config.conv_dim, config.conv_kernel_size),
        ).astype(mx.bfloat16)
        z_projection = mx.random.uniform(
            -0.03,
            0.03,
            shape=(config.value_dim, config.hidden_size),
        ).astype(mx.bfloat16)
        b_projection = mx.random.uniform(
            -0.03,
            0.03,
            shape=(config.num_v_heads, config.hidden_size),
        ).astype(mx.bfloat16)
        a_projection = mx.random.uniform(
            -0.03,
            0.03,
            shape=(config.num_v_heads, config.hidden_size),
        ).astype(mx.bfloat16)
        dt_bias = mx.random.uniform(-2.0, 2.0, shape=(32,)).astype(mx.bfloat16)
        a_log = mx.random.uniform(-3.0, 2.0, shape=(32,)).astype(mx.bfloat16)
        expected_state, expected_convolved = mlx_gdn.fused_qkv_conv_silu_step(
            hidden,
            conv_state,
            projection,
            conv_weight,
        )
        expected_z = mx.matmul(z_projection, hidden)
        expected_b = mx.matmul(b_projection, hidden)
        expected_a = mx.matmul(a_projection, hidden)
        expected_beta, expected_decay = mlx_gdn.fused_beta_decay(
            expected_b,
            expected_a,
            dt_bias,
            a_log,
        )
        actual_state, actual_convolved, actual_z, actual_beta, actual_decay = (
            mlx_gdn.fused_qkv_conv_silu_transition_step(
                hidden,
                conv_state,
                projection,
                conv_weight,
                z_projection,
                b_projection,
                a_projection,
                dt_bias,
                a_log,
            )
        )
        mx.eval(
            expected_state,
            expected_convolved,
            expected_z,
            expected_beta,
            expected_decay,
            actual_state,
            actual_convolved,
            actual_z,
            actual_beta,
            actual_decay,
        )
        self.assertTrue(bool(mx.array_equal(actual_state, expected_state).item()))
        self.assertTrue(bool(mx.array_equal(actual_convolved, expected_convolved).item()))
        self.assertTrue(bool(mx.array_equal(actual_z, expected_z).item()))
        self.assertTrue(bool(mx.array_equal(actual_beta, expected_beta).item()))
        self.assertTrue(bool(mx.array_equal(actual_decay, expected_decay).item()))

    def test_fused_production_recurrence_matches_materialized_operations(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        mx.random.seed(7)
        recurrent = mx.random.uniform(
            -0.05,
            0.05,
            shape=(config.num_v_heads, config.head_k_dim, config.head_v_dim),
        ).astype(mx.float32)
        key = mx.random.uniform(-0.2, 0.2, shape=(32, 128)).astype(mx.float32)
        query = mx.random.uniform(-0.02, 0.02, shape=(32, 128)).astype(mx.float32)
        value = mx.random.uniform(-0.2, 0.2, shape=(32, 128)).astype(mx.float32)
        beta = mx.random.uniform(0.1, 0.9, shape=(32,)).astype(mx.float32)
        decay = mx.random.uniform(0.8, 1.0, shape=(32,)).astype(mx.float32)

        decayed = recurrent * decay[:, None, None]
        memory = mx.sum(decayed * key[:, :, None], axis=1)
        delta = (value - memory) * beta[:, None]
        expected_recurrent = decayed + key[:, :, None] * delta[:, None, :]
        expected_core = mx.sum(expected_recurrent * query[:, :, None], axis=1)
        actual_recurrent, actual_core = mlx_gdn.fused_recurrence_step(
            recurrent,
            key,
            query,
            value,
            beta,
            decay,
        )
        z = mx.random.uniform(-0.5, 0.5, shape=(32, 128)).astype(mx.bfloat16)
        norm = mx.random.uniform(0.7, 1.3, shape=(128,)).astype(mx.bfloat16)
        variance = mx.mean(expected_core * expected_core, axis=-1, keepdims=True)
        normalized = expected_core * mx.rsqrt(variance + 1e-6)
        weighted = (
            normalized.astype(mx.bfloat16) * norm.astype(mx.bfloat16)
        ).astype(mx.bfloat16)
        expected_gated = (
            weighted.astype(mx.float32) * mlx_gdn._silu(z.astype(mx.float32))
        ).astype(mx.bfloat16)
        combined_recurrent, combined_gated = mlx_gdn.fused_recurrence_core_gate_step(
            recurrent,
            key,
            query,
            value,
            beta,
            decay,
            z,
            norm,
        )
        minimal_recurrent, minimal_core = mlx_gdn.fused_recurrence_step(
            recurrent,
            key,
            query,
            value,
            beta,
            decay,
            simdgroups=8,
        )
        minimal_combined_recurrent, minimal_combined_gated = (
            mlx_gdn.fused_recurrence_core_gate_step(
                recurrent,
                key,
                query,
                value,
                beta,
                decay,
                z,
                norm,
                simdgroups=8,
            )
        )
        mx.eval(
            expected_recurrent,
            expected_core,
            expected_gated,
            actual_recurrent,
            actual_core,
            combined_recurrent,
            combined_gated,
            minimal_recurrent,
            minimal_core,
            minimal_combined_recurrent,
            minimal_combined_gated,
        )
        self.assertTrue(bool(mx.array_equal(actual_recurrent, expected_recurrent).item()))
        self.assertTrue(bool(mx.array_equal(actual_core, expected_core).item()))
        self.assertTrue(bool(mx.array_equal(combined_recurrent, expected_recurrent).item()))
        self.assertTrue(bool(mx.array_equal(combined_gated, expected_gated).item()))
        self.assertTrue(bool(mx.array_equal(actual_recurrent, minimal_recurrent).item()))
        self.assertTrue(bool(mx.array_equal(actual_core, minimal_core).item()))
        self.assertTrue(
            bool(mx.array_equal(combined_recurrent, minimal_combined_recurrent).item())
        )
        self.assertTrue(bool(mx.array_equal(combined_gated, minimal_combined_gated).item()))

    def test_convolved_recurrence_input_fusion_matches_split_path(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        mx.random.seed(20260718)
        recurrent = mx.random.uniform(
            -0.05,
            0.05,
            shape=(config.num_v_heads, config.head_k_dim, config.head_v_dim),
        ).astype(mx.float32)
        convolved = mx.random.uniform(
            -0.3,
            0.3,
            shape=(config.conv_dim,),
        ).astype(mx.bfloat16)
        beta = mx.random.uniform(0.1, 0.9, shape=(32,)).astype(mx.float32)
        decay = mx.random.uniform(0.8, 1.0, shape=(32,)).astype(mx.float32)
        z = mx.random.uniform(-0.5, 0.5, shape=(32, 128)).astype(mx.bfloat16)
        norm = mx.random.uniform(0.7, 1.3, shape=(128,)).astype(mx.bfloat16)

        query = convolved[: config.key_dim].reshape(16, 128)
        key = convolved[config.key_dim : config.key_dim * 2].reshape(16, 128)
        value = convolved[config.key_dim * 2 :].reshape(32, 128)
        query = mx.repeat(mlx_gdn._l2norm(query), 2, axis=0)
        query = query * (config.head_k_dim**-0.5)
        key = mx.repeat(mlx_gdn._l2norm(key), 2, axis=0)
        expected_recurrent, expected_gated = (
            mlx_gdn.fused_recurrence_core_gate_step(
                recurrent,
                key,
                query,
                value.astype(mx.float32),
                beta,
                decay,
                z,
                norm,
            )
        )
        actual_recurrent, actual_gated = (
            mlx_gdn.fused_recurrence_convolved_core_gate_step(
                recurrent,
                convolved,
                beta,
                decay,
                z,
                norm,
            )
        )
        mx.eval(expected_recurrent, expected_gated, actual_recurrent, actual_gated)
        self.assertTrue(bool(mx.array_equal(actual_recurrent, expected_recurrent).item()))
        self.assertTrue(bool(mx.array_equal(actual_gated, expected_gated).item()))

    def test_production_contract(self) -> None:
        config = mlx_gdn.PRODUCTION_CONFIG
        self.assertEqual(config.conv_dim, 8192)
        self.assertEqual(config.value_dim, 4096)
        state = mlx_gdn.zeros_state(config)
        self.assertEqual(state.conv.shape, (8192, 4))
        self.assertEqual(state.recurrent.shape, (32, 128, 128))
        self.assertEqual(state.recurrent.dtype, mx.float32)

    def test_multistep_scalar_parity_and_rollback(self) -> None:
        config, scalar_weights = make_fixture()
        gpu_weights = mlx_weights(scalar_weights)
        scalar_state = reference.zeros_state(config)
        gpu_state = mlx_gdn.zeros_state(config, conv_dtype=mx.float32)
        original_conv = gpu_state.conv
        original_recurrent = gpu_state.recurrent

        inputs = (
            [0.25, -0.5, 0.75, 0.1],
            [-0.2, 0.4, 0.3, -0.7],
            [0.9, 0.05, -0.6, 0.2],
        )
        for hidden in inputs:
            expected, scalar_state = reference.decode_step(
                hidden, scalar_state, scalar_weights, config
            )
            actual, gpu_state = mlx_gdn.decode_step(
                mx.array(hidden, dtype=mx.float32), gpu_state, gpu_weights, config
            )
            mx.eval(actual, gpu_state.conv, gpu_state.recurrent)
            for left, right in zip(actual.tolist(), expected):
                self.assertAlmostEqual(left, right, delta=2e-6)

        for left, right in zip(flatten(gpu_state.conv.tolist()), flatten(scalar_state.conv)):
            self.assertAlmostEqual(left, right, delta=2e-6)
        for left, right in zip(
            flatten(gpu_state.recurrent.tolist()), flatten(scalar_state.recurrent)
        ):
            self.assertAlmostEqual(left, right, delta=2e-6)

        mx.eval(original_conv, original_recurrent)
        self.assertEqual(mx.max(mx.abs(original_conv)).item(), 0.0)
        self.assertEqual(mx.max(mx.abs(original_recurrent)).item(), 0.0)

    def test_generic_prefill_chunk_matches_token_steps(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        hidden = mx.array(
            (
                [0.25, -0.5, 0.75, 0.1],
                [-0.2, 0.4, 0.3, -0.7],
                [0.9, 0.05, -0.6, 0.2],
            ),
            dtype=mx.float32,
        )
        state = mlx_gdn.zeros_state(config, conv_dtype=mx.float32)
        expected = []
        expected_state = state
        for token in hidden:
            output, expected_state = mlx_gdn.decode_step(token, expected_state, weights, config)
            expected.append(output)
        expected_output = mx.stack(expected)
        actual_output, actual_state = mlx_gdn.prefill_chunk(hidden, state, weights, config)
        mx.eval(
            expected_output,
            expected_state.conv,
            expected_state.recurrent,
            actual_output,
            actual_state.conv,
            actual_state.recurrent,
        )
        self.assertTrue(bool(mx.array_equal(actual_output, expected_output).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.conv, expected_state.conv).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.recurrent, expected_state.recurrent).item()))

    def test_rejects_non_gdn_layer(self) -> None:
        with self.assertRaisesRegex(reference.GDNError, "not an Ornith GDN layer"):
            mlx_gdn.load_layer(Path("unused.safetensors"), 3)

    def test_rejects_mixed_weight_dtypes(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        weights = mlx_gdn.MLXGDNWeights(
            **{**weights.__dict__, "norm": weights.norm.astype(mx.bfloat16)}
        )
        with self.assertRaisesRegex(reference.GDNError, "weight dtype mismatch"):
            mlx_gdn.validate_weights(weights, config)


if __name__ == "__main__":
    unittest.main(verbosity=2)

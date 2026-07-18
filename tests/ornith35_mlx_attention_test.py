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
import ornith35_context as context
import ornith35_mlx_attention as mlx_attention
import ornith35_mlx_turboquant_cache as turboquant_cache


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


def bf16_weights(weights: reference.AttentionWeights) -> mlx_attention.MLXAttentionWeights:
    values = mlx_weights(weights)
    return mlx_attention.MLXAttentionWeights(
        **{
            name: array.astype(mx.bfloat16)
            for name, array in values.__dict__.items()
        }
    )


def flatten(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [value]


class MLXAttentionTest(unittest.TestCase):
    def test_production_turboquant_decode_matches_explicit_packed_composition(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        weights = mlx_attention.MLXAttentionWeights(
            q_proj=mx.full((8192, 2048), 2**-15, dtype=mx.bfloat16),
            k_proj=mx.full((512, 2048), 2**-14, dtype=mx.bfloat16),
            v_proj=mx.full((512, 2048), 2**-13, dtype=mx.bfloat16),
            o_proj=mx.full((2048, 4096), 2**-15, dtype=mx.bfloat16),
            q_norm=mx.zeros((256,), dtype=mx.bfloat16),
            k_norm=mx.zeros((256,), dtype=mx.bfloat16),
        )
        mx.random.seed(20260718)
        prefix_keys = mx.random.normal((2, 3, 256), dtype=mx.float32).astype(mx.bfloat16)
        prefix_values = mx.random.normal((2, 3, 256), dtype=mx.float32).astype(mx.bfloat16)
        hidden = mx.sin(mx.arange(2048, dtype=mx.float32) * 0.013).astype(mx.bfloat16)
        actual_state = turboquant_cache.linearize_bf16_kv(prefix_keys, prefix_values, 8)
        expected_state = turboquant_cache.linearize_bf16_kv(prefix_keys, prefix_values, 8)

        rope = mlx_attention.make_text_rope(3, 1, config, mx.bfloat16)
        query_gate = mx.matmul(weights.q_proj, hidden)
        key = mx.matmul(weights.k_proj, hidden)
        value = mx.matmul(weights.v_proj, hidden).reshape(2, 256)
        query, gate, key = mlx_attention.fused_qk_norm_rope_step(
            query_gate,
            key,
            weights.q_norm,
            weights.k_norm,
            rope,
        )
        expected_state = turboquant_cache.advance_linear_state(
            expected_state,
            key[:, None, :],
            value[:, None, :],
        )
        attended, _ = turboquant_cache.packed_attention(query, expected_state)
        attended = attended.astype(mx.bfloat16) * mx.sigmoid(gate)
        expected = mx.matmul(weights.o_proj, attended.reshape(4096))

        actual, actual_state = mlx_attention.decode_step(
            hidden,
            actual_state,
            weights,
            config,
            rope=rope,
        )
        mx.eval(
            expected,
            actual,
            expected_state.packed_keys,
            expected_state.key_norms,
            expected_state.packed_values,
            expected_state.value_norms,
            actual_state.packed_keys,
            actual_state.key_norms,
            actual_state.packed_values,
            actual_state.value_norms,
        )

        self.assertTrue(bool(mx.array_equal(actual, expected).item()))
        self.assertEqual(mlx_attention.state_length(actual_state, config), 4)
        self.assertTrue(
            bool(mx.array_equal(actual_state.packed_keys, expected_state.packed_keys).item())
        )
        self.assertTrue(
            bool(mx.array_equal(actual_state.packed_values, expected_state.packed_values).item())
        )

    def test_analysis_projection_matches_authoritative_prefill_cache(self) -> None:
        config, scalar_weights = make_fixture()
        weights = bf16_weights(scalar_weights)
        hidden = mx.array(
            (
                [0.25, -0.5, 0.75, 0.1],
                [-0.2, 0.4, 0.3, -0.7],
                [0.9, 0.05, -0.6, 0.2],
            ),
            dtype=mx.bfloat16,
        )
        trace = mlx_attention.project_prefill_qkv_for_analysis(
            hidden,
            weights,
            0,
            config,
        )
        _, state = mlx_attention.prefill_chunk(
            hidden,
            mlx_attention.zeros_state(config, dtype=mx.bfloat16),
            weights,
            config,
            use_steel=False,
        )
        expected_keys = mx.swapaxes(trace.keys, 0, 1)
        expected_values = mx.swapaxes(trace.values, 0, 1)
        mx.eval(expected_keys, expected_values, state.keys, state.values)

        self.assertTrue(bool(mx.array_equal(state.keys, expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(state.values, expected_values).item()))

    def test_token_tiled_prefill_projection_matches_vmap(self) -> None:
        mx.random.seed(20260717)
        weight = mx.random.normal((64, 2048), dtype=mx.float32).astype(mx.bfloat16)
        hidden = mx.random.normal((8, 2048), dtype=mx.float32).astype(mx.bfloat16)
        expected = mlx_attention._prefill_linear(weight, hidden, False)
        actual = mlx_attention._prefill_linear(weight, hidden, True)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_fused_qkv_prefill_projection_matches_split_path(self) -> None:
        mx.random.seed(20260718)
        hidden = mx.random.normal((8, 2048), dtype=mx.float32).astype(mx.bfloat16)
        q_weight = mx.random.normal((8192, 2048), dtype=mx.float32).astype(
            mx.bfloat16
        )
        k_weight = mx.random.normal((512, 2048), dtype=mx.float32).astype(
            mx.bfloat16
        )
        v_weight = mx.random.normal((512, 2048), dtype=mx.float32).astype(
            mx.bfloat16
        )
        expected = tuple(
            mlx_attention._prefill_linear(weight, hidden, True)
            for weight in (q_weight, k_weight, v_weight)
        )
        actual = mlx_attention.fused_qkv_prefill_projection(
            q_weight,
            k_weight,
            v_weight,
            hidden,
        )
        mx.eval(*expected, *actual)
        for expected_projection, actual_projection in zip(expected, actual):
            self.assertTrue(
                bool(mx.array_equal(actual_projection, expected_projection).item())
            )

    def test_exact_long_prefill_threshold_is_quality_gated(self) -> None:
        self.assertEqual(mlx_attention.KEY_TILED_PREFILL_MIN_PREFIX, 4_096)
        self.assertEqual(mlx_attention.EXACT_LONG_PREFILL_MIN_PREFIX, 106_496)
        self.assertEqual(
            mlx_attention.EXACT_FUSED_SOFTMAX_VALUE_MAX_PREFIX,
            131_072,
        )
        self.assertEqual(
            mlx_attention.EXACT_FUSED_SOFTMAX_VALUE_MIN_TOKENS,
            64,
        )

    def test_key_tiled_scores_match_authoritative_gqa_gemv(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        mx.random.seed(20260718)
        tokens = 5
        groups = config.num_q_heads // config.num_kv_heads
        for key_length in (127, 1027, 4099):
            start_position = key_length - tokens
            queries = mx.random.normal(
                (tokens, config.num_q_heads, config.head_dim),
                dtype=mx.float32,
            ).astype(mx.bfloat16)
            keys = mx.random.normal(
                (config.num_kv_heads, key_length, config.head_dim),
                dtype=mx.float32,
            ).astype(mx.bfloat16)
            actual = mlx_attention._exact_batched_scores(
                queries,
                keys,
                mx.array(start_position, dtype=mx.uint32),
                mx.array(tokens, dtype=mx.uint32),
                mx.array(key_length, dtype=mx.uint32),
                queries_count=tokens,
                keys_count=key_length,
                key_tiled=True,
            )
            expected = []
            for offset, query in enumerate(queries):
                valid_length = start_position + offset + 1
                expected.append(
                    mx.matmul(
                        query.reshape(
                            config.num_kv_heads,
                            groups,
                            1,
                            config.head_dim,
                        ),
                        mx.swapaxes(keys[:, :valid_length, :], 1, 2)[
                            :, None, :, :
                        ],
                    ).reshape(config.num_q_heads, valid_length)
                )
            checks = [
                mx.array_equal(value, actual[index, :, : value.shape[1]])
                for index, value in enumerate(expected)
            ]
            mx.eval(*checks)
            self.assertTrue(all(bool(check.item()) for check in checks))

    def test_fused_long_softmax_value_matches_split_kernels(self) -> None:
        mx.random.seed(20260717)
        tokens = 2
        heads = mlx_attention.PRODUCTION_CONFIG.num_q_heads
        key_length = 8192
        start_position = key_length - tokens
        scores = mx.random.normal(
            (tokens, heads, key_length),
            dtype=mx.float32,
        ).astype(mx.bfloat16)
        values = mx.random.normal(
            (
                mlx_attention.PRODUCTION_CONFIG.num_kv_heads,
                key_length,
                mlx_attention.PRODUCTION_CONFIG.head_dim,
            ),
            dtype=mx.float32,
        ).astype(mx.bfloat16)
        start_scalar = mx.array(start_position, dtype=mx.uint32)
        length_scalar = mx.array(key_length, dtype=mx.uint32)
        rows = tokens * heads
        probabilities = mlx_attention._exact_looped_softmax_kernel(
            inputs=[scores, start_scalar, length_scalar],
            grid=(rows * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[scores.shape],
            output_dtypes=[mx.bfloat16],
        )[0]
        expected = mlx_attention._exact_batched_value_kernel(
            inputs=[probabilities, values, start_scalar, length_scalar],
            grid=(8 * 64, rows, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[
                (
                    tokens,
                    heads,
                    mlx_attention.PRODUCTION_CONFIG.head_dim,
                )
            ],
            output_dtypes=[mx.bfloat16],
        )[0]
        actual = mlx_attention._exact_fused_softmax_value_kernel(
            inputs=[scores, values, start_scalar, length_scalar],
            grid=(rows * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[expected.shape],
            output_dtypes=[mx.bfloat16],
        )[0]
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_linear_cache_prefill_matches_immutable_bf16_chunk(self) -> None:
        config, scalar_weights = make_fixture()
        weights = bf16_weights(scalar_weights)
        immutable = mlx_attention.zeros_state(config, dtype=mx.bfloat16)
        for values in ([0.1, -0.3, 0.2, 0.6], [-0.4, 0.7, -0.1, 0.25]):
            _, immutable = mlx_attention.decode_step(
                mx.array(values, dtype=mx.bfloat16),
                immutable,
                weights,
                config,
            )
        mx.eval(immutable.keys, immutable.values)
        prefix_keys = mx.array(immutable.keys)
        prefix_values = mx.array(immutable.values)
        linear = mlx_attention.linearize_state(immutable, 8, config)
        hidden = mx.array(
            (
                [0.25, -0.5, 0.75, 0.1],
                [-0.2, 0.4, 0.3, -0.7],
                [0.9, 0.05, -0.6, 0.2],
            ),
            dtype=mx.bfloat16,
        )

        expected, expected_state = mlx_attention.prefill_chunk(
            hidden,
            immutable,
            weights,
            config,
            use_steel=False,
        )
        actual, actual_state = mlx_attention.prefill_chunk(
            hidden,
            linear,
            weights,
            config,
            use_steel=False,
        )
        mx.eval(
            expected,
            expected_state.keys,
            expected_state.values,
            actual,
            actual_state.keys,
            actual_state.values,
        )

        self.assertTrue(bool(mx.array_equal(actual, expected).item()))
        self.assertEqual(actual_state.position, 5)
        self.assertTrue(
            bool(mx.array_equal(actual_state.keys[:, :5], expected_state.keys).item())
        )
        self.assertTrue(
            bool(mx.array_equal(actual_state.values[:, :5], expected_state.values).item())
        )
        self.assertTrue(bool(mx.array_equal(immutable.keys, prefix_keys).item()))
        self.assertTrue(bool(mx.array_equal(immutable.values, prefix_values).item()))

    def test_kv_only_prefill_matches_full_chunk_state(self) -> None:
        config, scalar_weights = make_fixture()
        weights = bf16_weights(scalar_weights)
        immutable = mlx_attention.zeros_state(config, dtype=mx.bfloat16)
        for values in ([0.1, -0.3, 0.2, 0.6], [-0.4, 0.7, -0.1, 0.25]):
            _, immutable = mlx_attention.decode_step(
                mx.array(values, dtype=mx.bfloat16),
                immutable,
                weights,
                config,
            )
        mx.eval(immutable.keys, immutable.values)
        linear = mlx_attention.linearize_state(immutable, 8, config)
        hidden = mx.array(
            (
                [0.25, -0.5, 0.75, 0.1],
                [-0.2, 0.4, 0.3, -0.7],
                [0.9, 0.05, -0.6, 0.2],
            ),
            dtype=mx.bfloat16,
        )

        _, expected = mlx_attention.prefill_chunk(
            hidden,
            immutable,
            weights,
            config,
            use_steel=False,
        )
        actual = mlx_attention.prefill_kv_chunk(
            hidden,
            immutable,
            weights,
            config,
        )
        actual_linear = mlx_attention.prefill_kv_chunk(
            hidden,
            linear,
            weights,
            config,
        )
        mx.eval(
            expected.keys,
            expected.values,
            actual.keys,
            actual.values,
            actual_linear.keys,
            actual_linear.values,
        )

        self.assertTrue(bool(mx.array_equal(actual.keys, expected.keys).item()))
        self.assertTrue(bool(mx.array_equal(actual.values, expected.values).item()))
        self.assertEqual(actual_linear.position, 5)
        self.assertTrue(
            bool(mx.array_equal(actual_linear.keys[:, :5], expected.keys).item())
        )
        self.assertTrue(
            bool(mx.array_equal(actual_linear.values[:, :5], expected.values).item())
        )

    def test_last_query_prefill_matches_full_chunk_final_output_and_state(self) -> None:
        config, scalar_weights = make_fixture()
        weights = bf16_weights(scalar_weights)
        immutable = mlx_attention.zeros_state(config, dtype=mx.bfloat16)
        for values in ([0.1, -0.3, 0.2, 0.6], [-0.4, 0.7, -0.1, 0.25]):
            _, immutable = mlx_attention.decode_step(
                mx.array(values, dtype=mx.bfloat16),
                immutable,
                weights,
                config,
            )
        mx.eval(immutable.keys, immutable.values)
        linear = mlx_attention.linearize_state(immutable, 8, config)
        hidden = mx.array(
            (
                [0.25, -0.5, 0.75, 0.1],
                [-0.2, 0.4, 0.3, -0.7],
                [0.9, 0.05, -0.6, 0.2],
            ),
            dtype=mx.bfloat16,
        )

        expected_output, expected_state = mlx_attention.prefill_chunk(
            hidden,
            immutable,
            weights,
            config,
            use_steel=False,
        )
        actual_output, actual_state = mlx_attention.prefill_last_query_chunk(
            hidden,
            immutable,
            weights,
            config,
        )
        linear_output, linear_state = mlx_attention.prefill_last_query_chunk(
            hidden,
            linear,
            weights,
            config,
        )
        mx.eval(
            expected_output,
            expected_state.keys,
            expected_state.values,
            actual_output,
            actual_state.keys,
            actual_state.values,
            linear_output,
            linear_state.keys,
            linear_state.values,
        )

        self.assertTrue(bool(mx.array_equal(actual_output, expected_output[-1]).item()))
        self.assertTrue(bool(mx.array_equal(linear_output, expected_output[-1]).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.keys, expected_state.keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.values, expected_state.values).item()))
        self.assertTrue(
            bool(mx.array_equal(linear_state.keys[:, :5], expected_state.keys).item())
        )
        self.assertTrue(
            bool(mx.array_equal(linear_state.values[:, :5], expected_state.values).item())
        )

    def test_linear_cache_matches_immutable_bf16_trajectory(self) -> None:
        config, scalar_weights = make_fixture()
        weights = bf16_weights(scalar_weights)
        immutable = mlx_attention.zeros_state(config, dtype=mx.bfloat16)
        prefix = (
            [0.1, -0.3, 0.2, 0.6],
            [-0.4, 0.7, -0.1, 0.25],
        )
        for values in prefix:
            _, immutable = mlx_attention.decode_step(
                mx.array(values, dtype=mx.bfloat16),
                immutable,
                weights,
                config,
            )
        mx.eval(immutable.keys, immutable.values)
        prefix_keys = mx.array(immutable.keys)
        prefix_values = mx.array(immutable.values)
        linear = mlx_attention.linearize_state(immutable, 8, config)
        mx.eval(linear.keys, linear.values)

        for values in (
            [0.25, -0.5, 0.75, 0.1],
            [-0.2, 0.4, 0.3, -0.7],
            [0.9, 0.05, -0.6, 0.2],
        ):
            hidden = mx.array(values, dtype=mx.bfloat16)
            expected, immutable = mlx_attention.decode_step(
                hidden,
                immutable,
                weights,
                config,
            )
            actual, linear = mlx_attention.decode_step(
                hidden,
                linear,
                weights,
                config,
            )
            mx.eval(expected, immutable.keys, immutable.values, actual, linear.keys, linear.values)
            self.assertTrue(bool(mx.array_equal(actual, expected).item()))
            self.assertTrue(
                bool(mx.array_equal(linear.keys[:, : linear.position], immutable.keys).item())
            )
            self.assertTrue(
                bool(mx.array_equal(linear.values[:, : linear.position], immutable.values).item())
            )

        self.assertEqual(linear.position, 5)
        self.assertEqual(linear.capacity, 8)
        self.assertTrue(bool(mx.array_equal(prefix_keys, immutable.keys[:, :2]).item()))
        self.assertTrue(bool(mx.array_equal(prefix_values, immutable.values[:, :2]).item()))

    def test_rope_chunk_matches_independent_positions(self) -> None:
        config = reference.AttentionConfig(
            hidden_size=4,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=8,
            rotary_dim=8,
            rope_theta=10_000_000.0,
        )
        values = mx.array(
            [math.sin((index + 1) * 0.17) for index in range(3 * 2 * 8)],
            dtype=mx.float32,
        ).reshape(3, 2, 8).astype(mx.bfloat16)
        expected = mx.stack(
            [
                mlx_attention._apply_text_rope(
                    value,
                    262_141 + offset,
                    config,
                    mx.bfloat16,
                )
                for offset, value in enumerate(values)
            ]
        )
        actual = mlx_attention._apply_text_rope_chunk(
            values,
            262_141,
            config,
            mx.bfloat16,
            mlx_attention.make_text_rope(
                262_141,
                values.shape[0],
                config,
                mx.bfloat16,
            ),
        )
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_production_contract(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        self.assertEqual(config.query_dim, 4096)
        self.assertEqual(config.kv_dim, 512)
        self.assertEqual(config.rotary_dim, 64)
        state = mlx_attention.zeros_state(config)
        self.assertEqual(state.keys.shape, (2, 0, 256))
        self.assertEqual(state.values.shape, (2, 0, 256))

    def test_fused_production_qk_norm_rope_matches_split_path(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        # This seed exposed a one-BF16-value drift when the custom kernel used
        # a different reduction topology than MLX's 256-wide row reduction.
        mx.random.seed(1022)
        query_gate = mx.random.normal((config.query_dim * 2,), dtype=mx.float32).astype(
            mx.bfloat16
        )
        key = mx.random.normal((config.kv_dim,), dtype=mx.float32).astype(mx.bfloat16)
        q_norm = mx.random.normal((config.head_dim,), dtype=mx.float32).astype(
            mx.bfloat16
        )
        k_norm = mx.random.normal((config.head_dim,), dtype=mx.float32).astype(
            mx.bfloat16
        )
        position = 262_143
        rope = mlx_attention.make_text_rope(
            position,
            1,
            config,
            mx.bfloat16,
        )
        split = query_gate.reshape(config.num_q_heads, config.head_dim * 2)
        expected_query = mlx_attention._apply_text_rope(
            mlx_attention._rms_norm(
                split[:, : config.head_dim],
                q_norm,
                config.rms_norm_eps,
                mx.bfloat16,
            ),
            position,
            config,
            mx.bfloat16,
            rope,
        )
        expected_gate = split[:, config.head_dim :]
        expected_key = mlx_attention._apply_text_rope(
            mlx_attention._rms_norm(
                key.reshape(config.num_kv_heads, config.head_dim),
                k_norm,
                config.rms_norm_eps,
                mx.bfloat16,
            ),
            position,
            config,
            mx.bfloat16,
            rope,
        )
        actual_query, actual_gate, actual_key = (
            mlx_attention.fused_qk_norm_rope_step(
                query_gate,
                key,
                q_norm,
                k_norm,
                rope,
            )
        )
        mx.eval(
            expected_query,
            expected_gate,
            expected_key,
            actual_query,
            actual_gate,
            actual_key,
        )
        self.assertTrue(bool(mx.array_equal(actual_query, expected_query).item()))
        self.assertTrue(bool(mx.array_equal(actual_gate, expected_gate).item()))
        self.assertTrue(bool(mx.array_equal(actual_key, expected_key).item()))

        yarn_position = 524_287
        yarn_rope = mlx_attention.make_text_rope(
            yarn_position,
            1,
            config,
            mx.bfloat16,
            context.YARN2_PROFILE_ID,
        )
        yarn_query = mlx_attention._apply_text_rope(
            mlx_attention._rms_norm(
                split[:, : config.head_dim],
                q_norm,
                config.rms_norm_eps,
                mx.bfloat16,
            ),
            yarn_position,
            config,
            mx.bfloat16,
            yarn_rope,
            context.YARN2_PROFILE_ID,
        )
        yarn_key = mlx_attention._apply_text_rope(
            mlx_attention._rms_norm(
                key.reshape(config.num_kv_heads, config.head_dim),
                k_norm,
                config.rms_norm_eps,
                mx.bfloat16,
            ),
            yarn_position,
            config,
            mx.bfloat16,
            yarn_rope,
            context.YARN2_PROFILE_ID,
        )
        actual_query, actual_gate, actual_key = mlx_attention.fused_qk_norm_rope_step(
            query_gate,
            key,
            q_norm,
            k_norm,
            yarn_rope,
        )
        mx.eval(yarn_query, yarn_key, actual_query, actual_gate, actual_key)
        self.assertTrue(bool(mx.array_equal(actual_query, yarn_query).item()))
        self.assertTrue(bool(mx.array_equal(actual_gate, expected_gate).item()))
        self.assertTrue(bool(mx.array_equal(actual_key, yarn_key).item()))

    def test_fused_production_qk_norm_rope_chunk_matches_split_path(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        mx.random.seed(20260717)
        tokens = 5
        query_gate = mx.random.normal(
            (tokens, config.query_dim * 2),
            dtype=mx.float32,
        ).astype(mx.bfloat16)
        key = mx.random.normal(
            (tokens, config.num_kv_heads, config.head_dim),
            dtype=mx.float32,
        ).astype(mx.bfloat16)
        q_norm = mx.random.normal((config.head_dim,), dtype=mx.float32).astype(
            mx.bfloat16
        )
        k_norm = mx.random.normal((config.head_dim,), dtype=mx.float32).astype(
            mx.bfloat16
        )
        position = 262_139
        rope = mlx_attention.make_text_rope(
            position,
            tokens,
            config,
            mx.bfloat16,
        )
        split = query_gate.reshape(
            tokens,
            config.num_q_heads,
            config.head_dim * 2,
        )
        expected_query = mlx_attention._apply_text_rope_chunk(
            mlx_attention._rms_norm(
                split[:, :, : config.head_dim],
                q_norm,
                config.rms_norm_eps,
                mx.bfloat16,
            ),
            position,
            config,
            mx.bfloat16,
            rope,
        )
        expected_gate = split[:, :, config.head_dim :]
        expected_key = mlx_attention._apply_text_rope_chunk(
            mlx_attention._rms_norm(
                key,
                k_norm,
                config.rms_norm_eps,
                mx.bfloat16,
            ),
            position,
            config,
            mx.bfloat16,
            rope,
        )
        actual_query, actual_gate, actual_key = (
            mlx_attention.fused_qk_norm_rope_chunk(
                query_gate,
                key,
                q_norm,
                k_norm,
                rope,
            )
        )
        actual_key_only = mlx_attention.fused_key_norm_rope_chunk(
            key,
            k_norm,
            rope,
        )
        mx.eval(
            expected_query,
            expected_gate,
            expected_key,
            actual_query,
            actual_gate,
            actual_key,
            actual_key_only,
        )
        self.assertTrue(bool(mx.array_equal(actual_query, expected_query).item()))
        self.assertTrue(bool(mx.array_equal(actual_gate, expected_gate).item()))
        self.assertTrue(bool(mx.array_equal(actual_key, expected_key).item()))
        self.assertTrue(bool(mx.array_equal(actual_key_only, expected_key).item()))

    def test_fused_production_prefill_rejects_wrong_rope_position(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        tokens = 2
        weights = mlx_attention.MLXAttentionWeights(
            q_proj=mx.zeros((config.query_dim * 2, config.hidden_size), dtype=mx.bfloat16),
            k_proj=mx.zeros((config.kv_dim, config.hidden_size), dtype=mx.bfloat16),
            v_proj=mx.zeros((config.kv_dim, config.hidden_size), dtype=mx.bfloat16),
            o_proj=mx.zeros((config.hidden_size, config.query_dim), dtype=mx.bfloat16),
            q_norm=mx.zeros((config.head_dim,), dtype=mx.bfloat16),
            k_norm=mx.zeros((config.head_dim,), dtype=mx.bfloat16),
        )
        hidden = mx.zeros((tokens, config.hidden_size), dtype=mx.bfloat16)
        state = mlx_attention.zeros_state(config, dtype=mx.bfloat16)
        wrong_rope = mlx_attention.make_text_rope(
            1,
            tokens,
            config,
            mx.bfloat16,
        )
        with self.assertRaisesRegex(reference.AttentionError, "RoPE range mismatch"):
            mlx_attention.prefill_kv_chunk(
                hidden,
                state,
                weights,
                config,
                rope=wrong_rope,
            )
        with self.assertRaisesRegex(reference.AttentionError, "RoPE range mismatch"):
            mlx_attention.prefill_chunk(
                hidden,
                state,
                weights,
                config,
                use_steel=False,
                rope=wrong_rope,
            )

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

    def test_yarn_multistep_matches_scalar_oracle(self) -> None:
        config, scalar_weights = make_fixture()
        gpu_weights = mlx_weights(scalar_weights)
        scalar_state = reference.zeros_state(
            config,
            context.YARN2_PROFILE_ID,
        )
        gpu_state = mlx_attention.zeros_state(
            config,
            dtype=mx.float32,
            context_profile=context.YARN2_PROFILE_ID,
        )
        for hidden in (
            [0.25, -0.5, 0.75, 0.1],
            [-0.2, 0.4, 0.3, -0.7],
            [0.9, 0.05, -0.6, 0.2],
        ):
            expected, scalar_state = reference.decode_step(
                hidden,
                scalar_state,
                scalar_weights,
                config,
            )
            actual, gpu_state = mlx_attention.decode_step(
                mx.array(hidden, dtype=mx.float32),
                gpu_state,
                gpu_weights,
                config,
                fused_qk_norm_rope=False,
            )
            mx.eval(actual, gpu_state.keys, gpu_state.values)
            for left, right in zip(actual.tolist(), expected):
                self.assertAlmostEqual(left, right, delta=3e-6)
        self.assertEqual(gpu_state.context_profile, context.YARN2_PROFILE_ID)
        self.assertEqual(scalar_state.context_profile, context.YARN2_PROFILE_ID)

    def test_grouped_gqa_decode_matches_repeated_cache_path(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        state = mlx_attention.zeros_state(config, dtype=mx.float32)
        for hidden in (
            [0.1, -0.3, 0.2, 0.6],
            [-0.4, 0.7, -0.1, 0.25],
        ):
            _, state = mlx_attention.decode_step(
                mx.array(hidden, dtype=mx.float32),
                state,
                weights,
                config,
            )
        hidden = mx.array([0.25, -0.5, 0.75, 0.1], dtype=mx.float32)
        expected, expected_state = mlx_attention.decode_step(
            hidden,
            state,
            weights,
            config,
            grouped_gqa=False,
        )
        actual, actual_state = mlx_attention.decode_step(
            hidden,
            state,
            weights,
            config,
            grouped_gqa=True,
        )
        mx.eval(
            expected,
            expected_state.keys,
            expected_state.values,
            actual,
            actual_state.keys,
            actual_state.values,
        )
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.keys, expected_state.keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.values, expected_state.values).item()))

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
        state = mlx_attention.zeros_state(config, dtype=mx.float32)
        expected = []
        expected_state = state
        for token in hidden:
            output, expected_state = mlx_attention.decode_step(
                token,
                expected_state,
                weights,
                config,
            )
            expected.append(output)
        expected_output = mx.stack(expected)
        actual_output, actual_state = mlx_attention.prefill_chunk(
            hidden,
            state,
            weights,
            config,
        )
        repeated_output, repeated_state = mlx_attention.prefill_chunk(
            hidden,
            state,
            weights,
            config,
            grouped_gqa=False,
        )
        mx.eval(
            expected_output,
            expected_state.keys,
            expected_state.values,
            actual_output,
            actual_state.keys,
            actual_state.values,
            repeated_output,
            repeated_state.keys,
            repeated_state.values,
        )
        self.assertTrue(bool(mx.array_equal(actual_output, expected_output).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.keys, expected_state.keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.values, expected_state.values).item()))
        self.assertTrue(bool(mx.array_equal(actual_output, repeated_output).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.keys, repeated_state.keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.values, repeated_state.values).item()))

    def test_generic_prefill_chunk_matches_token_steps_after_prefix(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        prefix = mx.array(
            ([0.1, -0.3, 0.2, 0.6], [-0.4, 0.7, -0.1, 0.25]),
            dtype=mx.float32,
        )
        hidden = mx.array(
            ([0.25, -0.5, 0.75, 0.1], [-0.2, 0.4, 0.3, -0.7]),
            dtype=mx.float32,
        )
        state = mlx_attention.zeros_state(config, dtype=mx.float32)
        for token in prefix:
            _, state = mlx_attention.decode_step(token, state, weights, config)
        expected = []
        expected_state = state
        for token in hidden:
            output, expected_state = mlx_attention.decode_step(
                token,
                expected_state,
                weights,
                config,
            )
            expected.append(output)
        expected_output = mx.stack(expected)
        actual_output, actual_state = mlx_attention.prefill_chunk(
            hidden,
            state,
            weights,
            config,
        )
        mx.eval(
            expected_output,
            expected_state.keys,
            expected_state.values,
            actual_output,
            actual_state.keys,
            actual_state.values,
        )
        self.assertTrue(bool(mx.array_equal(actual_output, expected_output).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.keys, expected_state.keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_state.values, expected_state.values).item()))

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

    def test_yarn_rope_matches_scalar_equation_and_profile_boundary(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        rope = mlx_attention.make_text_rope(
            1,
            1,
            config,
            mx.float32,
            context.YARN2_PROFILE_ID,
        )
        inverse, scaling = context.rope_parameters(
            context.YARN2_PROFILE_ID,
            config.rotary_dim,
            config.rope_theta,
        )
        expected_cosine = [math.cos(value) * scaling for value in (*inverse, *inverse)]
        expected_sine = [math.sin(value) * scaling for value in (*inverse, *inverse)]
        mx.eval(rope.cosine, rope.sine)
        for actual, expected in zip(rope.cosine.tolist(), expected_cosine):
            self.assertAlmostEqual(actual, expected, delta=2e-6)
        for actual, expected in zip(rope.sine.tolist(), expected_sine):
            self.assertAlmostEqual(actual, expected, delta=2e-6)

        final = mlx_attention.make_text_rope(
            524_287,
            1,
            config,
            mx.bfloat16,
            context.YARN2_PROFILE_ID,
        )
        mx.eval(final.cosine, final.sine)
        self.assertEqual(final.context_profile, context.YARN2_PROFILE_ID)
        self.assertTrue(bool(mx.all(mx.isfinite(final.cosine)).item()))
        self.assertTrue(bool(mx.all(mx.isfinite(final.sine)).item()))
        with self.assertRaisesRegex(context.ContextError, "yarn2-524k"):
            mlx_attention.make_text_rope(
                524_288,
                1,
                config,
                mx.bfloat16,
                context.YARN2_PROFILE_ID,
            )

    def test_rope_profile_mismatch_is_rejected(self) -> None:
        config = mlx_attention.PRODUCTION_CONFIG
        rope = mlx_attention.make_text_rope(
            7,
            1,
            config,
            mx.bfloat16,
            context.YARN2_PROFILE_ID,
        )
        value = mx.zeros((1, config.head_dim), dtype=mx.bfloat16)
        with self.assertRaisesRegex(reference.AttentionError, "context profile mismatch"):
            mlx_attention._apply_text_rope(
                value,
                7,
                config,
                mx.bfloat16,
                rope,
                context.NATIVE_PROFILE_ID,
            )

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

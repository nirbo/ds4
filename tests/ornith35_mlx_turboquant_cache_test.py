#!/usr/bin/env python3
"""Metal parity tests for direct packed Ornith-35 K4-MSE K/V primitives."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ornith35" / "tools"))

import ornith35_mlx_turboquant as turboquant
import ornith35_mlx_turboquant_cache as cache
import ornith35_turboquant_reference as reference


def fixture(tokens: int = 19) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(20260718)
    keys = mx.random.normal((2, tokens, 256), dtype=mx.float32).astype(mx.bfloat16)
    values = mx.random.normal((2, tokens, 256), dtype=mx.float32).astype(mx.bfloat16)
    queries = mx.random.normal((16, 256), dtype=mx.float32).astype(mx.bfloat16)
    return keys, values, queries


class MLXTurboQuantCacheTest(unittest.TestCase):
    def test_gpu_encoding_matches_unpacked_mlx_authority(self) -> None:
        transforms = cache.production_transforms()
        vectors = fixture(3)[0]
        actual = cache.encode_mse4(vectors, transforms.key)
        expected_packed = cache.encode_mse4_graph(vectors, transforms.key)
        expected = turboquant.quantize_mse(vectors, 4, transforms.key)
        unpacked = cache.unpack_indices(actual)
        mx.eval(
            unpacked,
            actual.norms,
            expected_packed.packed,
            expected_packed.norms,
            expected.indices,
            expected.norms,
        )

        self.assertTrue(bool(mx.array_equal(unpacked, expected.indices).item()))
        self.assertTrue(bool(mx.array_equal(actual.norms, expected.norms).item()))
        self.assertTrue(bool(mx.array_equal(actual.packed, expected_packed.packed).item()))
        self.assertTrue(bool(mx.array_equal(actual.norms, expected_packed.norms).item()))

    def test_gpu_encoder_materializes_noncontiguous_prefix_exactly(self) -> None:
        transforms = cache.production_transforms()
        vectors = fixture(257)[0][:, :256]
        actual = cache.encode_mse4(vectors, transforms.key)
        expected = cache.encode_mse4_graph(vectors, transforms.key)
        mx.eval(actual.packed, actual.norms, expected.packed, expected.norms)

        self.assertTrue(bool(mx.array_equal(actual.packed, expected.packed).item()))
        self.assertTrue(bool(mx.array_equal(actual.norms, expected.norms).item()))

    def test_zero_vectors_round_trip_without_cpu_validation(self) -> None:
        transforms = cache.production_transforms()
        encoded = cache.encode_mse4(
            mx.zeros((2, 1, 256), dtype=mx.bfloat16),
            transforms.key,
        )
        reconstructed = cache.dequantize_mse4(encoded, transforms.key)
        mx.eval(encoded.packed, encoded.norms, reconstructed)

        self.assertEqual(encoded.norms.tolist(), [[[0.0]], [[0.0]]])
        self.assertTrue(bool(mx.all(reconstructed == 0.0).item()))

    def test_direct_packed_scores_match_materialized_oracle(self) -> None:
        keys, values, queries = fixture()
        transforms = cache.production_transforms()
        state = cache.compress_bf16_kv(keys, values, transforms, exact_tail=1)
        reconstructed_keys, _ = cache.dequantize_state(state, transforms)
        repeated = mx.repeat(reconstructed_keys, 8, axis=0)
        expected = mx.sum(queries.astype(mx.float32)[:, None, :] * repeated, axis=-1)
        actual = cache.packed_scores(queries, state, transforms)
        mx.eval(expected, actual)
        error = float(mx.max(mx.abs(actual - expected)).item())

        self.assertLess(error, 2e-4)

    def test_direct_packed_values_match_materialized_oracle(self) -> None:
        keys, values, _ = fixture()
        transforms = cache.production_transforms()
        state = cache.compress_bf16_kv(keys, values, transforms, exact_tail=1)
        _, reconstructed_values = cache.dequantize_state(state, transforms)
        logits = mx.arange(16 * 19, dtype=mx.float32).reshape(16, 19) * 0.001
        probabilities = mx.softmax(logits, axis=-1)
        repeated = mx.repeat(reconstructed_values, 8, axis=0)
        expected = mx.sum(probabilities[:, :, None] * repeated, axis=1)
        actual = cache.packed_attend(probabilities, state, transforms)
        mx.eval(expected, actual)
        error = float(mx.max(mx.abs(actual - expected)).item())

        self.assertLess(error, 2e-4)

    def test_reduction_topology_matches_materialized_at_longer_history(self) -> None:
        keys, values, queries = fixture(257)
        transforms = cache.production_transforms()
        state = cache.compress_bf16_kv(keys, values, transforms, exact_tail=1)
        reconstructed_keys, reconstructed_values = cache.dequantize_state(state, transforms)
        repeated_keys = mx.repeat(reconstructed_keys, 8, axis=0)
        expected_scores = mx.sum(
            queries.astype(mx.float32)[:, None, :] * repeated_keys,
            axis=-1,
        )
        expected_probabilities = mx.softmax(expected_scores * (256**-0.5), axis=-1)
        repeated_values = mx.repeat(reconstructed_values, 8, axis=0)
        expected_output = mx.sum(
            expected_probabilities[:, :, None] * repeated_values,
            axis=1,
        )
        actual_output, actual_probabilities = cache.packed_attention(
            queries,
            state,
            transforms,
        )
        mx.eval(
            expected_probabilities,
            expected_output,
            actual_probabilities,
            actual_output,
        )

        probability_error = float(
            mx.max(mx.abs(actual_probabilities - expected_probabilities)).item()
        )
        output_error = float(mx.max(mx.abs(actual_output - expected_output)).item())
        self.assertLess(probability_error, 2e-6)
        self.assertLess(output_error, 2e-5)

    def test_fixed_capacity_stride_matches_immutable_attention(self) -> None:
        keys, values, queries = fixture()
        transforms = cache.production_transforms()
        immutable = cache.compress_bf16_kv(keys, values, transforms, exact_tail=1)
        linear = cache.linearize_state(immutable, 257)
        immutable_output, immutable_probabilities = cache.packed_attention(
            queries,
            immutable,
            transforms,
        )
        linear_output, linear_probabilities = cache.packed_attention(
            queries,
            linear,
            transforms,
        )
        mx.eval(
            linear.packed_keys,
            linear.key_norms,
            linear.packed_values,
            linear.value_norms,
            immutable_output,
            immutable_probabilities,
            linear_output,
            linear_probabilities,
        )

        self.assertTrue(bool(mx.array_equal(linear_probabilities, immutable_probabilities).item()))
        self.assertTrue(bool(mx.array_equal(linear_output, immutable_output).item()))
        self.assertEqual(cache.state_length(linear), 19)
        self.assertEqual(cache.packed_history(linear), 18)
        self.assertEqual(cache.stored_bytes(linear), 135_688)

    def test_linear_advance_matches_direct_compression(self) -> None:
        keys, values, _ = fixture(7)
        more_keys, more_values, _ = fixture(3)
        transforms = cache.production_transforms()
        state = cache.linearize_bf16_kv(keys, values, 32, transforms)
        state = cache.advance_linear_state(state, more_keys, more_values, transforms)
        mx.eval(
            state.packed_keys,
            state.key_norms,
            state.packed_values,
            state.value_norms,
            state.exact_keys,
            state.exact_values,
        )

        expected = cache.compress_bf16_kv(
            mx.concatenate((keys, more_keys), axis=1),
            mx.concatenate((values, more_values), axis=1),
            transforms,
            exact_tail=1,
        )
        actual_keys, actual_values = cache.dequantize_state(state, transforms)
        expected_keys, expected_values = cache.dequantize_state(expected, transforms)
        mx.eval(actual_keys, actual_values, expected_keys, expected_values)

        self.assertEqual(cache.state_length(state), 10)
        self.assertEqual(cache.packed_history(state), 9)
        self.assertTrue(bool(mx.array_equal(actual_keys, expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_values, expected_values).item()))

    def test_empty_linear_state_promotes_one_exact_tail(self) -> None:
        transforms = cache.production_transforms()
        empty = mx.zeros((2, 0, 256), dtype=mx.bfloat16)
        state = cache.linearize_bf16_kv(empty, empty, 4, transforms)
        keys, values, _ = fixture(1)
        state = cache.advance_linear_state(state, keys, values, transforms)
        mx.eval(state.exact_keys, state.exact_values)

        self.assertEqual(cache.state_length(state), 1)
        self.assertEqual(cache.packed_history(state), 0)
        self.assertTrue(bool(mx.array_equal(state.exact_keys, keys).item()))
        with self.assertRaisesRegex(reference.TurboQuantError, "capacity exhausted"):
            cache.advance_linear_state(
                state,
                mx.zeros((2, 4, 256), dtype=mx.bfloat16),
                mx.zeros((2, 4, 256), dtype=mx.bfloat16),
                transforms,
            )

    def test_lazy_linear_advance_chain_preserves_order(self) -> None:
        transforms = cache.production_transforms()
        empty = mx.zeros((2, 0, 256), dtype=mx.bfloat16)
        state = cache.linearize_bf16_kv(empty, empty, 8, transforms)
        key_parts = []
        value_parts = []
        for token in range(5):
            keys = mx.full((2, 1, 256), token + 1, dtype=mx.bfloat16)
            values = mx.full((2, 1, 256), token + 11, dtype=mx.bfloat16)
            key_parts.append(keys)
            value_parts.append(values)
            state = cache.advance_linear_state(state, keys, values, transforms)
        mx.eval(
            state.packed_keys,
            state.key_norms,
            state.packed_values,
            state.value_norms,
            state.exact_keys,
            state.exact_values,
        )

        expected = cache.compress_bf16_kv(
            mx.concatenate(key_parts, axis=1),
            mx.concatenate(value_parts, axis=1),
            transforms,
            exact_tail=1,
        )
        actual_keys, actual_values = cache.dequantize_state(state, transforms)
        expected_keys, expected_values = cache.dequantize_state(expected, transforms)
        mx.eval(actual_keys, actual_values, expected_keys, expected_values)

        self.assertTrue(bool(mx.array_equal(actual_keys, expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_values, expected_values).item()))

    def test_state_storage_is_physical_and_tail_bounded(self) -> None:
        keys, values, _ = fixture()
        state = cache.compress_bf16_kv(keys, values, exact_tail=1)
        mx.eval(
            state.packed_keys,
            state.key_norms,
            state.packed_values,
            state.value_norms,
        )

        self.assertEqual(cache.state_length(state), 19)
        self.assertEqual(cache.stored_bytes(state), 11_408)
        self.assertEqual(state.packed_keys.shape, (2, 18, 128))
        self.assertEqual(state.exact_keys.shape, (2, 1, 256))
        with self.assertRaisesRegex(reference.TurboQuantError, "exact tail"):
            cache.compress_bf16_kv(keys, values, exact_tail=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

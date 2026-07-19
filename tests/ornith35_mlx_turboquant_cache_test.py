#!/usr/bin/env python3
"""Metal parity tests for direct packed Ornith-35 K9-MSE K/V primitives."""

from __future__ import annotations

import gc
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
    def test_production_norm_policy_is_narrow_and_explicit(self) -> None:
        self.assertEqual(cache.PRODUCTION_BF16_NORM_LAYERS, frozenset())
        self.assertEqual(cache.production_norm_dtype(3), mx.float32)
        self.assertEqual(cache.production_norm_dtype(7), mx.float32)

    def test_gpu_encoding_matches_unpacked_mlx_authority(self) -> None:
        transforms = cache.production_transforms()
        vectors = fixture(3)[0]
        actual = cache.encode_mse(vectors, transforms.key)
        expected_packed = cache.encode_mse_graph(vectors, transforms.key)
        expected = turboquant.quantize_mse(
            vectors,
            cache.PACKED_BITS,
            transforms.key,
            norm_dtype=cache.PRODUCTION_NORM_DTYPE,
        )
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
        actual = cache.encode_mse(vectors, transforms.key)
        expected = cache.encode_mse_graph(vectors, transforms.key)
        mx.eval(actual.packed, actual.norms, expected.packed, expected.norms)

        self.assertTrue(bool(mx.array_equal(actual.packed, expected.packed).item()))
        self.assertTrue(bool(mx.array_equal(actual.norms, expected.norms).item()))

    def test_zero_vectors_round_trip_without_cpu_validation(self) -> None:
        transforms = cache.production_transforms()
        encoded = cache.encode_mse(
            mx.zeros((2, 1, 256), dtype=mx.bfloat16),
            transforms.key,
        )
        reconstructed = cache.dequantize_mse(encoded, transforms.key)
        mx.eval(encoded.packed, encoded.norms, reconstructed)

        self.assertEqual(encoded.norms.tolist(), [[[0.0]], [[0.0]]])
        self.assertTrue(bool(mx.all(reconstructed == 0.0).item()))

    def test_bf16_norm_ablation_survives_linearization_and_append(self) -> None:
        transforms = cache.production_transforms()
        keys, values, _ = fixture(7)
        state = cache.linearize_bf16_kv(
            keys,
            values,
            16,
            transforms,
            exact_head=1,
            exact_tail=1,
            norm_dtype=mx.bfloat16,
        )
        updates = fixture(3)
        state = cache.advance_linear_state(state, updates[0], updates[1], transforms)
        mx.eval(state.packed_keys, state.key_norms, state.value_norms)

        self.assertEqual(state.key_norms.dtype, mx.bfloat16)
        self.assertEqual(state.value_norms.dtype, mx.bfloat16)
        self.assertEqual(cache.state_length(state), 10)

    def test_direct_packed_scores_match_materialized_oracle(self) -> None:
        keys, values, queries = fixture()
        transforms = cache.production_transforms()
        state = cache.compress_bf16_kv(
            keys,
            values,
            transforms,
            exact_head=1,
            exact_tail=1,
        )
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
        state = cache.compress_bf16_kv(
            keys,
            values,
            transforms,
            exact_head=1,
            exact_tail=1,
        )
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
        state = cache.compress_bf16_kv(
            keys,
            values,
            transforms,
            exact_head=1,
            exact_tail=1,
        )
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
        immutable = cache.compress_bf16_kv(
            keys,
            values,
            transforms,
            exact_head=1,
            exact_tail=1,
        )
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
        self.assertEqual(cache.packed_history(linear), 17)
        self.assertEqual(cache.stored_bytes(linear), 304_272)

    def test_linear_advance_matches_direct_compression(self) -> None:
        keys, values, _ = fixture(7)
        more_keys, more_values, _ = fixture(3)
        transforms = cache.production_transforms()
        state = cache.linearize_bf16_kv(
            keys,
            values,
            32,
            transforms,
            exact_head=1,
            exact_tail=1,
        )
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
            exact_head=1,
            exact_tail=1,
        )
        actual_keys, actual_values = cache.dequantize_state(state, transforms)
        expected_keys, expected_values = cache.dequantize_state(expected, transforms)
        mx.eval(actual_keys, actual_values, expected_keys, expected_values)

        self.assertEqual(cache.state_length(state), 10)
        self.assertEqual(cache.packed_history(state), 8)
        self.assertTrue(bool(mx.array_equal(actual_keys, expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_values, expected_values).item()))

    def test_empty_linear_state_promotes_one_exact_head(self) -> None:
        transforms = cache.production_transforms()
        empty = mx.zeros((2, 0, 256), dtype=mx.bfloat16)
        state = cache.linearize_bf16_kv(
            empty,
            empty,
            4,
            transforms,
            exact_head=1,
            exact_tail=1,
        )
        keys, values, _ = fixture(1)
        state = cache.advance_linear_state(state, keys, values, transforms)
        mx.eval(state.exact_head_keys, state.exact_head_values)

        self.assertEqual(cache.state_length(state), 1)
        self.assertEqual(cache.packed_history(state), 0)
        self.assertTrue(bool(mx.array_equal(state.exact_head_keys, keys).item()))
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
        state = cache.linearize_bf16_kv(
            empty,
            empty,
            8,
            transforms,
            exact_head=1,
            exact_tail=1,
        )
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
            exact_head=1,
            exact_tail=1,
        )
        actual_keys, actual_values = cache.dequantize_state(state, transforms)
        expected_keys, expected_values = cache.dequantize_state(expected, transforms)
        mx.eval(actual_keys, actual_values, expected_keys, expected_values)

        self.assertTrue(bool(mx.array_equal(actual_keys, expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_values, expected_values).item()))

    def test_state_storage_is_physical_and_tail_bounded(self) -> None:
        keys, values, _ = fixture()
        state = cache.compress_bf16_kv(keys, values, exact_head=1, exact_tail=1)
        mx.eval(
            state.packed_keys,
            state.key_norms,
            state.packed_values,
            state.value_norms,
        )

        self.assertEqual(cache.state_length(state), 19)
        self.assertEqual(cache.stored_bytes(state), 23_952)
        self.assertEqual(state.packed_keys.shape, (2, 17, 288))
        self.assertEqual(state.exact_head_keys.shape, (2, 1, 256))
        self.assertEqual(state.exact_keys.shape, (2, 1, 256))
        with self.assertRaisesRegex(reference.TurboQuantError, "exact head"):
            cache.compress_bf16_kv(
                keys,
                values,
                exact_head=cache.PRODUCTION_EXACT_HEAD_TOKENS + 1,
            )
        with self.assertRaisesRegex(reference.TurboQuantError, "exact tail"):
            cache.compress_bf16_kv(
                keys,
                values,
                exact_tail=cache.PRODUCTION_EXACT_TAIL_TOKENS + 1,
            )

    def test_production_head_and_tail_bound_packed_middle(self) -> None:
        transforms = cache.production_transforms()
        initial = (
            cache.PRODUCTION_EXACT_HEAD_TOKENS
            + cache.PRODUCTION_EXACT_TAIL_TOKENS
            + 7
        )
        keys, values, _ = fixture(initial)
        state = cache.linearize_bf16_kv(keys, values, 1024, transforms)
        more_keys, more_values, _ = fixture(19)
        state = cache.advance_linear_state(state, more_keys, more_values, transforms)
        expected = cache.compress_bf16_kv(
            mx.concatenate((keys, more_keys), axis=1),
            mx.concatenate((values, more_values), axis=1),
            transforms,
        )
        actual_keys, actual_values = cache.dequantize_state(state, transforms)
        expected_keys, expected_values = cache.dequantize_state(expected, transforms)
        mx.eval(actual_keys, actual_values, expected_keys, expected_values)

        self.assertEqual(
            state.exact_head_keys.shape[1],
            cache.PRODUCTION_EXACT_HEAD_TOKENS,
        )
        self.assertEqual(state.exact_keys.shape[1], cache.PRODUCTION_EXACT_TAIL_TOKENS)
        self.assertEqual(cache.packed_history(state), 26)
        self.assertTrue(bool(mx.array_equal(actual_keys, expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(actual_values, expected_values).item()))

    def test_exact_boundaries_release_large_bf16_source(self) -> None:
        gc.collect()
        mx.clear_cache()
        keys, values, _ = fixture(8192)
        mx.eval(keys, values)
        source_bytes = keys.nbytes + values.nbytes
        state = cache.compress_bf16_kv(keys, values)
        mx.eval(
            state.packed_keys,
            state.key_norms,
            state.packed_values,
            state.value_norms,
            state.exact_head_keys,
            state.exact_head_values,
            state.exact_keys,
            state.exact_values,
        )
        mx.synchronize()
        paired_active = mx.get_active_memory()

        del keys
        del values
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        released = paired_active - mx.get_active_memory()
        try:
            self.assertGreaterEqual(released, int(source_bytes * 0.90))
        finally:
            del state
            gc.collect()
            mx.clear_cache()

    def test_large_append_does_not_retain_bf16_update(self) -> None:
        gc.collect()
        mx.clear_cache()
        empty = mx.zeros((2, 0, 256), dtype=mx.bfloat16)
        state = cache.linearize_bf16_kv(empty, empty, 8192)
        keys, values, _ = fixture(8192)
        mx.eval(keys, values)
        source_bytes = keys.nbytes + values.nbytes
        state = cache.advance_linear_state(state, keys, values)
        mx.eval(
            state.packed_keys,
            state.key_norms,
            state.packed_values,
            state.value_norms,
            state.exact_head_keys,
            state.exact_head_values,
            state.exact_keys,
            state.exact_values,
        )
        mx.synchronize()
        paired_active = mx.get_active_memory()

        del keys
        del values
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        released = paired_active - mx.get_active_memory()
        try:
            self.assertGreaterEqual(released, int(source_bytes * 0.90))
        finally:
            del state
            gc.collect()
            mx.clear_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)

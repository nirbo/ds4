#!/usr/bin/env python3
"""Aliasing, ordering, validation, and allocation tests for linear K/V cache."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_linear_cache as linear_cache


class MLXLinearCacheTest(unittest.TestCase):
    def test_packed_mse5_append_updates_four_aliased_buffers(self) -> None:
        packed_keys = mx.full((2, 12, 8), 255, dtype=mx.uint8)
        key_norms = mx.full((2, 12, 1), -3, dtype=mx.bfloat16)
        packed_values = mx.full((2, 12, 8), 254, dtype=mx.uint8)
        value_norms = mx.full((2, 12, 1), -4, dtype=mx.bfloat16)
        key_update = mx.arange(48, dtype=mx.uint8).reshape(2, 3, 8)
        key_norm_update = mx.arange(6, dtype=mx.float32).reshape(2, 3, 1).astype(mx.bfloat16)
        value_update = (key_update + mx.array(64, dtype=mx.uint8)).astype(mx.uint8)
        value_norm_update = (key_norm_update + 100).astype(mx.bfloat16)

        outputs = linear_cache.append_packed_mse5(
            packed_keys,
            key_norms,
            packed_values,
            value_norms,
            key_update,
            key_norm_update,
            value_update,
            value_norm_update,
            5,
        )
        mx.eval(*outputs)

        expected = (key_update, key_norm_update, value_update, value_norm_update)
        sources = (packed_keys, key_norms, packed_values, value_norms)
        for output, source, update in zip(outputs, sources, expected, strict=True):
            self.assertTrue(bool(mx.array_equal(output[:, 5:8], update).item()))
            self.assertTrue(bool(mx.array_equal(source[:, 5:8], update).item()))

    def test_transposed_paired_append_updates_aliased_buffers(self) -> None:
        keys = mx.full((2, 12, 4), -3, dtype=mx.bfloat16)
        values = mx.full((2, 12, 4), -4, dtype=mx.bfloat16)
        key_update = mx.arange(24, dtype=mx.float32).reshape(3, 2, 4).astype(mx.bfloat16)
        value_update = (key_update + 100).astype(mx.bfloat16)
        output_keys, output_values = linear_cache.append_kv_transposed_bf16(
            keys,
            values,
            key_update,
            value_update,
            5,
        )
        mx.eval(output_keys, output_values)

        expected_keys = mx.transpose(key_update, (1, 0, 2))
        expected_values = mx.transpose(value_update, (1, 0, 2))
        self.assertTrue(bool(mx.array_equal(output_keys[:, 5:8], expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(output_values[:, 5:8], expected_values).item()))
        self.assertTrue(bool(mx.array_equal(keys[:, 5:8], expected_keys).item()))
        self.assertTrue(bool(mx.array_equal(values[:, 5:8], expected_values).item()))
        self.assertTrue(bool(mx.all(output_keys[:, :5] == -3).item()))
        self.assertTrue(bool(mx.all(output_values[:, 8:] == -4).item()))

    def test_paired_append_updates_distinct_aliased_buffers(self) -> None:
        keys = mx.full((2, 12, 4), -3, dtype=mx.bfloat16)
        values = mx.full((2, 12, 4), -4, dtype=mx.bfloat16)
        key_update = mx.arange(16, dtype=mx.float32).reshape(2, 2, 4).astype(mx.bfloat16)
        value_update = (key_update + 100).astype(mx.bfloat16)
        output_keys, output_values = linear_cache.append_kv_bf16(
            keys,
            values,
            key_update,
            value_update,
            5,
        )
        mx.eval(output_keys, output_values)

        self.assertTrue(bool(mx.array_equal(output_keys[:, 5:7], key_update).item()))
        self.assertTrue(bool(mx.array_equal(output_values[:, 5:7], value_update).item()))
        self.assertTrue(bool(mx.array_equal(keys[:, 5:7], key_update).item()))
        self.assertTrue(bool(mx.array_equal(values[:, 5:7], value_update).item()))
        self.assertTrue(bool(mx.all(output_keys[:, :5] == -3).item()))
        self.assertTrue(bool(mx.all(output_values[:, 7:] == -4).item()))

    def test_block_append_aliases_without_touching_other_slots(self) -> None:
        cache = mx.full((2, 10, 4), -2, dtype=mx.bfloat16)
        update = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4).astype(mx.bfloat16)
        snapshot = cache
        result = linear_cache.append_bf16(cache, update, 4)
        mx.eval(result)

        self.assertTrue(bool(mx.array_equal(result[:, 4:7], update).item()))
        self.assertTrue(bool(mx.array_equal(snapshot[:, 4:7], update).item()))
        self.assertTrue(bool(mx.all(result[:, :4] == -2).item()))
        self.assertTrue(bool(mx.all(result[:, 7:] == -2).item()))

    def test_lazy_chain_preserves_append_order(self) -> None:
        cache = mx.zeros((2, 16, 8), dtype=mx.bfloat16)
        result = cache
        for offset in range(4):
            update = mx.full((2, 1, 8), offset + 1, dtype=mx.bfloat16)
            result = linear_cache.append_bf16(result, update, 6 + offset)
        mx.eval(result)

        expected = mx.array([1, 2, 3, 4], dtype=mx.bfloat16)
        self.assertTrue(bool(mx.array_equal(result[0, 6:10, 0], expected).item()))
        self.assertTrue(bool(mx.array_equal(cache[1, 6:10, 7], expected).item()))

    def test_append_does_not_allocate_a_second_cache(self) -> None:
        keys = mx.zeros((2, 65_536, 256), dtype=mx.bfloat16)
        values = mx.zeros((2, 65_536, 256), dtype=mx.bfloat16)
        key_update = mx.ones((2, 1, 256), dtype=mx.bfloat16)
        value_update = mx.full((2, 1, 256), 2, dtype=mx.bfloat16)
        keys, values = linear_cache.append_kv_bf16(
            keys,
            values,
            key_update,
            value_update,
            0,
        )
        mx.eval(keys, values)
        baseline = mx.get_active_memory()
        mx.reset_peak_memory()

        output_keys, output_values = linear_cache.append_kv_bf16(
            keys,
            values,
            key_update,
            value_update,
            32_768,
        )
        mx.eval(output_keys, output_values)
        active_delta = mx.get_active_memory() - baseline
        peak_delta = mx.get_peak_memory() - baseline

        self.assertLess(active_delta, 2 * 2**20)
        self.assertLess(peak_delta, 2 * 2**20)
        self.assertTrue(bool(mx.array_equal(keys[:, 32_768], key_update[:, 0]).item()))
        self.assertTrue(bool(mx.array_equal(values[:, 32_768], value_update[:, 0]).item()))

    def test_packed_append_does_not_allocate_second_buffers(self) -> None:
        shape = (2, 65_536, 160)
        norm_shape = (2, 65_536, 1)
        packed_keys = mx.zeros(shape, dtype=mx.uint8)
        packed_values = mx.zeros(shape, dtype=mx.uint8)
        key_norms = mx.zeros(norm_shape, dtype=mx.bfloat16)
        value_norms = mx.zeros(norm_shape, dtype=mx.bfloat16)
        packed_update = mx.ones((2, 1, 160), dtype=mx.uint8)
        norm_update = mx.ones((2, 1, 1), dtype=mx.bfloat16)
        buffers = linear_cache.append_packed_mse5(
            packed_keys,
            key_norms,
            packed_values,
            value_norms,
            packed_update,
            norm_update,
            packed_update,
            norm_update,
            0,
        )
        mx.eval(*buffers)
        baseline = mx.get_active_memory()
        mx.reset_peak_memory()

        outputs = linear_cache.append_packed_mse5(
            *buffers,
            packed_update,
            norm_update,
            packed_update,
            norm_update,
            32_768,
        )
        mx.eval(*outputs)
        active_delta = mx.get_active_memory() - baseline
        peak_delta = mx.get_peak_memory() - baseline

        self.assertLess(active_delta, 2 * 2**20)
        self.assertLess(peak_delta, 2 * 2**20)
        self.assertTrue(bool(mx.all(packed_keys[:, 32_768] == 1).item()))
        self.assertTrue(bool(mx.all(key_norms[:, 32_768] == 1).item()))

    def test_rejects_dtype_shape_and_range_drift(self) -> None:
        cache = mx.zeros((2, 8, 4), dtype=mx.bfloat16)
        with self.assertRaisesRegex(ValueError, "BF16"):
            linear_cache.append_bf16(cache, mx.zeros((2, 1, 4)), 0)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            linear_cache.append_bf16(
                cache,
                mx.zeros((1, 1, 4), dtype=mx.bfloat16),
                0,
            )
        with self.assertRaisesRegex(ValueError, "outside capacity"):
            linear_cache.append_bf16(
                cache,
                mx.zeros((2, 3, 4), dtype=mx.bfloat16),
                6,
            )

        packed = mx.zeros((2, 8, 4), dtype=mx.uint8)
        norms = mx.zeros((2, 8, 1), dtype=mx.bfloat16)
        packed_update = mx.zeros((2, 1, 4), dtype=mx.uint8)
        norm_update = mx.zeros((2, 1, 1), dtype=mx.bfloat16)
        with self.assertRaisesRegex(ValueError, "UINT8"):
            linear_cache.append_packed_mse5(
                packed.astype(mx.int32),
                norms,
                packed,
                norms,
                packed_update,
                norm_update,
                packed_update,
                norm_update,
                0,
            )
        with self.assertRaisesRegex(ValueError, "outside capacity"):
            linear_cache.append_packed_mse5(
                packed,
                norms,
                packed,
                norms,
                mx.zeros((2, 3, 4), dtype=mx.uint8),
                mx.zeros((2, 3, 1), dtype=mx.bfloat16),
                mx.zeros((2, 3, 4), dtype=mx.uint8),
                mx.zeros((2, 3, 1), dtype=mx.bfloat16),
                6,
            )

        values = mx.zeros((2, 8, 4), dtype=mx.bfloat16)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            linear_cache.append_kv_transposed_bf16(
                cache,
                values,
                mx.zeros((3, 3, 4), dtype=mx.bfloat16),
                mx.zeros((3, 3, 4), dtype=mx.bfloat16),
                0,
            )
        with self.assertRaisesRegex(ValueError, "outside capacity"):
            linear_cache.append_kv_transposed_bf16(
                cache,
                values,
                mx.zeros((3, 2, 4), dtype=mx.bfloat16),
                mx.zeros((3, 2, 4), dtype=mx.bfloat16),
                6,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

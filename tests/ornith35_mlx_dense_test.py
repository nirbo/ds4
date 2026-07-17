#!/usr/bin/env python3
"""Bitwise checks for Ornith-35 token-tiled BF16 projections."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_dense as dense
from ornith35_moe_reference import MoEError


def reference(weight: mx.array, vectors: mx.array) -> mx.array:
    return mx.vmap(lambda vector: mx.matmul(weight, vector))(vectors)


class MLXDenseTest(unittest.TestCase):
    def test_token_tiles_match_authoritative_bf16_gemv(self) -> None:
        mx.random.seed(20260717)
        weight = mx.random.normal((64, 2048), dtype=mx.float32).astype(mx.bfloat16)
        vectors = mx.random.normal((9, 2048), dtype=mx.float32).astype(mx.bfloat16)
        expected = reference(weight, vectors)
        actual = [
            dense.token_tiled_matvec(weight, vectors, token_tile=tile)
            for tile in (1, 2, 4, 8)
        ]
        mx.eval(expected, *actual)
        for value in actual:
            self.assertTrue(bool(mx.array_equal(value, expected).item()))

    def test_output_projection_width_matches_authoritative_bf16_gemv(self) -> None:
        mx.random.seed(20260718)
        weight = mx.random.normal((32, 4096), dtype=mx.float32).astype(mx.bfloat16)
        vectors = mx.random.normal((7, 4096), dtype=mx.float32).astype(mx.bfloat16)
        expected = reference(weight, vectors)
        actual = dense.token_tiled_matvec(weight, vectors)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_rejects_dtype_shape_and_alignment_drift(self) -> None:
        weight = mx.zeros((4, 128), dtype=mx.bfloat16)
        vectors = mx.zeros((3, 128), dtype=mx.bfloat16)
        with self.assertRaisesRegex(MoEError, "BF16 matrix"):
            dense.token_tiled_matvec(weight.astype(mx.float32), vectors)
        with self.assertRaisesRegex(MoEError, "shape mismatch"):
            dense.token_tiled_matvec(weight, vectors[:, :64])
        with self.assertRaisesRegex(MoEError, "128-aligned"):
            dense.token_tiled_matvec(weight[:, :64], vectors[:, :64])


if __name__ == "__main__":
    unittest.main(verbosity=2)

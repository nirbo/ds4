#!/usr/bin/env python3
"""Metal-backed parity tests for the Ornith-35 TurboQuant composition."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ornith35" / "tools"))

import ornith35_mlx_turboquant as mlx_tq
import ornith35_turboquant_reference as reference


class MLXTurboQuantTest(unittest.TestCase):
    def test_calibrated_split_reassembles_and_scores_in_original_order(self) -> None:
        vectors = mx.array(
            [
                [math.sin((row + 1) * (column + 1) * 0.013) for column in range(256)]
                for row in range(3)
            ],
            dtype=mx.float32,
        )
        queries = mx.array(
            [
                [math.cos((row + 2) * (column + 1) * 0.017) for column in range(256)]
                for row in range(2)
            ],
            dtype=mx.float32,
        )
        split = mlx_tq.channel_split(256, tuple(range(0, 256, 2)))
        high_rotation = mlx_tq.haar_rotation(128, 801)
        low_rotation = mlx_tq.haar_rotation(128, 802)
        high_projection = mlx_tq.qjl_projection(128, 803)
        low_projection = mlx_tq.qjl_projection(128, 804)

        product = mlx_tq.quantize_split_product(
            vectors,
            split,
            4,
            3,
            high_rotation,
            low_rotation,
            high_projection,
            low_projection,
            norm_dtype=mx.float32,
        )
        reconstructed = mlx_tq.dequantize_split_product(
            product,
            high_rotation,
            low_rotation,
            high_projection,
            low_projection,
        )
        direct = mlx_tq.split_product_inner_products(
            queries,
            product,
            high_rotation,
            low_rotation,
            high_projection,
            low_projection,
        )
        expected = queries @ mx.swapaxes(reconstructed, -2, -1)

        mse = mlx_tq.quantize_split_mse(
            vectors,
            split,
            4,
            3,
            high_rotation,
            low_rotation,
            norm_dtype=mx.float32,
        )
        mse_reconstructed = mlx_tq.dequantize_split_mse(
            mse,
            high_rotation,
            low_rotation,
        )
        mse_direct = mlx_tq.split_mse_inner_products(
            queries,
            mse,
            high_rotation,
            low_rotation,
        )
        mse_expected = queries @ mx.swapaxes(mse_reconstructed, -2, -1)
        mx.eval(
            direct,
            expected,
            reconstructed,
            mse_reconstructed,
            mse_direct,
            mse_expected,
        )

        self.assertEqual(reconstructed.shape, vectors.shape)
        self.assertEqual(mse_reconstructed.shape, vectors.shape)
        self.assertLess(float(mx.max(mx.abs(direct - expected)).item()), 2e-5)
        self.assertLess(float(mx.max(mx.abs(mse_direct - mse_expected)).item()), 2e-5)

    def test_channel_split_rejects_aliases_and_missing_groups(self) -> None:
        with self.assertRaisesRegex(reference.TurboQuantError, "high-precision"):
            mlx_tq.channel_split(256, ())
        with self.assertRaisesRegex(reference.TurboQuantError, "high-precision"):
            mlx_tq.channel_split(256, (1, 1))
        with self.assertRaisesRegex(reference.TurboQuantError, "high-precision"):
            mlx_tq.channel_split(256, tuple(range(256)))

    def test_gaussian_qr_rotation_is_orthogonal_and_stable(self) -> None:
        rotation = mlx_tq.haar_rotation(128, 20260718)
        product = mx.swapaxes(rotation.matrix, -2, -1) @ rotation.matrix
        error = mx.max(mx.abs(product - mx.eye(128, dtype=mx.float32)))
        mx.eval(error)
        self.assertLess(float(error.item()), 2e-6)
        self.assertEqual(
            rotation.sha256,
            "0efe03fc87e4972f995cbe3ae3960b4e089949dd166fe8a6a9b0edd6162c0baa",
        )

    def test_mse_matches_scalar_authority_with_fp32_norm(self) -> None:
        dimension = 128
        rotation = mlx_tq.haar_rotation(dimension, 71)
        vector = tuple(math.sin(index * 0.071) * 0.25 for index in range(dimension))
        encoding = mlx_tq.quantize_mse(
            mx.array([vector], dtype=mx.float32),
            3,
            rotation,
            norm_dtype=mx.float32,
        )
        reconstructed = mlx_tq.dequantize_mse(encoding, rotation)
        mx.eval(encoding.indices, reconstructed)
        matrix = rotation.matrix.tolist()
        scalar = reference.quantize_mse(vector, 3, matrix)
        expected = reference.dequantize_mse(scalar, matrix)
        self.assertEqual(encoding.indices[0].tolist(), list(scalar.indices))
        for actual, target in zip(reconstructed[0].tolist(), expected):
            self.assertAlmostEqual(actual, target, delta=2e-6)

    def test_qjl_direct_scores_match_reconstructed_keys(self) -> None:
        dimension = 128
        rotation = mlx_tq.haar_rotation(dimension, 83)
        projection = mlx_tq.qjl_projection(dimension, 1083)
        keys = mx.array(
            [
                [math.sin((row + 1) * (column + 3) * 0.003) for column in range(dimension)]
                for row in range(5)
            ],
            dtype=mx.bfloat16,
        )
        queries = mx.array(
            [
                [math.cos((row + 2) * (column + 1) * 0.005) for column in range(dimension)]
                for row in range(3)
            ],
            dtype=mx.bfloat16,
        )
        encoding = mlx_tq.quantize_product(keys, 3, rotation, projection)
        direct = mlx_tq.product_inner_products(
            queries,
            encoding,
            rotation,
            projection,
        )
        reconstructed = mlx_tq.dequantize_product(encoding, rotation, projection)
        expected = queries.astype(mx.float32) @ mx.swapaxes(reconstructed, -2, -1)
        mx.eval(direct, expected)
        difference = mx.max(mx.abs(direct - expected))
        mx.eval(difference)
        self.assertLess(float(difference.item()), 3e-5)

    def test_k9_mse_uses_uint16_indices_and_matches_scalar_authority(self) -> None:
        dimension = 256
        rotation = mlx_tq.haar_rotation(dimension, 79)
        vector = tuple(math.sin(index * 0.037) * 0.25 for index in range(dimension))
        encoding = mlx_tq.quantize_mse(
            mx.array([vector], dtype=mx.float32),
            9,
            rotation,
            norm_dtype=mx.float32,
        )
        reconstructed = mlx_tq.dequantize_mse(encoding, rotation)
        mx.eval(encoding.indices, reconstructed)
        matrix = rotation.matrix.tolist()
        scalar = reference.quantize_mse(vector, 9, matrix)
        expected = reference.dequantize_mse(scalar, matrix)
        self.assertEqual(encoding.indices.dtype, mx.uint16)
        self.assertEqual(encoding.indices[0].tolist(), list(scalar.indices))
        self.assertLess(
            max(abs(actual - target) for actual, target in zip(reconstructed[0].tolist(), expected)),
            2e-6,
        )

    def test_bf16_norm_storage_is_explicitly_lossy(self) -> None:
        rotation = mlx_tq.haar_rotation(128, 97)
        vector = mx.array(
            [[math.sin(index * 0.13) * 0.12345 for index in range(128)]],
            dtype=mx.bfloat16,
        )
        fp32 = mlx_tq.quantize_mse(vector, 4, rotation, norm_dtype=mx.float32)
        bf16 = mlx_tq.quantize_mse(vector, 4, rotation, norm_dtype=mx.bfloat16)
        self.assertEqual(fp32.indices.tolist(), bf16.indices.tolist())
        self.assertEqual(bf16.norms.dtype, mx.bfloat16)
        self.assertNotEqual(
            fp32.norms.astype(mx.float32).tolist(),
            bf16.norms.astype(mx.float32).tolist(),
        )

    def test_rejects_zero_vectors_and_bad_geometry(self) -> None:
        rotation = mlx_tq.haar_rotation(128, 101)
        with self.assertRaisesRegex(reference.TurboQuantError, "zero vector"):
            mlx_tq.quantize_mse(mx.zeros((1, 128)), 3, rotation)
        with self.assertRaisesRegex(reference.TurboQuantError, "geometry mismatch"):
            mlx_tq.quantize_mse(mx.ones((1, 256)), 3, rotation)


if __name__ == "__main__":
    unittest.main(verbosity=2)

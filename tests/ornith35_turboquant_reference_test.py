#!/usr/bin/env python3
"""Tests for the dependency-free Ornith-35 TurboQuant authority."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ornith35" / "tools"))

import ornith35_turboquant_reference as tq


def identity(dimension: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )


class TurboQuantReferenceTest(unittest.TestCase):
    def test_exact_target_codebooks_and_distortion(self) -> None:
        one = tq.codebook(256, 1)
        four = tq.codebook(256, 4)
        five = tq.codebook(256, 5)
        self.assertEqual(one.centroids, (-0.049916507721605746, 0.049916507721605746))
        self.assertAlmostEqual(four.centroids[0], -0.1693834023763135)
        self.assertAlmostEqual(four.centroids[-1], 0.1693834023763135)
        self.assertAlmostEqual(four.expected_total_mse, 0.009407533529022872)
        self.assertAlmostEqual(five.expected_total_mse, 0.00247794223995182)
        self.assertEqual(len(five.centroids), 32)
        self.assertAlmostEqual(five.centroids[0], -five.centroids[-1])
        self.assertEqual(len(four.boundaries), 17)
        self.assertTrue(all(a < b for a, b in zip(four.boundaries, four.boundaries[1:])))

    def test_mse_quantization_uses_spherical_codebook(self) -> None:
        dimension = 128
        rotation = identity(dimension)
        vector = tuple(1.0 if index == 0 else 0.0 for index in range(dimension))
        encoded = tq.quantize_mse(vector, 3, rotation)
        self.assertEqual(encoded.indices[0], 7)
        self.assertEqual(encoded.indices[1:], (4,) * (dimension - 1))
        reconstructed = tq.dequantize_mse(encoded, rotation)
        self.assertAlmostEqual(reconstructed[0], tq.codebook(128, 3).centroids[-1])
        self.assertAlmostEqual(reconstructed[1], tq.codebook(128, 3).centroids[4])

    def test_qjl_direct_score_matches_reconstructed_vector(self) -> None:
        dimension = 128
        rotation = identity(dimension)
        projection = tuple(
            tuple(
                1.0 if row == column else (0.125 if (row + column) % 17 == 0 else 0.0)
                for column in range(dimension)
            )
            for row in range(dimension)
        )
        vector = tuple(math.sin(index * 0.17) * 0.3 for index in range(dimension))
        query = tuple(math.cos(index * 0.11) * 0.2 for index in range(dimension))
        encoded = tq.quantize_product(vector, 3, rotation, projection)
        reconstructed = tq.dequantize_product(encoded, rotation, projection)
        direct = tq.product_inner_product(query, encoded, rotation, projection)
        self.assertAlmostEqual(direct, tq.dot(query, reconstructed), places=12)

    def test_version_stable_gaussian_stream(self) -> None:
        matrix = tq.gaussian_matrix(2, 3, 20260718)
        expected = (
            (-0.31822156567314974, 0.08454975091451954, -0.022414314950703167),
            (1.4744643123642376, 0.552951890517841, -1.2159145454749702),
        )
        for actual_row, expected_row in zip(matrix, expected):
            for actual, target in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, target, places=15)

    def test_physical_cache_accounting(self) -> None:
        bf16_vector = 256 * 2
        k3q = tq.product_vector_bytes(256, 3)
        v3 = tq.mse_vector_bytes(256, 3)
        k4 = tq.mse_vector_bytes(256, 4)
        k5 = tq.mse_vector_bytes(256, 5)
        self.assertEqual((k3q, v3, k4, k5), (100, 98, 130, 162))
        native = tq.cache_payload_bytes(262_144, 10, 2, bf16_vector, bf16_vector)
        aggressive = tq.cache_payload_bytes(262_144, 10, 2, k3q, v3)
        conservative = tq.cache_payload_bytes(262_144, 10, 2, k4, v3)
        self.assertEqual(native, 5 * 2**30)
        self.assertAlmostEqual(aggressive / 2**30, 0.966796875)
        self.assertAlmostEqual(conservative / 2**30, 1.11328125)
        mixed = tq.split_product_vector_bytes((128, 128), (4, 3))
        mixed += tq.split_mse_vector_bytes((128, 128), (4, 3))
        self.assertEqual(mixed, 236)

    def test_rejects_unsupported_or_invalid_payloads(self) -> None:
        with self.assertRaisesRegex(tq.TurboQuantError, "unsupported"):
            tq.codebook(64, 3)
        with self.assertRaisesRegex(tq.TurboQuantError, "zero vector"):
            tq.quantize_mse((0.0,) * 128, 3, identity(128))
        with self.assertRaisesRegex(tq.TurboQuantError, "bit width"):
            tq.product_vector_bytes(256, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

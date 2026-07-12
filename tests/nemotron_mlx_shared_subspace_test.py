#!/usr/bin/env python3
"""Tests for paired-expert shared-subspace screening helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_shared_subspace import (  # noqa: E402
    projected_pair_bytes,
    input_union_decompose_pair,
    randomized_svd,
    reconstruct_pair,
    reconstruct_input_union,
    reconstruct_union,
    select_functional_pairs,
    union_decompose_pair,
)


class SharedSubspaceTest(unittest.TestCase):
    def test_pair_selection_is_disjoint_and_similarity_ordered(self) -> None:
        counts = np.zeros((5, 5), dtype=np.uint32)
        sums = np.zeros((5, 5), dtype=np.float32)
        for first, second, cosine in ((0, 1, 0.9), (0, 2, 0.8), (2, 3, 0.7)):
            counts[first, second] = counts[second, first] = 3
            sums[first, second] = sums[second, first] = 3 * cosine
        pairs = select_functional_pairs(counts, sums, [0, 1, 2, 3], 2, 2)
        self.assertEqual([row["experts"] for row in pairs], [[0, 1], [2, 3]])

    def test_randomized_svd_recovers_exact_low_rank_matrix(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.standard_normal((12, 2), dtype=np.float32) @ rng.standard_normal(
            (2, 9), dtype=np.float32
        )
        u, singular, v = randomized_svd(matrix, 2, 9)
        np.testing.assert_allclose((u * singular) @ v, matrix, rtol=2e-5, atol=2e-5)

    def test_pair_reconstruction_is_exact_when_delta_fits_rank(self) -> None:
        rng = np.random.default_rng(11)
        base = rng.standard_normal((10, 8), dtype=np.float32)
        delta = rng.standard_normal((10, 2), dtype=np.float32) @ rng.standard_normal(
            (2, 8), dtype=np.float32
        )
        first, second, _ = reconstruct_pair(base + delta, base - delta, 3.0, 1.0, 2, 13)
        np.testing.assert_allclose(first, base + delta, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(second, base - delta, rtol=3e-5, atol=3e-5)

    def test_storage_projection_counts_one_prototype_and_one_residual(self) -> None:
        result = projected_pair_bytes(100, 80, 8, 4.5, 16.0)
        expected = 100 * 80 * 4.5 / 8 + 8 * (100 + 80) * 16 / 8
        self.assertEqual(result["proposed_bytes"], expected)
        self.assertLess(result["ratio"], 1.0)

    def test_union_basis_recovers_matrices_with_shared_column_space(self) -> None:
        rng = np.random.default_rng(19)
        basis = rng.standard_normal((10, 2), dtype=np.float32)
        first = basis @ rng.standard_normal((2, 7), dtype=np.float32)
        second = basis @ rng.standard_normal((2, 7), dtype=np.float32)
        decomposition = union_decompose_pair(first, second, 2, 23)
        actual_first, actual_second, _ = reconstruct_union(decomposition, 2)
        np.testing.assert_allclose(actual_first, first, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(actual_second, second, rtol=3e-5, atol=3e-5)

    def test_input_union_recovers_matrices_with_shared_row_space(self) -> None:
        rng = np.random.default_rng(29)
        basis = rng.standard_normal((2, 7), dtype=np.float32)
        first = rng.standard_normal((10, 2), dtype=np.float32) @ basis
        second = rng.standard_normal((10, 2), dtype=np.float32) @ basis
        decomposition = input_union_decompose_pair(first, second, 2, 31)
        actual_first, actual_second, _ = reconstruct_input_union(decomposition, 2)
        np.testing.assert_allclose(actual_first, first, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(actual_second, second, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

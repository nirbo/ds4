#!/usr/bin/env python3
"""Tests for dense functional expert proxy fitting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_dense_proxy import fit_dense_mapping  # noqa: E402


class DenseProxyTest(unittest.TestCase):
    def test_mapping_uses_selected_context_and_preserves_retained_experts(self) -> None:
        projected = np.zeros((3, 4, 2), dtype=np.float32)
        projected[:, 0] = [[1, 0], [9, 0], [9, 0]]
        projected[:, 2] = [[8, 0], [2, 0], [2, 0]]
        projected[:, 1] = [[1.1, 0], [2.1, 0], [2.1, 0]]
        projected[:, 3] = [[7.9, 0], [8.9, 0], [8.9, 0]]
        indices = np.array([[1, 0], [3, 2], [3, 2]], dtype=np.int64)
        scores = np.ones_like(indices, dtype=np.float32)
        mapping, rows = fit_dense_mapping(projected, indices, scores, [0, 2])
        self.assertEqual(mapping, [0, 0, 2, 0])
        self.assertEqual([row["selected_tokens"] for row in rows], [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_gefen import (  # noqa: E402
    GefenMLX,
    find_period,
    nearest_codebook_indices,
    weighted_lloyd_codebook,
)


class GefenMLXTest(unittest.TestCase):
    def test_codebook_is_sorted_and_nearest_lookup_is_exact(self):
        counts = np.ones((4096,), dtype=np.int64)
        codebook = weighted_lloyd_codebook(counts)
        self.assertEqual(codebook.shape, (256,))
        self.assertEqual((codebook[0], codebook[-1]), (-1.0, 1.0))
        self.assertTrue(np.all(codebook[1:] >= codebook[:-1]))
        values = mx.array([-1.0, -0.2, 0.2, 1.0])
        indices = nearest_codebook_indices(mx.array(codebook), values).tolist()
        expected = [int(np.argmin(np.abs(codebook - value))) for value in values.tolist()]
        self.assertEqual(indices, expected)

    def test_period_selection_returns_a_bounded_divisor(self):
        values = np.tile(np.linspace(0.01, 1.0, 32), 16) ** 2
        period = find_period(values)
        self.assertEqual(values.size % period, 0)
        self.assertLessEqual(period, 1024)

    def test_update_uses_uint8_momentum_and_shared_second_moment(self):
        parameter = mx.zeros((64, 32))
        gradient = mx.array(np.tile(np.linspace(-1, 1, 32), (64, 1)).astype(np.float32))
        optimizer = GefenMLX(1e-3)
        updated = optimizer.update({"weight": parameter}, {"weight": gradient})
        state = optimizer.states["weight"]
        self.assertEqual(state.momentum_indices.dtype, mx.uint8)
        self.assertLessEqual(state.vmean.size, gradient.size)
        self.assertLess(optimizer.state_bytes(), gradient.nbytes * 2)
        self.assertGreater(float(mx.max(mx.abs(updated["weight"]))), 0.0)


if __name__ == "__main__":
    unittest.main()

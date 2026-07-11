#!/usr/bin/env python3
"""Tests for layerwise affine distillation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_layer_distill import fit_affine  # noqa: E402


class LayerDistillTest(unittest.TestCase):
    def test_affine_fit_recovers_channelwise_transform(self) -> None:
        candidate = np.array([[1.0, 2.0], [2.0, 0.0], [3.0, -2.0]], dtype=np.float32)
        teacher = candidate * np.array([2.0, -0.5]) + np.array([3.0, 4.0])
        scale, bias = fit_affine(candidate, teacher, 0.0)
        np.testing.assert_allclose(scale, [2.0, -0.5], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(bias, [3.0, 4.0], rtol=1e-6, atol=1e-6)

    def test_ridge_shrinks_scale_toward_identity(self) -> None:
        candidate = np.array([[0.0], [1.0]], dtype=np.float32)
        teacher = candidate * 3.0
        unregularized, _ = fit_affine(candidate, teacher, 0.0)
        regularized, _ = fit_affine(candidate, teacher, 10.0)
        self.assertGreater(unregularized[0], regularized[0])
        self.assertGreater(regularized[0], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

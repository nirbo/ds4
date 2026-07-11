#!/usr/bin/env python3
"""Tests for layerwise affine distillation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_layer_distill import (  # noqa: E402
    apply_latent_relu2,
    apply_low_rank,
    fit_affine,
    fit_latent_relu2,
    fit_low_rank,
)


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

    def test_low_rank_fit_recovers_predictable_residual(self) -> None:
        hidden = np.array(
            [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        residual = np.column_stack((hidden[:, 0] * 3.0, hidden[:, 0] * -2.0)).astype(np.float32)
        correction = fit_low_rank(hidden, residual, rank=1, ridge=0.0)
        predicted = apply_low_rank(hidden, *correction)
        np.testing.assert_allclose(predicted, residual, rtol=1e-5, atol=1e-5)

    def test_latent_relu2_fits_nonlinear_residual(self) -> None:
        values = np.linspace(-2.0, 2.0, 17, dtype=np.float32)
        latent = np.column_stack((values, np.zeros_like(values))).astype(np.float32)
        residual = np.column_stack(
            (np.maximum(values, 0.0) ** 2, np.maximum(-values, 0.0) ** 2)
        ).astype(np.float32)
        correction = fit_latent_relu2(latent, residual, rank=1, ridge=0.0)
        predicted = apply_latent_relu2(latent, correction)
        np.testing.assert_allclose(predicted, residual, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for gradient-capable Nemotron BF16 fallback projections."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_kd_gradient_audit import NativeBF16Linear  # noqa: E402


class KDGradientAuditTest(unittest.TestCase):
    def test_native_bf16_projection_has_exact_equation_and_input_gradient(self) -> None:
        weight = mx.array([[1.0, -2.0], [0.5, 0.25]], dtype=mx.bfloat16)
        linear = NativeBF16Linear(SimpleNamespace(weight=weight))
        x = mx.array([[0.2, -0.3]], dtype=mx.float32)

        def loss(value):
            return mx.sum(mx.square(linear(value)))

        output = linear(x)
        value, gradient = mx.value_and_grad(loss)(x)
        mx.eval(output, value, gradient)
        expected = x @ weight.T.astype(mx.float32)
        self.assertEqual(output.tolist(), expected.tolist())
        self.assertGreater(float(mx.linalg.norm(gradient)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

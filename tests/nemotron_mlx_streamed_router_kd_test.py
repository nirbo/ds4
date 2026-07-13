#!/usr/bin/env python3
"""Focused tests for streamed Router KD loss and guarded update helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_streamed_router_kd import (  # noqa: E402
    atomic_npy,
    changed_rows,
    decode_e4m3fn,
    first_adam_step,
    kl_divergence,
)


class StreamedRouterKDTest(unittest.TestCase):
    def test_vector_e4m3fn_decode(self) -> None:
        encoded = mx.array([0x00, 0x01, 0x38, 0x3C, 0x7E, 0x80, 0xB8], dtype=mx.uint8)
        decoded = decode_e4m3fn(encoded)
        mx.eval(decoded)
        np.testing.assert_array_equal(
            np.asarray(decoded),
            np.array([0.0, 2**-9, 1.0, 1.5, 448.0, -0.0, -1.0], dtype=np.float32),
        )

    def test_kl_is_zero_for_identical_logits_and_has_gradient(self) -> None:
        teacher = mx.array([[[-1.0, 0.5, 2.0]]], dtype=mx.float32)
        student = mx.array([[[-0.8, 0.1, 1.4]]], dtype=mx.float32)
        identical = kl_divergence(teacher, teacher, 1.0)
        value, gradient = mx.value_and_grad(lambda logits: kl_divergence(teacher, logits, 1.0))(student)
        mx.eval(identical, value, gradient)
        self.assertAlmostEqual(float(identical), 0.0, places=6)
        self.assertGreater(float(value), 0.0)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertGreater(float(mx.linalg.norm(gradient)), 0.0)

    def test_bf16_update_changes_only_nonzero_gradient_rows(self) -> None:
        source = mx.array([[0.25, -0.5], [0.125, 0.375]], dtype=mx.bfloat16)
        gradient = mx.array([[0.0, 0.0], [1.0, -1.0]], dtype=mx.float32)
        candidate = first_adam_step(source, gradient, 1e-3)
        mx.eval(candidate)
        source_np = np.asarray(source.astype(mx.float32))
        candidate_np = np.asarray(candidate.astype(mx.float32))
        self.assertEqual(changed_rows(source_np, candidate_np), [1])
        self.assertEqual(source_np[0].tolist(), candidate_np[0].tolist())

    def test_atomic_npy_never_leaves_part_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "value.npy"
            expected = np.arange(6, dtype=np.float32).reshape(2, 3)
            atomic_npy(path, expected)
            np.testing.assert_array_equal(np.load(path), expected)
            self.assertFalse(path.with_name(path.name + ".part").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Synthetic tests for Nemotron mixed-precision MLX linear primitives."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_linear import (  # noqa: E402
    bf16_matvec,
    fp8_matvec,
    fp8_matvec_custom,
    nvfp4_matvec,
)
from nemotron_mlx_nvfp4 import nvfp4_matvec as nvfp4_matvec_custom  # noqa: E402
from nemotron_nvfp4 import decode_e4m3fn  # noqa: E402


class MLXLinearTest(unittest.TestCase):
    def test_fp8_matches_scalar_decode(self) -> None:
        rows = 7
        columns = 64
        options = [0x01, 0x20, 0x38, 0x3C, 0x40, 0x58, 0x70, 0xFE]
        encoded = [options[(index * 5 + 3) % len(options)] for index in range(rows * columns)]
        values = [math.sin(index * 0.13) * 0.4 for index in range(columns)]
        scale_value = 0.00390625
        weight = mx.array(encoded, dtype=mx.uint8).reshape(rows, columns)
        scale = mx.array([scale_value], dtype=mx.float32)
        vector = mx.array(values, dtype=mx.float32)
        actual = fp8_matvec(weight, scale, vector)
        custom = fp8_matvec_custom(weight, scale, vector)
        mx.eval(actual, custom)
        expected = [
            math.fsum(
                decode_e4m3fn(encoded[row * columns + column]) * values[column]
                for column in range(columns)
            )
            * scale_value
            for row in range(rows)
        ]
        error2 = math.fsum((left - right) ** 2 for left, right in zip(actual.tolist(), expected))
        reference2 = math.fsum(value * value for value in expected)
        self.assertLessEqual(math.sqrt(error2 / reference2), 2e-5)
        self.assertLessEqual(max(abs(left - right) for left, right in zip(actual.tolist(), custom.tolist())), 2e-5)

    def test_bf16_uses_mlx_matvec(self) -> None:
        weight = mx.array([[1.0, 2.0], [-3.0, 0.5]], dtype=mx.bfloat16)
        vector = mx.array([0.25, -2.0], dtype=mx.float32)
        actual = bf16_matvec(weight, vector)
        reference = weight @ vector
        mx.eval(actual, reference)
        self.assertEqual(actual.tolist(), [-3.75, -1.75])
        self.assertEqual(actual.tolist(), reference.tolist())

    def test_native_nvfp4_matches_custom_kernel(self) -> None:
        rows = 7
        columns = 64
        packed = mx.array([(index * 19 + 7) & 0xFF for index in range(rows * columns // 2)], dtype=mx.uint8).reshape(
            rows, columns // 2
        )
        scales = mx.full((rows, columns // 16), 0x38, dtype=mx.uint8)
        global_scale = mx.array([0.03125], dtype=mx.float32)
        vector = mx.array([math.cos(index * 0.07) for index in range(columns)], dtype=mx.float32)
        native = nvfp4_matvec(packed, scales, global_scale, vector)
        custom = nvfp4_matvec_custom(packed, scales, global_scale, vector)
        mx.eval(native, custom)
        self.assertLessEqual(max(abs(left - right) for left, right in zip(native.tolist(), custom.tolist())), 2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

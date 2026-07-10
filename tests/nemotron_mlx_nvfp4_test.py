#!/usr/bin/env python3
"""Synthetic MLX composition test for the packed NVFP4 kernel."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_nvfp4 import nvfp4_matvec  # noqa: E402
from nemotron_nvfp4 import decode_e2m1, decode_e4m3fn  # noqa: E402


class MLXNVFP4Test(unittest.TestCase):
    def test_matches_scalar_decode(self) -> None:
        rows = 7
        columns = 64
        packed_bytes = []
        for index in range(rows * columns // 2):
            low = (index * 5 + 1) & 15
            high = (index * 7 + 3) & 15
            packed_bytes.append(low | (high << 4))
        scale_options = [0x01, 0x20, 0x38, 0x3C, 0x40, 0x58, 0x70, 0x7E]
        scale_bytes = [scale_options[index % len(scale_options)] for index in range(rows * columns // 16)]
        values = [math.sin(index * 0.17) * 0.5 for index in range(columns)]
        packed = mx.array(packed_bytes, dtype=mx.uint8).reshape(rows, columns // 2)
        scales = mx.array(scale_bytes, dtype=mx.uint8).reshape(rows, columns // 16)
        global_scale = mx.array([0.03125], dtype=mx.float32)
        vector = mx.array(values, dtype=mx.float32)
        actual = nvfp4_matvec(packed, scales, global_scale, vector)
        mx.eval(actual)

        expected = []
        for row in range(rows):
            terms = []
            for column in range(columns):
                byte = packed_bytes[row * columns // 2 + column // 2]
                nibble = byte >> 4 if column & 1 else byte & 15
                scale = scale_bytes[row * columns // 16 + column // 16]
                terms.append(decode_e2m1(nibble) * decode_e4m3fn(scale) * 0.03125 * values[column])
            expected.append(math.fsum(terms))
        error2 = math.fsum((left - right) ** 2 for left, right in zip(actual.tolist(), expected))
        reference2 = math.fsum(value * value for value in expected)
        self.assertLessEqual(math.sqrt(error2 / reference2), 2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

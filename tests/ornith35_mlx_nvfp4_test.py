#!/usr/bin/env python3
"""Metal-backed synthetic checks for Ornith-35 NVFP4 composition."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))
import ornith35_nvfp4 as reference

MODULE_PATH = TOOLS / "ornith35_mlx_nvfp4.py"
SPEC = importlib.util.spec_from_file_location("ornith35_mlx_nvfp4", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MLXNVFP4Test(unittest.TestCase):
    def test_production_expert_shapes_compile(self) -> None:
        for rows, columns in ((512, 2048), (2048, 512)):
            packed = mx.zeros((rows, columns // 2), dtype=mx.uint8)
            scales = mx.full((rows, columns // 16), 0x38, dtype=mx.uint8)
            global_scale = mx.ones((1,), dtype=mx.float32)
            vector = mx.ones((columns,), dtype=mx.float32)
            output = MODULE.nvfp4_matvec(packed, scales, global_scale, vector)
            mx.eval(output)
            self.assertEqual(output.shape, (rows,))
            self.assertEqual(mx.max(mx.abs(output)).item(), 0.0)

    def test_matches_scalar_reference(self) -> None:
        packed_values = [
            low | ((low + 1) << 4)
            for low in range(0, 16, 2)
        ] + [0x22] * 8
        packed = mx.array(packed_values, dtype=mx.uint8).reshape(2, 8)
        scales = mx.array([0x38, 0x40], dtype=mx.uint8).reshape(2, 1)
        global_scale = mx.array([2.0], dtype=mx.float32)
        values = [math.sin(index * 0.17) for index in range(16)]
        vector = mx.array(values, dtype=mx.float32)
        output = MODULE.nvfp4_matvec(packed, scales, global_scale, vector)
        mx.eval(output)
        actual = output.tolist()

        expected = []
        for row in range(2):
            total = 0.0
            for column, value in enumerate(values):
                byte = packed_values[row * 8 + column // 2]
                nibble = byte >> 4 if column & 1 else byte & 0xF
                scale = reference.decode_e4m3fn([0x38, 0x40][row])
                total += reference.decode_e2m1(nibble) * scale / 2.0 * value
            expected.append(total)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=2e-6)

    def test_rejects_scale_shape_mismatch(self) -> None:
        packed = mx.zeros((2, 8), dtype=mx.uint8)
        scales = mx.zeros((2, 2), dtype=mx.uint8)
        global_scale = mx.ones((1,), dtype=mx.float32)
        vector = mx.ones((16,), dtype=mx.float32)
        with self.assertRaisesRegex(MODULE.NVFP4Error, "scale shape mismatch"):
            MODULE.nvfp4_matvec(packed, scales, global_scale, vector)

    def test_selected_experts_stay_batched_for_gate_and_down(self) -> None:
        packed_values = [0x11] * 16 + [0x22] * 16
        packed = mx.array(packed_values, dtype=mx.uint8).reshape(2, 2, 8)
        scales = mx.full((2, 2, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.ones((2,), dtype=mx.float32)
        selected = mx.array([1, 0], dtype=mx.uint32)
        shared_vector = mx.array(
            [math.sin(index * 0.17) for index in range(16)],
            dtype=mx.float32,
        )
        gate = MODULE.nvfp4_selected_matvec(
            packed,
            scales,
            globals_,
            selected,
            shared_vector,
            batched_input=False,
        )
        down_vectors = mx.stack([shared_vector, shared_vector * 0.25])
        down = MODULE.nvfp4_selected_matvec(
            packed,
            scales,
            globals_,
            selected,
            down_vectors,
            batched_input=True,
        )
        mx.eval(gate, down)

        vector_sum = math.fsum(shared_vector.tolist())
        for row, expected in zip(gate.tolist(), (vector_sum, vector_sum * 0.5)):
            self.assertAlmostEqual(row[0], expected, delta=2e-6)
            self.assertAlmostEqual(row[1], expected, delta=2e-6)
        for row, expected in zip(down.tolist(), (vector_sum, vector_sum * 0.125)):
            self.assertAlmostEqual(row[0], expected, delta=2e-6)
            self.assertAlmostEqual(row[1], expected, delta=2e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

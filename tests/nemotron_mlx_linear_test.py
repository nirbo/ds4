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
    ModelOptBF16Linear,
    ModelOptNVFP4Linear,
    bf16_batch_matmul,
    bf16_gather_matvec,
    bf16_matvec,
    bf16_switch_matmul,
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

    def test_bf16_batch_matches_individual_matvecs(self) -> None:
        rows = 7
        columns = 64
        weight = mx.array(
            [
                [math.sin(row * 0.17 + column * 0.11) for column in range(columns)]
                for row in range(rows)
            ],
            dtype=mx.bfloat16,
        )
        matrix = mx.array(
            [
                [math.cos(token * 0.23 + column * 0.07) for column in range(columns)]
                for token in range(4)
            ],
            dtype=mx.float32,
        )
        batched = bf16_batch_matmul(weight, matrix)
        individual = mx.stack([bf16_matvec(weight, matrix[token]) for token in range(4)])
        mx.eval(batched, individual)
        self.assertLessEqual(float(mx.max(mx.abs(batched - individual))), 1e-6)

    def test_bf16_gather_matches_selected_rows_without_copying_weights(self) -> None:
        weight = mx.arange(12 * 64, dtype=mx.float32).reshape(12, 64).astype(mx.bfloat16)
        vector = mx.linspace(-0.5, 0.75, 64, dtype=mx.float32)
        indices = mx.array([9, 1, 7, 3], dtype=mx.int32)
        actual = bf16_gather_matvec(weight, indices, vector)
        expected = bf16_matvec(weight[indices], vector)
        mx.eval(actual, expected)
        self.assertLessEqual(float(mx.max(mx.abs(actual - expected))), 1e-6)

    def test_bf16_linear_chunks_long_sequences(self) -> None:
        rows = 7
        columns = 64
        tokens = 35
        weight = mx.array(
            [
                [math.sin(row * 0.17 + column * 0.11) for column in range(columns)]
                for row in range(rows)
            ],
            dtype=mx.bfloat16,
        )
        matrix = mx.array(
            [
                [math.cos(token * 0.23 + column * 0.07) for column in range(columns)]
                for token in range(tokens)
            ],
            dtype=mx.float32,
        )
        actual = ModelOptBF16Linear(weight)(matrix.reshape(1, tokens, columns))
        reference = mx.stack([bf16_matvec(weight, matrix[token]) for token in range(tokens)])
        mx.eval(actual, reference)
        self.assertLessEqual(float(mx.max(mx.abs(actual.reshape(tokens, rows) - reference))), 1e-6)

    def test_bf16_switch_matches_selected_individual_matvecs(self) -> None:
        experts = 4
        rows = 7
        columns = 64
        selected = [3, 1]
        weight = mx.array(
            [
                [
                    [math.sin(expert * 0.31 + row * 0.17 + column * 0.11) for column in range(columns)]
                    for row in range(rows)
                ]
                for expert in range(experts)
            ],
            dtype=mx.bfloat16,
        )
        indices = mx.array(selected, dtype=mx.int32).reshape(1, 1, -1)
        shared = mx.array(
            [math.cos(column * 0.07) for column in range(columns)],
            dtype=mx.float32,
        ).reshape(1, 1, columns)
        shared_output = bf16_switch_matmul(weight, shared, indices)
        shared_reference = mx.stack(
            [bf16_matvec(weight[expert], shared.reshape(-1)) for expert in selected]
        ).reshape(1, 1, len(selected), rows)

        per_expert = mx.array(
            [
                [math.cos(slot * 0.23 + column * 0.07) for column in range(columns)]
                for slot in range(len(selected))
            ],
            dtype=mx.float32,
        ).reshape(1, 1, len(selected), columns)
        per_expert_output = bf16_switch_matmul(weight, per_expert, indices)
        per_expert_reference = mx.stack(
            [
                bf16_matvec(weight[expert], per_expert[0, 0, slot])
                for slot, expert in enumerate(selected)
            ]
        ).reshape(1, 1, len(selected), rows)
        mx.eval(shared_output, shared_reference, per_expert_output, per_expert_reference)
        self.assertLessEqual(float(mx.max(mx.abs(shared_output - shared_reference))), 1e-6)
        self.assertLessEqual(float(mx.max(mx.abs(per_expert_output - per_expert_reference))), 1e-6)

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
        self.assertLessEqual(
            max(abs(left - right) for left, right in zip(native.tolist(), custom.tolist())),
            2e-5,
        )

    def test_native_nvfp4_sequence_matches_individual_rows(self) -> None:
        rows = 7
        columns = 64
        packed = mx.array(
            [(index * 29 + 11) & 0xFF for index in range(rows * columns // 2)],
            dtype=mx.uint8,
        ).reshape(rows, columns // 2)
        scales = mx.full((rows, columns // 16), 0x38, dtype=mx.uint8)
        global_scale = mx.array([0.03125], dtype=mx.float32)
        vectors = mx.array(
            [[math.cos(index * 0.07 + token * 0.19) for index in range(columns)] for token in range(3)],
            dtype=mx.float32,
        )
        linear = ModelOptNVFP4Linear(packed, scales, global_scale)
        batched = linear(vectors.reshape(1, 3, columns))
        individual = mx.stack([linear(vectors[token]) for token in range(3)])
        mx.eval(batched, individual)
        difference = mx.abs(batched.reshape(3, rows) - individual)
        self.assertLessEqual(float(mx.max(difference)), 2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

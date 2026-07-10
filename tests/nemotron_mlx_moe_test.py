#!/usr/bin/env python3
"""Numerical tests for GPU-owned Nemotron NVFP4 selected-expert execution."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_moe import NVFP4ExpertMLP, NVFP4SwitchWeight, expert_mlp  # noqa: E402
from nemotron_nvfp4 import decode_e2m1, decode_e4m3fn  # noqa: E402


def make_weight(experts: int, rows: int, columns: int, salt: int) -> tuple[NVFP4SwitchWeight, list]:
    packed = []
    scales = []
    globals_ = []
    decoded = []
    scale_options = [0x20, 0x30, 0x38, 0x3C, 0x40, 0x48, 0x50, 0x58]
    for expert in range(experts):
        expert_packed = []
        expert_scales = []
        global_scale = 0.015625 * (expert + 1)
        matrix = []
        for row in range(rows):
            row_packed = []
            row_scales = []
            for pair in range(columns // 2):
                low = (pair * 3 + row + expert + salt) & 15
                high = (pair * 5 + row * 2 + expert + salt + 1) & 15
                row_packed.append(low | (high << 4))
            for block in range(columns // 16):
                row_scales.append(scale_options[(block + row + expert + salt) % len(scale_options)])
            expert_packed.extend(row_packed)
            expert_scales.extend(row_scales)
            values = []
            for column in range(columns):
                byte = row_packed[column // 2]
                nibble = byte >> 4 if column & 1 else byte & 15
                values.append(
                    decode_e2m1(nibble)
                    * decode_e4m3fn(row_scales[column // 16])
                    * global_scale
                )
            matrix.append(values)
        packed.extend(expert_packed)
        scales.extend(expert_scales)
        globals_.append(global_scale)
        decoded.append(matrix)
    return (
        NVFP4SwitchWeight(
            mx.array(packed, dtype=mx.uint8).reshape(experts, rows, columns // 2),
            mx.array(scales, dtype=mx.uint8).reshape(experts, rows, columns // 16),
            mx.array(globals_, dtype=mx.float32),
        ),
        decoded,
    )


class MLXMoETest(unittest.TestCase):
    def test_selected_expert_equation_matches_scalar_decode(self) -> None:
        experts = 5
        latent = 64
        intermediate = 32
        up, up_decoded = make_weight(experts, intermediate, latent, 1)
        down, down_decoded = make_weight(experts, latent, intermediate, 7)
        weights = NVFP4ExpertMLP(up=up, down=down)
        values = [math.sin(index * 0.19) * 0.3 for index in range(latent)]
        selected = [4, 1, 3]
        score_values = [0.2, 0.5, 0.3]
        indices = mx.array([[selected]], dtype=mx.uint32)
        scores = mx.array([[score_values]], dtype=mx.float32)
        x = mx.array(values, dtype=mx.float32).reshape(1, 1, latent)
        actual = expert_mlp(x, weights, indices, scores)
        mx.eval(actual)

        expected = [0.0] * latent
        for expert, score in zip(selected, score_values):
            hidden = [
                max(0.0, math.fsum(weight * value for weight, value in zip(row, values))) ** 2
                for row in up_decoded[expert]
            ]
            projected = [
                math.fsum(weight * value for weight, value in zip(row, hidden))
                for row in down_decoded[expert]
            ]
            expected = [left + score * right for left, right in zip(expected, projected)]

        actual_values = actual.reshape(-1).tolist()
        error2 = math.fsum((left - right) ** 2 for left, right in zip(actual_values, expected))
        reference2 = math.fsum(value * value for value in expected)
        self.assertLessEqual(math.sqrt(error2 / max(reference2, 1e-30)), 2e-5)
        self.assertLessEqual(max(abs(left - right) for left, right in zip(actual_values, expected)), 2e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

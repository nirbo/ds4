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
from nemotron_mlx_linear import ModelOptBF16Linear, ModelOptFP8Linear  # noqa: E402
from nemotron_mlx_moe import (  # noqa: E402
    NVFP4ExpertMLP,
    NVFP4SwitchWeight,
    expert_mlp,
    expert_outputs,
)
from nemotron_mlx_moe_layer import _compiled_bf16_bf16_fp8_fp8_tail  # noqa: E402
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
    def test_compiled_dominant_tail_matches_eager_across_sequence_shapes(self) -> None:
        experts = 5
        latent = 64
        intermediate = 32
        up, _ = make_weight(experts, intermediate, latent, 2)
        down, _ = make_weight(experts, latent, intermediate, 8)
        weights = NVFP4ExpertMLP(up=up, down=down)
        fc1_weight = mx.array(
            [
                [math.sin(row * 0.17 + column * 0.11) for column in range(latent)]
                for row in range(latent)
            ],
            dtype=mx.bfloat16,
        )
        fc2_weight = mx.array(
            [
                [math.cos(row * 0.13 + column * 0.07) for column in range(latent)]
                for row in range(latent)
            ],
            dtype=mx.bfloat16,
        )
        fp8_options = [0x01, 0x20, 0x38, 0x3C, 0x40, 0x58, 0x70, 0xFE]
        shared_up_weight = mx.array(
            [fp8_options[(index * 5 + 3) % len(fp8_options)] for index in range(latent * latent)],
            dtype=mx.uint8,
        ).reshape(latent, latent)
        shared_down_weight = mx.array(
            [fp8_options[(index * 3 + 1) % len(fp8_options)] for index in range(latent * latent)],
            dtype=mx.uint8,
        ).reshape(latent, latent)
        shared_up_scale = mx.array([0.00390625], dtype=mx.float32)
        shared_down_scale = mx.array([0.001953125], dtype=mx.float32)

        fc1 = ModelOptBF16Linear(fc1_weight)
        fc2 = ModelOptBF16Linear(fc2_weight)
        shared_up = ModelOptFP8Linear(shared_up_weight, shared_up_scale)
        shared_down = ModelOptFP8Linear(shared_down_weight, shared_down_scale)
        for tokens in (1, 3):
            x = mx.array(
                [
                    [math.sin(token * 0.23 + column * 0.09) * 0.1 for column in range(latent)]
                    for token in range(tokens)
                ],
                dtype=mx.float32,
            ).reshape(1, tokens, latent)
            hidden = x * mx.array(0.75, dtype=mx.float32)
            indices = mx.array(
                [[[4, 1, 3] if token % 2 == 0 else [0, 2, 4] for token in range(tokens)]],
                dtype=mx.uint32,
            )
            scores = mx.array(
                [[[0.2, 0.5, 0.3] for _ in range(tokens)]],
                dtype=mx.float32,
            )
            selected = expert_outputs(fc1(hidden), weights, indices)
            routed = fc2((selected * scores[..., None]).sum(axis=-2))
            shared_hidden = mx.square(
                mx.maximum(shared_up(hidden), mx.array(0.0, dtype=hidden.dtype))
            )
            expected = x + routed + shared_down(shared_hidden)
            actual = _compiled_bf16_bf16_fp8_fp8_tail(
                x,
                hidden,
                indices,
                scores,
                fc1_weight,
                fc2_weight,
                shared_up_weight,
                shared_up_scale,
                shared_down_weight,
                shared_down_scale,
                up.weight,
                up.scales,
                up.global_scales,
                down.weight,
                down.scales,
                down.global_scales,
            )
            mx.eval(actual, expected)
            self.assertLessEqual(float(mx.max(mx.abs(actual - expected))), 1e-6)

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

    def test_selected_expert_sequence_matches_individual_tokens(self) -> None:
        experts = 5
        latent = 64
        intermediate = 32
        up, _ = make_weight(experts, intermediate, latent, 3)
        down, _ = make_weight(experts, latent, intermediate, 9)
        weights = NVFP4ExpertMLP(up=up, down=down)
        x = mx.array(
            [
                [math.sin(index * 0.13 + token * 0.17) * 0.2 for index in range(latent)]
                for token in range(3)
            ],
            dtype=mx.float32,
        ).reshape(1, 3, latent)
        indices = mx.array([[[4, 1, 3], [0, 2, 4], [3, 0, 1]]], dtype=mx.uint32)
        scores = mx.array(
            [[[0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.25, 0.5, 0.25]]],
            dtype=mx.float32,
        )
        batched = expert_mlp(x, weights, indices, scores)
        individual = mx.concatenate(
            [
                expert_mlp(
                    x[:, token : token + 1],
                    weights,
                    indices[:, token : token + 1],
                    scores[:, token : token + 1],
                )
                for token in range(3)
            ],
            axis=1,
        )
        mx.eval(batched, individual)
        self.assertLessEqual(float(mx.max(mx.abs(batched - individual))), 2e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

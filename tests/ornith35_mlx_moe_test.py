#!/usr/bin/env python3
"""Packed scalar parity checks for the GPU-owned Ornith-35 MoE block."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_moe_reference as reference
import ornith35_mlx_moe as mlx_moe


def packed_weight(
    rows: int, columns: int, nibble: int, dequant_scale: float
) -> reference.PackedWeight:
    byte = nibble | (nibble << 4)
    return reference.PackedWeight(
        packed=tuple(tuple(byte for _ in range(columns // 2)) for _ in range(rows)),
        scales=tuple(tuple(0x38 for _ in range(columns // 16)) for _ in range(rows)),
        global_scale=1.0 / dequant_scale,
    )


def expert(config: reference.MoEConfig, index: int) -> reference.ExpertWeights:
    return reference.ExpertWeights(
        gate=packed_weight(config.intermediate_size, config.hidden_size, 1 + index % 4, 0.03),
        up=packed_weight(config.intermediate_size, config.hidden_size, 2 + index % 3, 0.025),
        down=packed_weight(config.hidden_size, config.intermediate_size, 1 + index % 5, 0.02),
    )


def make_fixture() -> tuple[reference.MoEConfig, reference.MoEWeights]:
    config = reference.MoEConfig(hidden_size=16, intermediate_size=16, num_experts=4, top_k=2)
    router = tuple(
        tuple(math.sin((row * config.hidden_size + column + 1) * 0.13) * 0.2 for column in range(16))
        for row in range(config.num_experts)
    )
    return config, reference.MoEWeights(
        router=router,
        experts=tuple(expert(config, index) for index in range(config.num_experts)),
        shared_expert=expert(config, 5),
        shared_gate=tuple(math.cos((index + 1) * 0.17) * 0.1 for index in range(16)),
    )


def arrays(weight: reference.PackedWeight) -> mlx_moe.NVFP4Arrays:
    return mlx_moe.NVFP4Arrays(
        packed=mx.array(weight.packed, dtype=mx.uint8),
        scales=mx.array(weight.scales, dtype=mx.uint8),
        global_scale=mx.array([weight.global_scale], dtype=mx.float32),
    )


def stack(experts: tuple[reference.ExpertWeights, ...], name: str) -> mlx_moe.NVFP4Stack:
    values = [arrays(getattr(expert_weights, name)) for expert_weights in experts]
    return mlx_moe.NVFP4Stack(
        packed=mx.stack([value.packed for value in values]),
        scales=mx.stack([value.scales for value in values]),
        global_scale=mx.concatenate([value.global_scale for value in values]),
    )


def mlx_weights(weights: reference.MoEWeights) -> mlx_moe.MLXMoEWeights:
    return mlx_moe.MLXMoEWeights(
        router=mx.array(weights.router, dtype=mx.float32),
        experts=mlx_moe.ExpertStack(
            gate=stack(weights.experts, "gate"),
            up=stack(weights.experts, "up"),
            down=stack(weights.experts, "down"),
        ),
        shared_expert=mlx_moe.ExpertArrays(
            gate=arrays(weights.shared_expert.gate),
            up=arrays(weights.shared_expert.up),
            down=arrays(weights.shared_expert.down),
        ),
        shared_gate=mx.array([weights.shared_gate], dtype=mx.float32),
    )


class MLXMoETest(unittest.TestCase):
    def test_token_batch_matches_independent_one_token_paths(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        hidden = mx.array(
            [
                [math.sin((token * config.hidden_size + index + 1) * 0.21) * 0.4 for index in range(config.hidden_size)]
                for token in range(3)
            ],
            dtype=mx.float32,
        )
        batched = mlx_moe.forward_batch(hidden, weights, config)
        independent = [mlx_moe.forward(vector, weights, config) for vector in hidden]
        expected_output = mx.stack([result.output for result in independent])
        expected_selected = mx.stack([result.selected_experts for result in independent])
        expected_routing = mx.stack([result.routing_weights for result in independent])
        mx.eval(
            batched.output,
            batched.selected_experts,
            batched.routing_weights,
            expected_output,
            expected_selected,
            expected_routing,
        )
        self.assertTrue(bool(mx.array_equal(batched.selected_experts, expected_selected).item()))
        self.assertLess(
            float(mx.max(mx.abs(batched.routing_weights - expected_routing)).item()),
            2e-6,
        )
        self.assertLess(float(mx.max(mx.abs(batched.output - expected_output)).item()), 2e-5)

    def test_paired_gate_up_matches_separate_dispatches(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        hidden = mx.array(
            [math.sin((index + 1) * 0.21) * 0.4 for index in range(config.hidden_size)],
            dtype=mx.float32,
        )
        paired = mlx_moe.forward(hidden, weights, config, paired_gate_up=True)
        separate = mlx_moe.forward(hidden, weights, config, paired_gate_up=False)
        mx.eval(
            paired.output,
            paired.selected_experts,
            paired.routing_weights,
            separate.output,
            separate.selected_experts,
            separate.routing_weights,
        )
        self.assertTrue(bool(mx.all(paired.selected_experts == separate.selected_experts).item()))
        self.assertEqual(
            float(mx.max(mx.abs(paired.routing_weights - separate.routing_weights)).item()),
            0.0,
        )
        self.assertEqual(float(mx.max(mx.abs(paired.output - separate.output)).item()), 0.0)

    def test_fused_routed_down_matches_materialized_reduction(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        hidden = mx.array(
            [math.cos((index + 1) * 0.19) * 0.3 for index in range(config.hidden_size)],
            dtype=mx.float32,
        )
        fused = mlx_moe.forward(hidden, weights, config, fused_routed_down=True)
        materialized = mlx_moe.forward(hidden, weights, config, fused_routed_down=False)
        mx.eval(fused.output, materialized.output)
        self.assertEqual(float(mx.max(mx.abs(fused.output - materialized.output)).item()), 0.0)

    def test_production_contract(self) -> None:
        config = mlx_moe.PRODUCTION_CONFIG
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.intermediate_size, 512)
        self.assertEqual(config.num_experts, 256)
        self.assertEqual(config.top_k, 8)

    def test_matches_packed_scalar_router_experts_and_shared_path(self) -> None:
        config, scalar_weights = make_fixture()
        hidden = [math.sin((index + 1) * 0.21) * 0.4 for index in range(config.hidden_size)]
        expected = reference.forward(hidden, scalar_weights, config)
        actual = mlx_moe.forward(
            mx.array(hidden, dtype=mx.float32),
            mlx_weights(scalar_weights),
            config,
        )
        mx.eval(actual.output, actual.selected_experts, actual.routing_weights)
        self.assertEqual(tuple(actual.selected_experts.tolist()), expected.selected_experts)
        for left, right in zip(actual.routing_weights.tolist(), expected.routing_weights):
            self.assertAlmostEqual(left, right, delta=2e-6)
        for left, right in zip(actual.output.tolist(), expected.output):
            self.assertAlmostEqual(left, right, delta=2e-5)

    def test_rejects_layer_outside_text_model(self) -> None:
        with self.assertRaisesRegex(reference.MoEError, "outside the Ornith text model"):
            mlx_moe.load_layer(Path("unused.safetensors"), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)

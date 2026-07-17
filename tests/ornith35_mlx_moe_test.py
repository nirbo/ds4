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
    router = mx.array(weights.router, dtype=mx.float32)
    shared_gate = mx.array([weights.shared_gate], dtype=mx.float32)
    return mlx_moe.MLXMoEWeights(
        router_shared=mx.concatenate((router, shared_gate), axis=0),
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
        self.assertTrue(bool(mx.array_equal(batched.routing_weights, expected_routing).item()))
        self.assertTrue(bool(mx.array_equal(batched.output, expected_output).item()))

    def test_token_tiled_shared_path_matches_batched_reference(self) -> None:
        config = reference.MoEConfig(
            hidden_size=2048,
            intermediate_size=512,
            num_experts=4,
            top_k=2,
        )

        def single(rows: int, columns: int, byte: int) -> mlx_moe.NVFP4Arrays:
            return mlx_moe.NVFP4Arrays(
                packed=mx.full((rows, columns // 2), byte, dtype=mx.uint8),
                scales=mx.full((rows, columns // 16), 0x38, dtype=mx.uint8),
                global_scale=mx.ones((1,), dtype=mx.float32),
            )

        def expert_stack(rows: int, columns: int, byte: int) -> mlx_moe.NVFP4Stack:
            return mlx_moe.NVFP4Stack(
                packed=mx.full(
                    (config.num_experts, rows, columns // 2),
                    byte,
                    dtype=mx.uint8,
                ),
                scales=mx.full(
                    (config.num_experts, rows, columns // 16),
                    0x38,
                    dtype=mx.uint8,
                ),
                global_scale=mx.ones((config.num_experts,), dtype=mx.float32),
            )

        mx.random.seed(20260717)
        weights = mlx_moe.MLXMoEWeights(
            router_shared=mx.random.normal(
                (config.num_experts + 1, config.hidden_size),
                dtype=mx.float32,
            ).astype(mx.bfloat16),
            experts=mlx_moe.ExpertStack(
                gate=expert_stack(config.intermediate_size, config.hidden_size, 0x12),
                up=expert_stack(config.intermediate_size, config.hidden_size, 0x23),
                down=expert_stack(config.hidden_size, config.intermediate_size, 0x34),
            ),
            shared_expert=mlx_moe.ExpertArrays(
                gate=single(config.intermediate_size, config.hidden_size, 0x21),
                up=single(config.intermediate_size, config.hidden_size, 0x32),
                down=single(config.hidden_size, config.intermediate_size, 0x43),
            ),
        )
        hidden = mx.random.normal((9, config.hidden_size), dtype=mx.float32).astype(
            mx.bfloat16
        )
        expected = mlx_moe.forward_batch(
            hidden,
            weights,
            config,
            token_tiled_shared=False,
        )
        actual = mlx_moe.forward_batch(hidden, weights, config)
        pairs = (
            (actual.output, expected.output),
            (actual.selected_experts, expected.selected_experts),
            (actual.routing_weights, expected.routing_weights),
        )
        mx.eval(*(array for pair in pairs for array in pair))
        for candidate, baseline in pairs:
            self.assertTrue(bool(mx.array_equal(candidate, baseline).item()))

    def test_batched_route_matches_independent_rows_bit_exactly(self) -> None:
        mx.random.seed(20260717)
        logits = mx.random.normal((128, 256), dtype=mx.float32) * 3.0
        selected, routing = mlx_moe._route_batch(logits, 8, mx.bfloat16)
        independent = [mlx_moe._route_token(row, 8, mx.bfloat16) for row in logits]
        expected_selected = mx.stack([item[0] for item in independent])
        expected_routing = mx.stack([item[1] for item in independent])
        mx.eval(selected, routing, expected_selected, expected_routing)
        self.assertTrue(bool(mx.array_equal(selected, expected_selected).item()))
        self.assertTrue(bool(mx.array_equal(routing, expected_routing).item()))

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

    def test_fused_bf16_gate_up_activation_matches_separate_dispatches(self) -> None:
        config, scalar_weights = make_fixture()
        base = mlx_weights(scalar_weights)
        weights = mlx_moe.MLXMoEWeights(
            router_shared=base.router_shared.astype(mx.bfloat16),
            experts=base.experts,
            shared_expert=base.shared_expert,
        )
        hidden = mx.array(
            [math.sin((index + 1) * 0.21) * 0.4 for index in range(config.hidden_size)],
            dtype=mx.bfloat16,
        )
        fused = mlx_moe.forward(hidden, weights, config, paired_gate_up=True)
        separate = mlx_moe.forward(hidden, weights, config, paired_gate_up=False)
        mx.eval(
            fused.output,
            fused.selected_experts,
            fused.routing_weights,
            separate.output,
            separate.selected_experts,
            separate.routing_weights,
        )
        self.assertTrue(bool(mx.array_equal(fused.output, separate.output).item()))
        self.assertTrue(
            bool(mx.array_equal(fused.selected_experts, separate.selected_experts).item())
        )
        self.assertTrue(bool(mx.array_equal(fused.routing_weights, separate.routing_weights).item()))

    def test_combined_router_shared_gate_matches_split_projection(self) -> None:
        config, scalar_weights = make_fixture()
        base = mlx_weights(scalar_weights)
        weights = mlx_moe.MLXMoEWeights(
            router_shared=base.router_shared.astype(mx.bfloat16),
            experts=base.experts,
            shared_expert=base.shared_expert,
        )
        hidden = mx.array(
            [math.cos((index + 1) * 0.23) * 0.35 for index in range(config.hidden_size)],
            dtype=mx.bfloat16,
        )
        combined = mlx_moe.forward(hidden, weights, config, fused_shared_gate=True)
        split = mlx_moe.forward(hidden, weights, config, fused_shared_gate=False)
        batched_combined = mlx_moe.forward_batch(
            hidden[None, :], weights, config, fused_shared_gate=True
        )
        batched_split = mlx_moe.forward_batch(
            hidden[None, :], weights, config, fused_shared_gate=False
        )
        pairs = (
            (combined.output, split.output),
            (combined.selected_experts, split.selected_experts),
            (combined.routing_weights, split.routing_weights),
            (batched_combined.output, batched_split.output),
            (batched_combined.selected_experts, batched_split.selected_experts),
            (batched_combined.routing_weights, batched_split.routing_weights),
        )
        mx.eval(*(array for pair in pairs for array in pair))
        for actual, expected in pairs:
            self.assertTrue(bool(mx.array_equal(actual, expected).item()))

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

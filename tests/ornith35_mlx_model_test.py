#!/usr/bin/env python3
"""Text-model ordering and state checks for Ornith-35 MLX composition."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_attention as mlx_attention
import ornith35_mlx_layer as mlx_layer
import ornith35_mlx_layer_test as layer_fixture
import ornith35_mlx_model as model
import ornith35_mlx_moe_test as moe_fixture
import ornith35_moe_reference as moe_reference


def matrix(rows: int, columns: int, phase: float) -> mx.array:
    return mx.array(
        [
            [math.sin((row * columns + column + 1) * phase) * 0.11 for column in range(columns)]
            for row in range(rows)
        ],
        dtype=mx.float32,
    )


def make_fixture():
    gdn_config, scalar_gdn = layer_fixture.make_gdn_fixture()
    attention_config, scalar_attention = layer_fixture.make_attention_fixture()
    moe_config, scalar_moe = moe_fixture.make_fixture()
    gpu_moe = moe_fixture.mlx_weights(scalar_moe)
    input_norm = mx.array([math.sin((index + 1) * 0.07) * 0.1 for index in range(16)])
    post_norm = mx.array([math.cos((index + 1) * 0.09) * 0.1 for index in range(16)])
    norms = mlx_layer.LayerNorms(input_norm, post_norm)
    layers = (
        mlx_layer.GDNLayerWeights(
            token_mixer=layer_fixture.gdn_fixture.mlx_weights(scalar_gdn),
            moe=gpu_moe,
            norms=norms,
        ),
        mlx_layer.AttentionLayerWeights(
            token_mixer=layer_fixture.attention_fixture.mlx_weights(scalar_attention),
            moe=gpu_moe,
            norms=norms,
        ),
    )
    config = model.TextModelConfig(
        vocab_size=32,
        hidden_size=16,
        layer_types=(model.LAYER_GDN, model.LAYER_ATTENTION),
        gdn=gdn_config,
        attention=attention_config,
        moe=moe_config,
    )
    weights = model.TextModelWeights(
        embedding=matrix(config.vocab_size, config.hidden_size, 0.013),
        layers=layers,
        final_norm=mx.array([math.sin((index + 1) * 0.05) * 0.1 for index in range(16)]),
        lm_head=matrix(config.vocab_size, config.hidden_size, 0.017),
    )
    return config, weights


class MLXModelTest(unittest.TestCase):
    def test_parses_bounded_smoke_tokens(self) -> None:
        self.assertEqual(model.parse_token_ids(" 7,19 ", 32), (7, 19))
        with self.assertRaisesRegex(moe_reference.MoEError, "out of range"):
            model.parse_token_ids("32", 32)

    def test_production_contract_is_text_only_and_untied(self) -> None:
        config = model.PRODUCTION_CONFIG
        self.assertEqual(len(config.layer_types), 40)
        self.assertEqual(config.layer_types.count(model.LAYER_GDN), 30)
        self.assertEqual(config.layer_types.count(model.LAYER_ATTENTION), 10)
        self.assertEqual(config.vocab_size, 248_320)
        self.assertFalse(hasattr(model.TextModelWeights, "visual"))

    def test_two_token_model_matches_explicit_layer_order(self) -> None:
        config, weights = make_fixture()
        state = model.initial_state(weights, config)
        manual_states = list(state.layers)
        for position, token_id in enumerate((7, 19), start=1):
            hidden = weights.embedding[token_id]
            first = mlx_layer.forward_gdn(
                hidden,
                manual_states[0],
                weights.layers[0],
                config.gdn,
                config.moe,
            )
            second = mlx_layer.forward_attention(
                first.output,
                manual_states[1],
                weights.layers[1],
                config.attention,
                config.moe,
            )
            manual_states = [first.state, second.state]
            manual_hidden = mlx_layer.qwen_rms_norm(
                second.output,
                weights.final_norm,
                config.rms_norm_eps,
            )
            manual_logits = mx.matmul(weights.lm_head, manual_hidden)

            result = model.forward_token(token_id, state, weights, config)
            state = result.state
            mx.eval(result.logits, manual_logits)
            self.assertLess(mx.max(mx.abs(result.logits - manual_logits)).item(), 2e-6)
            self.assertEqual(state.position, position)
        self.assertEqual(mlx_attention.state_length(state.layers[1], config.attention), 2)

    def test_rejects_attention_cache_at_wrong_position(self) -> None:
        config, weights = make_fixture()
        state = model.initial_state(weights, config)
        broken = model.TextModelState(position=1, layers=state.layers)
        with self.assertRaisesRegex(moe_reference.MoEError, "attention position mismatch"):
            model.validate_state(broken, config)

    def test_rejects_layer_type_drift(self) -> None:
        config, weights = make_fixture()
        broken = model.TextModelWeights(
            embedding=weights.embedding,
            layers=(weights.layers[1], weights.layers[0]),
            final_norm=weights.final_norm,
            lm_head=weights.lm_head,
        )
        with self.assertRaisesRegex(moe_reference.MoEError, "type mismatch"):
            model.validate_weights(broken, config)


if __name__ == "__main__":
    unittest.main(verbosity=2)

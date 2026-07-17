#!/usr/bin/env python3
"""Text-model ordering and state checks for Ornith-35 MLX composition."""

from __future__ import annotations

import copy
import math
import sys
import unittest
from dataclasses import replace
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
import ornith35_gdn_reference as gdn_reference
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


def make_bf16_fixture():
    config, weights = make_fixture()

    def convert_moe(value):
        return replace(value, router_shared=value.router_shared.astype(mx.bfloat16))

    converted_layers = []
    for value in weights.layers:
        norms = mlx_layer.LayerNorms(
            value.norms.input_layernorm.astype(mx.bfloat16),
            value.norms.post_attention_layernorm.astype(mx.bfloat16),
        )
        if isinstance(value, mlx_layer.GDNLayerWeights):
            mixer = replace(
                value.token_mixer,
                **{
                    name: array.astype(mx.bfloat16)
                    for name, array in value.token_mixer.__dict__.items()
                },
            )
            converted_layers.append(
                replace(value, token_mixer=mixer, moe=convert_moe(value.moe), norms=norms)
            )
        else:
            mixer = replace(
                value.token_mixer,
                **{
                    name: array.astype(mx.bfloat16)
                    for name, array in value.token_mixer.__dict__.items()
                },
            )
            converted_layers.append(
                replace(value, token_mixer=mixer, moe=convert_moe(value.moe), norms=norms)
            )
    return config, replace(
        weights,
        embedding=weights.embedding.astype(mx.bfloat16),
        layers=tuple(converted_layers),
        final_norm=weights.final_norm.astype(mx.bfloat16),
        lm_head=weights.lm_head.astype(mx.bfloat16),
    )


class MLXModelTest(unittest.TestCase):
    def test_linear_prefill_session_matches_immutable_chunk_and_decode(self) -> None:
        config, weights = make_bf16_fixture()
        initial = model.initial_state(weights, config)
        expected = model.prefill_chunk(
            (7, 19, 11),
            initial,
            weights,
            config,
            use_steel=False,
        )
        model.evaluate_chunk_result(expected, diagnostics=True)

        linear = model.start_linear_decode_session(weights, initial, 5, config)
        actual = model.prefill_linear_session_chunk(
            (7, 19, 11),
            linear,
            project_logits=True,
            use_steel=False,
        )
        self.assertIsInstance(actual, model.TextModelChunkResult)
        self.assertTrue(bool(mx.array_equal(actual.logits, expected.logits).item()))
        self.assertTrue(bool(mx.array_equal(actual.hidden, expected.hidden).item()))
        self.assertEqual(linear.state.position, 3)
        for expected_state, actual_state in zip(expected.state.layers, linear.state.layers):
            if isinstance(expected_state, mlx_attention.MLXAttentionState):
                self.assertIsInstance(actual_state, mlx_attention.MLXLinearAttentionState)
                self.assertTrue(
                    bool(mx.array_equal(actual_state.keys[:, :3], expected_state.keys).item())
                )
                self.assertTrue(
                    bool(mx.array_equal(actual_state.values[:, :3], expected_state.values).item())
                )
            else:
                self.assertTrue(bool(mx.array_equal(actual_state.conv, expected_state.conv).item()))
                self.assertTrue(
                    bool(mx.array_equal(actual_state.recurrent, expected_state.recurrent).item())
                )

        expected_next = model.forward_token(5, expected.state, weights, config)
        model.evaluate_result(expected_next)
        actual_next = model.forward_linear_session_token(5, linear)
        self.assertTrue(bool(mx.array_equal(actual_next.logits, expected_next.logits).item()))
        self.assertEqual(linear.state.position, 4)

        expected_tail = model.forward_hidden_token(3, expected_next.state, weights, config)
        model.evaluate_transition(expected_tail)
        actual_tail = model.forward_linear_session_hidden_token(3, linear)
        self.assertTrue(bool(mx.array_equal(actual_tail.hidden, expected_tail.hidden).item()))
        self.assertEqual(linear.state.position, 5)
        with self.assertRaisesRegex(moe_reference.MoEError, "capacity exhausted"):
            model.prefill_linear_session_chunk(
                (2,),
                linear,
                project_logits=False,
                use_steel=False,
            )

    def test_state_only_prefill_matches_full_persistent_state(self) -> None:
        config, weights = make_bf16_fixture()
        initial = model.initial_state(weights, config)
        tokens = (7, 19, 11)
        expected = model.prefill_hidden_chunk(
            tokens,
            initial,
            weights,
            config,
            use_steel=False,
        )
        actual = model.prefill_state_chunk(
            tokens,
            initial,
            weights,
            config,
            use_steel=False,
        )
        model.evaluate_chunk_transition(expected)
        model.evaluate_state(actual)

        self.assertEqual(actual.position, expected.state.position)
        for actual_state, expected_state in zip(actual.layers, expected.state.layers):
            if isinstance(expected_state, mlx_attention.MLXAttentionState):
                self.assertIsInstance(actual_state, mlx_attention.MLXAttentionState)
                self.assertTrue(bool(mx.array_equal(actual_state.keys, expected_state.keys).item()))
                self.assertTrue(
                    bool(mx.array_equal(actual_state.values, expected_state.values).item())
                )
            else:
                self.assertTrue(bool(mx.array_equal(actual_state.conv, expected_state.conv).item()))
                self.assertTrue(
                    bool(mx.array_equal(actual_state.recurrent, expected_state.recurrent).item())
                )

        expected_linear = model.start_linear_decode_session(weights, initial, 4, config)
        actual_linear = model.start_linear_decode_session(weights, initial, 4, config)
        model.prefill_linear_session_chunk(
            tokens,
            expected_linear,
            project_logits=False,
            use_steel=False,
        )
        returned = model.prefill_linear_session_state_chunk(
            tokens,
            actual_linear,
            use_steel=False,
        )
        self.assertIs(returned, actual_linear.state)
        self.assertEqual(returned.position, expected_linear.state.position)
        for actual_state, expected_state in zip(
            returned.layers,
            expected_linear.state.layers,
        ):
            if isinstance(expected_state, mlx_attention.MLXLinearAttentionState):
                self.assertIsInstance(actual_state, mlx_attention.MLXLinearAttentionState)
                position = actual_state.position
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            actual_state.keys[:, :position],
                            expected_state.keys[:, :position],
                        ).item()
                    )
                )
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            actual_state.values[:, :position],
                            expected_state.values[:, :position],
                        ).item()
                    )
                )
            else:
                self.assertTrue(bool(mx.array_equal(actual_state.conv, expected_state.conv).item()))
                self.assertTrue(
                    bool(mx.array_equal(actual_state.recurrent, expected_state.recurrent).item())
                )

    def test_final_token_prefill_matches_full_chunk_result_and_state(self) -> None:
        config, weights = make_bf16_fixture()
        initial = model.initial_state(weights, config)
        tokens = (7, 19, 11)
        expected = model.prefill_chunk(
            tokens,
            initial,
            weights,
            config,
            use_steel=False,
        )
        actual = model.prefill_final_chunk(
            tokens,
            initial,
            weights,
            config,
            use_steel=False,
        )
        model.evaluate_chunk_result(expected, diagnostics=True)
        model.evaluate_result(actual)

        self.assertTrue(bool(mx.array_equal(actual.hidden, expected.hidden[-1]).item()))
        self.assertTrue(bool(mx.array_equal(actual.logits, expected.logits).item()))
        for actual_route, expected_route in zip(
            actual.selected_experts,
            expected.selected_experts,
        ):
            self.assertTrue(bool(mx.array_equal(actual_route, expected_route[-1]).item()))
        for actual_route, expected_route in zip(
            actual.routing_weights,
            expected.routing_weights,
        ):
            self.assertTrue(bool(mx.array_equal(actual_route, expected_route[-1]).item()))
        for actual_state, expected_state in zip(actual.state.layers, expected.state.layers):
            if isinstance(expected_state, mlx_attention.MLXAttentionState):
                self.assertIsInstance(actual_state, mlx_attention.MLXAttentionState)
                self.assertTrue(bool(mx.array_equal(actual_state.keys, expected_state.keys).item()))
                self.assertTrue(
                    bool(mx.array_equal(actual_state.values, expected_state.values).item())
                )
            else:
                self.assertTrue(bool(mx.array_equal(actual_state.conv, expected_state.conv).item()))
                self.assertTrue(
                    bool(mx.array_equal(actual_state.recurrent, expected_state.recurrent).item())
                )

        expected_linear = model.start_linear_decode_session(weights, initial, 4, config)
        actual_linear = model.start_linear_decode_session(weights, initial, 4, config)
        expected_result = model.prefill_linear_session_chunk(
            tokens,
            expected_linear,
            project_logits=True,
            use_steel=False,
        )
        actual_result = model.prefill_linear_session_final_chunk(
            tokens,
            actual_linear,
            use_steel=False,
        )
        self.assertTrue(bool(mx.array_equal(actual_result.hidden, expected_result.hidden[-1]).item()))
        self.assertTrue(bool(mx.array_equal(actual_result.logits, expected_result.logits).item()))
        for actual_state, expected_state in zip(
            actual_result.state.layers,
            expected_result.state.layers,
        ):
            if isinstance(expected_state, mlx_attention.MLXLinearAttentionState):
                self.assertIsInstance(actual_state, mlx_attention.MLXLinearAttentionState)
                position = actual_state.position
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            actual_state.keys[:, :position],
                            expected_state.keys[:, :position],
                        ).item()
                    )
                )
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            actual_state.values[:, :position],
                            expected_state.values[:, :position],
                        ).item()
                    )
                )
            else:
                self.assertTrue(bool(mx.array_equal(actual_state.conv, expected_state.conv).item()))
                self.assertTrue(
                    bool(mx.array_equal(actual_state.recurrent, expected_state.recurrent).item())
                )

    def test_linear_decode_session_matches_immutable_bf16_path(self) -> None:
        config, weights = make_bf16_fixture()
        state = model.initial_state(weights, config)
        for token_id in (7, 19):
            result = model.forward_token(token_id, state, weights, config)
            model.evaluate_result(result)
            state = result.state

        immutable = model.start_decode_session(weights, state, config)
        linear = model.start_linear_decode_session(weights, state, 4, config)
        copied = copy.copy(linear)
        with self.assertRaisesRegex(moe_reference.MoEError, "invalid linear"):
            model.forward_linear_session_token(11, copied)
        source_attention = state.layers[1]
        self.assertIsInstance(source_attention, mlx_attention.MLXAttentionState)
        source_keys = mx.array(source_attention.keys)
        source_values = mx.array(source_attention.values)

        for token_id in (11, 5):
            expected, immutable = model.forward_session_token(token_id, immutable)
            model.evaluate_result(expected)
            actual = model.forward_linear_session_token(token_id, linear)
            self.assertTrue(bool(mx.array_equal(actual.logits, expected.logits).item()))
            self.assertTrue(bool(mx.array_equal(actual.hidden, expected.hidden).item()))
            self.assertEqual(linear.state.position, expected.state.position)
            for expected_state, actual_state in zip(expected.state.layers, linear.state.layers):
                if isinstance(expected_state, mlx_attention.MLXAttentionState):
                    self.assertIsInstance(actual_state, mlx_attention.MLXLinearAttentionState)
                    self.assertTrue(
                        bool(
                            mx.array_equal(
                                actual_state.keys[:, : actual_state.position],
                                expected_state.keys,
                            ).item()
                        )
                    )
                    self.assertTrue(
                        bool(
                            mx.array_equal(
                                actual_state.values[:, : actual_state.position],
                                expected_state.values,
                            ).item()
                        )
                    )
                else:
                    self.assertTrue(bool(mx.array_equal(actual_state.conv, expected_state.conv).item()))
                    self.assertTrue(
                        bool(mx.array_equal(actual_state.recurrent, expected_state.recurrent).item())
                    )

        self.assertTrue(bool(mx.array_equal(source_attention.keys, source_keys).item()))
        self.assertTrue(bool(mx.array_equal(source_attention.values, source_values).item()))
        with self.assertRaisesRegex(moe_reference.MoEError, "capacity exhausted"):
            model.forward_linear_session_token(3, linear)

    def test_decode_session_matches_checked_path_and_preserves_rollback(self) -> None:
        config, weights = make_fixture()
        initial = model.initial_state(weights, config)
        session = model.start_decode_session(weights, initial, config)
        checked = model.forward_token(7, initial, weights, config)
        fast, next_session = model.forward_session_token(7, session)
        model.evaluate_result(checked)
        model.evaluate_result(fast)

        self.assertTrue(bool(mx.array_equal(checked.logits, fast.logits).item()))
        self.assertTrue(bool(mx.array_equal(checked.hidden, fast.hidden).item()))
        self.assertIs(session.state, initial)
        self.assertIs(next_session.state, fast.state)
        self.assertIs(next_session.weights, weights)
        for checked_state, fast_state in zip(checked.state.layers, fast.state.layers):
            if isinstance(checked_state, mlx_attention.MLXAttentionState):
                self.assertTrue(bool(mx.array_equal(checked_state.keys, fast_state.keys).item()))
                self.assertTrue(bool(mx.array_equal(checked_state.values, fast_state.values).item()))
            else:
                self.assertTrue(bool(mx.array_equal(checked_state.conv, fast_state.conv).item()))
                self.assertTrue(bool(mx.array_equal(checked_state.recurrent, fast_state.recurrent).item()))

        checked_second = model.forward_token(19, checked.state, weights, config)
        fast_second, _ = model.forward_session_token(19, next_session)
        model.evaluate_result(checked_second)
        model.evaluate_result(fast_second)
        self.assertTrue(bool(mx.array_equal(checked_second.logits, fast_second.logits).item()))

    def test_decode_session_rejects_invalid_state_and_deep_weight_drift(self) -> None:
        config, weights = make_fixture()
        state = model.initial_state(weights, config)
        broken_state = model.TextModelState(position=1, layers=state.layers)
        with self.assertRaisesRegex(moe_reference.MoEError, "attention position mismatch"):
            model.start_decode_session(weights, broken_state, config)

        gdn_layer = weights.layers[0]
        broken_mixer = replace(
            gdn_layer.token_mixer,
            in_proj_qkv=gdn_layer.token_mixer.in_proj_qkv[:1],
        )
        broken_weights = replace(
            weights,
            layers=(replace(gdn_layer, token_mixer=broken_mixer), weights.layers[1]),
        )
        with self.assertRaisesRegex(gdn_reference.GDNError, "in_proj_qkv shape mismatch"):
            model.start_decode_session(broken_weights, state, config)

    def test_prefill_chunk_matches_token_composition(self) -> None:
        config, weights = make_fixture()
        token_ids = (7, 19, 11)
        state = model.initial_state(weights, config)
        serial = []
        serial_state = state
        for token_id in token_ids:
            result = model.forward_token(token_id, serial_state, weights, config)
            model.evaluate_result(result)
            serial.append(result)
            serial_state = result.state

        chunk = model.prefill_chunk(token_ids, state, weights, config)
        model.evaluate_chunk_result(chunk, diagnostics=True)
        expected_hidden = mx.stack([result.hidden for result in serial])
        expected_selected = tuple(
            mx.stack([result.selected_experts[layer_index] for result in serial])
            for layer_index in range(len(config.layer_types))
        )
        mx.eval(expected_hidden, *expected_selected)
        self.assertLess(float(mx.max(mx.abs(chunk.hidden - expected_hidden)).item()), 2e-6)
        self.assertLess(
            float(mx.max(mx.abs(chunk.logits - serial[-1].logits)).item()),
            2e-6,
        )
        self.assertEqual(chunk.state.position, len(token_ids))
        for actual, expected in zip(chunk.selected_experts, expected_selected):
            self.assertTrue(bool(mx.array_equal(actual, expected).item()))
        for actual_state, expected_state in zip(chunk.state.layers, serial_state.layers):
            if isinstance(actual_state, mlx_attention.MLXAttentionState):
                self.assertLess(
                    float(mx.max(mx.abs(actual_state.keys - expected_state.keys)).item()),
                    2e-6,
                )
                self.assertLess(
                    float(mx.max(mx.abs(actual_state.values - expected_state.values)).item()),
                    2e-6,
                )
            else:
                self.assertLess(
                    float(mx.max(mx.abs(actual_state.conv - expected_state.conv)).item()),
                    2e-6,
                )
                self.assertLess(
                    float(mx.max(mx.abs(actual_state.recurrent - expected_state.recurrent)).item()),
                    2e-6,
                )

    def test_prefill_chunk_rejects_empty_input(self) -> None:
        config, weights = make_fixture()
        with self.assertRaisesRegex(moe_reference.MoEError, "at least one token"):
            model.prefill_chunk((), model.initial_state(weights, config), weights, config)

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

    def test_hidden_transition_matches_full_logit_transition(self) -> None:
        config, weights = make_fixture()
        state = model.initial_state(weights, config)
        hidden_only = model.forward_hidden_token(7, state, weights, config)
        full = model.forward_token(7, state, weights, config)
        model.evaluate_transition(hidden_only)
        model.evaluate_result(full)

        self.assertTrue(mx.array_equal(hidden_only.hidden, full.hidden).item())
        self.assertEqual(hidden_only.state.position, full.state.position)
        for hidden_state, full_state in zip(hidden_only.state.layers, full.state.layers):
            if isinstance(hidden_state, mlx_attention.MLXAttentionState):
                self.assertTrue(mx.array_equal(hidden_state.keys, full_state.keys).item())
                self.assertTrue(mx.array_equal(hidden_state.values, full_state.values).item())
            else:
                self.assertTrue(mx.array_equal(hidden_state.conv, full_state.conv).item())
                self.assertTrue(mx.array_equal(hidden_state.recurrent, full_state.recurrent).item())
        for hidden_selected, full_selected in zip(
            hidden_only.selected_experts,
            full.selected_experts,
        ):
            self.assertTrue(mx.array_equal(hidden_selected, full_selected).item())
        for hidden_routing, full_routing in zip(
            hidden_only.routing_weights,
            full.routing_weights,
        ):
            self.assertTrue(mx.array_equal(hidden_routing, full_routing).item())
        expected_logits = mx.matmul(weights.lm_head, hidden_only.hidden)
        mx.eval(expected_logits)
        self.assertTrue(mx.array_equal(expected_logits, full.logits).item())

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

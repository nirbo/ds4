#!/usr/bin/env python3
"""Exact target block-verification tests for Ornith-35."""

from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_attention as attention
import ornith35_mlx_compiled as compiled
import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_model as model
import ornith35_mlx_model_test as model_fixture
import ornith35_mlx_speculative as speculative
from ornith35_moe_reference import MoEError


def assert_state_equal(
    testcase: unittest.TestCase,
    actual: model.TextModelState,
    expected: model.TextModelState,
) -> None:
    testcase.assertEqual(actual.position, expected.position)
    testcase.assertEqual(len(actual.layers), len(expected.layers))
    for actual_layer, expected_layer in zip(actual.layers, expected.layers):
        if isinstance(expected_layer, attention.MLXAttentionState):
            testcase.assertIsInstance(
                actual_layer,
                (attention.MLXAttentionState, attention.MLXLinearAttentionState),
            )
            actual_keys = (
                actual_layer.keys[:, : actual_layer.position]
                if isinstance(actual_layer, attention.MLXLinearAttentionState)
                else actual_layer.keys
            )
            actual_values = (
                actual_layer.values[:, : actual_layer.position]
                if isinstance(actual_layer, attention.MLXLinearAttentionState)
                else actual_layer.values
            )
            testcase.assertTrue(bool(mx.array_equal(actual_keys, expected_layer.keys).item()))
            testcase.assertTrue(bool(mx.array_equal(actual_values, expected_layer.values).item()))
        else:
            testcase.assertTrue(bool(mx.array_equal(actual_layer.conv, expected_layer.conv).item()))
            testcase.assertTrue(
                bool(mx.array_equal(actual_layer.recurrent, expected_layer.recurrent).item())
            )


class MLXSpeculativeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.weights = model_fixture.make_fixture()
        initial = model.initial_state(self.weights, self.config)
        first = model.forward_token(7, initial, self.weights, self.config)
        model.evaluate_result(first)
        self.initial_cursor = speculative.cursor_from_result(first)

        cursor = self.initial_cursor
        self.correct = []
        self.serial_cursors = [cursor]
        for _ in range(4):
            token_id = speculative.greedy_token(cursor.logits, cursor.hidden, self.weights.lm_head)
            self.correct.append(token_id)
            result = model.forward_token(token_id, cursor.state, self.weights, self.config)
            model.evaluate_result(result)
            cursor = speculative.cursor_from_result(result)
            self.serial_cursors.append(cursor)
        self.bonus = speculative.greedy_token(cursor.logits, cursor.hidden, self.weights.lm_head)

        self.chunk_cursors = [self.initial_cursor]
        for length in range(1, len(self.correct) + 1):
            transition = model.prefill_hidden_chunk(
                self.correct[:length],
                self.initial_cursor.state,
                self.weights,
                self.config,
                use_steel=False,
            )
            logits = model.project_lm_head(self.weights.lm_head, transition.hidden[-1])
            model.evaluate_chunk_transition(transition)
            mx.eval(logits)
            self.chunk_cursors.append(
                speculative.GreedyTargetCursor(
                    state=transition.state,
                    hidden=transition.hidden[-1],
                    logits=logits,
                )
            )

    def test_accepts_complete_block_and_returns_bonus(self) -> None:
        auxiliary_indices = (1, 2)
        session = speculative.start_greedy_verifier(
            self.weights,
            self.initial_cursor,
            self.config,
            auxiliary_hidden_state_indices=auxiliary_indices,
        )
        verification, next_session = speculative.verify_greedy_block(self.correct, session)
        expected_auxiliary = model.prefill_hidden_chunk_with_aux(
            self.correct,
            self.initial_cursor.state,
            self.weights,
            auxiliary_indices,
            self.config,
            use_steel=False,
        )
        model.evaluate_chunk_transition(expected_auxiliary)
        mx.eval(*verification.committed_auxiliary_hidden_states)

        self.assertTrue(verification.all_accepted)
        self.assertEqual(verification.accepted_count, len(self.correct))
        self.assertEqual(verification.committed_tokens, tuple(self.correct))
        self.assertEqual(verification.emitted_tokens, tuple(self.correct) + (self.bonus,))
        self.assertEqual(
            verification.verified_target_ids,
            tuple(self.correct) + (self.bonus,),
        )
        self.assertEqual(verification.target_forward_tokens, len(self.correct))
        self.assertEqual(verification.rollback_replay_tokens, 0)
        self.assertEqual(verification.rollback_recurrent_tokens, 0)
        self.assertEqual(verification.auxiliary_hidden_state_indices, auxiliary_indices)
        for actual, expected in zip(
            verification.committed_auxiliary_hidden_states,
            expected_auxiliary.auxiliary_hidden_states,
        ):
            self.assertTrue(bool(mx.array_equal(actual, expected).item()))
        assert_state_equal(self, verification.cursor.state, self.chunk_cursors[-1].state)
        self.assertTrue(
            bool(mx.array_equal(verification.cursor.hidden, self.chunk_cursors[-1].hidden).item())
        )
        self.assertTrue(
            bool(mx.array_equal(verification.cursor.logits, self.chunk_cursors[-1].logits).item())
        )
        self.assertIs(next_session.cursor, verification.cursor)

    def test_exact_block_head_batches_decisions_and_keeps_lowest_ties(self) -> None:
        logits = []
        expected = (*self.correct[1:], self.bonus)
        tied_rows = 0
        for token_id in expected:
            row = [-5.0] * self.config.vocab_size
            row[token_id] = 7.0
            if token_id + 1 < self.config.vocab_size:
                row[token_id + 1] = 7.0
                tied_rows += 1
            logits.append(row)
        self.assertGreater(tied_rows, 0)
        block_logits = mx.array(logits, dtype=mx.bfloat16)
        exact_head = mx.zeros(
            (self.config.vocab_size, self.config.hidden_size),
            dtype=mx.bfloat16,
        )
        session = speculative.start_greedy_verifier(
            self.weights,
            self.initial_cursor,
            self.config,
            block_tokens=len(self.correct),
            exact_block_lm_head=exact_head,
        )
        with mock.patch.object(
            speculative,
            "_project_block_logits",
            return_value=block_logits,
        ):
            verification, _ = speculative.verify_greedy_block(self.correct, session)

        self.assertTrue(verification.all_accepted)
        self.assertEqual(verification.verified_target_ids, (*self.correct, self.bonus))

    def test_compiled_prefill_tail_matches_uncompiled_layer(self) -> None:
        token_ids = (7, 19, 11, 5)
        state = model.initial_state(self.weights, self.config)
        hidden = model.embed_tokens(self.weights.embedding, token_ids)
        layer_weights = self.weights.layers[0]
        self.assertIsInstance(layer_weights, layer.GDNLayerWeights)
        mixed_input = layer.qwen_rms_norm_batch(
            hidden,
            layer_weights.norms.input_layernorm,
            self.config.rms_norm_eps,
        )
        mixed, _ = gdn.prefill_chunk(
            mixed_input,
            state.layers[0],
            layer_weights.token_mixer,
            self.config.gdn,
        )
        expected = layer.prefill_gdn(
            hidden,
            state.layers[0],
            layer_weights,
            self.config.gdn,
            self.config.moe,
            next_input_norm=self.weights.layers[1].norms.input_layernorm,
        )
        tail = compiled.compile_prefill_tail(
            0,
            len(token_ids),
            layer_weights,
            self.weights.layers[1].norms.input_layernorm,
            self.config.moe,
        )
        actual = tail(hidden, mixed)
        mx.eval(
            expected.output,
            expected.selected_experts,
            expected.routing_weights,
            expected.normalized_output,
            actual.output,
            actual.selected_experts,
            actual.routing_weights,
            actual.normalized_output,
        )
        self.assertTrue(bool(mx.array_equal(actual.output, expected.output).item()))
        self.assertTrue(
            bool(mx.array_equal(actual.selected_experts, expected.selected_experts).item())
        )
        self.assertTrue(
            bool(mx.array_equal(actual.routing_weights, expected.routing_weights).item())
        )
        self.assertTrue(
            bool(mx.array_equal(actual.normalized_output, expected.normalized_output).item())
        )

    def test_rolls_back_every_mismatch_position(self) -> None:
        auxiliary_indices = (1, 2)
        for mismatch in range(len(self.correct)):
            with self.subTest(mismatch=mismatch):
                proposals = list(self.correct)
                proposals[mismatch] = (proposals[mismatch] + 1) % self.config.vocab_size
                session = speculative.start_greedy_verifier(
                    self.weights,
                    self.initial_cursor,
                    self.config,
                    auxiliary_hidden_state_indices=auxiliary_indices,
                )
                verification, next_session = speculative.verify_greedy_block(
                    proposals,
                    session,
                )

                expected_cursor = self.chunk_cursors[mismatch]
                self.assertFalse(verification.all_accepted)
                self.assertEqual(verification.accepted_count, mismatch)
                self.assertEqual(
                    verification.committed_tokens,
                    tuple(self.correct[:mismatch]),
                )
                self.assertEqual(
                    verification.emitted_tokens,
                    tuple(self.correct[: mismatch + 1]),
                )
                self.assertEqual(
                    verification.verified_target_ids,
                    tuple(self.correct[: mismatch + 1]),
                )
                self.assertEqual(
                    verification.target_forward_tokens,
                    0 if mismatch == 0 else len(proposals),
                )
                self.assertEqual(
                    verification.rollback_replay_tokens,
                    0,
                )
                self.assertEqual(
                    verification.rollback_recurrent_tokens,
                    0 if mismatch == 0 else mismatch,
                )
                self.assertEqual(
                    verification.auxiliary_hidden_state_indices,
                    auxiliary_indices,
                )
                if mismatch == 0:
                    self.assertEqual(verification.committed_auxiliary_hidden_states, ())
                else:
                    self.assertEqual(len(verification.committed_auxiliary_hidden_states), 2)
                    self.assertTrue(
                        all(
                            value.shape == (mismatch, self.config.hidden_size)
                            for value in verification.committed_auxiliary_hidden_states
                        )
                    )
                assert_state_equal(self, verification.cursor.state, expected_cursor.state)
                self.assertTrue(
                    bool(mx.array_equal(verification.cursor.hidden, expected_cursor.hidden).item())
                )
                self.assertTrue(
                    bool(mx.array_equal(verification.cursor.logits, expected_cursor.logits).item())
                )
                self.assertIs(next_session.cursor, verification.cursor)

    def test_linear_verifier_commits_and_rolls_back_owned_kv(self) -> None:
        config, weights = model_fixture.make_bf16_fixture()
        initial = model.initial_state(weights, config)
        first = model.forward_token(7, initial, weights, config)
        model.evaluate_result(first)
        initial_cursor = speculative.cursor_from_result(first)

        correct = []
        serial_cursor = initial_cursor
        for _ in range(4):
            token_id = speculative.greedy_token(
                serial_cursor.logits,
                serial_cursor.hidden,
                weights.lm_head,
            )
            correct.append(token_id)
            result = model.forward_token(token_id, serial_cursor.state, weights, config)
            model.evaluate_result(result)
            serial_cursor = speculative.cursor_from_result(result)

        expected_full = model.prefill_hidden_chunk(
            correct,
            initial_cursor.state,
            weights,
            config,
            use_steel=False,
        )
        expected_full_logits = model.project_lm_head(weights.lm_head, expected_full.hidden[-1])
        model.evaluate_chunk_transition(expected_full)
        mx.eval(expected_full_logits)
        expected_full_cursor = speculative.GreedyTargetCursor(
            state=expected_full.state,
            hidden=expected_full.hidden[-1],
            logits=expected_full_logits,
        )

        linear = model.start_linear_decode_session(
            weights,
            initial_cursor.state,
            initial_cursor.state.position + 8,
            config,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )
        linear_cursor = speculative.GreedyTargetCursor(
            state=linear.state,
            hidden=initial_cursor.hidden,
            logits=initial_cursor.logits,
        )
        session = speculative.start_greedy_verifier(
            weights,
            linear_cursor,
            config,
            block_tokens=len(correct),
            compile_prefill_tails=False,
            linear_session=linear,
        )
        verification, next_session = speculative.verify_greedy_block(correct, session)
        self.assertTrue(verification.all_accepted)
        self.assertIs(linear.state, next_session.cursor.state)
        assert_state_equal(self, linear.state, expected_full_cursor.state)
        self.assertTrue(
            bool(mx.array_equal(next_session.cursor.hidden, expected_full_cursor.hidden).item())
        )
        self.assertTrue(
            bool(mx.array_equal(next_session.cursor.logits, expected_full_cursor.logits).item())
        )
        with self.assertRaisesRegex(MoEError, "stale linear verifier"):
            speculative.verify_greedy_block(correct, session)

        mismatch = 2
        proposals = list(correct)
        proposals[mismatch] = (proposals[mismatch] + 1) % config.vocab_size
        expected_prefix = model.prefill_hidden_chunk(
            correct[:mismatch],
            initial_cursor.state,
            weights,
            config,
            use_steel=False,
        )
        model.evaluate_chunk_transition(expected_prefix)
        rollback_linear = model.start_linear_decode_session(
            weights,
            initial_cursor.state,
            initial_cursor.state.position + 8,
            config,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )
        rollback_cursor = speculative.GreedyTargetCursor(
            state=rollback_linear.state,
            hidden=initial_cursor.hidden,
            logits=initial_cursor.logits,
        )
        rollback_session = speculative.start_greedy_verifier(
            weights,
            rollback_cursor,
            config,
            block_tokens=len(correct),
            compile_prefill_tails=False,
            auxiliary_hidden_state_indices=(1,),
            linear_session=rollback_linear,
        )
        rejected, rejected_session = speculative.verify_greedy_block(
            proposals,
            rollback_session,
        )
        self.assertFalse(rejected.all_accepted)
        self.assertEqual(rejected.accepted_count, mismatch)
        self.assertEqual(rejected.cursor.state.position, expected_prefix.state.position)
        self.assertIs(rollback_linear.state, rejected_session.cursor.state)
        assert_state_equal(self, rollback_linear.state, expected_prefix.state)
        self.assertEqual(len(rejected.committed_auxiliary_hidden_states), 1)
        self.assertEqual(
            rejected.committed_auxiliary_hidden_states[0].shape,
            (mismatch, config.hidden_size),
        )

    def test_rejects_invalid_blocks_and_sessions(self) -> None:
        session = speculative.start_greedy_verifier(
            self.weights,
            self.initial_cursor,
            self.config,
        )
        with self.assertRaisesRegex(MoEError, "must contain"):
            speculative.verify_greedy_block([], session)
        with self.assertRaisesRegex(MoEError, "out of range"):
            speculative.verify_greedy_block([self.config.vocab_size], session)

        bf16_config, bf16_weights = model_fixture.make_bf16_fixture()
        bf16_initial = model.initial_state(bf16_weights, bf16_config)
        bf16_result = model.forward_token(7, bf16_initial, bf16_weights, bf16_config)
        model.evaluate_result(bf16_result)
        bf16_cursor = speculative.cursor_from_result(bf16_result)
        linear = model.start_linear_decode_session(
            bf16_weights,
            bf16_cursor.state,
            8,
            bf16_config,
        )
        with self.assertRaisesRegex(MoEError, "immutable attention state or a linear owner"):
            speculative.start_greedy_verifier(
                bf16_weights,
                speculative.GreedyTargetCursor(
                    state=linear.state,
                    hidden=bf16_cursor.hidden,
                    logits=bf16_cursor.logits,
                ),
                bf16_config,
            )
        with self.assertRaisesRegex(MoEError, "cursor is not the owned"):
            speculative.start_greedy_verifier(
                bf16_weights,
                speculative.GreedyTargetCursor(
                    state=model.TextModelState(
                        position=linear.state.position,
                        layers=linear.state.layers,
                    ),
                    hidden=bf16_cursor.hidden,
                    logits=bf16_cursor.logits,
                ),
                bf16_config,
                linear_session=linear,
            )


if __name__ == "__main__":
    unittest.main()

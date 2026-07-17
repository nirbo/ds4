#!/usr/bin/env python3
"""Exact target block-verification tests for Ornith-35."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_attention as attention
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
            testcase.assertIsInstance(actual_layer, attention.MLXAttentionState)
            testcase.assertTrue(bool(mx.array_equal(actual_layer.keys, expected_layer.keys).item()))
            testcase.assertTrue(bool(mx.array_equal(actual_layer.values, expected_layer.values).item()))
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
        session = speculative.start_greedy_verifier(
            self.weights,
            self.initial_cursor,
            self.config,
        )
        verification, next_session = speculative.verify_greedy_block(self.correct, session)

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
        assert_state_equal(self, verification.cursor.state, self.chunk_cursors[-1].state)
        self.assertTrue(
            bool(mx.array_equal(verification.cursor.hidden, self.chunk_cursors[-1].hidden).item())
        )
        self.assertTrue(
            bool(mx.array_equal(verification.cursor.logits, self.chunk_cursors[-1].logits).item())
        )
        self.assertIs(next_session.cursor, verification.cursor)

    def test_rolls_back_every_mismatch_position(self) -> None:
        for mismatch in range(len(self.correct)):
            with self.subTest(mismatch=mismatch):
                proposals = list(self.correct)
                proposals[mismatch] = (proposals[mismatch] + 1) % self.config.vocab_size
                session = speculative.start_greedy_verifier(
                    self.weights,
                    self.initial_cursor,
                    self.config,
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
                assert_state_equal(self, verification.cursor.state, expected_cursor.state)
                self.assertTrue(
                    bool(mx.array_equal(verification.cursor.hidden, expected_cursor.hidden).item())
                )
                self.assertTrue(
                    bool(mx.array_equal(verification.cursor.logits, expected_cursor.logits).item())
                )
                self.assertIs(next_session.cursor, verification.cursor)

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
        with self.assertRaisesRegex(MoEError, "immutable attention"):
            speculative.start_greedy_verifier(
                bf16_weights,
                speculative.GreedyTargetCursor(
                    state=linear.state,
                    hidden=bf16_cursor.hidden,
                    logits=bf16_cursor.logits,
                ),
                bf16_config,
            )


if __name__ == "__main__":
    unittest.main()

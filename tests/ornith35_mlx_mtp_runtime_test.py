#!/usr/bin/env python3
"""Exact target-verifier integration tests for Ornith-35 Qwen3.5 MTP."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_model as model
import ornith35_mlx_model_test as target_fixture
import ornith35_mlx_mtp_runtime as runtime
import ornith35_mlx_mtp_test as mtp_fixture
import ornith35_mlx_sampling as sampling
import ornith35_mlx_speculative as speculative


def assert_mtp_state_equal(test: unittest.TestCase, actual, expected) -> None:
    test.assertEqual(actual.keys.shape, expected.keys.shape)
    test.assertEqual(actual.values.shape, expected.values.shape)
    test.assertTrue(bool(mx.array_equal(actual.keys, expected.keys).item()))
    test.assertTrue(bool(mx.array_equal(actual.values, expected.values).item()))


class MLXMTPRuntimeTest(unittest.TestCase):
    def test_shifted_prompt_and_reconciled_step_remain_target_authoritative(self) -> None:
        target_config, target_weights = target_fixture.make_fixture()
        mtp_config, scalar_mtp_weights = mtp_fixture.make_fixture()
        mtp_weights = mtp_fixture.mlx_weights(scalar_mtp_weights)
        prompt = (7, 19, 11, 5)
        initial = model.initial_state(target_weights, target_config)
        target_result = model.prefill_chunk(
            prompt,
            initial,
            target_weights,
            target_config,
            use_steel=False,
        )
        model.evaluate_chunk_result(target_result, diagnostics=True)
        cursor = speculative.cursor_from_result(target_result)
        anchor = speculative.greedy_token(
            cursor.logits,
            cursor.hidden,
            target_weights.lm_head,
        )
        context = runtime.build_prompt_context(
            prompt,
            target_result.hidden,
            anchor,
            target_weights.embedding,
            mtp_weights,
            mtp_config,
        )
        self.assertEqual(context.conditioned_token_id, anchor)
        self.assertEqual(context.state.keys.shape[1], len(prompt))

        session = runtime.start_greedy_session(
            target_weights,
            cursor,
            mtp_weights,
            context,
            target_config,
            mtp_config,
            block_tokens=3,
            compile_prefill_tails=False,
        )
        step, next_session = runtime.step_greedy(session)
        verification = step.verification
        self.assertEqual(step.proposal.anchor_token_id, anchor)
        self.assertGreaterEqual(verification.accepted_count, 1)
        self.assertIsNotNone(verification.committed_hidden_states)
        self.assertEqual(
            verification.committed_hidden_states.shape,
            (verification.accepted_count, target_config.hidden_size),
        )

        serial_cursor = cursor
        serial_emitted = []
        serial_hidden = []
        for index in range(len(verification.emitted_tokens)):
            token_id = speculative.greedy_token(
                serial_cursor.logits,
                serial_cursor.hidden,
                target_weights.lm_head,
            )
            serial_emitted.append(token_id)
            if index + 1 == len(verification.emitted_tokens):
                break
            serial_result = model.forward_token(
                token_id,
                serial_cursor.state,
                target_weights,
                target_config,
            )
            model.evaluate_result(serial_result)
            serial_hidden.append(serial_result.hidden)
            serial_cursor = speculative.cursor_from_result(serial_result)
        self.assertEqual(tuple(serial_emitted), verification.emitted_tokens)
        expected_hidden = mx.stack(serial_hidden)
        mx.eval(expected_hidden)
        self.assertTrue(
            bool(
                mx.array_equal(
                    verification.committed_hidden_states,
                    expected_hidden,
                ).item()
            )
        )

        shifted = (
            *verification.committed_tokens[1:],
            verification.emitted_tokens[-1],
        )
        rebuilt = runtime.append_authoritative_hidden(
            context.state,
            verification.committed_hidden_states,
            shifted,
            target_weights.embedding,
            mtp_weights,
            mtp_config,
            _validated=True,
        )
        assert_mtp_state_equal(self, next_session.mtp_context.state, rebuilt.state)
        self.assertTrue(
            bool(mx.array_equal(next_session.mtp_context.hidden, rebuilt.hidden).item())
        )
        self.assertEqual(
            next_session.mtp_context.conditioned_token_id,
            verification.emitted_tokens[-1],
        )
        self.assertEqual(
            next_session.mtp_context.state.keys.shape[1],
            next_session.verifier.cursor.state.position,
        )

        fallback_cursor = next_session.verifier.cursor
        fallback_anchor = speculative.greedy_token(
            fallback_cursor.logits,
            fallback_cursor.hidden,
            target_weights.lm_head,
        )
        expected_fallback = model.forward_token(
            fallback_anchor,
            fallback_cursor.state,
            target_weights,
            target_config,
        )
        model.evaluate_result(expected_fallback)
        expected_bonus = speculative.greedy_token(
            expected_fallback.logits,
            expected_fallback.hidden,
            target_weights.lm_head,
        )
        fallback_step, fallback_session = runtime.step_target_greedy(next_session)
        self.assertEqual(fallback_step.proposal.target_token_ids, (fallback_anchor,))
        self.assertEqual(
            fallback_step.verification.emitted_tokens,
            (fallback_anchor, expected_bonus),
        )
        self.assertEqual(fallback_step.verification.accepted_count, 1)
        self.assertTrue(fallback_step.verification.all_accepted)
        self.assertTrue(
            bool(
                mx.array_equal(
                    fallback_session.verifier.cursor.hidden,
                    expected_fallback.hidden,
                ).item()
            )
        )
        self.assertEqual(
            fallback_session.mtp_context.conditioned_token_id,
            expected_bonus,
        )
        self.assertEqual(
            fallback_session.mtp_context.state.keys.shape[1],
            fallback_session.verifier.cursor.state.position,
        )

        detached = runtime.detach_target_session(next_session)
        detached_step, detached_session = runtime.step_detached_target_greedy(detached)
        self.assertEqual(detached_step.proposal.target_token_ids, (fallback_anchor,))
        self.assertEqual(
            detached_step.verification.emitted_tokens,
            (fallback_anchor, expected_bonus),
        )
        self.assertTrue(
            bool(
                mx.array_equal(
                    detached_session.verifier.cursor.hidden,
                    expected_fallback.hidden,
                ).item()
            )
        )

        adaptive = runtime.start_adaptive_session(
            next_session,
            runtime.MTPAdaptivePolicy(
                minimum_mtp_blocks=2,
                window_blocks=2,
                minimum_future_acceptance=0.75,
            ),
        )
        self.assertEqual(adaptive.mode, "mtp")
        self.assertIsNone(runtime.adaptive_recent_future_acceptance(adaptive))

        low_yield = runtime.start_adaptive_session(
            session,
            runtime.MTPAdaptivePolicy(
                minimum_mtp_blocks=1,
                window_blocks=1,
                minimum_future_acceptance=0.70,
            ),
        )
        adaptive_step, low_yield = runtime.step_adaptive_greedy(low_yield)
        self.assertEqual(adaptive_step.verification.emitted_tokens, verification.emitted_tokens)
        self.assertEqual(low_yield.mode, "target")
        self.assertEqual(low_yield.detached_after_mtp_blocks, 1)
        self.assertEqual(runtime.adaptive_recent_future_acceptance(low_yield), 0.0)
        target_step, low_yield = runtime.step_adaptive_greedy(low_yield)
        self.assertEqual(target_step.verification.emitted_tokens, (fallback_anchor, expected_bonus))
        self.assertEqual(low_yield.mode, "target")

        with self.assertRaisesRegex(RuntimeError, "cannot be shorter"):
            runtime.start_adaptive_session(
                next_session,
                runtime.MTPAdaptivePolicy(
                    minimum_mtp_blocks=1,
                    window_blocks=2,
                ),
            )

        wrong_anchor = (anchor + 1) % target_config.vocab_size
        stale = replace(
            session,
            mtp_context=replace(
                session.mtp_context,
                conditioned_token_id=wrong_anchor,
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "target rejected the MTP-owned anchor",
        ):
            runtime.step_greedy(stale)

    def test_sampled_mtp_and_detached_target_emit_only_exact_target_support(self) -> None:
        target_config, target_weights = target_fixture.make_fixture()
        mtp_config, scalar_mtp_weights = mtp_fixture.make_fixture()
        mtp_weights = mtp_fixture.mlx_weights(scalar_mtp_weights)
        prompt = (7, 19, 11, 5)
        target_result = model.prefill_chunk(
            prompt,
            model.initial_state(target_weights, target_config),
            target_weights,
            target_config,
            use_steel=False,
        )
        model.evaluate_chunk_result(target_result, diagnostics=True)
        cursor = speculative.cursor_from_result(target_result)
        temperature = 0.6
        top_k = min(8, target_config.vocab_size)
        top_p = 0.95
        rng = random.Random(31)
        anchor = sampling.target_distribution(
            cursor.logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).sample(rng)
        context = runtime.build_prompt_context(
            prompt,
            target_result.hidden,
            anchor,
            target_weights.embedding,
            mtp_weights,
            mtp_config,
        )
        session = runtime.start_sampled_session(
            target_weights,
            cursor,
            mtp_weights,
            context,
            target_config,
            mtp_config,
            block_tokens=3,
            compile_prefill_tails=False,
        )
        step, next_session = runtime.step_sampled(
            session,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        self.assertEqual(step.proposal.anchor_token_id, anchor)
        self.assertEqual(
            len(step.proposal.future_distributions),
            len(step.proposal.future_token_ids),
        )
        for token_id, distribution in zip(
            step.proposal.future_token_ids,
            step.proposal.future_distributions,
        ):
            self.assertGreater(distribution.probability(token_id), 0.0)
        self.assertGreaterEqual(step.verification.accepted_count, 1)
        self.assertEqual(
            next_session.mtp_context.conditioned_token_id,
            step.verification.emitted_tokens[-1],
        )

        replay = cursor
        for index, token_id in enumerate(step.verification.emitted_tokens):
            distribution = sampling.target_distribution(
                replay.logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            self.assertGreater(distribution.probability(token_id), 0.0)
            if index + 1 == len(step.verification.emitted_tokens):
                break
            result = model.forward_token(
                token_id,
                replay.state,
                target_weights,
                target_config,
            )
            model.evaluate_result(result)
            replay = speculative.cursor_from_result(result)

        detached = runtime.detach_target_session(next_session)
        self.assertEqual(
            detached.conditioned_token_id,
            step.verification.emitted_tokens[-1],
        )
        target_step, detached = runtime.step_detached_target_sampled(
            detached,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        self.assertEqual(
            target_step.proposal.anchor_token_id,
            step.verification.emitted_tokens[-1],
        )
        self.assertEqual(
            detached.conditioned_token_id,
            target_step.verification.emitted_tokens[-1],
        )


if __name__ == "__main__":
    unittest.main()

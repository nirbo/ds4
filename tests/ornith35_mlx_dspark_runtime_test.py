#!/usr/bin/env python3
"""End-to-end target-verifier integration tests for Ornith-35 DSpark."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_dspark as dspark
import ornith35_mlx_dspark_runtime as runtime
import ornith35_mlx_model as model
import ornith35_mlx_model_test as target_fixture
import ornith35_mlx_speculative as speculative
from ornith35_dspark_reference import DSparkConfig


CONFIG = DSparkConfig(
    target_vocab_size=32,
    draft_vocab_size=16,
    hidden_size=16,
    aux_hidden_state_indices=(1, 2),
    block_size=4,
    mask_token_id=31,
    num_layers=2,
    intermediate_size=24,
    num_q_heads=4,
    num_kv_heads=2,
    head_dim=4,
    rotary_dim=4,
    rope_theta=10_000.0,
    max_position_embeddings=32,
    rms_norm_eps=1e-6,
    markov_rank=4,
)


def matrix(rows: int, columns: int, phase: float) -> mx.array:
    return mx.array(
        [
            [
                math.sin((row * columns + column + 1) * phase) * 0.08
                for column in range(columns)
            ]
            for row in range(rows)
        ],
        dtype=mx.bfloat16,
    )


def norm(size: int, phase: float) -> mx.array:
    return mx.array(
        [0.93 + math.cos((index + 1) * phase) * 0.06 for index in range(size)],
        dtype=mx.bfloat16,
    )


def draft_weights() -> dspark.MLXDSparkWeights:
    layers = []
    for index in range(CONFIG.num_layers):
        phase = 0.007 + index * 0.011
        layers.append(
            dspark.MLXDraftLayerWeights(
                attention=dspark.MLXDraftAttentionWeights(
                    q_proj=matrix(CONFIG.query_width, CONFIG.hidden_size, phase),
                    k_proj=matrix(CONFIG.kv_width, CONFIG.hidden_size, phase + 0.002),
                    v_proj=matrix(CONFIG.kv_width, CONFIG.hidden_size, phase + 0.003),
                    o_proj=matrix(CONFIG.hidden_size, CONFIG.query_width, phase + 0.004),
                    q_norm=norm(CONFIG.head_dim, phase + 0.005),
                    k_norm=norm(CONFIG.head_dim, phase + 0.006),
                ),
                input_norm=norm(CONFIG.hidden_size, phase + 0.007),
                post_attention_norm=norm(CONFIG.hidden_size, phase + 0.008),
                gate_proj=matrix(
                    CONFIG.intermediate_size,
                    CONFIG.hidden_size,
                    phase + 0.009,
                ),
                up_proj=matrix(
                    CONFIG.intermediate_size,
                    CONFIG.hidden_size,
                    phase + 0.010,
                ),
                down_proj=matrix(
                    CONFIG.hidden_size,
                    CONFIG.intermediate_size,
                    phase + 0.011,
                ),
            )
        )
    return dspark.MLXDSparkWeights(
        d2t=mx.zeros((CONFIG.draft_vocab_size,), dtype=mx.int64),
        t2d=mx.array(
            [index < CONFIG.draft_vocab_size for index in range(CONFIG.target_vocab_size)],
            dtype=mx.bool_,
        ),
        embedding=matrix(CONFIG.target_vocab_size, CONFIG.hidden_size, 0.013),
        fc=matrix(CONFIG.hidden_size, CONFIG.aux_width, 0.017),
        hidden_norm=norm(CONFIG.hidden_size, 0.019),
        layers=tuple(layers),
        norm=norm(CONFIG.hidden_size, 0.023),
        lm_head=matrix(CONFIG.draft_vocab_size, CONFIG.hidden_size, 0.029),
        markov_w1=matrix(CONFIG.target_vocab_size, CONFIG.markov_rank, 0.031),
        markov_w2=matrix(CONFIG.draft_vocab_size, CONFIG.markov_rank, 0.037),
        confidence_weight=mx.array(
            [
                math.cos((index + 1) * 0.041) * 0.09
                for index in range(CONFIG.hidden_size + CONFIG.markov_rank)
            ],
            dtype=mx.bfloat16,
        ),
        confidence_bias=mx.array([-0.03], dtype=mx.bfloat16),
    )


def assert_context_equal(
    test: unittest.TestCase,
    actual: dspark.MLXDSparkContextState | dspark.MLXDSparkLinearContextState,
    expected: dspark.MLXDSparkContextState | dspark.MLXDSparkLinearContextState,
) -> None:
    test.assertEqual(actual.position, expected.position)
    for index in range(CONFIG.num_layers):
        actual_keys = (
            mx.transpose(actual.keys[index][:, : actual.position], (1, 0, 2))
            if isinstance(actual, dspark.MLXDSparkLinearContextState)
            else actual.keys[index]
        )
        actual_values = (
            mx.transpose(actual.values[index][:, : actual.position], (1, 0, 2))
            if isinstance(actual, dspark.MLXDSparkLinearContextState)
            else actual.values[index]
        )
        expected_keys = (
            mx.transpose(expected.keys[index][:, : expected.position], (1, 0, 2))
            if isinstance(expected, dspark.MLXDSparkLinearContextState)
            else expected.keys[index]
        )
        expected_values = (
            mx.transpose(expected.values[index][:, : expected.position], (1, 0, 2))
            if isinstance(expected, dspark.MLXDSparkLinearContextState)
            else expected.values[index]
        )
        test.assertTrue(bool(mx.array_equal(actual_keys, expected_keys).item()))
        test.assertTrue(bool(mx.array_equal(actual_values, expected_values).item()))


class MLXDSparkRuntimeTest(unittest.TestCase):
    def test_verifier_commits_matching_target_states_into_draft_context(self) -> None:
        target_config, target_weights = target_fixture.make_bf16_fixture()
        weights = draft_weights()
        initial_target = model.initial_state(target_weights, target_config)
        first = model.forward_hidden_token_with_aux(
            7,
            initial_target,
            target_weights,
            CONFIG.aux_hidden_state_indices,
            target_config,
        )
        logits = model.project_lm_head(target_weights.lm_head, first.hidden)
        model.evaluate_transition(first)
        mx.eval(logits, *first.auxiliary_hidden_states)
        cursor = speculative.GreedyTargetCursor(
            state=first.state,
            hidden=first.hidden,
            logits=logits,
        )
        target_linear = model.start_linear_decode_session(
            target_weights,
            cursor.state,
            CONFIG.max_position_embeddings,
            target_config,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )
        cursor = speculative.GreedyTargetCursor(
            state=target_linear.state,
            hidden=cursor.hidden,
            logits=cursor.logits,
        )
        context = runtime.append_target_auxiliary(
            dspark.initial_linear_context(CONFIG, CONFIG.max_position_embeddings),
            first,
            weights,
            CONFIG,
        )
        immutable_context = runtime.append_target_auxiliary(
            dspark.initial_context(CONFIG, mx.bfloat16),
            first,
            weights,
            CONFIG,
        )
        session = runtime.start_greedy_session(
            target_weights,
            cursor,
            weights,
            context,
            target_config,
            CONFIG,
            compile_prefill_tails=False,
            target_linear_session=target_linear,
        )

        step, next_session = runtime.step_greedy(session)
        verification = step.verification
        self.assertEqual(
            step.anchor_token_id,
            speculative.greedy_token(cursor.logits, cursor.hidden, target_weights.lm_head),
        )
        self.assertEqual(step.proposal.target_token_ids.shape, (CONFIG.block_size - 1,))
        self.assertEqual(verification.accepted_count, 1)
        self.assertFalse(verification.all_accepted)
        self.assertEqual(
            verification.auxiliary_hidden_state_indices,
            CONFIG.aux_hidden_state_indices,
        )
        self.assertEqual(
            len(verification.committed_auxiliary_hidden_states),
            len(CONFIG.aux_hidden_state_indices),
        )
        expected_context = dspark.append_context(
            immutable_context,
            verification.committed_auxiliary_hidden_states,
            weights,
            CONFIG,
            _validated=True,
        )
        assert_context_equal(self, next_session.draft_context, expected_context)
        self.assertEqual(
            next_session.draft_context.position,
            next_session.verifier.cursor.state.position,
        )
        self.assertIs(target_linear.state, next_session.verifier.cursor.state)
        with self.assertRaisesRegex(dspark.MLXDSparkError, "stale"):
            runtime.step_greedy(session)

        second, final_session = runtime.step_greedy(next_session)
        self.assertGreaterEqual(second.verification.accepted_count, 1)
        self.assertGreater(
            final_session.draft_context.position,
            next_session.draft_context.position,
        )
        self.assertEqual(
            final_session.draft_context.position,
            final_session.verifier.cursor.state.position,
        )
        self.assertIs(target_linear.state, final_session.verifier.cursor.state)


if __name__ == "__main__":
    unittest.main(verbosity=2)

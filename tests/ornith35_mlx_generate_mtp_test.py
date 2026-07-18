#!/usr/bin/env python3
"""Normal-generation integration tests for streamed Ornith-35 MTP state."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_model_test as target_fixture
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_runtime as runtime
import ornith35_mlx_mtp_test as mtp_fixture
import ornith35_mlx_speculative as speculative
from ornith35_mlx_speculative_test import assert_state_equal


class MLXGenerateMTPTest(unittest.TestCase):
    def test_streamed_prompt_matches_full_target_and_mtp_context(self) -> None:
        target_config, target_weights = target_fixture.make_bf16_fixture()
        mtp_config, scalar_mtp_weights = mtp_fixture.make_fixture()
        mtp_weights = mtp_fixture.mlx_weights(scalar_mtp_weights, mx.bfloat16)
        prompt = [7, 19, 11, 5, 3, 13, 17]
        initial = model.initial_state(target_weights, target_config)
        expected = model.prefill_chunk(
            prompt,
            initial,
            target_weights,
            target_config,
            use_steel=False,
        )
        model.evaluate_chunk_result(expected, diagnostics=True)
        pending = speculative.greedy_token(
            expected.logits,
            expected.hidden[-1],
            target_weights.lm_head,
        )
        expected_context = runtime.build_prompt_context(
            prompt,
            expected.hidden,
            pending,
            target_weights.embedding,
            mtp_weights,
            mtp_config,
        )
        linear = model.start_linear_decode_session(
            target_weights,
            initial,
            len(prompt) + 4,
            target_config,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )

        actual, schedule, actual_context = generate.prefill_prompt_with_mtp(
            prompt,
            linear.state,
            target_weights,
            mtp_weights,
            max_chunk=8,
            mtp_capacity=len(prompt) + 4,
            select_pending=lambda logits, hidden: speculative.greedy_token(
                logits,
                hidden,
                target_weights.lm_head,
            ),
            linear_session=linear,
            target_config=target_config,
            mtp_config=mtp_config,
        )

        self.assertEqual(schedule, (1,) * len(prompt))
        self.assertTrue(bool(mx.array_equal(actual.hidden, expected.hidden[-1]).item()))
        self.assertTrue(bool(mx.array_equal(actual.logits, expected.logits).item()))
        assert_state_equal(self, actual.state, expected.state)
        self.assertEqual(actual_context.conditioned_token_id, pending)
        position = expected.state.position
        self.assertTrue(
            bool(
                mx.array_equal(
                    actual_context.state.keys[:, :position],
                    expected_context.state.keys,
                ).item()
            )
        )
        self.assertTrue(
            bool(
                mx.array_equal(
                    actual_context.state.values[:, :position],
                    expected_context.state.values,
                ).item()
            )
        )
        self.assertTrue(bool(mx.array_equal(actual_context.hidden, expected_context.hidden).item()))


if __name__ == "__main__":
    unittest.main(verbosity=2)

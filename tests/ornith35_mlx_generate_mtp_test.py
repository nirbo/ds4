#!/usr/bin/env python3
"""Normal-generation integration tests for streamed Ornith-35 MTP state."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_generate as generate
import ornith35_mlx_attention as attention
import ornith35_mlx_cache as cache
import ornith35_mlx_model as model
import ornith35_mlx_model_test as target_fixture
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_runtime as runtime
import ornith35_mlx_mtp_test as mtp_fixture
import ornith35_mlx_speculative as speculative
from ornith35_mlx_speculative_test import assert_state_equal


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def mtp_identity() -> cache.CacheIdentity:
    return cache.CacheIdentity(
        model_id="test-ornith35",
        model_revision="test-revision",
        source_sha256=digest("source"),
        runtime_revision="test-runtime",
        runtime_sha256=digest("runtime"),
        tokenizer_sha256=digest("tokenizer"),
        chat_template_sha256=digest("template"),
        quantization_policy_sha256=digest("nvfp4"),
        rope_profile="native-262k",
        cache_dtype="BF16",
        mtp_profile=cache.MTP_PROFILE_FOLDED,
        mtp_policy_sha256=digest("mtp-policy"),
    )


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

    def test_persisted_mtp_prefix_resumes_to_bit_exact_full_context(self) -> None:
        target_config, target_weights = target_fixture.make_bf16_fixture()
        mtp_config, scalar_mtp_weights = mtp_fixture.make_fixture()
        mtp_weights = mtp_fixture.mlx_weights(scalar_mtp_weights, mx.bfloat16)
        prompt = [7, 19, 11, 5, 3, 13, 17]
        prefix_tokens = prompt[:3]
        suffix_tokens = prompt[3:]
        capacity = len(prompt) + 4

        full_session = model.start_linear_decode_session(
            target_weights,
            model.initial_state(target_weights, target_config),
            capacity,
            target_config,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )
        full_result, _, full_context = generate.prefill_prompt_with_mtp(
            prompt,
            full_session.state,
            target_weights,
            mtp_weights,
            max_chunk=1,
            mtp_capacity=capacity,
            select_pending=lambda logits, hidden: speculative.greedy_token(
                logits,
                hidden,
                target_weights.lm_head,
            ),
            linear_session=full_session,
            target_config=target_config,
            mtp_config=mtp_config,
        )

        prefix_session = model.start_linear_decode_session(
            target_weights,
            model.initial_state(target_weights, target_config),
            capacity,
            target_config,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )
        prefix_state, _, mtp_prefix = generate.prefill_state_prompt_with_mtp(
            prefix_tokens,
            prefix_session.state,
            target_weights,
            mtp_weights,
            max_chunk=1,
            mtp_capacity=capacity,
            linear_session=prefix_session,
            target_config=target_config,
            mtp_config=mtp_config,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = cache.save_cache(
                Path(temporary),
                prefix_tokens,
                prefix_state,
                mtp_identity(),
                target_config,
                mtp_prefix=mtp_prefix,
                mtp_config=mtp_config,
            )
            restored = cache.load_cache(
                path,
                mtp_identity(),
                target_config,
                mtp_config=mtp_config,
            )
        resumed_session = model.start_linear_decode_session(
            target_weights,
            restored.state,
            capacity,
            target_config,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )
        resumed_result, _, resumed_context = generate.prefill_prompt_with_mtp(
            suffix_tokens,
            resumed_session.state,
            target_weights,
            mtp_weights,
            max_chunk=1,
            mtp_capacity=capacity,
            select_pending=lambda logits, hidden: speculative.greedy_token(
                logits,
                hidden,
                target_weights.lm_head,
            ),
            mtp_prefix=restored.mtp_prefix,
            linear_session=resumed_session,
            target_config=target_config,
            mtp_config=mtp_config,
        )

        self.assertTrue(bool(mx.array_equal(resumed_result.hidden, full_result.hidden).item()))
        self.assertTrue(bool(mx.array_equal(resumed_result.logits, full_result.logits).item()))
        self.assertEqual(resumed_result.state.position, full_result.state.position)
        for actual_layer, expected_layer in zip(
            resumed_result.state.layers,
            full_result.state.layers,
        ):
            if isinstance(
                expected_layer,
                (attention.MLXAttentionState, attention.MLXLinearAttentionState),
            ):
                actual_keys = (
                    actual_layer.keys[:, : actual_layer.position]
                    if isinstance(actual_layer, attention.MLXLinearAttentionState)
                    else actual_layer.keys
                )
                expected_keys = (
                    expected_layer.keys[:, : expected_layer.position]
                    if isinstance(expected_layer, attention.MLXLinearAttentionState)
                    else expected_layer.keys
                )
                actual_values = (
                    actual_layer.values[:, : actual_layer.position]
                    if isinstance(actual_layer, attention.MLXLinearAttentionState)
                    else actual_layer.values
                )
                expected_values = (
                    expected_layer.values[:, : expected_layer.position]
                    if isinstance(expected_layer, attention.MLXLinearAttentionState)
                    else expected_layer.values
                )
                self.assertTrue(bool(mx.array_equal(actual_keys, expected_keys).item()))
                self.assertTrue(bool(mx.array_equal(actual_values, expected_values).item()))
            else:
                self.assertTrue(
                    bool(mx.array_equal(actual_layer.conv, expected_layer.conv).item())
                )
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            actual_layer.recurrent,
                            expected_layer.recurrent,
                        ).item()
                    )
                )
        self.assertEqual(
            resumed_context.conditioned_token_id,
            full_context.conditioned_token_id,
        )
        position = full_result.state.position
        self.assertTrue(
            bool(
                mx.array_equal(
                    resumed_context.state.keys[:, :position],
                    full_context.state.keys[:, :position],
                ).item()
            )
        )
        self.assertTrue(
            bool(
                mx.array_equal(
                    resumed_context.state.values[:, :position],
                    full_context.state.values[:, :position],
                ).item()
            )
        )
        self.assertTrue(bool(mx.array_equal(resumed_context.hidden, full_context.hidden).item()))


if __name__ == "__main__":
    unittest.main(verbosity=2)

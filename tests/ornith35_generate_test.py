#!/usr/bin/env python3
"""Sampling-contract checks for Ornith-35 generation."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_generate as generate


class GenerateTest(unittest.TestCase):
    def test_cli_defaults_to_thinking_and_recommended_sampling(self) -> None:
        with mock.patch.object(sys, "argv", ["generate", "--prompt", "Question"]):
            args = generate.parse_args()
        self.assertTrue(args.enable_thinking)
        self.assertEqual(args.max_tokens, 1024)
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.prefill_chunk, 128)
        self.assertTrue(args.linear_kv_cache)
        self.assertTrue(args.mapped_embedding)
        self.assertTrue(args.quantized_lm_head)
        self.assertTrue(args.exact_long_attention)
        self.assertIsNone(args.load_cache)
        self.assertFalse(args.save_cache)
        self.assertIsNone(args.cache_root)
        self.assertFalse(args.cache_system_prefix)
        self.assertEqual(args.cache_max_gib, 24.0)

    def test_cli_can_explicitly_disable_thinking(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["generate", "--prompt", "Question", "--no-thinking"],
        ):
            args = generate.parse_args()
        self.assertFalse(args.enable_thinking)

    def test_cli_can_enable_quantized_lm_head(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["generate", "--prompt", "Question", "--quantized-lm-head"],
        ):
            args = generate.parse_args()
        self.assertTrue(args.quantized_lm_head)

    def test_cli_can_disable_quantized_lm_head(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["generate", "--prompt", "Question", "--no-quantized-lm-head"],
        ):
            args = generate.parse_args()
        self.assertFalse(args.quantized_lm_head)

    def test_cli_can_disable_exact_long_attention(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "ornith35_mlx_generate.py",
                "--prompt",
                "test",
                "--no-exact-long-attention",
            ],
        ):
            args = generate.parse_args()
        self.assertFalse(args.exact_long_attention)

    def test_splits_reasoning_from_final_response(self) -> None:
        self.assertEqual(
            generate.split_reasoning_response("analysis\n</think>\n\nfinal\n"),
            ("analysis", "final\n"),
        )
        self.assertEqual(
            generate.split_reasoning_response("plain"),
            (None, "plain"),
        )

    def test_top_p_can_retain_only_the_best_candidate(self) -> None:
        selected = generate.sample_candidates(
            [10, 20, 30],
            [4.0, 1.0, 0.0],
            temperature=1.0,
            top_p=0.5,
            rng=random.Random(7),
        )
        self.assertEqual(selected, 10)

    def test_sampling_is_seed_deterministic(self) -> None:
        def sample(seed: int) -> list[int]:
            rng = random.Random(seed)
            return [
                generate.sample_candidates(
                    [1, 2, 3],
                    [1.0, 0.9, 0.8],
                    temperature=0.6,
                    top_p=0.95,
                    rng=rng,
                )
                for _ in range(20)
            ]

        self.assertEqual(sample(11), sample(11))
        self.assertNotEqual(sample(11), sample(12))

    def test_hybrid_head_uses_exact_scores_and_lowest_token_tie_break(self) -> None:
        head = generate.vocab.MLXAffineQuantizedMatrix(
            packed=None,
            scales=None,
            biases=None,
            shape=(64, 2048),
            group_size=32,
            bits=8,
            reference=object(),
        )
        with mock.patch.object(
            generate.vocab,
            "exact_candidate_scores",
            return_value=([20, 10, 30], [4.0, 4.0, 1.0]),
        ) as exact:
            selected = generate.choose_next_token(
                mx.zeros((64,), dtype=mx.bfloat16),
                temperature=0.0,
                top_k=3,
                top_p=0.95,
                rng=random.Random(0),
                hidden=mx.zeros((2048,), dtype=mx.bfloat16),
                lm_head=head,
            )
        self.assertEqual(selected, 10)
        exact.assert_called_once()

    def test_prefill_schedule_uses_bounded_compiled_chunks_and_serial_tail(self) -> None:
        self.assertEqual(generate.prefill_schedule(1, 128), (1,))
        self.assertEqual(generate.prefill_schedule(25, 128), (16, 8, 1))
        self.assertEqual(generate.prefill_schedule(259, 128), (128, 128, 1, 1, 1))
        self.assertEqual(generate.prefill_schedule(17, 1), (1,) * 17)
        self.assertEqual(generate.format_prefill_schedule((128, 128, 32, 1, 1, 1)), "128x2,32,1x3")
        with self.assertRaisesRegex(generate.MoEError, "power of two"):
            generate.prefill_schedule(16, 12)
        with self.assertRaisesRegex(generate.MoEError, "through 128"):
            generate.prefill_schedule(256, 256)

    def test_prefill_uses_state_only_path_before_final_chunk(self) -> None:
        states = [generate.model.TextModelState(position=0, layers=())]

        def advance(token_ids, state, weights, **kwargs):
            self.assertIs(state, states[-1])
            self.assertFalse(kwargs["use_steel"])
            self.assertTrue(kwargs["exact_long_attention"])
            next_state = generate.model.TextModelState(
                position=state.position + len(token_ids),
                layers=(),
            )
            states.append(next_state)
            return next_state

        final_result = generate.model.TextModelResult(
            hidden=None,
            state=generate.model.TextModelState(position=25, layers=()),
            selected_experts=(),
            routing_weights=(),
            logits=None,
        )
        with (
            mock.patch.object(generate.model, "prefill_state_chunk", side_effect=advance) as state_only,
            mock.patch.object(generate.model, "evaluate_state") as evaluate_state,
            mock.patch.object(generate.model, "forward_token", return_value=final_result) as final,
            mock.patch.object(generate.model, "evaluate_result") as evaluate_result,
            mock.patch.object(generate.model, "prefill_hidden_chunk") as full_hidden,
        ):
            result, schedule = generate.prefill_prompt(
                list(range(25)),
                states[0],
                object(),
                max_chunk=128,
            )

        self.assertIs(result, final_result)
        self.assertEqual(schedule, (16, 8, 1))
        self.assertEqual(state_only.call_count, 2)
        self.assertEqual(evaluate_state.call_count, 2)
        final.assert_called_once_with(24, states[-1], mock.ANY)
        evaluate_result.assert_called_once_with(final_result)
        full_hidden.assert_not_called()

    def test_prefill_uses_last_token_path_for_final_multi_token_chunk(self) -> None:
        initial = generate.model.TextModelState(position=0, layers=())
        advanced = generate.model.TextModelState(position=16, layers=())
        final_result = generate.model.TextModelResult(
            hidden=None,
            state=generate.model.TextModelState(position=24, layers=()),
            selected_experts=(),
            routing_weights=(),
            logits=None,
        )
        with (
            mock.patch.object(
                generate.model,
                "prefill_state_chunk",
                return_value=advanced,
            ) as state_only,
            mock.patch.object(generate.model, "evaluate_state"),
            mock.patch.object(
                generate.model,
                "prefill_final_chunk",
                return_value=final_result,
            ) as final,
            mock.patch.object(generate.model, "evaluate_result") as evaluate_result,
            mock.patch.object(generate.model, "prefill_chunk") as full_chunk,
        ):
            result, schedule = generate.prefill_prompt(
                list(range(24)),
                initial,
                object(),
                max_chunk=128,
            )

        self.assertIs(result, final_result)
        self.assertEqual(schedule, (16, 8))
        state_only.assert_called_once_with(
            list(range(16)),
            initial,
            mock.ANY,
            use_steel=False,
            exact_long_attention=True,
        )
        final.assert_called_once_with(
            list(range(16, 24)),
            advanced,
            mock.ANY,
            use_steel=False,
            exact_long_attention=True,
        )
        evaluate_result.assert_called_once_with(final_result)
        full_chunk.assert_not_called()

    def test_state_prefill_avoids_observable_tail_for_stable_chunks(self) -> None:
        states = [generate.model.TextModelState(position=0, layers=())]

        def advance(token_ids, session, **kwargs):
            self.assertFalse(kwargs["use_steel"])
            self.assertTrue(kwargs["exact_long_attention"])
            next_state = generate.model.TextModelState(
                position=states[-1].position + len(token_ids),
                layers=(),
            )
            states.append(next_state)
            return next_state

        final_transition = generate.model.TextModelTransition(
            hidden=None,
            state=generate.model.TextModelState(position=25, layers=()),
            selected_experts=(),
            routing_weights=(),
        )
        session = object()
        with (
            mock.patch.object(
                generate.model,
                "prefill_linear_session_state_chunk",
                side_effect=advance,
            ) as state_only,
            mock.patch.object(
                generate.model,
                "forward_linear_session_hidden_token",
                return_value=final_transition,
            ) as singleton,
        ):
            state, schedule = generate.prefill_state_prompt(
                list(range(25)),
                states[0],
                object(),
                max_chunk=128,
                linear_session=session,
            )

        self.assertIs(state, final_transition.state)
        self.assertEqual(schedule, (16, 8, 1))
        self.assertEqual(state_only.call_count, 2)
        singleton.assert_called_once_with(24, session)


if __name__ == "__main__":
    unittest.main(verbosity=2)

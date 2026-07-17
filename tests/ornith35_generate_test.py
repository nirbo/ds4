#!/usr/bin/env python3
"""Sampling-contract checks for Ornith-35 generation."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest import mock


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
        self.assertFalse(args.quantized_lm_head)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

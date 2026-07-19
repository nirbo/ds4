#!/usr/bin/env python3
"""Pure contract tests for the long-context TurboQuant coding gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_turboquant_coding_gate as gate
from ornith35_moe_reference import MoEError


class CharacterTokenizer:
    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)


class TurboQuantCodingGateTest(unittest.TestCase):
    def test_long_prefix_is_deterministic_and_bounded(self) -> None:
        tokenizer = CharacterTokenizer()
        with mock.patch.object(gate, "render_system_prefix", side_effect=lambda text: f"<{text}>"):
            first = gate.build_long_system_prefix(tokenizer, 1024)
            second = gate.build_long_system_prefix(tokenizer, 1024)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first.token_ids), 1024)
        self.assertGreater(len(first.token_ids), 800)
        self.assertGreater(first.record_count, 0)

    def test_prompt_tail_reconstructs_complete_chat(self) -> None:
        tokenizer = CharacterTokenizer()
        prefix = gate.LongPrefix("archive", tuple(ord(char) for char in "S:archive\n"), 1)
        prompt = gate.CodingPrompt("case", "write code", "return tests", True)
        with mock.patch.object(gate, "render_system_prefix", side_effect=lambda text: f"S:{text}\n"), mock.patch.object(
            gate,
            "render_text_prompt",
            side_effect=lambda user, system=None, enable_thinking=True: (
                (f"S:{system}\n" if system is not None else "")
                + f"U:{user}:{enable_thinking}"
            ),
        ):
            tail = gate.prompt_tail_ids(tokenizer, prefix, prompt)
        self.assertEqual("".join(chr(token) for token in tail), "U:Response requirements:\nreturn tests\n\nCoding task:\nwrite code:True")

    def test_prompt_loader_is_strict_and_honors_limit(self) -> None:
        rows = (
            {"name": "one", "user": "task one", "enable_thinking": False},
            {"name": "two", "user": "task two", "system": "be exact"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            prompts = gate.load_coding_prompts(path, limit=1)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].name, "one")
        self.assertFalse(prompts[0].enable_thinking)

    def test_prompt_loader_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                '{"name":"same","user":"one"}\n'
                '{"name":"same","user":"two"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MoEError, "duplicate"):
                gate.load_coding_prompts(path)

    def test_quality_failures_report_each_broken_contract(self) -> None:
        summary = {
            "steps": 100,
            "top1": 97,
            "top8_recall_mean": 0.94,
            "kl_mean": 0.02,
            "kl_max": 0.2,
            "material_mismatches": 2,
        }
        failures = gate.quality_failures(
            summary,
            gate.QualityThresholds(0.99, 0.95, 0.01, 0.1, 0.5, 0),
        )
        self.assertEqual(len(failures), 5)

    def test_sample_seeds_must_be_unique_u32(self) -> None:
        self.assertEqual(gate.parse_sample_seeds([17, 29]), (17, 29))
        with self.assertRaisesRegex(MoEError, "unique"):
            gate.parse_sample_seeds([17, 17])
        with self.assertRaisesRegex(MoEError, "U32"):
            gate.parse_sample_seeds([-1])

    def test_exact_attention_layer_candidate_is_strict_and_bound(self) -> None:
        layers = gate.parse_exact_attention_layers("15,3,7")
        self.assertEqual(layers, frozenset((3, 7, 15)))
        policy = gate.candidate_policy(layers, frozenset((11,)))
        self.assertEqual(policy["exact_attention_layers"], [3, 7, 15])
        self.assertEqual(policy["k8_attention_layers"], [11])
        with self.assertRaisesRegex(Exception, "non-empty and unique"):
            gate.parse_exact_attention_layers("3,3")
        with self.assertRaisesRegex(Exception, "selected from"):
            gate.parse_exact_attention_layers("3,8")


if __name__ == "__main__":
    unittest.main(verbosity=2)

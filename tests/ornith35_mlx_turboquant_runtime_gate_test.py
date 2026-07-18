#!/usr/bin/env python3
"""Bounded prompt and quality-score checks for the TurboQuant runtime gate."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_turboquant_runtime_gate as gate


class CharacterTokenizer:
    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)


class TurboQuantRuntimeGateTest(unittest.TestCase):
    def test_head_tail_bound_preserves_final_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompt.txt"
            path.write_text("A" * 160 + "FINAL-TASK", encoding="utf-8")
            with mock.patch.object(
                gate,
                "render_text_prompt",
                side_effect=lambda text, enable_thinking: text,
            ):
                tokens = gate.encode_head_tail_prompt(
                    CharacterTokenizer(),
                    path,
                    96,
                    10,
                )
        text = "".join(chr(token) for token in tokens)
        self.assertLessEqual(len(tokens), 96)
        self.assertIn(gate.HEAD_TAIL_MARKER, text)
        self.assertTrue(text.endswith("FINAL-TASK"))
        self.assertTrue(text.startswith("A"))

    def test_head_tail_returns_complete_prompt_when_it_fits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompt.txt"
            path.write_text("complete", encoding="utf-8")
            with mock.patch.object(
                gate,
                "render_text_prompt",
                side_effect=lambda text, enable_thinking: text,
            ):
                tokens = gate.encode_head_tail_prompt(
                    CharacterTokenizer(),
                    path,
                    32,
                    4,
                )
        self.assertEqual("".join(chr(token) for token in tokens), "complete")

    def test_pre_final_padding_retains_source_and_final_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompt.txt"
            path.write_text(
                "Bob was assigned the number thirty-four.\nFinal task:\nReturn Bob.",
                encoding="utf-8",
            )
            with mock.patch.object(
                gate,
                "render_text_prompt",
                side_effect=lambda text, enable_thinking: text,
            ):
                tokens = gate.encode_padded_before_final_prompt(
                    CharacterTokenizer(),
                    path,
                    512,
                )
        text = "".join(chr(token) for token in tokens)
        self.assertLessEqual(len(tokens), 512)
        self.assertGreater(len(tokens), 400)
        self.assertEqual(text.count("was assigned the number"), 1)
        self.assertIn("Archive filler record", text)
        self.assertTrue(text.endswith("Final task:\nReturn Bob."))

    def test_required_line_coverage_reports_missing_entries_in_order(self) -> None:
        covered, missing = gate.required_line_coverage(
            "Bob=34\nClara=71\n",
            ("Bob=34", "Alice=52", "Clara=71"),
        )
        self.assertEqual(covered, 2)
        self.assertEqual(missing, ("Alice=52",))


if __name__ == "__main__":
    unittest.main(verbosity=2)

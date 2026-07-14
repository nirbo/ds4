#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mtp_teacher_prompts import balanced_prompts  # noqa: E402
from nemotron_metadata import MetadataError  # noqa: E402


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class TeacherPromptsTest(unittest.TestCase):
    def test_selection_is_balanced_interleaved_and_deterministic(self):
        corpus = {
            "b": ["b one", "b two", "b three"],
            "a": ["a one", "a two", "a three"],
        }
        first = balanced_prompts(corpus, FakeTokenizer(), 2, 2)
        second = balanced_prompts(corpus, FakeTokenizer(), 2, 2)
        self.assertEqual(first, second)
        self.assertEqual([row["category"] for row in first], ["a", "b", "a", "b"])

    def test_selection_rejects_insufficient_eligible_rows(self):
        with self.assertRaisesRegex(MetadataError, "only 0"):
            balanced_prompts({"a": ["too many words"]}, FakeTokenizer(), 1, 2)


if __name__ == "__main__":
    unittest.main()

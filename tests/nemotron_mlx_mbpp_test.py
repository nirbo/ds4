#!/usr/bin/env python3
"""Tests for deterministic MBPP evaluation helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_mbpp import (  # noqa: E402
    advance_completion,
    chat_token_ids,
    deterministic_items,
    execute_tests,
    extract_code,
    single_token_delimiter,
)


class MBPPTest(unittest.TestCase):
    def test_normalizes_chat_template_batch_encoding(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        self.assertEqual(chat_token_ids(Tokenizer(), "prompt"), [1, 2, 3])

    def test_chat_prefix_continues_assistant_message(self) -> None:
        class Tokenizer:
            kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                self.messages = messages
                return [1]

        tokenizer = Tokenizer()
        self.assertEqual(chat_token_ids(tokenizer, "prompt", "```python\n"), [1])
        self.assertTrue(tokenizer.kwargs["continue_final_message"])
        self.assertFalse(tokenizer.kwargs["add_generation_prompt"])
        self.assertEqual(tokenizer.messages[-1]["role"], "assistant")

    def test_chat_template_receives_reasoning_controls(self) -> None:
        class Tokenizer:
            kwargs = None

            def apply_chat_template(self, _messages, **kwargs):
                self.kwargs = kwargs
                return [1]

        tokenizer = Tokenizer()
        chat_token_ids(tokenizer, "prompt", enable_thinking=True, low_effort=True)
        self.assertTrue(tokenizer.kwargs["enable_thinking"])
        self.assertTrue(tokenizer.kwargs["low_effort"])

    def test_single_token_delimiter_is_exact(self) -> None:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.text = text
                self.add_special_tokens = add_special_tokens
                return [13]

            def decode(self, token_ids):
                self.token_ids = token_ids
                return self.text

        tokenizer = Tokenizer()
        self.assertEqual(single_token_delimiter(tokenizer, "</think>"), 13)
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertEqual(tokenizer.token_ids, [13])

    def test_reasoning_completion_ignores_fences_before_think_end(self) -> None:
        thinking_complete = False
        fence_count = 0
        complete = False
        for token in (1975, 13, 1975, 42, 1975):
            thinking_complete, fence_count, complete = advance_completion(
                token, True, 13, 1975, thinking_complete, fence_count
            )
        self.assertTrue(thinking_complete)
        self.assertEqual(fence_count, 2)
        self.assertTrue(complete)

    def test_extracts_python_fence(self) -> None:
        self.assertEqual(extract_code("text\n```python\ndef f():\n    return 1\n```"), "def f():\n    return 1")

    def test_extracts_unfenced_function_after_preamble(self) -> None:
        self.assertEqual(extract_code("answer:\ndef f():\n    return 1"), "def f():\n    return 1")

    def test_sampling_is_deterministic_and_offsettable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            rows = [
                {"task_id": value, "test_list": ["assert True"]}
                for value in range(6)
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            first = deterministic_items(path, 3, 0)
            again = deterministic_items(path, 3, 0)
            shifted = deterministic_items(path, 2, 1)
            self.assertEqual(first, again)
            self.assertEqual(shifted, first[1:3])

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS-specific")
    def test_generated_code_is_confined_to_task_directory(self) -> None:
        item = {"test_setup_code": "", "test_list": ["assert add(2, 3) == 5"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            passed, error = execute_tests(
                "def add(a, b):\n    return a + b", item, root, Path(sys.executable)
            )
            self.assertTrue(passed, error)

            outside = root.parent / "nemotron-mbpp-forbidden-write"
            outside.unlink(missing_ok=True)
            passed, _ = execute_tests(
                f"open({str(outside)!r}, 'w').write('bad')\n\ndef add(a, b):\n    return a + b",
                item,
                root,
                Path(sys.executable),
            )
            self.assertFalse(passed)
            self.assertFalse(outside.exists())

            passed, _ = execute_tests(
                "import subprocess\nsubprocess.run(['/usr/bin/true'])\n\ndef add(a, b):\n    return a + b",
                item,
                root,
                Path(sys.executable),
            )
            self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)

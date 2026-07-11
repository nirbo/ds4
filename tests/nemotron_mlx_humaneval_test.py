#!/usr/bin/env python3
"""Tests for HumanEval prompt completion and execution helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_humaneval import (  # noqa: E402
    deterministic_items,
    extract_completion,
    human_eval_tests,
)


class HumanEvalTest(unittest.TestCase):
    def test_extracts_last_function_and_restores_prompt_imports(self) -> None:
        prompt = "from typing import List\n\ndef total(xs: List[int]):\n    pass\n"
        response = "```python\ndef wrong():\n    pass\n```\n```python\ndef total(xs):\n    return sum(xs)\n```"
        self.assertEqual(
            extract_completion(response, prompt),
            "from typing import List\n\ndef total(xs):\n    return sum(xs)",
        )

    def test_joins_body_completion_to_prompt(self) -> None:
        prompt = "def add(a, b):\n    \"\"\"Add values.\"\"\"\n"
        self.assertEqual(extract_completion("    return a + b", prompt), prompt + "    return a + b")

    def test_sampling_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            rows = [
                {"task_id": f"HumanEval/{i}", "prompt": "def f():\n", "test": "def check(f): pass", "entry_point": "f"}
                for i in range(5)
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            self.assertEqual(deterministic_items(path, 3, 0), deterministic_items(path, 3, 0))

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS-specific")
    def test_executes_official_check_function_in_sandbox(self) -> None:
        item = {
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
            "entry_point": "add",
        }
        with tempfile.TemporaryDirectory() as temporary:
            passed, error = human_eval_tests(
                "def add(a, b):\n    return a + b",
                item,
                Path(temporary),
                Path(sys.executable),
            )
            self.assertTrue(passed, error)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for deterministic LiveCodeBench execution helpers."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_livecodebench import (  # noqa: E402
    check_cases,
    deterministic_items,
    normalize_output,
    nvidia_protocol_mismatches,
    split_reasoning,
    stratified_items,
    summarize_results,
)


class LiveCodeBenchTest(unittest.TestCase):
    def test_nvidia_protocol_audit_identifies_only_unverified_split(self) -> None:
        args = argparse.Namespace(
            enable_thinking=True,
            low_effort=False,
            temperature=1.0,
            top_p=0.95,
            repeats=8,
            max_new_tokens=131072,
        )
        self.assertEqual(
            nvidia_protocol_mismatches(args), ["official_dated_split_unverified"]
        )

    def test_splits_reasoning_from_final_content(self) -> None:
        self.assertEqual(
            split_reasoning("work here</think>```python\nprint(1)\n```", True),
            ("work here", "```python\nprint(1)\n```"),
        )
        self.assertEqual(split_reasoning("unfinished", True), ("unfinished", ""))

    def test_summarizes_repeated_samples_separately_from_task_pass_any(self) -> None:
        rows = [
            {
                "task_id": "a",
                "passed": False,
                "truncated": False,
                "generation_seconds": 1.0,
                "generated_tokens": 2,
            },
            {
                "task_id": "a",
                "passed": True,
                "truncated": False,
                "generation_seconds": 2.0,
                "generated_tokens": 3,
            },
            {
                "task_id": "b",
                "passed": False,
                "truncated": True,
                "generation_seconds": 3.0,
                "generated_tokens": 4,
            },
        ]
        summary = summarize_results(rows, 2)
        self.assertEqual(summary["sample_pass_at_1"], 1 / 3)
        self.assertEqual(summary["task_pass_any"], 0.5)
        self.assertEqual(summary["truncated_samples"], 1)

    def test_output_normalization_preserves_content(self) -> None:
        self.assertEqual(normalize_output("a  \n b\n\n"), "a\n b")

    def test_sampling_parses_stdin_cases_and_skips_other_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            rows = [
                {
                    "question_id": str(i),
                    "question_content": "problem",
                    "public_test_cases": json.dumps([{"input": "", "output": "", "testtype": kind}]),
                }
                for i, kind in enumerate(("stdin", "functional", "stdin"))
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            items = deterministic_items(path, 2, 0)
            self.assertEqual({item["question_id"] for item in items}, {"0", "2"})

    def test_stratified_sampling_is_balanced_interleaved_and_offsettable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            rows = []
            for difficulty in ("easy", "medium", "hard"):
                for index in range(4):
                    rows.append(
                        {
                            "question_id": f"{difficulty}-{index}",
                            "question_content": "problem",
                            "difficulty": difficulty,
                            "public_test_cases": json.dumps(
                                [{"input": "", "output": "", "testtype": "stdin"}]
                            ),
                        }
                    )
            path.write_text("\n".join(json.dumps(row) for row in rows))
            first = stratified_items(path, 2, 0)
            shifted = stratified_items(path, 1, 1)
            self.assertEqual(
                [item["difficulty"] for item in first],
                ["easy", "medium", "hard", "easy", "medium", "hard"],
            )
            self.assertEqual(
                [item["question_id"] for item in shifted],
                [item["question_id"] for item in first[3:6]],
            )

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS-specific")
    def test_executes_stdin_program_and_checks_output(self) -> None:
        item = {
            "public_test_cases": [
                {"input": "2 3\n", "output": "5\n", "testtype": "stdin"}
            ]
        }
        code = "a, b = map(int, input().split())\nprint(a + b)"
        with tempfile.TemporaryDirectory() as temporary:
            passed, error, cases = check_cases(
                code, item, Path(temporary), Path(sys.executable)
            )
        self.assertTrue(passed, error)
        self.assertEqual(cases, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

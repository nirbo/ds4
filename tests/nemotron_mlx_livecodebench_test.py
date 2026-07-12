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
    prompt_for,
    split_reasoning,
    stratified_items,
    summarize_results,
    selected_items,
)
from nemotron_mlx_livecodebench_rescore import (  # noqa: E402
    rescore_rows,
    selected_items as rescore_selected_items,
)


class LiveCodeBenchTest(unittest.TestCase):
    def test_nvidia_protocol_audit_identifies_only_unverified_split(self) -> None:
        args = argparse.Namespace(
            enable_thinking=True,
            low_effort=False,
            temperature=1.0,
            top_p=0.95,
            repeats=8,
            repeat_offset=0,
            max_new_tokens=131072,
            max_public_cases=0,
            protocol_profile="standard",
        )
        self.assertEqual(
            nvidia_protocol_mismatches(args), ["official_dated_split_unverified"]
        )

    def test_low_budget_protocol_accepts_low_effort(self) -> None:
        args = argparse.Namespace(
            enable_thinking=True,
            low_effort=True,
            temperature=1.0,
            top_p=0.95,
            repeats=8,
            repeat_offset=0,
            max_new_tokens=131072,
            max_public_cases=0,
            protocol_profile="low-budget",
        )
        state = {
            "config": "release_v6",
            "start_date": "2024-08-01",
            "end_date": "2025-05-31",
        }
        self.assertEqual(nvidia_protocol_mismatches(args, state), ["public_tests_only"])
        self.assertEqual(nvidia_protocol_mismatches(args, state, True), [])

    def test_protocol_audit_rejects_nonzero_repeat_offset(self) -> None:
        args = argparse.Namespace(
            enable_thinking=True,
            low_effort=False,
            temperature=1.0,
            top_p=0.95,
            repeats=8,
            repeat_offset=1,
            max_new_tokens=131072,
            max_public_cases=0,
            protocol_profile="standard",
        )
        self.assertEqual(
            nvidia_protocol_mismatches(args),
            ["repeat_offset_not_0", "official_dated_split_unverified"],
        )

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS-specific")
    def test_checks_private_cases_after_public_cases(self) -> None:
        item = {
            "public_test_cases": [
                {"input": "1\n", "output": "1\n", "testtype": "stdin"}
            ],
            "private_test_cases": [
                {"input": "2\n", "output": "2\n", "testtype": "stdin"}
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            passed, error, cases = check_cases(
                "print(input())", item, Path(temporary), Path(sys.executable)
            )
        self.assertTrue(passed, error)
        self.assertEqual(cases, 2)

    def test_uses_nvidia_aai_prompt_shape(self) -> None:
        prompt = prompt_for(
            {
                "question_content": "Solve it.",
                "starter_code": "class Solution:\n    def solve(self):",
            }
        )
        self.assertTrue(prompt.startswith("### Question:\nSolve it."))
        self.assertIn("Function header:\n```\nclass Solution:", prompt)
        self.assertTrue(
            prompt.endswith("### Answer: (use the provided format with backticks)")
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

    def test_sampling_accepts_stdin_and_functional_cases(self) -> None:
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
            items = deterministic_items(path, 3, 0)
            self.assertEqual({item["question_id"] for item in items}, {"0", "1", "2"})

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

    def test_explicit_task_selection_preserves_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            rows = [
                {
                    "question_id": task_id,
                    "question_content": "problem",
                    "public_test_cases": json.dumps(
                        [{"input": "", "output": "", "testtype": "stdin"}]
                    ),
                }
                for task_id in ("a", "b", "c")
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            items = selected_items(path, ["c", "a"])
            self.assertEqual([item["question_id"] for item in items], ["c", "a"])
            rescored = rescore_selected_items(
                path, {"mode": "task_ids", "task_ids": ["c", "a"]}
            )
            self.assertEqual([item["question_id"] for item in rescored], ["c", "a"])

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

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS-specific")
    def test_executes_functional_program_and_checks_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            item = {
                "metadata": {"func_name": "add"},
                "public_test_cases": [
                    {
                        "input": "2\n3",
                        "output": "5",
                        "testtype": "functional",
                    }
                ],
            }
            passed, error, cases = check_cases(
                "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b",
                item,
                Path(temporary),
                Path(sys.executable),
                1,
            )
            self.assertTrue(passed, error)
            self.assertEqual(cases, 1)

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS-specific")
    def test_rescores_stored_code_with_current_harness(self) -> None:
        item = {
            "question_id": "task",
            "difficulty": "easy",
            "contest_date": "2025-01-01T00:00:00",
            "public_test_cases": [
                {"input": "2 3\n", "output": "5\n", "testtype": "stdin"}
            ],
        }
        source = [
            {
                "task_id": "task",
                "passed": False,
                "code": "a, b = map(int, input().split())\nprint(a + b)",
                "generated_tokens": 1,
                "generation_seconds": 1.0,
                "truncated": False,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            rows = rescore_rows(
                source, [item], Path(temporary), Path(sys.executable), 6
            )
        self.assertTrue(rows[0]["passed"], rows[0]["error"])
        self.assertFalse(rows[0]["source_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

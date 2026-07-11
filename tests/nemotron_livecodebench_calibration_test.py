#!/usr/bin/env python3
"""Tests for leakage-safe LiveCodeBench calibration corpus construction."""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_livecodebench_calibration import build_corpus  # noqa: E402


def row(task_id: str, date: str, difficulty: str) -> dict:
    return {
        "question_id": task_id,
        "question_content": f"Solve task {task_id}.",
        "starter_code": "",
        "difficulty": difficulty,
        "contest_date": date,
        "public_test_cases": [{"testtype": "stdin", "input": "", "output": ""}],
    }


class LiveCodeBenchCalibrationTest(unittest.TestCase):
    def test_date_window_and_eval_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.jsonl"
            rows = [
                row("july-easy", "2024-07-01T00:00:00", "easy"),
                row("july-hard", "2024-07-15T00:00:00", "hard"),
                row("august", "2024-08-01T00:00:00", "hard"),
            ]
            dataset.write_text("\n".join(json.dumps(value) for value in rows) + "\n")
            corpus, selected = build_corpus(
                dataset,
                dt.date(2024, 7, 1),
                dt.date(2024, 7, 31),
                {"july-easy"},
            )
        self.assertEqual([row["task_id"] for row in selected], ["july-hard"])
        self.assertEqual(list(corpus), ["livecodebench_hard"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

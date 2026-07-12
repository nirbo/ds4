#!/usr/bin/env python3
"""Tests for paired LiveCodeBench report comparison."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_livecodebench_compare import compare_reports  # noqa: E402
from nemotron_metadata import MetadataError  # noqa: E402


def report(passes: list[bool]) -> dict:
    result = {
        "status": "complete",
        "format": "nemotron-livecodebench-v1",
        "task_ids": ["a", "b"],
        "results": [],
        "summary": {"passed_samples": sum(passes), "tasks_with_pass": sum(passes)},
    }
    for key in (
        "evaluator_sha256", "generation_helper_sha256", "dataset_sha256", "dataset_state_sha256",
        "private_state_sha256", "private_index_sha256", "sandbox_python", "sandbox_python_version",
        "sampling", "max_new_tokens", "max_public_cases", "execution_timeout", "generation",
        "nvidia_reference_protocol",
    ):
        result[key] = key
    for index, passed in enumerate(passes):
        result["results"].append(
            {
                "task_id": result["task_ids"][index],
                "repeat": 0,
                "seed": index,
                "difficulty": "hard" if index else "easy",
                "passed": passed,
                "truncated": False,
                "error": "" if passed else "wrong",
            }
        )
    return result


class LiveCodeBenchCompareTest(unittest.TestCase):
    def test_reports_paired_sample_and_task_flips(self) -> None:
        baseline = report([True, False])
        candidate = report([False, True])
        result = compare_reports(baseline, candidate)
        self.assertEqual(
            result["sample_matrix"],
            {"both_pass": 0, "baseline_only": 1, "candidate_only": 1, "both_fail": 0},
        )
        self.assertEqual(result["sample_pass_delta"], 0)
        self.assertEqual(len(result["task_flips"]), 2)

    def test_rejects_protocol_mismatch(self) -> None:
        baseline = report([True, False])
        candidate = copy.deepcopy(baseline)
        candidate["generation"] = "changed"
        with self.assertRaises(MetadataError):
            compare_reports(baseline, candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)

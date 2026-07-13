#!/usr/bin/env python3
"""Tests for paired Nemotron MBPP and HumanEval comparisons."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_benchmark_compare import compare_reports  # noqa: E402
from nemotron_metadata import MetadataError  # noqa: E402


def report(passes: list[bool], report_format: str = "nemotron-mbpp-eval-v1") -> dict:
    task_ids = [str(index) for index in range(len(passes))]
    result = {
        "format": report_format,
        "status": "complete",
        "sample_size": len(passes),
        "sample_offset": 0,
        "task_ids": task_ids,
        "results": [],
        "summary": {"passed": sum(passes)},
    }
    for key in (
        "evaluator_sha256",
        "dataset_sha256",
        "sandbox_python",
        "sandbox_python_version",
        "max_new_tokens",
    ):
        result[key] = key
    if report_format == "nemotron-humaneval-v1":
        result["code_eval_helper_sha256"] = "helper"
    for task, passed in zip(task_ids, passes):
        result["results"].append(
            {
                "task_id": task,
                "passed": passed,
                "generated_tokens": 10,
                "response": f"response-{task}",
                "code": f"code-{task}",
                "error": "" if passed else "failed",
            }
        )
    return result


class BenchmarkCompareTest(unittest.TestCase):
    def test_reports_paired_flips_and_exact_outputs(self) -> None:
        baseline = report([True, False, True])
        candidate = report([False, True, True])
        candidate["results"][2]["response"] = "changed"
        result = compare_reports(baseline, candidate)
        self.assertEqual(
            result["matrix"],
            {"both_pass": 1, "baseline_only": 1, "candidate_only": 1, "both_fail": 0},
        )
        self.assertEqual(result["pass_delta"], 0)
        self.assertEqual(result["baseline_only"], ["0"])
        self.assertEqual(result["candidate_only"], ["1"])
        self.assertEqual(result["exact_response_matches"], 2)

    def test_rejects_protocol_mismatch(self) -> None:
        baseline = report([True, False])
        candidate = copy.deepcopy(baseline)
        candidate["max_new_tokens"] = "changed"
        with self.assertRaises(MetadataError):
            compare_reports(baseline, candidate)

    def test_validates_humaneval_helper_identity(self) -> None:
        baseline = report([True], "nemotron-humaneval-v1")
        candidate = copy.deepcopy(baseline)
        candidate["code_eval_helper_sha256"] = "changed"
        with self.assertRaises(MetadataError):
            compare_reports(baseline, candidate)

    def test_rejects_inconsistent_summary(self) -> None:
        baseline = report([True])
        baseline["summary"]["passed"] = 0
        with self.assertRaises(MetadataError):
            compare_reports(baseline, report([True]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

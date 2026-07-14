#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_mtp_teacher_capture import (  # noqa: E402
    authoritative_cycle_rows,
    load_prompts,
    mtp_head_report_path,
    prompt_digest,
    validate_trace_arrays,
)


class TeacherCaptureTest(unittest.TestCase):
    def test_rejection_excludes_counterfactual_verifier_rows(self):
        current = mx.array([1.0, 2.0])
        verified = mx.array([[3.0, 4.0], [99.0, 99.0]])
        rows = authoritative_cycle_rows(current, 10, verified, [20, 30], 0, 8)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0][1], rows[0][2]), (10, 20))

    def test_accepted_prefix_becomes_contiguous_rows(self):
        current = mx.array([1.0, 2.0])
        verified = mx.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        rows = authoritative_cycle_rows(current, 10, verified, [20, 30, 40], 2, 8)
        self.assertEqual([(row[1], row[2]) for row in rows], [(10, 20), (20, 30), (30, 40)])
        self.assertEqual(rows[1][0].tolist(), [3.0, 4.0])

    def test_row_limit_never_overruns_requested_capture(self):
        rows = authoritative_cycle_rows(
            mx.zeros((2,)), 1, mx.zeros((2, 2)), [2, 3], 1, 1
        )
        self.assertEqual(len(rows), 1)

    def test_jsonl_prompts_and_digest_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(
                json.dumps({"prompt": "first"}) + "\n" + json.dumps({"text": "second"}) + "\n"
            )
            prompts = load_prompts(path)
        self.assertEqual(prompts, ["first", "second"])
        self.assertEqual(prompt_digest(prompts), prompt_digest(list(prompts)))

    def test_mtp_head_report_path_is_unambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "nemotron_mtp_vocab_head_report.json"
            expected.write_text("{}")
            self.assertEqual(mtp_head_report_path(root), expected)
            (root / "nemotron_mtp_head_report.json").write_text("{}")
            with self.assertRaisesRegex(Exception, "exactly one"):
                mtp_head_report_path(root)

    def test_trace_validation_requires_contiguous_target_rows(self):
        arrays = {
            "target_hidden": mx.zeros((3, 4096), dtype=mx.bfloat16),
            "accepted_token_ids": mx.array([10, 20, 30], dtype=mx.int32),
            "expected_token_ids": mx.array([20, 30, 40], dtype=mx.int32),
            "prompt_indices": mx.zeros((3,), dtype=mx.int32),
            "scored": mx.ones((3,), dtype=mx.int32),
        }
        metadata = {"format": "nemotron-mtp-teacher-capture-v2", "prompt_index": "0"}
        validate_trace_arrays(arrays, metadata, 0, 3)
        arrays["accepted_token_ids"] = mx.array([10, 99, 30], dtype=mx.int32)
        with self.assertRaisesRegex(Exception, "contiguous"):
            validate_trace_arrays(arrays, metadata, 0, 3)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Format, corruption, and resume tests for Ornith-35 MTP teacher capture."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_mtp_teacher_capture as capture


def fixture() -> tuple[dict[str, mx.array], dict[str, str]]:
    hidden = mx.arange(4 * 2048, dtype=mx.float32).reshape(4, 2048).astype(mx.bfloat16)
    arrays = {
        "target_hidden": hidden,
        "token_ids": mx.array([11, 12, 13, 14], dtype=mx.int32),
        "expected_token_ids": mx.array([-1, 14, 15], dtype=mx.int32),
        "candidate_token_ids": mx.array(
            [[-1, -1, -1, -1], [14, 9, 7, 3], [15, 8, 6, 2]],
            dtype=mx.int32,
        ),
        "candidate_logits": mx.array(
            [[0.0, 0.0, 0.0, 0.0], [4.0, 3.0, 2.0, 1.0], [5.0, 2.0, 1.0, -1.0]],
            dtype=mx.float32,
        ),
        "scored": mx.array([0, 1, 1], dtype=mx.int32),
        "prompt_indices": mx.array([2, 2, 2], dtype=mx.int32),
        "positions": mx.array([0, 1, 2], dtype=mx.int32),
    }
    metadata = {
        "format": capture.FORMAT,
        "prompt_index": "2",
        "prompt_sha256": "a" * 64,
        "prompt_tokens": "2",
        "generated_tokens": "2",
        "candidate_count": "4",
        "candidate_policy": capture.CANDIDATE_POLICY,
    }
    return arrays, metadata


class MTPTeacherCaptureTest(unittest.TestCase):
    def test_candidate_union_prioritizes_bootstrap_errors_and_deduplicates(self) -> None:
        selected = capture.select_candidate_union(
            10,
            [90, 10, 80, 70, 60, 50],
            [10, 20, 30, 40, 50, 60],
            6,
        )
        self.assertEqual(selected, [10, 90, 80, 70, 20, 30])
        self.assertEqual(len(selected), len(set(selected)))

    def test_trace_contract_accepts_contiguous_authoritative_rows(self) -> None:
        arrays, metadata = fixture()
        capture.validate_trace_arrays(
            arrays,
            metadata,
            prompt_index=2,
            expected_prompt_sha256="a" * 64,
        )

    def test_trace_rejects_counterfactual_and_misranked_labels(self) -> None:
        arrays, metadata = fixture()
        broken = dict(arrays)
        broken["expected_token_ids"] = mx.array([-1, 99, 15], dtype=mx.int32)
        with self.assertRaisesRegex(RuntimeError, "exact candidate winner"):
            capture.validate_trace_arrays(broken, metadata, prompt_index=2)

        broken = dict(arrays)
        broken["candidate_logits"] = mx.array(
            [[0.0, 0.0, 0.0, 0.0], [3.0, 2.0, 4.0, 1.0], [5.0, 2.0, 1.0, -1.0]],
            dtype=mx.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "deterministically ranked"):
            capture.validate_trace_arrays(broken, metadata, prompt_index=2)

    def test_completed_state_rereads_size_hash_and_tensor_contract(self) -> None:
        arrays, metadata = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "prompt-00002.safetensors"
            mx.save_safetensors(str(shard), arrays, metadata)
            state = {
                "format": capture.FORMAT,
                "status": "complete",
                "completed": {
                    "2": {
                        "file": shard.name,
                        "bytes": shard.stat().st_size,
                        "sha256": capture.sha256_file(shard),
                        "prompt_sha256": "a" * 64,
                        "rows": 3,
                        "scored_rows": 2,
                    }
                },
                "rows": 3,
                "scored_rows": 2,
            }
            capture.validate_completed(root, state)
            corrupt = copy.deepcopy(state)
            corrupt["completed"]["2"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                capture.validate_completed(root, corrupt)

    def test_jsonl_prompt_records_preserve_per_prompt_thinking_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                '{"user":"first","enable_thinking":false}\n'
                '{"prompt":"second","system":"review carefully"}\n',
                encoding="utf-8",
            )
            records = capture.load_prompts(path, default_thinking=True)
        self.assertEqual([record.user for record in records], ["first", "second"])
        self.assertEqual([record.enable_thinking for record in records], [False, True])
        self.assertEqual(records[1].system, "review carefully")


if __name__ == "__main__":
    unittest.main()

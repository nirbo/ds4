#!/usr/bin/env python3
"""Tests for long-trajectory pruning-attribution helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_trajectory_attribution import (  # noqa: E402
    rank_removed_experts,
    sampled_positions,
    trajectory_tokens,
)


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [10, 11]

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class TrajectoryAttributionTest(unittest.TestCase):
    def test_positions_cover_first_and_last_generated_state(self) -> None:
        self.assertEqual(sampled_positions(4, 10, 3, 10), [3, 6, 9])
        positions = sampled_positions(4, 100, 3, 4)
        self.assertEqual(positions[0], 3)
        self.assertEqual(positions[-1], 99)
        self.assertLessEqual(len(positions), 4)

    def test_removed_experts_rank_by_route_weighted_output(self) -> None:
        indices = np.array([[[0, 1], [2, 1]]], dtype=np.int32)
        scores = np.array([[[0.5, 0.5], [0.25, 0.75]]], dtype=np.float32)
        norms = np.array([[[2.0, 1.0], [8.0, 1.0]]], dtype=np.float32)
        ranking, importance = rank_removed_experts(indices, scores, norms, [0])
        self.assertEqual(ranking, [2, 1])
        self.assertAlmostEqual(importance[2], 2.0)
        self.assertAlmostEqual(importance[1], 1.25)

    def test_explicit_expert_count_preserves_unselected_tail(self) -> None:
        indices = np.array([[[0, 1]]], dtype=np.int32)
        scores = np.array([[[0.5, 0.5]]], dtype=np.float32)
        norms = np.ones_like(scores)
        ranking, importance = rank_removed_experts(indices, scores, norms, [0], 4)
        self.assertEqual(ranking, [1])
        self.assertEqual(len(importance), 4)

    def test_mbpp_trajectory_uses_mbpp_prompt_and_response(self) -> None:
        item = {
            "task_id": 7,
            "prompt": "Return one.",
            "test_list": ["assert answer() == 1"],
        }
        report = {
            "format": "nemotron-mbpp-eval-v1",
            "results": [
                {
                    "task_id": "7",
                    "passed": True,
                    "generated_tokens": 3,
                    "response": "abc",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "mbpp.jsonl"
            dataset.write_text(json.dumps(item) + "\n")
            token_ids, trajectory = trajectory_tokens(
                FakeTokenizer(), dataset, report, "7", 0, 0, "mbpp"
            )
        self.assertEqual(token_ids, [10, 11, ord("a"), ord("b"), ord("c")])
        self.assertEqual(trajectory["trajectory_format"], "mbpp")
        self.assertEqual(trajectory["used_generated_tokens"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

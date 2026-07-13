#!/usr/bin/env python3
"""Focused tests for adaptive Nemotron MTP drafting policy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_speculative import (  # noqa: E402
    accepted_draft_prefix,
    draft_choice_arrays,
    draft_margin,
    greedy_token_array,
    greedy_token_ids,
    margin_outcome_bins,
    matching_draft_prefix,
)


class MLXSpeculativeTest(unittest.TestCase):
    def test_margin_uses_largest_two_logits(self) -> None:
        self.assertAlmostEqual(draft_margin(mx.array([-2.0, 3.5, 1.25, 0.0])), 2.25)

    def test_draft_choice_combines_token_mapping_and_margin(self) -> None:
        token, margin = draft_choice_arrays(
            mx.array([-2.0, 3.5, 1.25, 0.0]),
            mx.array([11, 17, 23, 29], dtype=mx.int32),
        )
        mx.eval(token, margin)
        self.assertEqual(int(token), 17)
        self.assertAlmostEqual(float(margin), 2.25)

    def test_margin_outcome_bins_report_only_observed_buckets(self) -> None:
        self.assertEqual(
            margin_outcome_bins([(0.1, True), (0.3, False), (0.4, True), (9.0, True)]),
            "lt0.25:1/1,lt0.5:1/2,ge8:1/1",
        )

    def test_accepted_prefix_stops_at_first_rejection(self) -> None:
        logits = mx.array(
            [
                [0.0, 4.0, 1.0],
                [5.0, 1.0, 0.0],
                [0.0, 1.0, 6.0],
            ]
        )
        self.assertEqual(accepted_draft_prefix([1, 2, 2], logits), 1)
        self.assertEqual(accepted_draft_prefix([1, 0, 2], logits), 3)
        self.assertEqual(greedy_token_array(logits).tolist(), [1, 0, 2])
        self.assertEqual(greedy_token_ids(logits), [1, 0, 2])
        self.assertEqual(matching_draft_prefix([1, 2, 2], [1, 0, 2]), 1)

    def test_rejects_incomplete_verified_logits(self) -> None:
        with self.assertRaises(MetadataError):
            accepted_draft_prefix([0, 1], mx.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

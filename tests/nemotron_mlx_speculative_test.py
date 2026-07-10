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
from nemotron_mlx_speculative import accepted_draft_prefix, draft_margin  # noqa: E402


class MLXSpeculativeTest(unittest.TestCase):
    def test_margin_uses_largest_two_logits(self) -> None:
        self.assertAlmostEqual(draft_margin(mx.array([-2.0, 3.5, 1.25, 0.0])), 2.25)

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

    def test_rejects_incomplete_verified_logits(self) -> None:
        with self.assertRaises(MetadataError):
            accepted_draft_prefix([0, 1], mx.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

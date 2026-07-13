#!/usr/bin/env python3
"""Tests for recursive MTP hidden-adapter helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_mtp_hidden_adapter import apply_adapter, better_metrics  # noqa: E402


class MTPHiddenAdapterTest(unittest.TestCase):
    def test_zero_up_is_identity(self) -> None:
        hidden = mx.array([1.0, 2.0, 3.0, 4.0])
        up = mx.zeros((4, 2))
        down = mx.ones((2, 4))
        self.assertEqual(apply_adapter(hidden, up, down).tolist(), hidden.tolist())

    def test_checkpoint_selection_prioritizes_depth_two(self) -> None:
        current = {
            "epoch": 1,
            "depths": {"2": {"matches": 10}, "3": {"matches": 8}},
        }
        deeper_only = {
            "epoch": 2,
            "depths": {"2": {"matches": 9}, "3": {"matches": 12}},
        }
        tied_better_depth_three = {
            "epoch": 2,
            "depths": {"2": {"matches": 10}, "3": {"matches": 9}},
        }
        self.assertFalse(better_metrics(deeper_only, current))
        self.assertTrue(better_metrics(tied_better_depth_three, current))


if __name__ == "__main__":
    unittest.main(verbosity=2)

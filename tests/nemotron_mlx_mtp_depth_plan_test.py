#!/usr/bin/env python3
"""Focused tests for recursive-depth MTP expert planning."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_mtp_depth_plan import (  # noqa: E402
    parse_weights,
    rank_experts,
    report_cache_mode,
)


class MTPDepthPlanTest(unittest.TestCase):
    def test_report_cache_mode_defaults_old_reports_and_rejects_unknown_modes(self) -> None:
        self.assertEqual(report_cache_mode({}), "none")
        self.assertEqual(report_cache_mode({"cache_mode": "generated"}), "generated")
        self.assertEqual(report_cache_mode({"cache_mode": "prompt"}), "prompt")
        with self.assertRaises(MetadataError):
            report_cache_mode({"cache_mode": "shared-target"})

    def test_depth_weights_change_fixed_budget_priority(self) -> None:
        masses = {
            "1": {"0": 9.0, "1": 1.0},
            "2": {"0": 1.0, "1": 9.0},
        }
        counts = {
            "1": {"0": 9, "1": 1},
            "2": {"0": 1, "1": 9},
        }
        self.assertEqual(rank_experts(masses, counts, (2.0, 1.0))[0], [0, 1])
        self.assertEqual(rank_experts(masses, counts, (1.0, 2.0))[0], [1, 0])

    def test_parse_weights_rejects_empty_negative_and_zero(self) -> None:
        self.assertEqual(parse_weights("1,2.5,0"), (1.0, 2.5, 0.0))
        for value in ("", "-1,2", "0,0", "nan,1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_weights(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)

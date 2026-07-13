#!/usr/bin/env python3
"""Focused tests for provenance-bound plan comparison options."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_metadata import MetadataError
from nemotron_mlx_plan_compare import parse_router_revert_layers


class PlanCompareTest(unittest.TestCase):
    def test_parses_sorted_router_reversions(self) -> None:
        self.assertEqual(parse_router_revert_layers("1,3,8"), [1, 3, 8])
        self.assertEqual(parse_router_revert_layers(None), [])

    def test_rejects_duplicate_router_reversions(self) -> None:
        with self.assertRaises(MetadataError):
            parse_router_revert_layers("1,1")


if __name__ == "__main__":
    unittest.main()

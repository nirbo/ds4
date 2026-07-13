#!/usr/bin/env python3
"""Tests for the resident Nemotron layer profiler helpers."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_runtime_profile import parse_sizes  # noqa: E402


class MLXRuntimeProfileTest(unittest.TestCase):
    def test_sizes_are_sorted_unique_and_bounded(self) -> None:
        self.assertEqual(parse_sizes("3,1,2,2"), [1, 2, 3])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_sizes("0,2")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_sizes("2,9")


if __name__ == "__main__":
    unittest.main(verbosity=2)

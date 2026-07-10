#!/usr/bin/env python3
"""Tests for Nemotron resident-runtime memory preflight."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_resident import resident_requirement  # noqa: E402


class MLXResidentTest(unittest.TestCase):
    def test_requirement_includes_explicit_margin(self) -> None:
        self.assertEqual(resident_requirement(10 * 2**30, 1.5), int(11.5 * 2**30))


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Bounded-input checks for the Ornith-35 restored-prefix TTFT harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_cache_bench as bench
from ornith35_moe_reference import MoEError


class MLXCacheBenchTest(unittest.TestCase):
    def test_deterministic_tokens_resume_at_the_exact_suffix_boundary(self) -> None:
        prefix = bench.deterministic_tokens(0, 33)
        suffix = bench.deterministic_tokens(33, 17)
        self.assertEqual(prefix + suffix, bench.deterministic_tokens(0, 50))
        self.assertTrue(all(0 <= token < 248_320 for token in prefix + suffix))

    def test_suffix_parser_is_ordered_unique_and_positive(self) -> None:
        self.assertEqual(bench.parse_suffixes("1, 16,128"), (1, 16, 128))
        for invalid in ("", "0", "1,-1", "8,8", "one"):
            with self.subTest(invalid=invalid), self.assertRaises(MoEError):
                bench.parse_suffixes(invalid)


if __name__ == "__main__":
    unittest.main()

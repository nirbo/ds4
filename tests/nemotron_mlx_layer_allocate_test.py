#!/usr/bin/env python3
"""Tests for exact nonuniform expert-budget allocation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_layer_allocate import allocate_exact, category_proxy, parse_named_total  # noqa: E402


class LayerAllocateTest(unittest.TestCase):
    def test_exact_allocator_protects_sensitive_layer(self) -> None:
        layers = [1, 3]
        retained = {"r0": 10, "r50": 5}
        costs = {1: {"r0": 0.0, "r50": 100.0}, 3: {"r0": 0.0, "r50": 1.0}}
        allocation, cost = allocate_exact(layers, retained, costs, 15)
        self.assertEqual(allocation, {1: "r0", 3: "r50"})
        self.assertEqual(cost, 1.0)

    def test_category_proxy_is_root_sum_square(self) -> None:
        summary = {"r50": {"1": {"categories": {"code": 3.0}}, "3": {"categories": {"code": 4.0}}}}
        self.assertEqual(category_proxy(summary, {1: "r50", 3: "r50"})["code"], 5.0)

    def test_parses_named_exact_total(self) -> None:
        self.assertEqual(parse_named_total("r275=14860"), ("r275", 14860))


if __name__ == "__main__":
    unittest.main(verbosity=2)

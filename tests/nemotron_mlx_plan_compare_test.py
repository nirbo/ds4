#!/usr/bin/env python3
"""Focused tests for provenance-bound plan comparison options."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_metadata import MetadataError
from nemotron_mlx_plan_compare import (
    damp_router,
    parse_router_damp_layers,
    parse_router_revert_layers,
)


class PlanCompareTest(unittest.TestCase):
    def test_parses_sorted_router_reversions(self) -> None:
        self.assertEqual(parse_router_revert_layers("1,3,8"), [1, 3, 8])
        self.assertEqual(parse_router_revert_layers(None), [])

    def test_rejects_duplicate_router_reversions(self) -> None:
        with self.assertRaises(MetadataError):
            parse_router_revert_layers("1,1")

    def test_parses_sorted_router_damping(self) -> None:
        self.assertEqual(parse_router_damp_layers("3:0.75,8:0.5"), {3: 0.75, 8: 0.5})
        self.assertEqual(parse_router_damp_layers(None), {})

    def test_rejects_invalid_router_damping(self) -> None:
        for value in ("3", "3:1.1", "8:0.5,3:0.5", "3:0.5,3:0.4"):
            with self.subTest(value=value), self.assertRaises(MetadataError):
                parse_router_damp_layers(value)

    def test_damps_router_in_float32_and_returns_bf16(self) -> None:
        source = mx.array([[0.0, 2.0]], dtype=mx.bfloat16)
        trained = mx.array([[2.0, 4.0]], dtype=mx.bfloat16)
        actual = damp_router(trained, source, 0.25)
        mx.eval(actual)
        self.assertEqual(actual.dtype, mx.bfloat16)
        self.assertEqual(actual.astype(mx.float32).tolist(), [[0.5, 2.5]])


if __name__ == "__main__":
    unittest.main()

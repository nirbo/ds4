#!/usr/bin/env python3
"""Tests for behavioral expert-to-prototype mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_proxy_plan import build_layer_mapping  # noqa: E402


class ProxyPlanTest(unittest.TestCase):
    def test_retained_experts_map_to_self_and_removed_uses_supported_similarity(self) -> None:
        counts = np.zeros((4, 4), dtype=np.uint32)
        cosine = np.zeros((4, 4), dtype=np.float32)
        counts[1, 0] = counts[0, 1] = 3
        counts[1, 2] = counts[2, 1] = 4
        cosine[1, 0] = cosine[0, 1] = 1.2
        cosine[1, 2] = cosine[2, 1] = 3.2
        categories = np.array([[2, 1, 3, 0], [0, 2, 1, 0]], dtype=np.uint32)
        mapping, summary = build_layer_mapping([0, 2], counts, cosine, categories, 2)
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[1], 2)
        self.assertEqual(mapping[2], 2)
        self.assertIn(mapping[3], (0, 2))
        self.assertEqual(summary["removed"], 2)
        self.assertEqual(summary["unsupported"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

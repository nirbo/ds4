#!/usr/bin/env python3
"""Tests for proxy-expert co-occurrence calibration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_proxy_calibrate import initialize_arrays, merge_observations  # noqa: E402


class ProxyCalibrationTest(unittest.TestCase):
    def test_merge_preserves_token_pairing_and_category_counts(self) -> None:
        config = {
            "n_routed_experts": 4,
            "hybrid_override_pattern": "E",
        }
        arrays = initialize_arrays(config, ["code", "reasoning"])
        routing = {
            0: {
                "indices": [0, 2, 1, 2],
                "scores": [0.75, 0.25, 0.6, 0.4],
                "pair_cosines": [1.0, 0.5, 0.5, 1.0, 1.0, -0.25, -0.25, 1.0],
            }
        }
        merge_observations(arrays, routing, category_index=0, top_k=2)
        counts = arrays["layer_000_pair_counts"]
        cosines = arrays["layer_000_cosine_sums"]
        category = arrays["layer_000_category_counts"]
        self.assertEqual(counts[0, 2], 1)
        self.assertEqual(counts[1, 2], 1)
        self.assertEqual(counts[0, 1], 0)
        self.assertAlmostEqual(float(cosines[0, 2]), 0.5)
        self.assertAlmostEqual(float(cosines[1, 2]), -0.25)
        self.assertEqual(category[0].tolist(), [1, 1, 2, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

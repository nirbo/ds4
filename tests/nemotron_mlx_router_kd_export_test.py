#!/usr/bin/env python3
"""Focused tests for exact Router KD composition export."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_router_kd_export import compose_routers


class RouterKdExportTest(unittest.TestCase):
    def test_composition_reverts_only_requested_layers(self) -> None:
        trained = {"1": mx.ones((2, 2)), "3": mx.ones((2, 2)) * 3}
        source = {"1": mx.zeros((2, 2)), "3": mx.zeros((2, 2))}
        result = compose_routers(trained, source, [1])
        np.testing.assert_array_equal(np.asarray(result["1"]), np.zeros((2, 2)))
        np.testing.assert_array_equal(np.asarray(result["3"]), np.full((2, 2), 3))


if __name__ == "__main__":
    unittest.main()

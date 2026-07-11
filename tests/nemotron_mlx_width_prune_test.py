#!/usr/bin/env python3
"""Tests for aligned expert-width block selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_moe import NVFP4ExpertMLP, NVFP4SwitchWeight  # noqa: E402
from nemotron_mlx_width_prune import select_blocks, slice_experts  # noqa: E402


class WidthPruneTest(unittest.TestCase):
    def test_select_blocks_is_ranked_but_stored_in_source_order(self) -> None:
        importance = np.array([[1, 4, 3, 2], [9, 1, 8, 2]], dtype=np.float64)
        np.testing.assert_array_equal(select_blocks(importance, 2), [[1, 2], [0, 2]])

    def test_select_blocks_breaks_ties_by_source_order(self) -> None:
        importance = np.ones((1, 4), dtype=np.float64)
        np.testing.assert_array_equal(select_blocks(importance, 2), [[0, 1]])

    def test_slice_experts_copies_aligned_up_rows_and_down_columns(self) -> None:
        up_weight = np.arange(2 * 32 * 8, dtype=np.uint16).reshape(2, 32, 8).astype(np.uint8)
        up_scales = np.arange(2 * 32, dtype=np.uint16).reshape(2, 32, 1).astype(np.uint8)
        down_weight = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16).astype(np.uint8)
        down_scales = np.arange(2 * 16 * 2, dtype=np.uint16).reshape(2, 16, 2).astype(np.uint8)
        experts = NVFP4ExpertMLP(
            NVFP4SwitchWeight(mx.array(up_weight), mx.array(up_scales), mx.ones((2,))),
            NVFP4SwitchWeight(mx.array(down_weight), mx.array(down_scales), mx.ones((2,))),
        )
        sliced = slice_experts(experts, np.array([[1], [0]], dtype=np.int32))
        mx.eval(sliced.up.weight, sliced.up.scales, sliced.down.weight, sliced.down.scales)
        np.testing.assert_array_equal(np.asarray(sliced.up.weight)[0], up_weight[0, 16:32])
        np.testing.assert_array_equal(np.asarray(sliced.up.weight)[1], up_weight[1, 0:16])
        np.testing.assert_array_equal(np.asarray(sliced.up.scales)[0], up_scales[0, 16:32])
        np.testing.assert_array_equal(np.asarray(sliced.down.weight)[0], down_weight[0, :, 8:16])
        np.testing.assert_array_equal(np.asarray(sliced.down.weight)[1], down_weight[1, :, 0:8])
        np.testing.assert_array_equal(np.asarray(sliced.down.scales)[0], down_scales[0, :, 1:2])


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for Nemotron resident-runtime memory preflight."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_resident import (  # noqa: E402
    resident_requirement,
    restore_caches,
    snapshot_caches,
)


class MLXResidentTest(unittest.TestCase):
    def test_requirement_includes_explicit_margin(self) -> None:
        self.assertEqual(resident_requirement(10 * 2**30, 1.5), int(11.5 * 2**30))

    def test_cache_snapshot_restores_recurrent_state_and_kv_offset(self) -> None:
        recurrent = ArraysCache(size=2)
        recurrent[0] = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
        recurrent[1] = mx.array([[[[3.0, 4.0]]]], dtype=mx.float32)
        attention = KVCache()
        initial_key = mx.array([[[[5.0, 6.0]]]], dtype=mx.float32)
        initial_value = mx.array([[[[7.0, 8.0]]]], dtype=mx.float32)
        attention.update_and_fetch(initial_key, initial_value)
        mx.eval(*recurrent.state, attention.keys, attention.values)

        caches = {0: recurrent, 1: attention}
        snapshot = snapshot_caches(caches)
        recurrent[0] = mx.array([[[9.0, 10.0]]], dtype=mx.float32)
        recurrent[1] = mx.array([[[[11.0, 12.0]]]], dtype=mx.float32)
        attention.update_and_fetch(initial_key + 10.0, initial_value + 10.0)
        mx.eval(*recurrent.state, attention.keys, attention.values)

        restore_caches(caches, snapshot)
        self.assertEqual(recurrent[0].tolist(), [[[1.0, 2.0]]])
        self.assertEqual(recurrent[1].tolist(), [[[[3.0, 4.0]]]])
        self.assertEqual(attention.offset, 1)
        self.assertEqual(
            attention.keys[..., : attention.offset, :].tolist(),
            initial_key.tolist(),
        )
        self.assertEqual(
            attention.values[..., : attention.offset, :].tolist(),
            initial_value.tolist(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

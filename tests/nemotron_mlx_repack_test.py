#!/usr/bin/env python3
"""Tests for fixed-size incremental Nemotron repacking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_repack import changed_group_names  # noqa: E402


class RepackTest(unittest.TestCase):
    def test_only_changed_moe_mappings_require_rewrite(self) -> None:
        base = {1: {2: 0, 4: 1}, 3: {1: 0, 5: 1}}
        target = {1: {2: 0, 4: 1}, 3: {1: 0, 6: 1}}
        self.assertEqual(changed_group_names(base, target), {"layer-003"})

    def test_rejects_different_layer_catalogs(self) -> None:
        with self.assertRaises(MetadataError):
            changed_group_names({1: {2: 0}}, {3: {2: 0}})


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Focused tests for bounded Ornith-35 metadata fetching."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ornith35" / "tools" / "ornith35_fetch_metadata.py"
SPEC = importlib.util.spec_from_file_location("ornith35_fetch_metadata", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FetchMetadataTest(unittest.TestCase):
    def test_finds_mtp_shards(self) -> None:
        index = {
            "weight_map": {
                "model.layer.weight": "model-00001-of-00002.safetensors",
                "mtp.layers.0.weight": "model-00002-of-00002.safetensors",
                "mtp.norm.weight": "model-00002-of-00002.safetensors",
            }
        }
        analysis = MODULE.analyze_mtp_index(index)
        self.assertEqual(analysis["tensor_count"], 2)
        self.assertEqual(analysis["shards"], ["model-00002-of-00002.safetensors"])
        self.assertEqual(
            analysis["tensor_shards"]["mtp.layers.0.weight"],
            "model-00002-of-00002.safetensors",
        )

    def test_rejects_index_without_mtp(self) -> None:
        with self.assertRaisesRegex(MODULE.FetchError, "no mtp"):
            MODULE.analyze_mtp_index({"weight_map": {"model.weight": "model.safetensors"}})

    def test_sums_mtp_payload_once(self) -> None:
        analysis = {
            "tensor_names": ["mtp.a", "mtp.b"],
            "tensor_count": 2,
            "tensor_shards": {"mtp.a": "a", "mtp.b": "b"},
            "shards": ["a", "b"],
        }
        headers = {
            "a": {"mtp.a": {"data_offsets": [10, 30]}},
            "b": {"mtp.b": {"data_offsets": [2, 14]}},
        }
        self.assertEqual(MODULE.mtp_payload_from_headers(analysis, headers), 32)

    def test_rejects_wrong_header_coverage(self) -> None:
        analysis = {
            "tensor_names": ["mtp.a"],
            "tensor_count": 1,
            "tensor_shards": {"mtp.a": "a"},
            "shards": ["a", "b"],
        }
        headers = {
            "b": {"mtp.a": {"data_offsets": [0, 1]}},
            "a": {},
        }
        with self.assertRaisesRegex(MODULE.FetchError, "coverage mismatch"):
            MODULE.mtp_payload_from_headers(analysis, headers)

    def test_rejects_duplicate_header_coverage(self) -> None:
        analysis = {
            "tensor_names": ["mtp.a"],
            "tensor_count": 1,
            "tensor_shards": {"mtp.a": "a"},
            "shards": ["a", "b"],
        }
        headers = {
            "a": {"mtp.a": {"data_offsets": [0, 1]}},
            "b": {"mtp.a": {"data_offsets": [0, 1]}},
        }
        with self.assertRaisesRegex(MODULE.FetchError, "multiple shard headers"):
            MODULE.mtp_payload_from_headers(analysis, headers)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for ModelOpt NVFP4 scalar decode semantics."""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
import nemotron_nvfp4 as nvfp4  # noqa: E402


def write_fixture(path: Path) -> str:
    prefix = "backbone.layers.0.mixer.experts.0.up_proj"
    packed = bytes((high << 4) | low for low, high in zip(range(0, 16, 2), range(1, 16, 2)))
    scales = bytes([0x38])
    global_scale = struct.pack("<f", 2.0)
    payloads = {
        prefix + ".weight": ("U8", [1, 8], packed),
        prefix + ".weight_scale": ("F8_E4M3", [1, 1], scales),
        prefix + ".weight_scale_2": ("F32", [], global_scale),
    }
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in payloads.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return prefix


class NVFP4Test(unittest.TestCase):
    def test_e2m1_exhaustive_table(self) -> None:
        expected = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        self.assertEqual([nvfp4.decode_e2m1(value) for value in range(16)], expected)

    def test_e4m3fn_reference_values(self) -> None:
        self.assertEqual(nvfp4.decode_e4m3fn(0x00), 0.0)
        self.assertEqual(nvfp4.decode_e4m3fn(0x01), 2**-9)
        self.assertEqual(nvfp4.decode_e4m3fn(0x38), 1.0)
        self.assertEqual(nvfp4.decode_e4m3fn(0x3C), 1.5)
        self.assertEqual(nvfp4.decode_e4m3fn(0x7E), 448.0)
        self.assertEqual(nvfp4.decode_e4m3fn(0xFE), -448.0)
        self.assertTrue(math.isnan(nvfp4.decode_e4m3fn(0x7F)))

    def test_low_nibble_is_even_input_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            prefix = write_fixture(path)
            with nvfp4.SafetensorsFile(path) as shard:
                weight = nvfp4.NVFP4Weight(shard, prefix)
                self.assertEqual((weight.rows, weight.columns), (1, 16))
                expected = [nvfp4.decode_e2m1(value) * 2.0 for value in range(16)]
                self.assertEqual([weight.value(0, column) for column in range(16)], expected)
                self.assertEqual(weight.matvec_row(0, [1.0] * 16), math.fsum(expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)

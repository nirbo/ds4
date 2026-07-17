#!/usr/bin/env python3
"""Focused tests for the Ornith-35 NVFP4 reference decoder."""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ornith35" / "tools" / "ornith35_nvfp4.py"
SPEC = importlib.util.spec_from_file_location("ornith35_nvfp4", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_fixture(path: Path, *, bad_scale_shape: bool = False) -> str:
    prefix = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    packed = bytes(low | ((low + 1) << 4) for low in range(0, 16, 2)) + bytes(
        [0x22] * 8
    )
    scales = bytes([0x38, 0x40])
    global_scale = struct.pack("<f", 2.0)
    tensors = [
        (prefix + ".weight_packed", "U8", [2, 8], packed),
        (
            prefix + ".weight_scale",
            "F8_E4M3",
            [2, 2] if bad_scale_shape else [2, 1],
            scales,
        ),
        (prefix + ".weight_global_scale", "F32", [1], global_scale),
    ]
    header: dict = {}
    offset = 0
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    return prefix


class NVFP4Test(unittest.TestCase):
    def test_decodes_all_e2m1_values(self) -> None:
        expected = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        self.assertEqual(tuple(MODULE.decode_e2m1(value) for value in range(8)), expected)
        self.assertEqual(
            tuple(MODULE.decode_e2m1(value) for value in range(8, 16)),
            tuple(-value for value in expected),
        )

    def test_decodes_e4m3fn_edges(self) -> None:
        self.assertEqual(MODULE.decode_e4m3fn(0x00), 0.0)
        self.assertEqual(MODULE.decode_e4m3fn(0x01), 2**-9)
        self.assertEqual(MODULE.decode_e4m3fn(0x38), 1.0)
        self.assertEqual(MODULE.decode_e4m3fn(0x7E), 448.0)
        self.assertEqual(MODULE.decode_e4m3fn(0xB8), -1.0)
        self.assertTrue(math.isnan(MODULE.decode_e4m3fn(0x7F)))

    def test_reads_low_then_high_nibbles_and_scales(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.safetensors"
            prefix = write_fixture(path)
            with MODULE.SafetensorsFile(path) as source:
                weight = MODULE.NVFP4Weight(source, prefix)
                self.assertEqual((weight.rows, weight.columns), (2, 16))
                self.assertEqual(weight.global_scale, 2.0)
                self.assertEqual(weight.value(0, 0), 0.0)
                self.assertEqual(weight.value(0, 1), 0.25)
                self.assertEqual(weight.value(0, 2), 0.5)
                self.assertEqual(weight.value(0, 15), -3.0)
                self.assertEqual(weight.value(1, 0), 1.0)
                self.assertEqual(weight.matvec_row(1, [1.0] * 16), 16.0)

    def test_rejects_weight_scale_block_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.safetensors"
            prefix = write_fixture(path, bad_scale_shape=True)
            with MODULE.SafetensorsFile(path) as source:
                with self.assertRaisesRegex(MODULE.NVFP4Error, "scale payload mismatch"):
                    MODULE.NVFP4Weight(source, prefix)


if __name__ == "__main__":
    unittest.main(verbosity=2)

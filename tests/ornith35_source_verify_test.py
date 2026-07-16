#!/usr/bin/env python3
"""Focused tests for Ornith-35 source acceptance."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ornith35" / "tools" / "ornith35_source_verify.py"
SPEC = importlib.util.spec_from_file_location("ornith35_source_verify", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture(root: Path) -> tuple[Path, Path, dict]:
    header = b'{"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}'
    payload = b"\x01\x02\x03\x04"
    source = root / "model.safetensors"
    source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    header_path = root / "model.safetensors.header.json"
    header_path.write_bytes(header)
    weight = {
        "name": "model.safetensors",
        "file_bytes": source.stat().st_size,
        "header_bytes": len(header),
        "payload_bytes": len(payload),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    return source, header_path, weight


class SourceVerifyTest(unittest.TestCase):
    def test_atomic_state_write_replaces_without_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text("old", encoding="utf-8")
            MODULE.atomic_write_json(path, {"status": "verified"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "verified"})
            self.assertFalse(path.with_suffix(".json.part").exists())

    def test_verifies_size_header_payload_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, header, weight = fixture(Path(temporary))
            result = MODULE.verify_weight_file(source, header, weight, progress_bytes=0)
            self.assertEqual(result["bytes"], weight["file_bytes"])
            self.assertEqual(result["payload_bytes"], 4)
            self.assertEqual(result["sha256"], weight["sha256"])

    def test_rejects_header_drift_before_full_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, header, weight = fixture(Path(temporary))
            header.write_bytes(header.read_bytes().replace(b"U8", b"I8"))
            with self.assertRaisesRegex(MODULE.VerificationError, "header differs"):
                MODULE.verify_weight_file(source, header, weight, progress_bytes=0)

    def test_rejects_payload_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, header, weight = fixture(Path(temporary))
            damaged = bytearray(source.read_bytes())
            damaged[-1] ^= 0xFF
            source.write_bytes(damaged)
            with self.assertRaisesRegex(MODULE.VerificationError, "SHA-256 mismatch"):
                MODULE.verify_weight_file(source, header, weight, progress_bytes=0)

    def test_validates_metadata_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, header, weight = fixture(root)
            config = root / "config.json"
            config.write_text(json.dumps({"model_type": "fixture"}), encoding="utf-8")
            state = {
                "format": MODULE.SOURCE_STATE_FORMAT,
                "repository": "fixture/repository",
                "revision": "fixture-revision",
                "metadata_files": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (config, header)
                },
                "weight": weight,
            }
            self.assertEqual(
                MODULE.validate_metadata_state(state, root, strict_target=False),
                weight,
            )
            broken = copy.deepcopy(state)
            broken["metadata_files"]["config.json"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(MODULE.VerificationError, "metadata hash mismatch"):
                MODULE.validate_metadata_state(broken, root, strict_target=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)

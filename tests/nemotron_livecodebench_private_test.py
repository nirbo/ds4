#!/usr/bin/env python3
"""Tests for bounded LiveCodeBench private-test materialization."""

from __future__ import annotations

import base64
import io
import json
import pickle
import sys
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_livecodebench_private import decode_private_cases  # noqa: E402
from nemotron_metadata import MetadataError  # noqa: E402


CASES = [{"input": "1", "output": "2", "testtype": "stdin"}]


class LiveCodeBenchPrivateTest(unittest.TestCase):
    def test_decodes_json_private_cases(self) -> None:
        self.assertEqual(decode_private_cases(json.dumps(CASES)), CASES)

    def test_decodes_restricted_legacy_payload(self) -> None:
        payload = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(CASES)))).decode()
        self.assertEqual(decode_private_cases(payload), CASES)

    def test_rejects_pickle_global_objects(self) -> None:
        payload = base64.b64encode(zlib.compress(pickle.dumps(Path("unsafe")))).decode()
        with self.assertRaisesRegex(MetadataError, "forbidden"):
            decode_private_cases(payload)

    def test_rejects_unsupported_test_type(self) -> None:
        bad = [{"input": "1", "output": "2", "testtype": "other"}]
        with self.assertRaisesRegex(MetadataError, "unsupported"):
            decode_private_cases(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)

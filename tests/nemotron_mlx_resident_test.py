#!/usr/bin/env python3
"""Tests for Nemotron resident-runtime memory preflight."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_resident import (  # noqa: E402
    preflight,
    resident_requirement,
    restore_caches,
    snapshot_caches,
)
from nemotron_prune_materialize import sha256_file  # noqa: E402


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

    def test_preflight_accounts_for_quantized_mtp_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            sidecar = root / "sidecar"
            head = root / "head"
            for directory in (model, sidecar, head):
                directory.mkdir()
            (model / "nemotron_mlx_pack_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-runtime-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "payload_bytes": 10 * 2**30,
                    }
                )
            )
            (sidecar / "nemotron_mtp_pack_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-mtp-sidecar-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "payload_bytes": 2 * 2**30,
                    }
                )
            )
            (head / "nemotron_mtp_head_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-mtp-head-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "source_report_sha256": sha256_file(
                            model / "nemotron_mlx_pack_report.json"
                        ),
                        "payload_bytes": 1 * 2**30,
                    }
                )
            )
            device = {
                "max_recommended_working_set_size": 16 * 2**30,
                "memory_size": 32 * 2**30,
            }
            with (
                patch("nemotron_mlx_resident.mx.device_info", return_value=device),
                patch("nemotron_mlx_resident.iogpu_wired_limit_bytes", return_value=0),
            ):
                result = preflight(model, 0.5, sidecar, head)
            self.assertEqual(result["payload_bytes"], 13 * 2**30)
            self.assertEqual(result["mtp_head_payload_gib"], 1.0)
            self.assertEqual(result["required_gib"], 13.5)
            self.assertTrue(result["safe_to_attempt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

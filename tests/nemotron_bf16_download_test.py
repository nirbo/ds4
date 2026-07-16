#!/usr/bin/env python3
"""Tests for bounded, resumable BF16 layer downloads."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_bf16_download import (  # noqa: E402
    STATE_FORMAT,
    disk_preflight,
    download_environment,
    load_or_create_state,
    state_identity,
    validate_file,
)


class BF16DownloadTest(unittest.TestCase):
    def test_state_and_direct_file_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            payload = b"immutable-bf16-test-payload"
            digest = hashlib.sha256(payload).hexdigest()
            contract = {
                "repository": "test/repository",
                "source_revision": "a" * 40,
                "layer": 1,
                "required_shards": [
                    {"name": "model-00001-of-00001.safetensors", "bytes": len(payload), "sha256": digest}
                ],
            }
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            identity = state_identity(contract_path, contract, raw)
            state_path = root / "state.json"
            state = load_or_create_state(state_path, identity)
            self.assertEqual(state["format"], STATE_FORMAT)
            entry = state["files"]["model-00001-of-00001.safetensors"]
            shard = raw / "model-00001-of-00001.safetensors"
            shard.write_bytes(payload)
            validate_file(shard, entry)
            preflight = disk_preflight(raw, state, 0.0)
            self.assertGreaterEqual(preflight["free_bytes"], preflight["remaining_download_bytes"])

    def test_cache_environment_is_job_local_and_chunkless(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "job"
            original_home = root / "original-home"
            original_home.mkdir()
            token = original_home / "token"
            token.write_text("secret-not-copied", encoding="utf-8")
            environment = download_environment(job, {"HF_HOME": str(original_home)})
            self.assertEqual(environment["HF_HOME"], str((job / "hf-home").resolve()))
            self.assertEqual(environment["HF_XET_CACHE"], str((job / "hf-xet").resolve()))
            self.assertEqual(environment["HF_XET_CHUNK_CACHE_SIZE_BYTES"], "0")
            self.assertEqual(environment["HF_XET_HIGH_PERFORMANCE"], "1")
            self.assertEqual(environment["HF_TOKEN_PATH"], str(token.resolve()))
            self.assertFalse((job / "hf-home" / "token").exists())
            anonymous = download_environment(
                job,
                {"HF_HOME": str(root / "missing-home"), "HF_TOKEN_PATH": str(root / "missing-token")},
            )
            self.assertNotIn("HF_TOKEN_PATH", anonymous)


if __name__ == "__main__":
    unittest.main(verbosity=2)

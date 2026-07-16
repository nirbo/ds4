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
    download_profile,
    load_or_create_state,
    partial_path,
    sha256_range,
    state_identity,
    validate_file,
    validate_partial,
)
from nemotron_metadata import MetadataError  # noqa: E402


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
            profile = download_profile(4, 64, 256, 128, 300)
            hub_runtime = {
                "python": "/test/python",
                "huggingface_hub": "1.23.0",
                "hf_xet": "1.5.1",
            }
            remote_files = {
                "model-00001-of-00001.safetensors": {
                    "commit": "a" * 40,
                    "etag": digest,
                    "bytes": len(payload),
                    "xet_file_hash": "b" * 64,
                    "refresh_route_sha256": "c" * 64,
                }
            }
            identity = state_identity(
                contract_path,
                contract,
                raw,
                profile,
                hub_runtime,
                remote_files,
            )
            state_path = root / "state.json"
            state = load_or_create_state(state_path, identity)
            self.assertEqual(state["format"], STATE_FORMAT)
            self.assertEqual(state["download_profile"], profile)
            self.assertEqual(state["hub_runtime"], hub_runtime)
            self.assertEqual(state["remote_files"], remote_files)
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
            profile = download_profile(4, 64, 256, 128, 300)
            environment = download_environment(
                job,
                profile,
                {"HF_HOME": str(original_home), "HF_XET_HIGH_PERFORMANCE": "1"},
            )
            self.assertEqual(environment["HF_HOME"], str((job / "hf-home").resolve()))
            self.assertEqual(environment["HF_XET_CACHE"], str((job / "hf-xet").resolve()))
            self.assertEqual(environment["HF_XET_CHUNK_CACHE_SIZE_BYTES"], "0")
            self.assertNotIn("HF_XET_HIGH_PERFORMANCE", environment)
            self.assertEqual(environment["HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY"], "false")
            self.assertEqual(environment["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"], "4")
            self.assertEqual(
                environment["HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE"],
                "64mb",
            )
            self.assertEqual(
                environment["HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE"],
                "256mb",
            )
            self.assertEqual(environment["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"], "1")
            self.assertEqual(environment["HF_TOKEN_PATH"], str(token.resolve()))
            self.assertFalse((job / "hf-home" / "token").exists())
            anonymous = download_environment(
                job,
                profile,
                {"HF_HOME": str(root / "missing-home"), "HF_TOKEN_PATH": str(root / "missing-token")},
            )
            self.assertNotIn("HF_TOKEN_PATH", anonymous)

    def test_partial_ranges_are_hash_bound_and_uncommitted_tail_is_removed(self) -> None:
        class Log:
            def __init__(self) -> None:
                self.rows = []

            def write(self, value: str) -> None:
                self.rows.append(value)

        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            raw.mkdir()
            name = "model-00001-of-00001.safetensors"
            path = partial_path(raw, name)
            path.parent.mkdir()
            committed = b"first-range" * 1024
            path.write_bytes(committed + b"uncommitted-tail")
            entry = {
                "expected_bytes": len(committed) + 4096,
                "completed_bytes": len(committed),
                "ranges": [
                    {
                        "start": 0,
                        "end": len(committed),
                        "bytes": len(committed),
                        "sha256": hashlib.sha256(committed).hexdigest(),
                    }
                ],
            }
            log = Log()
            validate_partial(path, entry, log)
            self.assertEqual(path.stat().st_size, len(committed))
            self.assertEqual(sha256_range(path, 0, len(committed)), entry["ranges"][0]["sha256"])
            self.assertTrue(any("recover-truncate" in row for row in log.rows))
            with path.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"X")
            with self.assertRaises(MetadataError):
                validate_partial(path, entry, log)


if __name__ == "__main__":
    unittest.main(verbosity=2)

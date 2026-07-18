#!/usr/bin/env python3
"""Resumption, integrity, and cleanup tests for Ornith-35 MTP extraction."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mtp_extract as extract


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> bytes:
    rendered = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(rendered)
    return rendered


def make_safetensors(
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> tuple[bytes, bytes]:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, value = tensors[name]
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    rendered = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    header_bytes = rendered + b" " * (-len(rendered) % 8)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + payload, header_bytes


def bf16_values(elements: int, seed: int) -> bytes:
    return b"".join(struct.pack("<H", (seed + index * 17) & 0xFFFF) for index in range(elements))


def read_safetensors(path: Path) -> dict[str, bytes]:
    with path.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_bytes))
        payload_base = 8 + header_bytes
        result = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            handle.seek(payload_base + start)
            result[name] = handle.read(end - start)
        return result


def make_fixture(root: Path) -> tuple[extract.ExtractionProfile, dict[str, bytes]]:
    metadata_dir = root / "metadata-mtp-source"
    raw_dir = root / "source-mtp-raw"
    metadata_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    shard_values = (
        {
            "model.unrelated": ("BF16", (2,), bf16_values(2, 3)),
            "mtp.fc.weight": ("BF16", (2, 4), bf16_values(8, 11)),
            "mtp.layers.0.self_attn.q_proj.weight": (
                "BF16",
                (4, 2),
                bf16_values(8, 29),
            ),
        },
        {
            "mtp.layers.0.input_layernorm.weight": (
                "BF16",
                (2,),
                bf16_values(2, 47),
            ),
            "mtp.layers.0.mlp.experts.0.down_proj.weight": (
                "BF16",
                (2, 2),
                bf16_values(4, 59),
            ),
        },
    )
    shard_names = ("fixture-00001.safetensors", "fixture-00002.safetensors")
    header_names = ("fixture-00001.header.json", "fixture-00002.header.json")
    pinned_shards = []
    expected_tensors: dict[str, bytes] = {}
    shard_inventory = []
    weight_map: dict[str, str] = {}
    for shard_name, header_name, tensors in zip(shard_names, header_names, shard_values):
        shard_bytes, header_bytes = make_safetensors(tensors)
        (raw_dir / shard_name).write_bytes(shard_bytes)
        (metadata_dir / header_name).write_bytes(header_bytes)
        mtp_values = {
            name: value for name, (_, _, value) in tensors.items() if name.startswith("mtp.")
        }
        expected_tensors.update(mtp_values)
        weight_map.update({name: shard_name for name in tensors})
        mtp_payload = sum(len(value) for value in mtp_values.values())
        pinned = extract.PinnedShard(
            name=shard_name,
            file_bytes=len(shard_bytes),
            sha256=sha256(shard_bytes),
            header_bytes=len(header_bytes),
            header_sha256=sha256(header_bytes),
            payload_bytes=len(shard_bytes) - 8 - len(header_bytes),
            mtp_tensor_count=len(mtp_values),
            mtp_payload_bytes=mtp_payload,
        )
        pinned_shards.append(pinned)
        shard_inventory.append(
            {
                "name": shard_name,
                "file_bytes": pinned.file_bytes,
                "sha256": pinned.sha256,
                "header_bytes": pinned.header_bytes,
                "header_sha256": pinned.header_sha256,
                "header_file": header_name,
                "payload_bytes": pinned.payload_bytes,
            }
        )

    config_bytes = write_json(metadata_dir / "config.json", {"fixture": True})
    index_bytes = write_json(
        metadata_dir / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": weight_map},
    )
    metadata_files = {
        "config.json": {
            "bytes": len(config_bytes),
            "sha256": sha256(config_bytes),
        },
        "model.safetensors.index.json": {
            "bytes": len(index_bytes),
            "sha256": sha256(index_bytes),
        },
    }
    for header_name in header_names:
        value = (metadata_dir / header_name).read_bytes()
        metadata_files[header_name] = {
            "bytes": len(value),
            "sha256": sha256(value),
        }
    profile = extract.ExtractionProfile(
        key="mtp-source",
        repository="fixture/qwen-mtp",
        revision="1" * 40,
        tensor_count=len(expected_tensors),
        payload_bytes=sum(len(value) for value in expected_tensors.values()),
        shards=tuple(pinned_shards),
    )
    write_json(
        metadata_dir / "source-state.json",
        {
            "format": extract.METADATA_FORMAT,
            "profile": profile.key,
            "repository": profile.repository,
            "revision": profile.revision,
            "metadata_files": metadata_files,
            "mtp": {
                "tensor_count": profile.tensor_count,
                "payload_bytes": profile.payload_bytes,
                "shards": shard_inventory,
            },
        },
    )
    return profile, expected_tensors


class MTPExtractionTest(unittest.TestCase):
    def test_plan_validates_metadata_without_creating_extraction_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            output = io.StringIO()
            with redirect_stdout(output):
                extract.print_extraction_plan(root, profile)
            self.assertIn("shards=2", output.getvalue())
            self.assertIn("conservative_peak_bytes=", output.getvalue())
            self.assertFalse((root / "source-mtp-state.json").exists())
            self.assertFalse((root / "source-mtp").exists())

    def test_resumes_one_shard_deletes_only_after_verification_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, expected = make_fixture(root)
            first = extract.extract_available_shards(
                root,
                profile=profile,
                delete_source=True,
            )
            self.assertEqual(first["status"], "partial")
            self.assertEqual(len(first["completed_shards"]), 1)
            self.assertFalse((root / "source-mtp-raw" / profile.shards[0].name).exists())
            self.assertTrue((root / "source-mtp-raw" / profile.shards[1].name).exists())
            self.assertTrue((root / "source-mtp" / "mtp.safetensors.part").exists())

            completed = extract.extract_available_shards(
                root,
                profile=profile,
                delete_source=True,
            )
            self.assertEqual(completed["status"], "complete")
            self.assertFalse((root / "source-mtp-raw" / profile.shards[1].name).exists())
            output = root / "source-mtp" / "mtp.safetensors"
            self.assertEqual(read_safetensors(output), expected)
            self.assertEqual(
                extract.extract_available_shards(root, profile=profile)["status"],
                "complete",
            )

    def test_rejects_corrupt_source_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            source = root / "source-mtp-raw" / profile.shards[0].name
            value = bytearray(source.read_bytes())
            value[-1] ^= 0xFF
            source.write_bytes(value)
            with self.assertRaisesRegex(extract.MTPExtractionError, "SHA-256"):
                extract.extract_available_shards(
                    root,
                    profile=profile,
                    delete_source=True,
                )
            self.assertTrue(source.exists())
            state = extract.load_json(root / "source-mtp-state.json")
            self.assertEqual(state["completed_shards"], {})

    def test_rejects_source_identity_change_after_full_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            catalog = extract.build_catalog(root, profile)
            shard = profile.shards[0]
            source = root / "source-mtp-raw" / shard.name
            identity = extract._verify_source_shard(source, shard)
            part = root / "source-mtp" / "mtp.safetensors.part"
            extract._initialize_output(part, catalog)
            current = source.stat()
            os.utime(
                source,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1),
            )
            with self.assertRaisesRegex(extract.MTPExtractionError, "changed before"):
                extract._copy_shard_tensors(
                    source,
                    part,
                    shard,
                    catalog.by_shard[shard.name],
                    catalog,
                    identity,
                )

    def test_resume_rejects_corrupt_verified_output_before_next_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            extract.extract_available_shards(root, profile=profile)
            part = root / "source-mtp" / "mtp.safetensors.part"
            value = bytearray(part.read_bytes())
            value[-profile.payload_bytes] ^= 0x80
            part.write_bytes(value)
            second = root / "source-mtp-raw" / profile.shards[1].name
            with self.assertRaisesRegex(extract.MTPExtractionError, "tensor drift"):
                extract.extract_available_shards(root, profile=profile)
            self.assertTrue(second.exists())

    def test_recovers_atomic_rename_before_complete_state_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            completed = extract.extract_available_shards(
                root,
                profile=profile,
                max_shards=2,
            )
            self.assertEqual(completed["status"], "complete")
            state_path = root / "source-mtp-state.json"
            interrupted = extract.load_json(state_path)
            interrupted["status"] = "partial"
            interrupted.pop("completed_at")
            interrupted.pop("output_sha256")
            write_json(state_path, interrupted)
            recovered = extract.extract_available_shards(root, profile=profile)
            self.assertEqual(recovered["status"], "complete")
            self.assertIn("output_sha256", recovered)

    def test_shard_retry_finalizes_state_committed_before_output_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            extract.extract_available_shards(root, profile=profile)
            with mock.patch.object(
                extract,
                "_complete_output",
                side_effect=RuntimeError("simulated stop before final output"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated stop"):
                    extract.extract_available_shards(root, profile=profile)
            interrupted = extract.load_json(root / "source-mtp-state.json")
            self.assertEqual(interrupted["status"], "partial")
            self.assertEqual(len(interrupted["completed_shards"]), 2)
            recovered = extract.extract_available_shards(
                root,
                profile=profile,
                shard_name=profile.shards[0].name,
            )
            self.assertEqual(recovered["status"], "complete")

    def test_recovers_initialized_output_before_first_state_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            catalog = extract.build_catalog(root, profile)
            part = root / "source-mtp" / "mtp.safetensors.part"
            extract._initialize_output(part, catalog)
            self.assertFalse((root / "source-mtp-state.json").exists())
            resumed = extract.extract_available_shards(root, profile=profile)
            self.assertEqual(resumed["status"], "partial")
            self.assertEqual(len(resumed["completed_shards"]), 1)

    def test_complete_rerun_verifies_then_deletes_stale_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            extract.extract_available_shards(root, profile=profile, max_shards=2)
            raw_dir = root / "source-mtp-raw"
            self.assertTrue(all((raw_dir / shard.name).exists() for shard in profile.shards))
            completed = extract.extract_available_shards(
                root,
                profile=profile,
                delete_source=True,
            )
            self.assertEqual(completed["status"], "complete")
            self.assertTrue(all(not (raw_dir / shard.name).exists() for shard in profile.shards))

    def test_rejects_completed_shard_metadata_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            extract.extract_available_shards(root, profile=profile)
            state_path = root / "source-mtp-state.json"
            state = extract.load_json(state_path)
            completed = state["completed_shards"][profile.shards[0].name]
            completed["source_sha256"] = "0" * 64
            write_json(state_path, state)
            with self.assertRaisesRegex(extract.MTPExtractionError, "source hash drift"):
                extract.extract_available_shards(root, profile=profile)

    def test_rejects_pinned_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = make_fixture(root)
            state_path = root / "metadata-mtp-source" / "source-state.json"
            state = extract.load_json(state_path)
            state["revision"] = "2" * 40
            write_json(state_path, state)
            with self.assertRaisesRegex(extract.MTPExtractionError, "revision"):
                extract.build_catalog(root, profile)


if __name__ == "__main__":
    unittest.main(verbosity=2)

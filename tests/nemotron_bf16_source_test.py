#!/usr/bin/env python3
"""Focused tests for strict Nemotron BF16 layer contracts."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))

from nemotron_bf16_source import (  # noqa: E402
    EXPECTED_REPOSITORY,
    EXPECTED_REVISION,
    build_layer_contract,
    validate_local_shards,
)
from nemotron_metadata import MetadataError  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class BF16SourceTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        metadata = root / "metadata"
        metadata.mkdir()
        pattern = "ME" * 40 + "*" * 8
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "model_type": "nemotron_h",
            "dtype": "bfloat16",
            "num_hidden_layers": 88,
            "hybrid_override_pattern": pattern,
            "n_routed_experts": 512,
            "num_experts_per_tok": 22,
            "moe_latent_size": 1024,
            "moe_intermediate_size": 2688,
            "mlp_hidden_act": "relu2",
        }
        weight_map = {
            "backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00002.safetensors",
        }
        for expert in range(512):
            shard = "model-00001-of-00002.safetensors" if expert < 300 else "model-00002-of-00002.safetensors"
            weight_map[f"backbone.layers.1.mixer.experts.{expert}.up_proj.weight"] = shard
            weight_map[f"backbone.layers.1.mixer.experts.{expert}.down_proj.weight"] = shard
        weight_map["backbone.layers.1.mixer.experts.299.down_proj.weight"] = (
            "model-00002-of-00002.safetensors"
        )
        index = {"metadata": {"total_size": 123456}, "weight_map": weight_map}
        config_path = metadata / "config.json"
        index_path = metadata / "model.safetensors.index.json"
        write_json(config_path, config)
        write_json(index_path, index)
        state = {
            "format": "nemotron-bf16-metadata-state-v1",
            "repository": EXPECTED_REPOSITORY,
            "revision": EXPECTED_REVISION,
            "indexed_payload_bytes": 123456,
            "indexed_shards": 2,
            "indexed_tensors": len(weight_map),
            "files": {
                "config.json": {"bytes": config_path.stat().st_size, "sha256": sha256(config_path)},
                "model.safetensors.index.json": {
                    "bytes": index_path.stat().st_size,
                    "sha256": sha256(index_path),
                },
            },
            "shards": {
                "model-00001-of-00002.safetensors": {"bytes": 17, "sha256": "1" * 64},
                "model-00002-of-00002.safetensors": {"bytes": 19, "sha256": "2" * 64},
            },
        }
        state_path = metadata / "state.json"
        write_json(state_path, state)
        return metadata, state_path

    def test_builds_exact_layer_contract_and_storage_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, state = self.fixture(Path(temporary))
            contract = build_layer_contract(metadata, state, 1)
            self.assertEqual(contract["required_download"]["bytes"], 36)
            self.assertEqual(contract["split_experts"], [299])
            self.assertEqual(contract["source_expert_payload"]["tensors"], 1024)
            self.assertLess(
                contract["uniform_affine_projection"]["1"]["bytes"],
                contract["uniform_affine_projection"]["4"]["bytes"],
            )
            self.assertEqual(
                contract["mixed_binary_native_projection"][3]["native_nvfp4_experts"],
                256,
            )
            self.assertEqual(
                contract["storage_units"]["native_nvfp4_bytes_per_expert"],
                3_096_584,
            )

    def test_rejects_missing_expert_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, state = self.fixture(Path(temporary))
            index_path = metadata / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            del index["weight_map"]["backbone.layers.1.mixer.experts.511.down_proj.weight"]
            write_json(index_path, index)
            state_value = json.loads(state.read_text())
            state_value["indexed_tensors"] -= 1
            state_value["files"]["model.safetensors.index.json"] = {
                "bytes": index_path.stat().st_size,
                "sha256": sha256(index_path),
            }
            write_json(state, state_value)
            with self.assertRaisesRegex(MetadataError, "missing BF16 expert tensor"):
                build_layer_contract(metadata, state, 1)

    def test_rejects_metadata_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, state = self.fixture(Path(temporary))
            with (metadata / "config.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(MetadataError, "metadata size mismatch"):
                build_layer_contract(metadata, state, 1)

    def test_local_shard_validation_checks_size_before_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, state = self.fixture(root)
            contract = build_layer_contract(metadata, state, 1)
            raw = root / "raw"
            raw.mkdir()
            (raw / "model-00001-of-00002.safetensors").write_bytes(b"x")
            (raw / "model-00002-of-00002.safetensors").write_bytes(b"y")
            with self.assertRaisesRegex(MetadataError, "size mismatch"):
                validate_local_shards(contract, raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)

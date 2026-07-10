#!/usr/bin/env python3
"""Tests for exact Nemotron safetensors inventory accounting."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
import nemotron_metadata as metadata  # noqa: E402
import nemotron_safetensors_inventory as inventory  # noqa: E402


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> tuple[dict, int]:
    header = {}
    cursor = 0
    for name, (dtype, shape) in tensors.items():
        size = inventory.tensor_nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(cursor))
    return header, cursor


def fixture(root: Path) -> tuple[dict, dict]:
    config = {
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "hybrid_override_pattern": "E",
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_latent_size": 2,
        "moe_intermediate_size": 4,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"},
    }
    tensors: dict[str, tuple[str, list[int]]] = {
        "backbone.layers.0.mixer.gate.weight": ("BF16", [2, 4]),
        "backbone.layers.0.mixer.gate.e_score_correction_bias": ("F32", [2]),
        "backbone.layers.0.norm.weight": ("BF16", [4]),
        "mtp.layers.0.norm.weight": ("BF16", [4]),
        "lm_head.weight": ("BF16", [8, 4]),
    }
    for expert in range(2):
        prefix = f"backbone.layers.0.mixer.experts.{expert}"
        for member in metadata.EXPECTED_EXPERT_MEMBERS:
            if member.endswith("weight_scale_2") or member.endswith("input_scale"):
                tensors[f"{prefix}.{member}"] = ("F32", [])
            elif member.endswith("weight_scale"):
                tensors[f"{prefix}.{member}"] = ("F8_E4M3", [2, 1])
            else:
                tensors[f"{prefix}.{member}"] = ("U8", [2, 2])

    shard = "model-00001-of-00001.safetensors"
    _, payload = write_shard(root / shard, tensors)
    index = {"metadata": {"total_size": payload}, "weight_map": {name: shard for name in tensors}}
    return config, index


class InventoryTest(unittest.TestCase):
    def test_inventory_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, index = fixture(root)
            result = inventory.build_inventory(root, config, index, strict_target=False)
            self.assertEqual(result["totals"]["tensors"], len(index["weight_map"]))
            self.assertEqual(result["roles"]["backbone_routed_expert"]["tensors"], 16)
            self.assertEqual(result["roles"]["mtp"]["bytes"], 8)
            fifty = next(item for item in result["uniform_pruning_projections"] if item["prune_percent"] == 40)
            self.assertEqual(fifty["retained_experts_per_layer"], 1)
            self.assertLess(fifty["without_mtp_bytes"], result["totals"]["payload_bytes"])

    def test_rejects_trailing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, index = fixture(root)
            shard = root / "model-00001-of-00001.safetensors"
            with shard.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(metadata.MetadataError, "file size mismatch"):
                inventory.build_inventory(root, config, index, strict_target=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)

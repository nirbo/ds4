#!/usr/bin/env python3
"""Tests for exact Nemotron NVFP4 structural materialization."""

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
import nemotron_prune_materialize as materialize  # noqa: E402
from nemotron_safetensors_inventory import read_safetensors_header  # noqa: E402


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_tensor(path: Path, name: str) -> bytes:
    tensors, header_size, _ = read_safetensors_header(path)
    start, end = tensors[name]["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(8 + header_size + start)
        return handle.read(end - start)


class MaterializeTest(unittest.TestCase):
    def test_remaps_experts_router_and_omits_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model-00001-of-00001.safetensors"
            output = root / "out.safetensors"
            tensors = {
                "backbone.layers.0.mixer.gate.weight": ("BF16", [4, 2], bytes(range(16))),
                "backbone.layers.0.mixer.gate.e_score_correction_bias": ("F32", [4], bytes(range(16, 32))),
                "backbone.layers.0.mixer.experts.0.up_proj.weight": ("U8", [2], b"a0"),
                "backbone.layers.0.mixer.experts.1.up_proj.weight": ("U8", [2], b"b1"),
                "backbone.layers.0.mixer.experts.2.up_proj.weight": ("U8", [2], b"c2"),
                "backbone.layers.0.mixer.experts.3.up_proj.weight": ("U8", [2], b"d3"),
                "backbone.layers.0.norm.weight": ("BF16", [2], b"norm"),
                "mtp.layers.0.norm.weight": ("BF16", [2], b"mtpx"),
            }
            write_shard(source, tensors)
            mappings = {0: {1: 0, 3: 1}}
            plan = materialize.build_shard_plan(source, mappings, 4, True)
            result = materialize.materialize_shard(source, output, plan)
            headers, _, _ = read_safetensors_header(output)
            self.assertEqual(result["status"], "done")
            self.assertNotIn("mtp.layers.0.norm.weight", headers)
            self.assertNotIn("backbone.layers.0.mixer.experts.2.up_proj.weight", headers)
            self.assertEqual(read_tensor(output, "backbone.layers.0.mixer.experts.0.up_proj.weight"), b"b1")
            self.assertEqual(read_tensor(output, "backbone.layers.0.mixer.experts.1.up_proj.weight"), b"d3")
            self.assertEqual(read_tensor(output, "backbone.layers.0.norm.weight"), b"norm")
            self.assertEqual(read_tensor(output, "backbone.layers.0.mixer.gate.weight"), bytes(range(4, 8)) + bytes(range(12, 16)))
            self.assertEqual(read_tensor(output, "backbone.layers.0.mixer.gate.e_score_correction_bias"), bytes(range(20, 24)) + bytes(range(28, 32)))

    def test_validates_plan_identity(self) -> None:
        config = {
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "hybrid_override_pattern": "E",
        }
        plan = {
            "format": materialize.PLAN_FORMAT,
            "source_revision": "abc",
            "old_num_experts": 4,
            "new_num_experts": 2,
            "model_moe_layers": [0],
            "kept_by_layer": {"0": [1, 3]},
            "old_to_new_by_layer": {"0": {"1": 0, "3": 1}},
        }
        mapping = materialize.validate_plan(plan, config, "abc")
        self.assertEqual(mapping, {0: {1: 0, 3: 1}})
        plan["source_revision"] = "wrong"
        with self.assertRaisesRegex(materialize.MetadataError, "revision mismatch"):
            materialize.validate_plan(plan, config, "abc")

    def test_all_excluded_tensors_omit_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model-00001-of-00001.safetensors"
            write_shard(
                source,
                {"mtp.layers.0.norm.weight": ("BF16", [2], b"mtpx")},
            )
            plan = materialize.build_shard_plan(source, {}, 4, True)
            self.assertTrue(plan["omitted"])
            self.assertEqual(plan["tensor_count"], 0)
            self.assertEqual(plan["file_bytes"], 0)

    def test_remaps_both_modelopt_maps(self) -> None:
        config = {
            "n_routed_experts": 4,
            "num_nextn_predict_layers": 1,
            "quantization_config": {
                "config_groups": {
                    "group_1": {
                        "targets": [
                            "backbone.layers.0.mixer.experts.1.up_proj",
                            "backbone.layers.0.mixer.experts.2.up_proj",
                        ]
                    }
                },
                "quantized_layers": {
                    "backbone.layers.0.mixer.experts.1.up_proj": {"quant_algo": "NVFP4"},
                    "backbone.layers.0.mixer.experts.2.up_proj": {"quant_algo": "NVFP4"},
                },
            },
        }
        transformed = materialize.transform_config(config, {0: {1: 0}}, 1, True)
        self.assertEqual(transformed["n_routed_experts"], 1)
        self.assertEqual(transformed["num_nextn_predict_layers"], 0)
        self.assertEqual(
            transformed["quantization_config"]["config_groups"]["group_1"]["targets"],
            ["backbone.layers.0.mixer.experts.0.up_proj"],
        )
        self.assertEqual(
            list(transformed["quantization_config"]["quantized_layers"]),
            ["backbone.layers.0.mixer.experts.0.up_proj"],
        )

    def test_finalizes_self_consistent_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            shard = "model-00001-of-00001.safetensors"
            tensors = {
                "backbone.layers.0.mixer.gate.weight": ("BF16", [2, 2], b"router00"),
                "backbone.layers.0.mixer.gate.e_score_correction_bias": ("F32", [2], b"bias0000"),
                "backbone.layers.0.norm.weight": ("BF16", [2], b"norm"),
                "lm_head.weight": ("BF16", [2, 2], b"lmhead00"),
                "mtp.layers.0.norm.weight": ("BF16", [2], b"mtpx"),
            }
            for expert in range(2):
                for projection in ("up_proj", "down_proj"):
                    prefix = f"backbone.layers.0.mixer.experts.{expert}.{projection}"
                    tensors[f"{prefix}.input_scale"] = ("F32", [], bytes([expert + 1]) * 4)
                    tensors[f"{prefix}.weight"] = ("U8", [2, 2], bytes([expert + 3]) * 4)
                    tensors[f"{prefix}.weight_scale"] = ("F8_E4M3", [2, 1], bytes([expert + 5]) * 2)
                    tensors[f"{prefix}.weight_scale_2"] = ("F32", [], bytes([expert + 7]) * 4)
            write_shard(source_dir / shard, tensors)

            quantized_layers = {
                f"backbone.layers.0.mixer.experts.{expert}.{projection}": {"quant_algo": "NVFP4"}
                for expert in range(2)
                for projection in ("up_proj", "down_proj")
            }
            config = {
                "architectures": ["NemotronHForCausalLM"],
                "hidden_size": 2,
                "num_hidden_layers": 1,
                "hybrid_override_pattern": "E",
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "n_group": 1,
                "moe_latent_size": 2,
                "moe_intermediate_size": 2,
                "num_nextn_predict_layers": 1,
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "config_groups": {"group_1": {"targets": list(quantized_layers)}},
                    "quantized_layers": quantized_layers,
                },
            }
            hf_quant = {"quantization": {"quantized_layers": quantized_layers}}
            plan = {
                "format": materialize.PLAN_FORMAT,
                "source_revision": "abc",
                "old_num_experts": 2,
                "new_num_experts": 1,
                "model_moe_layers": [0],
                "kept_by_layer": {"0": [1]},
                "old_to_new_by_layer": {"0": {"1": 0}},
            }
            mappings = {0: {1: 0}}
            shard_plan = materialize.build_shard_plan(source_dir / shard, mappings, 2, True)
            materialize.materialize_shard(source_dir / shard, output_dir / shard, shard_plan)
            report = materialize.finalize_artifact(
                source_dir,
                output_dir,
                [shard],
                config,
                hf_quant,
                plan,
                mappings,
                True,
                {"source_revision": "abc", "plan_sha256": "planhash"},
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["validation"]["modelopt_maps"], "passed")
            output_config = json.loads((output_dir / "config.json").read_text())
            self.assertEqual(output_config["n_routed_experts"], 1)
            self.assertEqual(output_config["num_nextn_predict_layers"], 0)
            output_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
            self.assertFalse(any(name.startswith("mtp.") for name in output_index["weight_map"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

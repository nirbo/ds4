#!/usr/bin/env python3
"""Tests for direct prune-to-MLX Nemotron runtime packing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_pack import build_groups, runtime_config, validate_pack_plan, write_group  # noqa: E402
from nemotron_safetensors_inventory import read_safetensors_header  # noqa: E402


def source(path: Path, offset: int, size: int, dtype: str, shape: list[int]) -> dict:
    return {"path": path, "offset": offset, "size": size, "dtype": dtype, "shape": shape}


class MLXPackTest(unittest.TestCase):
    def test_prunes_and_stacks_experts_with_matching_router(self) -> None:
        path = Path("source.safetensors")
        catalog = {
            "backbone.embeddings.weight": source(path, 0, 32, "BF16", [8, 2]),
            "backbone.layers.0.mixer.gate.weight": source(path, 32, 16, "BF16", [2, 4]),
            "backbone.layers.0.mixer.gate.e_score_correction_bias": source(path, 48, 8, "F32", [2]),
        }
        offset = 56
        for expert in range(2):
            for projection in ("up_proj", "down_proj"):
                prefix = f"backbone.layers.0.mixer.experts.{expert}.{projection}"
                for suffix, size, dtype, shape in (
                    ("weight", 128, "U8", [16, 8]),
                    ("weight_scale", 16, "U8", [16, 1]),
                    ("weight_scale_2", 4, "F32", [1]),
                    ("input_scale", 4, "F32", [1]),
                ):
                    catalog[f"{prefix}.{suffix}"] = source(path, offset, size, dtype, shape)
                    offset += size
        config = {
            "num_hidden_layers": 1,
            "hybrid_override_pattern": "E",
            "n_routed_experts": 2,
        }
        override_path = Path("router.safetensors")
        override = source(override_path, 8, 8, "BF16", [1, 4])
        groups = build_groups(
            catalog,
            config,
            {0: {1: 0}},
            omit_mtp=True,
            router_overrides={0: override},
        )
        layer = groups["layer-000"]
        self.assertEqual(layer["backbone.layers.0.mixer.switch_mlp.fc1.weight"]["shape"], [1, 16, 8])
        self.assertEqual(layer["backbone.layers.0.mixer.switch_mlp.fc2.scales"]["shape"], [1, 16, 1])
        self.assertEqual(layer["backbone.layers.0.mixer.switch_mlp.fc1.global_scales"]["shape"], [1])
        self.assertEqual(layer["backbone.layers.0.mixer.gate.weight"]["shape"], [1, 4])
        self.assertEqual(
            layer["backbone.layers.0.mixer.gate.weight"]["segments"][0]["path"],
            override_path,
        )
        self.assertEqual(layer["backbone.layers.0.mixer.gate.e_score_correction_bias"]["shape"], [1])
        segment = layer["backbone.layers.0.mixer.switch_mlp.fc1.weight"]["segments"][0]
        self.assertEqual(segment["offset"], catalog["backbone.layers.0.mixer.experts.1.up_proj.weight"]["offset"])

    def test_nonuniform_plan_and_runtime_config(self) -> None:
        config = {
            "hybrid_override_pattern": "EME",
            "n_routed_experts": 4,
            "num_experts_per_tok": 1,
        }
        kept = {"0": [0, 2], "2": [1, 2, 3]}
        plan = {
            "format": "nemotron-nonuniform-prune-plan-v1",
            "source_revision": "revision",
            "old_num_experts": 4,
            "model_moe_layers": [0, 2],
            "new_num_experts_by_layer": {"0": 2, "2": 3},
            "kept_by_layer": kept,
            "old_to_new_by_layer": {
                "0": {"0": 0, "2": 1},
                "2": {"1": 0, "2": 1, "3": 2},
            },
        }
        mappings = validate_pack_plan(plan, config, "revision")
        self.assertEqual(mappings[0], {0: 0, 2: 1})
        transformed = runtime_config(config, {0: 2, 2: 3}, True)
        self.assertEqual(transformed["n_routed_experts"], 3)
        self.assertEqual(transformed["nemotron_runtime"]["experts_by_layer"], {"0": 2, "2": 3})
        self.assertTrue(transformed["nemotron_runtime"]["nonuniform_experts"])

    def test_group_writer_preserves_segment_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.bin"
            source_path.write_bytes(bytes(range(64)))
            tensors = {
                "a": {
                    "dtype": "U8",
                    "shape": [8],
                    "size": 8,
                    "segments": [source(source_path, 5, 8, "U8", [8])],
                },
                "b": {
                    "dtype": "U8",
                    "shape": [4],
                    "size": 4,
                    "segments": [source(source_path, 30, 4, "U8", [4])],
                },
            }
            output = root / "group.safetensors"
            result = write_group(output, tensors)
            entries, header_bytes, payload_bytes = read_safetensors_header(output)
            self.assertEqual(result["status"], "done")
            self.assertEqual(payload_bytes, 12)
            self.assertEqual(entries["a"]["shape"], [8])
            with output.open("rb") as handle:
                handle.seek(8 + header_bytes)
                self.assertEqual(handle.read(), bytes(range(5, 13)) + bytes(range(30, 34)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

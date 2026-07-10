#!/usr/bin/env python3
"""Focused tests for the Nemotron metadata catalog."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nemotron" / "tools" / "nemotron_metadata.py"
SPEC = importlib.util.spec_from_file_location("nemotron_metadata", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture() -> tuple[dict, dict]:
    config = {
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "hybrid_override_pattern": "ME*",
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_latent_size": 4,
        "moe_intermediate_size": 8,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
        },
    }
    weight_map: dict[str, str] = {
        "backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00002.safetensors",
        "backbone.layers.1.mixer.gate.weight": "model-00001-of-00002.safetensors",
        "backbone.layers.1.mixer.gate.e_score_correction_bias": "model-00001-of-00002.safetensors",
        "backbone.layers.2.mixer.q_proj.weight": "model-00002-of-00002.safetensors",
        "mtp.layers.0.norm.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    for expert in range(2):
        for member in MODULE.EXPECTED_EXPERT_MEMBERS:
            weight_map[
                f"backbone.layers.1.mixer.experts.{expert}.{member}"
            ] = "model-00001-of-00002.safetensors"
    index = {"metadata": {"total_size": 1234}, "weight_map": weight_map}
    return config, index


class MetadataTest(unittest.TestCase):
    def test_catalogs_complete_expert_groups(self) -> None:
        config, index = fixture()
        catalog = MODULE.validate_metadata(config, index, strict_target=False)
        self.assertEqual(catalog["model"]["layer_type_counts"], {"M": 1, "E": 1, "*": 1})
        self.assertEqual(catalog["weights"]["shard_count"], 2)
        self.assertEqual(catalog["weights"]["tensor_role_counts"]["backbone_routed_expert"], 16)
        self.assertEqual(catalog["weights"]["tensor_role_counts"]["mtp"], 1)
        self.assertEqual(catalog["moe_layers"][0]["expert_tensor_count"], 16)

    def test_rejects_incomplete_expert_group(self) -> None:
        config, index = fixture()
        broken = copy.deepcopy(index)
        del broken["weight_map"]["backbone.layers.1.mixer.experts.0.up_proj.weight_scale_2"]
        with self.assertRaisesRegex(MODULE.MetadataError, "member mismatch"):
            MODULE.validate_metadata(config, broken, strict_target=False)

    def test_rejects_expert_on_wrong_layer_type(self) -> None:
        config, index = fixture()
        broken = copy.deepcopy(config)
        broken["hybrid_override_pattern"] = "M**"
        with self.assertRaisesRegex(MODULE.MetadataError, "non-MoE layer"):
            MODULE.validate_metadata(broken, index, strict_target=False)

    def test_rejects_noncontiguous_shards(self) -> None:
        config, index = fixture()
        broken = copy.deepcopy(index)
        broken["weight_map"]["lm_head.weight"] = "model-00003-of-00002.safetensors"
        with self.assertRaisesRegex(MODULE.MetadataError, "wrong shard total"):
            MODULE.validate_metadata(config, broken, strict_target=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)

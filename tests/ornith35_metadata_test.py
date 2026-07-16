#!/usr/bin/env python3
"""Focused tests for the Ornith 35B metadata catalog."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ornith35" / "tools" / "ornith35_metadata.py"
SPEC = importlib.util.spec_from_file_location("ornith35_metadata", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config_fixture() -> dict:
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    return {
        "architectures": ["TestArchitecture"],
        "model_type": "qwen3_5_moe",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "input_activations": None,
                    "weights": {
                        "num_bits": 4,
                        "group_size": 16,
                        "type": "float",
                        "scale_dtype": "torch.float8_e4m3fn",
                    },
                }
            },
        },
        "text_config": {
            "hidden_size": 16,
            "layer_types": layer_types,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 16,
            "partial_rotary_factor": 0.25,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "mrope_interleaved": True,
                "mrope_section": [1, 1, 0],
            },
        },
    }


def header_fixture() -> tuple[dict, int]:
    header: dict = {}
    offset = 0

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        size = MODULE.tensor_nbytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size

    add("model.language_model.embed_tokens.weight", "BF16", [8, 16])
    for layer in range(4):
        add(f"model.language_model.layers.{layer}.input_layernorm.weight", "BF16", [16])
        if layer == 3:
            add(f"model.language_model.layers.{layer}.self_attn.q_proj.weight", "BF16", [8, 16])
        else:
            add(f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight", "BF16", [16, 16])
        add(f"model.language_model.layers.{layer}.mlp.gate.weight", "BF16", [2, 16])
        for expert in range(2):
            add(
                f"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "U8",
                [8, 16],
            )
    add("model.visual.patch_embed.proj.weight", "BF16", [4, 4])
    add("lm_head.weight", "BF16", [8, 16])
    return header, offset


def state_fixture(payload_bytes: int) -> dict:
    return {
        "format": MODULE.SOURCE_STATE_FORMAT,
        "repository": MODULE.EXPECTED_REPOSITORY,
        "revision": MODULE.EXPECTED_REVISION,
        "weight": {
            "name": "model.safetensors",
            "file_bytes": payload_bytes + 24,
            "header_bytes": 16,
            "payload_bytes": payload_bytes,
            "sha256": MODULE.EXPECTED_WEIGHT_SHA256,
        },
    }


class MetadataTest(unittest.TestCase):
    def test_builds_role_and_context_catalog(self) -> None:
        config = config_fixture()
        header, payload = header_fixture()
        catalog = MODULE.build_catalog(
            config,
            header,
            state_fixture(payload),
            strict_target=False,
        )
        self.assertEqual(catalog["model"]["layer_type_counts"], {"full_attention": 1, "linear_attention": 3})
        self.assertEqual(catalog["roles"]["routed_expert"]["tensors"], 8)
        self.assertEqual(catalog["roles"]["vision"]["bytes"], 32)
        self.assertEqual(catalog["weights"]["text_payload_bytes"], payload - 32)
        self.assertEqual(catalog["cache"]["bf16_profiles"]["native-262k"]["kv_bytes_per_token"], 16)

    def test_rejects_missing_expert(self) -> None:
        config = config_fixture()
        header, payload = header_fixture()
        broken = copy.deepcopy(header)
        name = "model.language_model.layers.2.mlp.experts.1.gate_proj.weight"
        broken[name.replace("experts.1", "experts.2")] = broken.pop(name)
        with self.assertRaisesRegex(MODULE.MetadataError, "expert coverage mismatch"):
            MODULE.build_catalog(config, broken, state_fixture(payload), strict_target=False)

    def test_rejects_noncontiguous_payload(self) -> None:
        config = config_fixture()
        header, payload = header_fixture()
        broken = copy.deepcopy(header)
        name = "model.language_model.embed_tokens.weight"
        broken[name]["data_offsets"] = [2, broken[name]["data_offsets"][1] + 2]
        with self.assertRaisesRegex(MODULE.MetadataError, "non-contiguous payload"):
            MODULE.build_catalog(config, broken, state_fixture(payload + 2), strict_target=False)

    def test_context_math_matches_production_shape(self) -> None:
        config = config_fixture()
        config["text_config"]["layer_types"] = ["full_attention"] * 10
        config["text_config"]["num_key_value_heads"] = 2
        config["text_config"]["head_dim"] = 256
        profile = MODULE.context_profile(config, 524_288, kv_bytes=2)
        self.assertEqual(profile["kv_bytes_per_token"], 20_480)
        self.assertEqual(profile["kv_cache_bytes"], 10 * 2**30)

    def test_validates_complete_nvfp4_companions(self) -> None:
        text = {
            "hidden_size": 16,
            "moe_intermediate_size": 16,
            "num_experts": 1,
            "layer_types": ["linear_attention"],
        }
        entries: dict = {}
        for expert in ("experts.0", "shared_expert"):
            prefix = f"model.language_model.layers.0.mlp.{expert}"
            for projection, output, input_width in (
                ("down_proj", 16, 16),
                ("gate_proj", 16, 16),
                ("up_proj", 16, 16),
            ):
                entries[f"{prefix}.{projection}.weight_global_scale"] = {
                    "dtype": "F32",
                    "shape": [1],
                }
                entries[f"{prefix}.{projection}.weight_scale"] = {
                    "dtype": "F8_E4M3",
                    "shape": [output, input_width // 16],
                }
                entries[f"{prefix}.{projection}.weight_packed"] = {
                    "dtype": "U8",
                    "shape": [output, input_width // 2],
                }
        self.assertEqual(MODULE.validate_nvfp4_mlp(entries, text), 18)

        broken = copy.deepcopy(entries)
        name = "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale"
        broken[name]["dtype"] = "BF16"
        with self.assertRaisesRegex(MODULE.MetadataError, "dtype mismatch"):
            MODULE.validate_nvfp4_mlp(broken, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

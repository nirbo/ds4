#!/usr/bin/env python3
"""Focused tests for the pinned Ornith-35 DSpark draft contract."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ornith35" / "tools" / "ornith35_dspark.py"
SPEC = importlib.util.spec_from_file_location("ornith35_dspark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config_fixture() -> dict:
    return {
        "architectures": ["DSparkDraftModel"],
        "speculators_model_type": "dspark",
        "block_size": 8,
        "max_anchors": 3072,
        "aux_hidden_state_layer_ids": [9, 19, 29],
        "mask_token_id": 248077,
        "draft_vocab_size": 32000,
        "target_hidden_size": None,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "speculators_config": {
            "algorithm": "dspark",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 7,
                    "verifier_accept_k": 1,
                    "accept_tolerance": 0.0,
                }
            ],
            "verifier": {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "name_or_path": "/irrelevant/pinned-training-path",
            },
        },
        "transformer_layer_config": {
            "model_type": "qwen3",
            "vocab_size": 248320,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 3,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": False,
            "layer_types": ["full_attention"] * 3,
            "sliding_window": None,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        },
    }


def header_fixture() -> dict:
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in MODULE.expected_tensor_specs().items():
        size = MODULE.tensor_nbytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    assert offset == MODULE.EXPECTED_PAYLOAD_BYTES
    return header


def source_state_fixture() -> dict:
    return {
        "format": MODULE.SOURCE_STATE_FORMAT,
        "profile": "dspark",
        "repository": MODULE.EXPECTED_REPOSITORY,
        "revision": MODULE.EXPECTED_REVISION,
        "metadata_files": copy.deepcopy(MODULE.EXPECTED_METADATA_FILES),
        "weight": {
            "name": MODULE.EXPECTED_WEIGHT_NAME,
            "file_bytes": MODULE.EXPECTED_WEIGHT_BYTES,
            "header_bytes": MODULE.EXPECTED_HEADER_BYTES,
            "payload_bytes": MODULE.EXPECTED_PAYLOAD_BYTES,
            "sha256": MODULE.EXPECTED_WEIGHT_SHA256,
            "header_file": "model.safetensors.header.json",
            "header_sha256": MODULE.EXPECTED_HEADER_SHA256,
        },
    }


class DSparkTest(unittest.TestCase):
    def test_validates_legacy_anchor_and_auxiliary_state_contract(self) -> None:
        contract = MODULE.validate_config(config_fixture())
        self.assertEqual(contract["anchor_slot"], 0)
        self.assertEqual(contract["speculative_slots"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(contract["aux_hidden_state_indices"], [9, 19, 29])
        self.assertEqual(contract["aux_decoder_layer_indices"], [8, 18, 28])
        self.assertEqual(contract["aux_concatenated_width"], 6144)

    def test_rejects_post_release_anchor_semantics(self) -> None:
        broken = config_fixture()
        broken["sample_from_anchor"] = False
        with self.assertRaisesRegex(MODULE.DSparkError, "post-release"):
            MODULE.validate_config(broken)

    def test_rejects_auxiliary_layer_drift(self) -> None:
        broken = config_fixture()
        broken["aux_hidden_state_layer_ids"] = [8, 18, 28]
        with self.assertRaisesRegex(MODULE.DSparkError, "auxiliary"):
            MODULE.validate_config(broken)

    def test_validates_complete_weight_schema(self) -> None:
        summary = MODULE.validate_header(header_fixture())
        self.assertEqual(summary["tensor_count"], 44)
        self.assertEqual(summary["payload_bytes"], 1_657_163_778)
        self.assertEqual(summary["dtype_counts"], {"BF16": 42, "BOOL": 1, "I64": 1})
        self.assertEqual(summary["bf16_parameter_count"], 828_329_729)

    def test_rejects_tensor_dtype_and_payload_drift(self) -> None:
        broken_dtype = header_fixture()
        broken_dtype["fc.weight"]["dtype"] = "BOOL"
        with self.assertRaisesRegex(MODULE.DSparkError, "dtype mismatch"):
            MODULE.validate_header(broken_dtype)

        broken_offsets = copy.deepcopy(header_fixture())
        broken_offsets["hidden_norm.weight"]["data_offsets"][0] += 2
        with self.assertRaisesRegex(MODULE.DSparkError, "payload mismatch|non-contiguous"):
            MODULE.validate_header(broken_offsets)

    def test_source_state_binds_exact_metadata_hashes(self) -> None:
        summary = MODULE.validate_source_state(source_state_fixture())
        self.assertEqual(summary["revision"], MODULE.EXPECTED_REVISION)

        broken = source_state_fixture()
        broken["metadata_files"]["config.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.DSparkError, "metadata-file identities"):
            MODULE.validate_source_state(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)

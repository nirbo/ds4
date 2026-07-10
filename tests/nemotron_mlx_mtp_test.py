#!/usr/bin/env python3
"""Focused tests for Nemotron MTP tracing and payload accounting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_mtp import (  # noqa: E402
    NemotronMTPSidecar,
    QuantizedMTPHead,
    ReducedVocabMTPHead,
    mtp_payload_estimate,
    mtp_tensor_names,
)
from nemotron_mlx_mtp_bench import append_trace_rows  # noqa: E402
from nemotron_mlx_mtp_head_quantize import MODES, quantize_weight  # noqa: E402
from nemotron_mlx_mtp_pack import build_mtp_group  # noqa: E402
from nemotron_mlx_mtp_quantize import quantizable  # noqa: E402
from nemotron_mlx_mtp_vocab_head import rank_tokens, select_token_ids  # noqa: E402
from nemotron_mlx_linear import ModelOptBF16Linear  # noqa: E402
from nemotron_prune_materialize import sha256_file  # noqa: E402


class MLXMTPTest(unittest.TestCase):
    def test_tensor_catalog_contains_every_bf16_expert_pair(self) -> None:
        names = mtp_tensor_names({"n_routed_experts": 4})
        self.assertIn("mtp.layers.1.mixer.experts.0.up_proj.weight", names)
        self.assertIn("mtp.layers.1.mixer.experts.3.down_proj.weight", names)
        self.assertNotIn("mtp.layers.1.mixer.experts.4.up_proj.weight", names)

    def test_trace_rows_shift_tokens_and_score_only_generation(self) -> None:
        output = {
            "hidden": [],
            "accepted": [],
            "expected": [],
            "prompt_index": [],
            "scored": [],
        }
        hidden = [mx.array([float(index)]) for index in range(6)]
        append_trace_rows(hidden, [10, 11, 12, 20, 21, 22], 3, 7, output)
        self.assertEqual(output["accepted"], [11, 12, 20, 21])
        self.assertEqual(output["expected"], [12, 20, 21, 22])
        self.assertEqual(output["prompt_index"], [7, 7, 7, 7])
        self.assertEqual(output["scored"], [0, 0, 1, 1])
        self.assertEqual([float(row.item()) for row in output["hidden"]], [0.0, 1.0, 2.0, 3.0])

    def test_payload_estimate_preserves_fixed_head_cost(self) -> None:
        config = {
            "n_routed_experts": 512,
            "hidden_size": 4096,
            "moe_latent_size": 1024,
            "moe_intermediate_size": 2688,
        }
        full = mtp_payload_estimate(config, 512)
        half = mtp_payload_estimate(config, 256)
        self.assertEqual(full, 5_884_651_520)
        self.assertEqual(
            full - half,
            256 * (2 * 1024 * 2688 * 2 + 4096 * 2 + 4),
        )
        self.assertGreater(half, full // 2)

    def test_sidecar_stacks_selected_experts_and_slices_router_in_plan_order(self) -> None:
        def source(shape, size, offset=0, dtype="BF16"):
            return {
                "path": Path("source.safetensors"),
                "offset": offset,
                "size": size,
                "dtype": dtype,
                "shape": shape,
            }

        mixer = "mtp.layers.1.mixer"
        catalog = {
            "mtp.layers.0.enorm.weight": source([2], 4),
            f"{mixer}.gate.weight": source([4, 2], 16, offset=100),
            f"{mixer}.gate.e_score_correction_bias": source(
                [4], 16, offset=200, dtype="F32"
            ),
        }
        for expert in range(4):
            catalog[f"{mixer}.experts.{expert}.up_proj.weight"] = source(
                [3, 2], 12, offset=1000 + expert * 100
            )
            catalog[f"{mixer}.experts.{expert}.down_proj.weight"] = source(
                [2, 3], 12, offset=2000 + expert * 100
            )
        group = build_mtp_group(
            catalog,
            {"n_routed_experts": 4, "num_experts_per_tok": 2},
            [3, 1],
        )
        self.assertEqual(group[f"{mixer}.gate.weight"]["shape"], [2, 2])
        self.assertEqual(
            [segment["offset"] for segment in group[f"{mixer}.gate.weight"]["segments"]],
            [112, 104],
        )
        stacked = group[f"{mixer}.switch_mlp.up_proj.weight"]
        self.assertEqual(stacked["shape"], [2, 3, 2])
        self.assertEqual([segment["offset"] for segment in stacked["segments"]], [1300, 1100])

    def test_mtp_quantization_preserves_router_and_norms(self) -> None:
        matrix = mx.zeros((4, 64), dtype=mx.bfloat16)
        vector = mx.zeros((64,), dtype=mx.bfloat16)
        self.assertTrue(quantizable("mtp.layers.0.eh_proj.weight", matrix))
        self.assertFalse(quantizable("mtp.layers.1.mixer.gate.weight", matrix))
        self.assertFalse(quantizable("mtp.layers.0.norm.weight", vector))

    def test_quantized_mtp_head_loads_and_rejects_changed_artifact(self) -> None:
        original = mx.random.normal((64, 128)).astype(mx.bfloat16)
        settings = MODES["nvfp4"]
        weight, scales, biases = quantize_weight(original, settings)
        mx.eval(weight, scales)
        with tempfile.TemporaryDirectory() as temporary:
            head_dir = Path(temporary)
            artifact = head_dir / "lm_head.safetensors"
            tensors = {"weight": weight, "scales": scales}
            if biases is not None:
                tensors["biases"] = biases
            mx.save_safetensors(
                str(artifact),
                tensors,
                metadata={"format": "nemotron-mlx-mtp-head-v1", "mode": "nvfp4"},
            )
            report = {
                "format": "nemotron-mlx-mtp-head-v1",
                "status": "complete",
                "source_revision": "revision",
                "source_shape": [64, 128],
                "source_dtype": "mlx.core.bfloat16",
                "quantization": settings,
                "payload_bytes": sum(value.nbytes for value in tensors.values()),
                "artifact": artifact.name,
                "artifact_sha256": sha256_file(artifact),
            }
            report_path = head_dir / "nemotron_mtp_head_report.json"
            report_path.write_text(json.dumps(report))
            head = QuantizedMTPHead(head_dir, "revision")
            output = head(mx.ones((1, 128), dtype=mx.bfloat16))
            mx.eval(output)
            self.assertEqual(output.shape, (1, 64))

            report["artifact_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(MetadataError, "artifact hash mismatch"):
                QuantizedMTPHead(head_dir, "revision")

    def test_balanced_vocabulary_ranking_and_required_fill_are_deterministic(self) -> None:
        training = {
            "large": Counter({4: 90, 5: 10}),
            "small": Counter({5: 9, 6: 1}),
        }
        self.assertEqual(rank_tokens(training, "raw-frequency")[:3], [4, 5, 6])
        self.assertEqual(rank_tokens(training, "balanced-frequency")[:3], [5, 4, 6])
        self.assertEqual(
            select_token_ids(10, 6, {0, 9}, [5, 4, 6]),
            [0, 1, 4, 5, 6, 9],
        )

    def test_reduced_vocabulary_head_preserves_rows_and_maps_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            head_dir = Path(temporary)
            artifact = head_dir / "lm_head.safetensors"
            weight = mx.arange(32, dtype=mx.float32).reshape(4, 8).astype(mx.bfloat16)
            token_ids = mx.array([0, 3, 7, 11], dtype=mx.int32)
            tensors = {"weight": weight, "target_token_ids": token_ids}
            mx.save_safetensors(
                str(artifact),
                tensors,
                metadata={
                    "format": "nemotron-mlx-mtp-vocab-head-v1",
                    "selection": "balanced-frequency",
                },
            )
            report = {
                "format": "nemotron-mlx-mtp-vocab-head-v1",
                "status": "complete",
                "source_revision": "revision",
                "source_shape": [16, 8],
                "source_dtype": "mlx.core.bfloat16",
                "budget": 4,
                "payload_bytes": sum(value.nbytes for value in tensors.values()),
                "artifact": artifact.name,
                "artifact_sha256": sha256_file(artifact),
            }
            (head_dir / "nemotron_mtp_vocab_head_report.json").write_text(
                json.dumps(report)
            )
            head = ReducedVocabMTPHead(head_dir, "revision", None)
            output = head(mx.ones((1, 8), dtype=mx.bfloat16))
            mx.eval(output)
            self.assertEqual(output.shape, (1, 4))
            self.assertEqual(head.target_token_ids.tolist(), [0, 3, 7, 11])

            sidecar = object.__new__(NemotronMTPSidecar)
            sidecar.draft_token_ids = head.target_token_ids
            logits = mx.array([0.0, 4.0, 2.0, 3.0])
            self.assertEqual(sidecar.argmax_token(logits), 3)
            self.assertEqual(set(sidecar.top_token_ids(logits, 2)), {3, 11})

    def test_reduced_vocabulary_head_can_gather_shared_target_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            head_dir = Path(temporary)
            artifact = head_dir / "lm_head.safetensors"
            source = mx.arange(16 * 64, dtype=mx.float32).reshape(16, 64).astype(mx.bfloat16)
            token_ids = mx.array([0, 3, 7, 11], dtype=mx.int32)
            tensors = {"target_token_ids": token_ids}
            mx.save_safetensors(
                str(artifact),
                tensors,
                metadata={
                    "format": "nemotron-mlx-mtp-vocab-head-v1",
                    "selection": "balanced-frequency",
                    "storage": "shared-target-bf16",
                },
            )
            report = {
                "format": "nemotron-mlx-mtp-vocab-head-v1",
                "status": "complete",
                "source_revision": "revision",
                "source_shape": [16, 64],
                "source_dtype": "mlx.core.bfloat16",
                "storage": "shared-target-bf16",
                "budget": 4,
                "payload_bytes": token_ids.nbytes,
                "artifact": artifact.name,
                "artifact_sha256": sha256_file(artifact),
            }
            (head_dir / "nemotron_mtp_vocab_head_report.json").write_text(
                json.dumps(report)
            )
            head = ReducedVocabMTPHead(
                head_dir,
                "revision",
                ModelOptBF16Linear(source),
            )
            vector = mx.linspace(-0.5, 0.75, 64, dtype=mx.float32).reshape(1, 1, 64)
            actual = head(vector)
            expected = ModelOptBF16Linear(source[token_ids])(vector)
            mx.eval(actual, expected)
            self.assertEqual(actual.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)

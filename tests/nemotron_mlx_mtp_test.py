#!/usr/bin/env python3
"""Focused tests for Nemotron MTP tracing and payload accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_mlx_mtp import mtp_payload_estimate, mtp_tensor_names  # noqa: E402
from nemotron_mlx_mtp_bench import append_trace_rows  # noqa: E402
from nemotron_mlx_mtp_pack import build_mtp_group  # noqa: E402
from nemotron_mlx_mtp_quantize import quantizable  # noqa: E402


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


if __name__ == "__main__":
    unittest.main(verbosity=2)

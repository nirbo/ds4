#!/usr/bin/env python3
"""Tests for bounded Nemotron retained-router distillation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_mlx_router_distill import (  # noqa: E402
    adam_update,
    aggregate_table,
    load_router_artifact,
    normalized_aggregate_loss,
    parse_layers,
)
from nemotron_mlx_moe_layer import NemotronLatentMoELayer  # noqa: E402
from nemotron_prune_materialize import sha256_file  # noqa: E402


class RouterDistillTest(unittest.TestCase):
    def test_parse_layers_requires_plan_subset(self) -> None:
        self.assertEqual(parse_layers("1,5", [1, 3, 5]), [1, 5])
        with self.assertRaises(ValueError):
            parse_layers("5,1", [1, 3, 5])

    def test_sparse_router_gradient_improves_fixed_selection(self) -> None:
        hidden = mx.array([[0.8, -0.4]], dtype=mx.float32)
        weight = mx.array(
            [[0.7, 0.1], [-0.2, 0.9], [-0.8, 0.0], [0.0, -0.7]],
            dtype=mx.float32,
        )
        bias = mx.zeros((4,), dtype=mx.float32)
        values = mx.array(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]],
            dtype=mx.float32,
        )
        teacher = mx.array([[0.2, -0.8]], dtype=mx.float32)

        def objective(gate):
            candidate = aggregate_table(hidden, gate, bias, values, 2, 1.0)
            return normalized_aggregate_loss(candidate, teacher)

        value_and_grad = mx.value_and_grad(objective)
        initial = float(objective(weight))
        first = mx.zeros_like(weight)
        second = mx.zeros_like(weight)
        for step in range(1, 41):
            _, gradient = value_and_grad(weight)
            weight, first, second, _ = adam_update(
                weight, gradient, first, second, step, 2e-2, 1.0
            )
            mx.eval(weight, first, second)
        final = float(objective(weight))
        self.assertLess(final, initial * 0.75)

    def test_load_router_artifact_binds_plan_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text("{}\n")
            artifact = root / "router.safetensors"
            mx.save_safetensors(
                str(artifact),
                {"layer_001.gate.weight": mx.ones((2, 4), dtype=mx.bfloat16)},
            )
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "format": "nemotron-router-distill-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "plan_sha256": sha256_file(plan),
                        "artifact": artifact.name,
                        "artifact_sha256": sha256_file(artifact),
                    }
                )
            )
            tensors, _ = load_router_artifact(
                report, plan, "revision", {"1": [3, 7]}, 4
            )
            self.assertEqual(tensors["1"].shape, (2, 4))

    def test_load_router_artifact_accepts_multisample_kd_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text("{}\n")
            plan_hash = sha256_file(plan)
            artifact = root / "router.safetensors"
            mx.save_safetensors(
                str(artifact),
                {"layer_001.gate.weight": mx.ones((2, 4), dtype=mx.bfloat16)},
                metadata={
                    "format": "nemotron-multisample-router-kd-v1",
                    "source_revision": "revision",
                    "plan_sha256": plan_hash,
                },
            )
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "format": "nemotron-multisample-router-kd-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "plan_sha256": plan_hash,
                        "artifact": artifact.name,
                        "artifact_sha256": sha256_file(artifact),
                    }
                )
            )
            tensors, loaded = load_router_artifact(
                report, plan, "revision", {"1": [3, 7]}, 4
            )
            self.assertEqual(tensors["1"].shape, (2, 4))
            self.assertEqual(loaded["format"], "nemotron-multisample-router-kd-v1")

    def test_retained_override_matches_sliced_source_router(self) -> None:
        block = NemotronLatentMoELayer.__new__(NemotronLatentMoELayer)
        block.n_group = 1
        block.topk_group = 1
        block.top_k = 2
        block.routed_scaling_factor = 5.0
        block.norm_topk_prob = True
        block.gate_weight = mx.array(
            [[0.5, 0.1], [-0.4, 0.2], [0.2, 0.7], [-0.1, -0.6]],
            dtype=mx.bfloat16,
        )
        block.correction_bias = mx.array([0.1, -0.2, 0.05, 0.0])
        block.experts = SimpleNamespace(up=SimpleNamespace(experts=4))
        hidden = mx.array([[[0.3, -0.8]]], dtype=mx.float32)
        retained = [0, 2, 3]
        source_indices, source_scores = block.route_retained(hidden, retained)
        override_indices, override_scores = block.route_retained_gate(
            hidden,
            retained,
            block.gate_weight[mx.array(retained, dtype=mx.uint32)],
        )
        mx.eval(source_indices, source_scores, override_indices, override_scores)
        self.assertEqual(source_indices.tolist(), override_indices.tolist())
        self.assertEqual(source_scores.tolist(), override_scores.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)

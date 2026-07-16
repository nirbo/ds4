#!/usr/bin/env python3
"""Tests for representative backbone low-bit pilot reporting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_fit import STATE_FORMAT, atomic_expert_artifact  # noqa: E402
from nemotron_mlx_backbone_lowbit import fit_binary_expert  # noqa: E402
from nemotron_mlx_backbone_pilot import build_report  # noqa: E402
from nemotron_prune_materialize import atomic_json, sha256_file  # noqa: E402


class BackbonePilotTest(unittest.TestCase):
    def test_complete_improving_pilot_passes_mechanism_gate(self) -> None:
        up = mx.sin(mx.arange(128 * 128).reshape(128, 128) / 29.0).astype(mx.bfloat16)
        down = mx.cos(mx.arange(128 * 128).reshape(128, 128) / 31.0).astype(mx.bfloat16)
        context = mx.sin(mx.arange(24 * 128).reshape(24, 128) / 17.0)
        expert, _ = fit_binary_expert(up, down, context[:16], context[16:])
        with tempfile.TemporaryDirectory() as temporary:
            fit_dir = Path(temporary)
            artifacts = fit_dir / "experts"
            artifacts.mkdir()
            path = artifacts / "expert-007.safetensors"
            atomic_expert_artifact(
                path,
                expert,
                layer=1,
                expert_id=7,
                source_revision="bf16-revision",
                contract_sha256="a" * 64,
                context_state_sha256="b" * 64,
                validation_rows=mx.array([0], dtype=mx.int32),
                validation_weighted_residual=mx.zeros((1, 128), dtype=mx.float32),
            )
            entry = {
                "expert": 7,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "train_routes": 16,
                "validation_routes": 8,
                "elapsed_seconds": 1.0,
                "metrics": {
                    "validation": {
                        "initial_relative_l2": 0.8,
                        "fitted_relative_l2": 0.6,
                    }
                },
                "precision_tiers": {
                    "2": {"relative_l2": 0.4},
                    "3": {"relative_l2": 0.2},
                    "4": {"relative_l2": 0.1},
                },
            }
            atomic_json(
                fit_dir / "state.json",
                {
                    "format": STATE_FORMAT,
                    "status": "complete",
                    "source_repository": "nvidia/test-bf16",
                    "source_revision": "bf16-revision",
                    "proxy_source_revision": "nvfp4-revision",
                    "layer": 1,
                    "architecture": {"experts": 8, "latent_width": 128, "hidden_width": 128},
                    "validation_context_rows": 8,
                    "group_size": 128,
                    "contract_sha256": "a" * 64,
                    "context_state_sha256": "b" * 64,
                    "experts": [7],
                    "completed": [entry],
                    "skipped": [],
                },
            )
            report = build_report(fit_dir)
            self.assertEqual(report["mechanism_gate"]["result"], "passed")
            self.assertEqual(report["deployable_binary_experts"], [7])
            self.assertEqual(report["quality_status"], "not-accepted-full-model-evidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)

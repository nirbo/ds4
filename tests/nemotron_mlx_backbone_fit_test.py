#!/usr/bin/env python3
"""Focused tests for resumable BF16 backbone expert fitting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_fit import (  # noqa: E402
    atomic_expert_artifact,
    expert_context_rows,
    parse_experts,
    validate_expert_artifact,
)
from nemotron_mlx_backbone_lowbit import fit_binary_expert  # noqa: E402
from nemotron_prune_materialize import sha256_file  # noqa: E402


class BackboneFitTest(unittest.TestCase):
    def test_expert_parser_is_sorted_and_strict(self) -> None:
        self.assertEqual(parse_experts("0,2,7", 8), [0, 2, 7])
        with self.assertRaisesRegex(Exception, "sorted and unique"):
            parse_experts("2,1", 8)

    def test_context_rows_preserve_routes_and_square_scores(self) -> None:
        contexts = {
            "latent": mx.arange(4 * 8, dtype=mx.float32).reshape(4, 8),
            "indices": mx.array([[1, 2], [0, 1], [2, 3], [1, 3]], dtype=mx.int32),
            "scores": mx.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]),
        }
        latent, weights, rows, scores = expert_context_rows(contexts, 1, 2.0)
        mx.eval(latent, weights, scores)
        self.assertEqual(rows.tolist(), [0, 1, 3])
        self.assertTrue(bool(mx.allclose(weights, mx.array([0.64, 0.49, 0.25]))))
        self.assertTrue(bool(mx.allclose(scores, mx.array([0.8, 0.7, 0.5]))))

    def test_expert_artifact_roundtrip_binds_provenance(self) -> None:
        up = mx.sin(mx.arange(128 * 128).reshape(128, 128) / 29.0).astype(mx.bfloat16)
        down = mx.cos(mx.arange(128 * 128).reshape(128, 128) / 31.0).astype(mx.bfloat16)
        context = mx.sin(mx.arange(24 * 128).reshape(24, 128) / 17.0)
        expert, _ = fit_binary_expert(up, down, context[:16], context[16:])
        identity = {
            "source_revision": "revision",
            "layer": 1,
            "architecture": {"experts": 8, "latent_width": 128, "hidden_width": 128},
            "validation_context_rows": 8,
            "group_size": 128,
            "contract_sha256": "a" * 64,
            "context_state_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "expert-007.safetensors"
            atomic_expert_artifact(
                path,
                expert,
                layer=1,
                expert_id=7,
                source_revision="revision",
                contract_sha256=identity["contract_sha256"],
                context_state_sha256=identity["context_state_sha256"],
                validation_rows=mx.array([0, 2], dtype=mx.int32),
                validation_weighted_residual=mx.zeros((2, 128), dtype=mx.float32),
            )
            entry = {
                "expert": 7,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            arrays = validate_expert_artifact(path, entry, identity)
            self.assertEqual(arrays["up.weight"].shape, (128, 4))
            broken = dict(identity, contract_sha256="c" * 64)
            with self.assertRaisesRegex(Exception, "contract mismatch"):
                validate_expert_artifact(path, entry, broken)

            out_of_range = Path(temporary) / "expert-008.safetensors"
            atomic_expert_artifact(
                out_of_range,
                expert,
                layer=1,
                expert_id=8,
                source_revision="revision",
                contract_sha256=identity["contract_sha256"],
                context_state_sha256=identity["context_state_sha256"],
                validation_rows=mx.array([8], dtype=mx.int32),
                validation_weighted_residual=mx.zeros((1, 128), dtype=mx.float32),
            )
            out_of_range_entry = {
                "expert": 8,
                "bytes": out_of_range.stat().st_size,
                "sha256": sha256_file(out_of_range),
            }
            with self.assertRaisesRegex(Exception, "row is out of range"):
                validate_expert_artifact(out_of_range, out_of_range_entry, identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)

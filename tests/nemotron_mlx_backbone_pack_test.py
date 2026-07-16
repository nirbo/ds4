#!/usr/bin/env python3
"""Integration test for exact mixed backbone layer materialization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT  # noqa: E402
from nemotron_mlx_backbone_fit import STATE_FORMAT as FIT_FORMAT  # noqa: E402
from nemotron_mlx_backbone_fit import atomic_expert_artifact  # noqa: E402
from nemotron_mlx_backbone_lowbit import BinaryExpert, affine_weight  # noqa: E402
from nemotron_mlx_backbone_mixed import load_mixed_file  # noqa: E402
from nemotron_mlx_backbone_pack import main as pack_main  # noqa: E402
from nemotron_mlx_backbone_plan import FORMAT as PLAN_FORMAT  # noqa: E402
from nemotron_prune_materialize import atomic_json, sha256_file  # noqa: E402


class BackbonePackTest(unittest.TestCase):
    def test_pack_is_exact_and_resumable(self) -> None:
        experts = 4
        latent = hidden = 128
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            tensors = {}
            weight_map = {}
            shard = "model-00001-of-00001.safetensors"
            for expert in range(experts):
                for projection in ("up_proj", "down_proj"):
                    value = mx.sin(
                        mx.arange(hidden * latent).reshape(hidden, latent) / (31.0 + expert)
                        + expert * 0.1
                    ).astype(mx.bfloat16)
                    packed, scales = mx.quantize(value, group_size=16, bits=4, mode="nvfp4")
                    prefix = f"backbone.layers.1.mixer.experts.{expert}.{projection}"
                    projection_tensors = {
                        f"{prefix}.weight": packed.view(mx.uint8),
                        f"{prefix}.weight_scale": scales,
                        f"{prefix}.weight_scale_2": mx.array([1.0], dtype=mx.float32),
                    }
                    tensors.update(projection_tensors)
                    weight_map.update({name: shard for name in projection_tensors})
            mx.save_safetensors(str(source / shard), tensors)
            atomic_json(source / "config.json", {"hybrid_override_pattern": "ME", "n_routed_experts": experts})
            atomic_json(source / "model.safetensors.index.json", {"metadata": {}, "weight_map": weight_map})
            source_state = root / "source-state.json"
            atomic_json(
                source_state,
                {
                    "format": "nemotron-source-snapshot-v1",
                    "path": str(source.resolve()),
                    "revision": "native-revision",
                    "verification": {"result": "passed"},
                },
            )

            binary_bytes = (
                hidden * (latent // 8 + latent // 128 * 4)
                + latent * (hidden // 8 + hidden // 128 * 4)
            )
            native_bytes = (
                hidden * (latent // 2 + latent // 16) + 4
                + latent * (hidden // 2 + hidden // 16) + 4
            )
            contract_path = root / "contract.json"
            atomic_json(
                contract_path,
                {
                    "format": CONTRACT_FORMAT,
                    "layer": 1,
                    "architecture": {
                        "experts": experts,
                        "latent_width": latent,
                        "hidden_width": hidden,
                    },
                    "storage_units": {
                        "binary_affine_bytes_per_expert": binary_bytes,
                        "native_nvfp4_bytes_per_expert": native_bytes,
                    },
                },
            )

            fit_dir = root / "fit"
            artifacts = fit_dir / "experts"
            artifacts.mkdir(parents=True)
            completed = []
            for expert in range(experts):
                up = mx.sin(mx.arange(hidden * latent).reshape(hidden, latent) / (29.0 + expert))
                down = mx.cos(mx.arange(latent * hidden).reshape(latent, hidden) / (37.0 + expert))
                candidate = BinaryExpert(
                    affine_weight(up.astype(mx.bfloat16), 1, 128),
                    affine_weight(down.astype(mx.bfloat16), 1, 128),
                )
                path = artifacts / f"expert-{expert:03d}.safetensors"
                atomic_expert_artifact(
                    path,
                    candidate,
                    layer=1,
                    expert_id=expert,
                    source_revision="bf16-revision",
                    contract_sha256=sha256_file(contract_path),
                    context_state_sha256="c" * 64,
                    validation_rows=mx.array([expert], dtype=mx.int32),
                    validation_weighted_residual=mx.zeros((1, latent), dtype=mx.float32),
                )
                completed.append(
                    {
                        "expert": expert,
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            fit_state_path = fit_dir / "state.json"
            atomic_json(
                fit_state_path,
                {
                    "format": FIT_FORMAT,
                    "status": "complete",
                    "source_revision": "bf16-revision",
                    "proxy_source_revision": "native-revision",
                    "layer": 1,
                    "architecture": {
                        "experts": experts,
                        "latent_width": latent,
                        "hidden_width": hidden,
                    },
                    "validation_context_rows": experts,
                    "group_size": 128,
                    "contract_sha256": sha256_file(contract_path),
                    "context_state_sha256": "c" * 64,
                    "completed": completed,
                    "skipped": [],
                },
            )
            plan_path = root / "plan.json"
            atomic_json(
                plan_path,
                {
                    "format": PLAN_FORMAT,
                    "status": "complete",
                    "source_revision": "bf16-revision",
                    "proxy_source_revision": "native-revision",
                    "proxy_source_state_sha256": sha256_file(source_state),
                    "layer": 1,
                    "contract_sha256": sha256_file(contract_path),
                    "fit_state_sha256": sha256_file(fit_state_path),
                    "budgets": {
                        "2": {
                            "binary_experts": [0, 2],
                            "native_nvfp4_experts": [1, 3],
                            "layer_payload_bytes": binary_bytes * 2 + native_bytes * 2,
                        }
                    },
                },
            )
            output = root / "mixed.safetensors"
            arguments = [
                "nemotron_mlx_backbone_pack.py",
                "--contract",
                str(contract_path),
                "--fit-dir",
                str(fit_dir),
                "--plan",
                str(plan_path),
                "--native-budget",
                "2",
                "--proxy-source-dir",
                str(source),
                "--proxy-source-state",
                str(source_state),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(pack_main(), 0)
            mixed, _ = load_mixed_file(output)
            self.assertEqual(mixed.binary.up.experts, 2)
            self.assertEqual(mixed.native.up.experts, 2)
            self.assertEqual(mixed.binary_map.tolist(), [0, -1, 1, -1])
            self.assertEqual(mixed.native_map.tolist(), [-1, 0, -1, 1])
            first_hash = sha256_file(output)
            output.with_suffix(".report.json").unlink()
            with patch.object(sys, "argv", arguments):
                self.assertEqual(pack_main(), 0)
            self.assertEqual(sha256_file(output), first_hash)
            with patch.object(sys, "argv", arguments):
                self.assertEqual(pack_main(), 0)
            self.assertEqual(sha256_file(output), first_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)

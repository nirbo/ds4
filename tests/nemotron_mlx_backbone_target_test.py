#!/usr/bin/env python3
"""Numerical tests for the exact native-QAT backbone teacher."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_fit import NVFP4TargetWeights  # noqa: E402


class BackboneTargetTest(unittest.TestCase):
    def test_dense_teacher_matches_native_nvfp4_qmm(self) -> None:
        latent = 64
        hidden = 32
        layer = 1
        shard = "model-00001-of-00001.safetensors"
        tensors = {}
        weight_map = {}
        for projection, rows, columns in (
            ("up_proj", hidden, latent),
            ("down_proj", latent, hidden),
        ):
            prefix = f"backbone.layers.{layer}.mixer.experts.0.{projection}"
            packed = mx.array(
                [(index * 29 + 0x31) & 0xFF for index in range(rows * columns // 2)],
                dtype=mx.uint8,
            ).reshape(rows, columns // 2)
            scales = mx.array(
                [0x38 + (index & 1) * 8 for index in range(rows * columns // 16)],
                dtype=mx.uint8,
            ).reshape(rows, columns // 16)
            values = {
                f"{prefix}.weight": packed,
                f"{prefix}.weight_scale": scales,
                f"{prefix}.weight_scale_2": mx.array([0.03125], dtype=mx.float32),
            }
            tensors.update(values)
            weight_map.update({name: shard for name in values})

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            mx.save_safetensors(str(source / shard), tensors)
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}),
                encoding="utf-8",
            )
            target = NVFP4TargetWeights(source, layer, 1, latent, hidden)
            up, down = target.expert(0)
            for projection, dense, columns in (
                ("up_proj", up, latent),
                ("down_proj", down, hidden),
            ):
                prefix = f"backbone.layers.{layer}.mixer.experts.0.{projection}"
                packed = tensors[f"{prefix}.weight"]
                scales = tensors[f"{prefix}.weight_scale"]
                global_scale = tensors[f"{prefix}.weight_scale_2"].reshape(())
                values = mx.sin(mx.arange(3 * columns, dtype=mx.float32).reshape(3, columns) * 0.17)
                dense_output = values @ dense.T
                native_output = mx.quantized_matmul(
                    values * global_scale,
                    packed.view(mx.uint32),
                    scales,
                    transpose=True,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                )
                mx.eval(dense_output, native_output)
                difference = dense_output - native_output
                relative_l2 = math.sqrt(
                    float(mx.sum(mx.square(difference)))
                    / max(float(mx.sum(mx.square(native_output))), 1e-30)
                )
                self.assertLess(relative_l2, 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

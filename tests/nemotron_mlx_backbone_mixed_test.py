#!/usr/bin/env python3
"""Numerical tests for fused binary/native-NVFP4 expert execution."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_lowbit import affine_weight  # noqa: E402
from nemotron_mlx_backbone_mixed import (  # noqa: E402
    BinaryExpertMLP,
    MixedExpertMLP,
    binary_switch_from_affine,
    build_maps,
    mixed_expert_outputs,
    mixed_expert_mlp,
    mixed_file_tensors,
    mixed_layer_forward,
    load_mixed_file,
    mixed_switch,
    mixed_switch_reference,
    FILE_FORMAT,
)
from nemotron_mlx_moe import NVFP4ExpertMLP, NVFP4SwitchWeight  # noqa: E402


def binary_bank(original: mx.array):
    values = [affine_weight(original[index].astype(mx.bfloat16), 1, 128) for index in range(original.shape[0])]
    return binary_switch_from_affine(values)


def native_bank(original: mx.array):
    packed, scales = mx.quantize(
        original.astype(mx.bfloat16), group_size=16, bits=4, mode="nvfp4"
    )
    return NVFP4SwitchWeight(
        packed.view(mx.uint8),
        scales,
        mx.ones((original.shape[0],), dtype=mx.float32),
    )


class BackboneMixedTest(unittest.TestCase):
    def mixed(self) -> MixedExpertMLP:
        low_up = mx.sin(mx.arange(2 * 128 * 128).reshape(2, 128, 128) / 31.0) * 0.08
        low_down = mx.cos(mx.arange(2 * 128 * 128).reshape(2, 128, 128) / 37.0) * 0.07
        high_up = mx.sin(mx.arange(2 * 128 * 128).reshape(2, 128, 128) / 41.0) * 0.09
        high_down = mx.cos(mx.arange(2 * 128 * 128).reshape(2, 128, 128) / 43.0) * 0.06
        binary = BinaryExpertMLP(binary_bank(low_up), binary_bank(low_down))
        native = NVFP4ExpertMLP(native_bank(high_up), native_bank(high_down))
        binary_map, native_map = build_maps(4, [0, 2], [1, 3])
        result = MixedExpertMLP(binary, native, binary_map, native_map)
        result.validate()
        return result

    def test_fused_shared_projection_matches_reference(self) -> None:
        mixed = self.mixed()
        x = mx.sin(mx.arange(2 * 128).reshape(1, 2, 128) / 17.0)
        indices = mx.array([[[0, 1, 2, 3], [3, 2, 1, 0]]], dtype=mx.int32)
        expected = mixed_switch_reference(x, mixed, indices, "up")
        actual = mixed_switch(x, mixed, indices, "up")
        mx.eval(expected, actual)
        self.assertEqual(actual.shape, (1, 2, 4, 1, 128))
        self.assertTrue(bool(mx.allclose(actual, expected, rtol=3e-4, atol=3e-3)))

    def test_fused_complete_expert_matches_reference_composition(self) -> None:
        mixed = self.mixed()
        x = mx.cos(mx.arange(128).reshape(1, 1, 128) / 23.0)
        indices = mx.array([[[0, 3, 2, 1]]], dtype=mx.int32)
        up = mixed_switch_reference(x, mixed, indices, "up")
        hidden = mx.square(mx.maximum(up, 0.0))
        expected = mixed_switch_reference(hidden.squeeze(-2), mixed, indices, "down").squeeze(-2)
        actual = mixed_expert_outputs(x, mixed, indices)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.allclose(actual, expected, rtol=4e-4, atol=4e-3)))

    def test_maps_reject_overlap(self) -> None:
        with self.assertRaisesRegex(Exception, "overlap"):
            build_maps(4, [0, 1], [1, 2, 3])

    def test_mixed_file_roundtrip_and_complete_layer(self) -> None:
        mixed = self.mixed()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.safetensors"
            mx.save_safetensors(
                str(path),
                mixed_file_tensors(mixed),
                metadata={"format": FILE_FORMAT, "layer": "1"},
            )
            loaded, metadata = load_mixed_file(path)
            self.assertEqual(metadata["layer"], "1")

            class Block:
                def norm(self, value):
                    return value

                def route(self, value):
                    del value
                    return (
                        mx.array([[[0, 1, 2, 3]]], dtype=mx.int32),
                        mx.full((1, 1, 4), 0.25, dtype=mx.float32),
                    )

                def fc1_latent(self, value):
                    return value

                def fc2_latent(self, value):
                    return value

                def shared_up(self, value):
                    return mx.zeros_like(value)

                def shared_down(self, value):
                    return value

            x = mx.sin(mx.arange(128).reshape(1, 1, 128) / 19.0)
            output, indices, scores = mixed_layer_forward(Block(), loaded, x)
            expected = x + mixed_expert_mlp(x, loaded, indices, scores)
            mx.eval(output, expected)
            self.assertTrue(bool(mx.allclose(output, expected, rtol=4e-4, atol=4e-3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

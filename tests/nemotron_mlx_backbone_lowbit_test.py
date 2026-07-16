#!/usr/bin/env python3
"""Numerical tests for activation-fitted Nemotron backbone experts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_lowbit import (  # noqa: E402
    affine_weight,
    dequantize_affine,
    expert_output,
    fit_affine_expert,
    fit_binary_expert,
    fit_projection_endpoints,
    kmeans_affine_codes,
    kmeans_affine_weight,
    pack_codes,
    precision_metrics,
    refine_binary_expert,
    relu2_channel_equalize,
    reconstruct_affine,
    unpack_codes,
)


class BackboneLowBitTest(unittest.TestCase):
    def test_code_pack_roundtrip_all_supported_bits(self) -> None:
        columns = 128
        for bits in (1, 2, 3, 4):
            codes = (mx.arange(3 * columns).reshape(3, columns) * 7 + 3) % (1 << bits)
            packed = pack_codes(codes.astype(mx.uint32), bits)
            restored = unpack_codes(packed, columns, bits)
            mx.eval(restored)
            self.assertTrue(bool(mx.array_equal(codes.astype(mx.uint32), restored)), bits)

    def test_affine_storage_and_reconstruction_contract(self) -> None:
        value = mx.sin(mx.arange(7 * 128).reshape(7, 128) / 19.0).astype(mx.bfloat16)
        quantized = affine_weight(value, 1, 128)
        quantized.validate()
        restored = dequantize_affine(quantized)
        mx.eval(restored)
        self.assertEqual(restored.shape, value.shape)
        self.assertEqual(quantized.payload_bytes, 7 * (16 + 2 + 2))

    def test_float32_affine_input_stores_bf16_endpoints(self) -> None:
        value = mx.sin(mx.arange(7 * 128).reshape(7, 128) / 19.0).astype(mx.float32)
        for bits in (2, 3, 4):
            quantized = affine_weight(value, bits, 128)
            quantized.validate()
            self.assertEqual(quantized.scales.dtype, mx.bfloat16)
            self.assertEqual(quantized.biases.dtype, mx.bfloat16)

    def test_projection_endpoint_fit_reduces_activation_error(self) -> None:
        rows = 6
        columns = 128
        teacher = (
            mx.sin(mx.arange(rows * columns).reshape(rows, columns) / 23.0) * 0.3
            + mx.cos(mx.arange(rows * columns).reshape(rows, columns) / 71.0) * 0.1
        ).astype(mx.bfloat16)
        inputs = mx.sin(mx.arange(48 * columns).reshape(48, columns) / 17.0)
        target = inputs @ teacher.astype(mx.float32).T
        codes, scales, biases = kmeans_affine_codes(teacher, 1, 128)
        initial = reconstruct_affine(codes, scales, biases)
        fitted_scales, fitted_biases = fit_projection_endpoints(
            inputs,
            target,
            teacher,
            codes,
            scales,
            biases,
            mx.linspace(0.5, 1.5, inputs.shape[0]),
        )
        fitted = reconstruct_affine(codes, fitted_scales, fitted_biases)
        initial_error = mx.sum(mx.square(inputs @ initial.T - target))
        fitted_error = mx.sum(mx.square(inputs @ fitted.T - target))
        mx.eval(initial_error, fitted_error)
        self.assertLess(float(fitted_error), float(initial_error))

    def test_function_order_fit_improves_heldout_expert_output(self) -> None:
        latent = 128
        hidden = 128
        teacher_up = (
            mx.sin(mx.arange(hidden * latent).reshape(hidden, latent) / 29.0) * 0.08
            + mx.cos(mx.arange(hidden * latent).reshape(hidden, latent) / 101.0) * 0.03
        ).astype(mx.bfloat16)
        teacher_down = (
            mx.cos(mx.arange(latent * hidden).reshape(latent, hidden) / 31.0) * 0.07
            - mx.sin(mx.arange(latent * hidden).reshape(latent, hidden) / 89.0) * 0.02
        ).astype(mx.bfloat16)
        contexts = mx.sin(mx.arange(80 * latent).reshape(80, latent) / 37.0) + 0.15 * mx.cos(
            mx.arange(80 * latent).reshape(80, latent) / 13.0
        )
        fitted, metrics = fit_binary_expert(
            teacher_up,
            teacher_down,
            contexts[:64],
            contexts[64:],
            train_weights=mx.linspace(0.25, 1.75, 64),
            validation_weights=mx.ones((16,)),
        )
        fitted.validate()
        self.assertLess(
            metrics["train"]["fitted_relative_l2"],
            metrics["train"]["initial_relative_l2"],
        )
        self.assertLess(
            metrics["validation"]["fitted_relative_l2"],
            metrics["validation"]["initial_relative_l2"],
        )

    def test_precision_screen_orders_synthetic_tiers(self) -> None:
        up = mx.sin(mx.arange(128 * 128).reshape(128, 128) / 47.0).astype(mx.bfloat16)
        down = mx.cos(mx.arange(128 * 128).reshape(128, 128) / 43.0).astype(mx.bfloat16)
        latent = mx.sin(mx.arange(12 * 128).reshape(12, 128) / 17.0)
        metrics = precision_metrics(up, down, latent, mx.ones((12,)))
        self.assertGreater(metrics["2"]["relative_l2"], metrics["4"]["relative_l2"])
        self.assertLess(metrics["2"]["payload_bytes"], metrics["4"]["payload_bytes"])

    def test_multibit_endpoint_fit_reduces_training_function_error(self) -> None:
        up = (
            mx.sin(mx.arange(128 * 128).reshape(128, 128) / 31.0) * 0.08
            + mx.cos(mx.arange(128 * 128).reshape(128, 128) / 103.0) * 0.02
        ).astype(mx.bfloat16)
        down = (
            mx.cos(mx.arange(128 * 128).reshape(128, 128) / 37.0) * 0.07
            - mx.sin(mx.arange(128 * 128).reshape(128, 128) / 97.0) * 0.02
        ).astype(mx.bfloat16)
        contexts = mx.sin(mx.arange(64 * 128).reshape(64, 128) / 23.0)
        fitted, metrics = fit_affine_expert(
            up,
            down,
            contexts[:48],
            contexts[48:],
            bits=2,
        )
        fitted.validate()
        self.assertEqual(fitted.up.bits, 2)
        self.assertLess(
            metrics["train"]["fitted_relative_l2"],
            metrics["train"]["codebook_relative_l2"],
        )

    def test_kmeans_affine_weight_roundtrips_supported_storage(self) -> None:
        value = mx.sin(mx.arange(5 * 128).reshape(5, 128) / 29.0).astype(mx.bfloat16)
        for bits in (1, 2, 3, 4):
            quantized = kmeans_affine_weight(value, bits, 128)
            quantized.validate()
            self.assertEqual(dequantize_affine(quantized).shape, value.shape)

    def test_relu2_channel_equalization_preserves_expert_function(self) -> None:
        up = (
            mx.sin(mx.arange(256 * 128).reshape(256, 128) / 31.0) * 0.08
        ).astype(mx.float32)
        column_scale = mx.exp(mx.linspace(-1.5, 1.5, 256))
        down = (
            mx.cos(mx.arange(128 * 256).reshape(128, 256) / 43.0)
            * column_scale[None, :]
            * 0.05
        ).astype(mx.float32)
        contexts = mx.sin(mx.arange(16 * 128).reshape(16, 128) / 19.0)
        equalized_up, equalized_down, scale = relu2_channel_equalize(
            up,
            down,
            group_size=128,
            strength=0.75,
        )
        reference = expert_output(contexts, up, down)
        candidate = expert_output(contexts, equalized_up, equalized_down)
        relative = mx.linalg.norm(candidate - reference) / mx.maximum(mx.linalg.norm(reference), 1e-8)
        mx.eval(relative, scale)
        self.assertLess(float(relative), 2e-6)
        self.assertTrue(bool(mx.all(scale > 0.0)))

    def test_binary_codes_can_fit_a_distinct_qat_target(self) -> None:
        source_up = mx.sin(mx.arange(128 * 128).reshape(128, 128) / 41.0).astype(mx.bfloat16)
        source_down = mx.cos(mx.arange(128 * 128).reshape(128, 128) / 37.0).astype(mx.bfloat16)
        target_up = (source_up.astype(mx.float32) * 0.92 + 0.003).astype(mx.bfloat16)
        target_down = (source_down.astype(mx.float32) * 1.06 - 0.002).astype(mx.bfloat16)
        contexts = mx.sin(mx.arange(48 * 128).reshape(48, 128) / 19.0)
        _, metrics = fit_binary_expert(
            source_up,
            source_down,
            contexts[:32],
            contexts[32:],
            target_up=target_up,
            target_down=target_down,
        )
        self.assertLess(
            metrics["validation"]["fitted_relative_l2"],
            metrics["validation"]["initial_relative_l2"],
        )

    def test_heldout_refinement_never_returns_a_worse_checkpoint(self) -> None:
        source_up = mx.sin(mx.arange(128 * 128).reshape(128, 128) / 41.0).astype(mx.bfloat16)
        source_down = mx.cos(mx.arange(128 * 128).reshape(128, 128) / 37.0).astype(mx.bfloat16)
        target_up = (source_up.astype(mx.float32) * 0.93 + 0.002).astype(mx.bfloat16)
        target_down = (source_down.astype(mx.float32) * 1.04 - 0.001).astype(mx.bfloat16)
        contexts = mx.sin(mx.arange(64 * 128).reshape(64, 128) / 19.0)
        train_weights = mx.linspace(0.5, 1.5, 48)
        validation_weights = mx.ones((16,))
        initial, _ = fit_binary_expert(
            source_up,
            source_down,
            contexts[:48],
            contexts[48:],
            train_weights,
            validation_weights,
            target_up=target_up,
            target_down=target_down,
        )
        refined, report = refine_binary_expert(
            initial,
            source_up,
            source_down,
            contexts[:48],
            contexts[48:],
            train_weights,
            validation_weights,
            target_up=target_up,
            target_down=target_down,
            steps=8,
            batch_size=16,
            learning_rate=1e-3,
            code_learning_rate=2e-3,
            code_warmup_steps=4,
            evaluate_every=2,
        )
        refined.validate()
        self.assertLessEqual(report["best"]["error2"], report["initial"]["error2"])
        self.assertGreater(report["optimizer_state_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

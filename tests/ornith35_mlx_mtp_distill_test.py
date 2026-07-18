#!/usr/bin/env python3
"""Autograd, checkpoint, and strict artifact tests for Ornith-35 MTP tuning."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_attention as attention
import ornith35_mlx_layer as layer
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_distill as distill
import ornith35_mlx_mtp_test as fixture


class MTPDistillationTest(unittest.TestCase):
    def test_production_width_training_rmsnorm_matches_runtime_and_has_vjp(self) -> None:
        mx.random.seed(20260718)
        hidden = mx.random.uniform(-2.0, 2.0, shape=(3, 2048)).astype(mx.bfloat16)
        weight = mx.random.uniform(-0.2, 0.2, shape=(2048,)).astype(mx.bfloat16)
        expected = layer.qwen_rms_norm_batch(hidden, weight)
        actual = distill._training_rms_norm_batch(hidden, weight, 1e-6)

        def objective(value: mx.array) -> mx.array:
            normalized = distill._training_rms_norm_batch(value, weight, 1e-6)
            return mx.sum(normalized.astype(mx.float32))

        loss, gradient = mx.value_and_grad(objective)(hidden)
        mx.eval(expected, actual, loss, gradient)
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))
        self.assertTrue(bool(mx.all(mx.isfinite(gradient)).item()))
        self.assertGreater(float(mx.max(mx.abs(gradient.astype(mx.float32)))), 0.0)

    def test_autograd_safe_forward_matches_unadapted_reference_composition(self) -> None:
        config, scalar_weights = fixture.make_fixture()
        weights = fixture.mlx_weights(scalar_weights)
        token_embeddings = mx.array(
            [[(row + column + 1) * 0.003 for column in range(16)] for row in range(3)],
            dtype=mx.float32,
        )
        target_hidden = mx.array(
            [[(row * 3 - column + 1) * 0.002 for column in range(16)] for row in range(3)],
            dtype=mx.float32,
        )
        state = attention.zeros_state(config.attention, dtype=mx.float32)
        expected = mtp.prefill_steps(
            token_embeddings,
            target_hidden,
            state,
            weights,
            config,
            exact_long_attention=False,
        )
        actual, actual_state = distill.training_forward_chunk(
            token_embeddings,
            target_hidden,
            state,
            weights,
            config,
        )
        mx.eval(actual, actual_state.keys, actual_state.values, expected.hidden)
        self.assertTrue(bool(mx.allclose(actual, expected.hidden, rtol=1e-5, atol=1e-6).item()))
        self.assertTrue(bool(mx.allclose(actual_state.keys, expected.state.keys).item()))
        self.assertTrue(bool(mx.allclose(actual_state.values, expected.state.values).item()))

    def test_zero_initialized_lora_has_finite_nonzero_first_update(self) -> None:
        config, scalar_weights = fixture.make_fixture()
        weights = fixture.mlx_weights(scalar_weights)
        parameters = distill.initialize_adapter(config.hidden_size, rank=2, seed=9)
        token_embeddings = mx.ones((2, 16), dtype=mx.float32) * 0.03
        target_hidden = mx.ones((2, 16), dtype=mx.float32) * -0.02
        teacher_hidden = mx.ones((2, 16), dtype=mx.float32) * 0.07
        candidate_head = mx.arange(2 * 4 * 16, dtype=mx.float32).reshape(2, 4, 16) * 0.001
        teacher_logits = mx.array([[2.0, 1.0, 0.0, -1.0], [1.5, 0.5, 0.0, -0.5]])
        scored = mx.ones((2,), dtype=mx.int32)
        state = attention.zeros_state(config.attention, dtype=mx.float32)
        value_and_grad = mx.value_and_grad(distill.distillation_loss)
        result, gradients = value_and_grad(
            parameters,
            token_embeddings,
            target_hidden,
            teacher_hidden,
            candidate_head,
            teacher_logits,
            scored,
            state,
            weights,
            config,
            1.0,
            1.0,
            0.1,
            0.1,
            1.0,
        )
        mx.eval(result[0], gradients["lora_a"], gradients["lora_b"])
        self.assertTrue(bool(mx.all(mx.isfinite(gradients["lora_b"])).item()))
        self.assertGreater(float(mx.max(mx.abs(gradients["lora_b"]))), 0.0)
        self.assertEqual(float(mx.max(mx.abs(gradients["lora_a"]))), 0.0)

    def test_checkpoint_roundtrip_restores_optimizer_and_best_parameters(self) -> None:
        parameters = distill.initialize_adapter(2048, rank=2, seed=3)
        optimizer = distill.AdamW(learning_rate=1e-3, weight_decay=0.0)
        optimizer.initialize(parameters)
        gradients = {name: mx.ones_like(value) * 0.01 for name, value in parameters.items()}
        parameters = optimizer.update(parameters, gradients)
        mx.eval(*parameters.values())
        best = distill._copy_parameters(parameters)
        identity = {"test": "checkpoint"}
        metrics = {"candidate_acceptance": 0.5, "hidden_relative_l2": 0.25}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            distill.save_checkpoint(
                root,
                identity,
                1,
                parameters,
                optimizer,
                best,
                metrics,
                [{"epoch": 1}],
            )
            restored_optimizer = distill.AdamW(learning_rate=1e-3, weight_decay=0.0)
            epoch, restored, restored_best, restored_metrics, history = distill.load_checkpoint(
                root,
                identity,
                2048,
                2,
                restored_optimizer,
            )
        self.assertEqual(epoch, 1)
        self.assertEqual(restored_optimizer.step, 1)
        self.assertEqual(restored_metrics, metrics)
        self.assertEqual(history, [{"epoch": 1}])
        for name in parameters:
            self.assertTrue(bool(mx.array_equal(parameters[name], restored[name]).item()))
            self.assertTrue(bool(mx.array_equal(parameters[name], mx.array(restored_best[name])).item()))

    def test_strict_adaptation_loader_detects_artifact_corruption(self) -> None:
        source = mx.zeros((2048, 4096), dtype=mx.bfloat16)
        parameters = distill.initialize_adapter(2048, rank=2, seed=5)
        parameters["lora_b"] = mx.ones_like(parameters["lora_b"]) * 0.0001
        adapted = distill.merged_fc(source, parameters, scale=1.0)
        metrics = {"candidate_acceptance": 0.5, "hidden_relative_l2": 0.25}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            distill.write_adaptation(
                root,
                adapted,
                status="diagnostic",
                identity={"test": "artifact"},
                capture_state_sha256="c" * 64,
                baseline=metrics,
                best=metrics,
                merged=metrics,
                history=[],
                rank=2,
                alpha=2.0,
                elapsed_seconds=0.1,
            )
            loaded = mtp.load_fc_adaptation(root, source, allow_diagnostic=True)
            self.assertTrue(bool(mx.array_equal(loaded, adapted).item()))
            artifact = root / mtp.ADAPTATION_ARTIFACT
            with artifact.open("r+b") as handle:
                handle.seek(-1, 2)
                original = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([original[0] ^ 1]))
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                mtp.load_fc_adaptation(root, source, allow_diagnostic=True)


if __name__ == "__main__":
    unittest.main()

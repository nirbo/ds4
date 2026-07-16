#!/usr/bin/env python3
"""Focused tests for Nemotron MTP tracing and payload accounting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_mtp import (  # noqa: E402
    NemotronMTPCache,
    NemotronMTPSidecar,
    QuantizedMTPHead,
    ReducedVocabMTPHead,
    load_sidecar_linear,
    mtp_payload_estimate,
    mtp_tensor_names,
    sidecar_quantization_settings,
    validate_expert_bank_metadata,
)
from nemotron_mlx_mtp_bench import append_trace_rows, build_plan, load_prompts  # noqa: E402
from nemotron_mlx_mtp_binary_fit import (  # noqa: E402
    aggregate_projection_target,
    fit_group_endpoints,
    fit_projection_endpoints,
    replaced_aggregate,
    refine_projection_codes,
    reconstruct_binary,
)
from nemotron_mlx_mtp_chain_bench import cache_warms_unscored  # noqa: E402
from nemotron_mlx_mtp_head_quantize import MODES, quantize_weight  # noqa: E402
from nemotron_mlx_mtp_lowbit_plan import rank_experts  # noqa: E402
from nemotron_mlx_mtp_lowbit_sensitivity import masked_objective  # noqa: E402
from nemotron_mlx_mtp_mixed import mixed_affine_switch  # noqa: E402
from nemotron_mlx_mtp_pack import build_mtp_group  # noqa: E402
from nemotron_mlx_mtp_quantize import (  # noqa: E402
    MODES as SIDECAR_MODES,
    binary_affine_supported,
    binary_kmeans_quantize,
    binary_quantize,
    main as quantize_mtp_main,
    parse_tensor_modes,
    quantizable,
    quantize_tensor,
)
from nemotron_mlx_mtp_vocab_head import rank_tokens, select_token_ids  # noqa: E402
from nemotron_mlx_linear import ModelOptBF16Linear  # noqa: E402
from nemotron_prune_materialize import sha256_file  # noqa: E402


class MLXMTPTest(unittest.TestCase):
    def test_mixed_bank_metadata_requires_known_format(self) -> None:
        banks = [{"name": "low"}, {"name": "high"}]
        self.assertEqual(
            validate_expert_bank_metadata(
                {"format": "nemotron-mtp-lowbit-banks-v1", "banks": banks}
            ),
            banks,
        )
        with self.assertRaisesRegex(MetadataError, "metadata"):
            validate_expert_bank_metadata({"format": "unknown", "banks": banks})

    def test_mtp_prompt_json_interleaves_categories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            path.write_text(json.dumps({"b": ["b0"], "a": ["a0", "a1"]}))
            self.assertEqual(load_prompts(path), ["a0", "b0", "a1"])

    def test_recursive_cache_modes_warm_only_prompt_history(self) -> None:
        self.assertFalse(cache_warms_unscored("none"))
        self.assertFalse(cache_warms_unscored("generated"))
        self.assertTrue(cache_warms_unscored("prompt"))
        with self.assertRaises(MetadataError):
            cache_warms_unscored("invalid")

    def test_mtp_cache_checkpoint_restore_and_reset(self) -> None:
        cache = NemotronMTPCache()
        first_keys = mx.arange(8, dtype=mx.float32).reshape(1, 2, 1, 4)
        first_values = first_keys + 100
        cache.attention.update_and_fetch(first_keys, first_values)
        mx.eval(cache.attention.keys, cache.attention.values)
        checkpoint = cache.checkpoint()

        second_keys = first_keys + 10
        second_values = first_values + 10
        cache.attention.update_and_fetch(second_keys, second_values)
        mx.eval(cache.attention.keys, cache.attention.values)
        self.assertEqual(cache.offset, 2)
        self.assertGreater(cache.nbytes, 0)

        cache.restore(checkpoint)
        self.assertEqual(cache.offset, 1)
        self.assertEqual(cache.attention.state[0].tolist(), first_keys.tolist())
        self.assertEqual(cache.attention.state[1].tolist(), first_values.tolist())
        with self.assertRaises(MetadataError):
            cache.restore(2)

        cache.reset()
        self.assertEqual(cache.offset, 0)
        replacement_keys = first_keys + 20
        replacement_values = first_values + 20
        cache.attention.update_and_fetch(replacement_keys, replacement_values)
        mx.eval(cache.attention.keys, cache.attention.values)
        self.assertEqual(cache.attention.state[0].tolist(), replacement_keys.tolist())
        self.assertEqual(cache.attention.state[1].tolist(), replacement_values.tolist())

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

    def test_mtp_plan_appends_unobserved_source_experts_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({"n_routed_experts": 5}))
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "format": "nemotron-mtp-acceptance-v1",
                        "source_dir": str(source),
                        "scored_expert_score_mass": {"3": 0.5, "1": 0.75},
                        "scored_expert_counts": {"3": 8, "1": 2},
                    }
                )
            )
            output = root / "plan.json"
            args = type("Args", (), {"report": report_path, "output": output, "budgets": [3, 5]})
            self.assertEqual(build_plan(args), 0)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["budgets"]["3"], [1, 3, 0])
            self.assertEqual(plan["budgets"]["5"], [1, 3, 0, 2, 4])
            self.assertEqual(plan["declared_experts"], 5)
            self.assertEqual(plan["observed_experts"], 2)

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

    def test_ternary_mtp_quantization_uses_only_three_affine_codes(self) -> None:
        original = (
            mx.arange(256, dtype=mx.float32).reshape(2, 128) / 31.0 - 4.0
        ).astype(mx.bfloat16)
        weight, scales, biases = quantize_tensor(original, "ternary2-g128")
        self.assertIsNotNone(biases)
        mx.eval(weight, scales, biases)
        shifts = mx.arange(16, dtype=mx.uint32) * 2
        codes = (weight[..., None] >> shifts) & 3
        mx.eval(codes)
        self.assertLessEqual(int(mx.max(codes)), 2)
        restored = mx.dequantize(
            weight,
            scales,
            biases,
            **SIDECAR_MODES["ternary2-g128"],
            dtype=mx.float32,
        )
        mx.eval(restored)
        grouped = restored.reshape(2, 1, 128)
        expected_scale = scales.astype(mx.float32)
        normalized = mx.where(expected_scale[..., None] == 0, 0, grouped / expected_scale[..., None])
        mx.eval(normalized)
        self.assertTrue(bool(mx.all((normalized == -1) | (normalized == 0) | (normalized == 1))))
        self.assertEqual(weight.nbytes, original.size * 2 // 8)

    def test_binary_mtp_quantization_is_true_one_bit_symmetric_storage(self) -> None:
        original = (
            mx.arange(256, dtype=mx.float32).reshape(2, 128) / 31.0 - 4.0
        ).astype(mx.bfloat16)
        weight, scales, biases = binary_quantize(original, 128)
        mx.eval(weight, scales, biases)
        shifts = mx.arange(32, dtype=mx.uint32)
        codes = ((weight[..., None] >> shifts) & 1).reshape(original.shape)
        restored = (
            codes.reshape(2, 1, 128).astype(mx.float32) * scales.astype(mx.float32)[..., None]
            + biases.astype(mx.float32)[..., None]
        ).reshape(original.shape)
        mx.eval(codes, restored)
        magnitude = (-biases).astype(mx.float32)
        normalized = restored.reshape(2, 1, 128) / magnitude[..., None]
        mx.eval(normalized)
        self.assertTrue(bool(mx.all((codes == 0) | (codes == 1))))
        self.assertTrue(bool(mx.all((normalized == -1) | (normalized == 1))))
        self.assertEqual(weight.nbytes, original.size // 8)
        self.assertEqual(scales.shape, (2, 1))
        self.assertEqual(biases.shape, (2, 1))

    def test_binary_kmeans_reduces_weight_error_without_more_payload(self) -> None:
        original = mx.concatenate(
            [
                mx.linspace(-0.2, 0.4, 128),
                mx.linspace(-4.0, 1.0, 128),
            ]
        ).reshape(2, 128).astype(mx.bfloat16)
        symmetric = binary_quantize(original, 128)
        fitted = binary_kmeans_quantize(original, 128, chunk_values=128)

        def restore(payload: tuple[mx.array, mx.array, mx.array]) -> mx.array:
            weight, scales, biases = payload
            shifts = mx.arange(32, dtype=mx.uint32)
            codes = ((weight[..., None] >> shifts) & 1).reshape(original.shape)
            return (
                codes.reshape(2, 1, 128).astype(mx.float32)
                * scales.astype(mx.float32)[..., None]
                + biases.astype(mx.float32)[..., None]
            ).reshape(original.shape)

        symmetric_error = mx.sum(mx.square(restore(symmetric) - original.astype(mx.float32)))
        fitted_error = mx.sum(mx.square(restore(fitted) - original.astype(mx.float32)))
        mx.eval(symmetric_error, fitted_error)
        self.assertLess(float(fitted_error), float(symmetric_error))
        self.assertEqual(fitted[0].nbytes, original.size // 8)

    def test_activation_fit_reduces_heldout_projection_error(self) -> None:
        teacher = (
            mx.sin(mx.arange(4 * 128, dtype=mx.float32).reshape(4, 128) / 19.0) * 0.2
        )
        codes = (teacher >= 0).astype(mx.uint32)
        initial_scales = mx.full((4, 1), 0.1, dtype=mx.bfloat16)
        initial_biases = mx.full((4, 1), -0.05, dtype=mx.bfloat16)
        train = mx.sin(mx.arange(24 * 128, dtype=mx.float32).reshape(24, 128) / 17.0)
        heldout = mx.cos(mx.arange(12 * 128, dtype=mx.float32).reshape(12, 128) / 23.0)
        scales, biases = fit_group_endpoints(
            train,
            train,
            teacher,
            codes,
            initial_scales,
            initial_biases,
            mx.ones((24,), dtype=mx.float32),
            ridge=0.01,
        )
        baseline = reconstruct_binary(codes, initial_scales, initial_biases)
        fitted = reconstruct_binary(codes, scales, biases)
        target = heldout @ teacher.T
        baseline_error = mx.sum(mx.square(heldout @ baseline.T - target))
        fitted_error = mx.sum(mx.square(heldout @ fitted.T - target))
        mx.eval(baseline_error, fitted_error)
        self.assertLess(float(fitted_error), float(baseline_error))

    def test_joint_projection_fit_reduces_cross_group_error(self) -> None:
        teacher = (
            mx.sin(mx.arange(4 * 256, dtype=mx.float32).reshape(4, 256) / 19.0) * 0.2
        )
        codes = (teacher >= 0).astype(mx.uint32)
        initial_scales = mx.full((4, 2), 0.1, dtype=mx.bfloat16)
        initial_biases = mx.full((4, 2), -0.05, dtype=mx.bfloat16)
        train = mx.sin(mx.arange(32 * 256, dtype=mx.float32).reshape(32, 256) / 17.0)
        target = train @ teacher.T
        scales, biases = fit_projection_endpoints(
            train,
            target,
            teacher,
            codes,
            initial_scales,
            initial_biases,
            mx.ones((32,), dtype=mx.float32),
            ridge=0.01,
        )
        baseline = reconstruct_binary(codes, initial_scales, initial_biases)
        fitted = reconstruct_binary(codes, scales, biases)
        baseline_error = mx.sum(mx.square(train @ baseline.T - target))
        fitted_error = mx.sum(mx.square(train @ fitted.T - target))
        mx.eval(baseline_error, fitted_error)
        self.assertLess(float(fitted_error), float(baseline_error))

    def test_binary_code_refinement_improves_wrong_sign_assignments(self) -> None:
        teacher = (
            mx.sin(mx.arange(4 * 128, dtype=mx.float32).reshape(4, 128) / 13.0) * 0.2
        )
        correct_codes = (teacher >= 0).astype(mx.uint32)
        wrong_mask = (mx.arange(128) % 7 == 0)[None, :]
        codes = mx.where(wrong_mask, 1 - correct_codes, correct_codes).astype(mx.uint32)
        train = mx.sin(mx.arange(32 * 128, dtype=mx.float32).reshape(32, 128) / 17.0)
        target = train @ teacher.T
        initial_scales = mx.full((4, 1), 0.2, dtype=mx.bfloat16)
        initial_biases = mx.full((4, 1), -0.1, dtype=mx.bfloat16)
        refined_codes, scales, biases, history = refine_projection_codes(
            train,
            target,
            teacher,
            codes,
            initial_scales,
            initial_biases,
            mx.ones((32,), dtype=mx.float32),
            ridge=0.01,
            endpoint_margin=0.25,
            iterations=4,
            flip_fraction=0.05,
        )
        baseline = reconstruct_binary(codes, initial_scales, initial_biases)
        refined = reconstruct_binary(refined_codes, scales, biases)
        baseline_error = mx.sum(mx.square(train @ baseline.T - target))
        refined_error = mx.sum(mx.square(train @ refined.T - target))
        mx.eval(baseline_error, refined_error)
        self.assertTrue(history)
        self.assertLess(float(refined_error), float(baseline_error))

    def test_aggregate_target_exactly_replaces_one_routed_expert(self) -> None:
        current = mx.array([[3.0, -2.0], [1.5, 4.0]])
        teacher = mx.array([[2.0, 1.0], [-0.5, 3.0]])
        current_expert = mx.array([[4.0, -1.0], [2.0, 5.0]])
        scores = mx.array([0.25, 0.5])
        target_contribution = aggregate_projection_target(
            current,
            teacher,
            current_expert,
            scores,
        )
        replacement_output = target_contribution / scores[:, None]
        replaced = replaced_aggregate(
            current,
            current_expert,
            replacement_output,
            scores,
        )
        mx.eval(replaced)
        self.assertTrue(bool(mx.allclose(replaced, teacher)))

    def test_lowbit_plan_prioritizes_teacher_recovery_and_penalizes_regression(self) -> None:
        fit_report = {
            "binary_fit_validation": {
                "expert_metrics": [
                    {
                        "expert": expert,
                        "validation_after_error2": 10.0 - expert,
                        "validation_reference2": 20.0,
                        "route_score_mass": 1.0,
                        "validation_samples": 2,
                    }
                    for expert in range(3)
                ]
            }
        }
        baseline = {
            "scored_row_results": [
                {
                    "row": 0,
                    "expected_token_id": 7,
                    "predicted_token_id": 8,
                    "top5_token_ids": [8, 7],
                    "routed_expert_ids": [1, 0],
                    "route_scores": [0.8, 0.2],
                },
                {
                    "row": 1,
                    "expected_token_id": 9,
                    "predicted_token_id": 9,
                    "top5_token_ids": [9],
                    "routed_expert_ids": [2, 0],
                    "route_scores": [0.9, 0.1],
                },
            ]
        }
        teacher = {
            "scored_row_results": [
                {
                    **baseline["scored_row_results"][0],
                    "predicted_token_id": 7,
                },
                {
                    **baseline["scored_row_results"][1],
                    "predicted_token_id": 10,
                },
            ]
        }
        ranked = rank_experts(fit_report, 3, [(baseline, teacher)])
        self.assertEqual([row["expert"] for row in ranked], [1, 0, 2])
        self.assertGreater(ranked[0]["task_recovery_score"], 0.0)
        self.assertLess(ranked[-1]["task_recovery_score"], 0.0)

    def test_lowbit_sensitivity_objective_masks_padding_and_rewards_expected_token(self) -> None:
        teacher = mx.array([[3.0, 1.0, -99.0]])
        valid = mx.array([[True, True, False]])
        expected = mx.array([0], dtype=mx.int32)
        matched = masked_objective(teacher, teacher, valid, expected, teacher_weight=0.25)
        worse = masked_objective(
            mx.array([[1.0, 3.0, 99.0]]),
            teacher,
            valid,
            expected,
            teacher_weight=0.25,
        )
        mx.eval(matched, worse)
        self.assertLess(float(matched.item()), float(worse.item()))

    def test_binary_runtime_capability_probe_matches_direct_decode(self) -> None:
        try:
            decoded = mx.dequantize(
                mx.zeros((1, 4), dtype=mx.uint32),
                mx.ones((1, 1), dtype=mx.bfloat16),
                -mx.ones((1, 1), dtype=mx.bfloat16),
                group_size=128,
                bits=1,
                mode="affine",
            )
            mx.eval(decoded)
            direct = decoded.shape == (1, 128)
        except (RuntimeError, ValueError):
            direct = False
        self.assertEqual(binary_affine_supported(), direct)

    @unittest.skipUnless(binary_affine_supported(), "MLX build has no one-bit affine kernels")
    def test_binary_gather_qmm_matches_explicit_dequantization(self) -> None:
        original = (
            mx.arange(3 * 37 * 128, dtype=mx.float32).reshape(3, 37, 128) / 257.0
            - 27.0
        ).astype(mx.bfloat16)
        weight, scales, biases = binary_quantize(original, 128)
        x = (mx.arange(128, dtype=mx.float32) / 31.0 - 2.0).reshape(1, 1, 1, 128)
        indices = mx.array([[[2, 0]]], dtype=mx.int32)
        actual = mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            rhs_indices=indices,
            transpose=True,
            group_size=128,
            bits=1,
            mode="affine",
        )
        restored = mx.dequantize(
            weight,
            scales,
            biases,
            group_size=128,
            bits=1,
            mode="affine",
            dtype=mx.float32,
        )
        selected = restored[indices]
        expected = x @ mx.swapaxes(selected, -1, -2)
        mx.eval(actual, expected)
        self.assertEqual(actual.shape, (1, 1, 2, 1, 37))
        self.assertTrue(bool(mx.allclose(actual, expected, rtol=2e-4, atol=2e-3)))

    @unittest.skipUnless(binary_affine_supported(), "MLX build has no one-bit affine kernels")
    def test_mixed_metal_switch_matches_two_bank_reference(self) -> None:
        columns = 128
        rows = 37
        low_original = mx.sin(mx.arange(2 * rows * columns).reshape(2, rows, columns) / 31.0)
        high_original = mx.cos(mx.arange(2 * rows * columns).reshape(2, rows, columns) / 29.0)
        low_weight, low_scales, low_biases = binary_quantize(
            low_original.astype(mx.bfloat16),
            128,
        )
        high_weight, high_scales, high_biases = quantize_tensor(
            high_original.astype(mx.bfloat16),
            "affine3-g128",
        )
        low_map = mx.array([0, -1, 1, -1], dtype=mx.int32)
        high_map = mx.array([-1, 0, -1, 1], dtype=mx.int32)
        low_bank = {
            "original_to_local": low_map,
            "up": {
                "weight": low_weight,
                "scales": low_scales,
                "biases": low_biases,
                "settings": {"group_size": 128, "bits": 1, "mode": "affine"},
            },
        }
        high_bank = {
            "original_to_local": high_map,
            "up": {
                "weight": high_weight,
                "scales": high_scales,
                "biases": high_biases,
                "settings": {"group_size": 128, "bits": 3, "mode": "affine"},
            },
        }
        indices = mx.array([[[0, 1, 2, 3]]], dtype=mx.int32)
        inputs = mx.sin(mx.arange(4 * columns).reshape(1, 1, 4, 1, columns) / 17.0)

        def bank_output(bank: dict, bits: int) -> mx.array:
            mapped = bank["original_to_local"][indices]
            selected = mapped >= 0
            output = mx.gather_qmm(
                inputs,
                bank["up"]["weight"],
                bank["up"]["scales"],
                bank["up"]["biases"],
                rhs_indices=mx.maximum(mapped, 0),
                transpose=True,
                group_size=128,
                bits=bits,
                mode="affine",
            )
            return output, selected[..., None, None]

        low_output, low_selected = bank_output(low_bank, 1)
        high_output, _ = bank_output(high_bank, 3)
        reference = mx.where(low_selected, low_output, high_output)
        actual = mixed_affine_switch(inputs, indices, low_bank, high_bank, "up")
        mx.eval(reference, actual)
        self.assertEqual(actual.shape, reference.shape)
        self.assertTrue(bool(mx.allclose(actual, reference, rtol=2e-4, atol=2e-3)))

    def test_mtp_tensor_mode_parser_rejects_conflicts(self) -> None:
        self.assertEqual(
            parse_tensor_modes(["mtp.layers.1.mixer.switch_mlp.up_proj.weight=affine2-g128"]),
            {"mtp.layers.1.mixer.switch_mlp.up_proj.weight": "affine2-g128"},
        )
        with self.assertRaisesRegex(MetadataError, "duplicate"):
            parse_tensor_modes(["a=affine2-g128", "a=affine3-g128"])
        with self.assertRaisesRegex(MetadataError, "unsupported"):
            parse_tensor_modes(["a=not-a-mode"])

    def test_mtp_quantization_selectively_retains_exact_bf16(self) -> None:
        retained_name = "mtp.layers.0.eh_proj.weight"
        low_bit_name = "mtp.layers.1.mixer.switch_mlp.up_proj.weight"
        gate_name = "mtp.layers.1.mixer.gate.weight"
        retained = mx.arange(256, dtype=mx.float32).reshape(4, 64).astype(mx.bfloat16)
        low_bit = (mx.arange(512, dtype=mx.float32).reshape(4, 128) / 127.0 - 2.0).astype(
            mx.bfloat16
        )
        gate = mx.zeros((4, 64), dtype=mx.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            artifact = source / "mtp.safetensors"
            mx.save_safetensors(
                str(artifact),
                {retained_name: retained, low_bit_name: low_bit, gate_name: gate},
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "total_size": retained.nbytes + low_bit.nbytes + gate.nbytes
                        },
                        "weight_map": {
                            retained_name: artifact.name,
                            low_bit_name: artifact.name,
                            gate_name: artifact.name,
                        },
                    }
                )
            )
            (source / "config.json").write_text(
                json.dumps({"nemotron_mtp_runtime": {"format": "nemotron-mlx-mtp-sidecar-v1"}})
            )
            (source / "nemotron_mtp_pack_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-mtp-sidecar-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "budget": 4,
                    }
                )
            )
            argv = [
                "nemotron_mlx_mtp_quantize.py",
                "--source-sidecar",
                str(source),
                "--output-dir",
                str(output),
                "--mode",
                "nvfp4",
                "--keep-bf16",
                retained_name,
                "--tensor-mode",
                f"{low_bit_name}=ternary2-g128",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(quantize_mtp_main(), 0)
            mixed = mx.load(str(output / "mtp.safetensors"))
            self.assertEqual(mixed[retained_name].dtype, mx.bfloat16)
            self.assertTrue(bool(mx.array_equal(mixed[retained_name], retained)))
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(
                config["nemotron_mtp_runtime"]["quantization"]["bf16_tensors"],
                [retained_name],
            )
            low_settings = config["nemotron_mtp_runtime"]["quantization"]["tensor_modes"][
                low_bit_name
            ]
            self.assertEqual(low_settings["recipe"], "ternary2-g128")
            self.assertEqual(low_settings["bits"], 2)
            self.assertEqual(
                sidecar_quantization_settings(
                    config["nemotron_mtp_runtime"]["quantization"], low_bit_name
                ),
                low_settings,
            )
            linear = load_sidecar_linear(
                mixed,
                "mtp.layers.0.eh_proj",
                config["nemotron_mtp_runtime"]["quantization"],
            )
            self.assertIsInstance(linear, ModelOptBF16Linear)
            low_linear = load_sidecar_linear(
                mixed,
                low_bit_name[: -len(".weight")],
                config["nemotron_mtp_runtime"]["quantization"],
            )
            self.assertEqual(low_linear.bits, 2)
            self.assertEqual(low_linear.group_size, 128)
            self.assertEqual(low_linear.mode, "affine")

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

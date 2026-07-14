#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_mtp_predictor import (  # noqa: E402
    AdamWControl,
    GATED_ARCHITECTURE,
    acceptance_score,
    initialize_parameters,
    initialize_gated_parameters,
    initialize_token_classifier,
    predict_hidden,
    predict_token_logits,
    predict_token_logits_bf16,
    predictor_parameter_count,
    sequence_starts,
    validate_token_runtime_parameters,
    validated_predictor_report,
    validate_runtime_binding,
)
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_prune_materialize import sha256_file  # noqa: E402


class MTPPredictorTest(unittest.TestCase):
    def test_parameter_count_matches_arrays(self):
        parameters = initialize_parameters(16, 4, 3, 7)
        self.assertEqual(sum(value.size for value in parameters.values()), predictor_parameter_count(16, 4, 3))

    def test_predictor_shapes_and_depth_paths(self):
        parameters = initialize_parameters(16, 4, 3, 7)
        hidden = mx.ones((2, 16))
        embedding = mx.zeros((2, 16))
        self.assertEqual(predict_hidden(parameters, hidden, embedding, 2).shape, (2, 16))

    def test_gated_and_token_only_predictors_have_expected_shapes(self):
        gated = initialize_gated_parameters(16, 4, 1, 7)
        hidden = mx.ones((2, 16))
        embedding = mx.zeros((2, 16))
        self.assertEqual(
            predict_hidden(gated, hidden, embedding, 0, GATED_ARCHITECTURE).shape,
            (2, 16),
        )
        token = initialize_token_classifier(16, 4, 32, 7)
        self.assertEqual(predict_token_logits(token, hidden, embedding).shape, (2, 32))

    def test_bf16_token_runtime_matches_quantized_training_layout(self):
        parameters = {
            name: value.astype(mx.bfloat16)
            for name, value in initialize_token_classifier(16, 4, 32, 7).items()
        }
        hidden = mx.arange(16).astype(mx.float32) / 16
        embedding = mx.arange(16, 32).astype(mx.float32) / 32
        expected = predict_token_logits(
            parameters,
            hidden.reshape(1, -1),
            embedding.reshape(1, -1),
        ).reshape(-1)
        runtime = {
            "input_projection_t": parameters["input_projection"].T,
            "input_bias": parameters["input_bias"],
            "vocab_output_t": parameters["vocab_output"].T,
            "vocab_bias": parameters["vocab_bias"],
        }
        actual = predict_token_logits_bf16(runtime, hidden, embedding)
        mx.eval(expected, actual)
        self.assertLess(float(mx.max(mx.abs(expected - actual))), 2e-3)
        self.assertEqual(validate_token_runtime_parameters(runtime, True), (16, 32))

    def test_token_runtime_validation_rejects_non_bf16_payload(self):
        parameters = initialize_token_classifier(16, 4, 32, 7)
        with self.assertRaisesRegex(MetadataError, "must be BF16"):
            validate_token_runtime_parameters(parameters, False)

    def test_sequence_split_never_crosses_prompt(self):
        prompts = [0, 0, 0, 0, 1, 1, 1, 1]
        self.assertEqual(sequence_starts(prompts, 2, 0), [0, 1])
        self.assertEqual(sequence_starts(prompts, 2, 1), [4, 5])

    def test_adamw_state_is_two_full_precision_arrays(self):
        parameters = {"weight": mx.zeros((4, 4))}
        optimizer = AdamWControl(parameters, 1e-3, 0.0)
        self.assertEqual(optimizer.state_bytes(), parameters["weight"].nbytes * 2)
        updated = optimizer.update(parameters, {"weight": mx.ones((4, 4))})
        self.assertLess(float(mx.min(updated["weight"])), 0.0)

    def test_acceptance_score_counts_all_accepted_depth_events(self):
        metrics = {"1": {"matches": 10}, "2": {"matches": 7}, "3": {"matches": 4}}
        self.assertEqual(acceptance_score(metrics), 21)

    def test_runtime_binding_rejects_another_target_or_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            head = root / "head"
            model.mkdir()
            head.mkdir()
            model_report = model / "nemotron_mlx_pack_report.json"
            head_report = head / "nemotron_mtp_vocab_head_report.json"
            model_report.write_text("model")
            head_report.write_text("head")
            report = {
                "model_dir": str(model.resolve()),
                "model_report_sha256": sha256_file(model_report),
                "mtp_lm_head": str(head.resolve()),
                "mtp_lm_head_report_sha256": sha256_file(head_report),
            }
            validate_runtime_binding(report, model, head)
            model_report.write_text("changed")
            with self.assertRaisesRegex(MetadataError, "target identity mismatch"):
                validate_runtime_binding(report, model, head)

    def test_validated_report_rejects_changed_predictor_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            head = root / "head"
            artifact_dir = root / "predictor"
            model.mkdir()
            head.mkdir()
            artifact_dir.mkdir()
            model_report = model / "nemotron_mlx_pack_report.json"
            head_report = head / "nemotron_mtp_vocab_head_report.json"
            artifact = artifact_dir / "predictor.safetensors"
            model_report.write_text("model")
            head_report.write_text("head")
            artifact.write_bytes(b"predictor")
            report = {
                "format": "nemotron-mtp-learned-predictor-v1",
                "status": "diagnostic",
                "model_dir": str(model.resolve()),
                "model_report_sha256": sha256_file(model_report),
                "mtp_lm_head": str(head.resolve()),
                "mtp_lm_head_report_sha256": sha256_file(head_report),
                "artifact": artifact.name,
                "inference_payload_bytes": artifact.stat().st_size,
                "artifact_sha256": sha256_file(artifact),
            }
            (artifact_dir / "report.json").write_text(json.dumps(report))
            validated_predictor_report(artifact_dir, model, head)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(MetadataError, "artifact identity mismatch"):
                validated_predictor_report(artifact_dir, model, head)


if __name__ == "__main__":
    unittest.main()

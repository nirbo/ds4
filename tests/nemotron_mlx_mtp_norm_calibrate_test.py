#!/usr/bin/env python3

import sys
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_mtp_norm_calibrate import (  # noqa: E402
    accepted_checkpoint,
    calibration_examples,
    calibration_loss,
    checkpoint_score,
    damped_final_norm,
    parse_depth_weights,
    load_calibrated_final_norm,
)
from nemotron_prune_materialize import sha256_file  # noqa: E402


class MTPNormCalibrateTest(unittest.TestCase):
    def test_examples_are_prompt_disjoint_and_skip_missing_targets(self):
        rows, depths, labels = calibration_examples(
            [0, 0, 0, 1, 1, 1],
            [4, -1, 6, 7, 8, 9],
            3,
            0,
        )
        self.assertEqual(rows, [0, 0])
        self.assertEqual(depths, [0, 2])
        self.assertEqual(labels, [4, 6])

    def test_loss_has_scale_gradient(self):
        parameters = {"scale": mx.ones((4,))}
        hidden = mx.array([[[1.0, 0.0, 0.0, 0.0]]])
        head = mx.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
        value, gradient = mx.value_and_grad(calibration_loss)(
            parameters,
            mx.array([0], dtype=mx.int32),
            mx.array([0], dtype=mx.int32),
            mx.array([1], dtype=mx.int32),
            hidden,
            head,
            mx.array([1.0]),
            0.1,
        )
        mx.eval(value, gradient["scale"])
        self.assertGreater(float(value), 0)
        self.assertGreater(float(mx.max(mx.abs(gradient["scale"]))), 0)

    def test_checkpoint_requires_depth_one_and_prioritizes_later_depths(self):
        baseline = {"1": {"matches": 10}}
        rejected = {"1": {"matches": 9}, "2": {"matches": 8}}
        accepted = {"1": {"matches": 10}, "2": {"matches": 7}}
        self.assertFalse(accepted_checkpoint(rejected, baseline))
        self.assertTrue(accepted_checkpoint(accepted, baseline))
        self.assertGreater(checkpoint_score(rejected), checkpoint_score(accepted))

    def test_depth_weight_parser(self):
        self.assertEqual(parse_depth_weights("2,1,.5"), (2.0, 1.0, 0.5))

    def test_final_norm_damping_preserves_endpoints(self):
        original = mx.array([1.0, 2.0], dtype=mx.bfloat16)
        calibrated = mx.array([3.0, 4.0], dtype=mx.bfloat16)
        self.assertEqual(damped_final_norm(original, calibrated, 0).tolist(), [1.0, 2.0])
        self.assertEqual(damped_final_norm(original, calibrated, 1).tolist(), [3.0, 4.0])

    def test_calibrated_norm_loader_binds_sidecar_and_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            sidecar = root / "sidecar"
            head = root / "head"
            artifact_dir = root / "artifact"
            model.mkdir()
            sidecar.mkdir()
            head.mkdir()
            artifact_dir.mkdir()
            model_report = model / "nemotron_mlx_pack_report.json"
            sidecar_report = sidecar / "nemotron_mtp_pack_report.json"
            head_report = head / "nemotron_mtp_vocab_head_report.json"
            model_report.write_text("model")
            sidecar_report.write_text("sidecar")
            head_report.write_text("head")
            artifact = artifact_dir / "final_norm.safetensors"
            mx.save_safetensors(
                str(artifact),
                {"mtp.layers.1.final_layernorm.weight": mx.ones((4,), dtype=mx.bfloat16)},
                metadata={"format": "nemotron-mtp-final-norm-calibration-v1"},
            )
            report = {
                "format": "nemotron-mtp-final-norm-calibration-v1",
                "status": "diagnostic",
                "model_dir": str(model.resolve()),
                "model_report_sha256": sha256_file(model_report),
                "sidecar": str(sidecar.resolve()),
                "sidecar_report_sha256": sha256_file(sidecar_report),
                "mtp_lm_head_report_sha256": sha256_file(head_report),
                "artifact": artifact.name,
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": sha256_file(artifact),
            }
            (artifact_dir / "report.json").write_text(json.dumps(report))
            self.assertEqual(
                load_calibrated_final_norm(artifact_dir, model, sidecar, head).shape,
                (4,),
            )
            sidecar_report.write_text("changed")
            with self.assertRaisesRegex(MetadataError, "sidecar identity mismatch"):
                load_calibrated_final_norm(artifact_dir, model, sidecar, head)



if __name__ == "__main__":
    unittest.main()

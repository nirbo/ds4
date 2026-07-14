#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_mtp_distill import (  # noqa: E402
    distillation_loss,
    official_acceptance_metrics,
    reduced_target_indices,
    selected_logit_kl,
    token_classifier_loss,
    token_classifier_metrics,
)
from nemotron_mlx_mtp_predictor import (  # noqa: E402
    GATED_ARCHITECTURE,
    initialize_gated_parameters,
    initialize_token_classifier,
)


class MTPDistillTest(unittest.TestCase):
    def test_selected_logit_kl_is_zero_for_identical_projection(self):
        prediction = mx.array([[1.0, 0.0]])
        head = mx.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        indices = mx.array([[0, 1, 2]], dtype=mx.int32)
        teacher = mx.array([[1.0, 0.0, -1.0]])
        self.assertAlmostEqual(
            float(selected_logit_kl(prediction, indices, teacher, head, 1.0)[0]),
            0.0,
            places=6,
        )

    def test_official_acceptance_stops_at_first_rejection(self):
        metrics = official_acceptance_metrics(
            [0, 3],
            [1, 2, 3, 4, 5, 6],
            [[1, 9, 3], [0, 0, 0], [0, 0, 0], [4, 5, 6], [0, 0, 0], [0, 0, 0]],
            3,
        )
        self.assertEqual(
            [(metrics[str(depth)]["attempts"], metrics[str(depth)]["matches"]) for depth in (1, 2, 3)],
            [(2, 2), (2, 1), (1, 1)],
        )

    def test_distillation_loss_has_student_gradients(self):
        parameters = initialize_gated_parameters(4, 2, 1, 7)
        features = {
            "teacher_hidden": mx.ones((3, 2, 4)),
            "teacher_token_ids": mx.ones((3, 2), dtype=mx.int32),
            "teacher_top_indices": mx.zeros((3, 2, 2), dtype=mx.int32),
            "teacher_top_logits": mx.zeros((3, 2, 2)),
        }
        loss_and_grad = mx.value_and_grad(distillation_loss)
        loss, gradients = loss_and_grad(
            parameters,
            mx.array([0], dtype=mx.int32),
            features,
            mx.ones((3, 4)),
            mx.ones((3,), dtype=mx.int32),
            mx.zeros((3,), dtype=mx.int32),
            mx.ones((2, 4)),
            GATED_ARCHITECTURE,
            1,
            1.0,
            0.1,
            0.0,
            1.0,
        )
        mx.eval(loss, *gradients.values())
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(any(float(mx.max(mx.abs(value))) > 0 for value in gradients.values()))

    def test_token_classifier_loss_and_metrics(self):
        parameters = initialize_token_classifier(4, 2, 3, 7)
        features = {
            "teacher_hidden": mx.ones((3, 2, 4)),
            "teacher_token_ids": mx.array([[1, 2], [1, 2], [1, 2]], dtype=mx.int32),
            "teacher_top_indices": mx.array(
                [[[1, 0], [2, 1]], [[1, 0], [2, 1]], [[1, 0], [2, 1]]],
                dtype=mx.int32,
            ),
            "teacher_top_logits": mx.ones((3, 2, 2)),
        }
        loss = token_classifier_loss(
            parameters,
            mx.array([0], dtype=mx.int32),
            features,
            mx.ones((3, 4)),
            mx.array([0, 2, -1], dtype=mx.int32),
            0.1,
            1.0,
            1.0,
        )
        mx.eval(loss)
        self.assertGreater(float(loss), 0.0)
        metrics = token_classifier_metrics(
            parameters,
            [0],
            features,
            mx.ones((3, 4)),
            mx.array([1, 2, 0], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
            1,
        )
        self.assertEqual(metrics["1"]["matches"], 1)
        self.assertEqual(metrics["2"]["attempts"], 1)

    def test_reduced_target_indices_marks_tokens_outside_map(self):
        indices = reduced_target_indices([10, 30, 20], [20, 99, 10])
        self.assertEqual(indices.tolist(), [2, -1, 0])


if __name__ == "__main__":
    unittest.main()

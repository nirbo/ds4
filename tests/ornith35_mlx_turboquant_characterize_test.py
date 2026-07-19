#!/usr/bin/env python3
"""Focused contracts for bounded Ornith-35 TurboQuant characterization."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ornith35" / "tools"))

import ornith35_mlx_turboquant_characterize as characterize


class MLXTurboQuantCharacterizeTest(unittest.TestCase):
    def test_profiles_have_exact_physical_cache_accounting(self) -> None:
        reports = {
            profile.name: characterize.profile_storage(profile)
            for profile in characterize.PROFILES
        }
        self.assertEqual(reports["k4-qjl-v3-bf16norm"]["key_bytes_per_vector"], 132)
        self.assertEqual(reports["k4-qjl-v3-bf16norm"]["value_bytes_per_vector"], 98)
        self.assertEqual(reports["split35-qjl-bf16norm"]["native_cache_bytes"], 1_237_319_680)
        self.assertEqual(reports["split35-mse-bf16norm"]["native_cache_bytes"], 1_216_348_160)
        self.assertEqual(reports["bf16-control"]["native_cache_bytes"], 5 * 2**30)
        self.assertEqual(reports["bf16-k-v4-mse"]["native_cache_bytes"], 3_365_928_960)
        self.assertEqual(reports["k5-mse-v4-bf16norm"]["native_cache_bytes"], 1_530_920_960)
        self.assertEqual(reports["k5-mse-v5-bf16norm"]["native_cache_bytes"], 1_698_693_120)
        self.assertEqual(reports["k6-mse-v5-bf16norm"]["native_cache_bytes"], 1_866_465_280)
        self.assertEqual(reports["k6-mse-v6-bf16norm"]["native_cache_bytes"], 2_034_237_440)
        self.assertEqual(reports["k7-mse-v7-bf16norm"]["native_cache_bytes"], 2_369_781_760)
        self.assertEqual(reports["k8-mse-v8-bf16norm"]["native_cache_bytes"], 2_705_326_080)
        self.assertAlmostEqual(
            reports["k7-mse-v7-bf16norm"]["native_compression_ratio"],
            2.265486725663717,
        )
        self.assertAlmostEqual(
            reports["k3-qjl-v3-bf16norm"]["native_cache_gib"],
            0.966796875,
        )

    def test_channel_ranking_is_descending_and_tie_stable(self) -> None:
        values = [1.0] * 256
        values[200] = 4.0
        values[17] = 3.0
        selected = characterize.top_channels(values, 3)
        self.assertEqual(selected, (0, 17, 200))

    def test_metric_summary_uses_conservative_nearest_rank(self) -> None:
        summary = characterize.summarize([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["p50"], 3.0)
        self.assertEqual(summary["p95"], 4.0)
        self.assertEqual(summary["mean"], 2.5)

    def test_query_sampling_covers_prefix_and_final_token(self) -> None:
        self.assertEqual(
            characterize.sampled_query_positions(32),
            (3, 7, 11, 15, 19, 23, 27, 31),
        )
        self.assertEqual(characterize.sampled_query_positions(3), (0, 1, 2))

    def test_identity_attention_and_vector_metrics_are_exact(self) -> None:
        scores = mx.array([0.25, -0.5, 1.25, 0.75], dtype=mx.float32)
        probabilities = mx.softmax(scores)
        attention = characterize.attention_metrics(
            scores,
            scores,
            probabilities,
            probabilities,
        )
        vector = mx.array([math.sin(index * 0.3) for index in range(16)])
        output = characterize.vector_metrics(vector, vector)

        self.assertEqual(attention["score_rmse"], 0.0)
        self.assertEqual(attention["probability_l1"], 0.0)
        self.assertEqual(attention["top1_agreement"], 1.0)
        self.assertEqual(attention["top8_recall"], 1.0)
        self.assertEqual(output["relative_l2"], 0.0)
        self.assertAlmostEqual(output["cosine"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

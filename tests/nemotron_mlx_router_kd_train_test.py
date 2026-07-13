#!/usr/bin/env python3
"""Tests for multi-sample Router KD selection and held-out acceptance."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_router_kd_train import (  # noqa: E402
    acceptance_gate,
    expand_prefixes,
    parse_prefix_lengths,
    save_router_artifact,
    select_samples,
    validate_disjoint,
)


class FakeTokenizer:
    def encode(self, sample, add_special_tokens=False):
        self.assertFalse(add_special_tokens)
        return [ord(character) for character in sample]

    def assertFalse(self, value):
        if value:
            raise AssertionError("special tokens unexpectedly enabled")


def row(category: str, kl: float, teacher_top: int = 1, candidate_top: int = 1) -> dict:
    return {
        "category": category,
        "metrics": {
            "kl_baseline_candidate": kl,
            "baseline_top1": teacher_top,
            "candidate_top1": candidate_top,
        },
    }


class RouterKDTrainTest(unittest.TestCase):
    def test_prefix_parser_and_expansion(self) -> None:
        lengths = parse_prefix_lengths("2,4,full")
        self.assertEqual(lengths, [2, 4, None])
        samples = [{"category": "code", "sample_sha256": "a", "token_ids": [1, 2, 3, 4, 5]}]
        expanded = expand_prefixes(samples, lengths)
        self.assertEqual(
            [row["token_ids"] for row in expanded],
            [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        )

    def test_artifact_records_multisample_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "router.safetensors"
            save_router_artifact(
                path,
                {"1": mx.ones((2, 3), dtype=mx.bfloat16)},
                {"source_revision": "revision"},
            )
            tensors, metadata = mx.load(str(path), return_metadata=True)
            self.assertEqual(metadata["format"], "nemotron-multisample-router-kd-v1")
            self.assertEqual(metadata["source_revision"], "revision")
            self.assertEqual(set(tensors), {"layer_001.gate.weight"})

    def test_selects_filtered_samples_and_caps_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "corpus.json"
            corpus.write_text(json.dumps({"a": ["abcd"], "b": ["wxyz"]}))
            samples = select_samples(corpus, FakeTokenizer(), ["b"], 1, 3)
            self.assertEqual(samples[0]["category"], "b")
            self.assertEqual(samples[0]["token_ids"], [ord("w"), ord("x"), ord("y")])

    def test_disjoint_validation_rejects_matching_prefix(self) -> None:
        left = [{"sample_sha256": "a", "token_ids": [1, 2]}]
        right = [{"sample_sha256": "b", "token_ids": [1, 2]}]
        with self.assertRaises(MetadataError):
            validate_disjoint(left, right)

    def test_acceptance_requires_mean_worst_and_top1(self) -> None:
        baseline = [row("a", 0.4), row("b", 0.2)]
        accepted = acceptance_gate(baseline, [row("a", 0.3), row("b", 0.1)])
        self.assertTrue(accepted["accepted"])
        worst_regression = acceptance_gate(baseline, [row("a", 0.41), row("b", 0.01)])
        self.assertFalse(worst_regression["accepted"])
        top1_regression = acceptance_gate(
            baseline, [row("a", 0.3, candidate_top=2), row("b", 0.1)]
        )
        self.assertFalse(top1_regression["accepted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

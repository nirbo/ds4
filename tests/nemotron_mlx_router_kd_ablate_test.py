#!/usr/bin/env python3
"""Focused tests for Router KD layer-group ablation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_router_kd_ablate import gate, parse_indices, partition_layers


def metrics(kl: float, candidate: int = 1) -> dict:
    return {"kl_baseline_candidate": kl, "baseline_top1": 1, "candidate_top1": candidate}


class RouterKdAblateTest(unittest.TestCase):
    def test_partitions_layers_without_loss(self) -> None:
        groups = partition_layers([1, 3, 5, 7, 9], 2)
        self.assertEqual(groups, [[1, 3, 5], [7, 9]])

    def test_indices_are_sorted_and_unique(self) -> None:
        self.assertEqual(parse_indices("0,2,6"), [0, 2, 6])

    def test_gate_requires_mean_worst_and_top1(self) -> None:
        cases = [
            {"base": metrics(0.4), "candidate": metrics(0.3)},
            {"base": metrics(0.2), "candidate": metrics(0.1)},
        ]
        self.assertTrue(gate(cases, cases, "candidate")["accepted"])
        cases[0]["candidate"] = metrics(0.41)
        self.assertFalse(gate(cases, cases, "candidate")["accepted"])


if __name__ == "__main__":
    unittest.main()

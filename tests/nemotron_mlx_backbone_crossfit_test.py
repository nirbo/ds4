#!/usr/bin/env python3
"""Tests for train-only prompt-stratified low-bit allocation selection."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_crossfit import (  # noqa: E402
    aggregate_policy_losses,
    crossfit_policy_selection,
    selection_budget_indices,
    stratified_prompt_folds,
)


def brute_planner(
    losses: np.ndarray,
    payloads: np.ndarray,
    budgets: list[int],
    *,
    cost_quantum: int,
    labels: tuple[str, ...],
) -> list[dict]:
    del cost_quantum
    experts, tiers = losses.shape
    plans = []
    for budget in budgets:
        candidates = []
        for assignment in itertools.product(range(tiers), repeat=experts):
            payload = sum(int(payloads[tier]) for tier in assignment)
            if payload <= budget:
                error = sum(losses[expert, tier] for expert, tier in enumerate(assignment))
                candidates.append((error, assignment, payload))
        error, assignment, payload = min(candidates)
        plans.append(
            {
                "assignment_indices": list(assignment),
                "layer_payload_bytes": payload,
                "selection_error2": float(error),
                "tier_counts": {
                    label: assignment.count(tier) for tier, label in enumerate(labels)
                },
            }
        )
    return plans


class BackboneCrossfitTest(unittest.TestCase):
    def test_category_stratified_folds_are_deterministic_and_balanced(self) -> None:
        hashes = [f"{index:064x}" for index in range(20)]
        categories = ["code"] * 10 + ["reasoning"] * 10
        first = stratified_prompt_folds(hashes, categories, 5)
        second = stratified_prompt_folds(hashes, categories, 5)
        np.testing.assert_array_equal(first, second)
        for category in ("code", "reasoning"):
            counts = np.bincount(
                first[np.asarray(categories) == category],
                minlength=5,
            )
            self.assertEqual(counts.tolist(), [2, 2, 2, 2, 2])

    def test_policy_aggregation_has_explicit_prompt_and_category_weighting(self) -> None:
        losses = np.array([[[2.0]], [[18.0]], [[8.0]]], dtype=np.float64)
        rows = np.array([1, 9, 2], dtype=np.int64)
        categories = ["a", "a", "b"]
        reference = np.array([4.0, 36.0, 8.0], dtype=np.float64)
        selected = np.ones(3, dtype=bool)
        route_total = aggregate_policy_losses(
            losses, rows, categories, reference, selected, "route-total"
        )
        prompt_equal = aggregate_policy_losses(
            losses, rows, categories, reference, selected, "prompt-equal"
        )
        category_equal = aggregate_policy_losses(
            losses, rows, categories, reference, selected, "category-equal"
        )
        self.assertAlmostEqual(float(route_total[0, 0]), 28.0)
        self.assertAlmostEqual(float(prompt_equal[0, 0]), 32.0)
        self.assertAlmostEqual(float(category_equal[0, 0]), 36.0)

    def test_one_standard_error_rule_keeps_equal_route_total_control(self) -> None:
        prompts = 10
        losses = np.zeros((prompts, 2, 2), dtype=np.float64)
        losses[:, 0, 0] = 10.0
        losses[:, 1, 0] = 1.0
        result = crossfit_policy_selection(
            losses,
            np.arange(1, prompts + 1, dtype=np.int64),
            ["code"] * 5 + ["reasoning"] * 5,
            [f"{index:064x}" for index in range(prompts)],
            np.arange(1, prompts + 1, dtype=np.float64),
            np.array([1, 2], dtype=np.int64),
            [3],
            fold_count=5,
            cost_quantum=1,
            labels=("low", "native"),
            planner=brute_planner,
        )
        self.assertEqual(result["best_mean_policy"], "route-total")
        self.assertEqual(result["selected_policy"], "route-total")
        self.assertEqual(result["policies"]["route-total"]["paired_observations"], 5)

    def test_selection_budget_anchors_span_dense_frontier(self) -> None:
        self.assertEqual(selection_budget_indices(65), [0, 16, 32, 48, 64])
        self.assertEqual(selection_budget_indices(3), [0, 1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Train-only prompt-stratified allocation policy selection for low-bit experts."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Callable

import numpy as np

from nemotron_metadata import require


POLICIES = (
    "route-total",
    "prompt-equal",
    "category-equal",
    "prompt-category-equal",
    "prompt-relative",
    "prompt-category-relative",
)


def selection_budget_indices(budget_count: int, maximum: int = 5) -> list[int]:
    """Choose deterministic, evenly spaced policy-selection anchors."""

    require(budget_count > 0 and maximum > 0, "cross-fit budget catalog is empty")
    count = min(budget_count, maximum)
    if count == 1:
        return [0]
    return sorted(
        {
            round(index * (budget_count - 1) / (count - 1))
            for index in range(count)
        }
    )


def stratified_prompt_folds(
    sample_sha256: list[str],
    categories: list[str],
    fold_count: int,
) -> np.ndarray:
    """Assign complete prompts to deterministic, category-balanced folds."""

    require(len(sample_sha256) == len(categories) > 0, "cross-fit prompt provenance is empty")
    require(2 <= fold_count <= len(sample_sha256), "invalid cross-fit fold count")
    require(len(set(sample_sha256)) == len(sample_sha256), "cross-fit prompt hashes are not unique")
    assignments = np.full(len(sample_sha256), -1, dtype=np.int32)
    for category in sorted(set(categories)):
        prompt_indices = [
            index for index, observed in enumerate(categories) if observed == category
        ]
        require(len(prompt_indices) >= fold_count, f"category {category} has fewer prompts than folds")
        prompt_indices.sort(key=lambda index: sample_sha256[index])
        offset = int.from_bytes(
            hashlib.sha256(category.encode("utf-8")).digest()[:4], "little"
        ) % fold_count
        for position, prompt_index in enumerate(prompt_indices):
            assignments[prompt_index] = (position + offset) % fold_count
    require(np.all(assignments >= 0), "cross-fit prompt assignment is incomplete")
    return assignments


def policy_prompt_weights(
    prompt_rows: np.ndarray,
    categories: list[str],
    prompt_reference_energy2: np.ndarray,
    selected: np.ndarray,
    policy: str,
) -> np.ndarray:
    """Return policy weights whose overall scale remains comparable to raw loss."""

    require(policy in POLICIES, f"unsupported cross-fit policy: {policy}")
    prompt_rows = np.asarray(prompt_rows, dtype=np.float64)
    reference = np.asarray(prompt_reference_energy2, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    prompts = prompt_rows.size
    require(
        len(categories) == prompts and reference.shape == (prompts,) and selected.shape == (prompts,),
        "cross-fit policy input shape mismatch",
    )
    require(np.all(prompt_rows > 0.0), "cross-fit prompt row count is not positive")
    require(np.all(np.isfinite(reference)) and np.all(reference > 0.0), "cross-fit reference energy is invalid")
    require(np.any(selected), "cross-fit policy selected no prompts")

    active = np.nonzero(selected)[0]
    weights = np.zeros(prompts, dtype=np.float64)
    total_rows = float(np.sum(prompt_rows[active]))
    total_reference = float(np.sum(reference[active]))
    active_categories = sorted({categories[index] for index in active})
    require(active_categories, "cross-fit policy has no active categories")

    if policy == "route-total":
        weights[active] = 1.0
    elif policy == "prompt-equal":
        weights[active] = total_rows / (active.size * prompt_rows[active])
    elif policy == "category-equal":
        for category in active_categories:
            indices = np.asarray(
                [index for index in active if categories[index] == category],
                dtype=np.int64,
            )
            category_rows = float(np.sum(prompt_rows[indices]))
            weights[indices] = total_rows / (len(active_categories) * category_rows)
    elif policy == "prompt-category-equal":
        for category in active_categories:
            indices = np.asarray(
                [index for index in active if categories[index] == category],
                dtype=np.int64,
            )
            weights[indices] = total_rows / (
                len(active_categories) * indices.size * prompt_rows[indices]
            )
    elif policy == "prompt-relative":
        weights[active] = total_reference / (active.size * reference[active])
    else:
        require(policy == "prompt-category-relative", "invalid relative policy")
        for category in active_categories:
            indices = np.asarray(
                [index for index in active if categories[index] == category],
                dtype=np.int64,
            )
            weights[indices] = total_reference / (
                len(active_categories) * indices.size * reference[indices]
            )
    require(np.all(np.isfinite(weights)) and np.all(weights >= 0.0), "cross-fit policy weights are invalid")
    return weights


def aggregate_policy_losses(
    prompt_losses: np.ndarray,
    prompt_rows: np.ndarray,
    categories: list[str],
    prompt_reference_energy2: np.ndarray,
    selected: np.ndarray,
    policy: str,
) -> np.ndarray:
    """Aggregate prompt-local expert-option errors without crossing a split."""

    losses = np.asarray(prompt_losses, dtype=np.float64)
    require(losses.ndim == 3 and losses.shape[0] > 0, "cross-fit prompt loss tensor is invalid")
    require(np.all(np.isfinite(losses)) and np.all(losses >= 0.0), "cross-fit prompt loss is invalid")
    weights = policy_prompt_weights(
        prompt_rows,
        categories,
        prompt_reference_energy2,
        selected,
        policy,
    )
    return np.tensordot(weights, losses, axes=(0, 0))


def crossfit_policy_selection(
    prompt_losses: np.ndarray,
    prompt_rows: np.ndarray,
    categories: list[str],
    sample_sha256: list[str],
    prompt_reference_energy2: np.ndarray,
    payload_bytes: np.ndarray,
    budgets: list[int],
    *,
    fold_count: int,
    cost_quantum: int,
    labels: tuple[str, ...],
    planner: Callable[..., list[dict]],
) -> dict:
    """Select one allocation weighting policy with train-only cross-validation."""

    losses = np.asarray(prompt_losses, dtype=np.float64)
    require(losses.ndim == 3, "cross-fit loss catalog must be prompt/expert/option")
    prompts, experts, options = losses.shape
    require(prompts == len(sample_sha256) == len(categories), "cross-fit prompt count mismatch")
    require(payload_bytes.shape == (options,) and len(labels) == options, "cross-fit option catalog mismatch")
    require(budgets and budgets == sorted(set(budgets)), "cross-fit budgets are invalid")
    folds = stratified_prompt_folds(sample_sha256, categories, fold_count)
    all_selected = np.ones(prompts, dtype=bool)
    raw_total_losses = aggregate_policy_losses(
        losses,
        prompt_rows,
        categories,
        prompt_reference_energy2,
        all_selected,
        "route-total",
    )

    results: dict[str, dict] = {}
    raw_errors: dict[str, np.ndarray] = {}
    for policy in POLICIES:
        policy_errors = np.zeros((fold_count, len(budgets)), dtype=np.float64)
        fold_reports = []
        for fold in range(fold_count):
            heldout = folds == fold
            training = ~heldout
            planning_losses = aggregate_policy_losses(
                losses,
                prompt_rows,
                categories,
                prompt_reference_energy2,
                training,
                policy,
            )
            heldout_losses = aggregate_policy_losses(
                losses,
                prompt_rows,
                categories,
                prompt_reference_energy2,
                heldout,
                "route-total",
            )
            plans = planner(
                planning_losses,
                payload_bytes,
                budgets,
                cost_quantum=cost_quantum,
                labels=labels,
            )
            budget_reports = []
            for budget_index, plan in enumerate(plans):
                assignment = np.asarray(plan["assignment_indices"], dtype=np.int64)
                require(assignment.shape == (experts,), "cross-fit plan assignment shape mismatch")
                error2 = float(np.sum(heldout_losses[np.arange(experts), assignment]))
                policy_errors[fold, budget_index] = error2
                budget_reports.append(
                    {
                        "budget_bytes": budgets[budget_index],
                        "heldout_route_error2": error2,
                        "tier_counts": plan["tier_counts"],
                    }
                )
            fold_reports.append(
                {
                    "fold": fold,
                    "training_prompts": int(np.sum(training)),
                    "heldout_prompts": int(np.sum(heldout)),
                    "heldout_rows": int(np.sum(np.asarray(prompt_rows)[heldout])),
                    "heldout_categories": dict(Counter(categories[index] for index in np.nonzero(heldout)[0])),
                    "budgets": budget_reports,
                }
            )
        raw_errors[policy] = policy_errors
        results[policy] = {"folds": fold_reports}

    control = raw_errors["route-total"]
    informative = control > np.finfo(np.float64).tiny
    require(np.any(informative), "cross-fit budgets contain no nonzero heldout error")
    for policy in POLICIES:
        ratios = np.divide(
            raw_errors[policy],
            control,
            out=np.ones_like(control),
            where=informative,
        )
        values = ratios[informative]
        mean = float(np.mean(values))
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(values.size))
            if values.size > 1
            else 0.0
        )
        results[policy].update(
            {
                "mean_relative_to_route_total": mean,
                "standard_error": standard_error,
                "worst_relative_to_route_total": float(np.max(values)),
                "paired_observations": int(values.size),
            }
        )
        for fold_index, fold_report in enumerate(results[policy]["folds"]):
            for budget_index, budget_report in enumerate(fold_report["budgets"]):
                budget_report["relative_to_route_total"] = float(ratios[fold_index, budget_index])

    best = min(POLICIES, key=lambda policy: (results[policy]["mean_relative_to_route_total"], POLICIES.index(policy)))
    best_upper = results[best]["mean_relative_to_route_total"] + results[best]["standard_error"]
    if best != "route-total" and best_upper < 1.0:
        selected_policy = best
        reason = "lowest mean heldout error and its one-standard-error bound beats route-total"
    else:
        selected_policy = "route-total"
        reason = "one-standard-error rule retains the simpler route-total control"

    final_losses = {
        policy: aggregate_policy_losses(
            losses,
            prompt_rows,
            categories,
            prompt_reference_energy2,
            all_selected,
            policy,
        )
        for policy in POLICIES
    }
    return {
        "fold_count": fold_count,
        "fold_strategy": "category-stratified-prompt-sha256-round-robin-v1",
        "policies": results,
        "best_mean_policy": best,
        "selected_policy": selected_policy,
        "selection_reason": reason,
        "selection_rule": "one-standard-error-versus-route-total-v1",
        "final_losses": final_losses,
        "raw_total_losses": raw_total_losses,
        "fold_assignments": folds,
    }

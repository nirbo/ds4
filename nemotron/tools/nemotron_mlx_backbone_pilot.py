#!/usr/bin/env python3
"""Summarize a representative BF16-to-low-bit backbone fitting pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_backbone_fit import STATE_FORMAT as FIT_STATE_FORMAT
from nemotron_mlx_backbone_fit import validate_expert_artifact
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-backbone-lowbit-pilot-report-v1"


def finite_metric(row: dict, name: str) -> float:
    value = float(row[name])
    require(math.isfinite(value) and value >= 0.0, f"invalid pilot metric: {name}")
    return value


def summarize(values: list[float]) -> dict[str, float]:
    require(values, "cannot summarize an empty metric")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def build_report(fit_dir: Path) -> dict:
    state_path = fit_dir / "state.json"
    state = load_json(state_path)
    require(state.get("format") == FIT_STATE_FORMAT, "unsupported backbone fit state")
    require(state.get("status") == "complete", "representative backbone fit is incomplete")
    require(not state.get("skipped"), "representative backbone fit skipped experts")
    selected = state.get("experts")
    fit_strategy = state.get("fit_strategy", "bf16-endpoint")
    completed = state.get("completed")
    require(isinstance(selected, list) and selected, "pilot has no selected experts")
    require(
        isinstance(completed, list)
        and sorted(row.get("expert") for row in completed) == selected,
        "pilot completion does not match selected experts",
    )
    identity = {
        "source_revision": state["source_revision"],
        "layer": state["layer"],
        "architecture": state["architecture"],
        "validation_context_rows": state["validation_context_rows"],
        "group_size": state["group_size"],
        "fit_strategy": state.get("fit_strategy", "bf16-endpoint"),
        "contract_sha256": state["contract_sha256"],
        "context_state_sha256": state["context_state_sha256"],
    }
    initial_relative_l2 = []
    fitted_relative_l2 = []
    precision_relative_l2 = {str(bits): [] for bits in (2, 3, 4)}
    experts = []
    deployable = []
    refinement_improved = []
    refinement_code_flips = 0
    for row in sorted(completed, key=lambda value: value["expert"]):
        validate_expert_artifact(fit_dir / "experts" / row["file"], row, identity)
        validation = row.get("metrics", {}).get("validation", {})
        initial = finite_metric(validation, "initial_relative_l2")
        fitted = finite_metric(validation, "fitted_relative_l2")
        tiers = {}
        for bits in (2, 3, 4):
            tier = row.get("precision_tiers", {}).get(str(bits), {})
            relative = finite_metric(tier, "relative_l2")
            precision_relative_l2[str(bits)].append(relative)
            tiers[str(bits)] = relative
        improved = fitted < initial
        plan_eligible = row.get("candidate_plan_eligible", improved)
        require(isinstance(plan_eligible, bool), "invalid pilot planning eligibility")
        if plan_eligible:
            deployable.append(row["expert"])
        refinement = row.get("refinement")
        if isinstance(refinement, dict):
            if refinement.get("improved"):
                refinement_improved.append(row["expert"])
            refinement_code_flips += int(refinement.get("best_code_flips", 0))
        initial_relative_l2.append(initial)
        fitted_relative_l2.append(fitted)
        experts.append(
            {
                "expert": row["expert"],
                "train_routes": row["train_routes"],
                "validation_routes": row["validation_routes"],
                "initial_binary_relative_l2": initial,
                "fitted_binary_relative_l2": fitted,
                "relative_improvement": (initial - fitted) / max(initial, 1e-30),
                "fitted_improved_heldout": improved,
                "candidate_plan_eligible": plan_eligible,
                "precision_relative_l2": tiers,
                "elapsed_seconds": row["elapsed_seconds"],
                "refinement": refinement,
            }
        )
    improved_fraction = len(deployable) / len(selected)
    median_improved = statistics.median(fitted_relative_l2) < statistics.median(initial_relative_l2)
    mechanism_gate = improved_fraction >= 0.25 and median_improved
    if fit_strategy == "native-target-rtn":
        gate = {
            "result": "not-applicable",
            "criteria": "target-derived-rtn-is-not-a-bf16-fitting-mechanism",
            "next": "fit-and-causally-plan-full-layer",
        }
    else:
        gate = {
            "result": "passed" if mechanism_gate else "failed",
            "criteria": "at-least-25pct-heldout-improvements-and-median-improvement",
            "next": (
                "fit-and-causally-plan-full-layer"
                if mechanism_gate
                else "revise-fitting-before-more-bf16"
            ),
        }
    return {
        "format": FORMAT,
        "status": "complete",
        "scope": "representative-experts-only",
        "source_repository": state["source_repository"],
        "source_revision": state["source_revision"],
        "native_qat_proxy_revision": state["proxy_source_revision"],
        "fit_strategy": fit_strategy,
        "layer": state["layer"],
        "fit_state": str(state_path.resolve()),
        "fit_state_sha256": sha256_file(state_path),
        "tool_sha256": sha256_file(Path(__file__)),
        "selected_experts": selected,
        "deployable_binary_experts": deployable,
        "refinement_improved_experts": refinement_improved,
        "refinement_best_code_flips": refinement_code_flips,
        "heldout_improved_fraction": improved_fraction,
        "initial_binary_relative_l2": summarize(initial_relative_l2),
        "fitted_binary_relative_l2": summarize(fitted_relative_l2),
        "precision_relative_l2": {
            bits: summarize(values) for bits, values in precision_relative_l2.items()
        },
        "mechanism_gate": gate,
        "quality_status": "not-accepted-full-model-evidence",
        "experts": experts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args.fit_dir)
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"nemotron backbone pilot error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

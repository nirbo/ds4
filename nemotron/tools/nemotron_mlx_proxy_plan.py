#!/usr/bin/env python3
"""Build original-router expert-to-prototype maps from behavioral observations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import sha256_file


FORMAT = "nemotron-proxy-plan-v1"
CALIBRATION_FORMAT = "nemotron-proxy-calibration-v1"


def build_layer_mapping(
    retained: list[int],
    pair_counts: np.ndarray,
    cosine_sums: np.ndarray,
    category_counts: np.ndarray,
    minimum_support: int,
) -> tuple[list[int], dict]:
    experts = pair_counts.shape[0]
    require(
        pair_counts.shape == cosine_sums.shape == (experts, experts),
        "proxy pair matrices do not match",
    )
    require(category_counts.shape[1] == experts, "proxy category matrix does not match")
    require(retained == sorted(set(retained)) and retained, "invalid retained expert set")
    retained_set = set(retained)
    mapping = list(range(experts))
    selected_support = []
    selected_cosine = []
    selected_category = []
    unsupported = 0
    retained_array = np.asarray(retained, dtype=np.int64)
    retained_categories = category_counts[:, retained_array].astype(np.float64).T
    retained_category_norms = np.linalg.norm(retained_categories, axis=1)
    for expert in range(experts):
        if expert in retained_set:
            continue
        supports = pair_counts[expert, retained_array]
        means = np.divide(
            cosine_sums[expert, retained_array],
            supports,
            out=np.full(len(retained), -np.inf, dtype=np.float64),
            where=supports > 0,
        )
        expert_category = category_counts[:, expert].astype(np.float64)
        expert_norm = np.linalg.norm(expert_category)
        category_similarity = np.divide(
            retained_categories @ expert_category,
            retained_category_norms * expert_norm,
            out=np.zeros(len(retained), dtype=np.float64),
            where=(retained_category_norms * expert_norm) > 0,
        )
        supported = supports >= minimum_support
        if not np.any(supported):
            supported = supports > 0
        if np.any(supported):
            score = means + 0.05 * category_similarity + 0.005 * np.log1p(supports)
            score[~supported] = -np.inf
            position = int(np.argmax(score))
        else:
            unsupported += 1
            position = int(np.argmax(category_similarity + 1e-6 * retained_category_norms))
        prototype = retained[position]
        mapping[expert] = prototype
        selected_support.append(int(supports[position]))
        selected_cosine.append(float(means[position]) if math.isfinite(means[position]) else 0.0)
        selected_category.append(float(category_similarity[position]))
    return mapping, {
        "removed": experts - len(retained),
        "unsupported": unsupported,
        "support_min": min(selected_support, default=0),
        "support_median": float(np.median(selected_support)) if selected_support else 0.0,
        "support_max": max(selected_support, default=0),
        "cosine_min": min(selected_cosine, default=0.0),
        "cosine_median": float(np.median(selected_cosine)) if selected_cosine else 0.0,
        "cosine_max": max(selected_cosine, default=0.0),
        "category_cosine_median": (
            float(np.median(selected_category)) if selected_category else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-state", required=True, type=Path)
    parser.add_argument("--calibration-arrays", required=True, type=Path)
    parser.add_argument("--prune-plan", required=True, type=Path)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.minimum_support > 0, "minimum support must be positive")
        state = load_json(args.calibration_state)
        require(
            state.get("format") == CALIBRATION_FORMAT and state.get("status") == "complete",
            "proxy calibration is incomplete",
        )
        require(
            state.get("arrays_sha256") == sha256_file(args.calibration_arrays),
            "proxy calibration arrays hash mismatch",
        )
        prune_plan = load_json(args.prune_plan)
        require(
            prune_plan.get("format") == "nemotron-prune-plan-v1"
            and prune_plan.get("source_revision") == state.get("source_revision"),
            "proxy calibration/prune plan identity mismatch",
        )
        mappings = {}
        summaries = {}
        with np.load(args.calibration_arrays) as arrays:
            for layer in state["layers"]:
                retained = prune_plan["kept_by_layer"].get(str(layer))
                require(isinstance(retained, list), f"prune plan has no layer {layer}")
                prefix = f"layer_{layer:03d}"
                mapping, summary = build_layer_mapping(
                    retained,
                    arrays[f"{prefix}_pair_counts"],
                    arrays[f"{prefix}_cosine_sums"],
                    arrays[f"{prefix}_category_counts"],
                    args.minimum_support,
                )
                mappings[str(layer)] = mapping
                summaries[str(layer)] = summary
        report = {
            "format": FORMAT,
            "source_revision": state["source_revision"],
            "calibration_state_sha256": sha256_file(args.calibration_state),
            "calibration_arrays_sha256": sha256_file(args.calibration_arrays),
            "prune_plan_sha256": sha256_file(args.prune_plan),
            "old_num_experts": prune_plan["old_num_experts"],
            "physical_experts": prune_plan["new_num_experts"],
            "minimum_support": args.minimum_support,
            "prototype_by_original_layer": mappings,
            "summaries": summaries,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".part")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
        print(f"proxy-plan path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        print(f"nemotron proxy plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a provenance-bound expert/width hybrid plan from heldout evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import sha256_file
from nemotron_mlx_stream_forward import validate_virtual_hybrid_plan, validate_virtual_plan
from nemotron_prune_materialize import atomic_json


FORMAT = "nemotron-hybrid-width-plan-v1"
WIDTH_FORMAT = "nemotron-width-prune-v1"


def build_plan(
    base: dict,
    width_report: dict,
    max_mean_ratio: float,
    max_worst_ratio: float,
    include_layers: set[int] | None = None,
) -> dict:
    require(width_report.get("format") == WIDTH_FORMAT, "invalid width report format")
    require(base.get("source_revision") == width_report.get("source_revision"), "plan/report revision mismatch")
    layer_results = width_report.get("layer_results")
    require(isinstance(layer_results, dict), "width report has no layer results")
    layers = {}
    width_layers = []
    expert_equivalent_total = 0.0
    old_experts = base["old_num_experts"]
    for layer in base["model_moe_layers"]:
        key = str(layer)
        evidence = layer_results.get(key)
        use_width = False
        if isinstance(evidence, dict):
            summary = evidence["summary"]
            worst_ratio = summary["width_output_max_relative_l2"] / max(
                summary["hard_output_max_relative_l2"], 1e-30
            )
            use_width = (
                summary["width_to_hard_output_mean_ratio"] <= max_mean_ratio
                and worst_ratio <= max_worst_ratio
                and (include_layers is None or layer in include_layers)
            )
        if use_width:
            kept_blocks = evidence["kept_blocks"]
            source_blocks = evidence["source_blocks"]
            layers[key] = {"mode": "width", "kept_blocks": kept_blocks}
            width_layers.append(layer)
            expert_equivalent_total += old_experts * len(kept_blocks[0]) / source_blocks
        else:
            kept = base["kept_by_layer"][key]
            layers[key] = {"mode": "experts", "kept_experts": kept}
            expert_equivalent_total += len(kept)
    return {
        "format": FORMAT,
        "source_revision": base["source_revision"],
        "old_num_experts": old_experts,
        "model_moe_layers": base["model_moe_layers"],
        "base_plan_sha256": None,
        "width_report_sha256": None,
        "max_mean_ratio": max_mean_ratio,
        "max_worst_ratio": max_worst_ratio,
        "width_layers": width_layers,
        "expert_equivalent_total": expert_equivalent_total,
        "expert_equivalent_average": expert_equivalent_total / len(base["model_moe_layers"]),
        "layers": layers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--width-report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-mean-ratio", type=float, default=0.98)
    parser.add_argument("--max-worst-ratio", type=float, default=1.0)
    parser.add_argument("--include-layers", help="optional comma-separated width-layer allowlist")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.max_mean_ratio > 0 and args.max_worst_ratio > 0, "ratio gates must be positive")
        base = load_json(args.base_plan)
        report = load_json(args.width_report)
        config = load_json(args.config)
        validate_virtual_plan(base, config, base["source_revision"])
        include_layers = None
        if args.include_layers:
            include_layers = {int(value) for value in args.include_layers.split(",")}
            require(include_layers, "width-layer allowlist is empty")
        plan = build_plan(
            base, report, args.max_mean_ratio, args.max_worst_ratio, include_layers
        )
        plan["base_plan_sha256"] = sha256_file(args.base_plan)
        plan["width_report_sha256"] = sha256_file(args.width_report)
        validate_virtual_hybrid_plan(plan, config, plan["source_revision"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, plan)
        print(
            f"hybrid-plan path={args.output} width_layers={len(plan['width_layers'])} "
            f"equivalent_average={plan['expert_equivalent_average']:.3f} "
            f"sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        print(f"nemotron hybrid plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

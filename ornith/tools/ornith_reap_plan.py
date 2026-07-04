#!/usr/bin/env python3
"""Build a guarded expert-pruning plan from REAP-style observer metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def top_indices(values: list[float], n: int) -> set[int]:
    if n <= 0:
        return set()
    return {i for i, _ in sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:n]}


def super_experts(layers: dict[str, dict], include_last_layers: bool) -> dict[str, set[int]]:
    vals = []
    for layer in layers.values():
        vals.extend(float(v) for v in layer.get("max_activations", []))
    threshold = max(quantile(vals, 0.995), (max(vals) / 10.0 if vals else 0.0))
    out = {str(k): set() for k in layers}
    layer_ids = sorted(int(k) for k in layers)
    cutoff = len(layer_ids) if include_last_layers else int(len(layer_ids) * 0.75)
    eligible = set(layer_ids[:cutoff])
    for key, layer in layers.items():
        if int(key) not in eligible:
            continue
        out[str(key)] = {i for i, v in enumerate(layer.get("max_activations", [])) if float(v) > threshold}
    return out


def build_plan(data: dict, compression_ratio: float, metric: str, min_retained: int, preserve_top_fraction: float, preserve_outliers: bool) -> dict:
    layers = {str(k): v for k, v in data["layers"].items()}
    super_keep = super_experts(layers, include_last_layers=preserve_outliers)
    planned = {}
    for key in sorted(layers, key=lambda x: int(x)):
        layer = layers[key]
        scores = [float(v) for v in layer[metric]]
        n = len(scores)
        requested_prune = int(n * compression_ratio)
        max_prune = max(0, n - min_retained)
        keep = set(super_keep[key])
        keep |= top_indices([float(v) for v in layer.get("expert_frequency", [])], math.ceil(n * preserve_top_fraction))
        keep |= top_indices([float(v) for v in layer.get("reap", scores)], math.ceil(n * preserve_top_fraction))
        candidates = [i for i in range(n) if i not in keep]
        prune_n = min(requested_prune, max_prune, len(candidates))
        pruned = sorted(candidates, key=lambda i: scores[i])[:prune_n]
        retained = [i for i in range(n) if i not in set(pruned)]
        planned[key] = {
            "num_experts": n,
            "requested_prune": requested_prune,
            "pruned_count": len(pruned),
            "retained_count": len(retained),
            "preserved_count": len(keep),
            "pruned": pruned,
            "retained": retained,
        }
    return {
        "format": "ornith-reap-plan-v1",
        "metric": metric,
        "compression_ratio": compression_ratio,
        "min_retained": min_retained,
        "preserve_top_fraction": preserve_top_fraction,
        "preserve_outliers": preserve_outliers,
        "layers": planned,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--observer", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--metric", default="reap")
    p.add_argument("--compression-ratio", type=float, required=True)
    p.add_argument("--min-retained", type=int, default=16)
    p.add_argument("--preserve-top-fraction", type=float, default=0.02)
    p.add_argument("--preserve-outliers", action="store_true")
    args = p.parse_args()
    data = json.loads(args.observer.read_text(encoding="utf-8"))
    plan = build_plan(
        data,
        compression_ratio=args.compression_ratio,
        metric=args.metric,
        min_retained=args.min_retained,
        preserve_top_fraction=args.preserve_top_fraction,
        preserve_outliers=args.preserve_outliers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total_pruned = sum(layer["pruned_count"] for layer in plan["layers"].values())
    total = sum(layer["num_experts"] for layer in plan["layers"].values())
    print(f"layers={len(plan['layers'])} pruned={total_pruned}/{total} ratio={total_pruned / total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

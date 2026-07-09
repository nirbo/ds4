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


def normalized(values: list[float], n: int) -> list[float]:
    vals = (values + [0.0] * n)[:n]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.0] * n
    return [(v - lo) / (hi - lo) for v in vals]


def hybrid_scores(layer: dict, n: int) -> list[float]:
    freq = normalized([float(v) for v in layer.get("expert_frequency", [])], n)
    reap = normalized([float(v) for v in layer.get("reap", [])], n)
    ean = normalized([float(v) for v in layer.get("ean_mean", [])], n)
    maxa = normalized([float(v) for v in layer.get("max_activations", [])], n)
    return [0.45 * reap[i] + 0.25 * freq[i] + 0.20 * ean[i] + 0.10 * maxa[i] for i in range(n)]


def layer_prune_counts(layers: dict[str, dict], compression_ratio: float, min_retained: int, profile: str) -> dict[str, int]:
    keys = sorted(layers, key=lambda x: int(x))
    caps = {key: max(0, len(layers[key].get("expert_frequency", layers[key].get("reap", []))) - min_retained) for key in keys}
    target = sum(min(int(len(layers[key].get("expert_frequency", layers[key].get("reap", []))) * compression_ratio), caps[key]) for key in keys)
    if profile == "uniform" or not keys:
        return {key: min(int(len(layers[key].get("expert_frequency", layers[key].get("reap", []))) * compression_ratio), caps[key]) for key in keys}
    if profile != "late-protect":
        raise ValueError(f"unknown layer profile: {profile}")
    # ponytail: simple three-zone profile; replace with measured layer damage when available.
    raw = []
    for pos, key in enumerate(keys):
        frac = pos / max(1, len(keys) - 1)
        weight = 1.15 if frac < 0.50 else (0.95 if frac < 0.75 else 0.55)
        n = len(layers[key].get("expert_frequency", layers[key].get("reap", [])))
        raw.append((key, min(n * compression_ratio * weight, caps[key])))
    scale = target / sum(v for _, v in raw) if sum(v for _, v in raw) else 0.0
    counts = {key: min(int(v * scale), caps[key]) for key, v in raw}
    remaining = target - sum(counts.values())
    for key, _ in sorted(raw, key=lambda item: (item[1] * scale) - int(item[1] * scale), reverse=True):
        if remaining <= 0:
            break
        if counts[key] < caps[key]:
            counts[key] += 1
            remaining -= 1
    return counts


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


def observer_quality_errors(
    data: dict,
    expected_layers: int = 60,
    expected_experts: int = 512,
    min_tokens_per_layer: int = 32768,
    min_expert_frequency: int = 2,
) -> list[str]:
    errors = []
    if data.get("source_model") != "deepreinforce-ai/Ornith-1.0-397B":
        errors.append(f"wrong or missing source_model: {data.get('source_model')!r}")
    if data.get("source_precision") not in ("bf16", "fp16"):
        errors.append(f"REAP must be observed on original BF16/FP16 weights, got {data.get('source_precision')!r}")
    if not data.get("source_revision"):
        errors.append("missing immutable source_revision")
    layers = data.get("layers", {})
    if sorted(int(k) for k in layers) != list(range(expected_layers)):
        errors.append(f"expected layers 0..{expected_layers - 1}, got {len(layers)} layers")
    for key, layer in layers.items():
        freq = [int(v) for v in layer.get("expert_frequency", [])]
        if len(freq) != expected_experts:
            errors.append(f"layer {key}: expected {expected_experts} experts, got {len(freq)}")
            continue
        if int(layer.get("total_tokens", 0)) < min_tokens_per_layer:
            errors.append(f"layer {key}: only {layer.get('total_tokens', 0)} calibration tokens")
        low = sum(v < min_expert_frequency for v in freq)
        if low:
            errors.append(f"layer {key}: {low} experts selected fewer than {min_expert_frequency} times")
    return errors


def build_plan(data: dict, compression_ratio: float, metric: str, min_retained: int, preserve_top_fraction: float, preserve_outliers: bool, preserve_unobserved: bool = True, strategy: str = "reap", layer_profile: str = "uniform") -> dict:
    layers = {str(k): v for k, v in data["layers"].items()}
    super_keep = super_experts(layers, include_last_layers=preserve_outliers)
    prune_counts = layer_prune_counts(layers, compression_ratio, min_retained, layer_profile)
    planned = {}
    for key in sorted(layers, key=lambda x: int(x)):
        layer = layers[key]
        scores = hybrid_scores(layer, len(layer[metric])) if strategy == "hybrid" else [float(v) for v in layer[metric]]
        n = len(scores)
        freq = [int(v) for v in layer.get("expert_frequency", [0] * n)]
        if len(freq) < n:
            freq += [0] * (n - len(freq))
        freq = freq[:n]
        unobserved = {i for i, v in enumerate(freq) if v <= 0}
        observed_count = n - len(unobserved)
        requested_prune = prune_counts[key]
        max_prune = max(0, n - min_retained)
        keep = set(super_keep[key])
        if preserve_unobserved:
            keep |= unobserved
        keep |= top_indices([float(v) for v in freq], math.ceil(n * preserve_top_fraction))
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
            "observed_count": observed_count,
            "unobserved_count": len(unobserved),
            "unobserved_preserved_count": len(unobserved & keep),
            "observed_fraction": observed_count / n if n else 0.0,
            "candidate_count": len(candidates),
            "pruned": pruned,
            "retained": retained,
        }
    return {
        "format": "ornith-reap-plan-v1",
        "metric": metric,
        "strategy": strategy,
        "layer_profile": layer_profile,
        "compression_ratio": compression_ratio,
        "min_retained": min_retained,
        "preserve_top_fraction": preserve_top_fraction,
        "preserve_outliers": preserve_outliers,
        "preserve_unobserved": preserve_unobserved,
        "source_model": data.get("source_model", "unknown"),
        "source_precision": data.get("source_precision", "unknown"),
        "source_revision": data.get("source_revision"),
        "layers": planned,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--observer", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--metric", default="reap")
    p.add_argument("--strategy", choices=["reap", "hybrid"], default="reap")
    p.add_argument("--layer-profile", choices=["uniform", "late-protect"], default="uniform")
    p.add_argument("--compression-ratio", type=float, required=True)
    p.add_argument("--min-retained", type=int, default=16)
    p.add_argument("--preserve-top-fraction", type=float, default=0.0)
    p.add_argument("--preserve-outliers", action="store_true")
    p.add_argument("--allow-prune-unobserved", action="store_true")
    p.add_argument("--quality-profile", choices=["final", "experiment"], default="final")
    p.add_argument("--expected-layers", type=int, default=60)
    p.add_argument("--expected-experts", type=int, default=512)
    p.add_argument("--min-tokens-per-layer", type=int, default=32768)
    p.add_argument("--min-expert-frequency", type=int, default=2)
    args = p.parse_args()
    data = json.loads(args.observer.read_text(encoding="utf-8"))
    if args.quality_profile == "final":
        errors = observer_quality_errors(
            data,
            expected_layers=args.expected_layers,
            expected_experts=args.expected_experts,
            min_tokens_per_layer=args.min_tokens_per_layer,
            min_expert_frequency=args.min_expert_frequency,
        )
        if args.strategy != "reap" or args.layer_profile != "uniform":
            errors.append("final plans require the measured REAP metric with a uniform layer profile")
        if args.allow_prune_unobserved:
            errors.append("final plans cannot prune unobserved experts")
        if errors:
            raise SystemExit("REAP quality gate failed:\n  " + "\n  ".join(errors))
    plan = build_plan(
        data,
        compression_ratio=args.compression_ratio,
        metric=args.metric,
        min_retained=args.min_retained,
        preserve_top_fraction=args.preserve_top_fraction,
        preserve_outliers=args.preserve_outliers,
        preserve_unobserved=not args.allow_prune_unobserved,
        strategy=args.strategy,
        layer_profile=args.layer_profile,
    )
    plan["quality_profile"] = args.quality_profile
    plan["observer"] = str(args.observer.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total_pruned = sum(layer["pruned_count"] for layer in plan["layers"].values())
    total = sum(layer["num_experts"] for layer in plan["layers"].values())
    observed = sum(layer["observed_count"] for layer in plan["layers"].values())
    print(f"layers={len(plan['layers'])} pruned={total_pruned}/{total} ratio={total_pruned / total:.4f} observed={observed}/{total} coverage={observed / total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

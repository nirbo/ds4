#!/usr/bin/env python3
"""Merge Ornith REAP observer JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def zeros(n: int) -> list[float]:
    return [0.0] * n


def merge_layer(dst: dict, src: dict) -> None:
    n = len(src["expert_frequency"])
    if not dst:
        dst.update({
            "total_tokens": 0,
            "expert_frequency": [0] * n,
            "weighted_expert_frequency_sum": zeros(n),
            "ean_weighted_sum": zeros(n),
            "reap_weighted_sum": zeros(n),
            "max_activations": zeros(n),
        })
    if len(dst["expert_frequency"]) != n:
        raise ValueError("expert count mismatch")
    for i in range(n):
        freq = int(src["expert_frequency"][i])
        dst["expert_frequency"][i] += freq
        dst["weighted_expert_frequency_sum"][i] += float(src["weighted_expert_frequency_sum"][i])
        dst["ean_weighted_sum"][i] += float(src["ean_mean"][i]) * freq
        dst["reap_weighted_sum"][i] += float(src["reap"][i]) * freq
        dst["max_activations"][i] = max(dst["max_activations"][i], float(src["max_activations"][i]))
    dst["total_tokens"] += int(src["total_tokens"])


def finalize(acc: dict) -> dict:
    layers = {}
    for layer, data in sorted(acc.items(), key=lambda item: int(item[0])):
        freq = data["expert_frequency"]
        ean = zeros(len(freq))
        reap = zeros(len(freq))
        for i, count in enumerate(freq):
            if count:
                ean[i] = data["ean_weighted_sum"][i] / count
                reap[i] = data["reap_weighted_sum"][i] / count
        layers[layer] = {
            "total_tokens": data["total_tokens"],
            "expert_frequency": freq,
            "weighted_expert_frequency_sum": data["weighted_expert_frequency_sum"],
            "ean_mean": ean,
            "reap": reap,
            "max_activations": data["max_activations"],
        }
    return {"format": "ornith-reap-observer-v1", "layers": layers}


def run(paths: list[Path]) -> dict:
    acc: dict[str, dict] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for layer, src in data["layers"].items():
            merge_layer(acc.setdefault(str(layer), {}), src)
    return finalize(acc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("observations", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    report = run(args.observations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged={len(args.observations)} layers={len(report['layers'])} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

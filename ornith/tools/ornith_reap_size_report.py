#!/usr/bin/env python3
"""Estimate .ornq size after applying a REAP expert-retention plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ornith_quantize_safetensors import load_policy, quant_bytes, quant_mode


def gib(n: int) -> float:
    return n / 1024**3


def add(bucket: dict[str, dict], key: str, current: int, projected: int) -> None:
    row = bucket.setdefault(key, {"tensors": 0, "current": 0, "projected": 0})
    row["tensors"] += 1
    row["current"] += current
    row["projected"] += projected


def pruned_shape(name: str, meta: dict, plan: dict) -> list[int]:
    shape = [int(v) for v in meta["shape"]]
    layer = meta.get("layer")
    if meta.get("group") == "routed_expert" and layer is not None and str(layer) in plan["layers"]:
        shape[0] = int(plan["layers"][str(layer)]["retained_count"])
    elif meta.get("group") == "router" and str(meta.get("kind")) == "mlp.gate.weight" and layer is not None and str(layer) in plan["layers"]:
        shape[0] = int(plan["layers"][str(layer)]["retained_count"])
    return shape


def product(xs: list[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def run(catalog: Path, plan_path: Path, policy_path: Path, block: int) -> dict:
    data = json.loads(catalog.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    policy = load_policy(policy_path)
    by_group: dict[str, dict] = {}
    by_quant: dict[str, dict] = {}
    by_layer: dict[str, dict] = {}
    current = 0
    projected = 0
    for name, meta in data["tensors"].items():
        old_bytes = int(meta["nbytes"])
        shape = pruned_shape(name, meta, plan)
        nparams = product(shape)
        q = quant_mode(name, shape, nparams, policy)
        new_bytes = quant_bytes(nparams, q, block)
        current += old_bytes
        projected += new_bytes
        add(by_group, str(meta.get("group") or "unknown"), old_bytes, new_bytes)
        add(by_quant, q, old_bytes, new_bytes)
        add(by_layer, str(meta.get("layer") if meta.get("layer") is not None else "none"), old_bytes, new_bytes)
    return {
        "format": "ornith-reap-size-report-v1",
        "catalog": str(catalog),
        "plan": str(plan_path),
        "policy": policy.get("name", str(policy_path)) if policy else str(policy_path),
        "current_bytes": current,
        "projected_bytes": projected,
        "by_group": by_group,
        "by_quant": by_quant,
        "by_layer": by_layer,
    }


def print_bucket(title: str, bucket: dict[str, dict]) -> None:
    print(title)
    print("  name tensors current_gib projected_gib delta_gib")
    for key, row in sorted(bucket.items()):
        print(f"  {key} {row['tensors']} {gib(row['current']):.2f} {gib(row['projected']):.2f} {gib(row['projected'] - row['current']):+.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, default=Path("/Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.json"))
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()
    report = run(args.catalog, args.plan, args.policy, args.block)
    print(f"policy={report['policy']}")
    print(f"current={gib(report['current_bytes']):.2f}GiB projected={gib(report['projected_bytes']):.2f}GiB delta={gib(report['projected_bytes'] - report['current_bytes']):+.2f}GiB")
    print_bucket("by_quant", report["by_quant"])
    print_bucket("by_group", report["by_group"])
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

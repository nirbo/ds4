#!/usr/bin/env python3
"""Report projected .ornq bytes for a quantization policy."""

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


def run(catalog: Path, policy_path: Path, block: int) -> dict:
    policy = load_policy(policy_path)
    data = json.loads(catalog.read_text(encoding="utf-8"))
    by_quant: dict[str, dict] = {}
    by_group: dict[str, dict] = {}
    by_shard: dict[str, dict] = {}
    by_layer: dict[str, dict] = {}
    total_current = 0
    total_projected = 0
    for name, meta in data["tensors"].items():
        shape = [int(v) for v in meta["shape"]]
        nparams = int(meta["nparams"])
        current = int(meta["nbytes"])
        q = quant_mode(name, shape, nparams, policy)
        projected = quant_bytes(nparams, q, block)
        total_current += current
        total_projected += projected
        add(by_quant, q, current, projected)
        add(by_group, str(meta.get("group") or "unknown"), current, projected)
        add(by_shard, str(meta.get("shard") or "unknown"), current, projected)
        add(by_layer, str(meta.get("layer") if meta.get("layer") is not None else "none"), current, projected)
    return {
        "policy": policy.get("name", str(policy_path)) if policy else str(policy_path),
        "catalog": str(catalog),
        "block_size": block,
        "current_bytes": total_current,
        "projected_bytes": total_projected,
        "by_quant": by_quant,
        "by_group": by_group,
        "by_shard": by_shard,
        "by_layer": by_layer,
    }


def print_bucket(title: str, bucket: dict[str, dict]) -> None:
    print(title)
    print("  name tensors current_gib projected_gib delta_gib")
    for key, row in sorted(bucket.items()):
        delta = row["projected"] - row["current"]
        print(f"  {key} {row['tensors']} {gib(row['current']):.2f} {gib(row['projected']):.2f} {gib(delta):+.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, default=Path("/Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.json"))
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--print-layers", action="store_true")
    args = p.parse_args()
    report = run(args.catalog, args.policy, args.block)
    print(f"policy={report['policy']}")
    print(f"current={gib(report['current_bytes']):.2f}GiB projected={gib(report['projected_bytes']):.2f}GiB delta={gib(report['projected_bytes'] - report['current_bytes']):+.2f}GiB")
    print_bucket("by_quant", report["by_quant"])
    print_bucket("by_group", report["by_group"])
    if args.print_layers:
        print_bucket("by_layer", report["by_layer"])
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

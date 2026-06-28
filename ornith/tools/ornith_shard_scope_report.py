#!/usr/bin/env python3
"""Classify Ornith safetensors shards by tensor scope from a local index."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def tensor_scope(name: str) -> str:
    if name.startswith("model.visual."):
        return "visual"
    if name.startswith("model.language_model.") or name == "lm_head.weight":
        return "language"
    return "other"


def category(scopes: dict[str, int]) -> str:
    return "+".join(sorted(scopes))


def shard_scope_report(index: dict) -> dict:
    by_shard = defaultdict(list)
    for tensor, shard in index["weight_map"].items():
        by_shard[shard].append(tensor)

    shards = []
    for shard in sorted(by_shard):
        scopes = Counter(tensor_scope(name) for name in by_shard[shard])
        tensors = sorted(by_shard[shard])
        shards.append({
            "file": shard,
            "category": category(scopes),
            "tensor_count": len(tensors),
            "scopes": dict(sorted(scopes.items())),
            "first_tensor": tensors[0],
            "last_tensor": tensors[-1],
        })

    categories = Counter(shard["category"] for shard in shards)
    return {
        "total_weight_bytes": int((index.get("metadata") or {}).get("total_size") or 0),
        "shard_count": len(shards),
        "tensor_count": len(index["weight_map"]),
        "categories": dict(sorted(categories.items())),
        "shards": shards,
    }


def print_summary(report: dict) -> None:
    print(f"shards: {report['shard_count']}")
    print(f"tensors: {report['tensor_count']}")
    print(f"total weight bytes: {report['total_weight_bytes']}")
    print("categories:")
    for name, count in report["categories"].items():
        print(f"  {name}: {count}")
    print("mixed shards:")
    for shard in report["shards"]:
        if "+" in shard["category"]:
            print(f"  {shard['file']}: {shard['category']} {shard['scopes']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--out", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = shard_scope_report(load_json(args.index))
    print_summary(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

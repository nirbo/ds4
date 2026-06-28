#!/usr/bin/env python3
"""Plan text-only Ornith shard repacking from a safetensors index."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ornith_storage_manifest import is_text_tensor, load_json


def shard_plan(index: dict) -> dict:
    by_shard = defaultdict(list)
    for tensor, shard in index["weight_map"].items():
        by_shard[shard].append(tensor)

    shards = []
    for shard in sorted(by_shard):
        tensors = by_shard[shard]
        text = sorted(name for name in tensors if is_text_tensor(name))
        skipped = sorted(name for name in tensors if not is_text_tensor(name))
        if text and skipped:
            action = "filter"
        elif text:
            action = "copy"
        else:
            action = "skip"
        shards.append({
            "file": shard,
            "action": action,
            "text_tensor_count": len(text),
            "skipped_tensor_count": len(skipped),
        })

    actions = Counter(shard["action"] for shard in shards)
    return {
        "shard_count": len(shards),
        "text_tensor_count": sum(shard["text_tensor_count"] for shard in shards),
        "skipped_tensor_count": sum(shard["skipped_tensor_count"] for shard in shards),
        "actions": dict(sorted(actions.items())),
        "shards": shards,
    }


def print_summary(plan: dict) -> None:
    print(f"shards: {plan['shard_count']}")
    print(f"text tensors: {plan['text_tensor_count']}")
    print(f"skipped tensors: {plan['skipped_tensor_count']}")
    print("actions:")
    for action, count in plan["actions"].items():
        print(f"  {action}: {count}")
    print("filter shards:")
    for shard in plan["shards"]:
        if shard["action"] == "filter":
            print(f"  {shard['file']}: text={shard['text_tensor_count']} skipped={shard['skipped_tensor_count']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--out", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plan = shard_plan(load_json(args.index))
    print_summary(plan)
    if args.out:
        args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

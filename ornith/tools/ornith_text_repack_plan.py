#!/usr/bin/env python3
"""Plan text-only Ornith shard repacking from a safetensors index."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ornith_safetensors_filter import filter_safetensors, load_allowlist, read_header
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


def dry_run_actions(plan: dict, src_dir: Path, dst_dir: Path, allowlist_dir: Path) -> list[str]:
    out = []
    for shard in plan["shards"]:
        src = src_dir / shard["file"]
        dst = dst_dir / shard["file"]
        if shard["action"] == "copy":
            out.append(f"copy {src} {dst} # text={shard['text_tensor_count']}")
        elif shard["action"] == "filter":
            allowlist = allowlist_dir / f"{Path(shard['file']).stem}.text.allowlist"
            out.append(
                f"filter {src} {dst} --allowlist {allowlist} "
                f"# text={shard['text_tensor_count']} skipped={shard['skipped_tensor_count']}"
            )
    return out


def print_dry_run(plan: dict, src_dir: Path, dst_dir: Path, allowlist_dir: Path) -> None:
    print("dry-run actions:")
    for action in dry_run_actions(plan, src_dir, dst_dir, allowlist_dir):
        print(f"  {action}")


def execute_plan(plan: dict, src_dir: Path, dst_dir: Path, allowlist_dir: Path) -> dict[str, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    counts = {"copy": 0, "filter": 0}
    for shard in plan["shards"]:
        src = src_dir / shard["file"]
        dst = dst_dir / shard["file"]
        if shard["action"] == "copy":
            shutil.copy2(src, dst)
            counts["copy"] += 1
        elif shard["action"] == "filter":
            allowlist = allowlist_dir / f"{Path(shard['file']).stem}.text.allowlist"
            filter_safetensors(src, dst, allowlist=load_allowlist(allowlist))
            counts["filter"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--src-dir", type=Path, default=Path("."))
    p.add_argument("--dst-dir", type=Path, default=Path("ornith-text"))
    p.add_argument("--allowlist-dir", type=Path, default=Path("."))
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plan = shard_plan(load_json(args.index))
    print_summary(plan)
    if args.dry_run or not args.execute:
        print_dry_run(plan, args.src_dir, args.dst_dir, args.allowlist_dir)
    if args.execute:
        counts = execute_plan(plan, args.src_dir, args.dst_dir, args.allowlist_dir)
        print(f"executed copy: {counts['copy']}")
        print(f"executed filter: {counts['filter']}")
    if args.out:
        args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

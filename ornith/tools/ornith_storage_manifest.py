#!/usr/bin/env python3
"""Create a small storage/download manifest from the local Ornith index."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_REPO = "deepreinforce-ai/Ornith-1.0-397B"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def is_text_tensor(name: str) -> bool:
    return name.startswith("model.language_model.") or name == "lm_head.weight"


def shard_manifest(index: dict, repo: str, text_only: bool = False) -> dict:
    weight_map = index["weight_map"]
    shards = defaultdict(list)
    skipped = defaultdict(int)
    for tensor, shard in weight_map.items():
        if text_only and not is_text_tensor(tensor):
            skipped[shard] += 1
        else:
            shards[shard].append(tensor)
    total_size = int((index.get("metadata") or {}).get("total_size") or 0)
    ordered = []
    for shard in sorted(shards):
        tensors = sorted(shards[shard])
        ordered.append({
            "file": shard,
            "url": f"https://huggingface.co/{repo}/resolve/main/{shard}",
            "tensor_count": len(tensors),
            "skipped_tensor_count": skipped.get(shard, 0),
            "first_tensor": tensors[0],
            "last_tensor": tensors[-1],
        })
    return {
        "repo": repo,
        "text_only": text_only,
        "total_weight_bytes": total_size,
        "shard_count": len(ordered),
        "tensor_count": sum(len(tensors) for tensors in shards.values()),
        "skipped_tensor_count": sum(skipped.values()),
        "shards": ordered,
    }


def gib(n: int) -> float:
    return n / 1024**3


def print_summary(manifest: dict) -> None:
    total = int(manifest["total_weight_bytes"])
    shard_count = int(manifest["shard_count"])
    print(f"repo: {manifest['repo']}")
    print(f"total weight bytes: {total} ({gib(total):.2f} GiB)")
    print(f"shards: {shard_count}")
    if shard_count:
        print(f"average shard: {gib(total / shard_count):.2f} GiB")
    print(f"tensors: {manifest['tensor_count']}")
    if manifest.get("text_only"):
        print(f"text-only: yes")
        print(f"skipped tensors: {manifest['skipped_tensor_count']}")

    counts = Counter()
    for shard in manifest["shards"]:
        first = shard["first_tensor"]
        if first.startswith("model.visual."):
            counts["visual-start"] += 1
        elif ".layers." in first:
            counts["language-layer-start"] += 1
        else:
            counts["other-start"] += 1
    if counts:
        print("shard starts:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--out", type=Path)
    p.add_argument("--text-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = shard_manifest(load_json(args.index), args.repo, args.text_only)
    print_summary(manifest)
    if args.out:
        args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

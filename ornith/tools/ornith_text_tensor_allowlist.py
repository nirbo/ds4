#!/usr/bin/env python3
"""Write text tensor allowlists from an Ornith safetensors index."""

from __future__ import annotations

import argparse
from pathlib import Path

from ornith_storage_manifest import is_text_tensor, load_json


def text_tensors_for_shard(index: dict, shard: str) -> list[str]:
    return sorted(
        name
        for name, filename in index["weight_map"].items()
        if filename == shard and is_text_tensor(name)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--shard", required=True)
    p.add_argument("--out", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    names = text_tensors_for_shard(load_json(args.index), args.shard)
    text = "".join(f"{name}\n" for name in names)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote: {args.out}")
    else:
        print(text, end="")
    print(f"text tensors: {len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a checked per-layer tensor catalog for Ornith text weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ornith_layout_check import FULL_ATTN, GLOBAL_TEXT, LINEAR_ATTN, MOE, load_json, text_config


def tensor_entry(name: str, index: dict) -> dict:
    return {
        "name": name,
        "shard": index["weight_map"].get(name),
    }


def group_entries(names: list[str], index: dict) -> list[dict]:
    return [tensor_entry(name, index) for name in names]


def layer_catalog(config: dict, index: dict) -> dict:
    cfg = text_config(config)
    layers = []
    missing = []
    layer_types = cfg["layer_types"]

    global_entries = group_entries(GLOBAL_TEXT, index)
    missing.extend(entry["name"] for entry in global_entries if entry["shard"] is None)

    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}."
        if layer_type == "linear_attention":
            attn = LINEAR_ATTN
        elif layer_type == "full_attention":
            attn = FULL_ATTN
        else:
            raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer}")

        attention = group_entries([prefix + name for name in attn], index)
        moe = group_entries([prefix + name for name in MOE], index)
        missing.extend(entry["name"] for entry in attention + moe if entry["shard"] is None)
        shards = sorted({entry["shard"] for entry in attention + moe if entry["shard"]})
        layers.append({
            "layer": layer,
            "type": layer_type,
            "shards": shards,
            "groups": {
                "attention": attention,
                "moe": moe,
            },
        })

    language_shards = sorted({
        shard
        for name, shard in index["weight_map"].items()
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    })
    type_counts = Counter(layer_types)
    return {
        "num_layers": len(layer_types),
        "layer_type_counts": dict(sorted(type_counts.items())),
        "global": global_entries,
        "layers": layers,
        "language_shards": language_shards,
        "missing": sorted(missing),
    }


def print_summary(catalog: dict) -> None:
    print(f"text layers: {catalog['num_layers']}")
    print("layer types:")
    for name, count in catalog["layer_type_counts"].items():
        print(f"  {name}: {count}")
    print(f"language shards: {len(catalog['language_shards'])}")
    print(f"missing tensors: {len(catalog['missing'])}")
    if catalog["missing"]:
        for name in catalog["missing"]:
            print(f"  {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--out", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = layer_catalog(load_json(args.config), load_json(args.index))
    print_summary(catalog)
    if args.out:
        args.out.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    return 1 if catalog["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

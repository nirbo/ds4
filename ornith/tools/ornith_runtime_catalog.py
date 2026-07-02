#!/usr/bin/env python3
"""Build a compact runtime catalog from Ornith .ornq shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ornith_runtime import ORNQShard, layer_catalog, memory_report
from ornith_storage_manifest import is_text_tensor, load_json


FORMAT = "ornith-runtime-catalog-v1"
REQUIRED_GLOBALS = {
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "lm_head.weight",
}


def ornq_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.ornq")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"{root}: no .ornq files")
    return paths


def index_text_tensors(index: dict) -> set[str]:
    return {name for name in index["weight_map"] if is_text_tensor(name)}


def build_catalog(paths: list[Path], index: dict | None = None) -> dict:
    root = paths[0].parent if len(paths) > 1 else paths[0].parent
    shards = []
    tensors = {}
    open_shards = []
    try:
        for path in paths:
            shard = ORNQShard(path)
            open_shards.append(shard)
            st = path.stat()
            shards.append({
                "file": path.name,
                "size": st.st_size,
                "data_start": shard.data_start,
                "block_size": shard.block_size,
                "tensor_count": len(shard.tensors),
            })
            for name, tensor in sorted(shard.tensors.items()):
                if name in tensors:
                    raise ValueError(f"duplicate tensor: {name}")
                role = tensor.role
                tensors[name] = {
                    "shard": path.name,
                    "payload_offset": tensor.payload_offset,
                    "data_offsets": list(tensor.data_offsets),
                    "quant": tensor.quant,
                    "shape": tensor.shape,
                    "nparams": tensor.nparams,
                    "nbytes": tensor.nbytes,
                    "layer": role.layer,
                    "group": role.group,
                    "kind": role.kind,
                }

        report = memory_report(open_shards)
        layers = {str(k): v for k, v in layer_catalog(open_shards).items()}
    finally:
        for shard in open_shards:
            shard.close()

    missing = []
    extra = []
    if index is not None:
        expected = index_text_tensors(index)
        have = set(tensors)
        missing = sorted(expected - have)
        extra = sorted(have - expected)

    return {
        "format": FORMAT,
        "source_dir": str(root),
        "shard_count": len(shards),
        "tensor_count": len(tensors),
        "layer_count": len(layers),
        "bytes_by_quant": report["bytes_by_quant"],
        "params_by_quant": report["params_by_quant"],
        "tensors_by_group": report["tensors_by_group"],
        "shards": shards,
        "tensors": dict(sorted(tensors.items())),
        "layers": layers,
        "missing_text_tensors": missing,
        "unexpected_text_tensors": extra,
    }


def validate_catalog(catalog: dict) -> list[str]:
    errors = []
    tensors = catalog["tensors"]
    missing_globals = sorted(REQUIRED_GLOBALS - set(tensors))
    errors.extend(f"missing required tensor: {name}" for name in missing_globals)
    errors.extend(f"missing text tensor: {name}" for name in catalog.get("missing_text_tensors", []))
    errors.extend(f"unexpected text tensor: {name}" for name in catalog.get("unexpected_text_tensors", []))
    for name in tensors:
        if name.startswith("model.visual."):
            errors.append(f"vision tensor present: {name}")

    layers = sorted(int(k) for k in catalog["layers"])
    if layers:
        want = list(range(layers[-1] + 1))
        if layers != want:
            errors.append(f"layer ids are not contiguous 0..{layers[-1]}")
    return errors


def print_summary(catalog: dict) -> None:
    print(f"shards: {catalog['shard_count']}")
    print(f"tensors: {catalog['tensor_count']}")
    print(f"layers: {catalog['layer_count']}")
    print("bytes_by_quant:")
    for name, value in catalog["bytes_by_quant"].items():
        print(f"  {name}: {value}")
    if catalog["missing_text_tensors"] or catalog["unexpected_text_tensors"]:
        print(f"missing_text_tensors: {len(catalog['missing_text_tensors'])}")
        print(f"unexpected_text_tensors: {len(catalog['unexpected_text_tensors'])}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--index", type=Path)
    p.add_argument("--out", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    index = load_json(args.index) if args.index else None
    catalog = build_catalog(ornq_paths(args.path), index)
    print_summary(catalog)
    errors = validate_catalog(catalog)
    if args.out:
        args.out.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    if errors:
        for error in errors:
            print(error)
        return 1
    print("ornith runtime catalog: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

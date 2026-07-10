#!/usr/bin/env python3
"""Inventory Nemotron NVFP4 safetensors without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nemotron_metadata import (
    EXPERT_TENSOR_RE,
    MetadataError,
    load_json,
    require,
    validate_metadata,
)


INVENTORY_FORMAT = "nemotron-safetensors-inventory-v1"
SOURCE_STATE_FORMAT = "nemotron-source-snapshot-v1"
DTYPE_BYTES = {
    "F32": 4,
    "BF16": 2,
    "F8_E4M3": 1,
    "U8": 1,
}
BACKBONE_LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.")
ROUTER_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.gate\."
    r"(weight|e_score_correction_bias)$"
)


def tensor_nbytes(dtype: str, shape: list[int]) -> int:
    require(dtype in DTYPE_BYTES, f"unsupported safetensors dtype: {dtype}")
    require(
        isinstance(shape, list)
        and all(isinstance(dimension, int) and dimension >= 0 for dimension in shape),
        f"invalid tensor shape: {shape}",
    )
    return math.prod(shape) * DTYPE_BYTES[dtype]


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            require(len(prefix) == 8, f"truncated safetensors prefix: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            require(0 < header_size <= file_size - 8, f"invalid header size in {path}")
            header = json.loads(handle.read(header_size))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot read safetensors header {path}: {exc}") from exc

    require(isinstance(header, dict), f"invalid safetensors header object: {path}")
    tensors: dict[str, Any] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            require(isinstance(entry, dict), f"invalid safetensors metadata: {path}")
            continue
        require(isinstance(entry, dict), f"invalid tensor header for {name}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(offset, int) for offset in offsets),
            f"invalid data offsets for {name}",
        )
        start, end = offsets
        require(0 <= start <= end, f"invalid data interval for {name}")
        require(end - start == tensor_nbytes(dtype, shape), f"payload size mismatch for {name}")
        tensors[name] = entry
        intervals.append((start, end, name))

    require(tensors, f"safetensors shard has no tensors: {path}")
    cursor = 0
    for start, end, name in sorted(intervals):
        require(start == cursor, f"non-contiguous payload before {name} in {path.name}")
        cursor = end
    require(8 + header_size + cursor == file_size, f"file size mismatch: {path}")
    return tensors, header_size, cursor


def classify_tensor(name: str, pattern: str) -> str:
    if EXPERT_TENSOR_RE.fullmatch(name):
        return "backbone_routed_expert"
    if ROUTER_RE.fullmatch(name):
        return "backbone_router"
    layer_match = BACKBONE_LAYER_RE.match(name)
    if layer_match:
        layer_type = pattern[int(layer_match.group(1))]
        return {
            "M": "backbone_mamba",
            "E": "backbone_moe_fixed",
            "*": "backbone_attention",
        }[layer_type]
    if name.startswith("mtp."):
        return "mtp"
    if name == "backbone.embeddings.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm_head"
    return "top_level"


def validate_source_state(path: Path, source_dir: Path) -> dict[str, Any]:
    state = load_json(path)
    require(state.get("format") == SOURCE_STATE_FORMAT, "unsupported source state")
    require(state.get("verification", {}).get("result") == "passed", "source is not verified")
    require(Path(state.get("path", "")).resolve() == source_dir.resolve(), "source path mismatch")
    return state


def build_inventory(
    source_dir: Path,
    config: dict[str, Any],
    index: dict[str, Any],
    *,
    strict_target: bool = True,
    source_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_catalog = validate_metadata(config, index, strict_target=strict_target)
    weight_map = index["weight_map"]
    pattern = config["hybrid_override_pattern"]
    expected_shards = metadata_catalog["weights"]["shards"]

    tensor_entries: dict[str, dict[str, Any]] = {}
    shard_reports: list[dict[str, Any]] = []
    header_bytes = 0
    payload_bytes = 0
    for shard in expected_shards:
        path = source_dir / shard
        require(path.is_file(), f"missing safetensors shard: {path}")
        tensors, shard_header_bytes, shard_payload_bytes = read_safetensors_header(path)
        for name, entry in tensors.items():
            require(name not in tensor_entries, f"duplicate tensor across shards: {name}")
            require(weight_map.get(name) == shard, f"index/header shard mismatch for {name}")
            tensor_entries[name] = entry
        header_bytes += shard_header_bytes
        payload_bytes += shard_payload_bytes
        shard_reports.append(
            {
                "name": shard,
                "file_bytes": path.stat().st_size,
                "header_bytes": shard_header_bytes,
                "payload_bytes": shard_payload_bytes,
                "tensor_count": len(tensors),
            }
        )

    indexed_names = set(weight_map)
    header_names = set(tensor_entries)
    require(indexed_names == header_names, f"index/header tensor mismatch: missing={len(indexed_names - header_names)} extra={len(header_names - indexed_names)}")
    require(payload_bytes == index["metadata"]["total_size"], "header/index payload total mismatch")

    role_bytes: Counter[str] = Counter()
    role_tensors: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    dtype_tensors: Counter[str] = Counter()
    expert_bytes: dict[int, Counter[int]] = defaultdict(Counter)
    router_row_bytes: dict[int, int] = defaultdict(int)

    for name, entry in tensor_entries.items():
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        role = classify_tensor(name, pattern)
        role_bytes[role] += size
        role_tensors[role] += 1
        dtype_bytes[entry["dtype"]] += size
        dtype_tensors[entry["dtype"]] += 1

        expert_match = EXPERT_TENSOR_RE.fullmatch(name)
        if expert_match:
            expert_bytes[int(expert_match.group(1))][int(expert_match.group(2))] += size
        router_match = ROUTER_RE.fullmatch(name)
        if router_match:
            layer = int(router_match.group(1))
            shape = entry["shape"]
            require(shape and shape[0] == config["n_routed_experts"], f"router leading dimension mismatch: {name}")
            require(size % shape[0] == 0, f"router rows are not fixed-size: {name}")
            router_row_bytes[layer] += size // shape[0]

    moe_layers = [layer for layer, layer_type in enumerate(pattern) if layer_type == "E"]
    per_layer_expert_bytes: dict[str, int] = {}
    for layer in moe_layers:
        sizes = set(expert_bytes[layer].values())
        require(len(sizes) == 1, f"non-uniform expert storage in layer {layer}")
        require(len(expert_bytes[layer]) == config["n_routed_experts"], f"incomplete expert bytes in layer {layer}")
        per_layer_expert_bytes[str(layer)] = sizes.pop()
        require(router_row_bytes[layer] > 0, f"missing router row storage for layer {layer}")

    projections = []
    original_experts = config["n_routed_experts"]
    for prune_percent in (0, 5, 10, 15, 20, 25, 30, 35, 40):
        retained = round(original_experts * (100 - prune_percent) / 100)
        removed = original_experts - retained
        removed_bytes = sum(
            removed * (per_layer_expert_bytes[str(layer)] + router_row_bytes[layer])
            for layer in moe_layers
        )
        with_mtp = payload_bytes - removed_bytes
        without_mtp = with_mtp - role_bytes["mtp"]
        projections.append(
            {
                "prune_percent": prune_percent,
                "retained_experts_per_layer": retained,
                "with_mtp_bytes": with_mtp,
                "with_mtp_gib": with_mtp / 2**30,
                "without_mtp_bytes": without_mtp,
                "without_mtp_gib": without_mtp / 2**30,
            }
        )

    return {
        "format": INVENTORY_FORMAT,
        "source": {
            "repository": source_state.get("repository") if source_state else None,
            "revision": source_state.get("revision") if source_state else None,
            "path": str(source_dir.resolve()),
        },
        "totals": {
            "shards": len(expected_shards),
            "tensors": len(tensor_entries),
            "payload_bytes": payload_bytes,
            "header_bytes": header_bytes,
            "file_bytes": sum(report["file_bytes"] for report in shard_reports),
        },
        "roles": {
            role: {"tensors": role_tensors[role], "bytes": role_bytes[role], "gib": role_bytes[role] / 2**30}
            for role in sorted(role_bytes)
        },
        "dtypes": {
            dtype: {"tensors": dtype_tensors[dtype], "bytes": dtype_bytes[dtype], "gib": dtype_bytes[dtype] / 2**30}
            for dtype in sorted(dtype_bytes)
        },
        "moe": {
            "layers": moe_layers,
            "experts_per_layer": original_experts,
            "expert_bytes_per_layer": per_layer_expert_bytes,
            "router_row_bytes_per_layer": {str(layer): router_row_bytes[layer] for layer in moe_layers},
        },
        "uniform_pruning_projections": projections,
        "shard_reports": shard_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-strict-target", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = args.source_dir / "config.json"
        index_path = args.source_dir / "model.safetensors.index.json"
        source_state = validate_source_state(args.source_state, args.source_dir) if args.source_state else None
        inventory = build_inventory(
            args.source_dir,
            load_json(config_path),
            load_json(index_path),
            strict_target=not args.no_strict_target,
            source_state=source_state,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.out.with_name(args.out.name + ".part")
            temporary.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.out)

        totals = inventory["totals"]
        print(
            f"nemotron safetensors ok: shards={totals['shards']} tensors={totals['tensors']} "
            f"payload={totals['payload_bytes'] / 2**30:.2f}GiB headers={totals['header_bytes'] / 2**20:.2f}MiB"
        )
        print(f"mtp: {inventory['roles']['mtp']['gib']:.2f}GiB")
        print("uniform prune projections (MTP omitted):")
        for projection in inventory["uniform_pruning_projections"]:
            if projection["prune_percent"] in (0, 10, 15, 20, 25, 30, 35):
                print(
                    f"  {projection['prune_percent']:2d}% keep={projection['retained_experts_per_layer']:3d} "
                    f"payload={projection['without_mtp_gib']:.2f}GiB"
                )
        if args.out:
            print(f"inventory: {args.out}")
        return 0
    except MetadataError as exc:
        print(f"nemotron safetensors error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

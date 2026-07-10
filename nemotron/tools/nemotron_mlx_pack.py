#!/usr/bin/env python3
"""Directly materialize a pruned, MLX-ready Nemotron runtime checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import (
    OperationLog,
    atomic_json,
    load_source_state,
    sha256_file,
    validate_plan,
)
from nemotron_safetensors_inventory import read_safetensors_header


FORMAT = "nemotron-mlx-runtime-v1"
STATE_FORMAT = "nemotron-mlx-pack-state-v1"
COPY_CHUNK_BYTES = 16 * 1024 * 1024
LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.")
EXPERT_RE = re.compile(
    r"^(backbone\.layers\.(\d+)\.mixer)\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)$"
)
ROUTER_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.gate\."
    r"(weight|e_score_correction_bias)$"
)
SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        try:
            header = json.loads(handle.read(header_bytes))
        except json.JSONDecodeError as exc:
            raise MetadataError(f"invalid safetensors header {path}: {exc}") from exc
    require(isinstance(header, dict), f"invalid safetensors header object: {path}")
    return header, header_bytes


def source_catalog(source_dir: Path, index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        shard_path = source_dir / shard_name
        header, header_bytes = read_header(shard_path)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            require(index["weight_map"].get(name) == shard_name, f"index/header mismatch: {name}")
            start, end = entry["data_offsets"]
            catalog[name] = {
                "path": shard_path,
                "offset": 8 + header_bytes + start,
                "size": end - start,
                "dtype": entry["dtype"],
                "shape": list(entry["shape"]),
            }
    require(set(catalog) == set(index["weight_map"]), "source catalog/index tensor mismatch")
    return catalog


def identity_mappings(config: dict[str, Any]) -> dict[int, dict[int, int]]:
    pattern = config["hybrid_override_pattern"]
    experts = config["n_routed_experts"]
    return {
        layer: {expert: expert for expert in range(experts)}
        for layer, kind in enumerate(pattern)
        if kind == "E"
    }


def destination_expert_name(match: re.Match[str]) -> str:
    projection = "fc1" if match.group(4) == "up_proj" else "fc2"
    suffix = {
        "weight": "weight",
        "weight_scale": "scales",
        "weight_scale_2": "global_scales",
        "input_scale": "input_scales",
    }[match.group(5)]
    return f"{match.group(1)}.switch_mlp.{projection}.{suffix}"


def append_tensor(
    output: dict[str, dict[str, Any]],
    name: str,
    dtype: str,
    shape: list[int],
    segments: list[dict[str, Any]],
) -> None:
    require(name not in output, f"duplicate runtime tensor: {name}")
    expected = 1
    for dimension in shape:
        expected *= dimension
    item_bytes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "U8": 1}.get(dtype)
    require(item_bytes is not None, f"unsupported runtime dtype {dtype}: {name}")
    total_bytes = sum(segment["size"] for segment in segments)
    require(expected * item_bytes == total_bytes, f"runtime tensor size mismatch: {name}")
    output[name] = {"dtype": dtype, "shape": shape, "segments": segments, "size": total_bytes}


def build_groups(
    catalog: dict[str, dict[str, Any]],
    config: dict[str, Any],
    mappings: dict[int, dict[int, int]],
    omit_mtp: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {
        "global": {},
        **{f"layer-{layer:03d}": {} for layer in range(config["num_hidden_layers"])},
    }
    expert_parts: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    old_experts = config["n_routed_experts"]

    for name, source in catalog.items():
        if omit_mtp and name.startswith("mtp."):
            continue
        expert_match = EXPERT_RE.fullmatch(name)
        if expert_match:
            layer = int(expert_match.group(2))
            old_expert = int(expert_match.group(3))
            require(layer in mappings, f"expert tensor in unplanned layer: {name}")
            new_expert = mappings[layer].get(old_expert)
            if new_expert is not None:
                destination = destination_expert_name(expert_match)
                expert_parts.setdefault(destination, []).append((new_expert, source))
            continue

        layer_match = LAYER_RE.match(name)
        group = f"layer-{int(layer_match.group(1)):03d}" if layer_match else "global"
        router_match = ROUTER_RE.fullmatch(name)
        if router_match:
            layer = int(router_match.group(1))
            require(source["shape"] and source["shape"][0] == old_experts, f"router shape mismatch: {name}")
            row_bytes = source["size"] // old_experts
            require(row_bytes * old_experts == source["size"], f"router row bytes mismatch: {name}")
            kept = sorted(mappings[layer], key=mappings[layer].get)
            segments = [
                {**source, "offset": source["offset"] + expert * row_bytes, "size": row_bytes}
                for expert in kept
            ]
            shape = [len(kept), *source["shape"][1:]]
            append_tensor(groups[group], name, source["dtype"], shape, segments)
        else:
            append_tensor(groups[group], name, source["dtype"], source["shape"], [source])

    for destination, parts in expert_parts.items():
        parts.sort(key=lambda item: item[0])
        layer_match = LAYER_RE.match(destination)
        require(layer_match is not None, f"stacked expert has no layer: {destination}")
        layer = int(layer_match.group(1))
        expected_experts = len(mappings[layer])
        require([index for index, _ in parts] == list(range(expected_experts)), f"expert stack is incomplete: {destination}")
        first = parts[0][1]
        require(
            all(part["dtype"] == first["dtype"] and part["shape"] == first["shape"] for _, part in parts),
            f"expert stack metadata mismatch: {destination}",
        )
        shape = [expected_experts, *first["shape"]]
        if destination.endswith((".global_scales", ".input_scales")) and first["shape"] == [1]:
            shape = [expected_experts]
        append_tensor(
            groups[f"layer-{layer:03d}"],
            destination,
            first["dtype"],
            shape,
            [part for _, part in parts],
        )
    require(all(group for group in groups.values()), "runtime pack produced an empty group")
    return groups


def encode_group(tensors: dict[str, dict[str, Any]]) -> tuple[bytes, int]:
    header: dict[str, Any] = {"__metadata__": {"format": FORMAT}}
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        header[name] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + tensor["size"]],
        }
        offset += tensor["size"]
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return encoded, offset


def copy_bytes(source: Any, output: Any, offset: int, size: int, digest: Any) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        require(chunk, "unexpected EOF while packing runtime tensor")
        output.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def payload_sha256(path: Path, header_bytes: int, payload_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(8 + header_bytes)
        remaining = payload_bytes
        while remaining:
            chunk = handle.read(min(remaining, COPY_CHUNK_BYTES))
            require(chunk, f"unexpected EOF while verifying {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def write_group(path: Path, tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    encoded, payload_bytes = encode_group(tensors)
    temporary = path.with_name(path.name + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    source_paths = sorted({segment["path"] for tensor in tensors.values() for segment in tensor["segments"]})
    try:
        with ExitStack() as stack:
            sources = {source_path: stack.enter_context(source_path.open("rb")) for source_path in source_paths}
            output = stack.enter_context(temporary.open("wb"))
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for name in sorted(tensors):
                for segment in tensors[name]["segments"]:
                    copy_bytes(sources[segment["path"]], output, segment["offset"], segment["size"], digest)
            output.flush()
            os.fsync(output.fileno())
        expected_size = 8 + len(encoded) + payload_bytes
        require(temporary.stat().st_size == expected_size, "runtime group file size mismatch")
        _, header_bytes, actual_payload = read_safetensors_header(temporary)
        require(actual_payload == payload_bytes, "runtime group payload mismatch")
        require(payload_sha256(temporary, header_bytes, payload_bytes) == digest.hexdigest(), "runtime group digest mismatch")
        temporary.replace(path)
        return {
            "status": "done",
            "bytes": expected_size,
            "payload_bytes": payload_bytes,
            "tensors": len(tensors),
            "payload_sha256": digest.hexdigest(),
        }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_done(path: Path, record: dict[str, Any]) -> None:
    require(path.is_file() and path.stat().st_size == record["bytes"], f"completed runtime group changed: {path.name}")
    _, header_bytes, payload_bytes = read_safetensors_header(path)
    require(payload_bytes == record["payload_bytes"], f"completed runtime payload changed: {path.name}")
    require(payload_sha256(path, header_bytes, payload_bytes) == record["payload_sha256"], f"completed runtime digest changed: {path.name}")


def runtime_config(config: dict[str, Any], new_experts: int, omit_mtp: bool) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("quantization_config", None)
    result["n_routed_experts"] = new_experts
    if omit_mtp:
        result["num_nextn_predict_layers"] = 0
    result["nemotron_runtime"] = {
        "format": FORMAT,
        "expert_layout": "stacked-modelopt-nvfp4-u8",
        "global_scale_application": "fold-into-activation",
        "mtp_omitted": omit_mtp,
    }
    return result


def finalize(
    source_dir: Path,
    output_dir: Path,
    group_names: list[str],
    groups: dict[str, dict[str, dict[str, Any]]],
    config: dict[str, Any],
    new_experts: int,
    omit_mtp: bool,
    source_revision: str,
    plan_sha256: str,
) -> dict[str, Any]:
    weight_map: dict[str, str] = {}
    total_size = 0
    for group_name in group_names:
        filename = f"{group_name}.safetensors"
        total_size += sum(tensor["size"] for tensor in groups[group_name].values())
        for name in groups[group_name]:
            require(name not in weight_map, f"duplicate final runtime tensor: {name}")
            weight_map[name] = filename
    for path in source_dir.iterdir():
        if not path.is_file() or SHARD_RE.fullmatch(path.name):
            continue
        if path.name in {"config.json", "hf_quant_config.json", "model.safetensors.index.json"}:
            continue
        shutil.copy2(path, output_dir / path.name)
    atomic_json(output_dir / "config.json", runtime_config(config, new_experts, omit_mtp))
    atomic_json(output_dir / "model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    report = {
        "format": FORMAT,
        "status": "complete",
        "source_revision": source_revision,
        "plan_sha256": plan_sha256,
        "experts": new_experts,
        "mtp_omitted": omit_mtp,
        "groups": len(group_names),
        "tensors": len(weight_map),
        "payload_bytes": total_size,
        "payload_gib": total_size / 2**30,
        "retained_payloads": "byte-identical",
    }
    atomic_json(output_dir / "nemotron_mlx_pack_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--omit-mtp", action="store_true")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.source_dir.resolve() != args.output_dir.resolve(), "source and output are identical")
        require(args.max_groups is None or args.max_groups > 0, "max-groups must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        index = load_json(args.source_dir / "model.safetensors.index.json")
        catalog = source_catalog(args.source_dir, index)
        if args.plan:
            plan = load_json(args.plan)
            mappings = validate_plan(plan, config, source_state["revision"])
            new_experts = plan["new_num_experts"]
            plan_digest = sha256_file(args.plan)
        else:
            mappings = identity_mappings(config)
            new_experts = config["n_routed_experts"]
            plan_digest = "unpruned"
        groups = build_groups(catalog, config, mappings, args.omit_mtp)
        group_names = ["global", *[f"layer-{layer:03d}" for layer in range(config["num_hidden_layers"])]]
        projected_payload = sum(tensor["size"] for group in groups.values() for tensor in group.values())
        projected_tensors = sum(len(group) for group in groups.values())
        message = (
            f"projected: experts={config['n_routed_experts']}->{new_experts} "
            f"groups={len(group_names)} tensors={projected_tensors} "
            f"payload={projected_payload / 2**30:.2f}GiB omit_mtp={args.omit_mtp}"
        )
        print(message)
        if args.dry_run:
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "pack.log")
        operation_log.write(f"run-start source_revision={source_state['revision']} plan_sha256={plan_digest}")
        operation_log.write(message)
        state_path = args.output_dir / "pack-state.json"
        identity = {
            "format": STATE_FORMAT,
            "source_dir": str(args.source_dir.resolve()),
            "source_revision": source_state["revision"],
            "plan_sha256": plan_digest,
            "experts": new_experts,
            "omit_mtp": args.omit_mtp,
        }
        if state_path.exists():
            state = load_json(state_path)
            for key, value in identity.items():
                require(state.get(key) == value, f"runtime pack state identity mismatch: {key}")
        else:
            require(not list(args.output_dir.glob("*.safetensors*")), "runtime output has shards without state")
            state = {**identity, "status": "running", "groups": {}}
            atomic_json(state_path, state)
        remaining = projected_payload - sum(
            record.get("payload_bytes", 0)
            for record in state["groups"].values()
            if record.get("status") == "done"
        )
        require(shutil.disk_usage(args.output_dir).free >= remaining + 2 * 2**30, "insufficient disk with 2 GiB safety margin")

        processed = 0
        for group_name in group_names:
            output_path = args.output_dir / f"{group_name}.safetensors"
            record = state["groups"].get(group_name)
            if record and record.get("status") == "done":
                validate_done(output_path, record)
                operation_log.write(f"group-skip group={group_name} status=verified")
                continue
            if args.max_groups is not None and processed >= args.max_groups:
                break
            operation_log.write(f"group-start group={group_name} tensors={len(groups[group_name])}")
            state["groups"][group_name] = {"status": "processing"}
            atomic_json(state_path, state)
            result = write_group(output_path, groups[group_name])
            state["groups"][group_name] = result
            atomic_json(state_path, state)
            processed += 1
            operation_log.write(
                f"group-done group={group_name} tensors={result['tensors']} "
                f"bytes={result['bytes']} sha256={result['payload_sha256']}"
            )

        if all(state["groups"].get(group, {}).get("status") == "done" for group in group_names):
            report = finalize(
                args.source_dir,
                args.output_dir,
                group_names,
                groups,
                config,
                new_experts,
                args.omit_mtp,
                source_state["revision"],
                plan_digest,
            )
            state["status"] = "complete"
            state["report"] = report
            atomic_json(state_path, state)
            operation_log.write(f"run-complete payload_gib={report['payload_gib']:.2f} validation=passed")
        else:
            operation_log.write(f"run-paused processed={processed}")
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        if operation_log:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron MLX pack error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

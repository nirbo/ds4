#!/usr/bin/env python3
"""Materialize an exact, structurally pruned Nemotron NVFP4 checkpoint."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nemotron_metadata import MetadataError, load_json, require, validate_metadata
from nemotron_safetensors_inventory import build_inventory, read_safetensors_header


STATE_FORMAT = "nemotron-prune-materialize-state-v1"
REPORT_FORMAT = "nemotron-prune-materialize-report-v1"
PLAN_FORMAT = "nemotron-prune-plan-v1"
SOURCE_STATE_FORMAT = "nemotron-source-snapshot-v1"
COPY_CHUNK_BYTES = 16 * 1024 * 1024
EXPERT_NAME_RE = re.compile(
    r"^(backbone\.layers\.(\d+)\.mixer\.experts\.)(\d+)(\..+)$"
)
ROUTER_NAME_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.gate\."
    r"(weight|e_score_correction_bias)$"
)
SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


class OperationLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{timestamp} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_source_state(path: Path, source_dir: Path) -> dict[str, Any]:
    state = load_json(path)
    require(state.get("format") == SOURCE_STATE_FORMAT, "unsupported source state")
    require(state.get("verification", {}).get("result") == "passed", "source snapshot is not verified")
    require(Path(state.get("path", "")).resolve() == source_dir.resolve(), "source state path mismatch")
    return state


def validate_plan(
    plan: dict[str, Any],
    config: dict[str, Any],
    source_revision: str | None,
    *,
    require_revision: bool = True,
) -> dict[int, dict[int, int]]:
    require(plan.get("format") == PLAN_FORMAT, "unsupported prune plan format")
    old_experts = config.get("n_routed_experts")
    require(plan.get("old_num_experts") == old_experts, "plan source expert count mismatch")
    new_experts = plan.get("new_num_experts")
    require(isinstance(new_experts, int) and 0 < new_experts <= old_experts, "invalid retained expert count")
    require(config.get("num_experts_per_tok", 0) <= new_experts, "retained experts are below router top-k")
    require(config.get("n_group") == 1, "materializer currently requires n_group=1")
    if require_revision:
        require(isinstance(source_revision, str) and source_revision, "source revision is unavailable")
        require(plan.get("source_revision") == source_revision, "plan/source revision mismatch")

    pattern = config.get("hybrid_override_pattern")
    require(isinstance(pattern, str), "source config has no layer pattern")
    expected_layers = [layer for layer, layer_type in enumerate(pattern) if layer_type == "E"]
    model_layers = plan.get("model_moe_layers")
    require(model_layers == expected_layers, "plan MoE layer list mismatch")
    kept_by_layer = plan.get("kept_by_layer")
    require(isinstance(kept_by_layer, dict), "plan has no kept_by_layer map")

    mappings: dict[int, dict[int, int]] = {}
    for layer in expected_layers:
        kept = kept_by_layer.get(str(layer))
        require(isinstance(kept, list), f"plan has no retained experts for layer {layer}")
        require(len(kept) == new_experts, f"layer {layer} retained count mismatch")
        require(all(isinstance(expert, int) for expert in kept), f"layer {layer} has non-integer expert IDs")
        require(kept == sorted(set(kept)), f"layer {layer} retained experts must be sorted and unique")
        require(kept and kept[0] >= 0 and kept[-1] < old_experts, f"layer {layer} expert ID out of range")
        mappings[layer] = {old: new for new, old in enumerate(kept)}

    recorded = plan.get("old_to_new_by_layer")
    if recorded is not None:
        require(isinstance(recorded, dict), "invalid old_to_new_by_layer map")
        for layer, mapping in mappings.items():
            expected = {str(old): new for old, new in mapping.items()}
            require(recorded.get(str(layer)) == expected, f"layer {layer} recorded remap mismatch")
    return mappings


def remap_name(name: str, mappings: dict[int, dict[int, int]], omit_mtp: bool) -> str | None:
    if omit_mtp and name.startswith("mtp."):
        return None
    match = EXPERT_NAME_RE.fullmatch(name)
    if not match:
        return name
    layer = int(match.group(2))
    old_expert = int(match.group(3))
    require(layer in mappings, f"expert tensor belongs to unplanned layer {layer}: {name}")
    new_expert = mappings[layer].get(old_expert)
    if new_expert is None:
        return None
    return f"{match.group(1)}{new_expert}{match.group(4)}"


def read_raw_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            require(len(prefix) == 8, f"truncated safetensors prefix: {path}")
            size = struct.unpack("<Q", prefix)[0]
            header = json.loads(handle.read(size))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot read safetensors header {path}: {exc}") from exc
    require(isinstance(header, dict), f"invalid safetensors header: {path}")
    return header, size


def build_shard_plan(
    source_path: Path,
    mappings: dict[int, dict[int, int]],
    old_experts: int,
    omit_mtp: bool,
) -> dict[str, Any]:
    source_header, source_header_bytes = read_raw_header(source_path)
    metadata = source_header.get("__metadata__")
    source_tensors = [
        (name, entry)
        for name, entry in source_header.items()
        if name != "__metadata__"
    ]
    source_tensors.sort(key=lambda item: item[1]["data_offsets"][0])

    items = []
    output_header: dict[str, Any] = {}
    if metadata is not None:
        output_header["__metadata__"] = metadata
    output_offset = 0
    for source_name, source_entry in source_tensors:
        destination_name = remap_name(source_name, mappings, omit_mtp)
        if destination_name is None:
            continue
        require(destination_name not in output_header, f"duplicate remapped tensor: {destination_name}")
        source_start, source_end = source_entry["data_offsets"]
        segments = [(source_start, source_end - source_start)]
        destination_shape = list(source_entry["shape"])

        router_match = ROUTER_NAME_RE.fullmatch(source_name)
        if router_match:
            layer = int(router_match.group(1))
            require(layer in mappings, f"router belongs to unplanned layer {layer}")
            require(destination_shape and destination_shape[0] == old_experts, f"router shape mismatch: {source_name}")
            row_bytes = (source_end - source_start) // old_experts
            require(row_bytes * old_experts == source_end - source_start, f"router row size mismatch: {source_name}")
            kept = sorted(mappings[layer], key=mappings[layer].get)
            segments = [(source_start + old * row_bytes, row_bytes) for old in kept]
            destination_shape[0] = len(kept)

        output_size = sum(size for _, size in segments)
        output_header[destination_name] = {
            "dtype": source_entry["dtype"],
            "shape": destination_shape,
            "data_offsets": [output_offset, output_offset + output_size],
        }
        items.append(
            {
                "source_name": source_name,
                "destination_name": destination_name,
                "segments": segments,
                "output_bytes": output_size,
            }
        )
        output_offset += output_size

    if not items:
        return {
            "source_header_bytes": source_header_bytes,
            "output_header": {},
            "encoded_header": b"",
            "items": [],
            "payload_bytes": 0,
            "file_bytes": 0,
            "tensor_count": 0,
            "omitted": True,
        }
    encoded = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return {
        "source_header_bytes": source_header_bytes,
        "output_header": output_header,
        "encoded_header": encoded,
        "items": items,
        "payload_bytes": output_offset,
        "file_bytes": 8 + len(encoded) + output_offset,
        "tensor_count": len(items),
        "omitted": False,
    }


def copy_segment(
    source_handle: Any,
    output_handle: Any,
    source_offset: int,
    size: int,
    digest: Any,
) -> None:
    source_handle.seek(source_offset)
    remaining = size
    while remaining:
        chunk = source_handle.read(min(COPY_CHUNK_BYTES, remaining))
        require(chunk, "unexpected EOF while copying tensor payload")
        output_handle.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def hash_payload(path: Path, header_bytes: int, payload_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(8 + header_bytes)
        remaining = payload_bytes
        while remaining:
            chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
            require(chunk, f"unexpected EOF while validating {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def materialize_shard(source_path: Path, output_path: Path, shard_plan: dict[str, Any]) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    temporary.unlink(missing_ok=True)
    expected_digest = hashlib.sha256()
    source_payload_base = 8 + shard_plan["source_header_bytes"]
    encoded_header = shard_plan["encoded_header"]

    try:
        with source_path.open("rb") as source, temporary.open("wb") as output:
            output.write(struct.pack("<Q", len(encoded_header)))
            output.write(encoded_header)
            for item in shard_plan["items"]:
                for relative_offset, size in item["segments"]:
                    copy_segment(source, output, source_payload_base + relative_offset, size, expected_digest)
            output.flush()
            os.fsync(output.fileno())

        require(temporary.stat().st_size == shard_plan["file_bytes"], "materialized shard size mismatch")
        tensors, header_bytes, payload_bytes = read_safetensors_header(temporary)
        require(len(tensors) == shard_plan["tensor_count"], "materialized tensor count mismatch")
        require(payload_bytes == shard_plan["payload_bytes"], "materialized payload size mismatch")
        actual_digest = hash_payload(temporary, header_bytes, payload_bytes)
        require(actual_digest == expected_digest.hexdigest(), "materialized payload digest mismatch")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "status": "done",
        "source_file_bytes": source_path.stat().st_size,
        "output_file_bytes": output_path.stat().st_size,
        "output_payload_bytes": shard_plan["payload_bytes"],
        "output_tensors": shard_plan["tensor_count"],
        "payload_sha256": expected_digest.hexdigest(),
    }


def remap_quantized_layers(
    layers: dict[str, Any], mappings: dict[int, dict[int, int]], omit_mtp: bool
) -> dict[str, Any]:
    result = {}
    for name, value in layers.items():
        remapped = remap_name(name, mappings, omit_mtp)
        if remapped is not None:
            require(remapped not in result, f"duplicate remapped quantized layer: {remapped}")
            result[remapped] = value
    return result


def transform_config(
    config: dict[str, Any],
    mappings: dict[int, dict[int, int]],
    new_experts: int,
    omit_mtp: bool,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["n_routed_experts"] = new_experts
    if omit_mtp:
        result["num_nextn_predict_layers"] = 0
    quantization = result.get("quantization_config")
    require(isinstance(quantization, dict), "config has no quantization_config")
    groups = quantization.get("config_groups")
    require(isinstance(groups, dict), "config has no quantization groups")
    for group in groups.values():
        targets = group.get("targets")
        require(isinstance(targets, list), "quantization group has no targets")
        remapped_targets = []
        for target in targets:
            remapped = remap_name(target, mappings, omit_mtp)
            if remapped is not None:
                remapped_targets.append(remapped)
        group["targets"] = remapped_targets
    layers = quantization.get("quantized_layers")
    require(isinstance(layers, dict), "config has no quantized_layers map")
    quantization["quantized_layers"] = remap_quantized_layers(layers, mappings, omit_mtp)
    return result


def transform_hf_quant_config(
    config: dict[str, Any], mappings: dict[int, dict[int, int]], omit_mtp: bool
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    quantization = result.get("quantization")
    require(isinstance(quantization, dict), "hf_quant_config has no quantization object")
    layers = quantization.get("quantized_layers")
    require(isinstance(layers, dict), "hf_quant_config has no quantized_layers map")
    quantization["quantized_layers"] = remap_quantized_layers(layers, mappings, omit_mtp)
    return result


def initialize_state(
    state_path: Path,
    source_dir: Path,
    output_dir: Path,
    source_state: dict[str, Any],
    plan_path: Path,
    omit_mtp: bool,
) -> dict[str, Any]:
    identity = {
        "format": STATE_FORMAT,
        "source_dir": str(source_dir.resolve()),
        "source_revision": source_state["revision"],
        "output_dir": str(output_dir.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "omit_mtp": omit_mtp,
    }
    if state_path.exists():
        state = load_json(state_path)
        for key, value in identity.items():
            require(state.get(key) == value, f"existing state identity mismatch: {key}")
        require(isinstance(state.get("shards"), dict), "existing state has no shard map")
        return state
    foreign_outputs = list(output_dir.glob("model-*.safetensors")) + list(output_dir.glob("model-*.safetensors.part"))
    require(not foreign_outputs, "output directory has shard files but no matching state")
    state = {**identity, "status": "running", "shards": {}}
    atomic_json(state_path, state)
    return state


def collect_output_index(output_dir: Path, shards: list[str]) -> tuple[dict[str, str], int]:
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        tensors, _, payload_bytes = read_safetensors_header(output_dir / shard)
        total_size += payload_bytes
        for name in tensors:
            require(name not in weight_map, f"duplicate output tensor: {name}")
            weight_map[name] = shard
    return weight_map, total_size


def finalize_artifact(
    source_dir: Path,
    output_dir: Path,
    shards: list[str],
    config: dict[str, Any],
    hf_quant_config: dict[str, Any],
    plan: dict[str, Any],
    mappings: dict[int, dict[int, int]],
    omit_mtp: bool,
    state: dict[str, Any],
) -> dict[str, Any]:
    weight_map, total_size = collect_output_index(output_dir, shards)
    transformed_config = transform_config(config, mappings, plan["new_num_experts"], omit_mtp)
    transformed_hf_quant = transform_hf_quant_config(hf_quant_config, mappings, omit_mtp)

    config_layers = transformed_config["quantization_config"]["quantized_layers"]
    group_targets = [
        target
        for group in transformed_config["quantization_config"]["config_groups"].values()
        for target in group["targets"]
    ]
    hf_layers = transformed_hf_quant["quantization"]["quantized_layers"]
    require(len(group_targets) == len(set(group_targets)), "duplicate ModelOpt group target after remap")
    require(set(group_targets) == set(config_layers), "ModelOpt target/layer map mismatch after remap")
    require(set(config_layers) == set(hf_layers), "config/hf_quant_config layer map mismatch after remap")

    for source_path in source_dir.iterdir():
        if not source_path.is_file() or SHARD_RE.fullmatch(source_path.name):
            continue
        if source_path.name in {"config.json", "hf_quant_config.json", "model.safetensors.index.json"}:
            continue
        shutil.copy2(source_path, output_dir / source_path.name)

    atomic_json(output_dir / "config.json", transformed_config)
    atomic_json(output_dir / "hf_quant_config.json", transformed_hf_quant)
    atomic_json(
        output_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    atomic_json(output_dir / "nemotron_prune_plan.json", plan)
    output_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    validate_metadata(transformed_config, output_index, strict_target=False)
    output_inventory = build_inventory(
        output_dir,
        transformed_config,
        output_index,
        strict_target=False,
    )
    require(output_inventory["totals"]["payload_bytes"] == total_size, "final inventory payload mismatch")
    if omit_mtp:
        require("mtp" not in output_inventory["roles"], "MTP tensors remain in omit-MTP artifact")
    report = {
        "format": REPORT_FORMAT,
        "source_revision": state["source_revision"],
        "plan_sha256": state["plan_sha256"],
        "omit_mtp": omit_mtp,
        "old_num_experts": config["n_routed_experts"],
        "new_num_experts": plan["new_num_experts"],
        "shards": len(shards),
        "tensors": len(weight_map),
        "payload_bytes": total_size,
        "payload_gib": total_size / 2**30,
        "quantized_layers": len(transformed_config["quantization_config"]["quantized_layers"]),
        "validation": {
            "metadata": "passed",
            "safetensors": "passed",
            "modelopt_maps": "passed",
            "retained_payload_sha256_per_shard": "passed",
        },
        "status": "complete",
    }
    atomic_json(output_dir / "nemotron_materialization_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--omit-mtp", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        source_resolved = args.source_dir.resolve()
        output_resolved = args.output_dir.resolve()
        require(source_resolved != output_resolved, "source and output directories are identical")
        require(source_resolved not in output_resolved.parents, "output directory is inside immutable source")
        require(output_resolved not in source_resolved.parents, "output directory contains immutable source")
        require(args.max_shards is None or args.max_shards > 0, "max-shards must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        index = load_json(args.source_dir / "model.safetensors.index.json")
        hf_quant_config = load_json(args.source_dir / "hf_quant_config.json")
        plan = load_json(args.plan)
        mappings = validate_plan(plan, config, source_state["revision"])
        shards = sorted(set(index["weight_map"].values()))

        projected_payload = 0
        projected_files = 0
        projected_tensors = 0
        shard_plans = {}
        for shard in shards:
            shard_plan = build_shard_plan(
                args.source_dir / shard,
                mappings,
                config["n_routed_experts"],
                args.omit_mtp,
            )
            shard_plans[shard] = shard_plan
            projected_payload += shard_plan["payload_bytes"]
            projected_files += shard_plan["file_bytes"]
            projected_tensors += shard_plan["tensor_count"]

        output_sources = [shard for shard in shards if not shard_plans[shard]["omitted"]]
        output_names = {
            source_shard: f"model-{position:05d}-of-{len(output_sources):05d}.safetensors"
            for position, source_shard in enumerate(output_sources, start=1)
        }

        projection_message = (
            f"projected: experts={config['n_routed_experts']}->{plan['new_num_experts']} "
            f"shards={len(shards)}->{len(output_sources)} tensors={projected_tensors} "
            f"payload={projected_payload / 2**30:.2f}GiB files={projected_files / 2**30:.2f}GiB "
            f"omit_mtp={args.omit_mtp}"
        )
        print(projection_message)
        if args.dry_run:
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.log or (args.output_dir / "materialize.log"))
        operation_log.write(
            f"run-start source_revision={source_state['revision']} "
            f"plan_sha256={sha256_file(args.plan)} output={args.output_dir}"
        )
        operation_log.write(projection_message)
        state_path = args.output_dir / "materialize-state.json"
        state = initialize_state(
            state_path,
            args.source_dir,
            args.output_dir,
            source_state,
            args.plan,
            args.omit_mtp,
        )
        free_bytes = shutil.disk_usage(args.output_dir).free
        existing_output_bytes = sum(
            (args.output_dir / record["output_name"]).stat().st_size
            for shard, record in state["shards"].items()
            if record.get("status") == "done"
            and isinstance(record.get("output_name"), str)
            and (args.output_dir / record["output_name"]).is_file()
        )
        remaining_bytes = max(0, projected_files - existing_output_bytes)
        require(free_bytes >= remaining_bytes + 2 * 2**30, "insufficient disk space with 2 GiB safety margin")

        processed = 0
        for shard in shards:
            record = state["shards"].get(shard)
            if shard_plans[shard]["omitted"]:
                require(record is None or record.get("status") == "omitted", f"omitted shard state mismatch: {shard}")
                if record is None:
                    state["shards"][shard] = {"status": "omitted", "reason": "all tensors excluded"}
                    atomic_json(state_path, state)
                    operation_log.write(f"shard-omitted source={shard} reason=all-tensors-excluded")
                continue
            output_name = output_names[shard]
            output_path = args.output_dir / output_name
            if record and record.get("status") == "done":
                require(record.get("output_name") == output_name, f"completed shard output name changed: {shard}")
                require(output_path.is_file(), f"completed shard is missing: {output_name}")
                require(output_path.stat().st_size == record["output_file_bytes"], f"completed shard size changed: {output_name}")
                operation_log.write(f"shard-skip source={shard} output={output_name} status=verified")
                continue
            if args.max_shards is not None and processed >= args.max_shards:
                break
            operation_log.write(f"shard-start source={shard} output={output_name}")
            state["shards"][shard] = {"status": "processing", "output_name": output_name}
            atomic_json(state_path, state)
            result = materialize_shard(args.source_dir / shard, output_path, shard_plans[shard])
            result["output_name"] = output_name
            state["shards"][shard] = result
            atomic_json(state_path, state)
            processed += 1
            operation_log.write(
                f"shard-done source={shard} output={output_name} "
                f"tensors={result['output_tensors']} bytes={result['output_file_bytes']} "
                f"sha256={result['payload_sha256']}"
            )

        complete = all(
            state["shards"].get(shard, {}).get("status") in {"done", "omitted"}
            for shard in shards
        )
        if complete:
            report = finalize_artifact(
                args.source_dir,
                args.output_dir,
                [output_names[shard] for shard in output_sources],
                config,
                hf_quant_config,
                plan,
                mappings,
                args.omit_mtp,
                state,
            )
            state["status"] = "complete"
            state["report"] = report
            atomic_json(state_path, state)
            operation_log.write(
                f"run-complete payload_gib={report['payload_gib']:.2f} "
                f"tensors={report['tensors']} validation=passed"
            )
        else:
            operation_log.write(f"run-paused processed={processed}")
        return 0
    except (MetadataError, OSError) as exc:
        message = f"nemotron materialization error: {exc}"
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

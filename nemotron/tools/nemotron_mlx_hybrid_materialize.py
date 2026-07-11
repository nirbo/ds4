#!/usr/bin/env python3
"""Incrementally materialize a validated width layer over an MLX runtime."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import sha256_file
from nemotron_mlx_mamba import layer_tensors
from nemotron_mlx_moe import slice_expert_blocks
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_paged_embeddings import write_catalog
from nemotron_mlx_stream_forward import validate_virtual_hybrid_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state
from nemotron_safetensors_inventory import read_safetensors_header


FORMAT = "nemotron-mlx-hybrid-runtime-v1"
RUNTIME_FORMAT = "nemotron-mlx-runtime-v1"
EXCLUDED = {
    "config.json",
    "model.safetensors.index.json",
    "nemotron_mlx_pack_report.json",
    "pack-state.json",
    "pack.log",
    "nemotron_paged_embedding_catalog.json",
}


def tensor_bytes(tensors: dict[str, mx.array]) -> int:
    return sum(value.nbytes for value in tensors.values())


def load_expert_input_scales(source_dir: Path, layer: int, projection: str) -> mx.array:
    index = load_json(source_dir / "model.safetensors.index.json")
    names = [
        f"backbone.layers.{layer}.mixer.experts.{expert}.{projection}.input_scale"
        for expert in range(512)
    ]
    shards = {}
    values = []
    for name in names:
        shard = index["weight_map"].get(name)
        require(isinstance(shard, str), f"missing expert input scale: {name}")
        if shard not in shards:
            shards[shard] = mx.load(str(source_dir / shard))
        values.append(shards[shard][name].reshape(()).astype(mx.float32))
    result = mx.stack(values)
    mx.eval(result)
    return result


def replacement_layer(source_dir: Path, layer: int, blocks: np.ndarray) -> dict[str, mx.array]:
    block = load_moe_layer(source_dir, layer)
    narrowed = slice_expert_blocks(block.experts, blocks)
    tensors = layer_tensors(source_dir, layer)
    base = f"backbone.layers.{layer}.mixer.switch_mlp"
    tensors.update(
        {
            f"{base}.fc1.weight": narrowed.up.weight,
            f"{base}.fc1.scales": narrowed.up.scales,
            f"{base}.fc1.global_scales": narrowed.up.global_scales,
            f"{base}.fc1.input_scales": load_expert_input_scales(source_dir, layer, "up_proj"),
            f"{base}.fc2.weight": narrowed.down.weight,
            f"{base}.fc2.scales": narrowed.down.scales,
            f"{base}.fc2.global_scales": narrowed.down.global_scales,
            f"{base}.fc2.input_scales": load_expert_input_scales(source_dir, layer, "down_proj"),
        }
    )
    mx.eval(*tensors.values())
    return tensors


def transformed_config(base: dict, layer: int, width: int, plan_sha256: str) -> dict:
    result = copy.deepcopy(base)
    runtime = result.get("nemotron_runtime")
    require(isinstance(runtime, dict) and runtime.get("format") == RUNTIME_FORMAT, "invalid base runtime config")
    runtime["experts_by_layer"][str(layer)] = 512
    runtime["nonuniform_experts"] = True
    runtime["expert_width_by_layer"] = {str(layer): width}
    runtime["hybrid_plan_sha256"] = plan_sha256
    runtime["hybrid_format"] = FORMAT
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--hybrid-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    staging = args.output_dir.with_name(args.output_dir.name + ".part")
    try:
        require(not args.output_dir.exists(), "hybrid output already exists")
        require(not staging.exists(), "hybrid staging directory already exists")
        source_state = load_source_state(args.source_state, args.source_dir)
        source_config = load_json(args.source_dir / "config.json")
        plan = load_json(args.hybrid_plan)
        retained, width = validate_virtual_hybrid_plan(plan, source_config, source_state["revision"])
        require(len(width) == 1, "incremental materializer currently requires one width layer")
        layer_key, blocks = next(iter(width.items()))
        layer = int(layer_key)
        require(layer_key not in retained, "hybrid layer cannot also prune experts")
        base_config = load_json(args.base_runtime / "config.json")
        base_report = load_json(args.base_runtime / "nemotron_mlx_pack_report.json")
        base_index = load_json(args.base_runtime / "model.safetensors.index.json")
        require(base_report.get("status") == "complete", "base runtime is incomplete")
        plan_digest = sha256_file(args.hybrid_plan)
        staging.mkdir(parents=True)
        operation_log = OperationLog(staging / "materialize.log")
        operation_log.write(
            f"run-start layer={layer} width={blocks.shape[1] * 16} plan_sha256={plan_digest}"
        )
        replacement_name = f"layer-{layer:03d}.safetensors"
        linked = 0
        for source in args.base_runtime.iterdir():
            if not source.is_file() or source.name in EXCLUDED or source.name == replacement_name:
                continue
            os.link(source, staging / source.name)
            linked += 1
        operation_log.write(f"hardlinks-done files={linked}")

        tensors = replacement_layer(args.source_dir, layer, blocks)
        replacement_part = staging / (replacement_name + ".part.safetensors")
        mx.save_safetensors(
            str(replacement_part),
            tensors,
            metadata={"format": RUNTIME_FORMAT, "hybrid_plan_sha256": plan_digest},
        )
        replacement_path = staging / replacement_name
        replacement_part.replace(replacement_path)
        header, _, payload = read_safetensors_header(replacement_path)
        require(set(header) - {"__metadata__"} == set(tensors), "replacement tensor catalog mismatch")
        require(payload == tensor_bytes(tensors), "replacement payload size mismatch")
        operation_log.write(
            f"layer-write-done layer={layer} payload={payload} sha256={sha256_file(replacement_path)}"
        )

        index = copy.deepcopy(base_index)
        old_layer_payload = 0
        old_tensors = mx.load(str(args.base_runtime / replacement_name))
        old_layer_payload = tensor_bytes(old_tensors)
        index["metadata"]["total_size"] += payload - old_layer_payload
        atomic_json(staging / "model.safetensors.index.json", index)
        width_value = blocks.shape[1] * 16
        atomic_json(staging / "config.json", transformed_config(base_config, layer, width_value, plan_digest))
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "base_runtime": str(args.base_runtime.resolve()),
            "base_report_sha256": sha256_file(args.base_runtime / "nemotron_mlx_pack_report.json"),
            "hybrid_plan_sha256": plan_digest,
            "width_layer": layer,
            "width": width_value,
            "payload_bytes": index["metadata"]["total_size"],
            "payload_gib": index["metadata"]["total_size"] / 2**30,
            "replacement_payload_sha256": sha256_file(replacement_path),
            "unchanged_files": "hard-linked",
            "retained_width_payloads": "byte-identical",
        }
        atomic_json(staging / "nemotron_mlx_hybrid_report.json", report)
        atomic_json(
            staging / "nemotron_mlx_pack_report.json",
            {**report, "format": RUNTIME_FORMAT, "hybrid_format": FORMAT},
        )
        write_catalog(staging)

        physical = load_moe_layer(staging, layer)
        require(physical.experts.up.output_dims == width_value, "physical expert width mismatch")
        require(physical.experts.up.experts == 512, "physical expert count mismatch")
        for name, expected in tensors.items():
            actual = mx.load(str(replacement_path))[name]
            equal = mx.all(actual == expected)
            mx.eval(equal)
            require(bool(equal), f"physical tensor differs after save: {name}")
        operation_log.write("physical-validation-done tensors=exact")
        staging.replace(args.output_dir)
        print(
            f"hybrid-materialize output={args.output_dir} payload={report['payload_gib']:.4f}GiB "
            f"layer={layer} width={width_value}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron hybrid materialize error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

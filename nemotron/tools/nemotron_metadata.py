#!/usr/bin/env python3
"""Validate and catalog the pinned Nemotron 3 Super NVFP4 metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATALOG_FORMAT = "nemotron-metadata-catalog-v1"
SOURCE_FORMAT = "nemotron-source-metadata-v1"
EXPECTED_ARCHITECTURE = "NemotronHForCausalLM"
EXPECTED_LAYER_COUNTS = {"M": 40, "E": 40, "*": 8}
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_LAYER_COUNT = 88
EXPECTED_EXPERTS = 512
EXPECTED_EXPERT_TOP_K = 22
EXPECTED_MOE_LATENT_SIZE = 1024
EXPECTED_MOE_INTERMEDIATE_SIZE = 2688
EXPECTED_PAYLOAD_BYTES = 80_297_329_824
EXPECTED_SHARDS = 17
EXPECTED_TENSORS = 165_860

EXPERT_TENSOR_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.(input_scale|weight|weight_scale|weight_scale_2)$"
)
BACKBONE_LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.")
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")

EXPECTED_EXPERT_MEMBERS = {
    f"{projection}.{member}"
    for projection in ("up_proj", "down_proj")
    for member in ("input_scale", "weight", "weight_scale", "weight_scale_2")
}


class MetadataError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MetadataError(f"expected a JSON object in {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MetadataError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_manifest(
    manifest_path: Path, config_path: Path, index_path: Path
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    require(manifest.get("format") == SOURCE_FORMAT, "unsupported source manifest")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "source manifest has no files")

    base = manifest_path.parent
    for name, expected_hash in files.items():
        require(isinstance(name, str), "source manifest file name is not a string")
        require(
            isinstance(expected_hash, str) and len(expected_hash) == 64,
            f"invalid SHA-256 for {name}",
        )
        path = base / name
        require(path.is_file(), f"source metadata file is missing: {path}")
        actual_hash = sha256_file(path)
        require(actual_hash == expected_hash, f"source metadata hash mismatch: {name}")

    require(
        config_path.resolve() == (base / "config.json").resolve(),
        "config path does not belong to source manifest",
    )
    require(
        index_path.resolve() == (base / "model.safetensors.index.json").resolve(),
        "index path does not belong to source manifest",
    )
    return manifest


def validate_shards(weight_map: dict[str, Any], expected_count: int) -> list[str]:
    require(weight_map, "weight map is empty")
    require(
        all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()),
        "weight map keys and shard names must be strings",
    )
    shards = sorted(set(weight_map.values()))
    require(len(shards) == expected_count, f"expected {expected_count} shards, got {len(shards)}")
    for position, shard in enumerate(shards, start=1):
        match = SHARD_RE.fullmatch(shard)
        require(match is not None, f"invalid shard name: {shard}")
        require(int(match.group(1)) == position, f"non-contiguous shard sequence at {shard}")
        require(int(match.group(2)) == expected_count, f"wrong shard total in {shard}")
    return shards


def classify_tensor(name: str) -> str:
    if EXPERT_TENSOR_RE.fullmatch(name):
        return "backbone_routed_expert"
    if re.fullmatch(
        r"backbone\.layers\.\d+\.mixer\.gate\."
        r"(weight|e_score_correction_bias)",
        name,
    ):
        return "backbone_router"
    if name.startswith("backbone.layers."):
        return "backbone_other"
    if name.startswith("mtp."):
        return "mtp"
    return "top_level"


def validate_metadata(
    config: dict[str, Any],
    index: dict[str, Any],
    *,
    strict_target: bool = True,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    architectures = config.get("architectures")
    require(
        isinstance(architectures, list)
        and len(architectures) == 1
        and isinstance(architectures[0], str),
        "config must declare exactly one architecture",
    )
    architecture = architectures[0]
    layer_count = config.get("num_hidden_layers")
    pattern = config.get("hybrid_override_pattern")
    require(isinstance(layer_count, int) and layer_count > 0, "invalid layer count")
    require(isinstance(pattern, str), "missing hybrid layer pattern")
    require(len(pattern) == layer_count, "hybrid layer pattern length mismatch")
    require(set(pattern) <= {"M", "E", "*"}, "hybrid layer pattern has unknown layer type")

    routed_experts = config.get("n_routed_experts")
    expert_top_k = config.get("num_experts_per_tok")
    require(isinstance(routed_experts, int) and routed_experts > 0, "invalid routed expert count")
    require(isinstance(expert_top_k, int) and 0 < expert_top_k <= routed_experts, "invalid expert top-k")

    quantization = config.get("quantization_config")
    require(isinstance(quantization, dict), "missing quantization config")
    require(quantization.get("quant_method") == "modelopt", "expected ModelOpt quantization")
    require(quantization.get("quant_algo") == "MIXED_PRECISION", "expected mixed precision")

    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    require(isinstance(metadata, dict), "index metadata is missing")
    require(isinstance(weight_map, dict), "index weight_map is missing")
    total_size = metadata.get("total_size")
    require(isinstance(total_size, int) and total_size > 0, "invalid indexed payload size")

    expected_shards = EXPECTED_SHARDS if strict_target else len(set(weight_map.values()))
    shards = validate_shards(weight_map, expected_shards)

    layer_types = Counter(pattern)
    layer_tensor_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    shard_tensor_counts: Counter[str] = Counter(weight_map.values())
    expert_members: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    for name in weight_map:
        role_counts[classify_tensor(name)] += 1
        layer_match = BACKBONE_LAYER_RE.match(name)
        if layer_match:
            layer = int(layer_match.group(1))
            require(0 <= layer < layer_count, f"tensor references invalid layer {layer}: {name}")
            layer_tensor_counts[layer] += 1

        expert_match = EXPERT_TENSOR_RE.fullmatch(name)
        if expert_match:
            layer = int(expert_match.group(1))
            expert = int(expert_match.group(2))
            member = f"{expert_match.group(3)}.{expert_match.group(4)}"
            require(pattern[layer] == "E", f"expert tensor found on non-MoE layer {layer}")
            require(0 <= expert < routed_experts, f"invalid expert {expert} on layer {layer}")
            expert_members[layer][expert].add(member)

    moe_layers = [layer for layer, layer_type in enumerate(pattern) if layer_type == "E"]
    for layer in moe_layers:
        experts = expert_members.get(layer, {})
        require(len(experts) == routed_experts, f"layer {layer} has {len(experts)} routed experts")
        for expert in range(routed_experts):
            members = experts.get(expert, set())
            require(
                members == EXPECTED_EXPERT_MEMBERS,
                f"layer {layer} expert {expert} member mismatch: "
                f"missing={sorted(EXPECTED_EXPERT_MEMBERS - members)} "
                f"extra={sorted(members - EXPECTED_EXPERT_MEMBERS)}",
            )
        for router_name in (
            f"backbone.layers.{layer}.mixer.gate.weight",
            f"backbone.layers.{layer}.mixer.gate.e_score_correction_bias",
        ):
            require(router_name in weight_map, f"missing router tensor: {router_name}")

    non_moe_expert_layers = sorted(set(expert_members) - set(moe_layers))
    require(not non_moe_expert_layers, f"expert tensors on non-MoE layers: {non_moe_expert_layers}")

    if strict_target:
        require(architecture == EXPECTED_ARCHITECTURE, f"unexpected architecture: {architecture}")
        require(layer_count == EXPECTED_LAYER_COUNT, f"expected {EXPECTED_LAYER_COUNT} layers")
        require(dict(layer_types) == EXPECTED_LAYER_COUNTS, f"unexpected layer mix: {dict(layer_types)}")
        require(config.get("hidden_size") == EXPECTED_HIDDEN_SIZE, "unexpected hidden size")
        require(routed_experts == EXPECTED_EXPERTS, "unexpected routed expert count")
        require(expert_top_k == EXPECTED_EXPERT_TOP_K, "unexpected expert top-k")
        require(config.get("moe_latent_size") == EXPECTED_MOE_LATENT_SIZE, "unexpected MoE latent size")
        require(
            config.get("moe_intermediate_size") == EXPECTED_MOE_INTERMEDIATE_SIZE,
            "unexpected MoE intermediate size",
        )
        require(total_size == EXPECTED_PAYLOAD_BYTES, "unexpected indexed payload size")
        require(len(weight_map) == EXPECTED_TENSORS, "unexpected indexed tensor count")

    if source is not None:
        require(source.get("weight_payload_bytes") == total_size, "manifest/index payload mismatch")
        require(source.get("weight_shards") == len(shards), "manifest/index shard mismatch")

    return {
        "format": CATALOG_FORMAT,
        "source": {
            "repository": source.get("repository") if source else None,
            "revision": source.get("revision") if source else None,
        },
        "model": {
            "architecture": architecture,
            "hidden_size": config.get("hidden_size"),
            "layer_count": layer_count,
            "layer_pattern": pattern,
            "layer_type_counts": {key: layer_types[key] for key in ("M", "E", "*")},
            "routed_experts": routed_experts,
            "experts_per_token": expert_top_k,
            "moe_latent_size": config.get("moe_latent_size"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
            "quant_method": quantization.get("quant_method"),
            "quant_algorithm": quantization.get("quant_algo"),
        },
        "weights": {
            "payload_bytes": total_size,
            "tensor_count": len(weight_map),
            "shard_count": len(shards),
            "shards": shards,
            "tensor_role_counts": dict(sorted(role_counts.items())),
            "shard_tensor_counts": {shard: shard_tensor_counts[shard] for shard in shards},
        },
        "moe_layers": [
            {
                "layer": layer,
                "routed_experts": len(expert_members[layer]),
                "expert_tensor_count": sum(len(members) for members in expert_members[layer].values()),
                "router_tensors": [
                    f"backbone.layers.{layer}.mixer.gate.weight",
                    f"backbone.layers.{layer}.mixer.gate.e_score_correction_bias",
                ],
            }
            for layer in moe_layers
        ],
        "layer_tensor_counts": {
            str(layer): layer_tensor_counts[layer] for layer in range(layer_count)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--no-strict-target",
        action="store_true",
        help="validate structure without enforcing the pinned official target constants",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = None
        if args.source_manifest:
            source = validate_source_manifest(args.source_manifest, args.config, args.index)
        catalog = validate_metadata(
            load_json(args.config),
            load_json(args.index),
            strict_target=not args.no_strict_target,
            source=source,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.out.with_name(args.out.name + ".part")
            temporary.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.out)

        model = catalog["model"]
        weights = catalog["weights"]
        roles = weights["tensor_role_counts"]
        print(
            f"nemotron metadata ok: architecture={model['architecture']} "
            f"layers={model['layer_type_counts']} experts={model['routed_experts']} "
            f"top_k={model['experts_per_token']}"
        )
        print(
            f"weights: tensors={weights['tensor_count']} shards={weights['shard_count']} "
            f"bytes={weights['payload_bytes']} GiB={weights['payload_bytes'] / 2**30:.2f}"
        )
        print(
            f"roles: routed_expert={roles.get('backbone_routed_expert', 0)} "
            f"router={roles.get('backbone_router', 0)} mtp={roles.get('mtp', 0)}"
        )
        if args.out:
            print(f"catalog: {args.out}")
        return 0
    except MetadataError as exc:
        print(f"nemotron metadata error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

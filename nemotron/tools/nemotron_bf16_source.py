#!/usr/bin/env python3
"""Build strict, revision-bound contracts for Nemotron BF16 expert layers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-bf16-layer-contract-v1"
STATE_FORMAT = "nemotron-bf16-metadata-state-v1"
EXPECTED_REPOSITORY = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
EXPECTED_REVISION = "d51eab0d1f979ebc26b546e634a04f450d99158e"
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
EXPERT_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.weight$"
)


def _validated_json_file(metadata_dir: Path, state: dict, name: str) -> dict:
    entry = state.get("files", {}).get(name)
    require(isinstance(entry, dict), f"BF16 metadata state has no {name} entry")
    path = metadata_dir / name
    require(path.is_file(), f"missing BF16 metadata file: {path}")
    require(path.stat().st_size == entry.get("bytes"), f"BF16 metadata size mismatch: {name}")
    require(sha256_file(path) == entry.get("sha256"), f"BF16 metadata hash mismatch: {name}")
    return load_json(path)


def load_metadata(metadata_dir: Path, state_path: Path) -> tuple[dict, dict, dict]:
    """Validate the immutable small-file snapshot and target architecture."""

    state = load_json(state_path)
    require(state.get("format") == STATE_FORMAT, "unsupported BF16 metadata state")
    require(state.get("repository") == EXPECTED_REPOSITORY, "unexpected BF16 repository")
    require(state.get("revision") == EXPECTED_REVISION, "unexpected BF16 revision")
    config = _validated_json_file(metadata_dir, state, "config.json")
    index = _validated_json_file(metadata_dir, state, "model.safetensors.index.json")

    require(config.get("architectures") == ["NemotronHForCausalLM"], "unexpected BF16 architecture")
    require(config.get("model_type") == "nemotron_h", "unexpected BF16 model type")
    require(config.get("dtype") == "bfloat16", "BF16 source does not declare bfloat16")
    require("quantization_config" not in config, "BF16 source unexpectedly declares quantization")
    pattern = config.get("hybrid_override_pattern")
    require(
        isinstance(pattern, str) and len(pattern) == config.get("num_hidden_layers"),
        "invalid BF16 layer pattern",
    )
    require(Counter(pattern) == Counter({"M": 40, "E": 40, "*": 8}), "unexpected BF16 layer mix")
    require(config.get("n_routed_experts") == 512, "unexpected BF16 routed-expert count")
    require(config.get("num_experts_per_tok") == 22, "unexpected BF16 router top-k")
    require(config.get("moe_latent_size") == 1024, "unexpected BF16 expert input width")
    require(config.get("moe_intermediate_size") == 2688, "unexpected BF16 expert hidden width")

    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "BF16 index has no weight map")
    require(
        index.get("metadata", {}).get("total_size") == state.get("indexed_payload_bytes"),
        "BF16 indexed payload mismatch",
    )
    require(len(weight_map) == state.get("indexed_tensors"), "BF16 indexed tensor-count mismatch")
    shards = sorted(set(weight_map.values()))
    require(len(shards) == state.get("indexed_shards"), "BF16 indexed shard-count mismatch")
    totals = set()
    ordinals = set()
    for shard in shards:
        match = SHARD_RE.fullmatch(shard)
        require(match is not None, f"invalid BF16 shard name: {shard}")
        ordinals.add(int(match.group(1)))
        totals.add(int(match.group(2)))
    require(totals == {len(shards)}, "BF16 shard names disagree on total")
    require(ordinals == set(range(1, len(shards) + 1)), "BF16 shard sequence is incomplete")
    return state, config, index


def _state_shard_entry(state: dict, shard: str) -> dict:
    entry = state.get("shards", {}).get(shard)
    if entry is None:
        entry = state.get("representative_layer_1_shards", {}).get(shard)
    require(isinstance(entry, dict), f"BF16 metadata state has no immutable identity for {shard}")
    size = entry.get("bytes")
    digest = entry.get("sha256")
    require(isinstance(size, int) and size > 0, f"invalid BF16 shard size: {shard}")
    require(
        isinstance(digest, str) and len(digest) == 64,
        f"invalid BF16 shard SHA-256: {shard}",
    )
    return {"bytes": size, "sha256": digest}


def build_layer_contract(
    metadata_dir: Path,
    state_path: Path,
    layer: int,
) -> dict[str, Any]:
    state, config, index = load_metadata(metadata_dir, state_path)
    pattern = config["hybrid_override_pattern"]
    require(0 <= layer < len(pattern), f"BF16 layer index is out of range: {layer}")
    require(pattern[layer] == "E", f"BF16 layer {layer} is not LatentMoE")

    experts = config["n_routed_experts"]
    latent = config["moe_latent_size"]
    hidden = config["moe_intermediate_size"]
    weight_map = index["weight_map"]
    tensor_map: dict[str, str] = {}
    expert_shards: dict[int, dict[str, str]] = defaultdict(dict)
    prefix = f"backbone.layers.{layer}.mixer.experts"
    for expert in range(experts):
        for projection in ("up_proj", "down_proj"):
            name = f"{prefix}.{expert}.{projection}.weight"
            shard = weight_map.get(name)
            require(isinstance(shard, str), f"missing BF16 expert tensor: {name}")
            tensor_map[name] = shard
            expert_shards[expert][projection] = shard

    indexed_names = {name for name in weight_map if name.startswith(prefix + ".")}
    require(indexed_names == set(tensor_map), f"unexpected BF16 expert tensor set in layer {layer}")
    required_shards = sorted(set(tensor_map.values()))
    shard_entries = {shard: _state_shard_entry(state, shard) for shard in required_shards}
    split_experts = [
        expert
        for expert, projections in expert_shards.items()
        if projections["up_proj"] != projections["down_proj"]
    ]
    up_bytes = hidden * latent * 2
    down_bytes = latent * hidden * 2
    expert_bytes = up_bytes + down_bytes
    source_expert_bytes = experts * expert_bytes

    # Affine storage includes packed codes plus BF16 scale and bias per group.
    def projected_bytes(bits: int, group_size: int = 128) -> int:
        require((latent * bits) % 8 == 0 and (hidden * bits) % 8 == 0, "unaligned projection")
        up = experts * hidden * (latent * bits // 8 + latent // group_size * 4)
        down = experts * latent * (hidden * bits // 8 + hidden // group_size * 4)
        return up + down

    projections = {}
    for bits in (1, 2, 3, 4):
        payload = projected_bytes(bits)
        projections[str(bits)] = {
            "bytes": payload,
            "gib": payload / 2**30,
            "fraction_of_bf16": payload / source_expert_bytes,
        }
    mixed_1b_nvfp4 = []
    # Each projection has one F32 global scale per expert, not one per row.
    nvfp4_expert_bytes = (
        hidden * (latent // 2 + latent // 16)
        + latent * (hidden // 2 + hidden // 16)
        + 2 * 4
    )
    for protected in (64, 128, 192, 256, 320, 384):
        binary = projections["1"]["bytes"] * (experts - protected) // experts
        native = nvfp4_expert_bytes * protected
        total = binary + native
        mixed_1b_nvfp4.append(
            {
                "binary_experts": experts - protected,
                "native_nvfp4_experts": protected,
                "bytes": total,
                "gib": total / 2**30,
                "average_code_bits": (experts - protected + 4 * protected) / experts,
            }
        )

    return {
        "format": FORMAT,
        "status": "ready",
        "repository": state["repository"],
        "source_revision": state["revision"],
        "metadata": {
            "directory": str(metadata_dir.resolve()),
            "state": str(state_path.resolve()),
            "state_sha256": sha256_file(state_path),
            "config_sha256": state["files"]["config.json"]["sha256"],
            "index_sha256": state["files"]["model.safetensors.index.json"]["sha256"],
            "tool_sha256": sha256_file(Path(__file__)),
        },
        "layer": layer,
        "architecture": {
            "experts": experts,
            "top_k": config["num_experts_per_tok"],
            "latent_width": latent,
            "hidden_width": hidden,
            "activation": config.get("mlp_hidden_act"),
        },
        "source_expert_payload": {
            "tensors": len(tensor_map),
            "bytes": source_expert_bytes,
            "gib": source_expert_bytes / 2**30,
            "bytes_per_expert": expert_bytes,
        },
        "required_shards": [
            {"name": shard, **shard_entries[shard]} for shard in required_shards
        ],
        "required_download": {
            "shards": len(required_shards),
            "bytes": sum(value["bytes"] for value in shard_entries.values()),
            "gib": sum(value["bytes"] for value in shard_entries.values()) / 2**30,
        },
        "split_experts": split_experts,
        "expert_shards": {
            str(expert): expert_shards[expert] for expert in range(experts)
        },
        "tensor_map": tensor_map,
        "uniform_affine_projection": projections,
        "storage_units": {
            "binary_affine_bytes_per_expert": projections["1"]["bytes"] // experts,
            "native_nvfp4_bytes_per_expert": nvfp4_expert_bytes,
        },
        "mixed_binary_native_projection": mixed_1b_nvfp4,
    }


def validate_local_shards(contract: dict, raw_dir: Path, hash_payloads: bool = True) -> None:
    require(contract.get("format") == FORMAT, "unsupported BF16 layer contract")
    for entry in contract.get("required_shards", []):
        path = raw_dir / entry["name"]
        require(path.is_file(), f"missing BF16 source shard: {path}")
        require(path.stat().st_size == entry["bytes"], f"BF16 source shard size mismatch: {path}")
        if hash_payloads:
            require(sha256_file(path) == entry["sha256"], f"BF16 source shard hash mismatch: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--metadata-state", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--size-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = build_layer_contract(args.metadata_dir, args.metadata_state, args.layer)
        if args.raw_dir is not None:
            validate_local_shards(contract, args.raw_dir, hash_payloads=not args.size_only)
            contract["local_validation"] = {
                "directory": str(args.raw_dir.resolve()),
                "hash_payloads": not args.size_only,
                "result": "passed",
            }
        if args.output is not None:
            atomic_json(args.output, contract)
        print(json.dumps({
            "layer": contract["layer"],
            "required_download": contract["required_download"],
            "source_expert_payload": contract["source_expert_payload"],
            "uniform_affine_projection": contract["uniform_affine_projection"],
            "split_experts": contract["split_experts"],
        }, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, KeyError) as exc:
        print(f"nemotron BF16 source error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

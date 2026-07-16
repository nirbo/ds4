#!/usr/bin/env python3
"""Validate and catalog the pinned Ornith 35B NVFP4 metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATALOG_FORMAT = "ornith35-metadata-catalog-v1"
SOURCE_STATE_FORMAT = "ornith35-source-metadata-v1"
EXPECTED_REPOSITORY = "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
EXPECTED_REVISION = "85ffd2d0629ae5fa4f860dda356ec33161806c9b"
EXPECTED_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
EXPECTED_WEIGHT_FILE_BYTES = 23_741_821_016
EXPECTED_WEIGHT_SHA256 = "68a4b2b8605076825302be20132cf69342b44a0385c19e6de741af5ec3114ca0"
EXPECTED_TENSOR_COUNT = 93_346
EXPECTED_LAYER_COUNT = 40
EXPECTED_LINEAR_LAYERS = 30
EXPECTED_FULL_LAYERS = 10
EXPECTED_HIDDEN_SIZE = 2_048
EXPECTED_EXPERTS = 256
EXPECTED_EXPERT_TOP_K = 8
EXPECTED_EXPERT_INTERMEDIATE = 512
EXPECTED_NATIVE_CONTEXT = 262_144
EXPECTED_ROPE_THETA = 10_000_000
EXPECTED_MROPE_SECTION = [11, 11, 10]

DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "U8": 1,
}
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


class MetadataError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MetadataError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_nbytes(dtype: str, shape: Any) -> int:
    require(dtype in DTYPE_BYTES, f"unsupported safetensors dtype: {dtype}")
    require(
        isinstance(shape, list)
        and all(isinstance(dimension, int) and dimension >= 0 for dimension in shape),
        f"invalid tensor shape: {shape}",
    )
    return math.prod(shape) * DTYPE_BYTES[dtype]


def validate_config(config: dict[str, Any], *, strict_target: bool = True) -> dict[str, Any]:
    architectures = config.get("architectures")
    require(
        isinstance(architectures, list)
        and len(architectures) == 1
        and isinstance(architectures[0], str),
        "config must declare exactly one architecture",
    )
    text = config.get("text_config")
    require(isinstance(text, dict), "missing text_config")
    layer_types = text.get("layer_types")
    require(isinstance(layer_types, list) and layer_types, "missing layer_types")
    require(
        set(layer_types) <= {"linear_attention", "full_attention"},
        "unknown text layer type",
    )
    layer_counts = Counter(layer_types)

    quant = config.get("quantization_config")
    require(isinstance(quant, dict), "missing quantization_config")
    groups = quant.get("config_groups")
    require(isinstance(groups, dict) and len(groups) == 1, "expected one quantization group")
    group = next(iter(groups.values()))
    require(isinstance(group, dict), "invalid quantization group")
    weights = group.get("weights")
    require(isinstance(weights, dict), "missing weight quantization settings")
    require(group.get("input_activations") is None, "expected weight-only quantization")

    if strict_target:
        require(architectures[0] == EXPECTED_ARCHITECTURE, "unexpected architecture")
        require(config.get("model_type") == "qwen3_5_moe", "unexpected model type")
        require(len(layer_types) == EXPECTED_LAYER_COUNT, "unexpected layer count")
        require(layer_counts["linear_attention"] == EXPECTED_LINEAR_LAYERS, "unexpected linear layer count")
        require(layer_counts["full_attention"] == EXPECTED_FULL_LAYERS, "unexpected full layer count")
        require(
            all(
                layer_type == ("full_attention" if (index + 1) % 4 == 0 else "linear_attention")
                for index, layer_type in enumerate(layer_types)
            ),
            "unexpected hybrid layer order",
        )
        require(text.get("hidden_size") == EXPECTED_HIDDEN_SIZE, "unexpected hidden size")
        require(text.get("num_experts") == EXPECTED_EXPERTS, "unexpected expert count")
        require(text.get("num_experts_per_tok") == EXPECTED_EXPERT_TOP_K, "unexpected expert top-k")
        require(text.get("moe_intermediate_size") == EXPECTED_EXPERT_INTERMEDIATE, "unexpected expert width")
        require(text.get("shared_expert_intermediate_size") == EXPECTED_EXPERT_INTERMEDIATE, "unexpected shared expert width")
        require(text.get("num_attention_heads") == 16, "unexpected query head count")
        require(text.get("num_key_value_heads") == 2, "unexpected KV head count")
        require(text.get("head_dim") == 256, "unexpected attention head dimension")
        require(text.get("linear_num_key_heads") == 16, "unexpected linear key heads")
        require(text.get("linear_num_value_heads") == 32, "unexpected linear value heads")
        require(text.get("linear_key_head_dim") == 128, "unexpected linear key dimension")
        require(text.get("linear_value_head_dim") == 128, "unexpected linear value dimension")
        require(text.get("max_position_embeddings") == EXPECTED_NATIVE_CONTEXT, "unexpected native context")
        require(text.get("partial_rotary_factor") == 0.25, "unexpected partial rotary factor")
        rope = text.get("rope_parameters")
        require(isinstance(rope, dict), "missing rope_parameters")
        require(rope.get("rope_type") == "default", "unexpected native RoPE type")
        require(rope.get("rope_theta") == EXPECTED_ROPE_THETA, "unexpected RoPE theta")
        require(rope.get("mrope_interleaved") is True, "expected interleaved mRoPE")
        require(rope.get("mrope_section") == EXPECTED_MROPE_SECTION, "unexpected mRoPE sections")
        require(quant.get("quant_method") == "compressed-tensors", "unexpected quantization method")
        require(quant.get("format") == "nvfp4-pack-quantized", "unexpected quantization format")
        require(group.get("format") == "nvfp4-pack-quantized", "unexpected group format")
        require(weights.get("num_bits") == 4, "unexpected quantization bits")
        require(weights.get("group_size") == 16, "unexpected NVFP4 group size")
        require(weights.get("type") == "float", "expected floating-point quantization")
        require(weights.get("scale_dtype") == "torch.float8_e4m3fn", "unexpected scale dtype")

    return {
        "architecture": architectures[0],
        "hidden_size": text.get("hidden_size"),
        "layer_count": len(layer_types),
        "layer_types": layer_types,
        "layer_type_counts": dict(sorted(layer_counts.items())),
        "experts": text.get("num_experts"),
        "experts_per_token": text.get("num_experts_per_tok"),
        "native_context": text.get("max_position_embeddings"),
        "mtp_configured_layers": text.get("mtp_num_hidden_layers", 0),
        "quantization": {
            "method": quant.get("quant_method"),
            "format": quant.get("format"),
            "bits": weights.get("num_bits"),
            "group_size": weights.get("group_size"),
            "weight_only": group.get("input_activations") is None,
        },
    }


def classify_tensor(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name == "model.language_model.embed_tokens.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm_head"
    if name.startswith("mtp."):
        return "mtp"
    if ".mlp.experts." in name:
        return "routed_expert"
    if ".mlp.shared_expert." in name:
        return "shared_expert"
    if name.endswith(".mlp.gate.weight") or ".mlp.shared_expert_gate." in name:
        return "router"
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "full_attention"
    if "layernorm" in name or name == "model.language_model.norm.weight":
        return "norm"
    if name.startswith("model.language_model.layers."):
        return "text_other"
    return "top_level"


def validate_nvfp4_mlp(entries: dict[str, dict[str, Any]], text: dict[str, Any]) -> int:
    """Require every routed/shared MLP projection and its scales exactly once."""
    hidden = text["hidden_size"]
    intermediate = text["moe_intermediate_size"]
    require(hidden % 2 == 0 and hidden % 16 == 0, "unsupported hidden width for NVFP4")
    require(
        intermediate % 2 == 0 and intermediate % 16 == 0,
        "unsupported expert width for NVFP4",
    )
    projection_specs = {
        "down_proj": {
            "weight_global_scale": ("F32", [1]),
            "weight_scale": ("F8_E4M3", [hidden, intermediate // 16]),
            "weight_packed": ("U8", [hidden, intermediate // 2]),
        },
        "gate_proj": {
            "weight_global_scale": ("F32", [1]),
            "weight_scale": ("F8_E4M3", [intermediate, hidden // 16]),
            "weight_packed": ("U8", [intermediate, hidden // 2]),
        },
        "up_proj": {
            "weight_global_scale": ("F32", [1]),
            "weight_scale": ("F8_E4M3", [intermediate, hidden // 16]),
            "weight_packed": ("U8", [intermediate, hidden // 2]),
        },
    }
    expected_names: set[str] = set()
    for layer in range(len(text["layer_types"])):
        prefixes = [
            f"model.language_model.layers.{layer}.mlp.experts.{expert}"
            for expert in range(text["num_experts"])
        ]
        prefixes.append(f"model.language_model.layers.{layer}.mlp.shared_expert")
        for prefix in prefixes:
            for projection, tensors in projection_specs.items():
                for suffix, (dtype, shape) in tensors.items():
                    name = f"{prefix}.{projection}.{suffix}"
                    expected_names.add(name)
                    entry = entries.get(name)
                    require(isinstance(entry, dict), f"missing NVFP4 tensor: {name}")
                    require(entry.get("dtype") == dtype, f"NVFP4 dtype mismatch: {name}")
                    require(entry.get("shape") == shape, f"NVFP4 shape mismatch: {name}")

    actual_names = {
        name
        for name in entries
        if ".mlp.experts." in name or ".mlp.shared_expert." in name
    }
    unexpected = sorted(actual_names - expected_names)
    missing = sorted(expected_names - actual_names)
    require(not unexpected, f"unexpected NVFP4 tensor: {unexpected[0] if unexpected else ''}")
    require(not missing, f"missing NVFP4 tensor: {missing[0] if missing else ''}")
    return len(expected_names)


def context_profile(config: dict[str, Any], tokens: int, *, kv_bytes: int) -> dict[str, Any]:
    text = config["text_config"]
    full_layers = text["layer_types"].count("full_attention")
    bytes_per_token = (
        full_layers
        * 2
        * text["num_key_value_heads"]
        * text["head_dim"]
        * kv_bytes
    )
    attention_pairs = tokens * (tokens + 1) // 2
    attention_flops = (
        full_layers
        * 4
        * text["num_attention_heads"]
        * text["head_dim"]
        * attention_pairs
    )
    return {
        "tokens": tokens,
        "kv_bytes_per_token": bytes_per_token,
        "kv_cache_bytes": bytes_per_token * tokens,
        "kv_cache_gib": bytes_per_token * tokens / 2**30,
        "cold_attention_flops": attention_flops,
        "cold_attention_pflops": attention_flops / 1e15,
    }


def validate_source_state(
    state_path: Path, config_path: Path, header_path: Path
) -> dict[str, Any]:
    state = load_json(state_path)
    require(state.get("format") == SOURCE_STATE_FORMAT, "unsupported source state")
    require(state.get("repository") == EXPECTED_REPOSITORY, "source repository mismatch")
    require(state.get("revision") == EXPECTED_REVISION, "source revision mismatch")
    files = state.get("metadata_files")
    require(isinstance(files, dict), "source state has no metadata file hashes")
    for path in (config_path, header_path):
        entry = files.get(path.name)
        require(isinstance(entry, dict), f"source state does not cover {path.name}")
        require(entry.get("bytes") == path.stat().st_size, f"source size mismatch: {path.name}")
        require(entry.get("sha256") == sha256_file(path), f"source hash mismatch: {path.name}")
    return state


def build_catalog(
    config: dict[str, Any],
    header: dict[str, Any],
    source_state: dict[str, Any],
    *,
    strict_target: bool = True,
) -> dict[str, Any]:
    model = validate_config(config, strict_target=strict_target)
    weight = source_state.get("weight")
    require(isinstance(weight, dict), "source state has no weight metadata")
    file_bytes = weight.get("file_bytes")
    header_bytes = weight.get("header_bytes")
    payload_bytes = weight.get("payload_bytes")
    require(all(isinstance(value, int) and value > 0 for value in (file_bytes, header_bytes, payload_bytes)), "invalid weight sizes")
    require(file_bytes == 8 + header_bytes + payload_bytes, "weight size decomposition mismatch")

    entries: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            require(isinstance(entry, dict), "invalid safetensors metadata")
            continue
        require(isinstance(entry, dict), f"invalid tensor header: {name}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(offset, int) for offset in offsets),
            f"invalid tensor offsets: {name}",
        )
        start, end = offsets
        require(0 <= start <= end, f"invalid tensor interval: {name}")
        require(end - start == tensor_nbytes(dtype, shape), f"tensor payload mismatch: {name}")
        entries[name] = entry
        intervals.append((start, end, name))

    require(entries, "safetensors header has no tensors")
    cursor = 0
    for start, end, name in sorted(intervals):
        require(start == cursor, f"non-contiguous payload before {name}")
        cursor = end
    require(cursor == payload_bytes, "header payload total mismatch")

    role_bytes: Counter[str] = Counter()
    role_tensors: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    dtype_tensors: Counter[str] = Counter()
    expert_ids: dict[int, set[int]] = defaultdict(set)
    layer_tensor_counts: Counter[int] = Counter()
    for name, entry in entries.items():
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        role = classify_tensor(name)
        role_bytes[role] += size
        role_tensors[role] += 1
        dtype_bytes[entry["dtype"]] += size
        dtype_tensors[entry["dtype"]] += 1
        layer_match = LAYER_RE.match(name)
        if layer_match:
            layer = int(layer_match.group(1))
            require(0 <= layer < model["layer_count"], f"invalid layer in tensor: {name}")
            layer_tensor_counts[layer] += 1
        expert_match = EXPERT_RE.match(name)
        if expert_match:
            expert_ids[int(expert_match.group(1))].add(int(expert_match.group(2)))

    for layer in range(model["layer_count"]):
        require(layer_tensor_counts[layer] > 0, f"layer {layer} has no tensors")
        require(
            expert_ids[layer] == set(range(model["experts"])),
            f"layer {layer} routed expert coverage mismatch",
        )

    if strict_target:
        require(file_bytes == EXPECTED_WEIGHT_FILE_BYTES, "unexpected weight file size")
        require(weight.get("sha256") == EXPECTED_WEIGHT_SHA256, "unexpected weight SHA-256")
        require(len(entries) == EXPECTED_TENSOR_COUNT, "unexpected tensor count")
        require(role_tensors["mtp"] == 0, "released Ornith target unexpectedly contains MTP tensors")
        require(
            validate_nvfp4_mlp(entries, config["text_config"]) == 92_520,
            "unexpected NVFP4 MLP tensor count",
        )
        for name, entry in entries.items():
            if ".mlp.experts." not in name and ".mlp.shared_expert." not in name:
                require(entry["dtype"] == "BF16", f"unexpected non-MLP dtype: {name}")

    text_payload = payload_bytes - role_bytes["vision"]
    text = config["text_config"]
    recurrent_state_bytes = (
        text["layer_types"].count("linear_attention")
        * text["linear_num_value_heads"]
        * text["linear_key_head_dim"]
        * text["linear_value_head_dim"]
        * 4
    )
    conv_state_bytes = (
        text["layer_types"].count("linear_attention")
        * (text["linear_conv_kernel_dim"] - 1)
        * (
            2 * text["linear_num_key_heads"] * text["linear_key_head_dim"]
            + text["linear_num_value_heads"] * text["linear_value_head_dim"]
        )
        * 2
    )

    return {
        "format": CATALOG_FORMAT,
        "source": {
            "repository": source_state["repository"],
            "revision": source_state["revision"],
            "weight_file": weight.get("name"),
            "weight_sha256": weight.get("sha256"),
        },
        "model": model,
        "weights": {
            "file_bytes": file_bytes,
            "header_bytes": header_bytes,
            "payload_bytes": payload_bytes,
            "payload_gib": payload_bytes / 2**30,
            "tensor_count": len(entries),
            "text_payload_bytes": text_payload,
            "text_payload_gib": text_payload / 2**30,
            "vision_payload_bytes": role_bytes["vision"],
            "vision_payload_gib": role_bytes["vision"] / 2**30,
        },
        "roles": {
            role: {
                "tensors": role_tensors[role],
                "bytes": role_bytes[role],
                "gib": role_bytes[role] / 2**30,
            }
            for role in sorted(role_bytes)
        },
        "dtypes": {
            dtype: {
                "tensors": dtype_tensors[dtype],
                "bytes": dtype_bytes[dtype],
                "gib": dtype_bytes[dtype] / 2**30,
            }
            for dtype in sorted(dtype_bytes)
        },
        "cache": {
            "gated_delta_recurrent_state_bytes": recurrent_state_bytes,
            "gated_delta_conv_state_bytes": conv_state_bytes,
            "bf16_profiles": {
                "native-262k": context_profile(config, EXPECTED_NATIVE_CONTEXT, kv_bytes=2),
                "yarn2-524k": context_profile(config, EXPECTED_NATIVE_CONTEXT * 2, kv_bytes=2),
                "yarn4-1010k": context_profile(config, 1_010_000, kv_bytes=2),
            },
            "int8_profiles": {
                "native-262k": context_profile(config, EXPECTED_NATIVE_CONTEXT, kv_bytes=1),
                "yarn2-524k": context_profile(config, EXPECTED_NATIVE_CONTEXT * 2, kv_bytes=1),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-strict-target", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_json(args.config)
        header = load_json(args.header)
        state = validate_source_state(args.source_state, args.config, args.header)
        catalog = build_catalog(
            config,
            header,
            state,
            strict_target=not args.no_strict_target,
        )
    except (MetadataError, OSError) as exc:
        print(f"ornith35 metadata error: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".part")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(args.out)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the pinned Ornith-35 DSpark draft contract without its weights."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


CATALOG_FORMAT = "ornith35-dspark-catalog-v1"
SOURCE_STATE_FORMAT = "ornith35-companion-metadata-v1"
EXPECTED_REPOSITORY = (
    "pablogrant/"
    "ORNITH-1.0_35B_AEON_PABLOG-OPTIMIZED_UNCENSORED_DSPARK-DRAFT_NVFP4"
)
EXPECTED_REVISION = "9383b3c33ddf982114a4f72e07c890bfd6c35df2"
EXPECTED_WEIGHT_NAME = "model.safetensors"
EXPECTED_WEIGHT_BYTES = 1_657_168_394
EXPECTED_HEADER_BYTES = 4_608
EXPECTED_PAYLOAD_BYTES = 1_657_163_778
EXPECTED_WEIGHT_SHA256 = (
    "7ab36d46959066cbb68925239e069498f2847cd0ef4be87b08a995222ee4d06b"
)
EXPECTED_HEADER_SHA256 = (
    "ad921b86e64c4b5a8a2c9363257d0fa9f08c2fc3894f59d5afd676536fe620f7"
)
EXPECTED_METADATA_FILES = {
    "README.md": {
        "bytes": 10_061,
        "sha256": "adea06bba11115fa71d2c730a42588bf4fa89bd7d8a63dad559a1d427519fdbd",
    },
    "config.json": {
        "bytes": 1_970,
        "sha256": "151b7279861667c67d44fd71e4d847f1c3a530ae29acfe28e8a16bfaba2a73ae",
    },
    "config.py": {
        "bytes": 1_883,
        "sha256": "e51c93f57579ca6919e55ab2087ab6ca44d6674baa92527432b3ef01a1013fc0",
    },
    "model.safetensors.header.json": {
        "bytes": EXPECTED_HEADER_BYTES,
        "sha256": EXPECTED_HEADER_SHA256,
    },
    "repo-api.json": {
        "bytes": 2_888,
        "sha256": "e6153deffd9907c519dcc78605ea94cbf3bf5131e0b4491db9936cae47b66d67",
    },
    "val_metrics.json": {
        "bytes": 739,
        "sha256": "9a5b3da89f1e2e3d85e58e9cda7c0d04f69885d8bb321fd2288584b682fc3fe5",
    },
}
EXPECTED_AUX_HIDDEN_STATE_INDICES = (9, 19, 29)
EXPECTED_AUX_DECODER_LAYER_INDICES = (8, 18, 28)
EXPECTED_BLOCK_SIZE = 8
EXPECTED_SPECULATIVE_TOKENS = 7
EXPECTED_TARGET_VOCAB_SIZE = 248_320
EXPECTED_DRAFT_VOCAB_SIZE = 32_000
EXPECTED_HIDDEN_SIZE = 2_048
EXPECTED_DRAFT_LAYERS = 3
EXPECTED_MARKOV_RANK = 256

DTYPE_BYTES = {
    "BF16": 2,
    "BOOL": 1,
    "I64": 8,
}


class DSparkError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DSparkError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DSparkError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise DSparkError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def tensor_nbytes(dtype: str, shape: Any) -> int:
    require(dtype in DTYPE_BYTES, f"unsupported DSpark dtype: {dtype}")
    require(
        isinstance(shape, list)
        and all(isinstance(dimension, int) and dimension >= 0 for dimension in shape),
        f"invalid DSpark tensor shape: {shape}",
    )
    return math.prod(shape) * DTYPE_BYTES[dtype]


def expected_tensor_specs() -> dict[str, tuple[str, list[int]]]:
    specs: dict[str, tuple[str, list[int]]] = {
        "d2t": ("I64", [EXPECTED_DRAFT_VOCAB_SIZE]),
        "confidence_head.proj.bias": ("BF16", [1]),
        "confidence_head.proj.weight": (
            "BF16",
            [1, EXPECTED_HIDDEN_SIZE + EXPECTED_MARKOV_RANK],
        ),
        "embed_tokens.weight": (
            "BF16",
            [EXPECTED_TARGET_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE],
        ),
        "fc.weight": (
            "BF16",
            [EXPECTED_HIDDEN_SIZE, len(EXPECTED_AUX_HIDDEN_STATE_INDICES) * EXPECTED_HIDDEN_SIZE],
        ),
        "hidden_norm.weight": ("BF16", [EXPECTED_HIDDEN_SIZE]),
    }
    for layer in range(EXPECTED_DRAFT_LAYERS):
        prefix = f"layers.{layer}"
        specs.update(
            {
                f"{prefix}.input_layernorm.weight": ("BF16", [EXPECTED_HIDDEN_SIZE]),
                f"{prefix}.mlp.down_proj.weight": (
                    "BF16",
                    [EXPECTED_HIDDEN_SIZE, 6_144],
                ),
                f"{prefix}.mlp.gate_proj.weight": (
                    "BF16",
                    [6_144, EXPECTED_HIDDEN_SIZE],
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    "BF16",
                    [6_144, EXPECTED_HIDDEN_SIZE],
                ),
                f"{prefix}.post_attention_layernorm.weight": (
                    "BF16",
                    [EXPECTED_HIDDEN_SIZE],
                ),
                f"{prefix}.self_attn.k_norm.weight": ("BF16", [256]),
                f"{prefix}.self_attn.k_proj.weight": (
                    "BF16",
                    [512, EXPECTED_HIDDEN_SIZE],
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    "BF16",
                    [EXPECTED_HIDDEN_SIZE, 4_096],
                ),
                f"{prefix}.self_attn.q_norm.weight": ("BF16", [256]),
                f"{prefix}.self_attn.q_proj.weight": (
                    "BF16",
                    [4_096, EXPECTED_HIDDEN_SIZE],
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    "BF16",
                    [512, EXPECTED_HIDDEN_SIZE],
                ),
            }
        )
    specs.update(
        {
            "lm_head.weight": (
                "BF16",
                [EXPECTED_DRAFT_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE],
            ),
            "markov_head.markov_w1.weight": (
                "BF16",
                [EXPECTED_TARGET_VOCAB_SIZE, EXPECTED_MARKOV_RANK],
            ),
            "markov_head.markov_w2.weight": (
                "BF16",
                [EXPECTED_DRAFT_VOCAB_SIZE, EXPECTED_MARKOV_RANK],
            ),
            "norm.weight": ("BF16", [EXPECTED_HIDDEN_SIZE]),
            "t2d": ("BOOL", [EXPECTED_TARGET_VOCAB_SIZE]),
        }
    )
    return specs


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    require(config.get("architectures") == ["DSparkDraftModel"], "unexpected DSpark architecture")
    require(config.get("speculators_model_type") == "dspark", "unexpected speculator type")
    require(config.get("block_size") == EXPECTED_BLOCK_SIZE, "unexpected DSpark block size")
    require(config.get("max_anchors") == 3_072, "unexpected DSpark max anchors")
    require(
        config.get("aux_hidden_state_layer_ids") == list(EXPECTED_AUX_HIDDEN_STATE_INDICES),
        "unexpected DSpark auxiliary hidden-state indices",
    )
    require(config.get("mask_token_id") == 248_077, "unexpected DSpark mask token")
    require(
        config.get("draft_vocab_size") == EXPECTED_DRAFT_VOCAB_SIZE,
        "unexpected DSpark draft vocabulary",
    )
    require(config.get("target_hidden_size") is None, "unexpected target hidden-size override")
    require(config.get("markov_rank") == EXPECTED_MARKOV_RANK, "unexpected Markov rank")
    require(config.get("markov_head_type") == "vanilla", "unexpected Markov head type")
    require(config.get("enable_confidence_head") is True, "confidence head is disabled")
    require(
        config.get("confidence_head_with_markov") is True,
        "confidence head does not consume Markov features",
    )
    require(
        "sample_from_anchor" not in config,
        "post-release sample_from_anchor semantics cannot be applied to this checkpoint",
    )

    speculators = config.get("speculators_config")
    require(isinstance(speculators, dict), "missing speculators configuration")
    require(speculators.get("algorithm") == "dspark", "unexpected proposal algorithm")
    require(speculators.get("default_proposal_method") == "greedy", "unexpected proposal method")
    proposals = speculators.get("proposal_methods")
    require(isinstance(proposals, list) and len(proposals) == 1, "expected one proposal profile")
    proposal = proposals[0]
    require(isinstance(proposal, dict), "invalid proposal profile")
    require(proposal.get("proposal_type") == "greedy", "unexpected proposal profile type")
    require(
        proposal.get("speculative_tokens") == EXPECTED_SPECULATIVE_TOKENS,
        "unexpected speculative-token count",
    )
    require(proposal.get("verifier_accept_k") == 1, "unexpected verifier acceptance contract")
    require(proposal.get("accept_tolerance") == 0.0, "DSpark proposal is not exact greedy")
    require(
        "sample_from_anchor" not in proposal,
        "post-release proposal semantics cannot be applied to this checkpoint",
    )
    verifier = speculators.get("verifier")
    require(isinstance(verifier, dict), "missing verifier configuration")
    require(
        verifier.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"],
        "unexpected DSpark verifier architecture",
    )

    transformer = config.get("transformer_layer_config")
    require(isinstance(transformer, dict), "missing DSpark transformer configuration")
    expected_values = {
        "model_type": "qwen3",
        "vocab_size": EXPECTED_TARGET_VOCAB_SIZE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "intermediate_size": 6_144,
        "num_hidden_layers": EXPECTED_DRAFT_LAYERS,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262_144,
        "tie_word_embeddings": False,
    }
    for key, expected in expected_values.items():
        require(transformer.get(key) == expected, f"unexpected draft transformer {key}")
    require(
        transformer.get("layer_types") == ["full_attention"] * EXPECTED_DRAFT_LAYERS,
        "DSpark draft layers must all use full attention",
    )
    require(transformer.get("sliding_window") is None, "unexpected draft sliding window")
    rope = transformer.get("rope_parameters")
    require(isinstance(rope, dict), "missing DSpark RoPE configuration")
    require(rope.get("rope_type") == "default", "unexpected DSpark RoPE type")
    require(rope.get("rope_theta") == 10_000_000, "unexpected DSpark RoPE theta")
    require(rope.get("partial_rotary_factor") == 0.25, "unexpected partial RoPE factor")

    return {
        "architecture": "DSparkDraftModel",
        "block_size": EXPECTED_BLOCK_SIZE,
        "anchor_slot": 0,
        "speculative_slots": list(range(1, EXPECTED_BLOCK_SIZE)),
        "speculative_tokens": EXPECTED_SPECULATIVE_TOKENS,
        "aux_hidden_state_indices": list(EXPECTED_AUX_HIDDEN_STATE_INDICES),
        "aux_decoder_layer_indices": list(EXPECTED_AUX_DECODER_LAYER_INDICES),
        "aux_concatenated_width": len(EXPECTED_AUX_HIDDEN_STATE_INDICES)
        * EXPECTED_HIDDEN_SIZE,
        "draft_layers": EXPECTED_DRAFT_LAYERS,
        "draft_vocab_size": EXPECTED_DRAFT_VOCAB_SIZE,
        "target_vocab_size": EXPECTED_TARGET_VOCAB_SIZE,
        "markov_rank": EXPECTED_MARKOV_RANK,
    }


def validate_header(header: dict[str, Any]) -> dict[str, Any]:
    require(header.get("__metadata__") == {"format": "pt"}, "unexpected safetensors metadata")
    entries = {name: value for name, value in header.items() if name != "__metadata__"}
    specs = expected_tensor_specs()
    missing = sorted(set(specs) - set(entries))
    unexpected = sorted(set(entries) - set(specs))
    require(not missing, f"missing DSpark tensor: {missing[0] if missing else ''}")
    require(not unexpected, f"unexpected DSpark tensor: {unexpected[0] if unexpected else ''}")

    ranges: list[tuple[int, int, str]] = []
    dtypes: Counter[str] = Counter()
    for name, (dtype, shape) in specs.items():
        entry = entries[name]
        require(isinstance(entry, dict), f"invalid DSpark tensor entry: {name}")
        require(entry.get("dtype") == dtype, f"DSpark tensor dtype mismatch: {name}")
        require(entry.get("shape") == shape, f"DSpark tensor shape mismatch: {name}")
        offsets = entry.get("data_offsets")
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(value, int) for value in offsets),
            f"invalid DSpark tensor offsets: {name}",
        )
        start, end = offsets
        require(0 <= start <= end, f"invalid DSpark tensor range: {name}")
        require(
            end - start == tensor_nbytes(dtype, shape),
            f"DSpark tensor payload mismatch: {name}",
        )
        ranges.append((start, end, name))
        dtypes[dtype] += 1

    cursor = 0
    for start, end, name in sorted(ranges):
        require(start == cursor, f"non-contiguous DSpark payload before tensor: {name}")
        cursor = end
    require(cursor == EXPECTED_PAYLOAD_BYTES, "unexpected DSpark payload size")
    require(len(entries) == 44, "unexpected DSpark tensor count")
    require(dtypes == Counter({"BF16": 42, "I64": 1, "BOOL": 1}), "unexpected dtype inventory")
    return {
        "tensor_count": len(entries),
        "payload_bytes": cursor,
        "dtype_counts": dict(sorted(dtypes.items())),
        "bf16_parameter_count": sum(
            math.prod(shape) for dtype, shape in specs.values() if dtype == "BF16"
        ),
    }


def validate_source_state(state: dict[str, Any]) -> dict[str, Any]:
    require(state.get("format") == SOURCE_STATE_FORMAT, "unexpected DSpark source-state format")
    require(state.get("profile") == "dspark", "unexpected companion metadata profile")
    require(state.get("repository") == EXPECTED_REPOSITORY, "unexpected DSpark repository")
    require(state.get("revision") == EXPECTED_REVISION, "unexpected DSpark revision")
    weight = state.get("weight")
    require(isinstance(weight, dict), "missing DSpark weight identity")
    require(weight.get("name") == EXPECTED_WEIGHT_NAME, "unexpected DSpark weight name")
    require(weight.get("file_bytes") == EXPECTED_WEIGHT_BYTES, "unexpected DSpark file size")
    require(weight.get("header_bytes") == EXPECTED_HEADER_BYTES, "unexpected DSpark header size")
    require(weight.get("payload_bytes") == EXPECTED_PAYLOAD_BYTES, "unexpected DSpark payload size")
    require(weight.get("sha256") == EXPECTED_WEIGHT_SHA256, "unexpected DSpark weight hash")
    require(weight.get("header_file") == "model.safetensors.header.json", "unexpected header file")
    require(weight.get("header_sha256") == EXPECTED_HEADER_SHA256, "unexpected header hash")
    require(
        state.get("metadata_files") == EXPECTED_METADATA_FILES,
        "unexpected DSpark metadata-file identities",
    )
    require(
        EXPECTED_WEIGHT_BYTES == 8 + EXPECTED_HEADER_BYTES + EXPECTED_PAYLOAD_BYTES,
        "inconsistent pinned DSpark file geometry",
    )
    return {
        "repository": EXPECTED_REPOSITORY,
        "revision": EXPECTED_REVISION,
        "weight_name": EXPECTED_WEIGHT_NAME,
        "weight_bytes": EXPECTED_WEIGHT_BYTES,
        "weight_sha256": EXPECTED_WEIGHT_SHA256,
    }


def validate_metadata_dir(metadata_dir: Path) -> dict[str, Any]:
    state = load_json(metadata_dir / "source-state.json")
    source = validate_source_state(state)
    metadata_files = state.get("metadata_files")
    require(isinstance(metadata_files, dict), "missing DSpark metadata-file identities")
    for name, identity in metadata_files.items():
        require(isinstance(name, str) and isinstance(identity, dict), "invalid metadata identity")
        path = metadata_dir / name
        require(path.is_file(), f"missing DSpark metadata file: {name}")
        require(path.stat().st_size == identity.get("bytes"), f"metadata size mismatch: {name}")
        require(sha256_file(path) == identity.get("sha256"), f"metadata hash mismatch: {name}")
    config = validate_config(load_json(metadata_dir / "config.json"))
    weights = validate_header(load_json(metadata_dir / "model.safetensors.header.json"))
    return {
        "format": CATALOG_FORMAT,
        "source": source,
        "contract": config,
        "weights": weights,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        catalog = validate_metadata_dir(args.metadata_dir)
        if args.out is not None:
            write_json_atomic(args.out, catalog)
        print(json.dumps(catalog, indent=2, sort_keys=True))
    except DSparkError as exc:
        print(f"ornith35 dspark metadata error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

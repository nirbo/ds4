#!/usr/bin/env python3
"""Atomic, provenance-bound persistent prefix state for Ornith-35."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Any, Sequence
from uuid import uuid4

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_runtime as mtp_runtime
import ornith35_mlx_turboquant_cache as turboquant_cache
import ornith35_mtp_reference as mtp_reference
import ornith35_nvfp4 as nvfp4
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import SafetensorsFile


STATE_SCHEMA = "ornith35-prefix-state-v2"
TURBOQUANT_STATE_SCHEMA = (
    "ornith35-prefix-state-turboquant-mixedk8-k9-fp32norm-"
    "exactl3-l7-l27-l31-l39-k9l15-l23-head256-tail256-min131072-v14"
)
CACHE_DTYPE_BF16 = "BF16"
CACHE_DTYPE_TURBOQUANT = (
    "MIXED_K8_K9_MSE_FP32_NORM_EXACT_L3_L7_L27_L31_L39_"
    "K9_L15_L23_HEAD256_TAIL256_MIN131072"
)
MANIFEST_NAME = "manifest.json"
TOKENS_NAME = "tokens.u32le"
MTP_PREFIX_NAME = "mtp-prefix.safetensors"
DEFAULT_MAX_ENTRIES = 64
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "key",
        "identity",
        "config_sha256",
        "position",
        "token_count",
        "tokens",
        "files",
        "mtp",
    }
)
_HASH_CHUNK = 8 * 1024 * 1024
_TOKEN_CHUNK = 8192
MTP_PROFILE_NONE = "none"
MTP_PROFILE_FOLDED = "ornith35-mtp-qwen35-folded-v1"
MTP_NONE_POLICY_SHA256 = hashlib.sha256(b'{"profile":"none"}').hexdigest()
PRODUCTION_MODEL_ID = "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
PRODUCTION_MODEL_REVISION = "85ffd2d0629ae5fa4f860dda356ec33161806c9b"
PRODUCTION_RUNTIME_FILES = (
    "ornith35/tools/ornith35_context.py",
    "ornith35/tools/ornith35_attention_reference.py",
    "ornith35/tools/ornith35_gdn_reference.py",
    "ornith35/tools/ornith35_moe_reference.py",
    "ornith35/tools/ornith35_nvfp4.py",
    "ornith35/tools/ornith35_tokenizer.py",
    "ornith35/tools/ornith35_turboquant_reference.py",
    "ornith35/tools/ornith35_mlx_compiled.py",
    "ornith35/tools/ornith35_mlx_dense.py",
    "ornith35/tools/ornith35_mlx_nvfp4.py",
    "ornith35/tools/ornith35_mlx_gdn.py",
    "ornith35/tools/ornith35_mlx_attention.py",
    "ornith35/tools/ornith35_mlx_moe.py",
    "ornith35/tools/ornith35_mlx_layer.py",
    "ornith35/tools/ornith35_mlx_model.py",
    "ornith35/tools/ornith35_mlx_vocab.py",
    "ornith35/tools/ornith35_mlx_linear_cache.py",
    "ornith35/tools/ornith35_mlx_turboquant.py",
    "ornith35/tools/ornith35_mlx_turboquant_cache.py",
    "ornith35/tools/ornith35_mlx_cache.py",
    "ornith35/tools/ornith35_runtime_coordination.py",
    "ornith35/tools/ornith35_mlx_generate.py",
    "ornith35/tools/ornith35_mlx_cache_warm.py",
    "ornith35/tools/ornith35_mlx_sampling.py",
    "ornith35/tools/ornith35_mlx_speculative.py",
    "ornith35/tools/ornith35_mlx_mtp.py",
    "ornith35/tools/ornith35_mlx_mtp_runtime.py",
    "ornith35/tools/ornith35_mtp_reference.py",
    "ornith35/extensions/kv_cache/bindings.cpp",
    "ornith35/extensions/kv_cache/kv_cache/kv_cache.cpp",
    "ornith35/extensions/kv_cache/kv_cache/kv_cache.h",
    "ornith35/extensions/kv_cache/kv_cache/kv_cache.metal",
)
PRODUCTION_RUNTIME_ARTIFACTS = (
    "ornith35/extensions/kv_cache/ornith35_mlx_kv_cache/ornith35_kv_cache_ext.metallib",
    "ornith35/extensions/kv_cache/ornith35_mlx_kv_cache/libornith35_kv_cache_ext.dylib",
)


@dataclass(frozen=True)
class CacheIdentity:
    model_id: str
    model_revision: str
    source_sha256: str
    runtime_revision: str
    runtime_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    quantization_policy_sha256: str
    rope_profile: str
    cache_dtype: str
    mtp_profile: str = MTP_PROFILE_NONE
    mtp_policy_sha256: str = MTP_NONE_POLICY_SHA256
    state_schema: str = STATE_SCHEMA


@dataclass(frozen=True)
class PersistentCache:
    path: Path
    key: str
    identity: CacheIdentity
    token_ids: tuple[int, ...]
    state: model.TextModelState
    mtp_prefix: mtp_runtime.MTPPrefixState | None
    load_timing: CacheLoadTiming


@dataclass(frozen=True)
class CacheLoadTiming:
    manifest_s: float
    tokens_s: float
    payload_verify_s: float
    payload_materialize_s: float
    finalize_s: float
    total_s: float
    payload_bytes: int


@dataclass(frozen=True)
class CacheLookupResult:
    path: Path | None
    token_count: int
    scanned_entries: int
    compatible_entries: int
    matching_entries: int
    elapsed_s: float


@dataclass(frozen=True)
class CachePruneResult:
    removed_entries: int
    removed_bytes: int
    retained_entries: int
    retained_bytes: int
    over_budget: bool


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_context_profile(value: str) -> context.ContextProfile:
    try:
        return context.resolve_profile(value)
    except context.ContextError as exc:
        raise MoEError(str(exc)) from exc


def validate_identity(identity: CacheIdentity) -> None:
    require(isinstance(identity, CacheIdentity), "invalid cache identity")
    require(
        all(
            isinstance(value, str) and value
            for value in asdict(identity).values()
        ),
        "cache identity fields must be nonempty strings",
    )
    for name in (
        "source_sha256",
        "runtime_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "quantization_policy_sha256",
        "mtp_policy_sha256",
    ):
        require(_is_sha256(getattr(identity, name)), f"invalid cache identity hash: {name}")
    require(
        identity.cache_dtype in (CACHE_DTYPE_BF16, CACHE_DTYPE_TURBOQUANT),
        "persistent cache dtype is unsupported",
    )
    profile = _resolve_context_profile(identity.rope_profile)
    require(
        identity.mtp_profile in (MTP_PROFILE_NONE, MTP_PROFILE_FOLDED),
        "persistent cache MTP profile is unsupported",
    )
    require(
        identity.mtp_profile != MTP_PROFILE_NONE
        or identity.mtp_policy_sha256 == MTP_NONE_POLICY_SHA256,
        "target-only cache has a noncanonical MTP policy",
    )
    require(
        identity.mtp_profile == MTP_PROFILE_NONE
        or profile.profile_id == context.NATIVE_PROFILE_ID,
        "MTP cache requires the native context profile",
    )
    expected_schema = (
        STATE_SCHEMA
        if identity.cache_dtype == CACHE_DTYPE_BF16
        else TURBOQUANT_STATE_SCHEMA
    )
    require(identity.state_schema == expected_schema, "persistent cache schema mismatch")
    require(
        identity.cache_dtype == CACHE_DTYPE_BF16
        or identity.mtp_profile == MTP_PROFILE_NONE,
        "TurboQuant cache cannot contain MTP state",
    )
    require(
        identity.cache_dtype == CACHE_DTYPE_BF16
        or profile.profile_id == context.NATIVE_PROFILE_ID,
        "TurboQuant cache requires the native context profile",
    )


def identity_uses_mtp(identity: CacheIdentity) -> bool:
    validate_identity(identity)
    return identity.mtp_profile != MTP_PROFILE_NONE


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _config_sha256(config: model.TextModelConfig) -> str:
    return hashlib.sha256(_canonical_json(asdict(config))).hexdigest()


def _token_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for offset in range(0, len(token_ids), _TOKEN_CHUNK):
        values = token_ids[offset : offset + _TOKEN_CHUNK]
        digest.update(struct.pack(f"<{len(values)}I", *values))
    return digest.hexdigest()


def _token_prefix_sha256s(
    token_ids: Sequence[int],
    lengths: Sequence[int],
) -> dict[int, str]:
    requested = sorted(set(lengths))
    require(
        all(isinstance(length, int) and 0 < length <= len(token_ids) for length in requested),
        "invalid cache prefix-hash length",
    )
    digest = hashlib.sha256()
    result = {}
    cursor = 0
    for length in requested:
        while cursor < length:
            end = min(cursor + _TOKEN_CHUNK, length)
            values = token_ids[cursor:end]
            digest.update(struct.pack(f"<{len(values)}I", *values))
            cursor = end
        result[length] = digest.hexdigest()
    return result


def _cache_key_from_token_sha256(
    token_sha256: str,
    position: int,
    identity: CacheIdentity,
    config: model.TextModelConfig,
) -> str:
    require(_is_sha256(token_sha256), "invalid cache token hash")
    descriptor = {
        "config_sha256": _config_sha256(config),
        "identity": asdict(identity),
        "position": position,
        "token_sha256": token_sha256,
    }
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest()


def cache_key(
    token_ids: Sequence[int],
    identity: CacheIdentity,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
) -> str:
    validate_identity(identity)
    tokens = tuple(token_ids)
    require(tokens, "persistent cache token prefix must not be empty")
    require(
        all(isinstance(token, int) and 0 <= token < config.vocab_size for token in tokens),
        "persistent cache token ID is out of range",
    )
    return _cache_key_from_token_sha256(
        _token_sha256(tokens),
        len(tokens),
        identity,
        config,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MoEError(f"cannot read cache identity JSON: {path}") from exc
    require(isinstance(value, dict), f"cache identity JSON is not an object: {path}")
    return value


def _runtime_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (*PRODUCTION_RUNTIME_FILES, *PRODUCTION_RUNTIME_ARTIFACTS):
        path = repo_root / relative
        if relative in PRODUCTION_RUNTIME_ARTIFACTS and not path.is_file():
            continue
        require(path.is_file(), f"cache runtime input is missing: {relative}")
        relative_bytes = relative.encode("ascii")
        digest.update(struct.pack("<I", len(relative_bytes)))
        digest.update(relative_bytes)
        digest.update(bytes.fromhex(_file_sha256(path)))
    digest.update(b"mlx=0.32.0;mlx-metal=0.32.0")
    return digest.hexdigest()


def _production_mtp_policy(
    model_root: Path,
    adaptation_dir: Path | None,
) -> tuple[str, str]:
    if adaptation_dir is None:
        return MTP_PROFILE_NONE, MTP_NONE_POLICY_SHA256

    source_path = model_root / "source-mtp-state.json"
    source = _load_json_object(source_path)
    require(source.get("format") == mtp.STATE_FORMAT, "cache MTP source format mismatch")
    require(source.get("status") == "complete", "cache MTP source is incomplete")
    require(source.get("repository") == mtp.EXPECTED_REPOSITORY, "cache MTP repository mismatch")
    require(source.get("revision") == mtp.EXPECTED_REVISION, "cache MTP revision mismatch")
    require(
        source.get("output_sha256") == mtp.EXPECTED_SIDECAR_SHA256,
        "cache MTP sidecar identity mismatch",
    )
    sidecar = model_root / "source-mtp" / "mtp.safetensors"
    require(sidecar.is_file() and not sidecar.is_symlink(), "cache MTP sidecar is absent")
    require(sidecar.stat().st_size == mtp.EXPECTED_SIDECAR_BYTES, "cache MTP sidecar size mismatch")

    require(
        adaptation_dir.is_dir() and not adaptation_dir.is_symlink(),
        "cache MTP adaptation is absent",
    )
    directory = adaptation_dir.resolve()
    adaptation_state_path = directory / "state.json"
    require(
        adaptation_state_path.is_file() and not adaptation_state_path.is_symlink(),
        "cache MTP adaptation state is absent",
    )
    adaptation = _load_json_object(adaptation_state_path)
    require(
        adaptation.get("format") == mtp.ADAPTATION_FORMAT,
        "cache MTP adaptation format mismatch",
    )
    require(
        adaptation.get("status") in ("candidate", "accepted"),
        "cache MTP adaptation is not runtime-eligible",
    )
    adaptation_source = adaptation.get("source")
    require(isinstance(adaptation_source, dict), "cache MTP adaptation source is absent")
    require(
        adaptation_source.get("target_weight_sha256") == nvfp4.EXPECTED_WEIGHT_SHA256,
        "cache MTP adaptation target mismatch",
    )
    require(
        adaptation_source.get("mtp_sidecar_sha256") == mtp.EXPECTED_SIDECAR_SHA256,
        "cache MTP adaptation sidecar mismatch",
    )
    require(
        adaptation_source.get("mtp_fc_sha256") == mtp.EXPECTED_FC_SHA256,
        "cache MTP adaptation base projection mismatch",
    )
    artifact = adaptation.get("artifact")
    require(isinstance(artifact, dict), "cache MTP adaptation artifact is absent")
    require(artifact.get("name") == mtp.ADAPTATION_ARTIFACT, "cache MTP artifact name mismatch")
    artifact_path = directory / mtp.ADAPTATION_ARTIFACT
    require(
        artifact_path.is_file() and not artifact_path.is_symlink(),
        "cache MTP adaptation artifact is absent",
    )
    require(
        artifact_path.stat().st_size == artifact.get("bytes"),
        "cache MTP adaptation artifact size mismatch",
    )
    artifact_sha256 = artifact.get("sha256")
    require(_is_sha256(artifact_sha256), "cache MTP adaptation SHA-256 is invalid")
    require(
        _file_sha256(artifact_path) == artifact_sha256,
        "cache MTP adaptation artifact hash mismatch",
    )
    policy = {
        "profile": MTP_PROFILE_FOLDED,
        "config_sha256": hashlib.sha256(
            _canonical_json(asdict(mtp.PRODUCTION_CONFIG))
        ).hexdigest(),
        "source_repository": mtp.EXPECTED_REPOSITORY,
        "source_revision": mtp.EXPECTED_REVISION,
        "source_state_sha256": _file_sha256(source_path),
        "sidecar_sha256": mtp.EXPECTED_SIDECAR_SHA256,
        "adaptation_format": mtp.ADAPTATION_FORMAT,
        "adaptation_status": adaptation["status"],
        "adaptation_state_sha256": _file_sha256(adaptation_state_path),
        "adaptation_sha256": artifact_sha256,
    }
    return MTP_PROFILE_FOLDED, hashlib.sha256(_canonical_json(policy)).hexdigest()


def production_identity(
    model_root: Path,
    repo_root: Path,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    mapped_embedding: bool,
    quantized_lm_head: bool,
    mtp_adaptation_dir: Path | None = None,
    rope_profile: str = "native-262k",
    turboquant_kv: bool = False,
) -> CacheIdentity:
    """Bind a cache entry to the verified source and exact local runtime bytes."""
    profile = _resolve_context_profile(rope_profile)
    source = _load_json_object(model_root / "source-nvfp4-state.json")
    require(source.get("repository") == PRODUCTION_MODEL_ID, "cache source repository mismatch")
    require(source.get("revision") == PRODUCTION_MODEL_REVISION, "cache source revision mismatch")
    weight = source.get("weight")
    require(isinstance(weight, dict), "cache source state has no weight identity")
    source_sha256 = weight.get("sha256")
    require(_is_sha256(source_sha256), "cache source SHA-256 is invalid")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_revision = result.stdout.strip()
    require(runtime_revision, "cache runtime revision is empty")
    policy = {
        "source": "modelopt-nvfp4-mixed-source",
        "mapped_embedding": mapped_embedding,
        "quantized_lm_head": quantized_lm_head,
        "turboquant_kv": (
            {
                "profile": turboquant_cache.PRODUCTION_PROFILE,
                "key_rotation_seed": turboquant_cache.KEY_ROTATION_SEED,
                "value_rotation_seed": turboquant_cache.VALUE_ROTATION_SEED,
                "exact_head_tokens": turboquant_cache.PRODUCTION_EXACT_HEAD_TOKENS,
                "exact_tail_tokens": turboquant_cache.PRODUCTION_EXACT_TAIL_TOKENS,
                "minimum_history_tokens": (
                    turboquant_cache.PRODUCTION_MINIMUM_HISTORY_TOKENS
                ),
                "bf16_norm_layers": sorted(
                    turboquant_cache.PRODUCTION_BF16_NORM_LAYERS
                ),
                "exact_attention_layers": sorted(
                    turboquant_cache.PRODUCTION_EXACT_ATTENTION_LAYERS
                ),
                "k8_attention_layers": sorted(
                    turboquant_cache.PRODUCTION_K8_ATTENTION_LAYERS
                ),
                "k9_attention_layers": sorted(
                    turboquant_cache.PRODUCTION_K9_ATTENTION_LAYERS
                ),
            }
            if turboquant_kv
            else None
        ),
    }
    mtp_profile, mtp_policy_sha256 = _production_mtp_policy(
        model_root,
        mtp_adaptation_dir,
    )
    require(
        not turboquant_kv or mtp_profile == MTP_PROFILE_NONE,
        "TurboQuant cache cannot be combined with MTP",
    )
    identity = CacheIdentity(
        model_id=PRODUCTION_MODEL_ID,
        model_revision=PRODUCTION_MODEL_REVISION,
        source_sha256=source_sha256,
        runtime_revision=runtime_revision,
        runtime_sha256=_runtime_sha256(repo_root),
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        quantization_policy_sha256=hashlib.sha256(_canonical_json(policy)).hexdigest(),
        rope_profile=profile.profile_id,
        cache_dtype=CACHE_DTYPE_TURBOQUANT if turboquant_kv else CACHE_DTYPE_BF16,
        mtp_profile=mtp_profile,
        mtp_policy_sha256=mtp_policy_sha256,
        state_schema=TURBOQUANT_STATE_SCHEMA if turboquant_kv else STATE_SCHEMA,
    )
    validate_identity(identity)
    return identity


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_tokens(path: Path, token_ids: tuple[int, ...]) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for offset in range(0, len(token_ids), _TOKEN_CHUNK):
            values = token_ids[offset : offset + _TOKEN_CHUNK]
            payload = struct.pack(f"<{len(values)}I", *values)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "name": TOKENS_NAME,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _dtype_name(value: mx.array) -> str:
    if value.dtype == mx.uint8:
        return "U8"
    if value.dtype == mx.bfloat16:
        return "BF16"
    if value.dtype == mx.float32:
        return "F32"
    raise MoEError(f"persistent cache tensor dtype is unsupported: {value.dtype}")


def _tensor_spec(value: mx.array) -> dict[str, Any]:
    return {
        "dtype": _dtype_name(value),
        "shape": list(value.shape),
        "bytes": value.size * value.itemsize,
    }


def _layer_payload(
    layer_index: int,
    layer_kind: str,
    layer_state: model.LayerState,
    position: int,
    identity: CacheIdentity,
) -> tuple[dict[str, mx.array], dict[str, str]]:
    metadata = {
        "schema": identity.state_schema,
        "layer": str(layer_index),
        "kind": layer_kind,
        "position": str(position),
    }
    if layer_kind == model.LAYER_GDN:
        require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {layer_index}")
        return {
            "conv": layer_state.conv,
            "recurrent": layer_state.recurrent,
        }, metadata
    if identity.cache_dtype == CACHE_DTYPE_TURBOQUANT:
        if layer_index in turboquant_cache.PRODUCTION_EXACT_ATTENTION_LAYERS:
            require(
                isinstance(
                    layer_state,
                    (attention.MLXAttentionState, attention.MLXLinearAttentionState),
                ),
                f"exact TurboQuant attention state mismatch at {layer_index}",
            )
            if isinstance(layer_state, attention.MLXLinearAttentionState):
                require(
                    layer_state.position == position,
                    f"exact TurboQuant linear position mismatch at {layer_index}",
                )
                keys = layer_state.keys[:, :position, :]
                values = layer_state.values[:, :position, :]
            else:
                keys = layer_state.keys
                values = layer_state.values
            return {"keys": keys, "values": values}, metadata
        require(
            isinstance(
                layer_state,
                (
                    attention.MLXTurboQuantImmutableAttentionState,
                    attention.MLXTurboQuantAttentionState,
                ),
            ),
            f"TurboQuant attention state mismatch at {layer_index}",
        )
        require(
            turboquant_cache.state_length(layer_state) == position,
            f"TurboQuant position mismatch at {layer_index}",
        )
        require(
            layer_state.bits == turboquant_cache.production_packed_bits(layer_index),
            f"TurboQuant packed bit width mismatch at {layer_index}",
        )
        history = turboquant_cache.packed_history(layer_state)
        arrays = {}
        if layer_state.exact_head_keys.shape[1]:
            arrays.update(
                {
                    "exact_head_keys": layer_state.exact_head_keys,
                    "exact_head_values": layer_state.exact_head_values,
                }
            )
        if layer_state.exact_keys.shape[1]:
            arrays.update(
                {
                    "exact_keys": layer_state.exact_keys,
                    "exact_values": layer_state.exact_values,
                }
            )
        if history:
            arrays.update(
                {
                    "packed_keys": layer_state.packed_keys[:, :history],
                    "key_norms": layer_state.key_norms[:, :history],
                    "packed_values": layer_state.packed_values[:, :history],
                    "value_norms": layer_state.value_norms[:, :history],
                }
            )
        return arrays, metadata
    require(
        isinstance(
            layer_state,
            (attention.MLXAttentionState, attention.MLXLinearAttentionState),
        ),
        f"attention state mismatch at {layer_index}",
    )
    if isinstance(layer_state, attention.MLXLinearAttentionState):
        require(layer_state.position == position, f"linear position mismatch at {layer_index}")
        keys = layer_state.keys[:, :position, :]
        values = layer_state.values[:, :position, :]
    else:
        keys = layer_state.keys
        values = layer_state.values
    return {"keys": keys, "values": values}, metadata


def _mtp_prefix_payload(
    prefix: mtp_runtime.MTPPrefixState,
    target_position: int,
    mtp_config: mtp_reference.MTPConfig,
) -> tuple[dict[str, mx.array], dict[str, str]]:
    mtp_runtime.validate_prefix_state(
        prefix,
        target_position,
        mtp_config,
        dtype=mx.bfloat16,
    )
    mtp_position = target_position - 1
    if isinstance(prefix.state, attention.MLXLinearAttentionState):
        keys = prefix.state.keys[:, :mtp_position, :]
        values = prefix.state.values[:, :mtp_position, :]
    else:
        keys = prefix.state.keys
        values = prefix.state.values
    arrays = {"boundary_hidden": prefix.boundary_hidden}
    if mtp_position:
        arrays.update({"keys": keys, "values": values})
    return arrays, {
        "schema": STATE_SCHEMA,
        "kind": "mtp-prefix",
        "target_position": str(target_position),
        "mtp_position": str(mtp_position),
    }


def _validate_persistable_state(
    state: model.TextModelState,
    config: model.TextModelConfig,
    identity: CacheIdentity,
) -> None:
    model.validate_state(state, config)
    require(state.position > 0, "persistent cache position must be positive")
    require(len(state.layers) == len(config.layer_types), "cache layer count mismatch")
    for index, (kind, layer_state) in enumerate(zip(config.layer_types, state.layers)):
        if kind == model.LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            gdn.validate_state(layer_state, config.gdn)
            require(layer_state.conv.dtype == mx.bfloat16, f"GDN cache dtype mismatch at {index}")
            continue
        if identity.cache_dtype == CACHE_DTYPE_TURBOQUANT:
            if index in turboquant_cache.PRODUCTION_EXACT_ATTENTION_LAYERS:
                require(
                    isinstance(
                        layer_state,
                        (attention.MLXAttentionState, attention.MLXLinearAttentionState),
                    ),
                    f"exact TurboQuant attention state mismatch at {index}",
                )
                require(
                    attention.state_length(layer_state, config.attention) == state.position,
                    f"exact TurboQuant position mismatch at {index}",
                )
                require(
                    layer_state.keys.dtype == layer_state.values.dtype == mx.bfloat16,
                    f"exact TurboQuant dtype mismatch at {index}",
                )
                continue
            require(
                config.attention == attention.PRODUCTION_CONFIG,
                "TurboQuant persistence requires production attention geometry",
            )
            require(
                isinstance(
                    layer_state,
                    (
                        attention.MLXTurboQuantImmutableAttentionState,
                        attention.MLXTurboQuantAttentionState,
                    ),
                ),
                f"TurboQuant attention state mismatch at {index}",
            )
            require(
                turboquant_cache.state_length(layer_state) == state.position,
                f"TurboQuant position mismatch at {index}",
            )
            require(
                layer_state.key_norms.dtype
                == layer_state.value_norms.dtype
                == turboquant_cache.production_norm_dtype(index),
                f"TurboQuant persistent norm dtype mismatch at {index}",
            )
            require(
                layer_state.bits == turboquant_cache.production_packed_bits(index),
                f"TurboQuant persistent bit width mismatch at {index}",
            )
            continue
        require(
            isinstance(
                layer_state,
                (attention.MLXAttentionState, attention.MLXLinearAttentionState),
            ),
            f"attention state mismatch at {index}",
        )
        require(
            attention.state_length(layer_state, config.attention) == state.position,
            f"attention position mismatch at {index}",
        )
        require(layer_state.keys.dtype == mx.bfloat16, f"attention cache dtype mismatch at {index}")


def _validate_persistable_mtp_prefix(
    prefix: mtp_runtime.MTPPrefixState | None,
    identity: CacheIdentity,
    target_position: int,
    mtp_config: mtp_reference.MTPConfig,
) -> None:
    expected = identity_uses_mtp(identity)
    require(
        (prefix is not None) == expected,
        "cache identity and MTP prefix presence disagree",
    )
    if prefix is not None:
        mtp_runtime.validate_prefix_state(
            prefix,
            target_position,
            mtp_config,
            dtype=mx.bfloat16,
        )


def _verify_safetensors(
    path: Path,
    tensors: dict[str, dict[str, Any]],
) -> None:
    with SafetensorsFile(path) as source:
        require(set(source.tensors) == set(tensors), f"cache tensor set mismatch: {path.name}")
        for name, spec in tensors.items():
            entry = source.entry(name)
            require(entry.get("dtype") == spec["dtype"], f"cache dtype mismatch: {path.name}:{name}")
            require(entry.get("shape") == spec["shape"], f"cache shape mismatch: {path.name}:{name}")
            require(
                source.tensor_nbytes(name) == spec["bytes"],
                f"cache payload mismatch: {path.name}:{name}",
            )


def _expected_tensor_specs(
    layer_index: int,
    kind: str,
    position: int,
    config: model.TextModelConfig,
    identity: CacheIdentity,
) -> dict[str, dict[str, Any]]:
    if kind == model.LAYER_GDN:
        shapes = {
            "conv": (
                config.gdn.conv_dim,
                config.gdn.conv_kernel_size,
            ),
            "recurrent": (
                config.gdn.num_v_heads,
                config.gdn.head_k_dim,
                config.gdn.head_v_dim,
            ),
        }
        dtypes = {"conv": "BF16", "recurrent": "F32"}
    elif (
        identity.cache_dtype == CACHE_DTYPE_TURBOQUANT
        and layer_index in turboquant_cache.PRODUCTION_EXACT_ATTENTION_LAYERS
    ):
        shape = (
            config.attention.num_kv_heads,
            position,
            config.attention.head_dim,
        )
        shapes = {"keys": shape, "values": shape}
        dtypes = {"keys": "BF16", "values": "BF16"}
    elif identity.cache_dtype == CACHE_DTYPE_TURBOQUANT:
        head = min(position, turboquant_cache.PRODUCTION_EXACT_HEAD_TOKENS)
        remaining = position - head
        tail = min(remaining, turboquant_cache.PRODUCTION_EXACT_TAIL_TOKENS)
        history = remaining - tail
        packed_shape = (
            config.attention.num_kv_heads,
            history,
            turboquant_cache.packed_dimension(
                turboquant_cache.production_packed_bits(layer_index)
            ),
        )
        norm_shape = (config.attention.num_kv_heads, history, 1)
        exact_shape = (
            config.attention.num_kv_heads,
            tail,
            config.attention.head_dim,
        )
        exact_head_shape = (
            config.attention.num_kv_heads,
            head,
            config.attention.head_dim,
        )
        shapes = {}
        dtypes = {}
        if head:
            shapes.update(
                {
                    "exact_head_keys": exact_head_shape,
                    "exact_head_values": exact_head_shape,
                }
            )
            dtypes.update(
                {
                    "exact_head_keys": "BF16",
                    "exact_head_values": "BF16",
                }
            )
        if tail:
            shapes.update(
                {
                    "exact_keys": exact_shape,
                    "exact_values": exact_shape,
                }
            )
            dtypes.update(
                {
                    "exact_keys": "BF16",
                    "exact_values": "BF16",
                }
            )
        if history:
            shapes.update(
                {
                    "packed_keys": packed_shape,
                    "key_norms": norm_shape,
                    "packed_values": packed_shape,
                    "value_norms": norm_shape,
                }
            )
            dtypes.update(
                {
                    "packed_keys": "U8",
                    "key_norms": (
                        "BF16"
                        if layer_index in turboquant_cache.PRODUCTION_BF16_NORM_LAYERS
                        else "F32"
                    ),
                    "packed_values": "U8",
                    "value_norms": (
                        "BF16"
                        if layer_index in turboquant_cache.PRODUCTION_BF16_NORM_LAYERS
                        else "F32"
                    ),
                }
            )
    else:
        shape = (
            config.attention.num_kv_heads,
            position,
            config.attention.head_dim,
        )
        shapes = {"keys": shape, "values": shape}
        dtypes = {"keys": "BF16", "values": "BF16"}
    return {
        name: {
            "dtype": dtypes[name],
            "shape": list(shape),
            "bytes": math.prod(shape) * {"U8": 1, "BF16": 2, "F32": 4}[dtypes[name]],
        }
        for name, shape in shapes.items()
    }


def _expected_mtp_tensor_specs(
    target_position: int,
    mtp_config: mtp_reference.MTPConfig,
) -> dict[str, dict[str, Any]]:
    mtp_position = target_position - 1
    shapes = {"boundary_hidden": (mtp_config.hidden_size,)}
    if mtp_position:
        kv_shape = (
            mtp_config.attention.num_kv_heads,
            mtp_position,
            mtp_config.attention.head_dim,
        )
        shapes.update({"keys": kv_shape, "values": kv_shape})
    return {
        name: {
            "dtype": "BF16",
            "shape": list(shape),
            "bytes": math.prod(shape) * 2,
        }
        for name, shape in shapes.items()
    }


def save_cache(
    root: Path,
    token_ids: Sequence[int],
    state: model.TextModelState,
    identity: CacheIdentity,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    *,
    mtp_prefix: mtp_runtime.MTPPrefixState | None = None,
    mtp_config: mtp_reference.MTPConfig = mtp.PRODUCTION_CONFIG,
) -> Path:
    """Write one immutable cache entry and publish it with an atomic rename."""
    validate_identity(identity)
    tokens = tuple(token_ids)
    require(len(tokens) == state.position, "cache token/state position mismatch")
    require(
        all(isinstance(token, int) and 0 <= token < config.vocab_size for token in tokens),
        "persistent cache token ID is out of range",
    )
    _validate_persistable_state(state, config, identity)
    require(
        state.context_profile == identity.rope_profile,
        "cache state and identity context profiles disagree",
    )
    _validate_persistable_mtp_prefix(
        mtp_prefix,
        identity,
        state.position,
        mtp_config,
    )
    key = cache_key(tokens, identity, config)
    root.mkdir(parents=True, exist_ok=True)
    final = root / key
    if final.exists():
        load_cache(
            final,
            identity,
            config,
            expected_tokens=tokens,
            mtp_config=mtp_config,
        )
        return final
    staging = root / f".{key}.part-{os.getpid()}-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        token_record = _write_tokens(staging / TOKENS_NAME, tokens)
        files = []
        for index, (kind, layer_state) in enumerate(zip(config.layer_types, state.layers)):
            arrays, metadata = _layer_payload(
                index,
                kind,
                layer_state,
                state.position,
                identity,
            )
            name = f"layer-{index:03d}.safetensors"
            path = staging / name
            mx.save_safetensors(path, arrays, metadata)
            _fsync_file(path)
            tensor_specs = {
                tensor_name: _tensor_spec(array)
                for tensor_name, array in arrays.items()
            }
            _verify_safetensors(path, tensor_specs)
            files.append(
                {
                    "name": name,
                    "layer": index,
                    "kind": kind,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                    "tensors": tensor_specs,
                }
            )
        mtp_record = None
        if mtp_prefix is not None:
            arrays, metadata = _mtp_prefix_payload(
                mtp_prefix,
                state.position,
                mtp_config,
            )
            path = staging / MTP_PREFIX_NAME
            mx.save_safetensors(path, arrays, metadata)
            _fsync_file(path)
            tensor_specs = {
                tensor_name: _tensor_spec(array)
                for tensor_name, array in arrays.items()
            }
            _verify_safetensors(path, tensor_specs)
            mtp_record = {
                "name": MTP_PREFIX_NAME,
                "kind": "mtp-prefix",
                "target_position": state.position,
                "mtp_position": state.position - 1,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "tensors": tensor_specs,
            }
        manifest = {
            "schema": identity.state_schema,
            "key": key,
            "identity": asdict(identity),
            "config_sha256": _config_sha256(config),
            "position": state.position,
            "token_count": len(tokens),
            "tokens": token_record,
            "files": files,
            "mtp": mtp_record,
        }
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        try:
            os.rename(staging, final)
        except OSError:
            if not final.is_dir():
                raise
            load_cache(
                final,
                identity,
                config,
                expected_tokens=tokens,
                mtp_config=mtp_config,
            )
            shutil.rmtree(staging)
        _fsync_directory(root)
        return final
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate cache manifest key: {key}")
        result[key] = value
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "cache manifest is missing or unsafe")
    require(path.stat().st_size <= 1024 * 1024, "cache manifest is too large")
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MoEError("cache manifest is invalid JSON") from exc
    require(isinstance(value, dict), "cache manifest root must be an object")
    return value


def _validate_manifest_envelope(
    manifest: dict[str, Any],
    *,
    entry_name: str,
) -> None:
    require(set(manifest) == _MANIFEST_FIELDS, "cache manifest fields mismatch")
    require(manifest["schema"] in (STATE_SCHEMA, TURBOQUANT_STATE_SCHEMA), "cache schema mismatch")
    require(_is_sha256(manifest["key"]), "cache manifest key is invalid")
    require(manifest["key"] == entry_name, "cache manifest key/path mismatch")
    require(isinstance(manifest["identity"], dict), "cache manifest identity is invalid")
    require(_is_sha256(manifest["config_sha256"]), "cache config hash is invalid")
    position = manifest["position"]
    require(type(position) is int and position > 0, "cache position is invalid")
    require(manifest["token_count"] == position, "cache token count mismatch")
    token_record = manifest["tokens"]
    require(
        isinstance(token_record, dict)
        and set(token_record) == {"name", "bytes", "sha256"}
        and token_record["name"] == TOKENS_NAME
        and token_record["bytes"] == position * 4
        and _is_sha256(token_record["sha256"]),
        "cache token manifest mismatch",
    )
    require(isinstance(manifest["files"], list), "cache layer manifest is invalid")


def _read_tokens(path: Path, count: int, expected_sha256: str) -> tuple[int, ...]:
    require(path.is_file() and not path.is_symlink(), "cache token file is missing or unsafe")
    require(path.stat().st_size == count * 4, "cache token byte count mismatch")
    require(_file_sha256(path) == expected_sha256, "cache token hash mismatch")
    values = []
    with path.open("rb") as handle:
        while payload := handle.read(_TOKEN_CHUNK * 4):
            require(len(payload) % 4 == 0, "cache token payload is truncated")
            values.extend(struct.unpack(f"<{len(payload) // 4}I", payload))
    require(len(values) == count, "cache token count mismatch")
    return tuple(values)


def find_longest_prefix(
    root: Path,
    token_ids: Sequence[int],
    identity: CacheIdentity,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    *,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    min_suffix_tokens: int = 1,
) -> CacheLookupResult:
    """Find the longest exact cache prefix under a required suffix length."""
    started = time.perf_counter()
    validate_identity(identity)
    tokens = tuple(token_ids)
    require(
        all(isinstance(token, int) and 0 <= token < config.vocab_size for token in tokens),
        "cache lookup token ID is out of range",
    )
    require(max_entries > 0, "cache lookup entry bound must be positive")
    require(min_suffix_tokens >= 0, "cache lookup suffix requirement is invalid")
    if len(tokens) <= min_suffix_tokens or not root.exists():
        return CacheLookupResult(None, 0, 0, 0, 0, time.perf_counter() - started)
    require(root.is_dir() and not root.is_symlink(), "cache root is missing or unsafe")

    expected_identity = asdict(identity)
    expected_config = _config_sha256(config)
    max_position = len(tokens) - min_suffix_tokens
    scanned_entries = 0
    compatible = []
    visible = sorted(
        (path for path in root.iterdir() if not path.name.startswith(".")),
        key=lambda entry: entry.name,
    )
    require(len(visible) <= max_entries, "cache root exceeds lookup entry bound")
    for path in visible:
        require(
            path.is_dir() and not path.is_symlink() and _is_sha256(path.name),
            f"unexpected cache-root entry: {path.name}",
        )
        scanned_entries += 1
        manifest = _read_manifest(path / MANIFEST_NAME)
        _validate_manifest_envelope(manifest, entry_name=path.name)
        if (
            manifest["identity"] != expected_identity
            or manifest["schema"] != identity.state_schema
            or manifest["config_sha256"] != expected_config
            or manifest["position"] > max_position
        ):
            continue
        compatible.append((manifest["position"], path, manifest))

    prefix_hashes = _token_prefix_sha256s(
        tokens,
        [position for position, _, _ in compatible],
    )
    matches = []
    for position, path, manifest in compatible:
        token_sha256 = prefix_hashes[position]
        expected_key = _cache_key_from_token_sha256(
            token_sha256,
            position,
            identity,
            config,
        )
        if (
            manifest["tokens"]["sha256"] == token_sha256
            and path.name == expected_key
        ):
            matches.append((position, path))
    selected = max(matches, default=None, key=lambda candidate: candidate[0])
    return CacheLookupResult(
        path=selected[1] if selected is not None else None,
        token_count=selected[0] if selected is not None else 0,
        scanned_entries=scanned_entries,
        compatible_entries=len(compatible),
        matching_entries=len(matches),
        elapsed_s=time.perf_counter() - started,
    )


def _load_layer(
    path: Path,
    record: dict[str, Any],
    position: int,
    context_profile: str,
    identity: CacheIdentity,
    config: model.TextModelConfig,
) -> tuple[model.LayerState, float, float]:
    verify_started = time.perf_counter()
    require(path.is_file() and not path.is_symlink(), f"cache layer is missing or unsafe: {path.name}")
    require(path.stat().st_size == record["bytes"], f"cache layer size mismatch: {path.name}")
    require(_file_sha256(path) == record["sha256"], f"cache layer hash mismatch: {path.name}")
    _verify_safetensors(path, record["tensors"])
    verify_elapsed = time.perf_counter() - verify_started
    materialize_started = time.perf_counter()
    arrays, metadata = mx.load(path, return_metadata=True)
    require(
        metadata
        == {
            "schema": identity.state_schema,
            "layer": str(record["layer"]),
            "kind": record["kind"],
            "position": str(position),
        },
        f"cache layer metadata mismatch: {path.name}",
    )
    require(set(arrays) == set(record["tensors"]), f"cache layer array mismatch: {path.name}")
    mx.eval(*arrays.values())
    mx.synchronize()
    if record["kind"] == model.LAYER_GDN:
        state: model.LayerState = gdn.MLXGDNState(
            conv=arrays["conv"],
            recurrent=arrays["recurrent"],
        )
        return state, verify_elapsed, time.perf_counter() - materialize_started
    if identity.cache_dtype == CACHE_DTYPE_TURBOQUANT:
        if record["layer"] in turboquant_cache.PRODUCTION_EXACT_ATTENTION_LAYERS:
            exact = attention.MLXAttentionState(
                keys=arrays["keys"],
                values=arrays["values"],
                context_profile=context_profile,
            )
            require(
                attention.state_length(exact, config.attention) == position
                and exact.keys.dtype == exact.values.dtype == mx.bfloat16,
                "restored exact TurboQuant attention state mismatch",
            )
            return exact, verify_elapsed, time.perf_counter() - materialize_started
        head = min(position, turboquant_cache.PRODUCTION_EXACT_HEAD_TOKENS)
        remaining = position - head
        tail = min(remaining, turboquant_cache.PRODUCTION_EXACT_TAIL_TOKENS)
        history = remaining - tail
        packed_shape = (
            config.attention.num_kv_heads,
            history,
            turboquant_cache.packed_dimension(
                turboquant_cache.production_packed_bits(record["layer"])
            ),
        )
        norm_shape = (config.attention.num_kv_heads, history, 1)
        exact_head_shape = (
            config.attention.num_kv_heads,
            head,
            config.attention.head_dim,
        )
        exact_shape = (
            config.attention.num_kv_heads,
            tail,
            config.attention.head_dim,
        )
        packed = turboquant_cache.MLXPackedMSEState(
            packed_keys=arrays.get("packed_keys", mx.zeros(packed_shape, dtype=mx.uint8)),
            key_norms=arrays.get(
                "key_norms",
                mx.zeros(
                    norm_shape,
                    dtype=turboquant_cache.production_norm_dtype(record["layer"]),
                ),
            ),
            packed_values=arrays.get("packed_values", mx.zeros(packed_shape, dtype=mx.uint8)),
            value_norms=arrays.get(
                "value_norms",
                mx.zeros(
                    norm_shape,
                    dtype=turboquant_cache.production_norm_dtype(record["layer"]),
                ),
            ),
            exact_head_keys=arrays.get(
                "exact_head_keys",
                mx.zeros(exact_head_shape, dtype=mx.bfloat16),
            ),
            exact_head_values=arrays.get(
                "exact_head_values",
                mx.zeros(exact_head_shape, dtype=mx.bfloat16),
            ),
            exact_keys=arrays.get("exact_keys", mx.zeros(exact_shape, dtype=mx.bfloat16)),
            exact_values=arrays.get("exact_values", mx.zeros(exact_shape, dtype=mx.bfloat16)),
            exact_head_capacity=turboquant_cache.PRODUCTION_EXACT_HEAD_TOKENS,
            exact_tail_capacity=turboquant_cache.PRODUCTION_EXACT_TAIL_TOKENS,
            context_profile=context_profile,
            bits=turboquant_cache.production_packed_bits(record["layer"]),
        )
        turboquant_cache.validate_state(packed)
        return packed, verify_elapsed, time.perf_counter() - materialize_started
    state = attention.MLXAttentionState(
        keys=arrays["keys"],
        values=arrays["values"],
        context_profile=context_profile,
    )
    return state, verify_elapsed, time.perf_counter() - materialize_started


def _load_mtp_prefix(
    path: Path,
    record: dict[str, Any],
    target_position: int,
    mtp_config: mtp_reference.MTPConfig,
) -> tuple[mtp_runtime.MTPPrefixState, float, float]:
    verify_started = time.perf_counter()
    require(path.is_file() and not path.is_symlink(), "cache MTP prefix is missing or unsafe")
    require(path.stat().st_size == record["bytes"], "cache MTP prefix size mismatch")
    require(_file_sha256(path) == record["sha256"], "cache MTP prefix hash mismatch")
    _verify_safetensors(path, record["tensors"])
    verify_elapsed = time.perf_counter() - verify_started
    materialize_started = time.perf_counter()
    arrays, metadata = mx.load(path, return_metadata=True)
    require(
        metadata
        == {
            "schema": STATE_SCHEMA,
            "kind": "mtp-prefix",
            "target_position": str(target_position),
            "mtp_position": str(target_position - 1),
        },
        "cache MTP prefix metadata mismatch",
    )
    require(set(arrays) == set(record["tensors"]), "cache MTP prefix array mismatch")
    mx.eval(*arrays.values())
    mx.synchronize()
    state = (
        attention.MLXAttentionState(keys=arrays["keys"], values=arrays["values"])
        if target_position > 1
        else attention.zeros_state(mtp_config.attention, dtype=mx.bfloat16)
    )
    prefix = mtp_runtime.MTPPrefixState(
        state=state,
        boundary_hidden=arrays["boundary_hidden"],
    )
    mtp_runtime.validate_prefix_state(
        prefix,
        target_position,
        mtp_config,
        dtype=mx.bfloat16,
    )
    return prefix, verify_elapsed, time.perf_counter() - materialize_started


def load_cache(
    path: Path,
    identity: CacheIdentity,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    *,
    expected_tokens: Sequence[int] | None = None,
    mtp_config: mtp_reference.MTPConfig = mtp.PRODUCTION_CONFIG,
) -> PersistentCache:
    """Verify every durable byte before returning an immutable model state."""
    load_started = time.perf_counter()
    validate_identity(identity)
    require(path.is_dir() and not path.is_symlink(), "cache entry is missing or unsafe")
    manifest_started = time.perf_counter()
    manifest = _read_manifest(path / MANIFEST_NAME)
    _validate_manifest_envelope(manifest, entry_name=path.name)
    require(
        manifest["schema"] == identity.state_schema,
        "cache manifest schema mismatch",
    )
    require(manifest["identity"] == asdict(identity), "cache identity mismatch")
    require(manifest["config_sha256"] == _config_sha256(config), "cache config mismatch")
    position = manifest["position"]
    token_record = manifest["tokens"]
    manifest_elapsed = time.perf_counter() - manifest_started
    tokens_started = time.perf_counter()
    tokens = _read_tokens(path / TOKENS_NAME, position, token_record["sha256"])
    require(
        all(token < config.vocab_size for token in tokens),
        "cached token ID is out of range",
    )
    if expected_tokens is not None:
        require(tokens == tuple(expected_tokens), "cache token prefix mismatch")
    tokens_elapsed = time.perf_counter() - tokens_started
    key = cache_key(tokens, identity, config)
    require(manifest["key"] == key, "cache key mismatch")
    records = manifest["files"]
    require(len(records) == len(config.layer_types), "cache layer file count mismatch")
    expected_names = {MANIFEST_NAME, TOKENS_NAME}
    states = []
    payload_verify_s = 0.0
    payload_materialize_s = 0.0
    payload_bytes = token_record["bytes"]
    for index, (kind, record) in enumerate(zip(config.layer_types, records)):
        name = f"layer-{index:03d}.safetensors"
        require(
            isinstance(record, dict)
            and set(record) == {"name", "layer", "kind", "bytes", "sha256", "tensors"}
            and record["name"] == name
            and record["layer"] == index
            and record["kind"] == kind
            and isinstance(record["bytes"], int)
            and record["bytes"] > 0
            and _is_sha256(record["sha256"])
            and isinstance(record["tensors"], dict),
            f"cache layer manifest mismatch at {index}",
        )
        require(
            record["tensors"] == _expected_tensor_specs(
                index,
                kind,
                position,
                config,
                identity,
            ),
            f"cache tensor manifest mismatch at {index}",
        )
        expected_names.add(name)
        layer_state, verify_s, materialize_s = _load_layer(
            path / name,
            record,
            position,
            identity.rope_profile,
            identity,
            config,
        )
        states.append(layer_state)
        payload_verify_s += verify_s
        payload_materialize_s += materialize_s
        payload_bytes += record["bytes"]
    mtp_prefix = None
    mtp_record = manifest["mtp"]
    if identity_uses_mtp(identity):
        require(
            isinstance(mtp_record, dict)
            and set(mtp_record)
            == {
                "name",
                "kind",
                "target_position",
                "mtp_position",
                "bytes",
                "sha256",
                "tensors",
            }
            and mtp_record["name"] == MTP_PREFIX_NAME
            and mtp_record["kind"] == "mtp-prefix"
            and mtp_record["target_position"] == position
            and mtp_record["mtp_position"] == position - 1
            and isinstance(mtp_record["bytes"], int)
            and mtp_record["bytes"] > 0
            and _is_sha256(mtp_record["sha256"])
            and isinstance(mtp_record["tensors"], dict),
            "cache MTP prefix manifest mismatch",
        )
        require(
            mtp_record["tensors"]
            == _expected_mtp_tensor_specs(position, mtp_config),
            "cache MTP tensor manifest mismatch",
        )
        expected_names.add(MTP_PREFIX_NAME)
        mtp_prefix, verify_s, materialize_s = _load_mtp_prefix(
            path / MTP_PREFIX_NAME,
            mtp_record,
            position,
            mtp_config,
        )
        payload_verify_s += verify_s
        payload_materialize_s += materialize_s
        payload_bytes += mtp_record["bytes"]
    else:
        require(mtp_record is None, "target-only cache unexpectedly contains MTP state")
    finalize_started = time.perf_counter()
    require(
        {entry.name for entry in path.iterdir()} == expected_names,
        "cache entry contains unexpected files",
    )
    state = model.TextModelState(
        position=position,
        layers=tuple(states),
        context_profile=identity.rope_profile,
    )
    model.validate_state(state, config)
    os.utime(path)
    finalize_elapsed = time.perf_counter() - finalize_started
    total_elapsed = time.perf_counter() - load_started
    return PersistentCache(
        path=path,
        key=key,
        identity=identity,
        token_ids=tokens,
        state=state,
        mtp_prefix=mtp_prefix,
        load_timing=CacheLoadTiming(
            manifest_s=manifest_elapsed,
            tokens_s=tokens_elapsed,
            payload_verify_s=payload_verify_s,
            payload_materialize_s=payload_materialize_s,
            finalize_s=finalize_elapsed,
            total_s=total_elapsed,
            payload_bytes=payload_bytes,
        ),
    )


def _entry_bytes(path: Path) -> int:
    total = 0
    for entry in path.iterdir():
        require(entry.is_file() and not entry.is_symlink(), "cache entry is not flat and immutable")
        total += entry.stat().st_size
    return total


def prune_cache(
    root: Path,
    *,
    max_bytes: int,
    max_entries: int,
    protect: Sequence[str] = (),
) -> CachePruneResult:
    """Remove oldest complete entries after atomically hiding each victim."""
    require(max_bytes > 0, "cache byte budget must be positive")
    require(max_entries > 0, "cache entry budget must be positive")
    if not root.exists():
        return CachePruneResult(0, 0, 0, 0, False)
    require(root.is_dir() and not root.is_symlink(), "cache root is missing or unsafe")
    protected = set(protect)
    require(all(_is_sha256(key) for key in protected), "invalid protected cache key")
    entries = []
    for path in root.iterdir():
        if path.name.startswith("."):
            continue
        require(
            path.is_dir() and not path.is_symlink() and _is_sha256(path.name),
            f"unexpected cache-root entry: {path.name}",
        )
        size = _entry_bytes(path)
        entries.append((path.stat().st_mtime_ns, path.name, path, size))
    retained_bytes = sum(entry[3] for entry in entries)
    retained_entries = len(entries)
    removed_entries = 0
    removed_bytes = 0
    for _, key, path, size in sorted(entries):
        if retained_bytes <= max_bytes and retained_entries <= max_entries:
            break
        if key in protected:
            continue
        hidden = root / f".{key}.delete-{uuid4().hex}"
        os.rename(path, hidden)
        _fsync_directory(root)
        shutil.rmtree(hidden)
        retained_bytes -= size
        retained_entries -= 1
        removed_entries += 1
        removed_bytes += size
    _fsync_directory(root)
    return CachePruneResult(
        removed_entries=removed_entries,
        removed_bytes=removed_bytes,
        retained_entries=retained_entries,
        retained_bytes=retained_bytes,
        over_budget=retained_bytes > max_bytes or retained_entries > max_entries,
    )

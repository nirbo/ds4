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
from typing import Any, Sequence
from uuid import uuid4

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import SafetensorsFile


STATE_SCHEMA = "ornith35-prefix-state-v1"
MANIFEST_NAME = "manifest.json"
TOKENS_NAME = "tokens.u32le"
_HASH_CHUNK = 8 * 1024 * 1024
_TOKEN_CHUNK = 8192
PRODUCTION_MODEL_ID = "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
PRODUCTION_MODEL_REVISION = "85ffd2d0629ae5fa4f860dda356ec33161806c9b"
PRODUCTION_RUNTIME_FILES = (
    "ornith35/tools/ornith35_mlx_nvfp4.py",
    "ornith35/tools/ornith35_mlx_gdn.py",
    "ornith35/tools/ornith35_mlx_attention.py",
    "ornith35/tools/ornith35_mlx_moe.py",
    "ornith35/tools/ornith35_mlx_layer.py",
    "ornith35/tools/ornith35_mlx_model.py",
    "ornith35/tools/ornith35_mlx_vocab.py",
    "ornith35/tools/ornith35_mlx_linear_cache.py",
    "ornith35/tools/ornith35_mlx_cache.py",
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
    state_schema: str = STATE_SCHEMA


@dataclass(frozen=True)
class PersistentCache:
    path: Path
    key: str
    identity: CacheIdentity
    token_ids: tuple[int, ...]
    state: model.TextModelState


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
    ):
        require(_is_sha256(getattr(identity, name)), f"invalid cache identity hash: {name}")
    require(identity.cache_dtype == "BF16", "persistent cache dtype must be BF16")
    require(identity.state_schema == STATE_SCHEMA, "persistent cache schema mismatch")


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
    descriptor = {
        "config_sha256": _config_sha256(config),
        "identity": asdict(identity),
        "position": len(tokens),
        "token_sha256": _token_sha256(tokens),
    }
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest()


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


def production_identity(
    model_root: Path,
    repo_root: Path,
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    mapped_embedding: bool,
    quantized_lm_head: bool,
    rope_profile: str = "native-262k",
) -> CacheIdentity:
    """Bind a cache entry to the verified source and exact local runtime bytes."""
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
    }
    identity = CacheIdentity(
        model_id=PRODUCTION_MODEL_ID,
        model_revision=PRODUCTION_MODEL_REVISION,
        source_sha256=source_sha256,
        runtime_revision=runtime_revision,
        runtime_sha256=_runtime_sha256(repo_root),
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=chat_template_sha256,
        quantization_policy_sha256=hashlib.sha256(_canonical_json(policy)).hexdigest(),
        rope_profile=rope_profile,
        cache_dtype="BF16",
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
) -> tuple[dict[str, mx.array], dict[str, str]]:
    metadata = {
        "schema": STATE_SCHEMA,
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


def _validate_persistable_state(
    state: model.TextModelState,
    config: model.TextModelConfig,
) -> None:
    require(state.position > 0, "persistent cache position must be positive")
    require(len(state.layers) == len(config.layer_types), "cache layer count mismatch")
    for index, (kind, layer_state) in enumerate(zip(config.layer_types, state.layers)):
        if kind == model.LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            gdn.validate_state(layer_state, config.gdn)
            require(layer_state.conv.dtype == mx.bfloat16, f"GDN cache dtype mismatch at {index}")
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
    kind: str,
    position: int,
    config: model.TextModelConfig,
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
            "bytes": math.prod(shape) * (2 if dtypes[name] == "BF16" else 4),
        }
        for name, shape in shapes.items()
    }


def save_cache(
    root: Path,
    token_ids: Sequence[int],
    state: model.TextModelState,
    identity: CacheIdentity,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
) -> Path:
    """Write one immutable cache entry and publish it with an atomic rename."""
    validate_identity(identity)
    tokens = tuple(token_ids)
    require(len(tokens) == state.position, "cache token/state position mismatch")
    require(
        all(isinstance(token, int) and 0 <= token < config.vocab_size for token in tokens),
        "persistent cache token ID is out of range",
    )
    _validate_persistable_state(state, config)
    key = cache_key(tokens, identity, config)
    root.mkdir(parents=True, exist_ok=True)
    final = root / key
    if final.exists():
        load_cache(final, identity, config, expected_tokens=tokens)
        return final
    staging = root / f".{key}.part-{os.getpid()}-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        token_record = _write_tokens(staging / TOKENS_NAME, tokens)
        files = []
        for index, (kind, layer_state) in enumerate(zip(config.layer_types, state.layers)):
            arrays, metadata = _layer_payload(index, kind, layer_state, state.position)
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
        manifest = {
            "schema": STATE_SCHEMA,
            "key": key,
            "identity": asdict(identity),
            "config_sha256": _config_sha256(config),
            "position": state.position,
            "token_count": len(tokens),
            "tokens": token_record,
            "files": files,
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
            load_cache(final, identity, config, expected_tokens=tokens)
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


def _load_layer(
    path: Path,
    record: dict[str, Any],
    position: int,
) -> model.LayerState:
    require(path.is_file() and not path.is_symlink(), f"cache layer is missing or unsafe: {path.name}")
    require(path.stat().st_size == record["bytes"], f"cache layer size mismatch: {path.name}")
    require(_file_sha256(path) == record["sha256"], f"cache layer hash mismatch: {path.name}")
    _verify_safetensors(path, record["tensors"])
    arrays, metadata = mx.load(path, return_metadata=True)
    require(
        metadata
        == {
            "schema": STATE_SCHEMA,
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
        return gdn.MLXGDNState(conv=arrays["conv"], recurrent=arrays["recurrent"])
    return attention.MLXAttentionState(keys=arrays["keys"], values=arrays["values"])


def load_cache(
    path: Path,
    identity: CacheIdentity,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    *,
    expected_tokens: Sequence[int] | None = None,
) -> PersistentCache:
    """Verify every durable byte before returning an immutable model state."""
    validate_identity(identity)
    require(path.is_dir() and not path.is_symlink(), "cache entry is missing or unsafe")
    manifest = _read_manifest(path / MANIFEST_NAME)
    required = {
        "schema",
        "key",
        "identity",
        "config_sha256",
        "position",
        "token_count",
        "tokens",
        "files",
    }
    require(set(manifest) == required, "cache manifest fields mismatch")
    require(manifest["schema"] == STATE_SCHEMA, "cache manifest schema mismatch")
    require(manifest["identity"] == asdict(identity), "cache identity mismatch")
    require(manifest["config_sha256"] == _config_sha256(config), "cache config mismatch")
    position = manifest["position"]
    require(isinstance(position, int) and position > 0, "cache position is invalid")
    require(manifest["token_count"] == position, "cache token position mismatch")
    token_record = manifest["tokens"]
    require(
        isinstance(token_record, dict)
        and set(token_record) == {"name", "bytes", "sha256"}
        and token_record["name"] == TOKENS_NAME
        and token_record["bytes"] == position * 4
        and _is_sha256(token_record["sha256"]),
        "cache token manifest mismatch",
    )
    tokens = _read_tokens(path / TOKENS_NAME, position, token_record["sha256"])
    require(
        all(token < config.vocab_size for token in tokens),
        "cached token ID is out of range",
    )
    if expected_tokens is not None:
        require(tokens == tuple(expected_tokens), "cache token prefix mismatch")
    key = cache_key(tokens, identity, config)
    require(manifest["key"] == key and path.name == key, "cache key mismatch")
    records = manifest["files"]
    require(isinstance(records, list), "cache layer manifest is invalid")
    require(len(records) == len(config.layer_types), "cache layer file count mismatch")
    expected_names = {MANIFEST_NAME, TOKENS_NAME}
    states = []
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
            record["tensors"] == _expected_tensor_specs(kind, position, config),
            f"cache tensor manifest mismatch at {index}",
        )
        expected_names.add(name)
        states.append(_load_layer(path / name, record, position))
    require(
        {entry.name for entry in path.iterdir()} == expected_names,
        "cache entry contains unexpected files",
    )
    state = model.TextModelState(position=position, layers=tuple(states))
    model.validate_state(state, config)
    os.utime(path)
    return PersistentCache(
        path=path,
        key=key,
        identity=identity,
        token_ids=tokens,
        state=state,
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

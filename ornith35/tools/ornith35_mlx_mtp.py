#!/usr/bin/env python3
"""Strict BF16 Qwen3.5 MTP sidecar and MLX composition for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_layer as layer
import ornith35_mtp_reference as reference
import ornith35_nvfp4 as nvfp4
from ornith35_moe_reference import MoEConfig, require
from ornith35_nvfp4 import DEFAULT_ROOT, SafetensorsFile


STATE_FORMAT = "ornith35-mtp-extract-v1"
EXPECTED_REPOSITORY = "Qwen/Qwen3.5-35B-A3B"
EXPECTED_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXPECTED_RUNTIME_REVISION = "69c0d8f4cab717389a65afd4d021641263ed0a18"
EXPECTED_METADATA_STATE_SHA256 = (
    "f06507aa425ee6ce8d654fea311f499e1f2e599ead3fe5211e630002aac800ea"
)
EXPECTED_TOOL_SHA256 = (
    "8d1b3f2a3cc5646b8a43440b1f7e268e17e8e9ad7adc3e65f0390bc1929031a9"
)
EXPECTED_SIDECAR_BYTES = 1_689_376_064
EXPECTED_HEADER_BYTES = 94_520
EXPECTED_HEADER_SHA256 = (
    "1cfe6e40b9dacadc8d9f65f65abebd29aa611f9f2395330c406ab05a3c428559"
)
EXPECTED_PAYLOAD_BYTES = 1_689_281_536
EXPECTED_TENSOR_COUNT = 785
EXPECTED_SIDECAR_SHA256 = (
    "11c9043bf0c92c1eea7b4c6ffbadeb890a080a84d301872a29839be209099c1f"
)
EXPECTED_FC_SHA256 = (
    "484d48f41aff830601b575fa76af975cee459550efe55099d80dbb2fc1400910"
)
ADAPTATION_FORMAT = "ornith35-mtp-fc-adaptation-v1"
ADAPTATION_ARTIFACT = "mtp-fc.safetensors"
EXPECTED_SHARDS = {
    "model.safetensors-00013-of-00014.safetensors": (
        3,
        67_108_864,
        "da8cc27a2a99eeba5674bfef03c80b3c4f08edfd839ad319a7702fcaedfd874f",
    ),
    "model.safetensors-00014-of-00014.safetensors": (
        782,
        1_622_172_672,
        "d5e08a7dd670d7ef8da38e7af29cb8cdf42ecded5dfb2d9fcaf06cae79dfaa41",
    ),
}


PRODUCTION_CONFIG = reference.PRODUCTION_CONFIG


@dataclass(frozen=True)
class DenseExpertArrays:
    gate: mx.array
    up: mx.array
    down: mx.array


@dataclass(frozen=True)
class DenseExpertStack:
    gate: mx.array
    up: mx.array
    down: mx.array


@dataclass(frozen=True)
class MLXDenseMoEWeights:
    router_shared: mx.array
    experts: DenseExpertStack
    shared_expert: DenseExpertArrays

    @property
    def router(self) -> mx.array:
        return self.router_shared[:-1]

    @property
    def shared_gate(self) -> mx.array:
        return self.router_shared[-1:]


@dataclass(frozen=True)
class MLXMTPWeights:
    fc: mx.array
    pre_fc_norm_embedding: mx.array
    pre_fc_norm_hidden: mx.array
    input_layernorm: mx.array
    attention: attention.MLXAttentionWeights
    moe: MLXDenseMoEWeights
    post_attention_layernorm: mx.array
    norm: mx.array


@dataclass(frozen=True)
class MLXDenseMoEResult:
    output: mx.array
    selected_experts: mx.array
    routing_weights: mx.array


@dataclass(frozen=True)
class MLXMTPResult:
    hidden: mx.array
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState
    selected_experts: mx.array
    routing_weights: mx.array


@dataclass(frozen=True)
class MLXMTPChunkResult:
    hidden: mx.array
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState
    selected_experts: mx.array
    routing_weights: mx.array


def expected_tensor_shapes(config: reference.MTPConfig = PRODUCTION_CONFIG) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    intermediate = config.moe.intermediate_size
    prefix = "mtp.layers.0"
    values: dict[str, tuple[int, ...]] = {
        "mtp.fc.weight": (hidden, hidden * 2),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
        f"{prefix}.input_layernorm.weight": (hidden,),
        f"{prefix}.self_attn.q_proj.weight": (
            config.attention.query_dim * 2,
            hidden,
        ),
        f"{prefix}.self_attn.k_proj.weight": (config.attention.kv_dim, hidden),
        f"{prefix}.self_attn.v_proj.weight": (config.attention.kv_dim, hidden),
        f"{prefix}.self_attn.o_proj.weight": (hidden, config.attention.query_dim),
        f"{prefix}.self_attn.q_norm.weight": (config.attention.head_dim,),
        f"{prefix}.self_attn.k_norm.weight": (config.attention.head_dim,),
        f"{prefix}.mlp.gate.weight": (config.moe.num_experts, hidden),
        f"{prefix}.mlp.shared_expert_gate.weight": (1, hidden),
        f"{prefix}.mlp.shared_expert.gate_proj.weight": (intermediate, hidden),
        f"{prefix}.mlp.shared_expert.up_proj.weight": (intermediate, hidden),
        f"{prefix}.mlp.shared_expert.down_proj.weight": (hidden, intermediate),
        f"{prefix}.post_attention_layernorm.weight": (hidden,),
        "mtp.norm.weight": (hidden,),
    }
    for expert in range(config.moe.num_experts):
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        values[f"{expert_prefix}.gate_proj.weight"] = (intermediate, hidden)
        values[f"{expert_prefix}.up_proj.weight"] = (intermediate, hidden)
        values[f"{expert_prefix}.down_proj.weight"] = (hidden, intermediate)
    return values


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read MTP state {path}: {exc}") from exc
    require(isinstance(value, dict), "MTP state is not a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_extraction_state(state: dict[str, Any]) -> None:
    require(state.get("format") == STATE_FORMAT, "unsupported MTP state format")
    require(state.get("status") == "complete", "MTP extraction is incomplete")
    require(state.get("profile") == "mtp-source", "MTP extraction profile mismatch")
    require(state.get("repository") == EXPECTED_REPOSITORY, "MTP repository mismatch")
    require(state.get("revision") == EXPECTED_REVISION, "MTP revision mismatch")
    require(
        state.get("runtime_revision") == EXPECTED_RUNTIME_REVISION,
        "MTP extraction runtime revision mismatch",
    )
    require(
        state.get("metadata_state_sha256") == EXPECTED_METADATA_STATE_SHA256,
        "MTP metadata identity mismatch",
    )
    require(state.get("tool_sha256") == EXPECTED_TOOL_SHA256, "MTP extractor identity mismatch")
    require(
        state.get("output_sha256") == EXPECTED_SIDECAR_SHA256,
        "MTP state output hash mismatch",
    )
    output = state.get("output")
    require(isinstance(output, dict), "MTP state output is absent")
    require(output.get("name") == "mtp.safetensors", "MTP sidecar name mismatch")
    require(output.get("bytes") == EXPECTED_SIDECAR_BYTES, "MTP sidecar size mismatch")
    require(output.get("header_bytes") == EXPECTED_HEADER_BYTES, "MTP header size mismatch")
    require(
        output.get("header_sha256") == EXPECTED_HEADER_SHA256,
        "MTP header identity mismatch",
    )
    require(output.get("payload_bytes") == EXPECTED_PAYLOAD_BYTES, "MTP payload size mismatch")
    require(output.get("tensor_count") == EXPECTED_TENSOR_COUNT, "MTP tensor count mismatch")

    completed = state.get("completed_shards")
    require(isinstance(completed, dict), "MTP completed-shard state is absent")
    require(set(completed) == set(EXPECTED_SHARDS), "MTP completed-shard set mismatch")
    tensor_hash_names: set[str] = set()
    for shard_name, (tensor_count, payload_bytes, source_sha256) in EXPECTED_SHARDS.items():
        shard = completed[shard_name]
        require(isinstance(shard, dict), f"invalid MTP shard state: {shard_name}")
        require(shard.get("tensor_count") == tensor_count, f"MTP shard tensor count mismatch: {shard_name}")
        require(shard.get("payload_bytes") == payload_bytes, f"MTP shard payload mismatch: {shard_name}")
        require(shard.get("source_sha256") == source_sha256, f"MTP source hash mismatch: {shard_name}")
        tensor_hashes = shard.get("tensor_sha256")
        require(isinstance(tensor_hashes, dict), f"MTP tensor hashes absent: {shard_name}")
        require(len(tensor_hashes) == tensor_count, f"MTP tensor hash count mismatch: {shard_name}")
        require(
            all(isinstance(name, str) and _is_sha256(value) for name, value in tensor_hashes.items()),
            f"invalid MTP tensor hash: {shard_name}",
        )
        require(
            tensor_hash_names.isdisjoint(tensor_hashes),
            f"duplicate MTP tensor state: {shard_name}",
        )
        tensor_hash_names.update(tensor_hashes)
    require(
        tensor_hash_names == set(expected_tensor_shapes()),
        "MTP state tensor schema mismatch",
    )


def _validate_sidecar_schema(path: Path) -> None:
    expected = expected_tensor_shapes()
    with SafetensorsFile(path) as source:
        require(len(source.tensors) == EXPECTED_TENSOR_COUNT, "MTP header tensor count mismatch")
        require(set(source.tensors) == set(expected), "MTP header tensor schema mismatch")
        require(source.header_bytes == EXPECTED_HEADER_BYTES, "MTP safetensors header size mismatch")
        require(
            source.payload_offset + EXPECTED_PAYLOAD_BYTES == path.stat().st_size,
            "MTP safetensors payload extent mismatch",
        )
        ranges = []
        for name, shape in expected.items():
            entry = source.entry(name)
            require(entry.get("dtype") == "BF16", f"MTP tensor is not BF16: {name}")
            require(entry.get("shape") == list(shape), f"MTP tensor shape mismatch: {name}")
            expected_bytes = 2
            for dimension in shape:
                expected_bytes *= dimension
            require(source.tensor_nbytes(name) == expected_bytes, f"MTP tensor size mismatch: {name}")
            ranges.append(tuple(entry["data_offsets"]))
        ranges.sort()
        cursor = 0
        for start, end in ranges:
            require(start == cursor and end > start, "MTP tensor payload is not contiguous")
            cursor = end
        require(cursor == EXPECTED_PAYLOAD_BYTES, "MTP tensor payload is incomplete")


def load_fc_adaptation(
    directory: Path,
    source_fc: mx.array,
    *,
    allow_diagnostic: bool = False,
) -> mx.array:
    """Load one provenance-bound replacement for only the MTP fusion projection."""
    require(directory.is_dir() and not directory.is_symlink(), "MTP adaptation directory is absent")
    state_path = directory / "state.json"
    require(state_path.is_file() and not state_path.is_symlink(), "MTP adaptation state is absent")
    state = _load_json(state_path)
    require(state.get("format") == ADAPTATION_FORMAT, "MTP adaptation format mismatch")
    allowed_status = {"candidate", "accepted"}
    if allow_diagnostic:
        allowed_status.update(("diagnostic", "rejected"))
    require(state.get("status") in allowed_status, "MTP adaptation is not runtime-eligible")
    source = state.get("source")
    require(isinstance(source, dict), "MTP adaptation source identity is absent")
    require(
        source.get("target_weight_sha256") == nvfp4.EXPECTED_WEIGHT_SHA256,
        "MTP adaptation target identity mismatch",
    )
    require(
        source.get("mtp_sidecar_sha256") == EXPECTED_SIDECAR_SHA256,
        "MTP adaptation sidecar identity mismatch",
    )
    require(
        source.get("mtp_fc_sha256") == EXPECTED_FC_SHA256,
        "MTP adaptation base projection mismatch",
    )
    artifact = state.get("artifact")
    require(isinstance(artifact, dict), "MTP adaptation artifact record is absent")
    require(artifact.get("name") == ADAPTATION_ARTIFACT, "MTP adaptation artifact name mismatch")
    path = directory / ADAPTATION_ARTIFACT
    require(path.is_file() and not path.is_symlink(), "MTP adaptation artifact is absent")
    require(path.stat().st_size == artifact.get("bytes"), "MTP adaptation byte size mismatch")
    require(_sha256_file(path) == artifact.get("sha256"), "MTP adaptation SHA-256 mismatch")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    require(set(arrays) == {"mtp.fc.weight"}, "MTP adaptation tensor set mismatch")
    require(metadata.get("format") == ADAPTATION_FORMAT, "MTP adaptation metadata mismatch")
    require(
        metadata.get("base_mtp_sha256") == EXPECTED_SIDECAR_SHA256,
        "MTP adaptation metadata source mismatch",
    )
    adapted = arrays["mtp.fc.weight"]
    require(
        adapted.dtype == mx.bfloat16 and adapted.shape == source_fc.shape,
        "MTP adapted projection tensor mismatch",
    )
    require(
        source_fc.dtype == mx.bfloat16 and source_fc.shape == (2048, 4096),
        "MTP source projection tensor mismatch",
    )
    mx.eval(adapted)
    return adapted


def require_verified_mtp_sidecar(
    root: Path = DEFAULT_ROOT,
    *,
    verify_hash: bool = True,
) -> Path:
    """Accept only the exact revision-bound MTP extraction artifact."""
    state_path = root / "source-mtp-state.json"
    require(state_path.is_file() and not state_path.is_symlink(), "verified MTP state is absent")
    state = _load_json(state_path)
    _validate_extraction_state(state)
    sidecar = root / "source-mtp" / "mtp.safetensors"
    require(sidecar.is_file() and not sidecar.is_symlink(), "MTP sidecar is absent")
    require(sidecar.stat().st_size == EXPECTED_SIDECAR_BYTES, "MTP sidecar byte size mismatch")
    if verify_hash:
        require(_sha256_file(sidecar) == EXPECTED_SIDECAR_SHA256, "MTP sidecar SHA-256 mismatch")
    with sidecar.open("rb") as handle:
        encoded_size = handle.read(8)
        header = handle.read(EXPECTED_HEADER_BYTES)
    require(len(encoded_size) == 8 and len(header) == EXPECTED_HEADER_BYTES, "truncated MTP header")
    require(
        int.from_bytes(encoded_size, "little") == EXPECTED_HEADER_BYTES,
        "MTP encoded header size mismatch",
    )
    require(hashlib.sha256(header).hexdigest() == EXPECTED_HEADER_SHA256, "MTP header SHA-256 mismatch")
    _validate_sidecar_schema(sidecar)
    return sidecar


def _validate_expert_arrays(
    expert: DenseExpertArrays,
    config: MoEConfig,
    name: str,
    dtype: mx.Dtype,
) -> None:
    require(
        expert.gate.shape == (config.intermediate_size, config.hidden_size),
        f"{name} gate shape mismatch",
    )
    require(expert.up.shape == expert.gate.shape, f"{name} up shape mismatch")
    require(
        expert.down.shape == (config.hidden_size, config.intermediate_size),
        f"{name} down shape mismatch",
    )
    require(
        expert.gate.dtype == expert.up.dtype == expert.down.dtype == dtype,
        f"{name} dtype mismatch",
    )


def validate_moe_weights(weights: MLXDenseMoEWeights, config: MoEConfig) -> None:
    dtype = weights.router_shared.dtype
    require(dtype in (mx.bfloat16, mx.float32), "invalid MTP MoE dtype")
    require(
        weights.router_shared.shape == (config.num_experts + 1, config.hidden_size),
        "MTP router/shared-gate shape mismatch",
    )
    expected_gate = (config.num_experts, config.intermediate_size, config.hidden_size)
    expected_down = (config.num_experts, config.hidden_size, config.intermediate_size)
    require(weights.experts.gate.shape == expected_gate, "MTP expert gate stack mismatch")
    require(weights.experts.up.shape == expected_gate, "MTP expert up stack mismatch")
    require(weights.experts.down.shape == expected_down, "MTP expert down stack mismatch")
    require(
        weights.experts.gate.dtype
        == weights.experts.up.dtype
        == weights.experts.down.dtype
        == dtype,
        "MTP expert stack dtype mismatch",
    )
    _validate_expert_arrays(weights.shared_expert, config, "MTP shared expert", dtype)


def validate_weights(weights: MLXMTPWeights, config: reference.MTPConfig) -> None:
    dtype = weights.fc.dtype
    require(dtype in (mx.bfloat16, mx.float32), "invalid MTP model dtype")
    require(weights.fc.shape == (config.hidden_size, config.hidden_size * 2), "MTP fc shape mismatch")
    for name in (
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ):
        value = getattr(weights, name)
        require(value.shape == (config.hidden_size,), f"MTP {name} shape mismatch")
        require(value.dtype == dtype, f"MTP {name} dtype mismatch")
    attention.validate_weights(weights.attention, config.attention)
    require(weights.attention.q_proj.dtype == dtype, "MTP attention dtype mismatch")
    validate_moe_weights(weights.moe, config.moe)
    require(weights.moe.router.dtype == dtype, "MTP MoE/model dtype mismatch")


def _route_token(logits: mx.array, top_k: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    selected = mx.argsort(probabilities)[-top_k:][::-1]
    routing = mx.take(probabilities, selected)
    return selected, (routing / mx.sum(routing)).astype(dtype)


def _dense_expert(hidden: mx.array, weights: DenseExpertArrays) -> mx.array:
    gate = mx.matmul(weights.gate, hidden)
    up = mx.matmul(weights.up, hidden)
    return mx.matmul(weights.down, gate * mx.sigmoid(gate) * up)


def forward_moe(
    hidden: mx.array,
    weights: MLXDenseMoEWeights,
    config: MoEConfig,
    *,
    _validated: bool = False,
) -> MLXDenseMoEResult:
    require(hidden.ndim == 1 and hidden.shape == (config.hidden_size,), "MTP MoE input mismatch")
    if not _validated:
        validate_moe_weights(weights, config)
    dtype = weights.router.dtype
    hidden = hidden.astype(dtype)
    router_shared = mx.matmul(weights.router_shared, hidden)
    selected, routing = _route_token(router_shared[:-1], config.top_k, dtype)

    gate_weights = mx.take(weights.experts.gate, selected, axis=0)
    up_weights = mx.take(weights.experts.up, selected, axis=0)
    down_weights = mx.take(weights.experts.down, selected, axis=0)
    gate = mx.matmul(gate_weights, hidden)
    up = mx.matmul(up_weights, hidden)
    intermediate = gate * mx.sigmoid(gate) * up
    expert_outputs = mx.matmul(down_weights, intermediate[..., None]).squeeze(-1)
    routed = mx.sum(expert_outputs * routing[:, None], axis=0)
    shared = _dense_expert(hidden, weights.shared_expert)
    shared_multiplier = mx.sigmoid(router_shared[-1])
    output = (routed + shared * shared_multiplier).astype(dtype)
    return MLXDenseMoEResult(
        output=output,
        selected_experts=selected,
        routing_weights=routing,
    )


def forward_moe_batch(
    hidden: mx.array,
    weights: MLXDenseMoEWeights,
    config: MoEConfig,
    *,
    _validated: bool = False,
) -> MLXDenseMoEResult:
    """Evaluate a nonempty token batch while keeping expert IDs on Metal."""
    require(
        hidden.ndim == 2
        and hidden.shape[0] > 0
        and hidden.shape[1] == config.hidden_size,
        "MTP batched MoE input mismatch",
    )
    if not _validated:
        validate_moe_weights(weights, config)
    dtype = weights.router.dtype
    hidden = hidden.astype(dtype)
    router_shared = mx.matmul(hidden, mx.transpose(weights.router_shared))
    probabilities = mx.softmax(router_shared[:, :-1].astype(mx.float32), axis=-1)
    selected = mx.argsort(probabilities, axis=-1)[:, -config.top_k :][:, ::-1]
    routing = mx.take_along_axis(probabilities, selected, axis=-1)
    routing = (routing / mx.sum(routing, axis=-1, keepdims=True)).astype(dtype)

    gate_weights = mx.take(weights.experts.gate, selected, axis=0)
    up_weights = mx.take(weights.experts.up, selected, axis=0)
    down_weights = mx.take(weights.experts.down, selected, axis=0)
    expanded_hidden = hidden[:, None, :, None]
    gate = mx.matmul(gate_weights, expanded_hidden).squeeze(-1)
    up = mx.matmul(up_weights, expanded_hidden).squeeze(-1)
    intermediate = gate * mx.sigmoid(gate) * up
    expert_outputs = mx.matmul(down_weights, intermediate[..., None]).squeeze(-1)
    routed = mx.sum(expert_outputs * routing[:, :, None], axis=1)

    shared_gate = mx.matmul(hidden, mx.transpose(weights.shared_expert.gate))
    shared_up = mx.matmul(hidden, mx.transpose(weights.shared_expert.up))
    shared_intermediate = shared_gate * mx.sigmoid(shared_gate) * shared_up
    shared = mx.matmul(
        shared_intermediate,
        mx.transpose(weights.shared_expert.down),
    )
    shared_multiplier = mx.sigmoid(router_shared[:, -1:])
    output = (routed + shared * shared_multiplier).astype(dtype)
    return MLXDenseMoEResult(
        output=output,
        selected_experts=selected,
        routing_weights=routing,
    )


def initial_state(
    weights: MLXMTPWeights,
    config: reference.MTPConfig,
) -> attention.MLXAttentionState:
    validate_weights(weights, config)
    return attention.zeros_state(config.attention, dtype=weights.fc.dtype)


def forward_step(
    next_token_embedding: mx.array,
    target_hidden: mx.array,
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState,
    weights: MLXMTPWeights,
    config: reference.MTPConfig,
    *,
    _validated: bool = False,
) -> MLXMTPResult:
    """Evaluate one advancing MTP position without a CPU routing boundary."""
    require(
        next_token_embedding.ndim == 1
        and next_token_embedding.shape == (config.hidden_size,),
        "MTP token embedding mismatch",
    )
    require(
        target_hidden.ndim == 1 and target_hidden.shape == (config.hidden_size,),
        "MTP target hidden mismatch",
    )
    if not _validated:
        validate_weights(weights, config)
        require(
            attention.state_length(state, config.attention) >= 0,
            "invalid MTP attention state",
        )
    dtype = weights.fc.dtype
    normalized_embedding = layer.qwen_rms_norm(
        next_token_embedding.astype(dtype),
        weights.pre_fc_norm_embedding,
        config.rms_norm_eps,
    )
    normalized_hidden = layer.qwen_rms_norm(
        target_hidden.astype(dtype),
        weights.pre_fc_norm_hidden,
        config.rms_norm_eps,
    )
    hidden = mx.matmul(
        weights.fc,
        mx.concatenate((normalized_embedding, normalized_hidden)),
    ).astype(dtype)
    attention_input = layer.qwen_rms_norm(
        hidden,
        weights.input_layernorm,
        config.rms_norm_eps,
    )
    mixed, next_state = attention.decode_step(
        attention_input,
        state,
        weights.attention,
        config.attention,
        _validated=_validated,
    )
    hidden, moe_input = layer.residual_and_rms_norm(
        hidden,
        mixed,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
        fused_rmsnorm=True,
        fused_mean_square=True,
    )
    moe_result = forward_moe(
        moe_input,
        weights.moe,
        config.moe,
        _validated=_validated,
    )
    _, normalized = layer.residual_and_rms_norm(
        hidden,
        moe_result.output,
        weights.norm,
        config.rms_norm_eps,
        fused_rmsnorm=True,
        fused_mean_square=True,
    )
    return MLXMTPResult(
        hidden=normalized,
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
    )


def prefill_steps(
    next_token_embeddings: mx.array,
    target_hidden_states: mx.array,
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState,
    weights: MLXMTPWeights,
    config: reference.MTPConfig,
    *,
    exact_long_attention: bool = True,
    _validated: bool = False,
) -> MLXMTPChunkResult:
    """Evaluate known target-authoritative MTP rows as one causal chunk."""
    require(
        next_token_embeddings.ndim == 2
        and next_token_embeddings.shape[0] > 0
        and next_token_embeddings.shape[1] == config.hidden_size,
        "MTP embedding chunk mismatch",
    )
    require(
        target_hidden_states.shape == next_token_embeddings.shape,
        "MTP target-hidden chunk mismatch",
    )
    if not _validated:
        validate_weights(weights, config)
        require(
            attention.state_length(state, config.attention) >= 0,
            "invalid MTP chunk state",
        )
    dtype = weights.fc.dtype
    normalized_embedding = layer.qwen_rms_norm_batch(
        next_token_embeddings.astype(dtype),
        weights.pre_fc_norm_embedding,
        config.rms_norm_eps,
    )
    normalized_hidden = layer.qwen_rms_norm_batch(
        target_hidden_states.astype(dtype),
        weights.pre_fc_norm_hidden,
        config.rms_norm_eps,
    )
    fused_input = mx.concatenate((normalized_embedding, normalized_hidden), axis=1)
    hidden = mx.matmul(fused_input, mx.transpose(weights.fc)).astype(dtype)
    attention_input = layer.qwen_rms_norm_batch(
        hidden,
        weights.input_layernorm,
        config.rms_norm_eps,
    )
    mixed, next_state = attention.prefill_chunk(
        attention_input,
        state,
        weights.attention,
        config.attention,
        use_steel=False,
        grouped_gqa=True,
        exact_long_prefill=exact_long_attention,
    )
    hidden, moe_input = layer.residual_and_rms_norm_batch(
        hidden,
        mixed,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
    )
    moe_result = forward_moe_batch(
        moe_input,
        weights.moe,
        config.moe,
        _validated=_validated,
    )
    _, normalized = layer.residual_and_rms_norm_batch(
        hidden,
        moe_result.output,
        weights.norm,
        config.rms_norm_eps,
    )
    return MLXMTPChunkResult(
        hidden=normalized,
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
    )


def _load_bf16(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "BF16", f"expected BF16 MTP tensor: {name}")
    require(entry.get("shape") == list(shape), f"MTP tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    expected_bytes = 2
    for dimension in shape:
        expected_bytes *= dimension
    require(len(payload) == expected_bytes, f"MTP tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.bfloat16).reshape(shape)


def _load_expert_projection(
    source: SafetensorsFile,
    projection: str,
    shape: tuple[int, ...],
    config: MoEConfig,
) -> mx.array:
    arrays = [
        _load_bf16(
            source,
            f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight",
            shape,
        )
        for expert in range(config.num_experts)
    ]
    stacked = mx.stack(arrays)
    mx.eval(stacked)
    return stacked


def load_weights(
    root: Path = DEFAULT_ROOT,
    config: reference.MTPConfig = PRODUCTION_CONFIG,
    *,
    verify_hash: bool = True,
    adaptation_dir: Path | None = None,
    allow_diagnostic_adaptation: bool = False,
) -> MLXMTPWeights:
    """Load the exact verified sidecar without target embedding/head copies."""
    require(config == PRODUCTION_CONFIG, "production MTP loader requires the pinned config")
    path = require_verified_mtp_sidecar(root, verify_hash=verify_hash)
    hidden = config.hidden_size
    intermediate = config.moe.intermediate_size
    prefix = "mtp.layers.0"
    with SafetensorsFile(path) as source:
        expert_gate = _load_expert_projection(
            source,
            "gate_proj",
            (intermediate, hidden),
            config.moe,
        )
        expert_up = _load_expert_projection(
            source,
            "up_proj",
            (intermediate, hidden),
            config.moe,
        )
        expert_down = _load_expert_projection(
            source,
            "down_proj",
            (hidden, intermediate),
            config.moe,
        )
        router = _load_bf16(
            source,
            f"{prefix}.mlp.gate.weight",
            (config.moe.num_experts, hidden),
        )
        shared_gate = _load_bf16(
            source,
            f"{prefix}.mlp.shared_expert_gate.weight",
            (1, hidden),
        )
        router_shared = mx.concatenate((router, shared_gate), axis=0)
        weights = MLXMTPWeights(
            fc=_load_bf16(source, "mtp.fc.weight", (hidden, hidden * 2)),
            pre_fc_norm_embedding=_load_bf16(
                source,
                "mtp.pre_fc_norm_embedding.weight",
                (hidden,),
            ),
            pre_fc_norm_hidden=_load_bf16(
                source,
                "mtp.pre_fc_norm_hidden.weight",
                (hidden,),
            ),
            input_layernorm=_load_bf16(
                source,
                f"{prefix}.input_layernorm.weight",
                (hidden,),
            ),
            attention=attention.MLXAttentionWeights(
                q_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.q_proj.weight",
                    (config.attention.query_dim * 2, hidden),
                ),
                k_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.k_proj.weight",
                    (config.attention.kv_dim, hidden),
                ),
                v_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.v_proj.weight",
                    (config.attention.kv_dim, hidden),
                ),
                o_proj=_load_bf16(
                    source,
                    f"{prefix}.self_attn.o_proj.weight",
                    (hidden, config.attention.query_dim),
                ),
                q_norm=_load_bf16(
                    source,
                    f"{prefix}.self_attn.q_norm.weight",
                    (config.attention.head_dim,),
                ),
                k_norm=_load_bf16(
                    source,
                    f"{prefix}.self_attn.k_norm.weight",
                    (config.attention.head_dim,),
                ),
            ),
            moe=MLXDenseMoEWeights(
                router_shared=router_shared,
                experts=DenseExpertStack(
                    gate=expert_gate,
                    up=expert_up,
                    down=expert_down,
                ),
                shared_expert=DenseExpertArrays(
                    gate=_load_bf16(
                        source,
                        f"{prefix}.mlp.shared_expert.gate_proj.weight",
                        (intermediate, hidden),
                    ),
                    up=_load_bf16(
                        source,
                        f"{prefix}.mlp.shared_expert.up_proj.weight",
                        (intermediate, hidden),
                    ),
                    down=_load_bf16(
                        source,
                        f"{prefix}.mlp.shared_expert.down_proj.weight",
                        (hidden, intermediate),
                    ),
                ),
            ),
            post_attention_layernorm=_load_bf16(
                source,
                f"{prefix}.post_attention_layernorm.weight",
                (hidden,),
            ),
            norm=_load_bf16(source, "mtp.norm.weight", (hidden,)),
        )
        arrays = [
            weights.fc,
            weights.pre_fc_norm_embedding,
            weights.pre_fc_norm_hidden,
            weights.input_layernorm,
            *weights.attention.__dict__.values(),
            weights.moe.router_shared,
            weights.moe.experts.gate,
            weights.moe.experts.up,
            weights.moe.experts.down,
            *weights.moe.shared_expert.__dict__.values(),
            weights.post_attention_layernorm,
            weights.norm,
        ]
        mx.eval(*arrays)
    if adaptation_dir is not None:
        weights = replace(
            weights,
            fc=load_fc_adaptation(
                adaptation_dir,
                weights.fc,
                allow_diagnostic=allow_diagnostic_adaptation,
            ),
        )
    validate_weights(weights, config)
    return weights

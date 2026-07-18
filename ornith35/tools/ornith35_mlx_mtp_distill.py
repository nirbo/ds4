#!/usr/bin/env python3
"""Distill a zero-overhead Ornith-specific update into the official MTP layer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Iterable

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_layer as layer
import ornith35_mlx_model as model
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_teacher_capture as capture
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError, require
import ornith35_nvfp4 as nvfp4
from ornith35_tokenizer import DEFAULT_ROOT


FORMAT = "ornith35-mtp-fc-distillation-v1"
CHECKPOINT_FORMAT = "ornith35-mtp-fc-distillation-checkpoint-v1"


@dataclass(frozen=True)
class TraceShard:
    prompt_index: int
    path: Path
    rows: int
    scored_rows: int


@dataclass
class AdamW:
    learning_rate: float
    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    step: int = 0
    first: dict[str, mx.array] | None = None
    second: dict[str, mx.array] | None = None

    def initialize(self, parameters: dict[str, mx.array]) -> None:
        self.first = {name: mx.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: mx.zeros_like(value) for name, value in parameters.items()}
        mx.eval(*self.first.values(), *self.second.values())

    def update(
        self,
        parameters: dict[str, mx.array],
        gradients: dict[str, mx.array],
    ) -> dict[str, mx.array]:
        require(self.first is not None and self.second is not None, "AdamW is not initialized")
        self.step += 1
        correction1 = 1.0 - self.beta1**self.step
        correction2 = 1.0 - self.beta2**self.step
        updated: dict[str, mx.array] = {}
        next_first: dict[str, mx.array] = {}
        next_second: dict[str, mx.array] = {}
        for name, value in parameters.items():
            gradient = gradients[name].astype(mx.float32)
            first = self.beta1 * self.first[name] + (1.0 - self.beta1) * gradient
            second = self.beta2 * self.second[name] + (1.0 - self.beta2) * gradient * gradient
            direction = (first / correction1) / (mx.sqrt(second / correction2) + self.epsilon)
            updated[name] = (
                value - self.learning_rate * (direction + self.weight_decay * value)
            ).astype(mx.float32)
            next_first[name] = first
            next_second[name] = second
        self.first = next_first
        self.second = next_second
        return updated


def initialize_adapter(
    hidden_size: int,
    rank: int,
    seed: int,
) -> dict[str, mx.array]:
    require(hidden_size > 0 and rank > 0, "invalid MTP adapter shape")
    mx.random.seed(seed)
    return {
        "lora_a": (
            mx.random.normal((rank, hidden_size * 2)) / math.sqrt(hidden_size * 2)
        ).astype(mx.float32),
        "lora_b": mx.zeros((hidden_size, rank), dtype=mx.float32),
    }


def validate_adapter(
    parameters: dict[str, mx.array],
    hidden_size: int,
    rank: int,
) -> None:
    require(set(parameters) == {"lora_a", "lora_b"}, "MTP adapter tensor set mismatch")
    require(
        parameters["lora_a"].dtype == mx.float32
        and parameters["lora_a"].shape == (rank, hidden_size * 2),
        "MTP adapter A tensor mismatch",
    )
    require(
        parameters["lora_b"].dtype == mx.float32
        and parameters["lora_b"].shape == (hidden_size, rank),
        "MTP adapter B tensor mismatch",
    )


def merged_fc(
    source_fc: mx.array,
    parameters: dict[str, mx.array],
    scale: float,
) -> mx.array:
    hidden_size = source_fc.shape[0]
    rank = parameters["lora_a"].shape[0]
    validate_adapter(parameters, hidden_size, rank)
    require(
        source_fc.dtype == mx.bfloat16
        and source_fc.shape == (hidden_size, hidden_size * 2),
        "MTP source FC tensor mismatch",
    )
    require(math.isfinite(scale) and scale > 0.0, "invalid MTP adapter scale")
    delta = mx.matmul(parameters["lora_b"], parameters["lora_a"]) * scale
    result = (source_fc.astype(mx.float32) + delta).astype(mx.bfloat16)
    mx.eval(result)
    return result


def _adapter_hidden(
    fused_input: mx.array,
    source_fc: mx.array,
    parameters: dict[str, mx.array] | None,
    scale: float,
    fc_override: mx.array | None,
) -> mx.array:
    if fc_override is not None:
        require(parameters is None, "MTP training FC override conflicts with an adapter")
        return mx.matmul(fused_input, mx.transpose(fc_override)).astype(source_fc.dtype)
    base = mx.matmul(fused_input, mx.transpose(source_fc)).astype(source_fc.dtype)
    if parameters is None:
        return base
    update = mx.matmul(
        mx.matmul(fused_input.astype(mx.float32), mx.transpose(parameters["lora_a"])),
        mx.transpose(parameters["lora_b"]),
    ) * scale
    return (base.astype(mx.float32) + update).astype(source_fc.dtype)


def training_forward_chunk(
    token_embeddings: mx.array,
    target_hidden: mx.array,
    state: attention.MLXAttentionState,
    weights: mtp.MLXMTPWeights,
    config: mtp.reference.MTPConfig,
    *,
    parameters: dict[str, mx.array] | None = None,
    scale: float = 1.0,
    fc_override: mx.array | None = None,
) -> tuple[mx.array, attention.MLXAttentionState]:
    """Autograd-safe MTP composition with production BF16 boundaries."""
    require(
        token_embeddings.ndim == 2
        and token_embeddings.shape == target_hidden.shape
        and token_embeddings.shape[1] == config.hidden_size,
        "MTP training input shape mismatch",
    )
    require(isinstance(state, attention.MLXAttentionState), "MTP training state must be immutable")
    dtype = weights.fc.dtype
    normalized_embedding = layer.qwen_rms_norm_batch(
        token_embeddings.astype(dtype),
        weights.pre_fc_norm_embedding,
        config.rms_norm_eps,
    )
    normalized_hidden = layer.qwen_rms_norm_batch(
        target_hidden.astype(dtype),
        weights.pre_fc_norm_hidden,
        config.rms_norm_eps,
    )
    fused_input = mx.concatenate((normalized_embedding, normalized_hidden), axis=1)
    hidden = _adapter_hidden(
        fused_input,
        weights.fc,
        parameters,
        scale,
        fc_override,
    )
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
        exact_long_prefill=False,
        token_tiled_projections=False,
        fused_prefill_qkv_projection=False,
        fused_prefill_qk_norm_rope=False,
    )
    require(isinstance(next_state, attention.MLXAttentionState), "MTP training state changed type")
    residual = (hidden + mixed).astype(dtype)
    moe_input = layer.qwen_rms_norm_batch(
        residual,
        weights.post_attention_layernorm,
        config.rms_norm_eps,
    )
    moe_result = mtp.forward_moe_batch(
        moe_input,
        weights.moe,
        config.moe,
        _validated=True,
    )
    final = (residual + moe_result.output).astype(dtype)
    output = layer.qwen_rms_norm_batch(
        final,
        weights.norm,
        config.rms_norm_eps,
    )
    return output, next_state


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    count = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return mx.sum(values * mask) / count


def distillation_loss(
    parameters: dict[str, mx.array],
    token_embeddings: mx.array,
    target_hidden: mx.array,
    teacher_hidden: mx.array,
    candidate_head: mx.array,
    teacher_logits: mx.array,
    scored: mx.array,
    state: attention.MLXAttentionState,
    weights: mtp.MLXMTPWeights,
    config: mtp.reference.MTPConfig,
    scale: float,
    hidden_weight: float,
    kl_weight: float,
    hard_weight: float,
    temperature: float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    output, next_state = training_forward_chunk(
        token_embeddings,
        target_hidden,
        state,
        weights,
        config,
        parameters=parameters,
        scale=scale,
    )
    output32 = output.astype(mx.float32)
    teacher32 = teacher_hidden.astype(mx.float32)
    squared_error = mx.mean(mx.square(output32 - teacher32))
    squared_target = mx.maximum(
        mx.mean(mx.square(teacher32)),
        mx.array(1e-8, dtype=mx.float32),
    )
    hidden_loss = squared_error / squared_target
    student_logits = mx.sum(candidate_head.astype(mx.float32) * output32[:, None, :], axis=-1)
    teacher_scaled = teacher_logits / temperature
    student_scaled = student_logits / temperature
    teacher_log_probability = teacher_scaled - mx.logsumexp(
        teacher_scaled,
        axis=-1,
        keepdims=True,
    )
    student_log_probability = student_scaled - mx.logsumexp(
        student_scaled,
        axis=-1,
        keepdims=True,
    )
    teacher_probability = mx.exp(teacher_log_probability)
    mask = scored.astype(mx.float32)
    kl_rows = mx.sum(
        teacher_probability * (teacher_log_probability - student_log_probability),
        axis=-1,
    ) * temperature**2
    kl_loss = _masked_mean(kl_rows, mask)
    hard_rows = mx.logsumexp(student_logits, axis=-1) - student_logits[:, 0]
    hard_loss = _masked_mean(hard_rows, mask)
    total = hidden_weight * hidden_loss + kl_weight * kl_loss + hard_weight * hard_loss
    return (
        total,
        hidden_loss,
        kl_loss,
        hard_loss,
        next_state.keys,
        next_state.values,
    )


def trace_shards(capture_dir: Path, *, require_complete: bool = True) -> tuple[list[TraceShard], dict]:
    state_path = capture_dir / "state.json"
    state = capture.load_json(state_path)
    capture.validate_completed(capture_dir, state)
    if require_complete:
        require(state.get("status") == "complete", "MTP teacher capture is incomplete")
    completed = state["completed"]
    shards = [
        TraceShard(
            prompt_index=int(key),
            path=capture_dir / entry["file"],
            rows=entry["rows"],
            scored_rows=entry["scored_rows"],
        )
        for key, entry in sorted(completed.items(), key=lambda item: int(item[0]))
    ]
    require(shards, "MTP teacher capture contains no shards")
    return shards, state


def load_trace(shard: TraceShard) -> dict[str, mx.array]:
    arrays, metadata = mx.load(str(shard.path), return_metadata=True)
    capture.validate_trace_arrays(
        arrays,
        metadata,
        prompt_index=shard.prompt_index,
    )
    require(arrays["expected_token_ids"].size == shard.rows, "MTP trace row count drift")
    require(int(mx.sum(arrays["scored"]).item()) == shard.scored_rows, "MTP trace score count drift")
    return arrays


def _chunk_inputs(
    arrays: dict[str, mx.array],
    offset: int,
    end: int,
    embedding: vocab.MLXMappedBF16Matrix,
    head: vocab.MLXMappedBF16Matrix,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    following = arrays["token_ids"][offset + 1 : end + 1].tolist()
    embeddings = embedding.rows(following)
    candidate_ids = arrays["candidate_token_ids"][offset:end]
    flat_ids = [max(0, int(value)) for value in candidate_ids.reshape(-1).tolist()]
    candidate_head = head.rows(flat_ids).reshape(
        end - offset,
        candidate_ids.shape[1],
        model.PRODUCTION_CONFIG.hidden_size,
    )
    return (
        embeddings,
        arrays["target_hidden"][offset:end],
        arrays["target_hidden"][offset + 1 : end + 1],
        candidate_head,
        arrays["candidate_logits"][offset:end],
        arrays["scored"][offset:end],
    )


def _stop_state(state: attention.MLXAttentionState) -> attention.MLXAttentionState:
    return attention.MLXAttentionState(
        keys=mx.stop_gradient(state.keys),
        values=mx.stop_gradient(state.values),
    )


def evaluate(
    shards: Iterable[TraceShard],
    weights: mtp.MLXMTPWeights,
    embedding: vocab.MLXMappedBF16Matrix,
    head: vocab.MLXMappedBF16Matrix,
    *,
    unroll: int,
    parameters: dict[str, mx.array] | None = None,
    scale: float = 1.0,
    fc_override: mx.array | None = None,
) -> dict[str, float | int | None]:
    squared_error = 0.0
    squared_target = 0.0
    hidden_values = 0
    scored_rows = 0
    candidate_matches = 0
    started = time.perf_counter()
    for shard in shards:
        arrays = load_trace(shard)
        state = attention.zeros_state(mtp.PRODUCTION_CONFIG.attention, dtype=mx.bfloat16)
        for offset in range(0, shard.rows, unroll):
            end = min(offset + unroll, shard.rows)
            inputs = _chunk_inputs(arrays, offset, end, embedding, head)
            output, state = training_forward_chunk(
                inputs[0],
                inputs[1],
                state,
                weights,
                mtp.PRODUCTION_CONFIG,
                parameters=parameters,
                scale=scale,
                fc_override=fc_override,
            )
            output32 = output.astype(mx.float32)
            teacher32 = inputs[2].astype(mx.float32)
            error = mx.sum(mx.square(output32 - teacher32))
            target = mx.sum(mx.square(teacher32))
            logits = mx.sum(inputs[3].astype(mx.float32) * output32[:, None, :], axis=-1)
            predictions = mx.argmax(logits, axis=-1)
            mask = inputs[5].astype(mx.bool_)
            matches = mx.sum(mask & (predictions == 0))
            count = mx.sum(mask)
            mx.eval(error, target, matches, count, state.keys, state.values)
            squared_error += float(error)
            squared_target += float(target)
            hidden_values += output.size
            candidate_matches += int(matches)
            scored_rows += int(count)
            state = _stop_state(state)
    return {
        "rows": hidden_values // model.PRODUCTION_CONFIG.hidden_size,
        "scored_rows": scored_rows,
        "candidate_matches": candidate_matches,
        "candidate_acceptance": candidate_matches / scored_rows if scored_rows else None,
        "hidden_mse": squared_error / hidden_values if hidden_values else None,
        "hidden_relative_l2": math.sqrt(squared_error / squared_target) if squared_target else None,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _metric_score(metrics: dict[str, Any]) -> tuple[float, float]:
    acceptance = metrics.get("candidate_acceptance")
    relative = metrics.get("hidden_relative_l2")
    return (
        float(acceptance) if acceptance is not None else -1.0,
        -float(relative) if relative is not None else -math.inf,
    )


def _copy_parameters(parameters: dict[str, mx.array]) -> dict[str, mx.array]:
    values = {name: mx.stop_gradient(value) for name, value in parameters.items()}
    mx.eval(*values.values())
    return values


def _restore_parameters(parameters: dict[str, mx.array]) -> dict[str, mx.array]:
    values = {name: value.astype(mx.float32) for name, value in parameters.items()}
    mx.eval(*values.values())
    return values


def _save_safetensors_atomic(
    path: Path,
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
) -> None:
    temporary = path.with_name(f".{path.stem}.part.safetensors")
    temporary.unlink(missing_ok=True)
    try:
        mx.save_safetensors(str(temporary), arrays, metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        capture.fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    output_dir: Path,
    identity: dict[str, Any],
    epoch: int,
    parameters: dict[str, mx.array],
    optimizer: AdamW,
    best_parameters: dict[str, mx.array],
    best_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    require(optimizer.first is not None and optimizer.second is not None, "optimizer state is absent")
    arrays = {
        "lora_a": parameters["lora_a"],
        "lora_b": parameters["lora_b"],
        "adam_first_a": optimizer.first["lora_a"],
        "adam_first_b": optimizer.first["lora_b"],
        "adam_second_a": optimizer.second["lora_a"],
        "adam_second_b": optimizer.second["lora_b"],
        "best_lora_a": mx.array(best_parameters["lora_a"]),
        "best_lora_b": mx.array(best_parameters["lora_b"]),
    }
    artifact = output_dir / "checkpoint.safetensors"
    _save_safetensors_atomic(
        artifact,
        arrays,
        {"format": CHECKPOINT_FORMAT, "epoch": str(epoch)},
    )
    state = {
        "format": CHECKPOINT_FORMAT,
        "identity": identity,
        "epoch": epoch,
        "optimizer_step": optimizer.step,
        "artifact": {
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": capture.sha256_file(artifact),
        },
        "best_metrics": best_metrics,
        "history": history,
    }
    capture.atomic_json(output_dir / "checkpoint.json", state)


def load_checkpoint(
    output_dir: Path,
    identity: dict[str, Any],
    hidden_size: int,
    rank: int,
    optimizer: AdamW,
) -> tuple[int, dict[str, mx.array], dict[str, mx.array], dict[str, Any], list[dict[str, Any]]]:
    state = capture.load_json(output_dir / "checkpoint.json")
    require(state.get("format") == CHECKPOINT_FORMAT, "MTP checkpoint format mismatch")
    require(state.get("identity") == identity, "MTP checkpoint identity mismatch")
    record = state.get("artifact")
    require(isinstance(record, dict), "MTP checkpoint artifact record is absent")
    path = output_dir / record.get("name", "")
    require(path.is_file(), "MTP checkpoint artifact is absent")
    require(path.stat().st_size == record.get("bytes"), "MTP checkpoint size mismatch")
    require(capture.sha256_file(path) == record.get("sha256"), "MTP checkpoint hash mismatch")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    require(metadata.get("format") == CHECKPOINT_FORMAT, "MTP checkpoint metadata mismatch")
    required = {
        "lora_a",
        "lora_b",
        "adam_first_a",
        "adam_first_b",
        "adam_second_a",
        "adam_second_b",
        "best_lora_a",
        "best_lora_b",
    }
    require(set(arrays) == required, "MTP checkpoint tensor set mismatch")
    parameters = {"lora_a": arrays["lora_a"], "lora_b": arrays["lora_b"]}
    validate_adapter(parameters, hidden_size, rank)
    optimizer.first = {
        "lora_a": arrays["adam_first_a"],
        "lora_b": arrays["adam_first_b"],
    }
    optimizer.second = {
        "lora_a": arrays["adam_second_a"],
        "lora_b": arrays["adam_second_b"],
    }
    optimizer.step = int(state.get("optimizer_step", -1))
    require(optimizer.step >= 0, "MTP checkpoint optimizer step is invalid")
    best = {
        "lora_a": mx.stop_gradient(arrays["best_lora_a"]),
        "lora_b": mx.stop_gradient(arrays["best_lora_b"]),
    }
    mx.eval(*parameters.values(), *optimizer.first.values(), *optimizer.second.values())
    return (
        int(state.get("epoch", -1)),
        parameters,
        best,
        state.get("best_metrics", {}),
        state.get("history", []),
    )


def write_adaptation(
    output_dir: Path,
    adapted_fc: mx.array,
    *,
    status: str,
    identity: dict[str, Any],
    capture_state_sha256: str,
    baseline: dict[str, Any],
    best: dict[str, Any],
    merged: dict[str, Any],
    history: list[dict[str, Any]],
    rank: int,
    alpha: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    require(status in {"candidate", "diagnostic", "rejected"}, "invalid MTP adaptation status")
    require(
        adapted_fc.dtype == mx.bfloat16 and adapted_fc.shape == (2048, 4096),
        "invalid MTP adapted FC artifact",
    )
    artifact = output_dir / mtp.ADAPTATION_ARTIFACT
    _save_safetensors_atomic(
        artifact,
        {"mtp.fc.weight": adapted_fc},
        {
            "format": mtp.ADAPTATION_FORMAT,
            "base_mtp_sha256": mtp.EXPECTED_SIDECAR_SHA256,
            "base_fc_sha256": mtp.EXPECTED_FC_SHA256,
        },
    )
    state = {
        "format": mtp.ADAPTATION_FORMAT,
        "experiment_format": FORMAT,
        "status": status,
        "source": {
            "target_weight_sha256": nvfp4.EXPECTED_WEIGHT_SHA256,
            "mtp_sidecar_sha256": mtp.EXPECTED_SIDECAR_SHA256,
            "mtp_fc_sha256": mtp.EXPECTED_FC_SHA256,
        },
        "identity": identity,
        "capture_state_sha256": capture_state_sha256,
        "training": {
            "rank": rank,
            "alpha": alpha,
            "parameter_count": rank * (4096 + 2048),
            "runtime_extra_operations": 0,
            "history": history,
            "elapsed_seconds": elapsed_seconds,
            "peak_gib": mx.get_peak_memory() / 2**30,
        },
        "metrics": {
            "baseline": baseline,
            "best_low_rank": best,
            "merged_bf16": merged,
        },
        "artifact": {
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": capture.sha256_file(artifact),
        },
        "decision": (
            "candidate-pending-exact-resident-acceptance-and-throughput-gate"
            if status == "candidate"
            else "diagnostic-not-runtime-promoted"
        ),
    }
    capture.atomic_json(output_dir / "state.json", state)
    loaded = mtp.load_fc_adaptation(
        output_dir,
        mx.zeros((2048, 4096), dtype=mx.bfloat16),
        allow_diagnostic=True,
    )
    require(bool(mx.array_equal(loaded, adapted_fc).item()), "MTP adaptation readback mismatch")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--hard-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-incomplete-capture", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.epochs > 0 and args.rank > 0 and args.unroll > 0, "invalid training size")
        require(args.alpha > 0.0 and args.learning_rate > 0.0, "invalid optimizer scale")
        require(args.weight_decay >= 0.0, "weight decay cannot be negative")
        require(
            args.hidden_weight >= 0.0
            and args.kl_weight >= 0.0
            and args.hard_weight >= 0.0
            and args.hidden_weight + args.kl_weight + args.hard_weight > 0.0,
            "invalid MTP loss weights",
        )
        require(args.temperature > 0.0, "MTP temperature must be positive")
        require(
            args.holdout_modulus >= 2 and 0 <= args.holdout_fold < args.holdout_modulus,
            "invalid MTP holdout split",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        final_state_path = args.output_dir / "state.json"
        if final_state_path.exists():
            state = capture.load_json(final_state_path)
            require(state.get("format") == mtp.ADAPTATION_FORMAT, "existing MTP output is invalid")
            mtp.load_fc_adaptation(
                args.output_dir,
                mx.zeros((2048, 4096), dtype=mx.bfloat16),
                allow_diagnostic=True,
            )
            print(
                "mtp-distill-existing "
                f"status={state['status']} artifact={state['artifact']['name']}",
                flush=True,
            )
            return 0

        shards, capture_state = trace_shards(
            args.capture_dir,
            require_complete=not args.allow_incomplete_capture,
        )
        capture_state_hash = capture.sha256_file(args.capture_dir / "state.json")
        training_shards = [
            shard
            for shard in shards
            if shard.prompt_index % args.holdout_modulus != args.holdout_fold
        ]
        heldout_shards = [
            shard
            for shard in shards
            if shard.prompt_index % args.holdout_modulus == args.holdout_fold
        ]
        require(training_shards and heldout_shards, "MTP train/heldout split is empty")
        source_path = nvfp4.require_verified_source(args.root)
        mtp.require_verified_mtp_sidecar(args.root, verify_hash=True)
        scale = args.alpha / args.rank
        identity = {
            "root": str(args.root.resolve()),
            "capture_dir": str(args.capture_dir.resolve()),
            "capture_state_sha256": capture_state_hash,
            "capture_identity_sha256": capture.canonical_sha256(capture_state["identity"]),
            "target_weight_sha256": nvfp4.EXPECTED_WEIGHT_SHA256,
            "mtp_sidecar_sha256": mtp.EXPECTED_SIDECAR_SHA256,
            "repository_revision": capture.repository_revision(),
            "tool_sha256": capture.sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "epochs": args.epochs,
            "rank": args.rank,
            "alpha": args.alpha,
            "unroll": args.unroll,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_weight": args.hidden_weight,
            "kl_weight": args.kl_weight,
            "hard_weight": args.hard_weight,
            "temperature": args.temperature,
            "holdout_modulus": args.holdout_modulus,
            "holdout_fold": args.holdout_fold,
            "seed": args.seed,
        }
        if args.validate_only:
            print(
                "mtp-distill-validated "
                f"shards={len(shards)} train={len(training_shards)} heldout={len(heldout_shards)} "
                f"rows={capture_state['rows']} scored={capture_state['scored_rows']}",
                flush=True,
            )
            return 0

        free_bytes = shutil.disk_usage(args.output_dir).free
        estimated_bytes = 64 * 2**20 + args.rank * (4096 + 2048) * 32
        require(
            free_bytes >= estimated_bytes + 512 * 2**20,
            "insufficient disk for MTP checkpoints plus safety margin",
        )
        print(
            "mtp-distill-plan "
            f"train_shards={len(training_shards)} heldout_shards={len(heldout_shards)} "
            f"rank={args.rank} epochs={args.epochs} unroll={args.unroll} "
            f"estimated_mib={estimated_bytes / 2**20:.3f} free_gib={free_bytes / 2**30:.3f}",
            flush=True,
        )

        mx.set_cache_limit(256 * 2**20)
        weights = mtp.load_weights(args.root, verify_hash=False)
        embedding = vocab.MLXMappedBF16Matrix(
            source_path,
            "model.language_model.embed_tokens.weight",
            (model.PRODUCTION_CONFIG.vocab_size, model.PRODUCTION_CONFIG.hidden_size),
        )
        head = vocab.MLXMappedBF16Matrix(
            source_path,
            "lm_head.weight",
            (model.PRODUCTION_CONFIG.vocab_size, model.PRODUCTION_CONFIG.hidden_size),
        )
        optimizer = AdamW(args.learning_rate, args.weight_decay)
        checkpoint_path = args.output_dir / "checkpoint.json"
        if checkpoint_path.exists():
            (
                completed_epoch,
                parameters,
                best_parameters,
                best_metrics,
                history,
            ) = load_checkpoint(
                args.output_dir,
                identity,
                model.PRODUCTION_CONFIG.hidden_size,
                args.rank,
                optimizer,
            )
            print(
                f"mtp-distill-resume epoch={completed_epoch}/{args.epochs} step={optimizer.step}",
                flush=True,
            )
        else:
            completed_epoch = 0
            parameters = initialize_adapter(
                model.PRODUCTION_CONFIG.hidden_size,
                args.rank,
                args.seed,
            )
            optimizer.initialize(parameters)
            best_parameters = _copy_parameters(parameters)
            best_metrics = {}
            history: list[dict[str, Any]] = []
        validate_adapter(parameters, model.PRODUCTION_CONFIG.hidden_size, args.rank)
        baseline = evaluate(
            heldout_shards,
            weights,
            embedding,
            head,
            unroll=args.unroll,
        )
        print("mtp-distill-baseline " + json.dumps(baseline, separators=(",", ":")), flush=True)
        if not best_metrics:
            best_metrics = baseline

        value_and_grad = mx.value_and_grad(distillation_loss)
        started = time.perf_counter()
        for epoch in range(completed_epoch + 1, args.epochs + 1):
            ordered = list(training_shards)
            random.Random(args.seed + epoch).shuffle(ordered)
            epoch_losses = []
            epoch_hidden = []
            epoch_kl = []
            epoch_hard = []
            epoch_started = time.perf_counter()
            for shard in ordered:
                arrays = load_trace(shard)
                state = attention.zeros_state(mtp.PRODUCTION_CONFIG.attention, dtype=mx.bfloat16)
                accumulated = {
                    name: mx.zeros_like(value) for name, value in parameters.items()
                }
                accumulated_rows = 0
                for offset in range(0, shard.rows, args.unroll):
                    end = min(offset + args.unroll, shard.rows)
                    inputs = _chunk_inputs(arrays, offset, end, embedding, head)
                    (
                        (loss, hidden_loss, kl_loss, hard_loss, next_keys, next_values),
                        gradients,
                    ) = value_and_grad(
                        parameters,
                        *inputs,
                        state,
                        weights,
                        mtp.PRODUCTION_CONFIG,
                        scale,
                        args.hidden_weight,
                        args.kl_weight,
                        args.hard_weight,
                        args.temperature,
                    )
                    mx.eval(
                        loss,
                        hidden_loss,
                        kl_loss,
                        hard_loss,
                        next_keys,
                        next_values,
                        *gradients.values(),
                    )
                    accumulated = {
                        name: accumulated[name]
                        + gradients[name].astype(mx.float32) * (end - offset)
                        for name in parameters
                    }
                    state = _stop_state(
                        attention.MLXAttentionState(keys=next_keys, values=next_values)
                    )
                    epoch_losses.append(float(loss))
                    epoch_hidden.append(float(hidden_loss))
                    epoch_kl.append(float(kl_loss))
                    epoch_hard.append(float(hard_loss))
                    accumulated_rows += end - offset
                require(accumulated_rows > 0, "MTP training prompt produced no rows")
                averaged = {
                    name: value / accumulated_rows for name, value in accumulated.items()
                }
                parameters = optimizer.update(parameters, averaged)
                mx.eval(*parameters.values())
                mx.clear_cache()
            metrics = evaluate(
                heldout_shards,
                weights,
                embedding,
                head,
                unroll=args.unroll,
                parameters=parameters,
                scale=scale,
            )
            row = {
                "epoch": epoch,
                "loss": sum(epoch_losses) / len(epoch_losses),
                "hidden_loss": sum(epoch_hidden) / len(epoch_hidden),
                "kl_loss": sum(epoch_kl) / len(epoch_kl),
                "hard_loss": sum(epoch_hard) / len(epoch_hard),
                "heldout": metrics,
                "seconds": time.perf_counter() - epoch_started,
                "optimizer_step": optimizer.step,
            }
            history.append(row)
            if _metric_score(metrics) > _metric_score(best_metrics):
                best_metrics = metrics
                best_parameters = _copy_parameters(parameters)
            save_checkpoint(
                args.output_dir,
                identity,
                epoch,
                parameters,
                optimizer,
                best_parameters,
                best_metrics,
                history,
            )
            print("mtp-distill-epoch " + json.dumps(row, separators=(",", ":")), flush=True)

        best_values = _restore_parameters(best_parameters)
        adapted_fc = merged_fc(weights.fc, best_values, scale)
        merged_metrics = evaluate(
            heldout_shards,
            weights,
            embedding,
            head,
            unroll=args.unroll,
            fc_override=adapted_fc,
        )
        baseline_acceptance = float(baseline.get("candidate_acceptance") or 0.0)
        merged_acceptance = float(merged_metrics.get("candidate_acceptance") or 0.0)
        baseline_hidden = float(baseline.get("hidden_relative_l2") or math.inf)
        merged_hidden = float(merged_metrics.get("hidden_relative_l2") or math.inf)
        status = (
            "candidate"
            if merged_acceptance > baseline_acceptance and merged_hidden < baseline_hidden
            else "diagnostic"
        )
        state = write_adaptation(
            args.output_dir,
            adapted_fc,
            status=status,
            identity=identity,
            capture_state_sha256=capture_state_hash,
            baseline=baseline,
            best=best_metrics,
            merged=merged_metrics,
            history=history,
            rank=args.rank,
            alpha=args.alpha,
            elapsed_seconds=time.perf_counter() - started,
        )
        embedding.close()
        head.close()
        print("mtp-distill-done " + json.dumps(state, separators=(",", ":")), flush=True)
        return 0
    except (
        MoEError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        IndexError,
    ) as exc:
        print(f"mtp-distill-error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

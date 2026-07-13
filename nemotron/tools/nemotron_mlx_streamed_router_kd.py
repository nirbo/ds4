#!/usr/bin/env python3
"""Run one resumable end-to-end Router KD step with layer-streamed MLX VJPs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_attention import load_attention_layer
from nemotron_mlx_compare_logits import compare
from nemotron_mlx_kd_gradient_audit import NativeBF16Linear
from nemotron_mlx_linear import ModelOptFP8Linear
from nemotron_mlx_mamba import layer_tensors, load_mamba_layer
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_stream_forward import StreamingForward, validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-streamed-router-kd-v1"
STATE_FORMAT = "nemotron-streamed-router-kd-state-v1"


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def kl_divergence(teacher_logits: mx.array, student_logits: mx.array, temperature: float) -> mx.array:
    require(temperature > 0, "temperature must be positive")
    teacher = mx.stop_gradient(teacher_logits.astype(mx.float32) / temperature)
    student = student_logits.astype(mx.float32) / temperature
    teacher_log_probs = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    student_log_probs = student - mx.logsumexp(student, axis=-1, keepdims=True)
    return mx.mean(
        mx.sum(mx.exp(teacher_log_probs) * (teacher_log_probs - student_log_probs), axis=-1)
    ) * temperature**2


def first_adam_step(weight: mx.array, gradient: mx.array, learning_rate: float) -> mx.array:
    """Return the runtime BF16 result of one Adam step without optimizer state."""

    require(learning_rate > 0, "learning rate must be positive")
    gradient = gradient.astype(mx.float32)
    update = gradient / (mx.abs(gradient) + 1e-8)
    return (weight.astype(mx.float32) - learning_rate * update).astype(mx.bfloat16)


def changed_rows(source: np.ndarray, candidate: np.ndarray) -> list[int]:
    require(source.shape == candidate.shape and source.ndim == 2, "router shape mismatch")
    return np.flatnonzero(np.any(source != candidate, axis=1)).astype(int).tolist()


def global_tensor(source_dir: Path, index: dict, name: str) -> mx.array:
    shard = index.get("weight_map", {}).get(name)
    require(isinstance(shard, str), f"missing global tensor: {name}")
    tensors = mx.load(str(source_dir / shard))
    require(name in tensors, f"global tensor absent from shard: {name}")
    return tensors[name]


def gradient_safe_retained_moe(
    block,
    x: mx.array,
    retained: list[int],
    gate_weight: mx.array,
) -> mx.array:
    hidden = block.norm(x)
    indices, scores = block.route_retained_gate(hidden, retained, gate_weight)
    latent = block.fc1_latent(hidden)
    selected = expert_outputs(latent, block.experts, mx.stop_gradient(indices))
    aggregate = (selected * scores[..., None]).sum(axis=-2)
    routed_projection = gradient_safe_linear(block.fc2_latent)
    routed = routed_projection(aggregate)
    shared_hidden = mx.square(mx.maximum(block.shared_up(hidden), mx.array(0.0, hidden.dtype)))
    shared = block.shared_down(shared_hidden)
    return x + routed + shared


def gradient_safe_linear(source):
    """Replace inference kernels that lack VJPs while preserving frozen values."""

    if source.weight.dtype == mx.bfloat16:
        return NativeBF16Linear(source)
    if isinstance(source, ModelOptFP8Linear):
        return NativeFP8Linear(source)
    return source


def decode_e4m3fn(bits: mx.array) -> mx.array:
    """Decode a ModelOpt E4M3FN tensor with differentiable MLX primitives."""

    require(bits.dtype == mx.uint8, "E4M3FN decode requires uint8 storage")
    encoded = bits.astype(mx.uint32)
    exponent = (encoded >> 3) & 15
    mantissa = encoded & 7
    subnormal = mantissa.astype(mx.float32) * (2.0**-9)
    normal = (1.0 + mantissa.astype(mx.float32) * 0.125) * mx.power(
        mx.array(2.0, dtype=mx.float32), exponent.astype(mx.float32) - 7.0
    )
    magnitude = mx.where(exponent == 0, subnormal, normal)
    return mx.where((encoded & 128) != 0, -magnitude, magnitude)


class NativeFP8Linear:
    """Gradient-capable frozen ModelOpt FP8 projection used only during KD."""

    def __init__(self, source: ModelOptFP8Linear):
        self.weight = decode_e4m3fn(source.weight)
        self.scale = source.scale.reshape(()).astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        require(x.shape[-1] == self.weight.shape[1], "native FP8 input shape mismatch")
        return (x.astype(mx.float32) * self.scale) @ self.weight.T


def make_attention_gradient_safe(block) -> None:
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        source = getattr(block.mixer, name)
        setattr(block.mixer, name, gradient_safe_linear(source))


def make_mamba_gradient_safe(block) -> None:
    block.mixer.in_proj = gradient_safe_linear(block.mixer.in_proj)
    block.mixer.out_proj = gradient_safe_linear(block.mixer.out_proj)


def make_moe_gradient_safe(block) -> None:
    for name in ("fc1_latent", "fc2_latent", "shared_up", "shared_down"):
        setattr(block, name, gradient_safe_linear(getattr(block, name)))


def load_source_router(source_dir: Path, layer: int, retained: list[int]) -> mx.array:
    name = f"backbone.layers.{layer}.mixer.gate.weight"
    tensors = layer_tensors(source_dir, layer)
    require(name in tensors, f"missing router tensor: {name}")
    result = tensors[name][mx.array(retained, dtype=mx.uint32)]
    mx.eval(result)
    del tensors
    gc.collect()
    mx.clear_cache()
    return result


def forward_layer(
    source_dir: Path,
    layer: int,
    kind: str,
    x: mx.array,
    retained: dict[str, list[int]],
    routers: dict[str, mx.array] | None,
) -> mx.array:
    if kind == "M":
        block = load_mamba_layer(source_dir, layer)
        make_mamba_gradient_safe(block)
        output = block(x, mask=None, cache=None)
    elif kind == "*":
        block = load_attention_layer(source_dir, layer)
        make_attention_gradient_safe(block)
        output = block(x, mask=create_attention_mask(x, None), cache=None)
    elif kind == "E":
        block = load_moe_layer(source_dir, layer)
        make_moe_gradient_safe(block)
        kept = retained[str(layer)]
        gate = (
            block.gate_weight[mx.array(kept, dtype=mx.uint32)]
            if routers is None
            else routers[str(layer)]
        )
        output = gradient_safe_retained_moe(block, x, kept, gate)
    else:
        raise MetadataError(f"unsupported Nemotron layer type {kind!r} at {layer}")
    mx.eval(output)
    del block
    if kind == "E" and routers is None:
        del gate
    gc.collect()
    mx.clear_cache()
    return output


def score_head(
    source_dir: Path,
    index: dict,
    config: dict,
    hidden: mx.array,
) -> mx.array:
    norm = global_tensor(source_dir, index, "backbone.norm_f.weight")
    head = global_tensor(source_dir, index, "lm_head.weight")
    normalized = mx.fast.rms_norm(hidden[:, -1:, :], norm, config["layer_norm_epsilon"])
    logits = normalized.astype(mx.float32) @ head.T
    return logits


def student_forward(
    source_dir: Path,
    config: dict,
    retained: dict[str, list[int]],
    token_ids: list[int],
    activation_dir: Path | None,
    routers: dict[str, mx.array] | None,
    operation_log: OperationLog,
) -> np.ndarray:
    index = load_json(source_dir / "model.safetensors.index.json")
    start_layer = 0
    if activation_dir is not None:
        while (activation_dir / f"boundary-{start_layer + 1:03d}.npy").exists():
            start_layer += 1
            if start_layer == len(config["hybrid_override_pattern"]):
                break
    if start_layer:
        x = mx.array(np.load(activation_dir / f"boundary-{start_layer:03d}.npy"))
        operation_log.write(f"student-forward-resume layer={start_layer}")
    else:
        embeddings = global_tensor(source_dir, index, "backbone.embeddings.weight")
        x = embeddings[mx.array(token_ids, dtype=mx.uint32)].astype(mx.float32).reshape(
            1, len(token_ids), config["hidden_size"]
        )
        mx.eval(x)
        del embeddings
        if activation_dir is not None:
            atomic_npy(activation_dir / "boundary-000.npy", np.asarray(x, dtype=np.float32))
    for layer in range(start_layer, len(config["hybrid_override_pattern"])):
        kind = config["hybrid_override_pattern"][layer]
        started = time.perf_counter()
        x = forward_layer(source_dir, layer, kind, x, retained, routers)
        if activation_dir is not None:
            atomic_npy(activation_dir / f"boundary-{layer + 1:03d}.npy", np.asarray(x, dtype=np.float32))
        operation_log.write(
            f"student-forward layer={layer:02d} kind={kind} elapsed={time.perf_counter() - started:.3f}s "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}"
        )
    logits = score_head(source_dir, index, config, x)
    mx.eval(logits)
    result = np.asarray(logits, dtype=np.float32)
    del x, logits
    gc.collect()
    mx.clear_cache()
    return result


def backward_head(
    source_dir: Path,
    config: dict,
    final_hidden: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray]:
    index = load_json(source_dir / "model.safetensors.index.json")
    teacher = mx.array(teacher_logits)
    norm = global_tensor(source_dir, index, "backbone.norm_f.weight")
    head = global_tensor(source_dir, index, "lm_head.weight")

    def objective(hidden):
        normalized = mx.fast.rms_norm(
            hidden[:, -1:, :], norm, config["layer_norm_epsilon"]
        )
        return kl_divergence(teacher, normalized.astype(mx.float32) @ head.T, temperature)

    value, gradient = mx.value_and_grad(objective)(mx.array(final_hidden))
    mx.eval(value, gradient)
    result = float(value), np.asarray(gradient, dtype=np.float32)
    del value, gradient, teacher, norm, head
    gc.collect()
    mx.clear_cache()
    return result


def backward_layer(
    source_dir: Path,
    layer: int,
    kind: str,
    x_np: np.ndarray,
    cotangent_np: np.ndarray,
    retained: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray | None]:
    x = mx.array(x_np)
    cotangent = mx.array(cotangent_np)
    if kind == "M":
        block = load_mamba_layer(source_dir, layer)
        make_mamba_gradient_safe(block)
        _, gradients = mx.vjp(lambda value: block(value, mask=None, cache=None), [x], [cotangent])
        input_gradient = gradients[0]
        router_gradient = None
    elif kind == "*":
        block = load_attention_layer(source_dir, layer)
        make_attention_gradient_safe(block)
        mask = create_attention_mask(x, None)
        _, gradients = mx.vjp(lambda value: block(value, mask=mask, cache=None), [x], [cotangent])
        input_gradient = gradients[0]
        router_gradient = None
    elif kind == "E":
        block = load_moe_layer(source_dir, layer)
        make_moe_gradient_safe(block)
        kept = retained[str(layer)]
        gate = block.gate_weight[mx.array(kept, dtype=mx.uint32)].astype(mx.float32)
        _, gradients = mx.vjp(
            lambda value, weight: gradient_safe_retained_moe(block, value, kept, weight),
            [x, gate],
            [cotangent],
        )
        input_gradient, router_gradient = gradients
    else:
        raise MetadataError(f"unsupported Nemotron layer type {kind!r} at {layer}")
    values = [input_gradient]
    if router_gradient is not None:
        values.append(router_gradient)
    mx.eval(*values)
    input_np = np.asarray(input_gradient, dtype=np.float32)
    router_np = None if router_gradient is None else np.asarray(router_gradient, dtype=np.float32)
    del block, input_gradient, gradients
    gc.collect()
    mx.clear_cache()
    return input_np, router_np


def load_router_set(
    source_dir: Path,
    retained: dict[str, list[int]],
    gradient_dir: Path,
    learning_rate: float,
) -> tuple[dict[str, mx.array], list[dict]]:
    routers = {}
    rows = []
    for layer_text, kept in retained.items():
        layer = int(layer_text)
        source = load_source_router(source_dir, layer, kept)
        gradient = mx.array(np.load(gradient_dir / f"layer-{layer:03d}.npy"))
        candidate = first_adam_step(source, gradient, learning_rate)
        mx.eval(candidate)
        source_np = np.asarray(source.astype(mx.float32))
        candidate_np = np.asarray(candidate.astype(mx.float32))
        gradient_np = np.asarray(gradient, dtype=np.float32)
        changed = changed_rows(source_np, candidate_np)
        active = np.flatnonzero(np.any(gradient_np != 0, axis=1)).astype(int).tolist()
        require(set(changed) <= set(active), f"layer {layer} changed a zero-gradient row")
        routers[layer_text] = candidate
        rows.append(
            {
                "layer": layer,
                "retained_rows": len(kept),
                "active_gradient_rows": len(active),
                "changed_rows": len(changed),
                "unchanged_rows": len(kept) - len(changed),
                "zero_gradient_rows_exact": all(
                    np.array_equal(source_np[index], candidate_np[index])
                    for index in set(range(len(kept))) - set(active)
                ),
                "changed_source_experts": [kept[index] for index in changed],
                "gradient_norm": float(np.linalg.norm(gradient_np.astype(np.float64))),
                "gradient_finite": bool(np.isfinite(gradient_np).all()),
            }
        )
    return routers, rows


def save_router_artifact(
    path: Path,
    routers: dict[str, mx.array],
    source_revision: str,
    plan_sha256: str,
    learning_rate: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    mx.save_safetensors(
        str(temporary),
        {f"layer_{int(layer):03d}.gate.weight": value for layer, value in routers.items()},
        metadata={
            "format": FORMAT,
            "source_revision": source_revision,
            "plan_sha256": plan_sha256,
            "learning_rate": f"{learning_rate:.17g}",
        },
    )
    temporary.replace(path)


def parse_token_ids(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise MetadataError(f"invalid token ID list: {value}") from exc
    require(result, "token sequence is empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--token-ids")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--line-search-steps", type=int, default=5)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trace-production", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = OperationLog(args.output_dir / "run.log")
    try:
        require(args.temperature > 0 and args.learning_rate > 0, "invalid optimization setting")
        require(args.line_search_steps > 0, "line-search steps must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained = validate_virtual_plan(plan, config, source_state["revision"])
        if args.prompt is not None:
            tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
            token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            prompt_sha256 = hashlib.sha256(args.prompt.encode()).hexdigest()
        else:
            token_ids = parse_token_ids(args.token_ids)
            prompt_sha256 = None
        require(1 <= len(token_ids) <= 8, "one-step proof requires 1-8 tokens")
        require(all(0 <= token < config["vocab_size"] for token in token_ids), "token ID out of range")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plan_hash = sha256_file(args.plan)
        tool_hash = sha256_file(Path(__file__))
        source_state_hash = sha256_file(args.source_state)
        identity = {
            "format": STATE_FORMAT,
            "source_revision": source_state["revision"],
            "source_state_sha256": source_state_hash,
            "plan_sha256": plan_hash,
            "tool_sha256": tool_hash,
            "prompt_sha256": prompt_sha256,
            "token_ids": token_ids,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
        }
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            require(all(state.get(key) == value for key, value in identity.items()), "resume identity mismatch")
        else:
            state = {**identity, "phase": "initialized"}
            atomic_json(state_path, state)
        if state.get("phase") == "complete":
            report_path = args.output_dir / "report.json"
            require(report_path.is_file(), "completed run has no report")
            require(state.get("report_sha256") == sha256_file(report_path), "completed report hash mismatch")
            report = load_json(report_path)
            require(report.get("format") == FORMAT and report.get("status") == "complete", "invalid report")
            operation_log.write(f"run-already-complete report_sha256={state['report_sha256']}")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        activation_dir = args.output_dir / "activations"
        gradient_dir = args.output_dir / "router-gradients"

        teacher_path = args.output_dir / "teacher-logits.npy"
        if not teacher_path.exists():
            operation_log.write("teacher-forward-start")
            teacher_runner = StreamingForward(args.source_dir)
            teacher_logits = teacher_runner.forward_sequence(token_ids, trace=args.trace_production)
            atomic_npy(teacher_path, np.asarray(teacher_logits, dtype=np.float32))
            operation_log.write(f"teacher-forward-done peak_gib={mx.get_peak_memory() / 2**30:.3f}")
            del teacher_runner, teacher_logits
            gc.collect()
            mx.clear_cache()
        teacher_np = np.load(teacher_path)

        production_path = args.output_dir / "production-student-logits.npy"
        if not production_path.exists():
            operation_log.write("production-student-forward-start")
            production_runner = StreamingForward(args.source_dir, retained)
            production_logits = production_runner.forward_sequence(token_ids, trace=args.trace_production)
            atomic_npy(production_path, np.asarray(production_logits, dtype=np.float32))
            operation_log.write(f"production-student-forward-done peak_gib={mx.get_peak_memory() / 2**30:.3f}")
            del production_runner, production_logits
            gc.collect()
            mx.clear_cache()

        student_path = args.output_dir / "training-student-logits.npy"
        final_boundary = activation_dir / f"boundary-{len(config['hybrid_override_pattern']):03d}.npy"
        if not student_path.exists() or not final_boundary.exists():
            operation_log.write("training-student-forward-start")
            student_np = student_forward(
                args.source_dir, config, retained, token_ids, activation_dir, None, operation_log
            )
            atomic_npy(student_path, student_np)
            state["phase"] = "forward-complete"
            atomic_json(state_path, state)
            operation_log.write(f"training-student-forward-done peak_gib={mx.get_peak_memory() / 2**30:.3f}")
        student_np = np.load(student_path)
        production_np = np.load(production_path)
        parity = compare(production_np, student_np, 64)
        require(parity["centered_relative_l2"] <= 2e-4, "gradient-safe full forward parity failed")

        final_cotangent_path = args.output_dir / f"cotangent-{len(config['hybrid_override_pattern']):03d}.npy"
        if not final_cotangent_path.exists():
            initial_kl, final_cotangent = backward_head(
                args.source_dir,
                config,
                np.load(final_boundary),
                teacher_np,
                args.temperature,
            )
            require(math.isfinite(initial_kl), "initial KL is not finite")
            require(np.isfinite(final_cotangent).all(), "head cotangent is not finite")
            atomic_npy(final_cotangent_path, final_cotangent)
            state["initial_kl"] = initial_kl
            state["backward_next_layer"] = len(config["hybrid_override_pattern"]) - 1
            state["phase"] = "backward"
            atomic_json(state_path, state)
            operation_log.write(
                f"head-backward-done kl={initial_kl:.9g} gradient_norm={np.linalg.norm(final_cotangent):.9g}"
            )
        else:
            initial_kl = float(state["initial_kl"])

        next_layer = int(state.get("backward_next_layer", -1))
        for layer in range(next_layer, -1, -1):
            kind = config["hybrid_override_pattern"][layer]
            started = time.perf_counter()
            input_gradient, router_gradient = backward_layer(
                args.source_dir,
                layer,
                kind,
                np.load(activation_dir / f"boundary-{layer:03d}.npy"),
                np.load(args.output_dir / f"cotangent-{layer + 1:03d}.npy"),
                retained,
            )
            require(np.isfinite(input_gradient).all(), f"layer {layer} input gradient is not finite")
            atomic_npy(args.output_dir / f"cotangent-{layer:03d}.npy", input_gradient)
            if router_gradient is not None:
                require(np.isfinite(router_gradient).all(), f"layer {layer} router gradient is not finite")
                require(float(np.linalg.norm(router_gradient)) > 0, f"layer {layer} router gradient is zero")
                atomic_npy(gradient_dir / f"layer-{layer:03d}.npy", router_gradient)
            state["backward_next_layer"] = layer - 1
            atomic_json(state_path, state)
            operation_log.write(
                f"backward layer={layer:02d} kind={kind} input_gradient_norm={np.linalg.norm(input_gradient):.9g} "
                f"router_gradient_norm={0.0 if router_gradient is None else np.linalg.norm(router_gradient):.9g} "
                f"elapsed={time.perf_counter() - started:.3f}s peak_gib={mx.get_peak_memory() / 2**30:.3f}"
            )
        state["phase"] = "backward-complete"
        atomic_json(state_path, state)

        trials = []
        accepted = None
        accepted_routers = None
        accepted_rows = None
        for step in range(args.line_search_steps):
            learning_rate = args.learning_rate / (2**step)
            operation_log.write(f"line-search-start step={step} learning_rate={learning_rate:.9g}")
            routers, gradient_rows = load_router_set(
                args.source_dir, retained, gradient_dir, learning_rate
            )
            candidate_np = student_forward(
                args.source_dir, config, retained, token_ids, None, routers, operation_log
            )
            candidate_logits = mx.array(candidate_np)
            candidate_kl_array = kl_divergence(mx.array(teacher_np), candidate_logits, args.temperature)
            mx.eval(candidate_kl_array)
            candidate_kl = float(candidate_kl_array)
            row = {
                "step": step,
                "learning_rate": learning_rate,
                "kl": candidate_kl,
                "improvement": initial_kl - candidate_kl,
                "logits": compare(teacher_np, candidate_np, 64),
            }
            trials.append(row)
            operation_log.write(
                f"line-search-done step={step} learning_rate={learning_rate:.9g} "
                f"kl={candidate_kl:.9g} improvement={initial_kl - candidate_kl:.9g}"
            )
            if math.isfinite(candidate_kl) and candidate_kl < initial_kl:
                accepted = row
                accepted_routers = routers
                accepted_rows = gradient_rows
                atomic_npy(args.output_dir / "updated-student-logits.npy", candidate_np)
                break
            del routers
            gc.collect()
            mx.clear_cache()
        require(accepted is not None and accepted_routers is not None, "line search found no improving BF16 update")
        artifact = args.output_dir / "router.safetensors"
        save_router_artifact(
            artifact,
            accepted_routers,
            source_state["revision"],
            plan_hash,
            accepted["learning_rate"],
        )
        require(all(row["gradient_finite"] for row in accepted_rows), "router gradients are not finite")
        require(
            all(row["zero_gradient_rows_exact"] for row in accepted_rows),
            "a zero-gradient router row changed",
        )
        require(sum(row["changed_rows"] for row in accepted_rows) > 0, "accepted update changed no router rows")
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "source_state_sha256": source_state_hash,
            "plan_sha256": plan_hash,
            "tool_sha256": tool_hash,
            "prompt_sha256": prompt_sha256,
            "token_ids": token_ids,
            "temperature": args.temperature,
            "initial_kl": initial_kl,
            "accepted": accepted,
            "line_search": trials,
            "production_training_parity": parity,
            "initial_logits": compare(teacher_np, student_np, 64),
            "gradients": accepted_rows,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
        report_path = args.output_dir / "report.json"
        atomic_json(report_path, report)
        state["phase"] = "complete"
        state["report_sha256"] = sha256_file(report_path)
        atomic_json(state_path, state)
        operation_log.write(
            f"run-complete initial_kl={initial_kl:.9g} final_kl={accepted['kl']:.9g} "
            f"artifact={artifact} report_sha256={sha256_file(report_path)}"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        operation_log.write(f"run-failed error={exc}")
        print(f"nemotron streamed Router KD error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

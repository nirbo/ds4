#!/usr/bin/env python3
"""Fit retained Nemotron router rows against unpruned layer outputs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.nemotron_h import group_expert_select
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_layer_distill import capture_inputs
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_proxy_compare import error_metrics
from nemotron_mlx_stream_forward import validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-router-distill-v1"
ARTIFACT_FORMATS = {
    FORMAT,
    "nemotron-multisample-router-kd-v1",
    "nemotron-router-kd-rebalance-v1",
}


def load_router_artifact(
    report_path: Path,
    plan_path: Path,
    source_revision: str,
    retained: dict[str, list[int]],
    hidden_size: int,
) -> tuple[dict[str, mx.array], dict]:
    report = load_json(report_path)
    report_format = report.get("format")
    require(
        report_format in ARTIFACT_FORMATS and report.get("status") == "complete",
        "invalid router report",
    )
    require(report.get("source_revision") == source_revision, "router/source revision mismatch")
    require(report.get("plan_sha256") == sha256_file(plan_path), "router/plan hash mismatch")
    artifact_name = report.get("artifact")
    require(
        isinstance(artifact_name, str) and Path(artifact_name).name == artifact_name,
        "invalid router artifact name",
    )
    artifact = report_path.parent / artifact_name
    require(artifact.is_file(), "router artifact is missing")
    require(report.get("artifact_sha256") == sha256_file(artifact), "router artifact hash mismatch")
    tensors, metadata = mx.load(str(artifact), return_metadata=True)
    if report_format != FORMAT:
        require(metadata.get("format") == report_format, "router artifact format mismatch")
        require(metadata.get("source_revision") == source_revision, "router artifact source mismatch")
        require(metadata.get("plan_sha256") == report["plan_sha256"], "router artifact plan mismatch")
    expected = {f"layer_{int(layer):03d}.gate.weight" for layer in retained}
    require(set(tensors) == expected, "router artifact tensor catalog mismatch")
    result = {}
    for layer, experts in retained.items():
        tensor = tensors[f"layer_{int(layer):03d}.gate.weight"]
        require(tensor.dtype == mx.bfloat16, f"router layer {layer} is not BF16")
        require(
            tensor.shape == (len(experts), hidden_size),
            f"router layer {layer} shape mismatch",
        )
        result[layer] = tensor
    return result, report


def parse_layers(value: str | None, available: list[int]) -> list[int]:
    if value is None:
        return available
    try:
        layers = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise MetadataError(f"invalid layer list: {value}") from exc
    require(layers == sorted(set(layers)), "layers must be sorted and unique")
    require(set(layers) <= set(available), "requested layer is not in the prune plan")
    return layers


def route_indices_scores(
    hidden: mx.array,
    gate_weight: mx.array,
    correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[mx.array, mx.array]:
    return group_expert_select(
        hidden @ gate_weight.T,
        correction_bias,
        top_k,
        1,
        1,
        routed_scaling_factor,
        True,
    )


def aggregate_table(
    hidden: mx.array,
    gate_weight: mx.array,
    correction_bias: mx.array,
    expert_values: mx.array,
    top_k: int,
    routed_scaling_factor: float,
) -> mx.array:
    """Differentiable synthetic equivalent of retained sparse routing."""
    indices, scores = route_indices_scores(
        hidden,
        gate_weight,
        correction_bias,
        top_k,
        routed_scaling_factor,
    )
    selected = mx.take_along_axis(expert_values, indices[..., None], axis=-2)
    return (selected * scores[..., None]).sum(axis=-2)


def adam_update(
    weight: mx.array,
    gradient: mx.array,
    first: mx.array,
    second: mx.array,
    step: int,
    learning_rate: float,
    clip_norm: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[mx.array, mx.array, mx.array, float]:
    require(step > 0 and learning_rate > 0 and clip_norm > 0, "invalid optimizer setting")
    norm = mx.sqrt(mx.sum(mx.square(gradient.astype(mx.float32))))
    mx.eval(norm)
    norm_value = float(norm)
    gradient = gradient * min(1.0, clip_norm / max(norm_value, 1e-20))
    first = beta1 * first + (1.0 - beta1) * gradient
    second = beta2 * second + (1.0 - beta2) * mx.square(gradient)
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    weight = weight - learning_rate * first_hat / (mx.sqrt(second_hat) + epsilon)
    return weight, first, second, norm_value


def teacher_aggregate(block, hidden: mx.array, latent: mx.array) -> mx.array:
    indices, scores = block.route(hidden)
    selected = expert_outputs(latent, block.experts, indices)
    return (selected * scores[..., None]).sum(axis=-2)


def retained_aggregate(
    block,
    hidden: mx.array,
    latent: mx.array,
    retained_indices: mx.array,
    correction_bias: mx.array,
    gate_weight: mx.array,
) -> mx.array:
    local_indices, scores = route_indices_scores(
        hidden,
        gate_weight,
        correction_bias,
        block.top_k,
        block.routed_scaling_factor,
    )
    source_indices = mx.stop_gradient(retained_indices[local_indices])
    selected = expert_outputs(latent, block.experts, source_indices)
    return (selected * scores[..., None]).sum(axis=-2)


def prepare_examples(block, captures, layer: int) -> list[dict]:
    examples = []
    for batch, inputs in captures:
        x = mx.array(inputs[layer])
        hidden = block.norm(x)
        latent = block.fc1_latent(hidden)
        teacher = teacher_aggregate(block, hidden, latent)
        shared_hidden = mx.square(mx.maximum(block.shared_up(hidden), mx.array(0.0, hidden.dtype)))
        shared = block.shared_down(shared_hidden)
        mx.eval(x, hidden, latent, teacher, shared)
        examples.append(
            {
                "category": batch["category"],
                "x": x,
                "hidden": hidden,
                "latent": latent,
                "teacher": teacher,
                "shared": shared,
            }
        )
    return examples


def normalized_aggregate_loss(candidate: mx.array, teacher: mx.array) -> mx.array:
    denominator = mx.mean(mx.square(teacher.astype(mx.float32))) + 1e-8
    return mx.mean(mx.square(candidate.astype(mx.float32) - teacher.astype(mx.float32))) / denominator


def validation_loss(
    block,
    examples: list[dict],
    retained_indices: mx.array,
    correction_bias: mx.array,
    gate_weight: mx.array,
) -> float:
    losses = []
    for example in examples:
        candidate = retained_aggregate(
            block,
            example["hidden"],
            example["latent"],
            retained_indices,
            correction_bias,
            gate_weight,
        )
        loss = normalized_aggregate_loss(candidate, example["teacher"])
        mx.eval(loss)
        losses.append(float(loss))
    return float(np.mean(losses))


def evaluate_layer(
    block,
    examples: list[dict],
    retained_indices: mx.array,
    correction_bias: mx.array,
    source_gate: mx.array,
    trained_gate: mx.array,
    layer: int,
) -> list[dict]:
    rows = []
    for case, example in enumerate(examples):
        source_aggregate = retained_aggregate(
            block,
            example["hidden"],
            example["latent"],
            retained_indices,
            correction_bias,
            source_gate,
        )
        trained_aggregate = retained_aggregate(
            block,
            example["hidden"],
            example["latent"],
            retained_indices,
            correction_bias,
            trained_gate,
        )
        teacher_routed = block.fc2_latent(example["teacher"])
        source_routed = block.fc2_latent(source_aggregate)
        trained_routed = block.fc2_latent(trained_aggregate)
        mx.eval(teacher_routed, source_routed, trained_routed)
        x = np.asarray(example["x"], dtype=np.float32)
        shared = np.asarray(example["shared"], dtype=np.float32)
        teacher_aggregate_np = np.asarray(example["teacher"], dtype=np.float32)
        source_aggregate_np = np.asarray(source_aggregate, dtype=np.float32)
        trained_aggregate_np = np.asarray(trained_aggregate, dtype=np.float32)
        teacher_routed_np = np.asarray(teacher_routed, dtype=np.float32)
        source_routed_np = np.asarray(source_routed, dtype=np.float32)
        trained_routed_np = np.asarray(trained_routed, dtype=np.float32)
        teacher_output = x + shared + teacher_routed_np
        rows.append(
            {
                "layer": layer,
                "case": case,
                "category": example["category"],
                "source_aggregate": error_metrics(source_aggregate_np, teacher_aggregate_np),
                "trained_aggregate": error_metrics(trained_aggregate_np, teacher_aggregate_np),
                "source_routed": error_metrics(source_routed_np, teacher_routed_np),
                "trained_routed": error_metrics(trained_routed_np, teacher_routed_np),
                "source_output": error_metrics(x + shared + source_routed_np, teacher_output),
                "trained_output": error_metrics(x + shared + trained_routed_np, teacher_output),
            }
        )
    return rows


def summarize(rows: list[dict], key: str) -> dict[str, float]:
    values = [row[key]["relative_l2"] for row in rows]
    return {"mean_relative_l2": float(np.mean(values)), "max_relative_l2": max(values)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--train-corpus", required=True, type=Path)
    parser.add_argument("--validation-corpus", required=True, type=Path)
    parser.add_argument("--layers", help="comma-separated MoE layer indexes")
    parser.add_argument("--max-sample-tokens", type=int, default=24)
    parser.add_argument("--train-cases", type=int, default=8)
    parser.add_argument("--validation-cases", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.max_sample_tokens > 0, "max sample tokens must be positive")
        require(args.train_cases > 0 and args.validation_cases > 0, "case counts must be positive")
        require(args.epochs > 0 and args.patience > 0, "epoch settings must be positive")
        require(args.learning_rate > 0 and args.anchor >= 0, "invalid fit settings")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained = validate_virtual_plan(plan, config, source_state["revision"])
        layers = parse_layers(args.layers, [int(layer) for layer in plan["model_moe_layers"]])
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        train_captures = capture_inputs(
            args.source_dir,
            tokenizer,
            args.train_corpus,
            layers,
            args.max_sample_tokens,
            args.train_cases,
            "train",
        )
        validation_captures = capture_inputs(
            args.source_dir,
            tokenizer,
            args.validation_corpus,
            layers,
            args.max_sample_tokens,
            args.validation_cases,
            "validation",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "fit.log")
        artifact_tensors = {}
        histories = {}
        results = []
        for layer in layers:
            started = time.perf_counter()
            operation_log.write(f"layer-start layer={layer}")
            block = load_moe_layer(args.source_dir, layer)
            retained_list = retained[str(layer)]
            retained_indices = mx.array(retained_list, dtype=mx.uint32)
            source_gate_bf16 = block.gate_weight[retained_indices]
            source_gate = source_gate_bf16.astype(mx.float32)
            correction_bias = block.correction_bias[retained_indices]
            train_examples = prepare_examples(block, train_captures, layer)
            validation_examples = prepare_examples(block, validation_captures, layer)
            weight = source_gate
            first = mx.zeros_like(weight)
            second = mx.zeros_like(weight)
            best_weight = np.asarray(source_gate, dtype=np.float32).copy()
            best_loss = validation_loss(
                block,
                validation_examples,
                retained_indices,
                correction_bias,
                source_gate_bf16,
            )
            history = [{"epoch": 0, "validation_loss": best_loss, "improved": False}]
            stale = 0
            step = 0

            def objective(gate_weight, example):
                candidate = retained_aggregate(
                    block,
                    example["hidden"],
                    example["latent"],
                    retained_indices,
                    correction_bias,
                    gate_weight,
                )
                data_loss = normalized_aggregate_loss(candidate, example["teacher"])
                anchor_loss = mx.mean(mx.square(gate_weight - source_gate)) / (
                    mx.mean(mx.square(source_gate)) + 1e-8
                )
                return data_loss + args.anchor * anchor_loss

            value_and_grad = mx.value_and_grad(objective)
            for epoch in range(1, args.epochs + 1):
                train_losses = []
                gradient_norms = []
                for example in train_examples:
                    step += 1
                    loss, gradient = value_and_grad(weight, example)
                    weight, first, second, gradient_norm = adam_update(
                        weight,
                        gradient,
                        first,
                        second,
                        step,
                        args.learning_rate,
                        args.clip_norm,
                    )
                    mx.eval(loss, weight, first, second)
                    train_losses.append(float(loss))
                    gradient_norms.append(gradient_norm)
                runtime_weight = weight.astype(mx.bfloat16)
                current_loss = validation_loss(
                    block,
                    validation_examples,
                    retained_indices,
                    correction_bias,
                    runtime_weight,
                )
                improved = current_loss < best_loss
                if improved:
                    best_loss = current_loss
                    best_weight = np.asarray(weight, dtype=np.float32).copy()
                    stale = 0
                else:
                    stale += 1
                row = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "validation_loss": current_loss,
                    "gradient_norm_mean": float(np.mean(gradient_norms)),
                    "improved": improved,
                }
                history.append(row)
                operation_log.write(
                    f"epoch-done layer={layer} epoch={epoch} train={row['train_loss']:.6g} "
                    f"validation={current_loss:.6g} improved={int(improved)}"
                )
                if stale >= args.patience:
                    break
            best_runtime = mx.array(best_weight).astype(mx.bfloat16)
            artifact_tensors[f"layer_{layer:03d}.gate.weight"] = best_runtime
            layer_results = evaluate_layer(
                block,
                validation_examples,
                retained_indices,
                correction_bias,
                source_gate_bf16,
                best_runtime,
                layer,
            )
            results.extend(layer_results)
            histories[str(layer)] = history
            operation_log.write(
                f"layer-done layer={layer} best_epoch="
                f"{min(history, key=lambda row: row['validation_loss'])['epoch']} "
                f"validation={best_loss:.6g} elapsed={time.perf_counter() - started:.2f}s"
            )
            del block, train_examples, validation_examples, weight, first, second
            gc.collect()
            mx.clear_cache()

        artifact = args.output_dir / "router.safetensors"
        temporary = artifact.with_name(artifact.stem + ".part" + artifact.suffix)
        mx.save_safetensors(
            str(temporary),
            artifact_tensors,
            metadata={
                "format": FORMAT,
                "source_revision": source_state["revision"],
                "plan_sha256": sha256_file(args.plan),
            },
        )
        temporary.replace(artifact)
        summary = {
            key: summarize(results, key)
            for key in (
                "source_aggregate",
                "trained_aggregate",
                "source_routed",
                "trained_routed",
                "source_output",
                "trained_output",
            )
        }
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "tool_sha256": sha256_file(Path(__file__)),
            "plan_sha256": sha256_file(args.plan),
            "train_corpus_sha256": sha256_file(args.train_corpus),
            "validation_corpus_sha256": sha256_file(args.validation_corpus),
            "layers": layers,
            "max_sample_tokens": args.max_sample_tokens,
            "train_cases": args.train_cases,
            "validation_cases": args.validation_cases,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "anchor": args.anchor,
            "clip_norm": args.clip_norm,
            "patience": args.patience,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "histories": histories,
            "results": results,
            "summary": summary,
        }
        atomic_json(args.output_dir / "report.json", report)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(
            f"router-distill path={args.output_dir / 'report.json'} "
            f"sha256={sha256_file(args.output_dir / 'report.json')}"
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError, RuntimeError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron router distill error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

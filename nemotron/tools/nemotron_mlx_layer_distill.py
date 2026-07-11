#!/usr/bin/env python3
"""Fit and validate small linear corrections for a structurally pruned plan."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import build_batches, corpus_samples, sha256_file
from nemotron_mlx_layer_sensitivity import baseline_components, pruned_routed_output
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_proxy_compare import error_metrics
from nemotron_mlx_stream_forward import StreamingForward, validate_virtual_plan
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state


FORMAT = "nemotron-layer-affine-distill-v1"


def fit_affine(candidate: np.ndarray, teacher: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    require(candidate.shape == teacher.shape and candidate.ndim == 2, "affine fit shape mismatch")
    require(candidate.shape[0] >= 2, "affine fit requires at least two observations")
    require(ridge >= 0.0, "affine ridge must be nonnegative")
    candidate = candidate.astype(np.float64)
    teacher = teacher.astype(np.float64)
    candidate_mean = candidate.mean(axis=0)
    teacher_mean = teacher.mean(axis=0)
    centered_candidate = candidate - candidate_mean
    centered_teacher = teacher - teacher_mean
    variance = np.mean(centered_candidate * centered_candidate, axis=0)
    covariance = np.mean(centered_candidate * centered_teacher, axis=0)
    regularizer = ridge * max(float(np.mean(variance)), 1e-12)
    # Penalize deviation from the exact no-correction identity, not from zero.
    scale = (covariance + regularizer) / (variance + regularizer)
    bias = teacher_mean - scale * candidate_mean
    return scale.astype(np.float32), bias.astype(np.float32)


def fit_low_rank(
    hidden: np.ndarray,
    residual: np.ndarray,
    rank: int,
    ridge: float,
    fit_bias: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require(hidden.ndim == residual.ndim == 2, "low-rank fit requires matrices")
    require(hidden.shape == residual.shape, "low-rank hidden/residual shape mismatch")
    require(0 < rank < hidden.shape[0], "low-rank rank exceeds observations")
    require(ridge >= 0.0, "low-rank ridge must be nonnegative")
    hidden = hidden.astype(np.float64)
    residual = residual.astype(np.float64)
    hidden_mean = hidden.mean(axis=0)
    residual_mean = residual.mean(axis=0) if fit_bias else np.zeros(residual.shape[1])
    centered_hidden = hidden - hidden_mean
    centered_residual = residual - residual_mean
    _, _, right = np.linalg.svd(centered_hidden, full_matrices=False)
    input_basis = right[:rank]
    coefficients = centered_hidden @ input_basis.T
    gram = coefficients.T @ coefficients
    regularizer = ridge * max(float(np.trace(gram) / rank), 1e-12)
    output_basis = np.linalg.solve(
        gram + regularizer * np.eye(rank),
        coefficients.T @ centered_residual,
    )
    return tuple(
        value.astype(np.float32)
        for value in (input_basis, output_basis, hidden_mean, residual_mean)
    )


def apply_low_rank(
    hidden: np.ndarray,
    input_basis: np.ndarray,
    output_basis: np.ndarray,
    hidden_mean: np.ndarray,
    residual_mean: np.ndarray,
) -> np.ndarray:
    centered = hidden.astype(np.float32) - hidden_mean
    return (centered @ input_basis.T) @ output_basis + residual_mean


def expert_aggregate(block, x: mx.array, retained: list[int] | None) -> tuple[mx.array, mx.array]:
    hidden = block.norm(x)
    latent = block.fc1_latent(hidden)
    if retained is None:
        indices, scores = block.route(hidden)
    else:
        indices, scores = block.route_retained(hidden, retained)
    selected = expert_outputs(latent, block.experts, indices)
    return latent, (selected * scores[..., None]).sum(axis=-2)


def relu2_features(
    latent: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    projected = (latent.astype(np.float32) - mean) @ basis.T
    features = np.concatenate(
        (np.maximum(projected, 0.0) ** 2, np.maximum(-projected, 0.0) ** 2),
        axis=1,
    )
    return features / feature_scale


def fit_latent_relu2(
    latent: np.ndarray,
    residual: np.ndarray,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require(latent.ndim == residual.ndim == 2, "latent adapter requires matrices")
    require(latent.shape == residual.shape, "latent adapter shape mismatch")
    require(0 < rank < latent.shape[0], "latent adapter rank exceeds observations")
    mean = latent.astype(np.float64).mean(axis=0)
    centered = latent.astype(np.float64) - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    basis = right[:rank].astype(np.float32)
    raw = relu2_features(
        latent,
        basis,
        mean.astype(np.float32),
        np.ones(2 * rank, dtype=np.float32),
    )
    feature_scale = np.maximum(np.sqrt(np.mean(raw * raw, axis=0)), 1e-6).astype(np.float32)
    features = raw / feature_scale
    gram = features.T @ features
    regularizer = ridge * max(float(np.trace(gram) / (2 * rank)), 1e-12)
    output = np.linalg.solve(
        gram + regularizer * np.eye(2 * rank),
        features.T @ residual.astype(np.float32),
    ).astype(np.float32)
    return basis, mean.astype(np.float32), feature_scale, output


def apply_latent_relu2(latent: np.ndarray, correction: tuple[np.ndarray, ...]) -> np.ndarray:
    basis, mean, feature_scale, output = correction
    return relu2_features(latent, basis, mean, feature_scale) @ output


def capture_inputs(
    source_dir: Path,
    tokenizer,
    corpus: Path,
    layers: list[int],
    max_sample_tokens: int,
    max_cases: int,
    label: str,
    concatenate: bool = False,
) -> list[tuple[dict, dict[int, np.ndarray]]]:
    batches = build_batches(
        tokenizer,
        corpus_samples(corpus),
        max_sample_tokens,
        max_sample_tokens,
    )[:max_cases]
    require(len(batches) == max_cases, f"{label} corpus has too few cases")
    if concatenate:
        batches = [
            {
                "category": "concatenated",
                "token_ids": [token for batch in batches for token in batch["token_ids"]],
            }
        ]
    captured = []
    for index, batch in enumerate(batches):
        started = time.perf_counter()
        print(
            f"distill-capture-start split={label} case={index} category={batch['category']} "
            f"tokens={len(batch['token_ids'])}",
            flush=True,
        )
        runner = StreamingForward(source_dir)
        runner.forward_sequence(
            batch["token_ids"],
            max_layers=max(layers) + 1,
            score_head=False,
            capture_layer_inputs=set(layers),
        )
        require(set(runner.layer_inputs) == set(layers), f"{label} layer capture is incomplete")
        captured.append((batch, runner.layer_inputs))
        print(
            f"distill-capture-done split={label} case={index} "
            f"elapsed={time.perf_counter() - started:.2f}s",
            flush=True,
        )
        del runner
        gc.collect()
        mx.clear_cache()
    return captured


def aggregate_metrics(rows: list[dict], key: str) -> dict[str, float]:
    values = [row[key]["relative_l2"] for row in rows]
    return {"mean_relative_l2": float(np.mean(values)), "max_relative_l2": max(values)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--train-corpus", required=True, type=Path)
    parser.add_argument("--validation-corpus", required=True, type=Path)
    parser.add_argument("--max-sample-tokens", type=int, default=24)
    parser.add_argument("--train-cases", type=int, default=4)
    parser.add_argument("--validation-cases", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--method",
        choices=("affine", "low-rank", "latent-relu2"),
        default="affine",
    )
    parser.add_argument("--granularity", choices=("scalar", "channel"), default="scalar")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--low-rank-bias", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.max_sample_tokens > 0, "max sample tokens must be positive")
        require(args.train_cases > 0 and args.validation_cases > 0, "case counts must be positive")
        require(args.rank > 0, "rank must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained = validate_virtual_plan(plan, config, source_state["revision"])
        layers = plan["model_moe_layers"]
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        train = capture_inputs(
            args.source_dir,
            tokenizer,
            args.train_corpus,
            layers,
            args.max_sample_tokens,
            args.train_cases,
            "train",
        )
        validation = capture_inputs(
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
        corrections = {}
        results = []
        for layer in layers:
            operation_log.write(f"layer-start layer={layer}")
            block = load_moe_layer(args.source_dir, layer)
            candidate_rows = []
            teacher_rows = []
            hidden_rows = []
            latent_rows = []
            for _, inputs in train:
                x = mx.array(inputs[layer])
                if args.method == "latent-relu2":
                    latent, teacher_value = expert_aggregate(block, x, None)
                    _, candidate_value = expert_aggregate(block, x, retained[str(layer)])
                    mx.eval(latent, teacher_value, candidate_value)
                    latent_rows.append(np.asarray(latent, dtype=np.float32).reshape(-1, latent.shape[-1]))
                else:
                    hidden = block.norm(x)
                    teacher_value, _ = baseline_components(block, x)
                    candidate_value = pruned_routed_output(block, x, retained[str(layer)])
                    mx.eval(hidden, teacher_value, candidate_value)
                    hidden_rows.append(np.asarray(hidden, dtype=np.float32).reshape(-1, config["hidden_size"]))
                teacher_rows.append(np.asarray(teacher_value, dtype=np.float32).reshape(-1, teacher_value.shape[-1]))
                candidate_rows.append(np.asarray(candidate_value, dtype=np.float32).reshape(-1, candidate_value.shape[-1]))
            candidate_fit = np.concatenate(candidate_rows)
            teacher_fit = np.concatenate(teacher_rows)
            if args.method == "affine":
                if args.granularity == "scalar":
                    candidate_fit = candidate_fit.reshape(-1, 1)
                    teacher_fit = teacher_fit.reshape(-1, 1)
                correction = fit_affine(candidate_fit, teacher_fit, args.ridge)
                scale, bias = correction
                corrections[f"layer_{layer:03d}.scale"] = mx.array(scale)
                corrections[f"layer_{layer:03d}.bias"] = mx.array(bias)
            elif args.method == "low-rank":
                correction = fit_low_rank(
                    np.concatenate(hidden_rows),
                    teacher_fit - candidate_fit,
                    args.rank,
                    args.ridge,
                    args.low_rank_bias,
                )
                names = ("input_basis", "output_basis", "hidden_mean", "residual_mean")
                for name, value in zip(names, correction):
                    corrections[f"layer_{layer:03d}.{name}"] = mx.array(value)
            else:
                correction = fit_latent_relu2(
                    np.concatenate(latent_rows),
                    teacher_fit - candidate_fit,
                    args.rank,
                    args.ridge,
                )
                names = ("basis", "latent_mean", "feature_scale", "output")
                for name, value in zip(names, correction):
                    corrections[f"layer_{layer:03d}.{name}"] = mx.array(value)
            for case, (_, inputs) in enumerate(validation):
                x_np = inputs[layer].astype(np.float32)
                x = mx.array(x_np)
                hidden = block.norm(x)
                teacher_routed, shared = baseline_components(block, x)
                if args.method == "latent-relu2":
                    latent, candidate_aggregate = expert_aggregate(block, x, retained[str(layer)])
                    mx.eval(hidden, teacher_routed, candidate_aggregate, shared, latent)
                    candidate_routed = block.fc2_latent(candidate_aggregate)
                    corrected_aggregate = np.asarray(candidate_aggregate, dtype=np.float32) + apply_latent_relu2(
                        np.asarray(latent, dtype=np.float32).reshape(-1, latent.shape[-1]),
                        correction,
                    ).reshape(candidate_aggregate.shape)
                    corrected_routed = block.fc2_latent(mx.array(corrected_aggregate))
                    mx.eval(candidate_routed, corrected_routed)
                else:
                    candidate_routed = pruned_routed_output(block, x, retained[str(layer)])
                    mx.eval(hidden, teacher_routed, candidate_routed, shared)
                teacher_np = np.asarray(teacher_routed, dtype=np.float32)
                candidate_np = np.asarray(candidate_routed, dtype=np.float32)
                shared_np = np.asarray(shared, dtype=np.float32)
                if args.method == "affine":
                    corrected_np = candidate_np * scale.reshape(1, 1, -1) + bias.reshape(1, 1, -1)
                elif args.method == "low-rank":
                    hidden_np = np.asarray(hidden, dtype=np.float32)
                    corrected_np = candidate_np + apply_low_rank(
                        hidden_np.reshape(-1, config["hidden_size"]),
                        *correction,
                    ).reshape(candidate_np.shape)
                else:
                    corrected_np = np.asarray(corrected_routed, dtype=np.float32)
                baseline_output = x_np + teacher_np + shared_np
                results.append(
                    {
                        "layer": layer,
                        "case": case,
                        "category": validation[case][0]["category"],
                        "uncorrected_routed": error_metrics(candidate_np, teacher_np),
                        "corrected_routed": error_metrics(corrected_np, teacher_np),
                        "uncorrected_output": error_metrics(x_np + candidate_np + shared_np, baseline_output),
                        "corrected_output": error_metrics(x_np + corrected_np + shared_np, baseline_output),
                    }
                )
            if args.method == "affine":
                detail = (
                    f"scale_min={float(scale.min()):.6g} scale_max={float(scale.max()):.6g} "
                    f"bias_max={float(np.max(np.abs(bias))):.6g}"
                )
            elif args.method == "low-rank":
                detail = (
                    f"rank={args.rank} residual_mean_max="
                    f"{float(np.max(np.abs(correction[3]))):.6g}"
                )
            else:
                detail = f"rank={args.rank} features={2 * args.rank} latent_dims={candidate_fit.shape[1]}"
            operation_log.write(f"layer-done layer={layer} {detail}")
            del block
            gc.collect()
            mx.clear_cache()
        artifact = args.output_dir / "corrections.safetensors"
        temporary = artifact.with_name(artifact.stem + ".part" + artifact.suffix)
        mx.save_safetensors(
            str(temporary),
            corrections,
            metadata={"format": FORMAT, "method": args.method},
        )
        temporary.replace(artifact)
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "plan_sha256": sha256_file(args.plan),
            "train_corpus_sha256": sha256_file(args.train_corpus),
            "validation_corpus_sha256": sha256_file(args.validation_corpus),
            "max_sample_tokens": args.max_sample_tokens,
            "train_cases": args.train_cases,
            "validation_cases": args.validation_cases,
            "ridge": args.ridge,
            "method": args.method,
            "granularity": args.granularity,
            "rank": args.rank if args.method == "low-rank" else None,
            "low_rank_bias": args.low_rank_bias if args.method == "low-rank" else None,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "results": results,
            "summary": {
                key: aggregate_metrics(results, key)
                for key in (
                    "uncorrected_routed",
                    "corrected_routed",
                    "uncorrected_output",
                    "corrected_output",
                )
            },
        }
        atomic_json(args.output_dir / "report.json", report)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(f"distill-report path={args.output_dir / 'report.json'} sha256={sha256_file(args.output_dir / 'report.json')}")
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError, RuntimeError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron layer distill error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Calibrate the deployed BF16 MTP final norm on exact recursive target rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_mtp_distill import load_features, reduced_target_indices
from nemotron_mlx_mtp_predictor import AdamWControl, load_capture, sequence_starts
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-final-norm-calibration-v1"
FINAL_NORM = "mtp.layers.1.final_layernorm.weight"


def load_calibrated_final_norm(
    artifact_dir: Path,
    model_dir: Path,
    sidecar: Path,
    mtp_lm_head: Path,
) -> mx.array:
    report = load_json(artifact_dir / "report.json")
    require(
        report.get("format") == FORMAT and report.get("status") == "diagnostic",
        "MTP final norm calibration report is invalid",
    )
    require(
        Path(report.get("sidecar", "")).resolve() == sidecar.resolve()
        and report.get("sidecar_report_sha256")
        == sha256_file(sidecar / "nemotron_mtp_pack_report.json"),
        "MTP final norm calibration sidecar identity mismatch",
    )
    require(
        Path(report.get("model_dir", "")).resolve() == model_dir.resolve()
        and report.get("model_report_sha256")
        == sha256_file(model_dir / "nemotron_mlx_pack_report.json"),
        "MTP final norm calibration target identity mismatch",
    )
    require(
        report.get("mtp_lm_head_report_sha256")
        == sha256_file(mtp_lm_head / "nemotron_mtp_vocab_head_report.json"),
        "MTP final norm calibration vocabulary identity mismatch",
    )
    artifact_name = report.get("artifact")
    require(
        isinstance(artifact_name, str)
        and artifact_name
        and Path(artifact_name).name == artifact_name,
        "MTP final norm calibration artifact name is invalid",
    )
    artifact = artifact_dir / artifact_name
    require(
        artifact.is_file()
        and artifact.stat().st_size == report.get("artifact_bytes")
        and sha256_file(artifact) == report.get("artifact_sha256"),
        "MTP final norm calibration artifact identity mismatch",
    )
    arrays, metadata = mx.load(str(artifact), return_metadata=True)
    require(
        metadata.get("format") == FORMAT
        and set(arrays) == {FINAL_NORM}
        and arrays[FINAL_NORM].dtype == mx.bfloat16
        and arrays[FINAL_NORM].ndim == 1,
        "MTP final norm calibration payload is invalid",
    )
    return arrays[FINAL_NORM]


def damped_final_norm(
    original: mx.array, calibrated: mx.array, damping: float
) -> mx.array:
    require(
        original.dtype == calibrated.dtype == mx.bfloat16
        and original.shape == calibrated.shape,
        "MTP final norm damping payload mismatch",
    )
    require(
        np.isfinite(damping) and 0 <= damping <= 1,
        "MTP final norm damping must be between zero and one",
    )
    return (
        original.astype(mx.float32)
        + damping * (calibrated.astype(mx.float32) - original.astype(mx.float32))
    ).astype(mx.bfloat16)


def parse_depth_weights(value: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depth weights must be comma-separated numbers") from exc
    if not weights or any(not np.isfinite(weight) or weight <= 0 for weight in weights):
        raise argparse.ArgumentTypeError("depth weights must be finite and positive")
    return weights


def calibration_examples(
    prompt_indices: list[int],
    expected_reduced: list[int],
    max_depth: int,
    parity: int,
) -> tuple[list[int], list[int], list[int]]:
    starts = sequence_starts(prompt_indices, max_depth - 1, parity)
    rows = []
    depths = []
    labels = []
    for start in starts:
        for depth in range(max_depth):
            label = expected_reduced[start + depth]
            if label >= 0:
                rows.append(start)
                depths.append(depth)
                labels.append(label)
    return rows, depths, labels


def calibration_loss(
    parameters: dict[str, mx.array],
    rows: mx.array,
    depths: mx.array,
    labels: mx.array,
    teacher_hidden: mx.array,
    head_weight: mx.array,
    depth_weights: mx.array,
    identity_weight: float,
) -> mx.array:
    scale = parameters["scale"]
    hidden = teacher_hidden[rows, depths] * scale
    logits = hidden @ head_weight.T.astype(mx.float32)
    cross_entropy = mx.logsumexp(logits, axis=-1) - logits[
        mx.arange(rows.size), labels
    ]
    weighted = cross_entropy * depth_weights[depths]
    return mx.mean(weighted) + identity_weight * mx.mean(mx.square(scale - 1))


def recursive_metrics(
    scale: mx.array,
    starts: list[int],
    teacher_hidden: mx.array,
    expected_token_ids: mx.array,
    target_token_ids: mx.array,
    head_weight: mx.array,
    max_depth: int,
    batch_size: int,
) -> dict:
    attempts = [0] * max_depth
    matches = [0] * max_depth
    for offset in range(0, len(starts), batch_size):
        batch = mx.array(starts[offset : offset + batch_size], dtype=mx.int32)
        alive = mx.ones((batch.size,), dtype=mx.bool_)
        for depth in range(max_depth):
            logits = (teacher_hidden[batch, depth] * scale) @ head_weight.T.astype(
                mx.float32
            )
            predicted = target_token_ids[mx.argmax(logits, axis=-1)]
            expected = expected_token_ids[batch + depth]
            mx.eval(predicted, alive)
            active = int(mx.sum(alive))
            attempts[depth] += active
            alive = alive & (predicted == expected)
            matches[depth] += int(mx.sum(alive))
    return {
        str(depth + 1): {
            "attempts": attempts[depth],
            "matches": matches[depth],
            "conditional_acceptance": (
                matches[depth] / attempts[depth] if attempts[depth] else None
            ),
        }
        for depth in range(max_depth)
    }


def accepted_checkpoint(metrics: dict, baseline: dict) -> bool:
    return metrics["1"]["matches"] >= baseline["1"]["matches"]


def checkpoint_score(metrics: dict) -> tuple[int, ...]:
    depths = sorted(metrics, key=int)
    return tuple(metrics[depth]["matches"] for depth in reversed(depths))


def load_final_norm(sidecar_dir: Path) -> mx.array:
    index = load_json(sidecar_dir / "model.safetensors.index.json")
    shard_name = index.get("weight_map", {}).get(FINAL_NORM)
    require(isinstance(shard_name, str), "MTP sidecar final norm is absent from index")
    tensors = mx.load(str(sidecar_dir / shard_name))
    require(
        FINAL_NORM in tensors
        and tensors[FINAL_NORM].dtype == mx.bfloat16
        and tensors[FINAL_NORM].ndim == 1,
        "MTP sidecar final norm is invalid",
    )
    result = tensors[FINAL_NORM]
    mx.eval(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-depth", type=int, choices=(2, 3), default=3)
    parser.add_argument("--depth-weights", type=parse_depth_weights, default=(2.0, 1.0, 0.5))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--identity-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(
            args.epochs > 0
            and args.batch_size > 0
            and args.eval_batch_size > 0
            and args.learning_rate > 0
            and args.identity_weight >= 0,
            "invalid MTP norm calibration settings",
        )
        require(
            len(args.depth_weights) >= args.max_depth,
            "MTP norm calibration has insufficient depth weights",
        )
        arrays, capture_state = load_capture(args.capture_dir)
        capture_hash = sha256_file(args.capture_dir / "state.json")
        features, feature_state = load_features(args.features_dir, capture_hash)
        identity = feature_state["identity"]
        require(
            identity["model_dir"] == str(args.model_dir.resolve())
            and identity["mtp_sidecar"] == str(args.sidecar.resolve())
            and identity["mtp_lm_head"] == str(args.mtp_lm_head.resolve()),
            "MTP norm calibration input identity mismatch",
        )
        require(
            features["teacher_hidden"].shape[1] >= args.max_depth,
            "MTP norm calibration features have insufficient depth",
        )

        map_arrays, map_metadata = mx.load(
            str(args.mtp_lm_head / "lm_head.safetensors"), return_metadata=True
        )
        require(
            map_metadata.get("storage") == "shared-target-bf16",
            "MTP norm calibration requires the shared target head",
        )
        target_token_ids = map_arrays["target_token_ids"]
        global_tensors = mx.load(str(args.model_dir / "global.safetensors"))
        head_weight = global_tensors["lm_head.weight"][target_token_ids]
        mx.eval(head_weight)
        del global_tensors

        prompt_indices = arrays["prompt_indices"].tolist()
        expected_tokens = arrays["expected_token_ids"]
        expected_reduced = reduced_target_indices(
            target_token_ids.tolist(), expected_tokens.tolist()
        ).tolist()
        train_rows, train_depths, train_labels = calibration_examples(
            prompt_indices, expected_reduced, args.max_depth, 0
        )
        held_starts = sequence_starts(prompt_indices, args.max_depth - 1, 1)
        require(train_rows and held_starts, "MTP norm calibration split is empty")

        original_norm = load_final_norm(args.sidecar)
        require(
            original_norm.shape == (features["teacher_hidden"].shape[-1],),
            "MTP final norm hidden size mismatch",
        )
        parameters = {"scale": mx.ones(original_norm.shape, dtype=mx.float32)}
        optimizer = AdamWControl(parameters, args.learning_rate, 0.0)
        depth_weights = mx.array(args.depth_weights[: args.max_depth], dtype=mx.float32)
        baseline = recursive_metrics(
            parameters["scale"],
            held_starts,
            features["teacher_hidden"],
            expected_tokens,
            target_token_ids,
            head_weight,
            args.max_depth,
            args.eval_batch_size,
        )

        def loss_fn(parameter_values, rows, depths, labels):
            return calibration_loss(
                parameter_values,
                rows,
                depths,
                labels,
                features["teacher_hidden"],
                head_weight,
                depth_weights,
                args.identity_weight,
            )

        value_and_grad = mx.value_and_grad(loss_fn)
        order = list(range(len(train_rows)))
        rng = random.Random(args.seed)
        history = []
        best = None
        best_scale = np.ones(original_norm.shape, dtype=np.float32)
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(order)
            losses = []
            for offset in range(0, len(order), args.batch_size):
                selected = order[offset : offset + args.batch_size]
                rows = mx.array([train_rows[index] for index in selected], dtype=mx.int32)
                depths = mx.array([train_depths[index] for index in selected], dtype=mx.int32)
                labels = mx.array([train_labels[index] for index in selected], dtype=mx.int32)
                loss, gradients = value_and_grad(parameters, rows, depths, labels)
                parameters = optimizer.update(parameters, gradients)
                mx.eval(loss, parameters["scale"])
                losses.append(float(loss))
            metrics = recursive_metrics(
                parameters["scale"],
                held_starts,
                features["teacher_hidden"],
                expected_tokens,
                target_token_ids,
                head_weight,
                args.max_depth,
                args.eval_batch_size,
            )
            row = {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "metrics": metrics,
            }
            history.append(row)
            if accepted_checkpoint(metrics, baseline) and (
                best is None
                or checkpoint_score(metrics) > checkpoint_score(best["metrics"])
            ):
                best = row
                best_scale = np.asarray(parameters["scale"], dtype=np.float32).copy()
            print("mtp-norm-calibrate-epoch " + json.dumps(row, separators=(",", ":")), flush=True)

        require(best is not None, "MTP norm calibration never preserved depth one")
        scale = mx.array(best_scale)
        calibrated_norm = (original_norm.astype(mx.float32) * scale).astype(mx.bfloat16)
        deployed_scale = calibrated_norm.astype(mx.float32) / original_norm.astype(mx.float32)
        require(bool(mx.all(mx.isfinite(deployed_scale))), "calibrated MTP norm is not finite")
        final_metrics = recursive_metrics(
            deployed_scale,
            held_starts,
            features["teacher_hidden"],
            expected_tokens,
            target_token_ids,
            head_weight,
            args.max_depth,
            args.eval_batch_size,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = args.output_dir / "final_norm.safetensors"
        temporary = artifact.with_name("final_norm.part.safetensors")
        mx.save_safetensors(
            str(temporary),
            {FINAL_NORM: calibrated_norm},
            metadata={"format": FORMAT},
        )
        temporary.replace(artifact)
        report = {
            "format": FORMAT,
            "status": "diagnostic",
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(
                args.model_dir / "nemotron_mlx_pack_report.json"
            ),
            "sidecar": str(args.sidecar.resolve()),
            "sidecar_report_sha256": sha256_file(
                args.sidecar / "nemotron_mtp_pack_report.json"
            ),
            "capture_state_sha256": capture_hash,
            "features_state_sha256": sha256_file(args.features_dir / "state.json"),
            "mtp_lm_head_report_sha256": sha256_file(
                args.mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
            ),
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "max_depth": args.max_depth,
            "depth_weights": list(args.depth_weights[: args.max_depth]),
            "identity_weight": args.identity_weight,
            "training_examples": len(train_rows),
            "heldout_starts": len(held_starts),
            "baseline": baseline,
            "best": best,
            "final": final_metrics,
            "scale_max_abs_delta": float(mx.max(mx.abs(deployed_scale - 1))),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "artifact": artifact.name,
            "artifact_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256_file(artifact),
            "decision": "diagnostic-pending-physical-recursive-gate",
        }
        atomic_json(args.output_dir / "report.json", report)
        print("mtp-norm-calibrate-done " + json.dumps(report, separators=(",", ":")), flush=True)
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron MTP norm calibration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

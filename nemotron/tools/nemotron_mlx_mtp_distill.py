#!/usr/bin/env python3
"""Distill official recursive Nemotron MTP calls into a compact Metal student."""

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
from nemotron_mlx_gefen import GefenMLX, UPSTREAM_REVISION
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp_distill_features import FORMAT as FEATURE_FORMAT
from nemotron_mlx_mtp_distill_features import validate_feature_shard
from nemotron_mlx_mtp_predictor import (
    AdamWControl,
    FORMAT as PREDICTOR_FORMAT,
    GATED_ARCHITECTURE,
    TANH_ARCHITECTURE,
    TOKEN_ARCHITECTURE,
    acceptance_score,
    initialize_gated_parameters,
    initialize_parameters,
    initialize_token_classifier,
    load_capture,
    predict_hidden,
    predict_token_logits,
    sequence_starts,
)
from nemotron_paged_embeddings import PagedBF16Embedding
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-recursive-distillation-v1"


def load_features(
    features_dir: Path,
    capture_state_sha256: str,
) -> tuple[dict[str, mx.array], dict]:
    state = load_json(features_dir / "state.json")
    identity = state.get("identity", {})
    require(
        state.get("format") == FEATURE_FORMAT and state.get("status") == "complete",
        "MTP distillation features are incomplete",
    )
    require(
        identity.get("capture_state_sha256") == capture_state_sha256,
        "MTP distillation feature capture identity mismatch",
    )
    max_depth = identity.get("max_depth")
    top_k = identity.get("top_k")
    hidden = []
    tokens = []
    top_indices = []
    top_logits_values = []
    for key, entry in sorted(state["completed"].items(), key=lambda item: int(item[0])):
        path = features_dir / entry["file"]
        validate_feature_shard(path, entry, int(key), max_depth, top_k)
        arrays = mx.load(str(path))
        hidden.append(arrays["teacher_hidden"])
        tokens.append(arrays["teacher_token_ids"])
        top_indices.append(arrays["teacher_top_indices"])
        top_logits_values.append(arrays["teacher_top_logits"])
    result = {
        "teacher_hidden": mx.concatenate(hidden),
        "teacher_token_ids": mx.concatenate(tokens),
        "teacher_top_indices": mx.concatenate(top_indices),
        "teacher_top_logits": mx.concatenate(top_logits_values),
    }
    require(
        result["teacher_hidden"].shape[0] == state.get("rows"),
        "MTP distillation feature row count mismatch",
    )
    mx.eval(*result.values())
    return result, state


def initialize_student(
    architecture: str,
    hidden_size: int,
    rank: int,
    learned_depths: int,
    seed: int,
    vocabulary_size: int,
) -> dict[str, mx.array]:
    if architecture == TOKEN_ARCHITECTURE:
        require(learned_depths == 1, "token-only MTP student supports only depth two")
        return initialize_token_classifier(hidden_size, rank, vocabulary_size, seed)
    if architecture == GATED_ARCHITECTURE:
        return initialize_gated_parameters(hidden_size, rank, learned_depths, seed)
    require(architecture == TANH_ARCHITECTURE, "unsupported MTP student architecture")
    return initialize_parameters(hidden_size, rank, learned_depths, seed)


def selected_logit_kl(
    prediction: mx.array,
    teacher_indices: mx.array,
    teacher_logits: mx.array,
    head_weight: mx.array,
    temperature: float,
) -> mx.array:
    selected_weight = head_weight[teacher_indices].astype(mx.float32)
    student_logits = mx.sum(selected_weight * prediction[:, None, :], axis=-1)
    teacher_scaled = teacher_logits / temperature
    student_scaled = student_logits / temperature
    teacher_probability = mx.softmax(teacher_scaled, axis=-1)
    teacher_log_probability = teacher_scaled - mx.logsumexp(
        teacher_scaled, axis=-1, keepdims=True
    )
    student_log_probability = student_scaled - mx.logsumexp(
        student_scaled, axis=-1, keepdims=True
    )
    return mx.sum(
        teacher_probability * (teacher_log_probability - student_log_probability),
        axis=-1,
    ) * temperature**2


def reduced_target_indices(
    target_token_ids: list[int], expected_token_ids: list[int]
) -> mx.array:
    """Map authoritative target tokens into the predictor's reduced vocabulary."""
    inverse = {token_id: index for index, token_id in enumerate(target_token_ids)}
    require(
        len(inverse) == len(target_token_ids),
        "MTP reduced vocabulary contains duplicate target tokens",
    )
    return mx.array(
        [inverse.get(token_id, -1) for token_id in expected_token_ids],
        dtype=mx.int32,
    )


def distillation_loss(
    parameters: dict[str, mx.array],
    starts: mx.array,
    features: dict[str, mx.array],
    accepted_embeddings: mx.array,
    expected_token_ids: mx.array,
    expected_reduced_indices: mx.array,
    head_weight: mx.array,
    architecture: str,
    learned_depths: int,
    hidden_weight: float,
    logit_weight: float,
    full_logit_weight: float,
    temperature: float,
) -> mx.array:
    hidden = features["teacher_hidden"][starts, 0]
    alive = mx.ones((starts.size,), dtype=mx.float32)
    total = mx.array(0.0)
    total_weight = mx.array(0.0)
    for depth in range(learned_depths):
        rows = starts + depth + 1
        prediction = predict_hidden(
            parameters,
            hidden,
            accepted_embeddings[rows].astype(mx.float32),
            depth,
            architecture,
        )
        teacher_hidden = features["teacher_hidden"][starts, depth + 1]
        hidden_error = mx.mean(mx.square(prediction - teacher_hidden), axis=-1) / (
            mx.mean(mx.square(teacher_hidden), axis=-1) + 1e-8
        )
        logit_error = selected_logit_kl(
            prediction,
            features["teacher_top_indices"][starts, depth + 1],
            features["teacher_top_logits"][starts, depth + 1],
            head_weight,
            temperature,
        )
        full_logit_error = mx.zeros_like(hidden_error)
        if full_logit_weight > 0:
            full_logits = prediction @ head_weight.T.astype(mx.float32)
            labels = expected_reduced_indices[rows]
            full_logit_error = mx.logsumexp(full_logits, axis=-1) - full_logits[
                mx.arange(starts.size), labels
            ]
        total = total + mx.sum(
            alive
            * (
                hidden_weight * hidden_error
                + logit_weight * logit_error
                + full_logit_weight * full_logit_error
            )
        )
        total_weight = total_weight + mx.sum(alive)
        teacher_token = features["teacher_token_ids"][starts, depth + 1]
        alive = alive * (teacher_token == expected_token_ids[rows]).astype(mx.float32)
        hidden = prediction
    return total / mx.maximum(total_weight, 1.0)


def official_acceptance_metrics(
    starts: list[int],
    expected: list[int],
    teacher_tokens: list[list[int]],
    max_depth: int,
) -> dict:
    attempts = [0] * max_depth
    matches = [0] * max_depth
    for start in starts:
        for depth in range(max_depth):
            attempts[depth] += 1
            if teacher_tokens[start][depth] != expected[start + depth]:
                break
            matches[depth] += 1
    return {
        str(depth + 1): {
            "attempts": attempts[depth],
            "matches": matches[depth],
            "conditional_acceptance": matches[depth] / attempts[depth]
            if attempts[depth]
            else None,
        }
        for depth in range(max_depth)
    }


def student_acceptance_metrics(
    parameters: dict[str, mx.array],
    starts: list[int],
    features: dict[str, mx.array],
    accepted_embeddings: mx.array,
    expected_token_ids: mx.array,
    target_token_ids: mx.array,
    head: ModelOptBF16Linear,
    architecture: str,
    learned_depths: int,
    batch_size: int = 32,
) -> dict:
    max_depth = learned_depths + 1
    attempts = [0] * max_depth
    matches = [0] * max_depth
    teacher_matches = [0] * learned_depths
    hidden_squared_error = [0.0] * learned_depths
    hidden_squared_target = [0.0] * learned_depths
    teacher_tokens = features["teacher_token_ids"]
    for offset in range(0, len(starts), batch_size):
        batch = mx.array(starts[offset : offset + batch_size], dtype=mx.int32)
        alive = teacher_tokens[batch, 0] == expected_token_ids[batch]
        attempts[0] += batch.size
        matches[0] += int(mx.sum(alive))
        hidden = features["teacher_hidden"][batch, 0]
        for depth in range(learned_depths):
            rows = batch + depth + 1
            prediction = predict_hidden(
                parameters,
                hidden,
                accepted_embeddings[rows].astype(mx.float32),
                depth,
                architecture,
            )
            prediction_indices = mx.argmax(head(prediction), axis=-1)
            prediction_tokens = target_token_ids[prediction_indices]
            target_tokens = expected_token_ids[rows]
            teacher_token = teacher_tokens[batch, depth + 1]
            teacher_hidden = features["teacher_hidden"][batch, depth + 1]
            mx.eval(prediction_tokens, alive, prediction, teacher_hidden)
            active = int(mx.sum(alive))
            attempts[depth + 1] += active
            matches[depth + 1] += int(mx.sum(alive & (prediction_tokens == target_tokens)))
            teacher_matches[depth] += int(mx.sum(alive & (prediction_tokens == teacher_token)))
            active_f32 = alive.astype(mx.float32)[:, None]
            hidden_squared_error[depth] += float(
                mx.sum(active_f32 * mx.square(prediction - teacher_hidden))
            )
            hidden_squared_target[depth] += float(
                mx.sum(active_f32 * mx.square(teacher_hidden))
            )
            alive = alive & (prediction_tokens == target_tokens)
            hidden = prediction
    result = {
        "1": {
            "attempts": attempts[0],
            "matches": matches[0],
            "conditional_acceptance": matches[0] / attempts[0] if attempts[0] else None,
            "source": "official-mtp",
        }
    }
    for depth in range(learned_depths):
        index = depth + 1
        result[str(index + 1)] = {
            "attempts": attempts[index],
            "matches": matches[index],
            "conditional_acceptance": matches[index] / attempts[index]
            if attempts[index]
            else None,
            "teacher_matches": teacher_matches[depth],
            "teacher_agreement": teacher_matches[depth] / attempts[index]
            if attempts[index]
            else None,
            "teacher_hidden_relative_l2": (
                hidden_squared_error[depth] / hidden_squared_target[depth]
            )
            ** 0.5
            if hidden_squared_target[depth]
            else None,
        }
    return result


def token_classifier_loss(
    parameters: dict[str, mx.array],
    starts: mx.array,
    features: dict[str, mx.array],
    accepted_embeddings: mx.array,
    expected_reduced_indices: mx.array,
    selected_logit_weight: float,
    full_logit_weight: float,
    temperature: float,
) -> mx.array:
    logits = predict_token_logits(
        parameters,
        features["teacher_hidden"][starts, 0],
        accepted_embeddings[starts + 1].astype(mx.float32),
    )
    labels = expected_reduced_indices[starts + 1]
    full_loss = mx.logsumexp(logits, axis=-1) - logits[mx.arange(starts.size), labels]
    indices = features["teacher_top_indices"][starts, 1]
    student_selected = mx.take_along_axis(logits, indices, axis=-1)
    teacher_selected = features["teacher_top_logits"][starts, 1]
    teacher_scaled = teacher_selected / temperature
    student_scaled = student_selected / temperature
    teacher_probability = mx.softmax(teacher_scaled, axis=-1)
    teacher_log_probability = teacher_scaled - mx.logsumexp(
        teacher_scaled, axis=-1, keepdims=True
    )
    student_log_probability = student_scaled - mx.logsumexp(
        student_scaled, axis=-1, keepdims=True
    )
    selected_loss = mx.sum(
        teacher_probability * (teacher_log_probability - student_log_probability),
        axis=-1,
    ) * temperature**2
    return mx.mean(full_logit_weight * full_loss + selected_logit_weight * selected_loss)


def token_classifier_metrics(
    parameters: dict[str, mx.array],
    starts: list[int],
    features: dict[str, mx.array],
    accepted_embeddings: mx.array,
    expected_token_ids: mx.array,
    target_token_ids: mx.array,
    batch_size: int,
) -> dict:
    first_matches = 0
    second_attempts = 0
    second_matches = 0
    teacher_matches = 0
    teacher_tokens = features["teacher_token_ids"]
    for offset in range(0, len(starts), batch_size):
        batch = mx.array(starts[offset : offset + batch_size], dtype=mx.int32)
        alive = teacher_tokens[batch, 0] == expected_token_ids[batch]
        prediction_indices = mx.argmax(
            predict_token_logits(
                parameters,
                features["teacher_hidden"][batch, 0],
                accepted_embeddings[batch + 1].astype(mx.float32),
            ),
            axis=-1,
        )
        prediction_tokens = target_token_ids[prediction_indices]
        target_tokens = expected_token_ids[batch + 1]
        teacher_token = teacher_tokens[batch, 1]
        mx.eval(alive, prediction_tokens)
        active = int(mx.sum(alive))
        first_matches += active
        second_attempts += active
        second_matches += int(mx.sum(alive & (prediction_tokens == target_tokens)))
        teacher_matches += int(mx.sum(alive & (prediction_tokens == teacher_token)))
    return {
        "1": {
            "attempts": len(starts),
            "matches": first_matches,
            "conditional_acceptance": first_matches / len(starts) if starts else None,
            "source": "official-mtp",
        },
        "2": {
            "attempts": second_attempts,
            "matches": second_matches,
            "conditional_acceptance": second_matches / second_attempts
            if second_attempts
            else None,
            "teacher_matches": teacher_matches,
            "teacher_agreement": teacher_matches / second_attempts
            if second_attempts
            else None,
            "teacher_hidden_relative_l2": None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--architecture",
        choices=(TANH_ARCHITECTURE, GATED_ARCHITECTURE, TOKEN_ARCHITECTURE),
        default=GATED_ARCHITECTURE,
    )
    parser.add_argument("--optimizer", choices=("adamw", "gefen"), default="adamw")
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--max-depth", type=int, choices=(2, 3), default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-loss-weight", type=float, default=1.0)
    parser.add_argument("--logit-loss-weight", type=float, default=0.1)
    parser.add_argument("--full-logit-loss-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(
            args.rank > 0
            and args.epochs > 0
            and args.batch_size > 0
            and args.eval_batch_size > 0,
            "invalid MTP distillation training size",
        )
        require(
            args.learning_rate > 0
            and args.weight_decay >= 0
            and args.hidden_loss_weight >= 0
            and args.logit_loss_weight >= 0
            and args.full_logit_loss_weight >= 0
            and args.temperature > 0,
            "invalid MTP distillation optimizer or loss settings",
        )
        require(
            args.architecture != TOKEN_ARCHITECTURE or args.max_depth == 2,
            "token-only MTP student requires max depth two",
        )
        arrays, capture_state = load_capture(args.capture_dir)
        capture_hash = sha256_file(args.capture_dir / "state.json")
        features, feature_state = load_features(args.features_dir, capture_hash)
        require(
            feature_state["identity"]["model_dir"] == str(args.model_dir.resolve()),
            "MTP distillation target model mismatch",
        )
        require(
            feature_state["identity"]["mtp_lm_head"]
            == str(args.mtp_lm_head.resolve()),
            "MTP distillation vocabulary head mismatch",
        )
        require(
            features["teacher_hidden"].shape[1] >= args.max_depth,
            "MTP distillation features have insufficient depth",
        )
        prompts = arrays["prompt_indices"].tolist()
        learned_depths = args.max_depth - 1
        raw_training_starts = sequence_starts(prompts, args.max_depth - 1, 0)
        held_starts = sequence_starts(prompts, args.max_depth - 1, 1)
        expected_list = arrays["expected_token_ids"].tolist()
        teacher_list = features["teacher_token_ids"].tolist()
        training_starts = [
            start
            for start in raw_training_starts
            if teacher_list[start][0] == expected_list[start]
        ]
        require(training_starts and held_starts, "MTP distillation split is empty")

        map_path = args.mtp_lm_head / "lm_head.safetensors"
        map_arrays, map_metadata = mx.load(str(map_path), return_metadata=True)
        require(
            map_metadata.get("storage") == "shared-target-bf16",
            "MTP distillation requires the shared target vocabulary head",
        )
        target_token_ids = map_arrays["target_token_ids"]
        require(
            mx.array_equal(
                target_token_ids[features["teacher_top_indices"][:, :, 0]],
                features["teacher_token_ids"],
            ),
            "MTP teacher top-logit index does not match its token ID",
        )
        global_tensors = mx.load(str(args.model_dir / "global.safetensors"))
        head_weight = global_tensors["lm_head.weight"]
        head = ModelOptBF16Linear(head_weight[target_token_ids])
        mx.eval(head.weight)
        del head_weight, global_tensors
        embeddings = PagedBF16Embedding(args.model_dir, cache_rows=0)
        accepted_embeddings = embeddings.rows(arrays["accepted_token_ids"].tolist())
        mx.eval(accepted_embeddings)
        embeddings.close()
        expected_token_ids = arrays["expected_token_ids"]
        expected_reduced_indices = reduced_target_indices(
            target_token_ids.tolist(), expected_list
        )
        expected_reduced_list = expected_reduced_indices.tolist()
        training_starts = [
            start
            for start in training_starts
            if all(
                expected_reduced_list[start + depth] >= 0
                for depth in range(1, args.max_depth)
            )
        ]
        require(
            training_starts,
            "no accepted training targets are in the reduced vocabulary",
        )

        hidden_size = features["teacher_hidden"].shape[-1]
        parameters = initialize_student(
            args.architecture,
            hidden_size,
            args.rank,
            learned_depths,
            args.seed,
            target_token_ids.size,
        )
        mx.eval(*parameters.values())
        optimizer = (
            AdamWControl(parameters, args.learning_rate, args.weight_decay)
            if args.optimizer == "adamw"
            else GefenMLX(args.learning_rate, weight_decay=args.weight_decay)
        )
        official_metrics = official_acceptance_metrics(
            held_starts, expected_list, teacher_list, args.max_depth
        )

        def evaluate(parameter_values):
            if args.architecture == TOKEN_ARCHITECTURE:
                return token_classifier_metrics(
                    parameter_values,
                    held_starts,
                    features,
                    accepted_embeddings,
                    expected_token_ids,
                    target_token_ids,
                    args.eval_batch_size,
                )
            return student_acceptance_metrics(
                parameter_values,
                held_starts,
                features,
                accepted_embeddings,
                expected_token_ids,
                target_token_ids,
                head,
                args.architecture,
                learned_depths,
                args.eval_batch_size,
            )

        baseline_metrics = evaluate(parameters)

        def loss_fn(parameter_values, batch):
            if args.architecture == TOKEN_ARCHITECTURE:
                return token_classifier_loss(
                    parameter_values,
                    batch,
                    features,
                    accepted_embeddings,
                    expected_reduced_indices,
                    args.logit_loss_weight,
                    args.full_logit_loss_weight,
                    args.temperature,
                )
            return distillation_loss(
                parameter_values,
                batch,
                features,
                accepted_embeddings,
                expected_token_ids,
                expected_reduced_indices,
                head.weight,
                args.architecture,
                learned_depths,
                args.hidden_loss_weight,
                args.logit_loss_weight,
                args.full_logit_loss_weight,
                args.temperature,
            )

        value_and_grad = mx.value_and_grad(loss_fn)
        rng = random.Random(args.seed)
        history = []
        best = None
        best_parameters = None
        started = time.perf_counter()
        steps = 0
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(training_starts)
            losses = []
            epoch_started = time.perf_counter()
            for offset in range(0, len(training_starts), args.batch_size):
                batch = mx.array(
                    training_starts[offset : offset + args.batch_size], dtype=mx.int32
                )
                loss, gradients = value_and_grad(parameters, batch)
                parameters = optimizer.update(parameters, gradients)
                mx.eval(loss, *parameters.values())
                losses.append(float(loss))
                steps += 1
            metrics = evaluate(parameters)
            row = {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "seconds": time.perf_counter() - epoch_started,
                "heldout": metrics,
            }
            history.append(row)
            if best is None or acceptance_score(metrics) > acceptance_score(best["heldout"]):
                best = row
                best_parameters = {
                    name: np.asarray(value).copy() for name, value in parameters.items()
                }
            print("mtp-distill-epoch " + json.dumps(row, separators=(",", ":")), flush=True)

        require(best is not None and best_parameters is not None, "distillation produced no checkpoint")
        parameters = {name: mx.array(value) for name, value in best_parameters.items()}
        mx.eval(*parameters.values())
        final_metrics = evaluate(parameters)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = args.output_dir / "predictor.safetensors"
        temporary = artifact.with_name("predictor.part.safetensors")
        runtime_parameters = parameters
        runtime_layout = "training-native"
        if args.architecture == TOKEN_ARCHITECTURE:
            runtime_parameters = {
                "input_projection_t": parameters["input_projection"].T,
                "input_bias": parameters["input_bias"],
                "vocab_output_t": parameters["vocab_output"].T,
                "vocab_bias": parameters["vocab_bias"],
            }
            runtime_layout = "contiguous-bf16-matvec-v1"
        mx.save_safetensors(
            str(temporary),
            {
                name: value.astype(mx.bfloat16)
                for name, value in runtime_parameters.items()
            },
            metadata={
                "format": PREDICTOR_FORMAT,
                "optimizer": args.optimizer,
                "architecture": args.architecture,
                "runtime_layout": runtime_layout,
            },
        )
        temporary.replace(artifact)
        parameter_count = sum(value.size for value in parameters.values())
        report = {
            "format": PREDICTOR_FORMAT,
            "experiment_format": FORMAT,
            "status": "diagnostic",
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(
                args.model_dir / "nemotron_mlx_pack_report.json"
            ),
            "capture_dir": str(args.capture_dir.resolve()),
            "capture_state_sha256": sha256_file(args.capture_dir / "state.json"),
            "features_dir": str(args.features_dir.resolve()),
            "features_state_sha256": sha256_file(args.features_dir / "state.json"),
            "mtp_lm_head": str(args.mtp_lm_head.resolve()),
            "mtp_lm_head_report_sha256": sha256_file(
                args.mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
            ),
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "optimizer": args.optimizer,
            "training_mode": "official-first-distilled",
            "architecture": args.architecture,
            "runtime_layout": runtime_layout,
            "gefen_upstream_revision": UPSTREAM_REVISION
            if args.optimizer == "gefen"
            else None,
            "rank": args.rank,
            "max_depth": args.max_depth,
            "hidden_loss_weight": args.hidden_loss_weight,
            "selected_logit_loss_weight": args.logit_loss_weight,
            "full_logit_loss_weight": args.full_logit_loss_weight,
            "hard_label_source": "authoritative-target-token",
            "temperature": args.temperature,
            "parameter_count": parameter_count,
            "inference_payload_bytes": artifact.stat().st_size,
            "optimizer_state_bytes": optimizer.state_bytes(),
            "training_starts": len(training_starts),
            "heldout_starts": len(held_starts),
            "official_heldout": official_metrics,
            "baseline": baseline_metrics,
            "final": final_metrics,
            "best_epoch": best["epoch"],
            "best_acceptance_score": acceptance_score(best["heldout"]),
            "history": history,
            "steps": steps,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "decision": "diagnostic-pending-matched-resident-throughput-gate",
        }
        atomic_json(args.output_dir / "report.json", report)
        print("mtp-distill-done " + json.dumps(report, separators=(",", ":")), flush=True)
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron MTP distillation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

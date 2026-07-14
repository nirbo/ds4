#!/usr/bin/env python3
"""Train a compact recursive MTP predictor on exact resident target traces."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_gefen import GefenMLX, UPSTREAM_REVISION
from nemotron_paged_embeddings import PagedBF16Embedding
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-learned-predictor-v1"
TRACE_FORMAT = "nemotron-mtp-teacher-capture-v2"


def validate_runtime_binding(report: dict, model_dir: Path, mtp_lm_head: Path) -> None:
    """Reject a learned artifact trained against another target or head map."""
    model_report = model_dir / "nemotron_mlx_pack_report.json"
    head_report = mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
    require(model_report.is_file(), "learned MTP target report is missing")
    require(head_report.is_file(), "learned MTP vocabulary-head report is missing")
    require(
        Path(report.get("model_dir", "")).resolve() == model_dir.resolve()
        and report.get("model_report_sha256") == sha256_file(model_report),
        "learned MTP predictor target identity mismatch",
    )
    require(
        Path(report.get("mtp_lm_head", "")).resolve() == mtp_lm_head.resolve()
        and report.get("mtp_lm_head_report_sha256") == sha256_file(head_report),
        "learned MTP predictor vocabulary-head identity mismatch",
    )


def validated_predictor_report(
    artifact_dir: Path, model_dir: Path, mtp_lm_head: Path
) -> dict:
    report = load_json(artifact_dir / "report.json")
    require(
        report.get("format") == FORMAT and report.get("status") == "diagnostic",
        "learned MTP predictor report is invalid",
    )
    validate_runtime_binding(report, model_dir, mtp_lm_head)
    artifact = artifact_dir / report.get("artifact", "")
    require(artifact.is_file(), "learned MTP predictor artifact is missing")
    require(
        artifact.stat().st_size == report.get("inference_payload_bytes")
        and sha256_file(artifact) == report.get("artifact_sha256"),
        "learned MTP predictor artifact identity mismatch",
    )
    return report


class LearnedMTPPredictor:
    """Inference-only learned draft over the resident embedding/head boundary."""

    def __init__(
        self,
        artifact_dir: Path,
        base_mtp,
        *,
        model_dir: Path,
        mtp_lm_head: Path,
    ):
        report = validated_predictor_report(artifact_dir, model_dir, mtp_lm_head)
        artifact = artifact_dir / report.get("artifact", "")
        parameters, metadata = mx.load(str(artifact), return_metadata=True)
        require(metadata.get("format") == FORMAT, "learned MTP predictor format mismatch")
        required = {"hidden_projection", "token_projection", "depth_output", "depth_bias"}
        require(set(parameters) == required, "learned MTP predictor tensor set mismatch")
        self.parameters = parameters
        self.training_mode = report.get("training_mode", "direct")
        require(
            self.training_mode in ("direct", "official-first-continuation"),
            "learned MTP predictor training mode is invalid",
        )
        self.learned_depths = parameters["depth_output"].shape[0]
        self.max_depth = self.learned_depths + (self.training_mode == "official-first-continuation")
        self.base_mtp = base_mtp
        self.embeddings = base_mtp.embeddings
        self.lm_head = base_mtp.lm_head
        self.draft_token_ids = base_mtp.draft_token_ids
        self.report = report
        mx.eval(*parameters.values())

    def token_ids(self, draft_indices: mx.array) -> mx.array:
        return draft_indices if self.draft_token_ids is None else self.draft_token_ids[draft_indices]

    def argmax_token(self, logits: mx.array) -> int:
        return int(self.token_ids(mx.argmax(logits)))

    def draft_step(
        self, target_hidden: mx.array, accepted_token_id: int, *, depth: int = 0
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        require(0 <= depth < self.max_depth, "learned MTP predictor depth is out of range")
        if self.training_mode == "official-first-continuation" and depth == 0:
            return self.base_mtp.draft_step(target_hidden, accepted_token_id)
        learned_depth = depth - 1 if self.training_mode == "official-first-continuation" else depth
        embedding = self.embeddings[accepted_token_id].astype(mx.float32).reshape(1, -1)
        prediction = predict_hidden(
            self.parameters,
            target_hidden.astype(mx.float32).reshape(1, -1),
            embedding,
            learned_depth,
        )
        logits = self.lm_head(prediction.reshape(1, 1, -1)).reshape(-1)
        empty_indices = mx.zeros((0,), dtype=mx.int32)
        empty_scores = mx.zeros((0,), dtype=mx.float32)
        return logits, prediction.reshape(-1), empty_indices, empty_scores


def predictor_parameter_count(hidden_size: int, rank: int, depths: int) -> int:
    return 2 * hidden_size * rank + depths * rank * hidden_size + depths * rank


def initialize_parameters(hidden_size: int, rank: int, depths: int, seed: int) -> dict[str, mx.array]:
    mx.random.seed(seed)
    scale = 1 / math.sqrt(hidden_size)
    return {
        "hidden_projection": mx.random.normal((hidden_size, rank)) * scale,
        "token_projection": mx.random.normal((hidden_size, rank)) * scale,
        "depth_output": mx.random.normal((depths, rank, hidden_size)) * 1e-4,
        "depth_bias": mx.zeros((depths, rank)),
    }


def predict_hidden(
    parameters: dict[str, mx.array],
    hidden: mx.array,
    token_embedding: mx.array,
    depth: int,
) -> mx.array:
    latent = mx.tanh(
        hidden @ parameters["hidden_projection"]
        + token_embedding @ parameters["token_projection"]
        + parameters["depth_bias"][depth]
    )
    return hidden + latent @ parameters["depth_output"][depth]


def load_capture(capture_dir: Path) -> tuple[dict[str, mx.array], dict]:
    from nemotron_mlx_mtp_teacher_capture import validate_completed

    state = load_json(capture_dir / "state.json")
    require(state.get("format") == TRACE_FORMAT, "unsupported teacher capture format")
    require(state.get("status") == "complete", "teacher capture is incomplete")
    validate_completed(capture_dir, state)
    shards = []
    for key, entry in sorted(state["completed"].items(), key=lambda item: int(item[0])):
        arrays, _ = mx.load(str(capture_dir / entry["file"]), return_metadata=True)
        shards.append(arrays)
    required = {
        "target_hidden",
        "accepted_token_ids",
        "expected_token_ids",
        "prompt_indices",
        "scored",
    }
    arrays = {name: mx.concatenate([shard[name] for shard in shards]) for name in required}
    mx.eval(*arrays.values())
    return arrays, state


def load_recursive_features(
    features_dir: Path, capture_state: dict, capture_state_sha256: str
) -> tuple[dict[str, mx.array], dict]:
    state = load_json(features_dir / "state.json")
    require(
        state.get("format") == "nemotron-mtp-recursive-features-v1"
        and state.get("status") == "complete",
        "recursive MTP features are incomplete",
    )
    require(
        state.get("identity", {}).get("capture_state_sha256")
        == capture_state_sha256,
        "recursive MTP feature capture identity mismatch",
    )
    hidden = []
    tokens = []
    for key, entry in sorted(state["completed"].items(), key=lambda item: int(item[0])):
        path = features_dir / entry["file"]
        require(path.is_file() and sha256_file(path) == entry["sha256"], "recursive feature hash mismatch")
        arrays = mx.load(str(path))
        hidden.append(arrays["mtp_hidden"])
        tokens.append(arrays["mtp_token_ids"])
    result = {"mtp_hidden": mx.concatenate(hidden), "mtp_token_ids": mx.concatenate(tokens)}
    require(result["mtp_hidden"].shape[0] == capture_state["rows"], "recursive feature rows mismatch")
    mx.eval(*result.values())
    return result, state


def sequence_starts(prompt_indices: list[int], max_depth: int, parity: int) -> list[int]:
    starts = []
    for index in range(len(prompt_indices) - max_depth):
        prompt = prompt_indices[index]
        if prompt % 2 != parity:
            continue
        if all(prompt_indices[index + offset] == prompt for offset in range(max_depth + 1)):
            starts.append(index)
    return starts


class AdamWControl:
    def __init__(self, parameters: dict[str, mx.array], learning_rate: float, weight_decay: float):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_count = 0
        self.mean = {name: mx.zeros_like(value) for name, value in parameters.items()}
        self.variance = {name: mx.zeros_like(value) for name, value in parameters.items()}

    def state_bytes(self) -> int:
        return sum(value.nbytes for value in (*self.mean.values(), *self.variance.values()))

    def update(self, parameters: dict[str, mx.array], gradients: dict[str, mx.array]) -> dict[str, mx.array]:
        self.step_count += 1
        result = {}
        for name, parameter in parameters.items():
            gradient = gradients[name]
            self.mean[name] = 0.9 * self.mean[name] + 0.1 * gradient
            self.variance[name] = 0.999 * self.variance[name] + 0.001 * mx.square(gradient)
            corrected_mean = self.mean[name] / (1 - 0.9**self.step_count)
            corrected_variance = self.variance[name] / (1 - 0.999**self.step_count)
            result[name] = parameter * (1 - self.learning_rate * self.weight_decay) - (
                self.learning_rate * corrected_mean / (mx.sqrt(corrected_variance) + 1e-8)
            )
        mx.eval(*result.values(), *self.mean.values(), *self.variance.values())
        return result


def recursive_loss(
    parameters: dict[str, mx.array],
    starts: mx.array,
    target_hidden: mx.array,
    token_embeddings: mx.array,
    expected_indices: mx.array,
    head_weight: mx.array,
    max_depth: int,
    hidden_weight: float,
) -> mx.array:
    hidden = target_hidden[starts].astype(mx.float32)
    total = 0.0
    for depth in range(max_depth):
        rows = starts + depth
        prediction = predict_hidden(
            parameters,
            hidden,
            token_embeddings[rows].astype(mx.float32),
            depth,
        )
        logits = prediction @ head_weight.T.astype(mx.float32)
        labels = expected_indices[rows]
        token_loss = mx.mean(mx.logsumexp(logits, axis=-1) - logits[mx.arange(len(starts)), labels])
        target = target_hidden[rows + 1].astype(mx.float32)
        hidden_loss = mx.mean(mx.square(prediction - target)) / (
            mx.mean(mx.square(target)) + 1e-8
        )
        total = total + token_loss + hidden_weight * hidden_loss
        hidden = prediction
    return total / max_depth


def continuation_loss(
    parameters: dict[str, mx.array],
    starts: mx.array,
    mtp_hidden: mx.array,
    target_hidden: mx.array,
    token_embeddings: mx.array,
    expected_indices: mx.array,
    head_weight: mx.array,
    learned_depths: int,
    hidden_weight: float,
) -> mx.array:
    hidden = mtp_hidden[starts].astype(mx.float32)
    total = 0.0
    for depth in range(learned_depths):
        rows = starts + depth + 1
        prediction = predict_hidden(
            parameters, hidden, token_embeddings[rows].astype(mx.float32), depth
        )
        logits = prediction @ head_weight.T.astype(mx.float32)
        labels = expected_indices[rows]
        token_loss = mx.mean(mx.logsumexp(logits, axis=-1) - logits[mx.arange(len(starts)), labels])
        target = target_hidden[rows + 1].astype(mx.float32)
        hidden_loss = mx.mean(mx.square(prediction - target)) / (
            mx.mean(mx.square(target)) + 1e-8
        )
        total = total + token_loss + hidden_weight * hidden_loss
        hidden = prediction
    return total / learned_depths


def acceptance_metrics(
    parameters: dict[str, mx.array],
    starts: list[int],
    arrays: dict[str, mx.array],
    token_embeddings: mx.array,
    token_ids: mx.array,
    head_weight: mx.array,
    max_depth: int,
) -> dict:
    attempts = [0] * max_depth
    matches = [0] * max_depth
    for start in starts:
        hidden = arrays["target_hidden"][start].astype(mx.float32)[None]
        for depth in range(max_depth):
            row = start + depth
            hidden = predict_hidden(
                parameters, hidden, token_embeddings[row][None].astype(mx.float32), depth
            )
            logits = hidden @ head_weight.T.astype(mx.float32)
            prediction = int(token_ids[int(mx.argmax(logits))])
            expected = int(arrays["expected_token_ids"][row])
            attempts[depth] += 1
            if prediction != expected:
                break
            matches[depth] += 1
    return {
        str(depth + 1): {
            "attempts": attempts[depth],
            "matches": matches[depth],
            "conditional_acceptance": matches[depth] / attempts[depth] if attempts[depth] else None,
        }
        for depth in range(max_depth)
    }


def continuation_acceptance_metrics(
    parameters: dict[str, mx.array],
    starts: list[int],
    arrays: dict[str, mx.array],
    features: dict[str, mx.array],
    token_embeddings: mx.array,
    token_ids: mx.array,
    head_weight: mx.array,
    learned_depths: int,
) -> dict:
    attempts = [0] * (learned_depths + 1)
    matches = [0] * (learned_depths + 1)
    expected = arrays["expected_token_ids"]
    for start in starts:
        attempts[0] += 1
        if int(features["mtp_token_ids"][start]) != int(expected[start]):
            continue
        matches[0] += 1
        hidden = features["mtp_hidden"][start].astype(mx.float32)[None]
        for depth in range(learned_depths):
            row = start + depth + 1
            attempts[depth + 1] += 1
            hidden = predict_hidden(
                parameters, hidden, token_embeddings[row][None].astype(mx.float32), depth
            )
            prediction = int(token_ids[int(mx.argmax(hidden @ head_weight.T.astype(mx.float32)))])
            if prediction != int(expected[row]):
                break
            matches[depth + 1] += 1
    return {
        str(depth + 1): {
            "attempts": attempts[depth],
            "matches": matches[depth],
            "conditional_acceptance": matches[depth] / attempts[depth] if attempts[depth] else None,
        }
        for depth in range(learned_depths + 1)
    }


def acceptance_score(metrics: dict) -> int:
    return sum(value["matches"] for value in metrics.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--recursive-features", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--optimizer", choices=("adamw", "gefen"), default="adamw")
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--max-depth", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.rank > 0 and args.epochs > 0 and args.batch_size > 0, "invalid training size")
        require(args.learning_rate > 0 and args.weight_decay >= 0, "invalid optimizer settings")
        arrays, capture_state = load_capture(args.capture_dir)
        prompts = arrays["prompt_indices"].tolist()
        features = None
        training_mode = "direct"
        sequence_depth = args.max_depth
        if args.recursive_features is not None:
            features, _ = load_recursive_features(
                args.recursive_features,
                capture_state,
                sha256_file(args.capture_dir / "state.json"),
            )
            training_mode = "official-first-continuation"
            sequence_depth += 1
        raw_training_starts = sequence_starts(prompts, sequence_depth, 0)
        raw_held_starts = sequence_starts(prompts, sequence_depth, 1)

        map_path = args.mtp_lm_head / "lm_head.safetensors"
        map_arrays, map_metadata = mx.load(str(map_path), return_metadata=True)
        require(map_metadata.get("storage") == "shared-target-bf16", "predictor requires shared target head")
        target_token_ids = map_arrays["target_token_ids"]
        token_to_index = {token: index for index, token in enumerate(target_token_ids.tolist())}
        expected = arrays["expected_token_ids"].tolist()
        mapped_offset = 1 if features is not None else 0
        mapped_window = lambda start: all(
            expected[start + mapped_offset + depth] in token_to_index
            for depth in range(args.max_depth)
        )
        mapped_training_starts = [start for start in raw_training_starts if mapped_window(start)]
        mapped_held_starts = [start for start in raw_held_starts if mapped_window(start)]
        training_starts = mapped_training_starts
        held_starts = mapped_held_starts
        official_first_rejected_training_starts = 0
        if features is not None:
            training_starts = [
                start
                for start in training_starts
                if int(features["mtp_token_ids"][start]) == expected[start]
            ]
            official_first_rejected_training_starts = (
                len(mapped_training_starts) - len(training_starts)
            )
        require(training_starts and held_starts, "reduced-head prompt-disjoint split is empty")
        expected_indices = mx.array(
            [token_to_index.get(token, 0) for token in expected], dtype=mx.int32
        )

        global_tensors = mx.load(str(args.model_dir / "global.safetensors"))
        head_weight = global_tensors["lm_head.weight"][target_token_ids]
        mx.eval(head_weight)
        del global_tensors
        embeddings = PagedBF16Embedding(args.model_dir, cache_rows=0)
        token_embeddings = embeddings.rows(arrays["accepted_token_ids"].tolist())
        mx.eval(token_embeddings)
        embeddings.close()
        mx.clear_cache()

        hidden_size = arrays["target_hidden"].shape[1]
        parameters = initialize_parameters(hidden_size, args.rank, args.max_depth, args.seed)
        parameter_count = predictor_parameter_count(hidden_size, args.rank, args.max_depth)
        mx.eval(*parameters.values())
        baseline = acceptance_metrics(
            parameters,
            held_starts,
            arrays,
            token_embeddings,
            target_token_ids,
            head_weight,
            args.max_depth,
        ) if features is None else continuation_acceptance_metrics(
            parameters,
            held_starts,
            arrays,
            features,
            token_embeddings,
            target_token_ids,
            head_weight,
            args.max_depth,
        )
        optimizer = (
            AdamWControl(parameters, args.learning_rate, args.weight_decay)
            if args.optimizer == "adamw"
            else GefenMLX(args.learning_rate, weight_decay=args.weight_decay)
        )

        def loss_fn(parameter_values, batch):
            if features is not None:
                return continuation_loss(
                    parameter_values,
                    batch,
                    features["mtp_hidden"],
                    arrays["target_hidden"],
                    token_embeddings,
                    expected_indices,
                    head_weight,
                    args.max_depth,
                    args.hidden_loss_weight,
                )
            return recursive_loss(
                parameter_values,
                batch,
                arrays["target_hidden"],
                token_embeddings,
                expected_indices,
                head_weight,
                args.max_depth,
                args.hidden_loss_weight,
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
                batch = mx.array(training_starts[offset : offset + args.batch_size], dtype=mx.int32)
                loss, gradients = value_and_grad(parameters, batch)
                parameters = optimizer.update(parameters, gradients)
                mx.eval(loss, *parameters.values())
                losses.append(float(loss))
                steps += 1
            metrics = acceptance_metrics(
                parameters,
                held_starts,
                arrays,
                token_embeddings,
                target_token_ids,
                head_weight,
                args.max_depth,
            ) if features is None else continuation_acceptance_metrics(
                parameters,
                held_starts,
                arrays,
                features,
                token_embeddings,
                target_token_ids,
                head_weight,
                args.max_depth,
            )
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
            print("mtp-predictor-epoch " + json.dumps(row, separators=(",", ":")), flush=True)

        require(best is not None and best_parameters is not None, "training produced no checkpoint")
        final_epoch_metrics = history[-1]["heldout"]
        parameters = {name: mx.array(value) for name, value in best_parameters.items()}
        mx.eval(*parameters.values())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = args.output_dir / "predictor.safetensors"
        temporary = artifact.with_name("predictor.part.safetensors")
        mx.save_safetensors(
            str(temporary),
            {name: value.astype(mx.bfloat16) for name, value in parameters.items()},
            metadata={"format": FORMAT, "optimizer": args.optimizer},
        )
        temporary.replace(artifact)
        final_metrics = acceptance_metrics(
            parameters,
            held_starts,
            arrays,
            token_embeddings,
            target_token_ids,
            head_weight,
            args.max_depth,
        ) if features is None else continuation_acceptance_metrics(
            parameters,
            held_starts,
            arrays,
            features,
            token_embeddings,
            target_token_ids,
            head_weight,
            args.max_depth,
        )
        report = {
            "format": FORMAT,
            "status": "diagnostic",
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(args.model_dir / "nemotron_mlx_pack_report.json"),
            "capture_dir": str(args.capture_dir.resolve()),
            "capture_state_sha256": sha256_file(args.capture_dir / "state.json"),
            "mtp_lm_head": str(args.mtp_lm_head.resolve()),
            "mtp_lm_head_report_sha256": sha256_file(
                args.mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
            ),
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "optimizer": args.optimizer,
            "training_mode": training_mode,
            "recursive_features": (
                str(args.recursive_features.resolve()) if args.recursive_features is not None else None
            ),
            "recursive_features_state_sha256": (
                sha256_file(args.recursive_features / "state.json")
                if args.recursive_features is not None
                else None
            ),
            "gefen_upstream_revision": UPSTREAM_REVISION if args.optimizer == "gefen" else None,
            "gefen_codebook_solver": "weighted-lloyd-4096-bin" if args.optimizer == "gefen" else None,
            "rank": args.rank,
            "max_depth": args.max_depth,
            "parameter_count": parameter_count,
            "inference_payload_bytes": artifact.stat().st_size,
            "optimizer_state_bytes": optimizer.state_bytes(),
            "training_starts": len(training_starts),
            "heldout_starts": len(held_starts),
            "excluded_unmapped_training_starts": (
                len(raw_training_starts) - len(mapped_training_starts)
            ),
            "excluded_unmapped_heldout_starts": (
                len(raw_held_starts) - len(mapped_held_starts)
            ),
            "official_first_rejected_training_starts": (
                official_first_rejected_training_starts
            ),
            "baseline": baseline,
            "final": final_metrics,
            "final_epoch": final_epoch_metrics,
            "best_epoch": best["epoch"],
            "best_acceptance_score": acceptance_score(best["heldout"]),
            "history": history,
            "steps": steps,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "decision": "diagnostic-pending-10k-trace-and-resident-speculative-gate",
        }
        atomic_json(args.output_dir / "report.json", report)
        print("mtp-predictor-done " + json.dumps(report, separators=(",", ":")), flush=True)
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron MTP predictor error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

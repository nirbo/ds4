#!/usr/bin/env python3
"""Train a tiny recursive-MTP hidden adapter on prompt-disjoint target traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import NemotronMTPSidecar, load_indexed_tensors
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-hidden-adapter-v1"
TRACE_FORMAT = "nemotron-mtp-target-trace-v1"


class NativeTrainingHead:
    """Gradient-capable exact-row fallback for the custom BF16 draft head."""

    def __init__(self, weight: mx.array, target_token_ids: mx.array):
        require(weight.dtype == mx.bfloat16 and weight.ndim == 2, "invalid training head weight")
        self.weight = weight[target_token_ids]
        self.target_token_ids = target_token_ids

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(mx.float32) @ self.weight.T.astype(mx.float32)


def apply_adapter(hidden: mx.array, up: mx.array, down: mx.array) -> mx.array:
    require(
        hidden.shape[-1] == up.shape[0]
        and up.ndim == down.ndim == 2
        and up.shape[1] == down.shape[0]
        and down.shape[1] == hidden.shape[-1],
        "MTP hidden adapter shape mismatch",
    )
    return hidden + (hidden @ down.T) @ up.T


def better_metrics(candidate: dict, current: dict | None) -> bool:
    """Prefer held-out depth two, then depth three, then the earlier epoch."""

    if current is None:
        return True
    return (
        candidate["depths"]["2"]["matches"],
        candidate["depths"]["3"]["matches"],
        -candidate["epoch"],
    ) > (
        current["depths"]["2"]["matches"],
        current["depths"]["3"]["matches"],
        -current["epoch"],
    )


def recursive_metrics(
    model: NemotronMTPSidecar,
    arrays: dict[str, mx.array],
    prompt_indices: list[int],
    scored: list[bool],
    up: mx.array,
    down: mx.array,
    epoch: int,
    held_parity: int = 1,
    max_depth: int = 3,
) -> dict:
    attempts = Counter()
    matches = Counter()
    rows = len(prompt_indices)
    for row in range(rows):
        if not scored[row] or prompt_indices[row] % 2 != held_parity:
            continue
        prompt = prompt_indices[row]
        hidden = arrays["target_hidden"][row]
        token = int(arrays["accepted_token_ids"][row])
        for depth in range(1, max_depth + 1):
            expected_row = row + depth - 1
            if expected_row >= rows or prompt_indices[expected_row] != prompt:
                break
            logits, next_hidden, _, _ = model.draft_step(hidden, token)
            mx.eval(logits, next_hidden)
            prediction = model.argmax_token(logits)
            expected = int(arrays["expected_token_ids"][expected_row])
            attempts[depth] += 1
            if prediction != expected:
                break
            matches[depth] += 1
            hidden = apply_adapter(next_hidden, up, down)
            token = prediction
    return {
        "epoch": epoch,
        "depths": {
            str(depth): {
                "attempts": attempts[depth],
                "matches": matches[depth],
                "conditional_acceptance": (
                    matches[depth] / attempts[depth] if attempts[depth] else None
                ),
            }
            for depth in range(1, max_depth + 1)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--init-seed", type=int, default=77)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.rank > 0 and args.epochs > 0, "rank and epochs must be positive")
        require(args.learning_rate > 0 and args.weight_decay >= 0, "invalid optimizer settings")
        arrays, trace_metadata = mx.load(str(args.trace), return_metadata=True)
        require(
            trace_metadata.get("format") == TRACE_FORMAT
            and trace_metadata.get("model_dir") == str(args.source_dir.resolve()),
            "MTP trace does not belong to the target candidate",
        )
        required = {
            "target_hidden",
            "accepted_token_ids",
            "expected_token_ids",
            "prompt_indices",
            "scored",
        }
        require(required <= set(arrays), "MTP trace is incomplete")
        rows = arrays["target_hidden"].shape[0]
        require(all(arrays[name].shape[0] == rows for name in required), "MTP trace row mismatch")

        globals_ = load_indexed_tensors(
            args.source_dir,
            {"backbone.embeddings.weight", "lm_head.weight"},
        )
        model = NemotronMTPSidecar(
            args.sidecar,
            globals_["backbone.embeddings.weight"],
            ModelOptBF16Linear(globals_["lm_head.weight"]),
            alternate_lm_head=args.mtp_lm_head,
        )
        production_head = model.lm_head
        require(
            getattr(production_head, "storage", None) == "shared-target-bf16"
            and getattr(production_head, "target_token_ids", None) is not None,
            "hidden-adapter training requires the shared-target reduced BF16 head",
        )
        target_token_ids = production_head.target_token_ids
        token_to_index = {
            token_id: index for index, token_id in enumerate(target_token_ids.tolist())
        }
        prompt_indices = arrays["prompt_indices"].tolist()
        scored = arrays["scored"].tolist()

        pairs = []
        for row in range(rows - 1):
            if not scored[row] or prompt_indices[row + 1] != prompt_indices[row]:
                continue
            logits, next_hidden, _, _ = model.draft_step(
                arrays["target_hidden"][row],
                int(arrays["accepted_token_ids"][row]),
            )
            mx.eval(logits, next_hidden)
            expected = int(arrays["expected_token_ids"][row])
            next_expected = int(arrays["expected_token_ids"][row + 1])
            if model.argmax_token(logits) == expected and next_expected in token_to_index:
                pairs.append(
                    (
                        prompt_indices[row],
                        next_hidden,
                        expected,
                        token_to_index[next_expected],
                    )
                )
        training_pairs = [pair for pair in pairs if pair[0] % 2 == 0]
        held_pairs = [pair for pair in pairs if pair[0] % 2 == 1]
        require(training_pairs and held_pairs, "prompt-disjoint adapter split is empty")

        training_head = NativeTrainingHead(production_head.linear.weight, target_token_ids)
        model.lm_head = training_head
        model.draft_token_ids = target_token_ids
        original_switch = model.moe.switch

        def gradient_safe_switch(self, x, indices, projection):
            return original_switch(x, mx.stop_gradient(indices), projection)

        model.moe.switch = types.MethodType(gradient_safe_switch, model.moe)
        hidden_size = arrays["target_hidden"].shape[1]
        mx.random.seed(args.init_seed)
        down = mx.random.normal((args.rank, hidden_size)) * 0.005
        up = mx.zeros((hidden_size, args.rank), dtype=mx.float32)
        moments = {
            "up_mean": mx.zeros_like(up),
            "up_variance": mx.zeros_like(up),
            "down_mean": mx.zeros_like(down),
            "down_variance": mx.zeros_like(down),
        }

        def loss_fn(up_value, down_value, hidden, token, target):
            logits, _, _, _ = model.draft_step(
                apply_adapter(hidden, up_value, down_value),
                token,
            )
            regularizer = args.weight_decay * (
                mx.mean(mx.square(up_value)) + mx.mean(mx.square(down_value))
            )
            return mx.logsumexp(logits) - logits[target] + regularizer

        value_and_grad = mx.value_and_grad(loss_fn, argnums=(0, 1))
        rng = random.Random(args.seed)
        step = 0
        history = []
        best_metrics = None
        best_up = np.asarray(up).copy()
        best_down = np.asarray(down).copy()

        def evaluate(epoch: int) -> dict:
            model.lm_head = production_head
            model.draft_token_ids = target_token_ids
            metrics = recursive_metrics(
                model,
                arrays,
                prompt_indices,
                scored,
                up,
                down,
                epoch,
            )
            model.lm_head = training_head
            model.draft_token_ids = target_token_ids
            return metrics

        baseline = evaluate(0)
        print("mtp-hidden-adapter-eval " + json.dumps(baseline, separators=(",", ":")), flush=True)
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(training_pairs)
            losses = []
            for _, hidden, token, target in training_pairs:
                value, (up_gradient, down_gradient) = value_and_grad(
                    up,
                    down,
                    hidden,
                    token,
                    target,
                )
                step += 1
                beta1 = 0.9
                beta2 = 0.999
                moments["up_mean"] = beta1 * moments["up_mean"] + (1 - beta1) * up_gradient
                moments["up_variance"] = (
                    beta2 * moments["up_variance"] + (1 - beta2) * mx.square(up_gradient)
                )
                moments["down_mean"] = (
                    beta1 * moments["down_mean"] + (1 - beta1) * down_gradient
                )
                moments["down_variance"] = (
                    beta2 * moments["down_variance"] + (1 - beta2) * mx.square(down_gradient)
                )
                up = up - args.learning_rate * (
                    moments["up_mean"] / (1 - beta1**step)
                ) / (mx.sqrt(moments["up_variance"] / (1 - beta2**step)) + 1e-8)
                down = down - args.learning_rate * (
                    moments["down_mean"] / (1 - beta1**step)
                ) / (mx.sqrt(moments["down_variance"] / (1 - beta2**step)) + 1e-8)
                mx.eval(value, up, down, *moments.values())
                losses.append(float(value))
            metrics = evaluate(epoch)
            metrics["training_loss"] = sum(losses) / len(losses)
            history.append(metrics)
            if better_metrics(metrics, best_metrics):
                best_metrics = metrics
                best_up = np.asarray(up).copy()
                best_down = np.asarray(down).copy()
            print("mtp-hidden-adapter-eval " + json.dumps(metrics, separators=(",", ":")), flush=True)

        require(best_metrics is not None, "adapter training produced no checkpoint")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = args.output_dir / "adapter.safetensors"
        temporary = artifact.with_name("adapter.part.safetensors")
        mx.save_safetensors(
            str(temporary),
            {
                "up": mx.array(best_up),
                "down": mx.array(best_down),
            },
            metadata={"format": FORMAT},
        )
        temporary.replace(artifact)
        sidecar_report = args.sidecar / "nemotron_mtp_pack_report.json"
        head_report = args.mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
        report = {
            "format": FORMAT,
            "status": "diagnostic",
            "source_revision": model.config["nemotron_mtp_runtime"]["source_revision"],
            "source_dir": str(args.source_dir.resolve()),
            "source_report_sha256": sha256_file(
                args.source_dir / "nemotron_mlx_pack_report.json"
            ),
            "trace": str(args.trace.resolve()),
            "trace_sha256": sha256_file(args.trace),
            "sidecar": str(args.sidecar.resolve()),
            "sidecar_report_sha256": sha256_file(sidecar_report),
            "mtp_lm_head": str(args.mtp_lm_head.resolve()),
            "mtp_lm_head_report_sha256": sha256_file(head_report),
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "rank": args.rank,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "init_seed": args.init_seed,
            "split": "even-prompt-training-odd-prompt-heldout",
            "training_pairs": len(training_pairs),
            "held_pairs": len(held_pairs),
            "baseline": baseline,
            "best": best_metrics,
            "history": history,
            "artifact": artifact.name,
            "artifact_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256_file(artifact),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gib": mx.get_peak_memory() / 2**30,
            "decision": "diagnostic-pending-independent-trace-and-resident-gate",
        }
        report_path = args.output_dir / "report.json"
        atomic_json(report_path, report)
        print(
            f"mtp-hidden-adapter-done path={report_path} sha256={sha256_file(report_path)}",
            flush=True,
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron MTP hidden adapter error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

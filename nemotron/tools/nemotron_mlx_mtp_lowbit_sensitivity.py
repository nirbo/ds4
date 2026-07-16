#!/usr/bin/env python3
"""Rank higher-precision MTP expert overlays by causal reduced-logit ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import NemotronMTPSidecar, load_indexed_tensors
from nemotron_mlx_mtp_bench import TRACE_FORMAT
from nemotron_mlx_mtp_binary_fit import (
    DOWN_WEIGHT,
    FORMAT as FIT_FORMAT,
    UP_WEIGHT,
    expert_weight,
    load_sidecar,
)
from nemotron_mlx_mtp_lowbit_plan import FORMAT as PLAN_FORMAT
from nemotron_mlx_mtp_quantize import BINARY_MODES, MODES, quantize_tensor
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-mtp-lowbit-sensitivity-v1"


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def quantized_weight(value: mx.array, recipe: str) -> mx.array:
    settings = MODES[recipe]
    weight, scales, biases = quantize_tensor(value, recipe)
    restored = mx.dequantize(
        weight,
        scales,
        biases,
        **settings,
        dtype=mx.float32,
    )
    mx.eval(restored)
    return restored


def masked_objective(
    logits: mx.array,
    teacher_logits: mx.array,
    valid: mx.array,
    expected_positions: mx.array,
    teacher_weight: float,
) -> mx.array:
    require(logits.shape == teacher_logits.shape == valid.shape, "logit objective shape mismatch")
    require(expected_positions.shape == (logits.shape[0],), "expected-token position mismatch")
    masked_logits = mx.where(valid, logits.astype(mx.float32), -1e9)
    masked_teacher = mx.where(valid, teacher_logits.astype(mx.float32), -1e9)
    log_probs = masked_logits - mx.logsumexp(masked_logits, axis=-1, keepdims=True)
    teacher_log_probs = masked_teacher - mx.logsumexp(
        masked_teacher, axis=-1, keepdims=True
    )
    teacher_probs = mx.exp(teacher_log_probs)
    kl = mx.sum(teacher_probs * (teacher_log_probs - log_probs), axis=-1)
    rows = mx.arange(logits.shape[0], dtype=mx.int32)
    cross_entropy = -log_probs[rows, expected_positions]
    return cross_entropy + teacher_weight * kl


def shortlist(
    baseline_logits: mx.array,
    teacher_logits: mx.array,
    expected_token: int,
    top_k: int,
) -> tuple[list[int], list[bool], int]:
    baseline_ids = mx.argpartition(-baseline_logits, kth=top_k - 1)[:top_k].tolist()
    teacher_ids = mx.argpartition(-teacher_logits, kth=top_k - 1)[:top_k].tolist()
    ordered = []
    seen = set()
    for token in [expected_token, *baseline_ids, *teacher_ids]:
        token = int(token)
        if token not in seen:
            ordered.append(token)
            seen.add(token)
    width = top_k * 2 + 1
    valid = [True] * len(ordered)
    ordered.extend([expected_token] * (width - len(ordered)))
    valid.extend([False] * (width - len(valid)))
    return ordered, valid, 0


def capture_rows(
    source_dir: Path,
    base_sidecar: Path,
    teacher_sidecar: Path,
    traces: list[Path],
    top_k: int,
    operation_log: OperationLog,
) -> tuple[dict[str, mx.array], NemotronMTPSidecar, dict, dict[str, mx.array]]:
    globals_ = load_indexed_tensors(
        source_dir,
        {"backbone.embeddings.weight", "lm_head.weight"},
    )
    base_model = NemotronMTPSidecar(
        base_sidecar,
        globals_["backbone.embeddings.weight"],
        ModelOptBF16Linear(globals_["lm_head.weight"]),
    )
    teacher_model = NemotronMTPSidecar(
        teacher_sidecar,
        globals_["backbone.embeddings.weight"],
        ModelOptBF16Linear(globals_["lm_head.weight"]),
    )
    require(
        base_model.retained_experts == teacher_model.retained_experts,
        "sensitivity sidecars use different expert mappings",
    )
    rows: dict[str, list[mx.array | int]] = {
        "fixed": [],
        "routed_sum": [],
        "latent": [],
        "indices": [],
        "scores": [],
        "token_ids": [],
        "valid_tokens": [],
        "teacher_logits": [],
        "baseline_logits": [],
        "expected_positions": [],
    }
    for trace_path in traces:
        arrays, metadata = mx.load(str(trace_path), return_metadata=True)
        require(metadata.get("format") == TRACE_FORMAT, f"unsupported MTP trace: {trace_path}")
        required = {
            "target_hidden",
            "accepted_token_ids",
            "expected_token_ids",
            "scored",
        }
        require(required <= set(arrays), f"MTP trace is incomplete: {trace_path}")
        captured = 0
        for row in range(arrays["target_hidden"].shape[0]):
            if not bool(int(arrays["scored"][row])):
                continue
            target_hidden = arrays["target_hidden"][row]
            accepted_token = int(arrays["accepted_token_ids"][row])
            expected_token = int(arrays["expected_token_ids"][row])
            fused = base_model._attention_step(target_hidden, accepted_token, cache=None)
            hidden = mx.fast.rms_norm(
                fused,
                base_model.moe.norm_weight,
                base_model.moe.epsilon,
            )
            indices, scores = base_model.moe.route(hidden)
            latent = base_model.moe.fc1_latent(hidden)
            expert_input = mx.expand_dims(latent, (-2, -3))
            expert_hidden = mx.square(
                mx.maximum(base_model.moe.switch(expert_input, indices, "up"), 0.0)
            )
            expert_output = base_model.moe.switch(
                expert_hidden, indices, "down"
            ).squeeze(-2)
            routed_sum = (expert_output * scores[..., None]).sum(axis=-2)
            shared = base_model.moe.shared_down(
                mx.square(mx.maximum(base_model.moe.shared_up(hidden), 0.0))
            )
            fixed = fused + shared
            candidate_hidden = fixed + base_model.moe.fc2_latent(routed_sum)
            candidate_hidden = mx.fast.rms_norm(
                candidate_hidden,
                base_model.final_norm_weight,
                base_model.epsilon,
            )
            baseline_full_logits = base_model.lm_head(candidate_hidden).reshape(-1)
            teacher_full_logits, teacher_hidden, teacher_indices, _ = teacher_model.draft_step(
                target_hidden,
                accepted_token,
            )
            require(
                bool(
                    mx.array_equal(
                        base_model.original_expert_ids[indices].reshape(-1),
                        teacher_indices.reshape(-1),
                    )
                ),
                "binary and teacher routes differ",
            )
            mx.eval(baseline_full_logits, teacher_full_logits)
            token_ids, valid_tokens, expected_position = shortlist(
                baseline_full_logits,
                teacher_full_logits,
                expected_token,
                top_k,
            )
            token_array = mx.array(token_ids, dtype=mx.int32)
            subset_head = base_model.lm_head.weight[token_array].astype(mx.float32)
            baseline_subset_logits = mx.sum(
                subset_head * candidate_hidden.reshape(1, -1).astype(mx.float32),
                axis=-1,
            )
            teacher_subset_logits = mx.sum(
                subset_head * teacher_hidden.reshape(1, -1).astype(mx.float32),
                axis=-1,
            )
            rows["fixed"].append(fixed.reshape(-1).astype(mx.float32))
            rows["routed_sum"].append(routed_sum.reshape(-1).astype(mx.float32))
            rows["latent"].append(latent.reshape(-1).astype(mx.float32))
            rows["indices"].append(indices.reshape(-1).astype(mx.int32))
            rows["scores"].append(scores.reshape(-1).astype(mx.float32))
            rows["token_ids"].append(token_array)
            rows["valid_tokens"].append(mx.array(valid_tokens, dtype=mx.bool_))
            rows["teacher_logits"].append(teacher_subset_logits.astype(mx.float32))
            rows["baseline_logits"].append(baseline_subset_logits.astype(mx.float32))
            rows["expected_positions"].append(expected_position)
            captured += 1
            mx.clear_cache()
        operation_log.write(f"mtp-lowbit-sensitivity-trace path={trace_path} rows={captured}")
    output = {
        name: (
            mx.array(values, dtype=mx.int32)
            if name == "expected_positions"
            else mx.stack(values)
        )
        for name, values in rows.items()
    }
    mx.eval(*output.values())
    base_config, base_tensors, _, _ = load_sidecar(base_sidecar)
    return output, base_model, base_config, base_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--base-sidecar", required=True, type=Path)
    parser.add_argument("--teacher-sidecar", required=True, type=Path)
    parser.add_argument("--bf16-sidecar", required=True, type=Path)
    parser.add_argument("--trace", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget", required=True, action="append", type=int)
    parser.add_argument("--high-mode", default="affine3-g128", choices=sorted(MODES))
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.high_mode not in BINARY_MODES, "sensitivity overlay must exceed one bit")
        require(args.top_k > 0, "shortlist size must be positive")
        require(args.teacher_weight >= 0.0, "teacher loss weight must be nonnegative")
        require(args.checkpoint_every > 0, "checkpoint interval must be positive")
        base_config, _, base_payload, base_report = load_sidecar(args.base_sidecar)
        teacher_config, _, teacher_payload, teacher_report = load_sidecar(args.teacher_sidecar)
        bf16_config, bf16_tensors, bf16_payload, bf16_report = load_sidecar(args.bf16_sidecar)
        revision = base_report["source_revision"]
        require(
            teacher_report["source_revision"] == bf16_report["source_revision"] == revision,
            "sensitivity source revisions differ",
        )
        original_expert_ids = base_config["nemotron_mtp_runtime"]["original_expert_ids"]
        require(
            teacher_config["nemotron_mtp_runtime"]["original_expert_ids"]
            == bf16_config["nemotron_mtp_runtime"]["original_expert_ids"]
            == original_expert_ids,
            "sensitivity source expert mappings differ",
        )
        expert_count = len(original_expert_ids)
        budgets = sorted(set(args.budget))
        require(
            len(budgets) == len(args.budget)
            and all(0 < budget <= expert_count for budget in budgets),
            "sensitivity budgets are invalid",
        )
        job = {
            "format": FORMAT,
            "source_revision": revision,
            "base_sidecar": str(args.base_sidecar.resolve()),
            "base_sidecar_sha256": sha256_file(base_payload),
            "teacher_sidecar": str(args.teacher_sidecar.resolve()),
            "teacher_sidecar_sha256": sha256_file(teacher_payload),
            "bf16_sidecar": str(args.bf16_sidecar.resolve()),
            "bf16_sidecar_sha256": sha256_file(bf16_payload),
            "traces": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.trace
            ],
            "high_mode": args.high_mode,
            "top_k": args.top_k,
            "teacher_weight": args.teacher_weight,
            "budgets": budgets,
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": getattr(mx, "__version__", "unknown"),
        }
        job_hash = canonical_hash(job)
        state_path = args.output.with_suffix(args.output.suffix + ".state.json")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output.with_suffix(args.output.suffix + ".log"))
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "format": FORMAT,
                "status": "running",
                "job": job,
                "job_hash": job_hash,
                "expert_metrics": [],
            }
        )
        require(state.get("job_hash") == job_hash, "existing sensitivity job differs")
        if state.get("status") == "complete":
            require(state.get("plan_sha256") == sha256_file(args.output), "sensitivity plan changed")
            operation_log.write(f"mtp-lowbit-sensitivity-already-complete job={job_hash}")
            return 0
        atomic_json(state_path, state)
        operation_log.write(f"mtp-lowbit-sensitivity-start job={job_hash}")
        captured, base_model, runtime_config, base_tensors = capture_rows(
            args.source_dir,
            args.base_sidecar,
            args.teacher_sidecar,
            args.trace,
            args.top_k,
            operation_log,
        )
        baseline_loss = masked_objective(
            captured["baseline_logits"],
            captured["teacher_logits"],
            captured["valid_tokens"],
            captured["expected_positions"],
            args.teacher_weight,
        )
        mx.eval(baseline_loss)
        indices_np = np.asarray(captured["indices"])
        scores_np = np.asarray(captured["scores"])
        completed = {row["expert"] for row in state["expert_metrics"]}
        metrics = {row["expert"]: row for row in state["expert_metrics"]}
        for position, expert in enumerate(range(expert_count), 1):
            if expert in completed:
                continue
            started = time.perf_counter()
            row_indices, slots = np.where(indices_np == expert)
            require(np.unique(row_indices).size == row_indices.size, "expert route is repeated in a row")
            if row_indices.size == 0:
                row = {
                    "expert": expert,
                    "rows": 0,
                    "baseline_loss": 0.0,
                    "candidate_loss": 0.0,
                    "loss_improvement": 0.0,
                    "baseline_expected_wins": 0,
                    "candidate_expected_wins": 0,
                    "net_expected_wins": 0,
                }
            else:
                selected_rows = mx.array(row_indices, dtype=mx.int32)
                latent = captured["latent"][selected_rows]
                base_up = expert_weight(runtime_config, base_tensors, UP_WEIGHT, expert)
                base_down = expert_weight(runtime_config, base_tensors, DOWN_WEIGHT, expert)
                high_up = quantized_weight(bf16_tensors[UP_WEIGHT][expert], args.high_mode)
                high_down = quantized_weight(bf16_tensors[DOWN_WEIGHT][expert], args.high_mode)
                base_hidden = mx.square(mx.maximum(latent @ base_up.T, 0.0))
                high_hidden = mx.square(mx.maximum(latent @ high_up.T, 0.0))
                base_output = base_hidden @ base_down.T
                high_output = high_hidden @ high_down.T
                scores = mx.array(scores_np[row_indices, slots], dtype=mx.float32)
                candidate_sum = captured["routed_sum"][selected_rows] + scores[:, None] * (
                    high_output - base_output
                )
                candidate_hidden = (
                    captured["fixed"][selected_rows]
                    + base_model.moe.fc2_latent(candidate_sum)
                )
                candidate_hidden = mx.fast.rms_norm(
                    candidate_hidden,
                    base_model.final_norm_weight,
                    base_model.epsilon,
                ).reshape(row_indices.size, -1)
                token_ids = captured["token_ids"][selected_rows]
                head_rows = base_model.lm_head.weight[token_ids].astype(mx.float32)
                candidate_logits = mx.sum(
                    head_rows * candidate_hidden[:, None, :].astype(mx.float32),
                    axis=-1,
                )
                candidate_loss = masked_objective(
                    candidate_logits,
                    captured["teacher_logits"][selected_rows],
                    captured["valid_tokens"][selected_rows],
                    captured["expected_positions"][selected_rows],
                    args.teacher_weight,
                )
                baseline_rows_loss = baseline_loss[selected_rows]
                baseline_predictions = mx.argmax(
                    mx.where(
                        captured["valid_tokens"][selected_rows],
                        captured["baseline_logits"][selected_rows],
                        -1e9,
                    ),
                    axis=-1,
                )
                candidate_predictions = mx.argmax(
                    mx.where(
                        captured["valid_tokens"][selected_rows],
                        candidate_logits,
                        -1e9,
                    ),
                    axis=-1,
                )
                expected_positions = captured["expected_positions"][selected_rows]
                values = {
                    "baseline_loss": mx.sum(baseline_rows_loss),
                    "candidate_loss": mx.sum(candidate_loss),
                    "baseline_wins": mx.sum(baseline_predictions == expected_positions),
                    "candidate_wins": mx.sum(candidate_predictions == expected_positions),
                }
                mx.eval(*values.values())
                baseline_value = float(values["baseline_loss"])
                candidate_value = float(values["candidate_loss"])
                baseline_wins = int(values["baseline_wins"])
                candidate_wins = int(values["candidate_wins"])
                row = {
                    "expert": expert,
                    "rows": int(row_indices.size),
                    "route_score_mass": float(scores_np[row_indices, slots].sum()),
                    "baseline_loss": baseline_value,
                    "candidate_loss": candidate_value,
                    "loss_improvement": baseline_value - candidate_value,
                    "baseline_expected_wins": baseline_wins,
                    "candidate_expected_wins": candidate_wins,
                    "net_expected_wins": candidate_wins - baseline_wins,
                }
            metrics[expert] = row
            state["expert_metrics"] = [metrics[key] for key in sorted(metrics)]
            operation_log.write(
                f"mtp-lowbit-sensitivity-expert expert={expert} rows={row['rows']} "
                f"loss_improvement={row['loss_improvement']:.9g} "
                f"net_expected_wins={row['net_expected_wins']} "
                f"elapsed={time.perf_counter() - started:.3f}s"
            )
            if position % args.checkpoint_every == 0 or position == expert_count:
                atomic_json(state_path, state)
            mx.clear_cache()

        ranking = sorted(
            metrics.values(),
            key=lambda row: (
                -row["loss_improvement"],
                -row["net_expected_wins"],
                -row.get("route_score_mass", 0.0),
                row["expert"],
            ),
        )
        plan = {
            "format": PLAN_FORMAT,
            "source_revision": revision,
            "fit_report": str((args.base_sidecar / "nemotron_mtp_pack_report.json").resolve()),
            "fit_report_sha256": sha256_file(
                args.base_sidecar / "nemotron_mtp_pack_report.json"
            ),
            "fit_artifact_sha256": sha256_file(base_payload),
            "expert_count": expert_count,
            "score": "causal-reduced-logit-ablation",
            "sensitivity_job": job,
            "ranked_experts": ranking,
            "budgets": {
                str(budget): sorted(row["expert"] for row in ranking[:budget])
                for budget in budgets
            },
        }
        atomic_json(args.output, plan)
        state.update(
            {
                "status": "complete",
                "plan_sha256": sha256_file(args.output),
                "positive_experts": sum(row["loss_improvement"] > 0.0 for row in ranking),
            }
        )
        atomic_json(state_path, state)
        operation_log.write(
            f"mtp-lowbit-sensitivity-complete positive={state['positive_experts']} "
            f"plan={args.output}"
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, KeyError) as exc:
        if operation_log is not None:
            operation_log.write(f"mtp-lowbit-sensitivity-failed error={exc}")
        print(f"nemotron MTP low-bit sensitivity error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train one validation-gated Router KD step from disjoint prompt corpora."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import corpus_samples
from nemotron_mlx_compare_logits import compare
from nemotron_mlx_stream_forward import StreamingForward, validate_virtual_plan
from nemotron_mlx_streamed_router_kd import (
    FORMAT as PROOF_FORMAT,
    atomic_npy,
    backward_head,
    backward_layer,
    load_router_set,
    student_forward,
)
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-multisample-router-kd-v1"
STATE_FORMAT = "nemotron-multisample-router-kd-state-v1"
SAMPLE_FORMAT = "nemotron-router-kd-gradient-sample-v1"


def parse_categories(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [part.strip() for part in value.split(",") if part.strip()]
    require(result == sorted(set(result)), "categories must be sorted and unique")
    return result


def select_samples(
    corpus_path: Path,
    tokenizer,
    categories: list[str] | None,
    cases: int,
    max_tokens: int,
) -> list[dict]:
    require(cases > 0 and max_tokens > 0, "sample limits must be positive")
    available = corpus_samples(corpus_path)
    if categories is not None:
        present = {category for category, _ in available}
        require(set(categories) <= present, "requested category is absent from corpus")
        available = [(category, sample) for category, sample in available if category in categories]
    require(cases <= len(available), "requested more cases than the corpus provides")
    result = []
    for category, sample in available[:cases]:
        token_ids = tokenizer.encode(sample, add_special_tokens=False)[:max_tokens]
        require(token_ids, f"sample encoded to no tokens: {category}")
        result.append(
            {
                "category": category,
                "sample_sha256": hashlib.sha256(sample.encode()).hexdigest(),
                "token_ids": token_ids,
            }
        )
    return result


def sample_identity(sample: dict) -> dict:
    return {
        "category": sample["category"],
        "sample_sha256": sample["sample_sha256"],
        "token_ids": sample["token_ids"],
    }


def validate_disjoint(train: list[dict], validation: list[dict]) -> None:
    train_hashes = {sample["sample_sha256"] for sample in train}
    validation_hashes = {sample["sample_sha256"] for sample in validation}
    require(not train_hashes & validation_hashes, "training and validation samples overlap")
    train_tokens = {tuple(sample["token_ids"]) for sample in train}
    validation_tokens = {tuple(sample["token_ids"]) for sample in validation}
    require(not train_tokens & validation_tokens, "training and validation token prefixes overlap")


def acceptance_gate(baseline: list[dict], candidate: list[dict]) -> dict:
    require(len(baseline) == len(candidate) and baseline, "validation row count mismatch")
    require(
        [row["category"] for row in baseline] == [row["category"] for row in candidate],
        "validation category order mismatch",
    )
    baseline_kls = [row["metrics"]["kl_baseline_candidate"] for row in baseline]
    candidate_kls = [row["metrics"]["kl_baseline_candidate"] for row in candidate]
    baseline_top1 = sum(
        row["metrics"]["baseline_top1"] == row["metrics"]["candidate_top1"] for row in baseline
    )
    candidate_top1 = sum(
        row["metrics"]["baseline_top1"] == row["metrics"]["candidate_top1"] for row in candidate
    )
    result = {
        "baseline_mean_kl": float(np.mean(baseline_kls)),
        "candidate_mean_kl": float(np.mean(candidate_kls)),
        "baseline_max_kl": max(baseline_kls),
        "candidate_max_kl": max(candidate_kls),
        "baseline_top1_matches": baseline_top1,
        "candidate_top1_matches": candidate_top1,
        "improved_cases": sum(right < left for left, right in zip(baseline_kls, candidate_kls)),
        "regressed_cases": sum(right > left for left, right in zip(baseline_kls, candidate_kls)),
    }
    result["mean_improved"] = result["candidate_mean_kl"] < result["baseline_mean_kl"]
    result["worst_not_regressed"] = result["candidate_max_kl"] <= result["baseline_max_kl"]
    result["top1_not_regressed"] = candidate_top1 >= baseline_top1
    result["accepted"] = all(
        result[key] for key in ("mean_improved", "worst_not_regressed", "top1_not_regressed")
    )
    return result


def logits_row(category: str, teacher: np.ndarray, candidate: np.ndarray) -> dict:
    return {"category": category, "metrics": compare(teacher, candidate, 64)}


def gradient_sample(
    source_dir: Path,
    config: dict,
    retained: dict[str, list[int]],
    source_revision: str,
    plan_sha256: str,
    tool_sha256: str,
    helper_sha256: str,
    sample: dict,
    output_dir: Path,
    temperature: float,
    operation_log: OperationLog,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    identity = {
        "format": SAMPLE_FORMAT,
        "source_revision": source_revision,
        "plan_sha256": plan_sha256,
        "tool_sha256": tool_sha256,
        "helper_sha256": helper_sha256,
        "temperature": temperature,
        **sample_identity(sample),
    }
    state_path = output_dir / "state.json"
    if state_path.exists():
        state = load_json(state_path)
        require(all(state.get(key) == value for key, value in identity.items()), "sample resume mismatch")
    else:
        state = {**identity, "status": "running", "backward_next_layer": len(config["hybrid_override_pattern"]) - 1}
        atomic_json(state_path, state)
    if state.get("status") == "complete":
        require(report_path.is_file(), "completed gradient sample has no report")
        require(state.get("report_sha256") == sha256_file(report_path), "gradient sample report hash mismatch")
        return load_json(report_path)

    teacher_path = output_dir / "teacher-logits.npy"
    if not teacher_path.exists():
        operation_log.write(
            f"train-teacher-start category={sample['category']} tokens={len(sample['token_ids'])}"
        )
        runner = StreamingForward(source_dir)
        logits = runner.forward_sequence(sample["token_ids"])
        atomic_npy(teacher_path, np.asarray(logits, dtype=np.float32))
        operation_log.write(f"train-teacher-done category={sample['category']}")
        del runner, logits
        gc.collect()
        mx.clear_cache()

    activation_dir = output_dir / "activations"
    student_path = output_dir / "student-logits.npy"
    final_boundary = activation_dir / f"boundary-{len(config['hybrid_override_pattern']):03d}.npy"
    if not student_path.exists() or not final_boundary.exists():
        operation_log.write(f"train-student-start category={sample['category']}")
        logits = student_forward(
            source_dir,
            config,
            retained,
            sample["token_ids"],
            activation_dir,
            None,
            operation_log,
        )
        atomic_npy(student_path, logits)
        operation_log.write(f"train-student-done category={sample['category']}")

    final_cotangent_path = output_dir / f"cotangent-{len(config['hybrid_override_pattern']):03d}.npy"
    if not final_cotangent_path.exists():
        initial_kl, cotangent = backward_head(
            source_dir,
            config,
            np.load(final_boundary),
            np.load(teacher_path),
            temperature,
        )
        require(math.isfinite(initial_kl) and np.isfinite(cotangent).all(), "sample head gradient failed")
        atomic_npy(final_cotangent_path, cotangent)
        state["initial_kl"] = initial_kl
        atomic_json(state_path, state)

    next_layer = int(state["backward_next_layer"])
    gradient_dir = output_dir / "router-gradients"
    rows = state.get("gradient_rows", {})
    for layer in range(next_layer, -1, -1):
        kind = config["hybrid_override_pattern"][layer]
        input_gradient, router_gradient = backward_layer(
            source_dir,
            layer,
            kind,
            np.load(activation_dir / f"boundary-{layer:03d}.npy"),
            np.load(output_dir / f"cotangent-{layer + 1:03d}.npy"),
            retained,
        )
        require(np.isfinite(input_gradient).all(), f"sample layer {layer} input gradient failed")
        atomic_npy(output_dir / f"cotangent-{layer:03d}.npy", input_gradient)
        if router_gradient is not None:
            norm = float(np.linalg.norm(router_gradient))
            require(np.isfinite(router_gradient).all() and norm > 0, f"sample layer {layer} router gradient failed")
            atomic_npy(gradient_dir / f"layer-{layer:03d}.npy", router_gradient)
            rows[str(layer)] = {"norm": norm, "finite": True}
        state["gradient_rows"] = rows
        state["backward_next_layer"] = layer - 1
        atomic_json(state_path, state)
        operation_log.write(
            f"train-backward category={sample['category']} layer={layer:02d} kind={kind} "
            f"router_gradient_norm={0.0 if router_gradient is None else np.linalg.norm(router_gradient):.9g}"
        )

    teacher = np.load(teacher_path)
    student = np.load(student_path)
    report = {
        **identity,
        "status": "complete",
        "initial_kl": float(state["initial_kl"]),
        "logits": compare(teacher, student, 64),
        "gradient_rows": rows,
        "peak_gib": mx.get_peak_memory() / 2**30,
    }
    atomic_json(report_path, report)
    state["status"] = "complete"
    state["report_sha256"] = sha256_file(report_path)
    atomic_json(state_path, state)
    return report


def aggregate_gradients(
    config: dict,
    sample_dirs: list[Path],
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer, kind in enumerate(config["hybrid_override_pattern"]):
        if kind != "E":
            continue
        gradients = [
            np.load(sample_dir / "router-gradients" / f"layer-{layer:03d}.npy")
            for sample_dir in sample_dirs
        ]
        shape = gradients[0].shape
        require(all(gradient.shape == shape for gradient in gradients), f"layer {layer} gradient shape mismatch")
        aggregate = np.mean(np.stack(gradients).astype(np.float64), axis=0).astype(np.float32)
        require(np.isfinite(aggregate).all(), f"layer {layer} aggregate gradient is not finite")
        atomic_npy(output_dir / f"layer-{layer:03d}.npy", aggregate)
        rows.append(
            {
                "layer": layer,
                "samples": len(gradients),
                "norm": float(np.linalg.norm(aggregate)),
                "nonzero_rows": int(np.count_nonzero(np.any(aggregate != 0, axis=1))),
            }
        )
    return rows


def save_router_artifact(
    path: Path,
    routers: dict[str, mx.array],
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    mx.save_safetensors(
        str(temporary),
        {f"layer_{int(layer):03d}.gate.weight": value for layer, value in routers.items()},
        metadata={"format": FORMAT, **metadata},
    )
    temporary.replace(path)


def validation_baseline(
    source_dir: Path,
    config: dict,
    retained: dict[str, list[int]],
    sample: dict,
    output_dir: Path,
    operation_log: OperationLog,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = output_dir / "teacher-logits.npy"
    if not teacher_path.exists():
        operation_log.write(f"validation-teacher-start category={sample['category']}")
        runner = StreamingForward(source_dir)
        logits = runner.forward_sequence(sample["token_ids"])
        atomic_npy(teacher_path, np.asarray(logits, dtype=np.float32))
        del runner, logits
        gc.collect()
        mx.clear_cache()
    student_path = output_dir / "baseline-logits.npy"
    if not student_path.exists():
        operation_log.write(f"validation-baseline-start category={sample['category']}")
        logits = student_forward(
            source_dir, config, retained, sample["token_ids"], None, None, operation_log
        )
        atomic_npy(student_path, logits)
    row = logits_row(sample["category"], np.load(teacher_path), np.load(student_path))
    operation_log.write(
        f"validation-baseline-done category={sample['category']} "
        f"kl={row['metrics']['kl_baseline_candidate']:.9g}"
    )
    return row


def validation_candidate(
    source_dir: Path,
    config: dict,
    retained: dict[str, list[int]],
    routers: dict[str, mx.array],
    sample: dict,
    baseline_dir: Path,
    output_path: Path,
    operation_log: OperationLog,
) -> dict:
    if not output_path.exists():
        operation_log.write(f"validation-candidate-start category={sample['category']}")
        logits = student_forward(
            source_dir, config, retained, sample["token_ids"], None, routers, operation_log
        )
        atomic_npy(output_path, logits)
    row = logits_row(sample["category"], np.load(baseline_dir / "teacher-logits.npy"), np.load(output_path))
    operation_log.write(
        f"validation-candidate-done category={sample['category']} "
        f"kl={row['metrics']['kl_baseline_candidate']:.9g}"
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--mechanism-report", required=True, type=Path)
    parser.add_argument("--train-corpus", required=True, type=Path)
    parser.add_argument("--validation-corpus", required=True, type=Path)
    parser.add_argument("--categories")
    parser.add_argument("--train-cases", type=int, default=8)
    parser.add_argument("--validation-cases", type=int, default=8)
    parser.add_argument("--max-sample-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--line-search-steps", type=int, default=6)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = OperationLog(args.output_dir / "run.log")
    try:
        require(args.train_cases > 0 and args.validation_cases > 0, "case counts must be positive")
        require(args.max_sample_tokens > 0, "max sample tokens must be positive")
        require(args.temperature > 0 and args.learning_rate > 0, "optimization settings must be positive")
        require(args.line_search_steps > 0, "line-search steps must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained = validate_virtual_plan(plan, config, source_state["revision"])
        mechanism = load_json(args.mechanism_report)
        require(
            mechanism.get("format") == PROOF_FORMAT and mechanism.get("status") == "complete",
            "invalid mechanism proof",
        )
        require(mechanism.get("source_revision") == source_state["revision"], "mechanism/source mismatch")
        require(mechanism.get("plan_sha256") == sha256_file(args.plan), "mechanism/plan mismatch")
        require(
            mechanism.get("production_training_parity", {}).get("kl_baseline_candidate", 1.0) <= 1e-7,
            "mechanism production parity is insufficient",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        categories = parse_categories(args.categories)
        train_samples = select_samples(
            args.train_corpus, tokenizer, categories, args.train_cases, args.max_sample_tokens
        )
        validation_samples = select_samples(
            args.validation_corpus,
            tokenizer,
            categories,
            args.validation_cases,
            args.max_sample_tokens,
        )
        validate_disjoint(train_samples, validation_samples)
        tool_hash = sha256_file(Path(__file__))
        helper_path = Path(__file__).with_name("nemotron_mlx_streamed_router_kd.py")
        helper_hash = sha256_file(helper_path)
        identity = {
            "format": STATE_FORMAT,
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "plan_sha256": sha256_file(args.plan),
            "mechanism_report_sha256": sha256_file(args.mechanism_report),
            "train_corpus_sha256": sha256_file(args.train_corpus),
            "validation_corpus_sha256": sha256_file(args.validation_corpus),
            "tool_sha256": tool_hash,
            "helper_sha256": helper_hash,
            "categories": categories,
            "train_samples": [sample_identity(sample) for sample in train_samples],
            "validation_samples": [sample_identity(sample) for sample in validation_samples],
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "line_search_steps": args.line_search_steps,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            require(all(state.get(key) == value for key, value in identity.items()), "run resume mismatch")
        else:
            state = {**identity, "status": "running"}
            atomic_json(state_path, state)
        report_path = args.output_dir / "report.json"
        if state.get("status") in ("complete", "rejected"):
            require(report_path.is_file(), "finished run has no report")
            require(state.get("report_sha256") == sha256_file(report_path), "finished report hash mismatch")
            print(json.dumps(load_json(report_path), indent=2, sort_keys=True))
            return 0

        sample_reports = []
        sample_dirs = []
        for index, sample in enumerate(train_samples):
            sample_dir = args.output_dir / "train" / f"sample-{index:03d}"
            sample_dirs.append(sample_dir)
            operation_log.write(f"train-sample-start index={index} category={sample['category']}")
            report = gradient_sample(
                args.source_dir,
                config,
                retained,
                source_state["revision"],
                identity["plan_sha256"],
                tool_hash,
                helper_hash,
                sample,
                sample_dir,
                args.temperature,
                operation_log,
            )
            sample_reports.append(report)
            operation_log.write(
                f"train-sample-done index={index} category={sample['category']} kl={report['initial_kl']:.9g}"
            )

        aggregate_dir = args.output_dir / "aggregate-gradients"
        gradient_rows = aggregate_gradients(config, sample_dirs, aggregate_dir)
        require(all(row["norm"] > 0 for row in gradient_rows), "aggregate contains zero router gradient")
        operation_log.write(
            f"gradient-aggregate-done samples={len(sample_reports)} layers={len(gradient_rows)}"
        )

        baseline_rows = []
        validation_dirs = []
        for index, sample in enumerate(validation_samples):
            validation_dir = args.output_dir / "validation" / f"sample-{index:03d}"
            validation_dirs.append(validation_dir)
            baseline_rows.append(
                validation_baseline(
                    args.source_dir, config, retained, sample, validation_dir, operation_log
                )
            )

        trials = []
        accepted = None
        accepted_routers = None
        accepted_gradient_rows = None
        for step in range(args.line_search_steps):
            learning_rate = args.learning_rate / (2**step)
            operation_log.write(f"validation-trial-start step={step} learning_rate={learning_rate:.9g}")
            routers, router_rows = load_router_set(
                args.source_dir, retained, aggregate_dir, learning_rate
            )
            candidate_rows = []
            trial_dir = args.output_dir / "validation" / f"trial-{step:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            for index, sample in enumerate(validation_samples):
                candidate_rows.append(
                    validation_candidate(
                        args.source_dir,
                        config,
                        retained,
                        routers,
                        sample,
                        validation_dirs[index],
                        trial_dir / f"sample-{index:03d}-logits.npy",
                        operation_log,
                    )
                )
            gate = acceptance_gate(baseline_rows, candidate_rows)
            trial = {
                "step": step,
                "learning_rate": learning_rate,
                "gate": gate,
                "rows": candidate_rows,
            }
            trials.append(trial)
            operation_log.write(
                f"validation-trial-done step={step} learning_rate={learning_rate:.9g} "
                f"baseline_mean={gate['baseline_mean_kl']:.9g} candidate_mean={gate['candidate_mean_kl']:.9g} "
                f"baseline_max={gate['baseline_max_kl']:.9g} candidate_max={gate['candidate_max_kl']:.9g} "
                f"top1={gate['candidate_top1_matches']}/{len(candidate_rows)} accepted={int(gate['accepted'])}"
            )
            if gate["accepted"]:
                accepted = trial
                accepted_routers = routers
                accepted_gradient_rows = router_rows
                break
            del routers
            gc.collect()
            mx.clear_cache()

        artifact = None
        if accepted is not None:
            artifact = args.output_dir / "router.safetensors"
            save_router_artifact(
                artifact,
                accepted_routers,
                {
                    "source_revision": source_state["revision"],
                    "plan_sha256": identity["plan_sha256"],
                    "train_corpus_sha256": identity["train_corpus_sha256"],
                    "validation_corpus_sha256": identity["validation_corpus_sha256"],
                    "tool_sha256": tool_hash,
                    "learning_rate": f"{accepted['learning_rate']:.17g}",
                },
            )
            status = "complete"
        else:
            status = "rejected"
        report = {
            "format": FORMAT,
            "status": status,
            **{key: value for key, value in identity.items() if key != "format"},
            "train": {
                "mean_initial_kl": float(np.mean([report["initial_kl"] for report in sample_reports])),
                "samples": sample_reports,
                "aggregate_gradients": gradient_rows,
            },
            "validation_baseline": baseline_rows,
            "trials": trials,
            "accepted": accepted,
            "accepted_router_rows": accepted_gradient_rows,
            "artifact": None if artifact is None else artifact.name,
            "artifact_sha256": None if artifact is None else sha256_file(artifact),
            "artifact_bytes": None if artifact is None else artifact.stat().st_size,
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
        atomic_json(report_path, report)
        state["status"] = status
        state["report_sha256"] = sha256_file(report_path)
        atomic_json(state_path, state)
        operation_log.write(f"run-{status} report_sha256={state['report_sha256']}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        operation_log.write(f"run-failed error={exc}")
        print(f"nemotron multi-sample Router KD error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebalance completed Router KD sample gradients and rerun heldout validation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_router_kd_train import (
    FORMAT as TRAIN_FORMAT,
    acceptance_gate,
    save_router_artifact,
    validation_candidate,
)
from nemotron_mlx_stream_forward import validate_virtual_plan
from nemotron_mlx_streamed_router_kd import atomic_npy, load_router_set
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-router-kd-rebalance-v1"
STATE_FORMAT = "nemotron-router-kd-rebalance-state-v1"
AGGREGATIONS = ("sample-global-unit", "layer-unit")


def global_gradient_norm(sample_report: dict) -> float:
    rows = sample_report.get("gradient_rows", {})
    require(rows, "training sample has no gradient rows")
    result = math.sqrt(sum(float(row["norm"]) ** 2 for row in rows.values()))
    require(math.isfinite(result) and result > 0, "training sample has invalid global gradient norm")
    return result


def aggregate_layer(gradients: list[np.ndarray], mode: str, global_norms: list[float]) -> np.ndarray:
    require(mode in AGGREGATIONS, f"unsupported aggregation mode: {mode}")
    require(len(gradients) == len(global_norms) and gradients, "gradient sample count mismatch")
    shape = gradients[0].shape
    require(all(gradient.shape == shape for gradient in gradients), "gradient shape mismatch")
    values = [np.asarray(gradient, dtype=np.float64) for gradient in gradients]
    if mode == "sample-global-unit":
        normalized = [gradient / norm for gradient, norm in zip(values, global_norms)]
    else:
        norms = [float(np.linalg.norm(gradient)) for gradient in values]
        require(all(math.isfinite(norm) and norm > 0 for norm in norms), "invalid layer gradient norm")
        normalized = [gradient / norm for gradient, norm in zip(values, norms)]
    result = np.mean(np.stack(normalized), axis=0).astype(np.float32)
    require(np.isfinite(result).all(), "rebalanced gradient is not finite")
    return result


def aggregate_gradients(
    config: dict,
    sample_dirs: list[Path],
    sample_reports: list[dict],
    mode: str,
    output_dir: Path,
) -> tuple[list[dict], list[float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_norms = [global_gradient_norm(report) for report in sample_reports]
    rows = []
    for layer, kind in enumerate(config["hybrid_override_pattern"]):
        if kind != "E":
            continue
        gradients = [
            np.load(sample_dir / "router-gradients" / f"layer-{layer:03d}.npy")
            for sample_dir in sample_dirs
        ]
        aggregate = aggregate_layer(gradients, mode, global_norms)
        atomic_npy(output_dir / f"layer-{layer:03d}.npy", aggregate)
        rows.append(
            {
                "layer": layer,
                "samples": len(gradients),
                "norm": float(np.linalg.norm(aggregate)),
                "nonzero_rows": int(np.count_nonzero(np.any(aggregate != 0, axis=1))),
            }
        )
        del gradients, aggregate
    return rows, global_norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--gradient-run", required=True, type=Path)
    parser.add_argument("--aggregation", choices=AGGREGATIONS, default="sample-global-unit")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--line-search-steps", type=int, default=6)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = OperationLog(args.output_dir / "run.log")
    try:
        require(args.learning_rate > 0 and args.line_search_steps > 0, "invalid line-search setting")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained = validate_virtual_plan(plan, config, source_state["revision"])
        base_report_path = args.gradient_run / "report.json"
        base_state_path = args.gradient_run / "state.json"
        base_report = load_json(base_report_path)
        base_state = load_json(base_state_path)
        require(base_report.get("format") == TRAIN_FORMAT, "gradient run has the wrong format")
        require(base_report.get("status") in ("complete", "rejected"), "gradient run is unfinished")
        require(base_state.get("report_sha256") == sha256_file(base_report_path), "gradient report hash mismatch")
        require(base_report.get("source_revision") == source_state["revision"], "gradient/source mismatch")
        require(base_report.get("source_state_sha256") == sha256_file(args.source_state), "source state mismatch")
        require(base_report.get("plan_sha256") == sha256_file(args.plan), "gradient/plan mismatch")
        trainer_path = Path(__file__).with_name("nemotron_mlx_router_kd_train.py")
        require(base_report.get("tool_sha256") == sha256_file(trainer_path), "gradient trainer changed")

        train_samples = base_report["train"]["samples"]
        validation_samples = base_report["validation_samples"]
        require(train_samples and validation_samples, "gradient run has no samples")
        sample_dirs = [args.gradient_run / "train" / f"sample-{index:03d}" for index in range(len(train_samples))]
        validation_dirs = [
            args.gradient_run / "validation" / f"sample-{index:03d}"
            for index in range(len(validation_samples))
        ]
        require(all(path.is_dir() for path in sample_dirs), "training sample directory is missing")
        require(
            all((path / "teacher-logits.npy").is_file() for path in validation_dirs),
            "heldout teacher logits are missing",
        )

        tool_hash = sha256_file(Path(__file__))
        identity = {
            "format": STATE_FORMAT,
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "plan_sha256": sha256_file(args.plan),
            "gradient_report_sha256": sha256_file(base_report_path),
            "gradient_tool_sha256": base_report["tool_sha256"],
            "tool_sha256": tool_hash,
            "aggregation": args.aggregation,
            "learning_rate": args.learning_rate,
            "line_search_steps": args.line_search_steps,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.output_dir / "state.json"
        report_path = args.output_dir / "report.json"
        if state_path.exists():
            state = load_json(state_path)
            require(all(state.get(key) == value for key, value in identity.items()), "rebalance resume mismatch")
        else:
            state = {**identity, "status": "running"}
            atomic_json(state_path, state)
        if state.get("status") in ("complete", "rejected"):
            require(report_path.is_file(), "finished rebalance run has no report")
            require(state.get("report_sha256") == sha256_file(report_path), "rebalance report hash mismatch")
            print(json.dumps(load_json(report_path), indent=2, sort_keys=True))
            return 0

        aggregate_dir = args.output_dir / "aggregate-gradients"
        rows, global_norms = aggregate_gradients(
            config, sample_dirs, train_samples, args.aggregation, aggregate_dir
        )
        require(all(row["norm"] > 0 for row in rows), "rebalanced aggregate contains a zero gradient")
        operation_log.write(
            f"gradient-rebalance-done mode={args.aggregation} samples={len(train_samples)} layers={len(rows)}"
        )

        baseline_rows = base_report["validation_baseline"]
        require(len(baseline_rows) == len(validation_samples), "heldout baseline count mismatch")
        trials = []
        accepted = None
        accepted_routers = None
        accepted_router_rows = None
        for step in range(args.line_search_steps):
            learning_rate = args.learning_rate / (2**step)
            operation_log.write(f"validation-trial-start step={step} learning_rate={learning_rate:.9g}")
            routers, router_rows = load_router_set(
                args.source_dir, retained, aggregate_dir, learning_rate
            )
            trial_dir = args.output_dir / "validation" / f"trial-{step:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            candidate_rows = []
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
                accepted_router_rows = router_rows
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
                    "gradient_report_sha256": identity["gradient_report_sha256"],
                    "aggregation": args.aggregation,
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
            "sample_global_norms": [
                {"category": sample["category"], "norm": norm}
                for sample, norm in zip(train_samples, global_norms)
            ],
            "aggregate_gradients": rows,
            "validation_baseline": baseline_rows,
            "trials": trials,
            "accepted": accepted,
            "accepted_router_rows": accepted_router_rows,
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
        print(f"nemotron Router KD rebalance error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

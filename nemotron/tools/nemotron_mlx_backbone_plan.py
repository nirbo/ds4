#!/usr/bin/env python3
"""Build and causally gate nested binary/native-NVFP4 backbone plans."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT
from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_backbone_context import load_context_rows
from nemotron_mlx_backbone_fit import STATE_FORMAT as FIT_STATE_FORMAT
from nemotron_mlx_backbone_fit import validate_expert_artifact
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-backbone-lowbit-plan-v1"


def rank_binary_residuals(fit_state: dict) -> tuple[list[dict], list[int]]:
    """Rank deployable binary experts by held-out routed function residual."""

    require(fit_state.get("format") == FIT_STATE_FORMAT, "unsupported backbone fit state")
    completed = fit_state.get("completed")
    skipped = fit_state.get("skipped")
    require(isinstance(completed, list) and isinstance(skipped, list), "invalid backbone fit state")
    forced_native = {int(row["expert"]) for row in skipped}
    deployable = []
    seen = set(forced_native)
    for row in completed:
        expert = row.get("expert")
        require(isinstance(expert, int) and expert not in seen, "duplicate backbone fit expert")
        seen.add(expert)
        validation = row.get("metrics", {}).get("validation", {})
        initial = float(validation.get("initial_error2", math.inf))
        fitted = float(validation.get("fitted_error2", math.inf))
        residual = float(row.get("validation_weighted_residual_error2", math.inf))
        if not all(math.isfinite(value) and value >= 0.0 for value in (initial, fitted, residual)):
            forced_native.add(expert)
            continue
        candidate_plan_eligible = row.get(
            "candidate_plan_eligible",
            row.get("candidate_deployable"),
        )
        if candidate_plan_eligible is None:
            candidate_plan_eligible = fitted < initial
        require(
            isinstance(candidate_plan_eligible, bool),
            "invalid binary planning eligibility decision",
        )
        if not candidate_plan_eligible:
            forced_native.add(expert)
            continue
        deployable.append(
            {
                "expert": expert,
                "validation_weighted_residual_error2": residual,
                "validation_initial_error2": initial,
                "validation_fitted_error2": fitted,
                "validation_improvement": (initial - fitted) / max(initial, 1e-30),
                "fit_strategy": row.get("fit_strategy", "bf16-endpoint"),
                "validation_routes": int(row["validation_routes"]),
            }
        )
    deployable.sort(
        key=lambda row: (
            -row["validation_weighted_residual_error2"],
            -row["validation_routes"],
            row["expert"],
        )
    )
    return deployable, sorted(forced_native)


def nested_assignments(
    expert_count: int,
    ranking: list[dict],
    forced_native: list[int],
    budgets: list[int],
) -> dict[str, dict[str, list[int]]]:
    require(expert_count > 0, "expert count must be positive")
    forced = set(forced_native)
    ranked_ids = [row["expert"] for row in ranking]
    require(len(ranked_ids) == len(set(ranked_ids)), "precision ranking repeats an expert")
    require(not forced.intersection(ranked_ids), "forced expert appears in binary ranking")
    require(forced.union(ranked_ids) == set(range(expert_count)), "precision ranking is incomplete")
    require(budgets == sorted(set(budgets)) and budgets, "native budgets must be sorted and unique")
    require(budgets[0] >= len(forced) and budgets[-1] <= expert_count, "native budget cannot satisfy forced set")
    result = {}
    for budget in budgets:
        native = forced.union(ranked_ids[: budget - len(forced)])
        binary = set(range(expert_count)) - native
        result[str(budget)] = {
            "native_nvfp4_experts": sorted(native),
            "binary_experts": sorted(binary),
        }
    return result


def aggregate_binary_residual(
    evidence: dict[int, tuple[np.ndarray, np.ndarray]],
    binary_experts: list[int],
    rows: int,
    latent_width: int,
) -> np.ndarray:
    result = np.zeros((rows, latent_width), dtype=np.float32)
    for expert in binary_experts:
        require(expert in evidence, f"missing binary evidence for expert {expert}")
        row_ids, residual = evidence[expert]
        require(row_ids.ndim == 1 and residual.shape == (row_ids.size, latent_width), "invalid binary evidence")
        require(np.all((0 <= row_ids) & (row_ids < rows)), "binary evidence row is out of range")
        result[row_ids] += residual
    return result


def load_evidence(fit_dir: Path, fit_state: dict, identity: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    evidence = {}
    artifacts_dir = fit_dir / "experts"
    for entry in fit_state["completed"]:
        arrays = validate_expert_artifact(artifacts_dir / entry["file"], entry, identity)
        rows = np.asarray(arrays["validation.rows"], dtype=np.int32)
        residual = np.asarray(arrays["validation.weighted_residual"], dtype=np.float32)
        evidence[entry["expert"]] = (rows, residual)
    return evidence


def canonical_routes(
    indices: mx.array,
    scores: mx.array,
) -> tuple[mx.array, mx.array]:
    """Order selected routes by expert ID so top-k tie order is irrelevant."""

    require(indices.shape == scores.shape, "route index/score shape mismatch")
    order = mx.argsort(indices, axis=-1)
    return (
        mx.take_along_axis(indices, order, axis=-1),
        mx.take_along_axis(scores, order, axis=-1),
    )


def native_reference_and_fc2(
    source_dir: Path,
    layer: int,
    validation: dict[str, mx.array],
    operation_log: OperationLog,
    chunk_rows: int = 32,
):
    block = load_moe_layer(source_dir, layer)
    rows = validation["layer_input"].shape[0]
    full_reference2 = 0.0
    routed_reference2 = 0.0
    routed_latent_reference2 = 0.0
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        x = validation["layer_input"][start:stop].reshape(1, stop - start, -1)
        output, indices, scores, _, selected_outputs = block.forward_with_expert_outputs(x)
        expected_indices = validation["indices"][start:stop].reshape(1, stop - start, -1)
        expected_scores = validation["scores"][start:stop].reshape(1, stop - start, -1)
        canonical_indices, canonical_scores = canonical_routes(indices, scores)
        expected_indices, expected_scores = canonical_routes(expected_indices, expected_scores)
        routed_latent = (selected_outputs * scores[..., None]).sum(axis=-2)
        routed_output = block.fc2_latent(routed_latent)
        mx.eval(
            output,
            canonical_indices,
            canonical_scores,
            expected_indices,
            expected_scores,
            routed_latent,
            routed_output,
        )
        require(
            bool(mx.array_equal(canonical_indices.astype(mx.int32), expected_indices)),
            "context/native route expert set drifted",
        )
        require(
            bool(mx.allclose(canonical_scores, expected_scores, rtol=1e-5, atol=2e-6)),
            "context/native route scores drifted",
        )
        full_reference2 += float(mx.sum(mx.square(output.astype(mx.float32))))
        routed_reference2 += float(mx.sum(mx.square(routed_output.astype(mx.float32))))
        routed_latent_reference2 += float(mx.sum(mx.square(routed_latent.astype(mx.float32))))
        if stop == rows or stop % 512 == 0:
            operation_log.write(f"backbone-plan-native-progress rows={stop}/{rows}")
    fc2 = block.fc2_latent
    del block
    gc.collect()
    mx.clear_cache()
    return {
        "full_layer_output2": full_reference2,
        "routed_output2": routed_reference2,
        "routed_latent2": routed_latent_reference2,
    }, fc2


def causal_error(
    residual: np.ndarray,
    fc2,
    references: dict[str, float],
    chunk_rows: int = 32,
) -> dict[str, float]:
    latent_error2 = float(np.sum(np.square(residual.astype(np.float64))))
    error2 = 0.0
    maximum = 0.0
    for start in range(0, residual.shape[0], chunk_rows):
        values = mx.array(residual[start : start + chunk_rows], dtype=mx.float32)
        output = fc2(values)
        mx.eval(output)
        error2 += float(mx.sum(mx.square(output.astype(mx.float32))))
        maximum = max(maximum, float(mx.max(mx.abs(output))))
    return {
        "error2": error2,
        "full_layer_reference2": references["full_layer_output2"],
        "routed_reference2": references["routed_output2"],
        "routed_latent_error2": latent_error2,
        "routed_latent_reference2": references["routed_latent2"],
        "full_layer_relative_l2": math.sqrt(
            error2 / max(references["full_layer_output2"], 1e-30)
        ),
        "routed_relative_l2": math.sqrt(
            error2 / max(references["routed_output2"], 1e-30)
        ),
        "routed_latent_relative_l2": math.sqrt(
            latent_error2 / max(references["routed_latent2"], 1e-30)
        ),
        "max_abs": maximum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--proxy-source-dir", required=True, type=Path)
    parser.add_argument("--proxy-source-state", required=True, type=Path)
    parser.add_argument("--native-budget", required=True, action="append", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        contract = load_json(args.contract)
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        fit_state_path = args.fit_dir / "state.json"
        fit_state = load_json(fit_state_path)
        require(fit_state.get("format") == FIT_STATE_FORMAT, "unsupported backbone fit state")
        require(fit_state.get("status") == "complete", "backbone fit is incomplete")
        require(fit_state.get("contract_sha256") == sha256_file(args.contract), "fit/contract mismatch")
        context_state_path = args.contexts / "state.json"
        require(
            fit_state.get("context_state_sha256") == sha256_file(context_state_path),
            "fit/context mismatch",
        )
        proxy_state = load_source_state(args.proxy_source_state, args.proxy_source_dir)
        require(
            fit_state.get("proxy_source_revision") == proxy_state["revision"],
            "fit/native target revision mismatch",
        )
        expert_count = contract["architecture"]["experts"]
        require(fit_state.get("experts") == list(range(expert_count)), "full precision planning requires all experts")
        budgets = sorted(set(args.native_budget))
        require(len(budgets) == len(args.native_budget), "native budgets must be unique")
        ranking, forced_native = rank_binary_residuals(fit_state)
        assignments = nested_assignments(expert_count, ranking, forced_native, budgets)
        identity = {
            "source_revision": fit_state["source_revision"],
            "layer": fit_state["layer"],
            "architecture": fit_state["architecture"],
            "validation_context_rows": fit_state["validation_context_rows"],
            "group_size": fit_state["group_size"],
            "fit_strategy": fit_state.get("fit_strategy", "bf16-endpoint"),
            "contract_sha256": fit_state["contract_sha256"],
            "context_state_sha256": fit_state["context_state_sha256"],
        }
        operation_log = OperationLog(args.output.with_suffix(".log"))
        operation_log.write(
            f"backbone-plan-start layer={fit_state['layer']} forced_native={len(forced_native)} "
            f"budgets={','.join(map(str, budgets))}"
        )
        validation = load_context_rows(args.contexts, "validation")
        evidence = load_evidence(args.fit_dir, fit_state, identity)
        references, fc2 = native_reference_and_fc2(
            args.proxy_source_dir,
            fit_state["layer"],
            validation,
            operation_log,
        )
        storage = contract["storage_units"]
        budget_results = {}
        for budget in budgets:
            assignment = assignments[str(budget)]
            started = time.perf_counter()
            residual = aggregate_binary_residual(
                evidence,
                assignment["binary_experts"],
                validation["latent"].shape[0],
                contract["architecture"]["latent_width"],
            )
            metrics = causal_error(residual, fc2, references)
            payload = (
                len(assignment["binary_experts"])
                * storage["binary_affine_bytes_per_expert"]
                + len(assignment["native_nvfp4_experts"])
                * storage["native_nvfp4_bytes_per_expert"]
            )
            budget_results[str(budget)] = {
                **assignment,
                "layer_payload_bytes": payload,
                "layer_payload_gib": payload / 2**30,
                "causal_layer_output": metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
            operation_log.write(
                f"backbone-plan-budget-done native={budget} payload_gib={payload / 2**30:.6f} "
                f"full_relative_l2={metrics['full_layer_relative_l2']:.9g} "
                f"routed_relative_l2={metrics['routed_relative_l2']:.9g} "
                f"latent_relative_l2={metrics['routed_latent_relative_l2']:.9g} "
                f"max_abs={metrics['max_abs']:.9g}"
            )
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_repository": fit_state["source_repository"],
            "source_revision": fit_state["source_revision"],
            "proxy_source_revision": fit_state["proxy_source_revision"],
            "fit_strategy": fit_state.get("fit_strategy", "bf16-endpoint"),
            "metric_labels": fit_state.get(
                "metric_labels",
                {
                    "initial": "bf16-source-rtn",
                    "fitted": "bf16-endpoint-or-refined",
                },
            ),
            "layer": fit_state["layer"],
            "contract": str(args.contract.resolve()),
            "contract_sha256": sha256_file(args.contract),
            "fit_state": str(fit_state_path.resolve()),
            "fit_state_sha256": sha256_file(fit_state_path),
            "context_state": str(context_state_path.resolve()),
            "context_state_sha256": sha256_file(context_state_path),
            "proxy_source_state_sha256": sha256_file(args.proxy_source_state),
            "tool_sha256": sha256_file(Path(__file__)),
            "ranking_metric": "score-weighted-heldout-native-function-residual-energy",
            "forced_native_experts": forced_native,
            "ranked_binary_residuals": ranking,
            "budgets": budget_results,
        }
        atomic_json(args.output, report)
        operation_log.write(f"backbone-plan-complete output={args.output} sha256={sha256_file(args.output)}")
        print(json.dumps({key: value["causal_layer_output"] for key, value in budget_results.items()}, indent=2))
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"backbone-plan-failed error={exc}")
        print(f"nemotron backbone plan error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

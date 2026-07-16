#!/usr/bin/env python3
"""Screen target-derived affine precision tiers on one Nemotron MoE layer."""

from __future__ import annotations

import argparse
from collections import Counter
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
from nemotron_mlx_backbone_context import STATE_FORMAT as CONTEXT_STATE_FORMAT
from nemotron_mlx_backbone_context import load_context_rows
from nemotron_mlx_backbone_fit import (
    NVFP4TargetWeights,
    STATE_FORMAT as FIT_STATE_FORMAT,
    expert_context_rows,
    validate_expert_artifact,
)
from nemotron_mlx_backbone_lowbit import (
    AffineExpert,
    AffineWeight,
    affine_weight,
    dequantize_affine,
    expert_output,
    kmeans_affine_weight,
    output_error,
    relu2_channel_equalize,
)
from nemotron_mlx_backbone_plan import causal_error, native_reference_and_fc2
from nemotron_prune_materialize import (
    OperationLog,
    atomic_json,
    load_source_state,
    sha256_file,
)
from nemotron_safetensors_inventory import INVENTORY_FORMAT


FORMAT = "nemotron-backbone-lowbit-tier-screen-v1"
TIER_LABELS = ("q1", "q2", "q3", "q4", "native_nvfp4")
AFFINE_BITS = (1, 2, 3, 4)
DEFAULT_PROJECTED_GIB = (30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 68.0, 69.31)
DEFAULT_COST_QUANTUM = 172_032
DEFAULT_EQUALIZATION_STRENGTHS = (0.25, 0.5, 0.75, 1.0)


def artifact_affine_expert(
    arrays: dict[str, mx.array],
    latent_width: int,
    hidden_width: int,
    group_size: int,
) -> AffineExpert:
    def projection(prefix: str, rows: int, columns: int) -> AffineWeight:
        return AffineWeight(
            weight=arrays[f"{prefix}.weight"],
            scales=arrays[f"{prefix}.scales"],
            biases=arrays[f"{prefix}.biases"],
            bits=1,
            group_size=group_size,
            rows=rows,
            columns=columns,
        )

    result = AffineExpert(
        up=projection("up", hidden_width, latent_width),
        down=projection("down", latent_width, hidden_width),
    )
    result.validate()
    return result


def quantized_expert_pair(
    up: mx.array,
    down: mx.array,
    bits: int,
    group_size: int,
    quantizer: str,
    equalization_strength: float,
) -> tuple[mx.array, mx.array]:
    require(quantizer in ("stock", "kmeans"), "unsupported tier quantizer")
    require(0.0 <= equalization_strength <= 1.0, "invalid equalization strength")
    if equalization_strength > 0.0:
        source_up, source_down, _ = relu2_channel_equalize(
            up,
            down,
            group_size=group_size,
            strength=equalization_strength,
        )
    else:
        source_up, source_down = up, down
    quantize = affine_weight if quantizer == "stock" else kmeans_affine_weight
    return (
        dequantize_affine(quantize(source_up, bits, group_size)),
        dequantize_affine(quantize(source_down, bits, group_size)),
    )


def aggregate_tier_residual(
    rows_by_expert: list[np.ndarray],
    residuals: dict[str, list[np.ndarray]],
    assignment: list[str],
    rows: int,
    latent_width: int,
) -> np.ndarray:
    require(len(rows_by_expert) == len(assignment), "tier assignment length mismatch")
    result = np.zeros((rows, latent_width), dtype=np.float32)
    for expert, label in enumerate(assignment):
        require(label in TIER_LABELS, f"unsupported tier assignment: {label}")
        if label == "native_nvfp4":
            continue
        require(label in residuals and expert < len(residuals[label]), f"missing {label} residual")
        row_ids = rows_by_expert[expert]
        values = residuals[label][expert]
        require(
            row_ids.ndim == 1 and values.shape == (row_ids.size, latent_width),
            f"invalid {label} residual for expert {expert}",
        )
        require(np.all((0 <= row_ids) & (row_ids < rows)), "tier residual row is out of range")
        result[row_ids] += values
    return result


def independent_tier_plans(
    losses: np.ndarray,
    payload_bytes: np.ndarray,
    budgets: list[int],
    *,
    cost_quantum: int = DEFAULT_COST_QUANTUM,
) -> list[dict]:
    """Solve a multiple-choice knapsack over independent expert residual energy."""

    require(losses.ndim == 2, "tier losses must be a matrix")
    experts, tiers = losses.shape
    require(experts > 0 and tiers > 1, "tier loss matrix is empty")
    require(payload_bytes.shape == (tiers,), "tier payload shape mismatch")
    require(np.all(np.isfinite(losses)) and np.all(losses >= 0.0), "tier loss is invalid")
    require(np.all(payload_bytes > 0), "tier payload must be positive")
    require(np.all(np.diff(payload_bytes) > 0), "tier payloads must increase strictly")
    require(cost_quantum > 0, "tier planning quantum must be positive")
    require(budgets and budgets == sorted(set(budgets)), "tier budgets must be sorted and unique")

    costs = np.rint(payload_bytes.astype(np.float64) / cost_quantum).astype(np.int32)
    require(np.all(costs > 0) and np.all(np.diff(costs) > 0), "tier planning quantum collapses costs")
    require(budgets[0] >= experts * int(payload_bytes[0]), "tier budget is below the minimum payload")
    maximum_units = min(budgets[-1] // cost_quantum, experts * int(costs[-1]))
    require(maximum_units >= experts * int(costs[0]), "tier budget has no feasible assignment")

    previous = np.full(maximum_units + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    decisions = np.full((experts, maximum_units + 1), 255, dtype=np.uint8)
    for expert in range(experts):
        current = np.full_like(previous, np.inf)
        selected = decisions[expert]
        for tier, cost in enumerate(costs.tolist()):
            if cost > maximum_units:
                continue
            candidate = previous[: maximum_units + 1 - cost] + losses[expert, tier]
            destination = current[cost:]
            better = candidate < destination
            destination[better] = candidate[better]
            selected_indices = np.nonzero(better)[0] + cost
            selected[selected_indices] = tier
        previous = current

    plans = []
    for budget in budgets:
        capacity = min(budget // cost_quantum, maximum_units)
        feasible = np.nonzero(np.isfinite(previous[: capacity + 1]))[0]
        require(feasible.size > 0, f"no feasible tier assignment for budget {budget}")
        selected_cost = int(feasible[np.argmin(previous[feasible])])
        assignment = [0] * experts
        cursor = selected_cost
        for expert in range(experts - 1, -1, -1):
            tier = int(decisions[expert, cursor])
            require(tier != 255, "tier planner backtrack reached an invalid state")
            assignment[expert] = tier
            cursor -= int(costs[tier])
        require(cursor == 0, "tier planner backtrack did not reach its origin")

        exact_payload = int(sum(int(payload_bytes[tier]) for tier in assignment))
        while exact_payload > budget:
            candidates = []
            for expert, current_tier in enumerate(assignment):
                for lower_tier in range(current_tier):
                    saved = int(payload_bytes[current_tier] - payload_bytes[lower_tier])
                    penalty = float(losses[expert, lower_tier] - losses[expert, current_tier])
                    candidates.append((penalty / saved, penalty, -saved, expert, lower_tier))
            require(candidates, "rounded tier plan cannot be repaired to its exact byte budget")
            _, _, _, expert, lower_tier = min(candidates)
            assignment[expert] = lower_tier
            exact_payload = int(sum(int(payload_bytes[tier]) for tier in assignment))

        plans.append(
            {
                "budget_bytes": budget,
                "layer_payload_bytes": exact_payload,
                "selection_error2": float(
                    sum(losses[expert, tier] for expert, tier in enumerate(assignment))
                ),
                "assignment_indices": assignment,
                "tier_counts": {
                    TIER_LABELS[tier]: assignment.count(tier) for tier in range(tiers)
                },
                "planning_cost_quantum": cost_quantum,
            }
        )
    return plans


def projected_model_bytes(fixed_without_mtp: int, layer_payload: int, moe_layers: int) -> int:
    require(fixed_without_mtp >= 0, "fixed model payload is negative")
    require(layer_payload > 0 and moe_layers > 0, "invalid layer projection input")
    return fixed_without_mtp + layer_payload * moe_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--proxy-source-dir", required=True, type=Path)
    parser.add_argument("--proxy-source-state", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--projected-gib", action="append", type=float)
    parser.add_argument(
        "--affine-method",
        choices=("stock", "kmeans", "train-selected", "train-selected-equalized"),
        default="stock",
    )
    parser.add_argument("--equalization-strength", action="append", type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        projected_gib = sorted(set(args.projected_gib or DEFAULT_PROJECTED_GIB))
        require(projected_gib and all(value > 0.0 for value in projected_gib), "invalid projected GiB targets")
        equalization_strengths = sorted(
            set(args.equalization_strength or DEFAULT_EQUALIZATION_STRENGTHS)
        )
        require(
            equalization_strengths
            and all(0.0 < value <= 1.0 for value in equalization_strengths),
            "equalization strengths must be in (0, 1]",
        )

        contract = load_json(args.contract)
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        fit_state_path = args.fit_dir / "state.json"
        fit_state = load_json(fit_state_path)
        require(fit_state.get("format") == FIT_STATE_FORMAT, "unsupported backbone fit state")
        require(fit_state.get("status") == "complete", "backbone fit is incomplete")
        require(fit_state.get("fit_strategy") == "native-target-rtn", "tier screen requires native-target-rtn evidence")
        require(fit_state.get("contract_sha256") == sha256_file(args.contract), "fit/contract mismatch")

        context_state_path = args.contexts / "state.json"
        context_state = load_json(context_state_path)
        require(context_state.get("format") == CONTEXT_STATE_FORMAT, "unsupported context state")
        require(context_state.get("status") == "complete", "context capture is incomplete")
        require(fit_state.get("context_state_sha256") == sha256_file(context_state_path), "fit/context mismatch")

        proxy_state = load_source_state(args.proxy_source_state, args.proxy_source_dir)
        require(fit_state.get("proxy_source_revision") == proxy_state["revision"], "fit/native target revision mismatch")
        inventory = load_json(args.inventory)
        require(inventory.get("format") == INVENTORY_FORMAT, "unsupported source inventory")
        require(inventory.get("source", {}).get("revision") == proxy_state["revision"], "inventory/source revision mismatch")

        expert_count = contract["architecture"]["experts"]
        require(fit_state.get("experts") == list(range(expert_count)), "tier screen requires all experts")
        require(len(fit_state.get("completed", [])) == expert_count, "tier screen fit evidence is incomplete")
        require(not fit_state.get("skipped"), "tier screen fit evidence has skipped experts")
        entries = {entry["expert"]: entry for entry in fit_state["completed"]}
        require(sorted(entries) == list(range(expert_count)), "tier screen fit entries are incomplete")

        roles = inventory.get("roles", {})
        total_payload = int(inventory["totals"]["payload_bytes"])
        mtp_payload = int(roles["mtp"]["bytes"])
        routed_payload = int(roles["backbone_routed_expert"]["bytes"])
        fixed_without_mtp = total_payload - mtp_payload - routed_payload
        moe_layers = len(inventory["moe"]["layers"])
        require(moe_layers == 40, "unexpected Nemotron MoE layer count")
        require(contract["layer"] in inventory["moe"]["layers"], "screen layer is not a source MoE layer")

        first_entry = entries[0]
        payloads = np.array(
            [
                int(first_entry["payload_bytes"]),
                int(first_entry["precision_tiers"]["2"]["payload_bytes"]),
                int(first_entry["precision_tiers"]["3"]["payload_bytes"]),
                int(first_entry["precision_tiers"]["4"]["payload_bytes"]),
                int(contract["storage_units"]["native_nvfp4_bytes_per_expert"]),
            ],
            dtype=np.int64,
        )
        require(np.all(np.diff(payloads) > 0), "tier payload ladder is not strictly increasing")
        for entry in entries.values():
            observed = [
                int(entry["payload_bytes"]),
                int(entry["precision_tiers"]["2"]["payload_bytes"]),
                int(entry["precision_tiers"]["3"]["payload_bytes"]),
                int(entry["precision_tiers"]["4"]["payload_bytes"]),
            ]
            require(observed == payloads[:4].tolist(), "expert tier payloads are not uniform")

        identity = {
            "format": FORMAT,
            "source_repository": fit_state["source_repository"],
            "source_revision": fit_state["source_revision"],
            "proxy_source_revision": fit_state["proxy_source_revision"],
            "layer": fit_state["layer"],
            "architecture": fit_state["architecture"],
            "fit_strategy": fit_state["fit_strategy"],
            "group_size": fit_state["group_size"],
            "contract": str(args.contract.resolve()),
            "contract_sha256": sha256_file(args.contract),
            "fit_state": str(fit_state_path.resolve()),
            "fit_state_sha256": sha256_file(fit_state_path),
            "context_state": str(context_state_path.resolve()),
            "context_state_sha256": sha256_file(context_state_path),
            "proxy_source_state_sha256": sha256_file(args.proxy_source_state),
            "inventory": str(args.inventory.resolve()),
            "inventory_sha256": sha256_file(args.inventory),
            "tool_sha256": sha256_file(Path(__file__)),
            "projected_gib_targets": projected_gib,
            "affine_method": args.affine_method,
            "equalization_strengths": (
                equalization_strengths
                if args.affine_method == "train-selected-equalized"
                else []
            ),
        }
        operation_log = OperationLog(args.output.with_suffix(".log"))
        if args.output.exists():
            existing = load_json(args.output)
            require(existing.get("status") == "complete", "existing tier screen is incomplete")
            for key, value in identity.items():
                require(existing.get(key) == value, f"existing tier screen identity mismatch: {key}")
            operation_log.write(f"backbone-tier-screen-validated output={args.output}")
            return 0

        operation_log.write(
            f"backbone-tier-screen-start layer={fit_state['layer']} experts={expert_count} "
            f"rows={fit_state['validation_context_rows']} method={args.affine_method} "
            f"free_artifact_bytes=report-only"
        )
        validation = load_context_rows(args.contexts, "validation")
        train = (
            load_context_rows(args.contexts, "train")
            if args.affine_method in ("train-selected", "train-selected-equalized")
            else None
        )
        rows = validation["latent"].shape[0]
        latent_width = contract["architecture"]["latent_width"]
        require(rows == fit_state["validation_context_rows"], "validation row count mismatch")

        target = NVFP4TargetWeights(
            args.proxy_source_dir,
            contract["layer"],
            expert_count,
            latent_width,
            contract["architecture"]["hidden_width"],
        )
        rows_by_expert: list[np.ndarray] = []
        residuals: dict[str, list[np.ndarray]] = {f"q{bits}": [] for bits in AFFINE_BITS}
        losses = np.zeros((expert_count, len(TIER_LABELS)), dtype=np.float64)
        expert_reports = []
        artifacts_dir = args.fit_dir / "experts"
        for expert in range(expert_count):
            started = time.perf_counter()
            entry = entries[expert]
            if train is not None:
                train_latent, train_weights, _, _ = expert_context_rows(
                    train,
                    expert,
                    fit_state["route_weight_power"],
                )
            else:
                train_latent = train_weights = None
            validation_latent, validation_weights, validation_rows_np, validation_scores = expert_context_rows(
                validation,
                expert,
                fit_state["route_weight_power"],
            )
            arrays = validate_expert_artifact(
                artifacts_dir / entry["file"],
                entry,
                fit_state,
            )
            binary_rows = np.asarray(arrays["validation.rows"], dtype=np.int32).copy()
            require(np.array_equal(binary_rows, validation_rows_np), "binary/tier validation rows differ")
            binary_residual = np.asarray(
                arrays["validation.weighted_residual"], dtype=np.float32
            ).copy()
            rows_by_expert.append(binary_rows)

            target_up, target_down = target.expert(expert)
            quant_source_up = target_up.astype(mx.bfloat16)
            quant_source_down = target_down.astype(mx.bfloat16)
            target_output = expert_output(validation_latent, target_up, target_down)
            mx.eval(target_output)
            target_train = (
                expert_output(train_latent, target_up, target_down)
                if train_latent is not None
                else None
            )
            if target_train is not None:
                mx.eval(target_train)
            heldout = {
                "q1": {
                    "relative_l2": float(entry["metrics"]["validation"]["fitted_relative_l2"]),
                    "max_abs": float(entry["metrics"]["validation"]["fitted_max_abs"]),
                    "quantizer": "fit-state-native-target-rtn",
                    "equalization_strength": 0.0,
                    "training_selection": None,
                }
            }

            if args.affine_method == "train-selected-equalized":
                binary = artifact_affine_expert(
                    arrays,
                    latent_width,
                    contract["architecture"]["hidden_width"],
                    fit_state["group_size"],
                )
                binary_train = expert_output(
                    train_latent,
                    dequantize_affine(binary.up),
                    dequantize_affine(binary.down),
                )
                binary_selection = output_error(binary_train, target_train, train_weights)
                q1_selection = {
                    "fit-state-native-target-rtn": {
                        "error2": binary_selection["error2"],
                        "relative_l2": binary_selection["relative_l2"],
                    }
                }
                q1_choices = [("fit-state-native-target-rtn", 0.0)]
                for strength in (0.0, *equalization_strengths):
                    candidate_up, candidate_down = quantized_expert_pair(
                        quant_source_up,
                        quant_source_down,
                        1,
                        fit_state["group_size"],
                        "stock",
                        strength,
                    )
                    candidate_train = expert_output(train_latent, candidate_up, candidate_down)
                    metrics = output_error(candidate_train, target_train, train_weights)
                    label = f"stock@{strength:g}"
                    q1_selection[label] = {
                        "error2": metrics["error2"],
                        "relative_l2": metrics["relative_l2"],
                    }
                    q1_choices.append(("stock", strength))
                    del candidate_up, candidate_down, candidate_train
                selected_index = min(
                    range(len(q1_choices)),
                    key=lambda index: q1_selection[
                        (
                            "fit-state-native-target-rtn"
                            if q1_choices[index][0] == "fit-state-native-target-rtn"
                            else f"stock@{q1_choices[index][1]:g}"
                        )
                    ]["error2"],
                )
                q1_quantizer, q1_strength = q1_choices[selected_index]
                if q1_quantizer != "fit-state-native-target-rtn":
                    candidate_up, candidate_down = quantized_expert_pair(
                        quant_source_up,
                        quant_source_down,
                        1,
                        fit_state["group_size"],
                        q1_quantizer,
                        q1_strength,
                    )
                    candidate = expert_output(validation_latent, candidate_up, candidate_down)
                    metrics = output_error(candidate, target_output, validation_weights)
                    weighted_residual = (candidate - target_output) * validation_scores[:, None]
                    mx.eval(weighted_residual)
                    binary_residual = np.asarray(weighted_residual, dtype=np.float32).copy()
                    heldout["q1"] = {
                        "relative_l2": metrics["relative_l2"],
                        "max_abs": metrics["max_abs"],
                        "quantizer": q1_quantizer,
                        "equalization_strength": q1_strength,
                        "training_selection": q1_selection,
                    }
                    del candidate_up, candidate_down, candidate, weighted_residual
                else:
                    heldout["q1"]["training_selection"] = q1_selection
                del binary, binary_train

            residuals["q1"].append(binary_residual)
            losses[expert, 0] = float(np.sum(np.square(binary_residual.astype(np.float64))))

            for bit_width in (2, 3, 4):
                quantizer = args.affine_method if args.affine_method in ("stock", "kmeans") else "stock"
                equalization_strength = 0.0
                selection = None
                if args.affine_method in ("train-selected", "train-selected-equalized"):
                    strengths = (
                        (0.0, *equalization_strengths)
                        if args.affine_method == "train-selected-equalized"
                        else (0.0,)
                    )
                    choices = [
                        (candidate_quantizer, strength)
                        for strength in strengths
                        for candidate_quantizer in ("stock", "kmeans")
                    ]
                    selection = {}
                    for candidate_quantizer, strength in choices:
                        candidate_up, candidate_down = quantized_expert_pair(
                            quant_source_up,
                            quant_source_down,
                            bit_width,
                            fit_state["group_size"],
                            candidate_quantizer,
                            strength,
                        )
                        candidate_train = expert_output(
                            train_latent,
                            candidate_up,
                            candidate_down,
                        )
                        metrics = output_error(candidate_train, target_train, train_weights)
                        selection[f"{candidate_quantizer}@{strength:g}"] = {
                            "error2": metrics["error2"],
                            "relative_l2": metrics["relative_l2"],
                        }
                        del candidate_up, candidate_down, candidate_train
                    quantizer, equalization_strength = min(
                        choices,
                        key=lambda choice: selection[f"{choice[0]}@{choice[1]:g}"]["error2"],
                    )
                up, down = quantized_expert_pair(
                    quant_source_up,
                    quant_source_down,
                    bit_width,
                    fit_state["group_size"],
                    quantizer,
                    equalization_strength,
                )
                candidate = expert_output(
                    validation_latent,
                    up,
                    down,
                )
                metrics = output_error(candidate, target_output, validation_weights)
                expected = entry["precision_tiers"][str(bit_width)]
                if quantizer == "stock" and equalization_strength == 0.0:
                    require(
                        math.isclose(metrics["relative_l2"], expected["relative_l2"], rel_tol=2e-5, abs_tol=1e-7),
                        f"q{bit_width} precision replay drift for expert {expert}",
                    )
                weighted_residual = (candidate - target_output) * validation_scores[:, None]
                mx.eval(weighted_residual)
                residual = np.asarray(weighted_residual, dtype=np.float32).copy()
                residuals[f"q{bit_width}"].append(residual)
                losses[expert, bit_width - 1] = float(
                    np.sum(np.square(residual.astype(np.float64)))
                )
                heldout[f"q{bit_width}"] = {
                    "relative_l2": metrics["relative_l2"],
                    "max_abs": metrics["max_abs"],
                    "quantizer": quantizer,
                    "equalization_strength": equalization_strength,
                    "training_selection": selection,
                }
                del (
                    up,
                    down,
                    candidate,
                    weighted_residual,
                    residual,
                )
            losses[expert, 4] = 0.0
            expert_reports.append(
                {
                    "expert": expert,
                    "validation_routes": int(validation_rows_np.size),
                    "weighted_residual_error2": {
                        label: float(losses[expert, tier])
                        for tier, label in enumerate(TIER_LABELS)
                    },
                    "heldout_function": heldout,
                }
            )
            variant_summary = ",".join(
                f"{heldout[f'q{bits}']['quantizer']}@"
                f"{heldout[f'q{bits}']['equalization_strength']:g}"
                for bits in (1, 2, 3, 4)
            )
            operation_log.write(
                f"backbone-tier-screen-expert-done expert={expert} routes={validation_rows_np.size} "
                f"q1_error2={losses[expert, 0]:.7g} q2_error2={losses[expert, 1]:.7g} "
                f"q3_error2={losses[expert, 2]:.7g} q4_error2={losses[expert, 3]:.7g} "
                f"variants={variant_summary} "
                f"elapsed={time.perf_counter() - started:.2f}s"
            )
            del (
                arrays,
                binary_residual,
                target_up,
                target_down,
                quant_source_up,
                quant_source_down,
                target_output,
                target_train,
            )
            gc.collect()
            mx.clear_cache()

        del target
        gc.collect()
        mx.clear_cache()
        references, fc2 = native_reference_and_fc2(
            args.proxy_source_dir,
            contract["layer"],
            validation,
            operation_log,
        )

        evaluated: dict[tuple[str, ...], dict] = {}

        def evaluate(assignment: list[str]) -> dict:
            key = tuple(assignment)
            if key not in evaluated:
                residual = aggregate_tier_residual(
                    rows_by_expert,
                    residuals,
                    assignment,
                    rows,
                    latent_width,
                )
                evaluated[key] = causal_error(residual, fc2, references)
            return evaluated[key]

        uniform = {}
        for tier, label in enumerate(TIER_LABELS):
            assignment = [label] * expert_count
            layer_payload = int(payloads[tier]) * expert_count
            metrics = evaluate(assignment)
            model_payload = projected_model_bytes(fixed_without_mtp, layer_payload, moe_layers)
            uniform[label] = {
                "tier_counts": {name: expert_count if name == label else 0 for name in TIER_LABELS},
                "layer_payload_bytes": layer_payload,
                "layer_payload_gib": layer_payload / 2**30,
                "projected_model_without_mtp_bytes": model_payload,
                "projected_model_without_mtp_gib": model_payload / 2**30,
                "causal_layer_output": metrics,
            }
            operation_log.write(
                f"backbone-tier-screen-uniform-done tier={label} "
                f"projected_gib={model_payload / 2**30:.6f} "
                f"full_relative_l2={metrics['full_layer_relative_l2']:.9g}"
            )

        budget_bytes = []
        usable_targets = []
        minimum_layer = int(payloads[0]) * expert_count
        maximum_layer = int(payloads[-1]) * expert_count
        for target_gib in projected_gib:
            available = math.floor((target_gib * 2**30 - fixed_without_mtp) / moe_layers)
            if available < minimum_layer:
                operation_log.write(
                    f"backbone-tier-screen-budget-skip target_gib={target_gib:.3f} reason=below-q1-minimum"
                )
                continue
            available = min(available, maximum_layer)
            if budget_bytes and available == budget_bytes[-1]:
                continue
            budget_bytes.append(available)
            usable_targets.append(target_gib)
        require(budget_bytes, "no projected model target can hold the q1 layer")
        plans = independent_tier_plans(losses, payloads, budget_bytes)
        mixed = {}
        for target_gib, plan in zip(usable_targets, plans, strict=True):
            assignment = [TIER_LABELS[tier] for tier in plan.pop("assignment_indices")]
            metrics = evaluate(assignment)
            model_payload = projected_model_bytes(
                fixed_without_mtp,
                plan["layer_payload_bytes"],
                moe_layers,
            )
            key = f"{target_gib:.3f}"
            mixed[key] = {
                "target_model_without_mtp_gib": target_gib,
                **plan,
                "layer_payload_gib": plan["layer_payload_bytes"] / 2**30,
                "projected_model_without_mtp_bytes": model_payload,
                "projected_model_without_mtp_gib": model_payload / 2**30,
                "tier_by_expert": assignment,
                "causal_layer_output": metrics,
            }
            operation_log.write(
                f"backbone-tier-screen-mixed-done target_gib={target_gib:.3f} "
                f"actual_gib={model_payload / 2**30:.6f} counts={plan['tier_counts']} "
                f"full_relative_l2={metrics['full_layer_relative_l2']:.9g}"
            )

        selection_summary = {}
        for label in TIER_LABELS[:-1]:
            counts = Counter(
                (
                    row["heldout_function"][label]["quantizer"],
                    row["heldout_function"][label]["equalization_strength"],
                )
                for row in expert_reports
            )
            selection_summary[label] = {
                f"{quantizer}@{strength:g}": count
                for (quantizer, strength), count in sorted(counts.items())
            }

        if args.affine_method == "train-selected-equalized":
            quantizer_selection = (
                "The existing fitted q1 artifact competes with target-derived q1, while q2-q4 "
                "compare stock and fixed k-means affine quantizers. Exact ReLU-squared channel "
                "equalization strengths are selected per expert and tier only from routed training "
                "contexts; held-out contexts are used after selection."
            )
        elif args.affine_method == "train-selected":
            quantizer_selection = (
                "Stock MLX affine or fixed k-means affine is selected per expert and tier only "
                "from routed training contexts; held-out contexts are used after selection."
            )
        else:
            quantizer_selection = f"All q2-q4 experts use the {args.affine_method} affine quantizer."

        report = {
            **identity,
            "status": "complete",
            "selection_metric": "sum-score-weighted-heldout-expert-latent-residual-error2",
            "quantizer_selection": quantizer_selection,
            "selection_summary": selection_summary,
            "acceptance_metric": "aggregate-residual-through-native-fc2-relative-to-full-layer-output",
            "planning_note": (
                "The multiple-choice knapsack is exact for the independent residual objective at the "
                "recorded cost quantum; every selected assignment is then measured with the actual "
                "aggregate residual and native fc2_latent. This is a layer screen, not a 40-layer "
                "quality claim."
            ),
            "accounting": {
                "source_payload_bytes": total_payload,
                "source_mtp_payload_bytes": mtp_payload,
                "source_routed_expert_payload_bytes": routed_payload,
                "fixed_without_mtp_or_routed_experts_bytes": fixed_without_mtp,
                "moe_layers": moe_layers,
                "tier_payload_bytes_per_expert": {
                    label: int(payloads[tier]) for tier, label in enumerate(TIER_LABELS)
                },
                "native_payload_note": (
                    "The packed runtime native expert removes eight source-container alignment bytes "
                    "per expert while preserving every tensor payload byte."
                ),
            },
            "native_references": references,
            "uniform_tiers": uniform,
            "mixed_budget_frontier": mixed,
            "experts": expert_reports,
        }
        atomic_json(args.output, report)
        operation_log.write(
            f"backbone-tier-screen-complete output={args.output} bytes={args.output.stat().st_size} "
            f"sha256={sha256_file(args.output)}"
        )
        print(
            json.dumps(
                {
                    "uniform": {
                        label: {
                            "projected_gib": row["projected_model_without_mtp_gib"],
                            "full_relative_l2": row["causal_layer_output"]["full_layer_relative_l2"],
                        }
                        for label, row in uniform.items()
                    },
                    "mixed": {
                        key: {
                            "actual_gib": row["projected_model_without_mtp_gib"],
                            "tier_counts": row["tier_counts"],
                            "full_relative_l2": row["causal_layer_output"]["full_layer_relative_l2"],
                        }
                        for key, row in mixed.items()
                    },
                },
                indent=2,
            )
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"backbone-tier-screen-failed error={exc}")
        print(f"nemotron backbone tier screen error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

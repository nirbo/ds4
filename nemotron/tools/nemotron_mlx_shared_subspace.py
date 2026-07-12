#!/usr/bin/env python3
"""Screen paired expert prototypes with shared low-rank residuals on real inputs."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_layer_distill import capture_inputs
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_proxy_compare import error_metrics
from nemotron_mlx_stream_forward import validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-shared-subspace-screen-v1"
BLOCK = 16


def parse_ranks(value: str) -> list[int]:
    try:
        ranks = sorted({int(item) for item in value.split(",")})
    except ValueError as exc:
        raise MetadataError(f"invalid rank list: {value}") from exc
    require(ranks and ranks[0] > 0, "ranks must be positive")
    return ranks


def select_functional_pairs(
    counts: np.ndarray,
    cosine_sums: np.ndarray,
    retained: list[int],
    minimum_support: int,
    pair_count: int,
) -> list[dict]:
    require(counts.shape == cosine_sums.shape and counts.ndim == 2, "invalid pair arrays")
    require(counts.shape[0] == counts.shape[1], "pair arrays must be square")
    require(minimum_support > 0 and pair_count > 0, "invalid pair selection limits")
    candidates = []
    for first_index, first in enumerate(retained):
        for second in retained[first_index + 1 :]:
            support = int(counts[first, second])
            if support < minimum_support:
                continue
            cosine = float(cosine_sums[first, second] / support)
            if math.isfinite(cosine):
                candidates.append((cosine, support, first, second))
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    selected = []
    used = set()
    for cosine, support, first, second in candidates:
        if first in used or second in used:
            continue
        selected.append(
            {"experts": [first, second], "mean_output_cosine": cosine, "support": support}
        )
        used.update((first, second))
        if len(selected) == pair_count:
            break
    require(len(selected) == pair_count, "insufficient disjoint supported expert pairs")
    return selected


def randomized_svd(
    matrix: np.ndarray,
    rank: int,
    seed: int,
    oversample: int = 8,
    power_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(matrix.ndim == 2, "randomized SVD requires a matrix")
    require(0 < rank <= min(matrix.shape), "rank exceeds matrix dimensions")
    require(oversample >= 0 and power_iterations >= 0, "invalid randomized SVD controls")
    width = min(min(matrix.shape), rank + oversample)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], width), dtype=np.float32)
    source = matrix.astype(np.float32, copy=False)
    sample = source @ omega
    for _ in range(power_iterations):
        sample = source @ (source.T @ sample)
    basis, _ = np.linalg.qr(sample, mode="reduced")
    small = basis.T @ source
    left, singular, right = np.linalg.svd(small, full_matrices=False)
    return (
        (basis @ left[:, :rank]).astype(np.float32),
        singular[:rank].astype(np.float32),
        right[:rank].astype(np.float32),
    )


def reconstruct_pair(
    first: np.ndarray,
    second: np.ndarray,
    first_frequency: float,
    second_frequency: float,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    decomposition = decompose_pair(
        first, second, first_frequency, second_frequency, rank, seed
    )
    return reconstruct_decomposition(decomposition, rank)


def decompose_pair(
    first: np.ndarray,
    second: np.ndarray,
    first_frequency: float,
    second_frequency: float,
    rank: int,
    seed: int,
) -> dict:
    require(first.shape == second.shape and first.ndim == 2, "expert matrices differ")
    total = first_frequency + second_frequency
    require(total > 0.0, "pair has no calibration frequency")
    first_mix = first_frequency / total
    second_mix = second_frequency / total
    prototype = first_mix * first + second_mix * second
    delta = first - second
    left, singular, right = randomized_svd(delta, rank, seed)
    return {
        "prototype": prototype,
        "left": left,
        "singular": singular,
        "right": right,
        "first_coefficient": second_mix,
        "second_coefficient": -first_mix,
    }


def reconstruct_decomposition(decomposition: dict, rank: int) -> tuple[np.ndarray, np.ndarray, dict]:
    singular = decomposition["singular"]
    require(0 < rank <= len(singular), "rank exceeds decomposition")
    delta = (
        decomposition["left"][:, :rank] * singular[:rank]
    ) @ decomposition["right"][:rank]
    first_coefficient = decomposition["first_coefficient"]
    second_coefficient = decomposition["second_coefficient"]
    prototype = decomposition["prototype"]
    return (
        prototype + first_coefficient * delta,
        prototype + second_coefficient * delta,
        {
            "first_coefficient": first_coefficient,
            "second_coefficient": second_coefficient,
            "singular_values": singular[:rank].tolist(),
        },
    )


def projected_pair_bytes(
    rows: int,
    columns: int,
    rank: int,
    prototype_bits: float,
    factor_bits: float,
) -> dict[str, float]:
    require(rows > 0 and columns > 0 and rank > 0, "invalid projection dimensions")
    require(prototype_bits > 0.0 and factor_bits > 0.0, "invalid projection precision")
    original = 2 * rows * columns * prototype_bits / 8
    proposed = rows * columns * prototype_bits / 8 + rank * (rows + columns) * factor_bits / 8
    return {
        "original_bytes": original,
        "proposed_bytes": proposed,
        "ratio": proposed / original,
        "saving_fraction": 1.0 - proposed / original,
    }


def union_decompose_pair(
    first: np.ndarray,
    second: np.ndarray,
    rank: int,
    seed: int,
) -> dict:
    require(first.shape == second.shape and first.ndim == 2, "expert matrices differ")
    joint = np.concatenate((first, second), axis=1)
    basis, singular, _ = randomized_svd(joint, rank, seed)
    return {
        "basis": basis,
        "singular": singular,
        "first_coefficients": basis.T @ first,
        "second_coefficients": basis.T @ second,
    }


def reconstruct_union(decomposition: dict, rank: int) -> tuple[np.ndarray, np.ndarray, dict]:
    require(0 < rank <= len(decomposition["singular"]), "rank exceeds union decomposition")
    basis = decomposition["basis"][:, :rank]
    return (
        basis @ decomposition["first_coefficients"][:rank],
        basis @ decomposition["second_coefficients"][:rank],
        {"singular_values": decomposition["singular"][:rank].tolist()},
    )


def input_union_decompose_pair(
    first: np.ndarray,
    second: np.ndarray,
    rank: int,
    seed: int,
) -> dict:
    require(first.shape == second.shape and first.ndim == 2, "expert matrices differ")
    joint = np.concatenate((first, second), axis=0)
    basis, singular, _ = randomized_svd(joint.T, rank, seed)
    return {
        "basis": basis,
        "singular": singular,
        "first_coefficients": first @ basis,
        "second_coefficients": second @ basis,
    }


def reconstruct_input_union(
    decomposition: dict,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    require(0 < rank <= len(decomposition["singular"]), "rank exceeds input union")
    basis = decomposition["basis"][:, :rank]
    return (
        decomposition["first_coefficients"][:, :rank] @ basis.T,
        decomposition["second_coefficients"][:, :rank] @ basis.T,
        {"singular_values": decomposition["singular"][:rank].tolist()},
    )


def projected_union_bytes(
    rows: int,
    columns: int,
    rank: int,
    original_bits: float,
    factor_bits: float,
) -> dict[str, float]:
    require(rows > 0 and columns > 0 and rank > 0, "invalid union dimensions")
    require(original_bits > 0.0 and factor_bits > 0.0, "invalid union precision")
    original = 2 * rows * columns * original_bits / 8
    proposed = rank * (rows + 2 * columns) * factor_bits / 8
    return {
        "original_bytes": original,
        "proposed_bytes": proposed,
        "ratio": proposed / original,
        "saving_fraction": 1.0 - proposed / original,
    }


def dequantize_expert(weight, expert: int) -> np.ndarray:
    value = mx.dequantize(
        weight.weight[expert].view(mx.uint32),
        weight.scales[expert],
        None,
        BLOCK,
        4,
        "nvfp4",
        dtype=mx.float32,
    ) * weight.global_scales[expert]
    mx.eval(value)
    result = np.asarray(value, dtype=np.float32)
    del value
    mx.clear_cache()
    return result


def expert_forward(latent: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    activated = np.maximum(latent @ up.T, 0.0) ** 2
    return activated @ down.T


def aggregate(rows: list[dict], key: str) -> dict[str, float]:
    values = [float(row[key]["relative_l2"]) for row in rows]
    return {
        "mean_relative_l2": float(np.mean(values)),
        "max_relative_l2": max(values),
        "samples": len(values),
    }


def pair_inputs(
    block,
    validation,
    layer: int,
    pair: list[int],
    retained: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    latent_rows = {expert: [] for expert in pair}
    exact_rows = {expert: [] for expert in pair}
    for _, inputs in validation:
        x = mx.array(inputs[layer])
        hidden = block.norm(x)
        latent = block.fc1_latent(hidden)
        indices, _ = block.route_retained(hidden, retained)
        outputs = expert_outputs(latent, block.experts, indices)
        mx.eval(latent, indices, outputs)
        latent_np = np.asarray(latent, dtype=np.float32).reshape(-1, latent.shape[-1])
        indices_np = np.asarray(indices, dtype=np.int64).reshape(-1, indices.shape[-1])
        outputs_np = np.asarray(outputs, dtype=np.float32).reshape(
            -1, indices.shape[-1], outputs.shape[-1]
        )
        for expert in pair:
            for token, slot in np.argwhere(indices_np == expert):
                latent_rows[expert].append(latent_np[token])
                exact_rows[expert].append(outputs_np[token, slot])
        del x, hidden, latent, indices, outputs
        mx.clear_cache()
    return (
        {
            expert: np.asarray(values, dtype=np.float32).reshape(-1, block.experts.up.input_dims)
            for expert, values in latent_rows.items()
        },
        {
            expert: np.asarray(values, dtype=np.float32).reshape(-1, block.experts.down.output_dims)
            for expert, values in exact_rows.items()
        },
    )


def evaluate_pair(
    block,
    validation,
    layer: int,
    pair: dict,
    frequencies: np.ndarray,
    retained: list[int],
    ranks: list[int],
    seed: int,
    operation_log: OperationLog,
) -> dict:
    first, second = pair["experts"]
    operation_log.write(
        f"pair-start layer={layer} experts={first},{second} cosine={pair['mean_output_cosine']:.6g}"
    )
    matrices = {
        "up": (
            dequantize_expert(block.experts.up, first),
            dequantize_expert(block.experts.up, second),
        ),
        "down": (
            dequantize_expert(block.experts.down, first),
            dequantize_expert(block.experts.down, second),
        ),
    }
    latent, exact = pair_inputs(block, validation, layer, [first, second], retained)
    require(sum(len(values) for values in latent.values()) > 0, "pair has no held-out observations")
    factorizations = {
        name: decompose_pair(
            first_matrix,
            second_matrix,
            max(float(frequencies[first]), 1.0),
            max(float(frequencies[second]), 1.0),
            max(ranks),
            seed + layer * 1009 + first * 17 + second * 31 + offset,
        )
        for offset, (name, (first_matrix, second_matrix)) in enumerate(matrices.items())
    }
    union_factorizations = {
        name: union_decompose_pair(
            first_matrix,
            second_matrix,
            max(ranks),
            seed + 500_000 + layer * 1009 + first * 17 + second * 31 + offset,
        )
        for offset, (name, (first_matrix, second_matrix)) in enumerate(matrices.items())
    }
    input_union_factorizations = {
        name: input_union_decompose_pair(
            first_matrix,
            second_matrix,
            max(ranks),
            seed + 900_000 + layer * 1009 + first * 17 + second * 31 + offset,
        )
        for offset, (name, (first_matrix, second_matrix)) in enumerate(matrices.items())
    }
    results = []
    for rank in ranks:
        reconstructed = {}
        union_reconstructed = {}
        input_union_reconstructed = {}
        matrix_metrics = {}
        union_matrix_metrics = {}
        input_union_matrix_metrics = {}
        decomposition = {}
        union_decomposition = {}
        input_union_decomposition = {}
        for name, (first_matrix, second_matrix) in matrices.items():
            first_value, second_value, details = reconstruct_decomposition(
                factorizations[name], rank
            )
            reconstructed[name] = {first: first_value, second: second_value}
            matrix_metrics[name] = {
                str(first): error_metrics(first_value, first_matrix),
                str(second): error_metrics(second_value, second_matrix),
            }
            decomposition[name] = details
            union_first, union_second, union_details = reconstruct_union(
                union_factorizations[name], rank
            )
            union_reconstructed[name] = {first: union_first, second: union_second}
            union_matrix_metrics[name] = {
                str(first): error_metrics(union_first, first_matrix),
                str(second): error_metrics(union_second, second_matrix),
            }
            union_decomposition[name] = union_details
            input_first, input_second, input_details = reconstruct_input_union(
                input_union_factorizations[name], rank
            )
            input_union_reconstructed[name] = {first: input_first, second: input_second}
            input_union_matrix_metrics[name] = {
                str(first): error_metrics(input_first, first_matrix),
                str(second): error_metrics(input_second, second_matrix),
            }
            input_union_decomposition[name] = input_details
        functional = []
        union_functional = []
        input_union_functional = []
        for expert in (first, second):
            if len(latent[expert]) == 0:
                continue
            approximate = expert_forward(
                latent[expert], reconstructed["up"][expert], reconstructed["down"][expert]
            )
            functional.append(
                {
                    "expert": expert,
                    "samples": len(latent[expert]),
                    "output": error_metrics(approximate, exact[expert]),
                }
            )
            union_approximate = expert_forward(
                latent[expert],
                union_reconstructed["up"][expert],
                union_reconstructed["down"][expert],
            )
            union_functional.append(
                {
                    "expert": expert,
                    "samples": len(latent[expert]),
                    "output": error_metrics(union_approximate, exact[expert]),
                }
            )
            input_union_approximate = expert_forward(
                latent[expert],
                input_union_reconstructed["up"][expert],
                input_union_reconstructed["down"][expert],
            )
            input_union_functional.append(
                {
                    "expert": expert,
                    "samples": len(latent[expert]),
                    "output": error_metrics(input_union_approximate, exact[expert]),
                }
            )
        require(functional, "pair produced no functional comparisons")
        storage = {}
        union_storage = {}
        for factor_bits in (16.0, 8.0):
            projections = [
                projected_pair_bytes(*value[0].shape, rank, 4.5, factor_bits)
                for value in matrices.values()
            ]
            original = sum(value["original_bytes"] for value in projections)
            proposed = sum(value["proposed_bytes"] for value in projections)
            storage[f"factor_{int(factor_bits)}bit"] = {
                "original_bytes": original,
                "proposed_bytes": proposed,
                "ratio": proposed / original,
                "saving_fraction": 1.0 - proposed / original,
            }
            union_projections = [
                projected_union_bytes(*value[0].shape, rank, 4.5, factor_bits)
                for value in matrices.values()
            ]
            union_original = sum(value["original_bytes"] for value in union_projections)
            union_proposed = sum(value["proposed_bytes"] for value in union_projections)
            union_storage[f"factor_{int(factor_bits)}bit"] = {
                "original_bytes": union_original,
                "proposed_bytes": union_proposed,
                "ratio": union_proposed / union_original,
                "saving_fraction": 1.0 - union_proposed / union_original,
            }
        result = {
            "rank": rank,
            "matrix": matrix_metrics,
            "functional": functional,
            "functional_summary": aggregate(functional, "output"),
            "storage": storage,
            "decomposition": decomposition,
            "union_matrix": union_matrix_metrics,
            "union_functional": union_functional,
            "union_functional_summary": aggregate(union_functional, "output"),
            "union_storage": union_storage,
            "union_decomposition": union_decomposition,
            "input_union_matrix": input_union_matrix_metrics,
            "input_union_functional": input_union_functional,
            "input_union_functional_summary": aggregate(input_union_functional, "output"),
            "input_union_storage": union_storage,
            "input_union_decomposition": input_union_decomposition,
        }
        results.append(result)
        operation_log.write(
            f"pair-rank layer={layer} experts={first},{second} rank={rank} "
            f"output_mean={result['functional_summary']['mean_relative_l2']:.6g} "
            f"union_output_mean={result['union_functional_summary']['mean_relative_l2']:.6g} "
            f"input_union_output_mean={result['input_union_functional_summary']['mean_relative_l2']:.6g} "
            f"bf16_saving={storage['factor_16bit']['saving_fraction']:.3%} "
            f"union_bf16_saving={union_storage['factor_16bit']['saving_fraction']:.3%}"
        )
    del matrices, factorizations, union_factorizations, input_union_factorizations, latent, exact
    gc.collect()
    mx.clear_cache()
    return {**pair, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--proxy-state", required=True, type=Path)
    parser.add_argument("--proxy-arrays", required=True, type=Path)
    parser.add_argument("--validation-corpus", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--pair-count", type=int, default=2)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--validation-cases", type=int, default=4)
    parser.add_argument("--max-sample-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        ranks = parse_ranks(args.ranks)
        require(args.pair_count > 0 and args.validation_cases > 0, "invalid screen size")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained_by_layer = validate_virtual_plan(plan, config, source_state["revision"])
        require(str(args.layer) in retained_by_layer, "layer is absent from prune plan")
        proxy_state = load_json(args.proxy_state)
        require(proxy_state.get("format") == "nemotron-proxy-calibration-v1", "invalid proxy state")
        require(proxy_state.get("status") == "complete", "proxy calibration is incomplete")
        require(proxy_state.get("source_revision") == source_state["revision"], "proxy revision mismatch")
        require(
            proxy_state.get("arrays_sha256") == sha256_file(args.proxy_arrays),
            "proxy arrays hash mismatch",
        )
        with np.load(args.proxy_arrays) as loaded:
            prefix = f"layer_{args.layer:03d}"
            counts = loaded[f"{prefix}_pair_counts"]
            cosine_sums = loaded[f"{prefix}_cosine_sums"]
            frequencies = loaded[f"{prefix}_category_counts"].sum(axis=0)
        retained = retained_by_layer[str(args.layer)]
        pairs = select_functional_pairs(
            counts, cosine_sums, retained, args.minimum_support, args.pair_count
        )
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        validation = capture_inputs(
            args.source_dir,
            tokenizer,
            args.validation_corpus,
            [args.layer],
            args.max_sample_tokens,
            args.validation_cases,
            "shared-subspace-validation",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "screen.log")
        operation_log.write(
            f"screen-start layer={args.layer} pairs={len(pairs)} ranks={','.join(map(str, ranks))}"
        )
        block = load_moe_layer(args.source_dir, args.layer)
        started = time.perf_counter()
        results = [
            evaluate_pair(
                block,
                validation,
                args.layer,
                pair,
                frequencies,
                retained,
                ranks,
                args.seed,
                operation_log,
            )
            for pair in pairs
        ]
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "plan_sha256": sha256_file(args.plan),
            "proxy_state_sha256": sha256_file(args.proxy_state),
            "proxy_arrays_sha256": sha256_file(args.proxy_arrays),
            "validation_corpus_sha256": sha256_file(args.validation_corpus),
            "tool_sha256": sha256_file(Path(__file__)),
            "layer": args.layer,
            "retained_experts": len(retained),
            "minimum_support": args.minimum_support,
            "ranks": ranks,
            "validation_cases": args.validation_cases,
            "max_sample_tokens": args.max_sample_tokens,
            "pairs": results,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(args.output_dir / "report.json", report)
        operation_log.write(
            f"screen-complete elapsed={report['elapsed_seconds']:.2f}s "
            f"report_sha256={sha256_file(args.output_dir / 'report.json')}"
        )
        print(
            f"shared-subspace-report path={args.output_dir / 'report.json'} "
            f"sha256={sha256_file(args.output_dir / 'report.json')}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
        if operation_log is not None:
            operation_log.write(f"screen-failed error={exc}")
        print(f"nemotron shared-subspace screen error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

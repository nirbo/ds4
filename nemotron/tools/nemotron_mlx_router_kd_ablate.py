#!/usr/bin/env python3
"""Screen contiguous Router KD layer-group ablations on selected heldout cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import corpus_samples
from nemotron_mlx_compare_logits import compare
from nemotron_mlx_plan_compare import score, summarize
from nemotron_mlx_router_distill import load_router_artifact
from nemotron_mlx_stream_forward import validate_virtual_plan
from nemotron_mlx_streamed_router_kd import atomic_npy, load_source_router
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-router-kd-layer-ablation-v1"


def parse_indices(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise MetadataError(f"invalid case index: {value}") from exc
    require(result == sorted(set(result)) and result, "case indices must be sorted and unique")
    require(result[0] >= 0, "case indices must be nonnegative")
    return result


def partition_layers(layers: list[int], groups: int) -> list[list[int]]:
    require(layers == sorted(set(layers)) and layers, "layers must be sorted and unique")
    require(1 <= groups <= len(layers), "invalid layer group count")
    quotient, remainder = divmod(len(layers), groups)
    result = []
    offset = 0
    for index in range(groups):
        size = quotient + (1 if index < remainder else 0)
        result.append(layers[offset : offset + size])
        offset += size
    require(offset == len(layers) and all(result), "layer partition failed")
    return result


def gate(base_cases: list[dict], candidate_cases: list[dict], key: str) -> dict:
    require(len(base_cases) == len(candidate_cases) and base_cases, "ablation case count mismatch")
    base_metrics = [case["base"] for case in base_cases]
    candidate_metrics = [case[key] for case in candidate_cases]
    base_kls = [row["kl_baseline_candidate"] for row in base_metrics]
    candidate_kls = [row["kl_baseline_candidate"] for row in candidate_metrics]
    base_top1 = sum(row["baseline_top1"] == row["candidate_top1"] for row in base_metrics)
    candidate_top1 = sum(
        row["baseline_top1"] == row["candidate_top1"] for row in candidate_metrics
    )
    result = {
        "base_mean_kl": float(np.mean(base_kls)),
        "candidate_mean_kl": float(np.mean(candidate_kls)),
        "base_max_kl": max(base_kls),
        "candidate_max_kl": max(candidate_kls),
        "base_top1_matches": base_top1,
        "candidate_top1_matches": candidate_top1,
        "improved_cases": sum(right < left for left, right in zip(base_kls, candidate_kls)),
    }
    result["mean_improved"] = result["candidate_mean_kl"] < result["base_mean_kl"]
    result["worst_not_regressed"] = result["candidate_max_kl"] <= result["base_max_kl"]
    result["top1_not_regressed"] = candidate_top1 >= base_top1
    result["accepted"] = all(
        result[name] for name in ("mean_improved", "worst_not_regressed", "top1_not_regressed")
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--router-report", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--case-indices", required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--max-sample-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = OperationLog(args.output_dir / "run.log")
    try:
        require(args.max_sample_tokens > 0 and args.top_k > 0, "invalid evaluation limits")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained = validate_virtual_plan(plan, config, source_state["revision"])
        trained_routers, router_report = load_router_artifact(
            args.router_report,
            args.plan,
            source_state["revision"],
            retained,
            config["hidden_size"],
        )
        layers = sorted(int(layer) for layer in retained)
        groups = partition_layers(layers, args.groups)
        indices = parse_indices(args.case_indices)
        available = corpus_samples(args.corpus)
        require(indices[-1] < len(available), "case index exceeds corpus")
        selected = [available[index] for index in indices]
        identity = {
            "format": FORMAT,
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "plan_sha256": sha256_file(args.plan),
            "router_report_sha256": sha256_file(args.router_report),
            "router_artifact_sha256": router_report["artifact_sha256"],
            "corpus_sha256": sha256_file(args.corpus),
            "case_indices": indices,
            "groups": groups,
            "max_sample_tokens": args.max_sample_tokens,
            "top_k": args.top_k,
            "tool_sha256": sha256_file(Path(__file__)),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / "report.json"
        if report_path.exists():
            report = load_json(report_path)
            require(all(report.get(key) == value for key, value in identity.items()), "ablation resume mismatch")
        else:
            report = {**identity, "status": "running", "cases": []}
            atomic_json(report_path, report)
        completed = {case["sample_sha256"] for case in report["cases"]}
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        source_routers = {
            layer: load_source_router(args.source_dir, int(layer), retained[layer])
            for layer in retained
        }
        variants = {"full": trained_routers}
        for group_index, group in enumerate(groups):
            routers = dict(trained_routers)
            for layer in group:
                routers[str(layer)] = source_routers[str(layer)]
            variants[f"ablate_{group_index}"] = routers

        cache_dir = args.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        for selected_index, (category, sample) in zip(indices, selected):
            sample_hash = hashlib.sha256(sample.encode()).hexdigest()
            if sample_hash in completed:
                continue
            token_ids = tokenizer.encode(sample, add_special_tokens=False)[: args.max_sample_tokens]
            require(token_ids, f"sample encoded to no tokens: {category}")
            operation_log.write(
                f"case-start index={selected_index} category={category} tokens={len(token_ids)}"
            )
            source_path = cache_dir / f"case-{selected_index:03d}-source.npy"
            base_path = cache_dir / f"case-{selected_index:03d}-base.npy"
            if not source_path.exists():
                atomic_npy(source_path, score(args.source_dir, token_ids, None))
            if not base_path.exists():
                atomic_npy(base_path, score(args.source_dir, token_ids, retained))
            source_logits = np.load(source_path)
            case = {
                "index": selected_index,
                "category": category,
                "sample_sha256": sample_hash,
                "tokens": len(token_ids),
                "base": compare(source_logits, np.load(base_path), args.top_k),
            }
            for name, routers in variants.items():
                logits = score(args.source_dir, token_ids, retained, routers=routers)
                case[name] = compare(source_logits, logits, args.top_k)
                operation_log.write(
                    f"variant-done index={selected_index} variant={name} "
                    f"kl={case[name]['kl_baseline_candidate']:.9g}"
                )
            report["cases"].append(case)
            atomic_json(report_path, report)
            operation_log.write(f"case-done index={selected_index} category={category}")

        require(len(report["cases"]) == len(selected), "ablation cases are incomplete")
        report["summary"] = {"base": summarize(report["cases"], "base")}
        report["gates"] = {}
        for name in variants:
            report["summary"][name] = summarize(report["cases"], name)
            report["gates"][name] = gate(report["cases"], report["cases"], name)
        report["status"] = "complete"
        atomic_json(report_path, report)
        operation_log.write(f"run-complete report_sha256={sha256_file(report_path)}")
        print(json.dumps({"summary": report["summary"], "gates": report["gates"]}, indent=2))
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        operation_log.write(f"run-failed error={exc}")
        print(f"nemotron Router KD ablation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

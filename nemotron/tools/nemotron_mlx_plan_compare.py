#!/usr/bin/env python3
"""Resumably compare uniform and nonuniform plans on independent full logits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import corpus_samples, sha256_file
from nemotron_mlx_compare_logits import compare
from nemotron_mlx_stream_forward import StreamingForward, validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state


FORMAT = "nemotron-plan-logit-comparison-v1"


def score(source_dir: Path, token_ids: list[int], retained: dict[str, list[int]] | None) -> np.ndarray:
    runner = StreamingForward(source_dir, retained)
    logits = runner.forward_sequence(token_ids)
    mx.eval(logits)
    result = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
    del runner, logits
    gc.collect()
    mx.clear_cache()
    return result


def summarize(cases: list[dict], method: str) -> dict:
    metrics = [case[method] for case in cases]
    fields = ("relative_l2", "centered_relative_l2", "kl_baseline_candidate", "cosine")
    return {
        "cases": len(cases),
        "top1_matches": sum(item["baseline_top1"] == item["candidate_top1"] for item in metrics),
        "mean_top_k_overlap": float(np.mean([item["top_k_overlap"] for item in metrics])),
        **{f"mean_{field}": float(np.mean([item[field] for item in metrics])) for field in fields},
        **{f"max_{field}": max(item[field] for item in metrics) for field in fields},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--uniform-plan", required=True, type=Path)
    parser.add_argument("--nonuniform-plan", required=True, type=Path)
    parser.add_argument("--max-sample-tokens", type=int, default=24)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.max_sample_tokens > 0 and args.top_k > 0, "token and top-k limits must be positive")
        require(args.max_cases is None or args.max_cases > 0, "max cases must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        uniform_plan = load_json(args.uniform_plan)
        nonuniform_plan = load_json(args.nonuniform_plan)
        uniform = validate_virtual_plan(uniform_plan, config, source_state["revision"])
        nonuniform = validate_virtual_plan(nonuniform_plan, config, source_state["revision"])
        identity = {
            "format": FORMAT,
            "source_revision": source_state["revision"],
            "corpus_sha256": sha256_file(args.corpus),
            "uniform_plan_sha256": sha256_file(args.uniform_plan),
            "nonuniform_plan_sha256": sha256_file(args.nonuniform_plan),
            "max_sample_tokens": args.max_sample_tokens,
            "top_k": args.top_k,
        }
        if args.output.exists():
            report = load_json(args.output)
            for key, value in identity.items():
                require(report.get(key) == value, f"comparison identity mismatch: {key}")
        else:
            report = {**identity, "status": "running", "cases": []}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        completed = {case["sample_sha256"] for case in report["cases"]}
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        processed = 0
        for category, sample in corpus_samples(args.corpus):
            sample_hash = hashlib.sha256(sample.encode()).hexdigest()
            if sample_hash in completed:
                continue
            if args.max_cases is not None and processed >= args.max_cases:
                break
            token_ids = tokenizer.encode(sample, add_special_tokens=False)[: args.max_sample_tokens]
            require(token_ids, f"validation sample encoded to no tokens: {category}")
            operation_log.write(
                f"case-start category={category} tokens={len(token_ids)} sample_sha256={sample_hash}"
            )
            started = time.perf_counter()
            baseline_logits = score(args.source_dir, token_ids, None)
            uniform_logits = score(args.source_dir, token_ids, uniform)
            nonuniform_logits = score(args.source_dir, token_ids, nonuniform)
            case = {
                "category": category,
                "sample_sha256": sample_hash,
                "tokens": len(token_ids),
                "uniform": compare(baseline_logits, uniform_logits, args.top_k),
                "nonuniform": compare(baseline_logits, nonuniform_logits, args.top_k),
            }
            report["cases"].append(case)
            report["summary"] = {
                "uniform": summarize(report["cases"], "uniform"),
                "nonuniform": summarize(report["cases"], "nonuniform"),
            }
            atomic_json(args.output, report)
            processed += 1
            operation_log.write(
                f"case-done category={category} elapsed={time.perf_counter() - started:.2f}s "
                f"uniform_kl={case['uniform']['kl_baseline_candidate']:.6g} "
                f"nonuniform_kl={case['nonuniform']['kl_baseline_candidate']:.6g}"
            )
        total_cases = len(corpus_samples(args.corpus))
        if len(report["cases"]) == total_cases:
            report["status"] = "complete"
            atomic_json(args.output, report)
        operation_log.write(
            f"run-stop status={report['status']} completed={len(report['cases'])} total={total_cases}"
        )
        print(
            f"plan-compare status={report['status']} completed={len(report['cases'])}/{total_cases} "
            f"output={args.output} sha256={sha256_file(args.output)}"
        )
        if report["cases"]:
            print(report["summary"])
        return 0
    except (MetadataError, OSError, ValueError, IndexError, KeyError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron plan comparison error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

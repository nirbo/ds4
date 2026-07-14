#!/usr/bin/env python3
"""Screen reduced routed-expert top-k values with low-memory full logits."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import corpus_samples
from nemotron_mlx_compare_logits import compare
from nemotron_mlx_stream_forward import StreamingForward
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-expert-topk-sweep-v1"


def parse_top_ks(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k values must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result) or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("top-k values must be positive and unique")
    return result


def aggregate(metrics: list[dict]) -> dict:
    require(metrics, "top-k sweep has no metrics")
    return {
        "positions": len(metrics),
        "top1_equal": sum(
            item["baseline_top1"] == item["candidate_top1"] for item in metrics
        ),
        "mean_kl": statistics.fmean(item["kl_baseline_candidate"] for item in metrics),
        "worst_kl": max(item["kl_baseline_candidate"] for item in metrics),
        "mean_centered_relative_l2": statistics.fmean(
            item["centered_relative_l2"] for item in metrics
        ),
        "worst_centered_relative_l2": max(
            item["centered_relative_l2"] for item in metrics
        ),
        "worst_baseline_top1_rank": max(
            item["candidate_rank_of_baseline_top1"] for item in metrics
        ),
        "mean_top64_overlap": statistics.fmean(
            item["top_k_overlap"] for item in metrics
        ),
    }


def run_cases(
    model_dir: Path,
    token_cases: list[tuple[str, list[int]]],
    expert_top_k: int,
) -> tuple[list[np.ndarray], float]:
    runner = StreamingForward(model_dir, expert_top_k=expert_top_k)
    logits = []
    started = time.perf_counter()
    for _, token_ids in token_cases:
        runner.caches.clear()
        runner.routing.clear()
        output = runner.forward_sequence(token_ids)
        logits.append(np.asarray(output.reshape(-1), dtype=np.float32))
    elapsed = time.perf_counter() - started
    del runner
    mx.clear_cache()
    return logits, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--top-ks", type=parse_top_ks, default=parse_top_ks("20,18,16,14,12"))
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-sample-tokens", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.max_cases > 0 and args.max_sample_tokens > 0, "invalid sweep limits")
        config = load_json(args.model_dir / "config.json")
        pack_report_path = args.model_dir / "nemotron_mlx_pack_report.json"
        pack_report = load_json(pack_report_path)
        require(
            pack_report.get("format") == "nemotron-mlx-runtime-v1"
            and pack_report.get("status") == "complete",
            "top-k sweep requires a complete packed runtime",
        )
        native_top_k = config["num_experts_per_tok"]
        require(
            all(top_k < native_top_k for top_k in args.top_ks),
            f"candidate top-k values must be below native top-{native_top_k}",
        )
        samples = corpus_samples(args.corpus)[: args.max_cases]
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        token_cases = []
        for category, sample in samples:
            token_ids = tokenizer.encode(sample, add_special_tokens=False)[: args.max_sample_tokens]
            require(token_ids, f"corpus category {category} encoded to no tokens")
            token_cases.append((category, token_ids))

        tool_sha256 = sha256_file(Path(__file__))
        identity = {
            "format": FORMAT,
            "source_revision": pack_report["source_revision"],
            "pack_report_sha256": sha256_file(pack_report_path),
            "prune_plan_sha256": pack_report["plan_sha256"],
            "model_config_sha256": sha256_file(args.model_dir / "config.json"),
            "model_index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
            "corpus_sha256": sha256_file(args.corpus),
            "tool_sha256": tool_sha256,
            "native_top_k": native_top_k,
            "candidate_top_ks": args.top_ks,
            "max_cases": args.max_cases,
            "max_sample_tokens": args.max_sample_tokens,
            "cases": [
                {
                    "category": category,
                    "tokens": len(token_ids),
                    "token_ids_sha256": hashlib.sha256(
                        np.asarray(token_ids, dtype=np.uint32).tobytes()
                    ).hexdigest(),
                }
                for category, token_ids in token_cases
            ],
        }
        report = {**identity, "status": "running", "results": {}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        operation_log.write(
            f"topk-sweep-start native={native_top_k} candidates={args.top_ks} "
            f"cases={len(token_cases)} tokens={sum(len(item[1]) for item in token_cases)}"
        )

        baseline_logits, baseline_seconds = run_cases(
            args.model_dir, token_cases, native_top_k
        )
        report["baseline_seconds"] = baseline_seconds
        atomic_json(args.output, report)
        operation_log.write(f"topk-baseline-done elapsed={baseline_seconds:.3f}s")

        for top_k in args.top_ks:
            candidate_logits, elapsed = run_cases(args.model_dir, token_cases, top_k)
            case_metrics = [
                compare(baseline, candidate, 64)
                for baseline, candidate in zip(baseline_logits, candidate_logits)
            ]
            summary = aggregate(case_metrics)
            report["results"][str(top_k)] = {
                "elapsed_seconds": elapsed,
                "summary": summary,
                "cases": [
                    {"category": category, **metrics}
                    for (category, _), metrics in zip(token_cases, case_metrics)
                ],
            }
            atomic_json(args.output, report)
            operation_log.write(
                f"topk-candidate-done top_k={top_k} elapsed={elapsed:.3f}s "
                f"top1={summary['top1_equal']}/{summary['positions']} "
                f"mean_kl={summary['mean_kl']:.6g} worst_kl={summary['worst_kl']:.6g}"
            )

        report["status"] = "complete"
        atomic_json(args.output, report)
        operation_log.write(
            f"topk-sweep-complete output={args.output} sha256={sha256_file(args.output)}"
        )
        print(f"topk-sweep output={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"topk-sweep-failed error={exc}")
        print(f"nemotron top-k sweep error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

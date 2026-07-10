#!/usr/bin/env python3
"""Evaluate conservative exact-winner certificates for an NVFP4 target head."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_mtp import QuantizedMTPHead
from nemotron_prune_materialize import sha256_file


FORMAT = "nemotron-head-certificate-v1"
TRACE_FORMAT = "nemotron-mtp-target-trace-v1"


def parse_counts(value: str) -> list[int]:
    try:
        counts = sorted({int(item) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate counts must be comma-separated integers") from exc
    if not counts or counts[0] <= 0:
        raise argparse.ArgumentTypeError("candidate counts must be positive")
    return counts


def evaluate_certificates(
    quant_logits: np.ndarray,
    exact_logits: np.ndarray,
    error_bounds: np.ndarray,
    counts: list[int],
    safety_epsilon: float,
) -> dict[str, dict[str, float | int]]:
    require(
        quant_logits.shape == exact_logits.shape == error_bounds.shape
        and quant_logits.ndim == 2,
        "head certificate arrays must have one matching matrix shape",
    )
    require(np.isfinite(quant_logits).all(), "quantized logits contain non-finite values")
    require(np.isfinite(exact_logits).all(), "exact logits contain non-finite values")
    require(np.isfinite(error_bounds).all() and np.all(error_bounds >= 0), "invalid error bounds")
    rows, vocabulary = quant_logits.shape
    true_ids = np.argmax(exact_logits, axis=1)
    results = {}
    for count in counts:
        require(count <= vocabulary, "candidate count exceeds vocabulary")
        recalled = 0
        certified = 0
        for row in range(rows):
            candidates = np.argpartition(-quant_logits[row], count - 1)[:count]
            candidate_scores = exact_logits[row, candidates]
            winner = int(candidates[int(np.argmax(candidate_scores))])
            recalled += winner == int(true_ids[row])
            upper = quant_logits[row] + error_bounds[row] + safety_epsilon
            upper[candidates] = -np.inf
            certified += float(np.max(candidate_scores)) > float(np.max(upper))
        results[str(count)] = {
            "candidate_count": count,
            "recalled": recalled,
            "recall": recalled / rows,
            "certified": certified,
            "certification_rate": certified / rows,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument(
        "--candidate-counts",
        type=parse_counts,
        default=parse_counts("1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768"),
    )
    parser.add_argument("--rows-per-chunk", type=int, default=2048)
    parser.add_argument("--safety-epsilon", type=float, default=1e-4)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.rows_per_chunk > 0, "rows per chunk must be positive")
        require(args.safety_epsilon >= 0, "safety epsilon cannot be negative")
        head_report_path = args.head_dir / "nemotron_mtp_head_report.json"
        head_report = load_json(head_report_path)
        revision = head_report.get("source_revision")
        require(isinstance(revision, str), "quantized head report has no revision")
        head = QuantizedMTPHead(args.head_dir, revision)
        index = load_json(args.model_dir / "model.safetensors.index.json")
        shard_name = index.get("weight_map", {}).get("lm_head.weight")
        require(isinstance(shard_name, str), "packed runtime has no BF16 target head")
        shard = args.model_dir / shard_name
        exact_weight = mx.load(str(shard))["lm_head.weight"]
        require(
            exact_weight.dtype == mx.bfloat16
            and exact_weight.shape == head.shape,
            "BF16/NVFP4 target head shape mismatch",
        )
        arrays, metadata = mx.load(str(args.trace), return_metadata=True)
        require(metadata.get("format") == TRACE_FORMAT, "unsupported target trace")
        scored_rows = [index for index, value in enumerate(arrays["scored"].tolist()) if value]
        require(scored_rows, "target trace has no scored rows")
        hidden = arrays["target_hidden"][mx.array(scored_rows)].astype(mx.float32)
        require(hidden.shape[1] == exact_weight.shape[1], "target trace hidden width mismatch")

        started = time.perf_counter()
        group_errors = []
        for start in range(0, exact_weight.shape[0], args.rows_per_chunk):
            end = min(start + args.rows_per_chunk, exact_weight.shape[0])
            restored = mx.dequantize(
                head.weight[start:end],
                head.scales[start:end],
                head.biases[start:end] if head.biases is not None else None,
                group_size=head.group_size,
                bits=head.bits,
                mode=head.mode,
                dtype=mx.float32,
            )
            difference = (restored - exact_weight[start:end].astype(mx.float32)).reshape(
                end - start,
                -1,
                head.group_size,
            )
            group_errors.append(mx.sqrt(mx.sum(mx.square(difference), axis=2)))
        group_errors = mx.concatenate(group_errors)
        hidden_groups = mx.sqrt(
            mx.sum(
                mx.square(hidden.reshape(hidden.shape[0], -1, head.group_size)),
                axis=2,
            )
        )
        quant_logits = head(hidden)
        exact_logits = mx.matmul(hidden, exact_weight.T.astype(mx.float32))
        error_bounds = mx.matmul(hidden_groups, group_errors.T)
        mx.eval(quant_logits, exact_logits, error_bounds)
        results = evaluate_certificates(
            np.asarray(quant_logits, dtype=np.float32),
            np.asarray(exact_logits, dtype=np.float32),
            np.asarray(error_bounds, dtype=np.float32),
            args.candidate_counts,
            args.safety_epsilon,
        )
        for result in results.values():
            result["candidate_bf16_mib"] = (
                result["candidate_count"] * exact_weight.shape[1] * 2 / 2**20
            )
        report = {
            "format": FORMAT,
            "source_revision": revision,
            "model_dir": str(args.model_dir.resolve()),
            "model_index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
            "head_report_sha256": sha256_file(head_report_path),
            "head_artifact_sha256": head_report.get("artifact_sha256"),
            "trace": str(args.trace.resolve()),
            "trace_sha256": sha256_file(args.trace),
            "rows": len(scored_rows),
            "vocabulary": exact_weight.shape[0],
            "hidden_size": exact_weight.shape[1],
            "bound": "sum_group_l2_error_times_hidden_l2",
            "group_size": head.group_size,
            "safety_epsilon": args.safety_epsilon,
            "results": results,
            "elapsed_seconds": time.perf_counter() - started,
            "active_gib": mx.get_active_memory() / 2**30,
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + ".part")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.report)
        print("head-certificate-result " + json.dumps(report, separators=(",", ":")))
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron head certificate error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

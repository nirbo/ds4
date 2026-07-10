#!/usr/bin/env python3
"""Compare full-vocabulary Nemotron baseline and candidate logits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from nemotron_metadata import MetadataError, require


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    return shifted - math.log(float(np.exp(shifted).sum()))


def compare(baseline: np.ndarray, candidate: np.ndarray, top_k: int) -> dict:
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    require(baseline.shape == candidate.shape and baseline.size > 0, "logit shape mismatch")
    require(np.isfinite(baseline).all() and np.isfinite(candidate).all(), "logits contain non-finite values")
    difference = candidate - baseline
    centered_baseline = baseline - baseline.mean()
    centered_candidate = candidate - candidate.mean()
    centered_difference = centered_candidate - centered_baseline
    k = min(top_k, baseline.size)
    baseline_top = np.argpartition(-baseline, k - 1)[:k]
    candidate_top = np.argpartition(-candidate, k - 1)[:k]
    baseline_log_probs = log_softmax(baseline)
    candidate_log_probs = log_softmax(candidate)
    baseline_probs = np.exp(baseline_log_probs)
    baseline_top1 = int(np.argmax(baseline))
    candidate_order = np.argsort(-candidate)
    candidate_rank_of_baseline_top1 = int(np.flatnonzero(candidate_order == baseline_top1)[0]) + 1
    return {
        "vocab": int(baseline.size),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(baseline), 1e-30)),
        "centered_relative_l2": float(
            np.linalg.norm(centered_difference) / max(np.linalg.norm(centered_baseline), 1e-30)
        ),
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(
            np.dot(centered_baseline, centered_candidate)
            / max(np.linalg.norm(centered_baseline) * np.linalg.norm(centered_candidate), 1e-30)
        ),
        "kl_baseline_candidate": float(np.sum(baseline_probs * (baseline_log_probs - candidate_log_probs))),
        "baseline_top1": baseline_top1,
        "candidate_top1": int(np.argmax(candidate)),
        "candidate_rank_of_baseline_top1": candidate_rank_of_baseline_top1,
        "top_k": k,
        "top_k_overlap": int(len(set(baseline_top.tolist()) & set(candidate_top.tolist()))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.top_k > 0, "top-k must be positive")
        report = compare(np.load(args.baseline), np.load(args.candidate), args.top_k)
        encoded = json.dumps(report, indent=2) + "\n"
        print(encoded, end="")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".part")
            temporary.write_text(encoded)
            temporary.replace(args.output)
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron logit comparison error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

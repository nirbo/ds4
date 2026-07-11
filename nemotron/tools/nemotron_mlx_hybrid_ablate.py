#!/usr/bin/env python3
"""Ablate hybrid width layers independently on full-vocabulary logits."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import corpus_samples, sha256_file
from nemotron_mlx_compare_logits import compare
from nemotron_mlx_plan_compare import score
from nemotron_mlx_stream_forward import validate_virtual_hybrid_plan, validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state


FORMAT = "nemotron-hybrid-width-ablation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--hybrid-plan", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--max-sample-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.case >= 0 and args.max_sample_tokens > 0 and args.top_k > 0, "invalid limits")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        base_plan = load_json(args.base_plan)
        hybrid_plan = load_json(args.hybrid_plan)
        base = validate_virtual_plan(base_plan, config, source_state["revision"])
        _, width = validate_virtual_hybrid_plan(hybrid_plan, config, source_state["revision"])
        samples = corpus_samples(args.corpus)
        require(args.case < len(samples), "case index exceeds corpus")
        category, sample = samples[args.case]
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        token_ids = tokenizer.encode(sample, add_special_tokens=False)[: args.max_sample_tokens]
        identity = {
            "format": FORMAT,
            "source_revision": source_state["revision"],
            "base_plan_sha256": sha256_file(args.base_plan),
            "hybrid_plan_sha256": sha256_file(args.hybrid_plan),
            "corpus_sha256": sha256_file(args.corpus),
            "case": args.case,
            "category": category,
            "sample_sha256": hashlib.sha256(sample.encode()).hexdigest(),
            "tokens": len(token_ids),
            "top_k": args.top_k,
            "width_layers": [int(layer) for layer in width],
        }
        report = {**identity, "status": "running", "results": {}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        operation_log.write(f"ablation-start category={category} tokens={len(token_ids)}")
        baseline_logits = score(args.source_dir, token_ids, None)
        base_logits = score(args.source_dir, token_ids, base)
        report["base"] = compare(baseline_logits, base_logits, args.top_k)
        atomic_json(args.output, report)
        for key in width:
            retained = dict(base)
            retained.pop(key)
            candidate_logits = score(args.source_dir, token_ids, retained, {key: width[key]})
            metrics = compare(baseline_logits, candidate_logits, args.top_k)
            report["results"][key] = metrics
            atomic_json(args.output, report)
            operation_log.write(
                f"ablation-layer-done layer={key} kl={metrics['kl_baseline_candidate']:.6g} "
                f"top1={metrics['candidate_top1']}"
            )
        report["status"] = "complete"
        atomic_json(args.output, report)
        operation_log.write("ablation-complete")
        print(f"hybrid-ablation output={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"ablation-failed error={exc}")
        print(f"nemotron hybrid ablation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

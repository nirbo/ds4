#!/usr/bin/env python3
"""Measure recursive packed-MTP draft acceptance on a contiguous target trace."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import (
    NemotronMTPSidecar,
    alternate_mtp_head_uses_shared_target,
    load_indexed_tensors,
)
from nemotron_prune_materialize import sha256_file


FORMAT = "nemotron-mtp-recursive-acceptance-v1"
TRACE_FORMAT = "nemotron-mtp-target-trace-v1"


def percentile(values: list[float], fraction: float) -> float:
    require(values, "cannot calculate percentile of no values")
    return sorted(values)[max(0, int(len(values) * fraction + 0.999) - 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", type=Path)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(1 <= args.max_depth <= 8, "recursive MTP depth must be between 1 and 8")
        arrays, metadata = mx.load(str(args.trace), return_metadata=True)
        require(metadata.get("format") == TRACE_FORMAT, "unsupported MTP target trace")
        required = {
            "target_hidden",
            "accepted_token_ids",
            "expected_token_ids",
            "prompt_indices",
            "scored",
        }
        require(required <= set(arrays), "MTP target trace is incomplete")
        rows = arrays["target_hidden"].shape[0]
        require(all(arrays[name].shape[0] == rows for name in required), "MTP trace row mismatch")

        needs_full_head = args.mtp_lm_head is None or alternate_mtp_head_uses_shared_target(
            args.mtp_lm_head
        )
        globals_ = load_indexed_tensors(
            args.source_dir,
            {"backbone.embeddings.weight"}
            | ({"lm_head.weight"} if needs_full_head else set()),
        )
        model = NemotronMTPSidecar(
            args.sidecar,
            globals_["backbone.embeddings.weight"],
            ModelOptBF16Linear(globals_["lm_head.weight"]) if needs_full_head else None,
            alternate_lm_head=args.mtp_lm_head,
        )
        sidecar_report = args.sidecar / "nemotron_mtp_pack_report.json"
        require(sidecar_report.is_file(), "MTP sidecar report is missing")
        alternate_report = None
        if args.mtp_lm_head is not None:
            alternate_reports = [
                path
                for name in (
                    "nemotron_mtp_head_report.json",
                    "nemotron_mtp_vocab_head_report.json",
                )
                if (path := args.mtp_lm_head / name).is_file()
            ]
            require(len(alternate_reports) == 1, "alternate MTP head report is ambiguous")
            alternate_report = alternate_reports[0]
        prompt_indices = arrays["prompt_indices"].tolist()
        scored = arrays["scored"].tolist()
        attempts = Counter()
        matches = Counter()
        chain_lengths = Counter()
        step_latencies: dict[int, list[float]] = {
            depth: [] for depth in range(1, args.max_depth + 1)
        }
        cycles = 0
        accepted_drafts = 0
        started = time.perf_counter()
        for row in range(rows):
            if not scored[row]:
                continue
            prompt = prompt_indices[row]
            hidden = arrays["target_hidden"][row]
            accepted_token = int(arrays["accepted_token_ids"][row])
            accepted = 0
            for depth in range(1, args.max_depth + 1):
                expected_row = row + depth - 1
                if expected_row >= rows or prompt_indices[expected_row] != prompt:
                    break
                if depth > 1:
                    require(
                        int(arrays["accepted_token_ids"][expected_row])
                        == int(arrays["expected_token_ids"][expected_row - 1]),
                        "MTP target trace is not contiguous",
                    )
                call_started = time.perf_counter()
                logits, next_hidden, _, _ = model.draft_step(hidden, accepted_token)
                mx.eval(logits, next_hidden)
                mx.synchronize()
                step_latencies[depth].append((time.perf_counter() - call_started) * 1000)
                prediction = model.argmax_token(logits)
                expected = int(arrays["expected_token_ids"][expected_row])
                attempts[depth] += 1
                if prediction != expected:
                    break
                matches[depth] += 1
                accepted += 1
                hidden = next_hidden
                accepted_token = prediction
            cycles += 1
            accepted_drafts += accepted
            chain_lengths[accepted] += 1

        require(cycles > 0 and attempts[1] == cycles, "recursive MTP benchmark found no cycles")
        depth_results = {}
        for depth in range(1, args.max_depth + 1):
            values = step_latencies[depth]
            depth_results[str(depth)] = {
                "attempts": attempts[depth],
                "matches": matches[depth],
                "conditional_acceptance": (
                    matches[depth] / attempts[depth] if attempts[depth] else None
                ),
                "median_ms": statistics.median(values) if values else None,
                "p95_ms": percentile(values, 0.95) if values else None,
            }
        report = {
            "format": FORMAT,
            "trace": str(args.trace.resolve()),
            "trace_sha256": sha256_file(args.trace),
            "sidecar": str(args.sidecar.resolve()),
            "sidecar_report_sha256": sha256_file(sidecar_report),
            "mtp_lm_head": (
                str(args.mtp_lm_head.resolve()) if args.mtp_lm_head is not None else None
            ),
            "mtp_lm_head_report_sha256": (
                sha256_file(alternate_report) if alternate_report is not None else None
            ),
            "source_revision": model.config["nemotron_mtp_runtime"]["source_revision"],
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "max_depth": args.max_depth,
            "cycles": cycles,
            "accepted_drafts": accepted_drafts,
            "accepted_drafts_per_cycle": accepted_drafts / cycles,
            "chain_lengths": {
                str(length): chain_lengths[length]
                for length in range(args.max_depth + 1)
            },
            "depths": depth_results,
            "elapsed_seconds": time.perf_counter() - started,
            "active_gib": mx.get_active_memory() / 2**30,
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
        print("mtp-recursive-result " + json.dumps(report, separators=(",", ":")), flush=True)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_name(args.report.name + ".part")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temporary.replace(args.report)
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron recursive MTP benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

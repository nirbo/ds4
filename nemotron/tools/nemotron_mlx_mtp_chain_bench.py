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

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import (
    NemotronMTPReference,
    NemotronMTPSidecar,
    alternate_mtp_head_uses_shared_target,
    load_indexed_tensors,
)
from nemotron_mlx_mtp_norm_calibrate import (
    damped_final_norm,
    load_calibrated_final_norm,
)
from nemotron_mlx_mtp_predictor import load_capture
from nemotron_prune_materialize import sha256_file


FORMAT = "nemotron-mtp-recursive-acceptance-v1"
TRACE_FORMAT = "nemotron-mtp-target-trace-v1"
CAPTURE_FORMAT = "nemotron-mtp-teacher-capture-v2"
PLAN_FORMAT = "nemotron-mtp-expert-plan-v1"


def percentile(values: list[float], fraction: float) -> float:
    require(values, "cannot calculate percentile of no values")
    return sorted(values)[max(0, int(len(values) * fraction + 0.999) - 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--capture-dir", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--mtp-lm-head", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--prompt-parity", type=int, choices=(0, 1))
    parser.add_argument("--final-norm-override", type=Path)
    parser.add_argument("--final-norm-damping", type=float, default=1.0)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(1 <= args.max_depth <= 8, "recursive MTP depth must be between 1 and 8")
        require(
            (args.plan is None) == (args.budget is None),
            "recursive MTP plan and budget must be supplied together",
        )
        require(
            args.sidecar is None or args.plan is None,
            "packed sidecar and source subset plan are mutually exclusive",
        )
        require(
            args.sidecar is not None or args.mtp_lm_head is None,
            "alternate MTP head requires a packed sidecar",
        )
        require(
            args.final_norm_override is None or args.sidecar is not None,
            "MTP final norm override requires a packed sidecar",
        )
        require(
            args.final_norm_override is None or args.mtp_lm_head is not None,
            "MTP final norm override requires its reduced vocabulary head",
        )
        require(
            0.0 <= args.final_norm_damping <= 1.0,
            "MTP final norm damping must be between zero and one",
        )
        require(
            args.final_norm_override is not None or args.final_norm_damping == 1.0,
            "MTP final norm damping requires an override",
        )
        capture_state_sha256 = None
        if args.capture_dir is not None:
            arrays, capture_state = load_capture(args.capture_dir)
            require(
                capture_state.get("format") == CAPTURE_FORMAT,
                "unsupported MTP teacher capture",
            )
            capture_state_sha256 = sha256_file(args.capture_dir / "state.json")
            scored = [True] * arrays["target_hidden"].shape[0]
        else:
            arrays, metadata = mx.load(str(args.trace), return_metadata=True)
            require(metadata.get("format") == TRACE_FORMAT, "unsupported MTP target trace")
            scored = arrays["scored"].tolist()
        required = {
            "target_hidden",
            "accepted_token_ids",
            "expected_token_ids",
            "prompt_indices",
        }
        if args.trace is not None:
            required.add("scored")
        require(required <= set(arrays), "MTP target trace is incomplete")
        rows = arrays["target_hidden"].shape[0]
        require(all(arrays[name].shape[0] == rows for name in required), "MTP trace row mismatch")

        plan = None
        plan_sha256 = None
        retained_experts = None
        if args.plan is not None:
            plan = load_json(args.plan)
            require(plan.get("format") == PLAN_FORMAT, "unsupported MTP expert plan")
            retained_experts = plan.get("budgets", {}).get(str(args.budget))
            require(
                isinstance(retained_experts, list),
                f"MTP plan has no budget {args.budget}",
            )
            plan_sha256 = sha256_file(args.plan)
        sidecar_report = None
        if args.sidecar is not None:
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
            source_revision = model.config["nemotron_mtp_runtime"]["source_revision"]
            if args.final_norm_override is not None:
                calibrated_norm = load_calibrated_final_norm(
                    args.final_norm_override,
                    args.source_dir,
                    args.sidecar,
                    args.mtp_lm_head,
                )
                require(
                    calibrated_norm.shape == model.final_norm_weight.shape,
                    "calibrated MTP final norm shape mismatch",
                )
                model.final_norm_weight = damped_final_norm(
                    model.final_norm_weight,
                    calibrated_norm,
                    args.final_norm_damping,
                )
        else:
            model = NemotronMTPReference(args.source_dir, retained_experts)
            source_state = load_json(args.source_dir.parent / "source-nvfp4-state.json")
            source_revision = source_state.get("revision")
            require(isinstance(source_revision, str), "source state has no revision")
        if plan is not None and plan.get("source_revision") is not None:
            require(
                plan["source_revision"] == source_revision,
                "MTP plan source revision does not match the checkpoint",
            )
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
        attempts = Counter()
        matches = Counter()
        chain_lengths = Counter()
        step_latencies: dict[int, list[float]] = {
            depth: [] for depth in range(1, args.max_depth + 1)
        }
        route_counts = {
            depth: Counter() for depth in range(1, args.max_depth + 1)
        }
        route_score_mass = {
            depth: Counter() for depth in range(1, args.max_depth + 1)
        }
        cycles = 0
        accepted_drafts = 0
        started = time.perf_counter()
        for row in range(rows):
            if not scored[row] or (
                args.prompt_parity is not None
                and prompt_indices[row] % 2 != args.prompt_parity
            ):
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
                logits, next_hidden, indices, scores = model.draft_step(
                    hidden,
                    accepted_token,
                )
                require(logits is not None, "recursive MTP step produced no logits")
                mx.eval(logits, next_hidden, indices, scores)
                mx.synchronize()
                step_latencies[depth].append((time.perf_counter() - call_started) * 1000)
                routed = [
                    (int(index), float(score))
                    for index, score in zip(indices.tolist(), scores.tolist())
                ]
                route_counts[depth].update(index for index, _ in routed)
                route_score_mass[depth].update(dict(routed))
                prediction = (
                    model.argmax_token(logits)
                    if hasattr(model, "argmax_token")
                    else int(mx.argmax(logits))
                )
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
            "trace": str(args.trace.resolve()) if args.trace is not None else None,
            "trace_sha256": sha256_file(args.trace) if args.trace is not None else None,
            "capture_dir": (
                str(args.capture_dir.resolve()) if args.capture_dir is not None else None
            ),
            "capture_state_sha256": capture_state_sha256,
            "sidecar": str(args.sidecar.resolve()) if args.sidecar is not None else None,
            "sidecar_report_sha256": (
                sha256_file(sidecar_report) if sidecar_report is not None else None
            ),
            "plan": str(args.plan.resolve()) if args.plan is not None else None,
            "plan_sha256": plan_sha256,
            "budget": args.budget,
            "mtp_lm_head": (
                str(args.mtp_lm_head.resolve()) if args.mtp_lm_head is not None else None
            ),
            "mtp_lm_head_report_sha256": (
                sha256_file(alternate_report) if alternate_report is not None else None
            ),
            "source_revision": source_revision,
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "max_depth": args.max_depth,
            "prompt_parity": args.prompt_parity,
            "final_norm_override": (
                str(args.final_norm_override.resolve())
                if args.final_norm_override is not None
                else None
            ),
            "final_norm_override_report_sha256": (
                sha256_file(args.final_norm_override / "report.json")
                if args.final_norm_override is not None
                else None
            ),
            "final_norm_damping": args.final_norm_damping,
            "cycles": cycles,
            "accepted_drafts": accepted_drafts,
            "accepted_drafts_per_cycle": accepted_drafts / cycles,
            "chain_lengths": {
                str(length): chain_lengths[length]
                for length in range(args.max_depth + 1)
            },
            "depths": depth_results,
            "expert_counts_by_depth": {
                str(depth): {
                    str(expert): count
                    for expert, count in route_counts[depth].most_common()
                }
                for depth in range(1, args.max_depth + 1)
            },
            "expert_score_mass_by_depth": {
                str(depth): {
                    str(expert): route_score_mass[depth][expert]
                    for expert, _ in route_counts[depth].most_common()
                }
                for depth in range(1, args.max_depth + 1)
            },
            "elapsed_seconds": time.perf_counter() - started,
            "active_gib": mx.get_active_memory() / 2**30,
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
        summary = {
            key: report[key]
            for key in (
                "source_revision",
                "max_depth",
                "cycles",
                "accepted_drafts",
                "accepted_drafts_per_cycle",
                "chain_lengths",
                "depths",
                "elapsed_seconds",
                "active_gib",
                "peak_gib",
            )
        }
        summary["unique_experts_by_depth"] = {
            str(depth): len(route_counts[depth])
            for depth in range(1, args.max_depth + 1)
        }
        print("mtp-recursive-result " + json.dumps(summary, separators=(",", ":")), flush=True)
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

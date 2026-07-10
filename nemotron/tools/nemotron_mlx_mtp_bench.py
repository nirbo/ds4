#!/usr/bin/env python3
"""Capture target traces and measure the official Nemotron MTP head offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import (
    NemotronMTPReference,
    NemotronMTPSidecar,
    alternate_mtp_head_uses_shared_target,
    load_indexed_tensors,
    mtp_payload_estimate,
)
from nemotron_mlx_resident import ResidentModel, preflight


TRACE_FORMAT = "nemotron-mtp-target-trace-v1"
PLAN_FORMAT = "nemotron-mtp-expert-plan-v1"
DEFAULT_PROMPTS = [
    "Complete this Python function:\n\ndef binary_search(values, target):\n",
    "Write a Rust function that returns the longest common prefix of a list of strings.\n",
    "Find and fix the bug:\n\ndef average(xs):\n    return sum(xs) / len(xs)\n",
    "Implement an LRU cache in Python with O(1) get and put operations.\n",
    "Explain the race condition in this Go code and provide a corrected implementation:\n",
    "Write a SQL query that returns the second highest salary in each department.\n",
    "Design a TypeScript function that retries an async operation with exponential backoff.\n",
    "Given a directed graph, implement cycle detection and state its time complexity.\n",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [line for line in path.read_text().splitlines() if line.strip()]
    require(prompts, "prompt file contains no non-empty lines")
    return prompts


def parse_budgets(value: str) -> list[int]:
    try:
        budgets = sorted({int(item) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from exc
    if not budgets or budgets[0] < 22 or budgets[-1] > 512:
        raise argparse.ArgumentTypeError("MTP expert budgets must be between 22 and 512")
    return budgets


def append_trace_rows(
    hidden_rows: list[mx.array],
    sequence: list[int],
    prompt_tokens: int,
    prompt_index: int,
    output: dict[str, list],
) -> None:
    require(len(hidden_rows) == len(sequence), "target hidden/token sequence mismatch")
    require(prompt_tokens > 0 and len(sequence) >= prompt_tokens + 2, "target trace is too short")
    for position in range(len(sequence) - 2):
        output["hidden"].append(hidden_rows[position])
        output["accepted"].append(sequence[position + 1])
        output["expected"].append(sequence[position + 2])
        output["prompt_index"].append(prompt_index)
        output["scored"].append(int(position >= prompt_tokens - 1))


def capture_trace(args: argparse.Namespace) -> int:
    result = preflight(args.model_dir, args.margin_gib)
    print("mtp-trace-preflight " + json.dumps(result, separators=(",", ":")), flush=True)
    require(result["safe_to_attempt"], "Metal wired cap is too low for target trace capture")
    prompts = load_prompts(args.prompts_file)
    previous_limit = mx.set_wired_limit(result["required_bytes"])
    mx.set_cache_limit(256 * 2**20)
    try:
        started = time.perf_counter()
        model = ResidentModel(args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        empty_snapshot = model.snapshot()
        rows: dict[str, list] = {
            "hidden": [],
            "accepted": [],
            "expected": [],
            "prompt_index": [],
            "scored": [],
        }
        generations = []
        for prompt_index, prompt in enumerate(prompts):
            model.restore(empty_snapshot)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            require(prompt_ids, f"prompt {prompt_index} encoded to no tokens")
            sequence = list(prompt_ids)
            hidden_rows = []
            logits = None
            for token_id in prompt_ids:
                logits, hidden = model.forward(token_id)
                hidden_rows.append(hidden)
            require(logits is not None, "target prompt produced no logits")
            generated = []
            for _ in range(args.tokens_per_prompt + 1):
                token_id = int(mx.argmax(logits))
                generated.append(token_id)
                sequence.append(token_id)
                logits, hidden = model.forward(token_id)
                hidden_rows.append(hidden)
            append_trace_rows(hidden_rows, sequence, len(prompt_ids), prompt_index, rows)
            generations.append(generated)
            print(
                f"mtp-trace-prompt index={prompt_index} prompt_tokens={len(prompt_ids)} "
                f"generated_tokens={len(generated)}",
                flush=True,
            )

        arrays = {
            "target_hidden": mx.stack(rows["hidden"]).astype(mx.float32),
            "accepted_token_ids": mx.array(rows["accepted"], dtype=mx.int32),
            "expected_token_ids": mx.array(rows["expected"], dtype=mx.int32),
            "prompt_indices": mx.array(rows["prompt_index"], dtype=mx.int32),
            "scored": mx.array(rows["scored"], dtype=mx.int32),
        }
        mx.eval(*arrays.values())
        metadata = {
            "format": TRACE_FORMAT,
            "model_dir": str(args.model_dir.resolve()),
            "tokens_per_prompt": str(args.tokens_per_prompt),
            "prompts_json": json.dumps(prompts, separators=(",", ":")),
            "generations_json": json.dumps(generations, separators=(",", ":")),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.stem + ".part" + args.output.suffix)
        temporary.unlink(missing_ok=True)
        mx.save_safetensors(str(temporary), arrays, metadata=metadata)
        temporary.replace(args.output)
        print(
            f"mtp-trace-done path={args.output} rows={len(rows['scored'])} "
            f"scored={sum(rows['scored'])} bytes={args.output.stat().st_size} "
            f"sha256={sha256_file(args.output)} elapsed={time.perf_counter() - started:.3f}s "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
    finally:
        mx.set_wired_limit(previous_limit)
    return 0


def evaluate_trace(args: argparse.Namespace) -> int:
    arrays, metadata = mx.load(str(args.trace), return_metadata=True)
    require(metadata.get("format") == TRACE_FORMAT, "unsupported MTP trace format")
    required = {
        "target_hidden",
        "accepted_token_ids",
        "expected_token_ids",
        "prompt_indices",
        "scored",
    }
    require(required <= set(arrays), "MTP trace is incomplete")
    row_count = arrays["target_hidden"].shape[0]
    require(all(arrays[name].shape[0] == row_count for name in required), "MTP trace row mismatch")
    require(row_count > 0, "MTP trace is empty")

    retained_experts = None
    plan_sha256 = None
    if args.plan is not None:
        plan = load_json(args.plan)
        require(plan.get("format") == PLAN_FORMAT, "unsupported MTP expert plan")
        retained_experts = plan.get("budgets", {}).get(str(args.budget))
        require(isinstance(retained_experts, list), f"MTP plan has no budget {args.budget}")
        plan_sha256 = sha256_file(args.plan)

    mx.set_cache_limit(256 * 2**20)
    started = time.perf_counter()
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
            (
                ModelOptBF16Linear(globals_["lm_head.weight"])
                if needs_full_head
                else None
            ),
            alternate_lm_head=args.mtp_lm_head,
        )
    else:
        model = NemotronMTPReference(args.source_dir, retained_experts)
    load_seconds = time.perf_counter() - started
    accepted = 0
    top5_accepted = 0
    scored_rows = 0
    route_counts: Counter[int] = Counter()
    route_score_mass: Counter[int] = Counter()
    scored_route_counts: Counter[int] = Counter()
    scored_route_score_mass: Counter[int] = Counter()
    latencies = []
    prompt_results: dict[int, Counter[str]] = defaultdict(Counter)
    for row in range(row_count):
        prompt_index = int(arrays["prompt_indices"][row])
        scored = bool(int(arrays["scored"][row]))
        if not scored:
            continue
        call_started = time.perf_counter()
        if isinstance(model, NemotronMTPReference):
            logits, indices, scores = model(
                arrays["target_hidden"][row],
                int(arrays["accepted_token_ids"][row]),
                project_logits=True,
            )
        else:
            logits, indices, scores = model(
                arrays["target_hidden"][row],
                int(arrays["accepted_token_ids"][row]),
            )
        require(logits is not None, "scored MTP row produced no logits")
        mx.eval(logits, indices, scores)
        routed = [(int(index), float(score)) for index, score in zip(indices.tolist(), scores.tolist())]
        route_counts.update(index for index, _ in routed)
        route_score_mass.update(dict(routed))
        mx.synchronize()
        latencies.append(time.perf_counter() - call_started)
        expected = int(arrays["expected_token_ids"][row])
        prediction = (
            int(mx.argmax(logits))
            if isinstance(model, NemotronMTPReference)
            else model.argmax_token(logits)
        )
        top5 = (
            mx.argpartition(-logits, kth=4)[:5].tolist()
            if isinstance(model, NemotronMTPReference)
            else model.top_token_ids(logits, 5)
        )
        accepted += int(prediction == expected)
        top5_accepted += int(expected in top5)
        scored_rows += 1
        prompt_results[prompt_index].update(
            {
                "rows": 1,
                "top1": int(prediction == expected),
                "top5": int(expected in top5),
            }
        )
        scored_route_counts.update(index for index, _ in routed)
        scored_route_score_mass.update(dict(routed))
        if (
            (prediction != expected and not args.quiet_mismatches)
            or (args.row_log_every and scored_rows % args.row_log_every == 0)
        ):
            print(
                f"mtp-row row={row} prompt={prompt_index} predicted={prediction} expected={expected} "
                f"accepted={prediction == expected} ms={latencies[-1] * 1000:.3f}",
                flush=True,
            )

    require(scored_rows > 0, "MTP trace has no scored rows")
    ordered_experts = sorted(
        route_score_mass,
        key=lambda expert: (route_score_mass[expert], route_counts[expert]),
        reverse=True,
    )
    coverage = {}
    total_routes = sum(route_counts.values())
    total_score_mass = sum(route_score_mass.values())
    for budget in (32, 48, 64, 96, 128, 192, 256):
        retained = set(ordered_experts[:budget])
        covered = sum(count for expert, count in route_counts.items() if expert in retained)
        covered_score = sum(
            score for expert, score in route_score_mass.items() if expert in retained
        )
        coverage[str(budget)] = {
            "selection_coverage": covered / total_routes,
            "score_mass_coverage": covered_score / total_score_mass,
            "payload_gib": mtp_payload_estimate(model.config, budget) / 2**30,
        }
    report = {
        "format": "nemotron-mtp-acceptance-v1",
        "source_dir": str(args.source_dir.resolve()),
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256_file(args.trace),
        "plan": str(args.plan.resolve()) if args.plan is not None else None,
        "plan_sha256": plan_sha256,
        "sidecar": str(args.sidecar.resolve()) if args.sidecar is not None else None,
        "mtp_lm_head": (
            str(args.mtp_lm_head.resolve()) if args.mtp_lm_head is not None else None
        ),
        "retained_experts": len(model.retained_experts),
        "rows": row_count,
        "scored_rows": scored_rows,
        "top1_acceptance": accepted / scored_rows,
        "top5_acceptance": top5_accepted / scored_rows,
        "prompt_acceptance": {
            str(prompt): {
                "rows": values["rows"],
                "top1_acceptance": values["top1"] / values["rows"],
                "top5_acceptance": values["top5"] / values["rows"],
            }
            for prompt, values in sorted(prompt_results.items())
        },
        "load_seconds": load_seconds,
        "median_ms": statistics.median(latencies) * 1000,
        "p95_ms": sorted(latencies)[max(0, int(0.95 * len(latencies) + 0.999) - 1)] * 1000,
        "unique_routed_experts": len(route_counts),
        "expert_counts": {str(expert): route_counts[expert] for expert in ordered_experts},
        "expert_score_mass": {
            str(expert): route_score_mass[expert] for expert in ordered_experts
        },
        "scored_expert_counts": {
            str(expert): scored_route_counts[expert] for expert in ordered_experts
        },
        "scored_expert_score_mass": {
            str(expert): scored_route_score_mass[expert] for expert in ordered_experts
        },
        "budget_coverage": coverage,
        "active_gib": mx.get_active_memory() / 2**30,
        "peak_gib": mx.get_peak_memory() / 2**30,
    }
    summary = {
        key: report[key]
        for key in (
            "retained_experts",
            "scored_rows",
            "top1_acceptance",
            "top5_acceptance",
            "median_ms",
            "p95_ms",
            "unique_routed_experts",
            "active_gib",
            "peak_gib",
            "budget_coverage",
        )
    }
    print("mtp-result " + json.dumps(summary, separators=(",", ":")), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + ".part")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.report)
    return 0


def build_plan(args: argparse.Namespace) -> int:
    report = load_json(args.report)
    require(report.get("format") == "nemotron-mtp-acceptance-v1", "unsupported MTP report")
    score_mass = report.get("scored_expert_score_mass")
    counts = report.get("scored_expert_counts")
    require(isinstance(score_mass, dict) and isinstance(counts, dict), "MTP report has no routing evidence")
    ordered = sorted(
        (int(expert) for expert in score_mass),
        key=lambda expert: (score_mass[str(expert)], counts[str(expert)]),
        reverse=True,
    )
    require(len(ordered) >= max(args.budgets), "routing evidence does not cover requested budget")
    plan = {
        "format": PLAN_FORMAT,
        "source_report": str(args.report.resolve()),
        "source_report_sha256": sha256_file(args.report),
        "ranking": "decode-only-summed-router-score-mass-then-selection-count",
        "budgets": {str(budget): ordered[:budget] for budget in args.budgets},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".part")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        f"mtp-plan-done path={args.output} budgets={','.join(str(value) for value in args.budgets)} "
        f"sha256={sha256_file(args.output)}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    trace = subparsers.add_parser("capture", help="capture target hidden-state traces")
    trace.add_argument("--model-dir", required=True, type=Path)
    trace.add_argument("--output", required=True, type=Path)
    trace.add_argument("--prompts-file", type=Path)
    trace.add_argument("--tokens-per-prompt", type=int, default=8)
    trace.add_argument("--margin-gib", type=float, default=1.5)
    evaluate = subparsers.add_parser("evaluate", help="evaluate official MTP on a target trace")
    evaluate.add_argument("--source-dir", required=True, type=Path)
    evaluate.add_argument("--trace", required=True, type=Path)
    evaluate.add_argument("--report", type=Path)
    evaluate.add_argument("--row-log-every", type=int, default=1)
    evaluate.add_argument("--quiet-mismatches", action="store_true")
    evaluate.add_argument("--plan", type=Path)
    evaluate.add_argument("--budget", type=int)
    evaluate.add_argument("--sidecar", type=Path)
    evaluate.add_argument("--mtp-lm-head", type=Path)
    plan = subparsers.add_parser("plan", help="build exact expert-subset plans from routing evidence")
    plan.add_argument("--report", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--budgets", type=parse_budgets, default=parse_budgets("96,128,192,256"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "capture":
            require(args.tokens_per_prompt > 0, "tokens per prompt must be positive")
            return capture_trace(args)
        if args.command == "evaluate":
            require(args.mtp_lm_head is None or args.sidecar is not None, "MTP head requires a sidecar")
            require(args.row_log_every >= 0, "row log interval cannot be negative")
            require(
                (args.plan is None and args.budget is None)
                or (args.plan is not None and args.budget is not None),
                "MTP plan and budget must be supplied together",
            )
            require(
                args.sidecar is None or args.plan is None,
                "MTP sidecar and source subset plan are mutually exclusive",
            )
            return evaluate_trace(args)
        return build_plan(args)
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron MTP benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

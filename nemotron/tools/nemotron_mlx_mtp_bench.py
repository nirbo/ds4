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
from nemotron_prune_materialize import atomic_json


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


def sidecar_payload_path(sidecar: Path) -> Path:
    index = load_json(sidecar / "model.safetensors.index.json")
    shard_names = set(index.get("weight_map", {}).values())
    require(len(shard_names) == 1, f"MTP sidecar must occupy one shard: {sidecar}")
    return sidecar / next(iter(shard_names))


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    if path.suffix == ".json":
        value = load_json(path)
        if isinstance(value, list):
            require(all(isinstance(prompt, str) and prompt.strip() for prompt in value), "prompt JSON list is invalid")
            return value
        require(isinstance(value, dict) and value, "prompt JSON must be a list or object")
        categories = []
        for name in sorted(value):
            prompts = value[name]
            require(
                isinstance(prompts, list)
                and all(isinstance(prompt, str) and prompt.strip() for prompt in prompts),
                f"prompt JSON category is invalid: {name}",
            )
            categories.append(prompts)
        interleaved = []
        for index in range(max(len(prompts) for prompts in categories)):
            interleaved.extend(
                prompts[index] for prompts in categories if index < len(prompts)
            )
        return interleaved
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
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
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
    scored_row_results = []
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
        scored_row_results.append(
            {
                "row": row,
                "prompt_index": prompt_index,
                "expected_token_id": expected,
                "predicted_token_id": prediction,
                "top5_token_ids": [int(token) for token in top5],
                "routed_expert_ids": [index for index, _ in routed],
                "route_scores": [score for _, score in routed],
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
        "sidecar_sha256": sha256_file(sidecar_payload_path(args.sidecar))
        if args.sidecar is not None
        else None,
        "mtp_lm_head": (
            str(args.mtp_lm_head.resolve()) if args.mtp_lm_head is not None else None
        ),
        "retained_experts": len(model.retained_experts),
        "rows": row_count,
        "scored_rows": scored_rows,
        "top1_acceptance": accepted / scored_rows,
        "top5_acceptance": top5_accepted / scored_rows,
        "scored_row_results": scored_row_results,
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


def compare_sidecars(args: argparse.Namespace) -> int:
    arrays, metadata = mx.load(str(args.trace), return_metadata=True)
    require(metadata.get("format") == TRACE_FORMAT, "unsupported MTP trace format")
    required = {"target_hidden", "accepted_token_ids", "scored"}
    require(required <= set(arrays), "MTP trace is incomplete")
    globals_ = load_indexed_tensors(
        args.source_dir,
        {"backbone.embeddings.weight", "lm_head.weight"},
    )
    model_a = NemotronMTPSidecar(
        args.sidecar_a,
        globals_["backbone.embeddings.weight"],
        ModelOptBF16Linear(globals_["lm_head.weight"]),
        use_mixed_metal=args.mixed_metal_a,
    )
    model_b = NemotronMTPSidecar(
        args.sidecar_b,
        globals_["backbone.embeddings.weight"],
        ModelOptBF16Linear(globals_["lm_head.weight"]),
        use_mixed_metal=args.mixed_metal_b,
    )
    require(model_a.retained_experts == model_b.retained_experts, "MTP sidecar mappings differ")
    rows = 0
    exact_rows = 0
    top1_equal = 0
    top5_equal = 0
    error2 = 0.0
    reference2 = 0.0
    max_abs = 0.0
    for row in range(arrays["target_hidden"].shape[0]):
        if not bool(int(arrays["scored"][row])):
            continue
        logits_a, indices_a, scores_a = model_a(
            arrays["target_hidden"][row],
            int(arrays["accepted_token_ids"][row]),
        )
        logits_b, indices_b, scores_b = model_b(
            arrays["target_hidden"][row],
            int(arrays["accepted_token_ids"][row]),
        )
        difference = logits_b.astype(mx.float32) - logits_a.astype(mx.float32)
        values = {
            "error2": mx.sum(mx.square(difference)),
            "reference2": mx.sum(mx.square(logits_a.astype(mx.float32))),
            "max_abs": mx.max(mx.abs(difference)),
            "exact": mx.array_equal(logits_a, logits_b),
        }
        mx.eval(*values.values(), indices_a, indices_b, scores_a, scores_b)
        require(
            bool(mx.array_equal(indices_a, indices_b))
            and bool(mx.array_equal(scores_a, scores_b)),
            f"MTP sidecar routes differ at trace row {row}",
        )
        exact_rows += int(values["exact"])
        error2 += float(values["error2"])
        reference2 += float(values["reference2"])
        max_abs = max(max_abs, float(values["max_abs"]))
        top1_equal += int(model_a.argmax_token(logits_a) == model_b.argmax_token(logits_b))
        top5_equal += int(
            sorted(model_a.top_token_ids(logits_a, 5))
            == sorted(model_b.top_token_ids(logits_b, 5))
        )
        rows += 1
        if args.max_rows and rows >= args.max_rows:
            break
    require(rows > 0, "MTP comparison trace has no scored rows")
    report = {
        "format": "nemotron-mtp-sidecar-parity-v1",
        "source_dir": str(args.source_dir.resolve()),
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256_file(args.trace),
        "sidecar_a": str(args.sidecar_a.resolve()),
        "sidecar_a_sha256": sha256_file(sidecar_payload_path(args.sidecar_a)),
        "sidecar_b": str(args.sidecar_b.resolve()),
        "sidecar_b_sha256": sha256_file(sidecar_payload_path(args.sidecar_b)),
        "mixed_metal_a": args.mixed_metal_a,
        "mixed_metal_b": args.mixed_metal_b,
        "rows": rows,
        "exact_rows": exact_rows,
        "top1_equal_rows": top1_equal,
        "top5_equal_rows": top5_equal,
        "relative_l2": (error2 / max(reference2, 1e-30)) ** 0.5,
        "max_abs": max_abs,
    }
    atomic_json(args.report, report)
    print("mtp-parity " + json.dumps(report, separators=(",", ":")), flush=True)
    return 0


def build_plan(args: argparse.Namespace) -> int:
    report = load_json(args.report)
    require(report.get("format") == "nemotron-mtp-acceptance-v1", "unsupported MTP report")
    score_mass = report.get("scored_expert_score_mass")
    counts = report.get("scored_expert_counts")
    require(isinstance(score_mass, dict) and isinstance(counts, dict), "MTP report has no routing evidence")
    source_dir_value = report.get("source_dir")
    require(isinstance(source_dir_value, str) and source_dir_value, "MTP report has no source directory")
    source_config_path = Path(source_dir_value) / "config.json"
    source_config = load_json(source_config_path)
    declared_experts = source_config.get("n_routed_experts")
    require(isinstance(declared_experts, int) and declared_experts > 0, "invalid source MTP expert count")
    require(set(score_mass) == set(counts), "MTP route score/count expert sets differ")
    observed = {int(expert) for expert in score_mass}
    require(
        len(observed) == len(score_mass)
        and all(0 <= expert < declared_experts for expert in observed),
        "MTP report contains an invalid expert id",
    )
    ordered = sorted(
        observed,
        key=lambda expert: (score_mass[str(expert)], counts[str(expert)]),
        reverse=True,
    )
    ordered.extend(expert for expert in range(declared_experts) if expert not in observed)
    require(max(args.budgets) <= declared_experts, "requested budget exceeds source expert count")
    plan = {
        "format": PLAN_FORMAT,
        "source_report": str(args.report.resolve()),
        "source_report_sha256": sha256_file(args.report),
        "source_config": str(source_config_path.resolve()),
        "source_config_sha256": sha256_file(source_config_path),
        "declared_experts": declared_experts,
        "observed_experts": len(observed),
        "ranking": (
            "decode-only-summed-router-score-mass-then-selection-count;"
            "unobserved-experts-ascending"
        ),
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
    trace.add_argument("--max-prompts", type=int)
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
    compare = subparsers.add_parser("compare", help="compare full logits from two MTP sidecars")
    compare.add_argument("--source-dir", required=True, type=Path)
    compare.add_argument("--trace", required=True, type=Path)
    compare.add_argument("--sidecar-a", required=True, type=Path)
    compare.add_argument("--sidecar-b", required=True, type=Path)
    compare.add_argument("--report", required=True, type=Path)
    compare.add_argument("--max-rows", type=int, default=0)
    compare.add_argument("--mixed-metal-a", action="store_true")
    compare.add_argument("--mixed-metal-b", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "capture":
            require(args.tokens_per_prompt > 0, "tokens per prompt must be positive")
            require(
                args.max_prompts is None or args.max_prompts > 0,
                "maximum prompt count must be positive",
            )
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
        if args.command == "compare":
            require(args.max_rows >= 0, "maximum comparison rows cannot be negative")
            return compare_sidecars(args)
        return build_plan(args)
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron MTP benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

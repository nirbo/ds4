#!/usr/bin/env python3
"""Exact adaptive MTP and lookup generation for resident Nemotron."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_resident import ResidentModel, preflight
from nemotron_mlx_mtp_predictor import LearnedMTPPredictor, validated_predictor_report
from nemotron_mlx_mtp_norm_calibrate import (
    damped_final_norm,
    load_calibrated_final_norm,
)
from nemotron_ngram_lookup import NGramLookup


def percentile(values: list[float], fraction: float) -> float:
    require(values, "cannot calculate a percentile of no values")
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def margin_outcome_bins(
    outcomes: list[tuple[float, bool]],
    boundaries: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> str:
    require(
        boundaries and tuple(sorted(set(boundaries))) == boundaries,
        "margin boundaries must be sorted and unique",
    )
    counts = [[0, 0] for _ in range(len(boundaries) + 1)]
    for margin, correct in outcomes:
        bucket = next(
            (index for index, boundary in enumerate(boundaries) if margin < boundary),
            len(boundaries),
        )
        counts[bucket][0] += int(correct)
        counts[bucket][1] += 1
    labels = [f"lt{boundary:g}" for boundary in boundaries] + [f"ge{boundaries[-1]:g}"]
    return ",".join(
        f"{label}:{correct}/{total}"
        for label, (correct, total) in zip(labels, counts)
        if total
    )


def draft_margin(logits: mx.array) -> float:
    require(logits.ndim == 1 and logits.size >= 2, "draft logits must contain two classes")
    top_two = mx.partition(logits, logits.size - 2)[-2:]
    return float(mx.max(top_two) - mx.min(top_two))


def draft_choice_arrays(
    logits: mx.array,
    draft_token_ids: mx.array | None,
) -> tuple[mx.array, mx.array]:
    """Build deterministic token and margin reductions for one shared evaluation."""

    require(logits.ndim == 1 and logits.size >= 2, "draft logits must contain two classes")
    winner = mx.argmax(logits)
    top_two = mx.topk(logits, 2)
    margin = mx.max(top_two) - mx.min(top_two)
    token = winner if draft_token_ids is None else draft_token_ids[winner]
    return token, margin


def timed_draft(
    sidecar,
    target_hidden: mx.array,
    accepted_token_id: int,
    depth: int | None = None,
) -> tuple[int, float, mx.array, float]:
    started = time.perf_counter()
    if depth is None:
        logits, next_hidden, _, _ = sidecar.draft_step(target_hidden, accepted_token_id)
    else:
        logits, next_hidden, _, _ = sidecar.draft_step(
            target_hidden, accepted_token_id, depth=depth
        )
    token, margin = draft_choice_arrays(logits, sidecar.draft_token_ids)
    mx.eval(next_hidden, token, margin)
    mx.synchronize()
    return (
        int(token),
        float(margin),
        next_hidden,
        time.perf_counter() - started,
    )


def greedy_token_array(logits: mx.array) -> mx.array:
    require(logits.ndim == 2 and logits.shape[0] > 0, "greedy logits must contain rows")
    return mx.argmax(logits, axis=-1)


def greedy_token_ids(logits: mx.array) -> list[int]:
    return [int(token_id) for token_id in greedy_token_array(logits).tolist()]


def matching_draft_prefix(draft_tokens: list[int], target_tokens: list[int]) -> int:
    require(len(target_tokens) >= len(draft_tokens), "target tokens do not cover every draft")
    accepted = 0
    for draft_token, target_token in zip(draft_tokens, target_tokens):
        if draft_token != target_token:
            break
        accepted += 1
    return accepted


def draft_gate(
    max_draft_tokens: int,
    remaining_tokens: int,
    current_drafts: int,
    previous_margin: float,
    threshold: float,
) -> bool:
    """Return whether another recursive draft can be attempted safely."""

    require(current_drafts >= 1, "draft gate requires an existing draft")
    return (
        max_draft_tokens >= current_drafts + 1
        and remaining_tokens >= current_drafts + 2
        and previous_margin >= threshold
    )


def accepted_draft_prefix(draft_tokens: list[int], verified_logits: mx.array) -> int:
    require(
        verified_logits.ndim == 2 and verified_logits.shape[0] >= len(draft_tokens),
        "verified logits do not cover every draft",
    )
    return matching_draft_prefix(draft_tokens, greedy_token_ids(verified_logits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--mtp-sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", type=Path)
    parser.add_argument("--mtp-final-norm-override", type=Path)
    parser.add_argument("--mtp-final-norm-damping", type=float, default=1.0)
    parser.add_argument("--learned-mtp-predictor", type=Path)
    parser.add_argument("--prompt", default="Complete this Python function:\n\ndef binary_search(values, target):\n")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--margin-gib", type=float, default=0.5)
    parser.add_argument("--cache-limit-mib", type=int)
    parser.add_argument("--capture-rollback", action="store_true")
    parser.add_argument("--max-draft-tokens", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--draft-margin-threshold", type=float, default=1.5)
    parser.add_argument("--second-draft-margin-threshold", type=float, default=1.0)
    parser.add_argument("--third-attempt-margin-threshold", type=float, default=2.0)
    parser.add_argument("--third-draft-margin-threshold", type=float, default=1.0)
    parser.add_argument("--lookup-max-draft-tokens", type=int, default=0)
    parser.add_argument("--lookup-min-key-tokens", type=int, default=3)
    parser.add_argument("--lookup-max-key-tokens", type=int, default=8)
    parser.add_argument("--lookup-table-entries", type=int, default=65_536)
    parser.add_argument("--lookup-positions-per-key", type=int, default=4)
    parser.add_argument("--lookup-min-matches", type=int, default=2)
    parser.add_argument("--lookup-mtp-agreement-tokens", type=int, choices=(1, 2), default=1)
    parser.add_argument("--token-timings", action="store_true")
    parser.add_argument("--cycle-trace", type=Path)
    parser.add_argument("--paged-embeddings", action="store_true")
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    parser.add_argument(
        "--compile-mamba",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.max_new_tokens >= 4, "speculative benchmark requires at least four tokens")
        require(args.warmup_cycles >= 0, "warmup cycles cannot be negative")
        require(
            math.isfinite(args.draft_margin_threshold)
            and math.isfinite(args.second_draft_margin_threshold)
            and math.isfinite(args.third_attempt_margin_threshold)
            and math.isfinite(args.third_draft_margin_threshold),
            "draft margin thresholds must be finite",
        )
        require(
            args.cache_limit_mib is None or args.cache_limit_mib >= 0,
            "cache limit cannot be negative",
        )
        require(args.embedding_cache_rows >= 0, "embedding cache rows cannot be negative")
        require(
            math.isfinite(args.mtp_final_norm_damping)
            and 0.0 <= args.mtp_final_norm_damping <= 1.0,
            "MTP final norm damping must be finite and between zero and one",
        )
        require(
            args.mtp_final_norm_override is not None
            or args.mtp_final_norm_damping == 1.0,
            "MTP final norm damping requires an override",
        )
        require(
            0 <= args.lookup_max_draft_tokens <= 4,
            "lookup draft limit must be between zero and four",
        )
        require(
            1 <= args.lookup_min_key_tokens <= args.lookup_max_key_tokens,
            "invalid lookup key-token range",
        )
        learned_report = None
        if args.learned_mtp_predictor is not None:
            require(
                args.mtp_lm_head is not None,
                "learned MTP predictor requires an explicit reduced vocabulary head",
            )
            learned_report = validated_predictor_report(
                args.learned_mtp_predictor,
                args.model_dir,
                args.mtp_lm_head,
            )
        learned_payload = (
            learned_report.get("inference_payload_bytes", 0) if learned_report else 0
        )
        require(
            isinstance(learned_payload, int) and learned_payload >= 0,
            "invalid learned predictor payload",
        )
        final_norm_payload = 0
        if args.mtp_final_norm_override is not None:
            require(
                args.mtp_lm_head is not None,
                "MTP final norm override requires its reduced vocabulary head",
            )
            final_norm_report = load_json(
                args.mtp_final_norm_override / "report.json"
            )
            load_calibrated_final_norm(
                args.mtp_final_norm_override,
                args.model_dir,
                args.mtp_sidecar,
                args.mtp_lm_head,
            )
            final_norm_payload = final_norm_report.get("artifact_bytes", 0)
            require(
                isinstance(final_norm_payload, int) and final_norm_payload > 0,
                "invalid MTP final norm override payload",
            )
        result = preflight(
            args.model_dir,
            args.margin_gib,
            args.mtp_sidecar,
            args.mtp_lm_head,
            paged_embeddings=args.paged_embeddings,
            additional_payload_bytes=learned_payload + final_norm_payload,
        )
        print("speculative-preflight " + json.dumps(result, separators=(",", ":")), flush=True)
        require(result["safe_to_attempt"], "Metal wired cap is too low for target plus MTP sidecar")
        # Keep the full pre-approved kernel cap available for transient verifier
        # and MTP buffers. The preflight still rejects payloads whose explicit
        # requirement exceeds this cap.
        mx.set_wired_limit(result["effective_cap_bytes"])
        cache_limit_mib = args.cache_limit_mib
        if cache_limit_mib is None:
            cache_limit_mib = (
                128
                if result["mtp_head_payload_gib"] > 1 / 1024
                else 256
            )
        mx.set_cache_limit(cache_limit_mib * 2**20)
        try:
            load_started = time.perf_counter()
            model = ResidentModel(
                args.model_dir,
                args.mtp_sidecar,
                args.mtp_lm_head,
                paged_embeddings=args.paged_embeddings,
                embedding_cache_rows=args.embedding_cache_rows,
                compile_mamba=args.compile_mamba,
            )
            require(model.mtp is not None, "resident MTP sidecar did not load")
            if args.mtp_final_norm_override is not None:
                calibrated_norm = load_calibrated_final_norm(
                    args.mtp_final_norm_override,
                    args.model_dir,
                    args.mtp_sidecar,
                    args.mtp_lm_head,
                )
                require(
                    calibrated_norm.shape == model.mtp.final_norm_weight.shape,
                    "calibrated MTP final norm shape mismatch",
                )
                model.mtp.final_norm_weight = damped_final_norm(
                    model.mtp.final_norm_weight,
                    calibrated_norm,
                    args.mtp_final_norm_damping,
                )
            draft_model = (
                LearnedMTPPredictor(
                    args.learned_mtp_predictor,
                    model.mtp,
                    model_dir=args.model_dir,
                    mtp_lm_head=args.mtp_lm_head,
                )
                if args.learned_mtp_predictor is not None
                else model.mtp
            )
            learned_depth = args.learned_mtp_predictor is not None
            print(
                f"speculative-loaded active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f} "
                f"compiled_mamba={str(args.compile_mamba).lower()}",
                flush=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            require(prompt_ids, "prompt encoded to no tokens")
            logits = hidden = None
            for token_id in prompt_ids:
                logits, hidden = model.forward(token_id)
            require(logits is not None and hidden is not None, "prompt prefill produced no state")
            print(
                f"speculative-prefilled prompt_tokens={len(prompt_ids)} "
                f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                flush=True,
            )
            prefill_snapshot = model.snapshot()
            load_prefill_seconds = time.perf_counter() - load_started

            ordinary = [int(mx.argmax(logits))]
            ordinary_seconds = []
            ordinary_logits = logits
            while len(ordinary) < args.max_new_tokens:
                started = time.perf_counter()
                ordinary_logits, _ = model.forward(ordinary[-1])
                ordinary_token = mx.argmax(ordinary_logits)
                mx.eval(ordinary_token)
                mx.synchronize()
                ordinary_seconds.append(time.perf_counter() - started)
                ordinary.append(int(ordinary_token))
            model.restore(prefill_snapshot)
            prefill_snapshot = None
            ordinary_logits = None
            mx.clear_cache()
            print(
                f"speculative-reference-done tokens={len(ordinary)} "
                f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                flush=True,
            )

            base_token = int(mx.argmax(logits))
            generated = []
            cycles = []
            lookup = (
                NGramLookup(
                    prompt_ids,
                    min_key_tokens=args.lookup_min_key_tokens,
                    max_key_tokens=args.lookup_max_key_tokens,
                    max_entries=args.lookup_table_entries,
                    positions_per_key=args.lookup_positions_per_key,
                    min_matching_continuations=args.lookup_min_matches,
                )
                if args.lookup_max_draft_tokens > 0
                else None
            )
            while len(generated) < args.max_new_tokens:
                cycle_started = time.perf_counter()
                accepted_snapshot = None
                if not cycles:
                    print(
                        f"speculative-cycle-start active_gib={mx.get_active_memory() / 2**30:.3f} "
                        f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                        flush=True,
                    )
                remaining = args.max_new_tokens - len(generated)
                lookup_started = time.perf_counter()
                lookup_draft = (
                    lookup.propose(
                        base_token,
                        min(args.lookup_max_draft_tokens, remaining - 1),
                    )
                    if lookup is not None and remaining > 1
                    else None
                )
                lookup_seconds = time.perf_counter() - lookup_started
                lookup_candidate = lookup_draft is not None
                lookup_key_tokens = lookup_draft.key_tokens if lookup_draft is not None else 0
                (
                    first_draft,
                    first_margin,
                    draft_hidden,
                    first_mtp_seconds,
                ) = timed_draft(
                    draft_model,
                    hidden,
                    base_token,
                    depth=0 if learned_depth else None,
                )
                mtp_seconds = first_mtp_seconds
                second_draft_attempted = False
                second_candidate = None
                second_margin = None
                second_hidden = None
                third_draft_attempted = False
                third_candidate = None
                third_margin = None
                lookup_agreed = (
                    lookup_draft is not None
                    and lookup_draft.token_ids[0] == first_draft
                )
                if (
                    lookup_agreed
                    and args.lookup_mtp_agreement_tokens >= 2
                    and len(lookup_draft.token_ids) >= 2
                ):
                    second_draft_attempted = True
                    (
                        second_candidate,
                        second_margin,
                        second_hidden,
                        second_mtp_seconds,
                    ) = timed_draft(
                        draft_model,
                        draft_hidden,
                        first_draft,
                        depth=1 if learned_depth else None,
                    )
                    mtp_seconds += second_mtp_seconds
                    lookup_agreed = lookup_draft.token_ids[1] == second_candidate
                draft_source = "lookup" if lookup_agreed else "mtp"
                draft_tokens = (
                    list(lookup_draft.token_ids) if lookup_agreed else [first_draft]
                )
                if not lookup_agreed:
                    if draft_gate(
                        args.max_draft_tokens,
                        remaining,
                        1,
                        first_margin,
                        args.draft_margin_threshold,
                    ):
                        if not second_draft_attempted:
                            second_draft_attempted = True
                            (
                                second_candidate,
                                second_margin,
                                second_hidden,
                                second_mtp_seconds,
                            ) = timed_draft(
                                draft_model,
                                draft_hidden,
                                first_draft,
                                depth=1 if learned_depth else None,
                            )
                            mtp_seconds += second_mtp_seconds
                        if second_margin >= args.second_draft_margin_threshold:
                            draft_tokens.append(second_candidate)
                            if draft_gate(
                                args.max_draft_tokens,
                                remaining,
                                2,
                                second_margin,
                                args.third_attempt_margin_threshold,
                            ):
                                third_draft_attempted = True
                                (
                                    third_candidate,
                                    third_margin,
                                    _,
                                    third_mtp_seconds,
                                ) = timed_draft(
                                    draft_model,
                                    second_hidden,
                                    second_candidate,
                                    depth=2 if learned_depth else None,
                                )
                                mtp_seconds += third_mtp_seconds
                                if third_margin >= args.third_draft_margin_threshold:
                                    draft_tokens.append(third_candidate)
                if not cycles:
                    print(
                        f"speculative-draft-ready source={draft_source} "
                        f"ms={(mtp_seconds + lookup_seconds) * 1000:.3f} "
                        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                        f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                        flush=True,
                    )
                draft_hidden = None
                second_hidden = None

                before_verify = (
                    model.snapshot()
                    if not args.capture_rollback or len(draft_tokens) > 1
                    else None
                )
                verify_started = time.perf_counter()
                if args.capture_rollback:
                    capture_index = len(draft_tokens) - 1
                    verified_logits, verified_hidden, accepted_snapshot = (
                        model.verify_sequence(
                            [base_token, *draft_tokens],
                            capture_index,
                        )
                    )
                    accepted_snapshots = {capture_index: accepted_snapshot}
                else:
                    verified_logits, verified_hidden = model.forward_sequence(
                        [base_token, *draft_tokens]
                    )
                    accepted_snapshots = None
                verified_token_array = greedy_token_array(verified_logits)
                mx.eval(verified_token_array)
                mx.synchronize()
                verify_seconds = time.perf_counter() - verify_started
                verified_tokens = [int(token_id) for token_id in verified_token_array.tolist()]
                accepted_drafts = matching_draft_prefix(draft_tokens, verified_tokens)
                second_correct = (
                    second_draft_attempted
                    and accepted_drafts >= 1
                    and second_candidate == verified_tokens[1]
                )
                third_correct = (
                    third_draft_attempted
                    and accepted_drafts >= 2
                    and third_candidate == verified_tokens[2]
                )
                generated.append(base_token)
                emitted_drafts = draft_tokens[:accepted_drafts][
                    : args.max_new_tokens - len(generated)
                ]
                generated.extend(emitted_drafts)
                if lookup is not None:
                    lookup.extend([base_token, *emitted_drafts])
                produced = 1 + len(emitted_drafts)
                if accepted_drafts == len(draft_tokens):
                    logits = verified_logits[accepted_drafts]
                    hidden = verified_hidden[accepted_drafts]
                    replay_seconds = 0.0
                else:
                    replay_started = time.perf_counter()
                    if (
                        accepted_snapshots is not None
                        and accepted_drafts in accepted_snapshots
                    ):
                        model.restore(accepted_snapshots[accepted_drafts])
                        logits = verified_logits[accepted_drafts]
                        hidden = verified_hidden[accepted_drafts]
                    else:
                        require(before_verify is not None, "rejection replay has no cache snapshot")
                        model.restore(before_verify)
                        _, replay_hidden = model.forward_sequence(
                            [base_token, *draft_tokens[:accepted_drafts]]
                        )
                        logits = verified_logits[accepted_drafts]
                        hidden = replay_hidden[-1]
                    mx.synchronize()
                    replay_seconds = time.perf_counter() - replay_started
                accepted_snapshots = None
                accepted_snapshot = None
                before_verify = None
                base_token = verified_tokens[accepted_drafts]
                cycles.append(
                    {
                        "accepted": accepted_drafts == len(draft_tokens),
                        "accepted_drafts": accepted_drafts,
                        "drafted": len(draft_tokens),
                        "draft_source": draft_source,
                        "lookup_candidate": lookup_candidate,
                        "lookup_agreed": lookup_agreed,
                        "lookup_key_tokens": lookup_key_tokens,
                        "lookup_seconds": lookup_seconds,
                        "first_margin": first_margin,
                        "second_margin": second_margin,
                        "second_draft_attempted": second_draft_attempted,
                        "second_correct": second_correct,
                        "third_margin": third_margin,
                        "third_draft_attempted": third_draft_attempted,
                        "third_correct": third_correct,
                        "produced": produced,
                        "mtp_seconds": mtp_seconds,
                        "verify_seconds": verify_seconds,
                        "replay_seconds": replay_seconds,
                        "cycle_seconds": time.perf_counter() - cycle_started,
                    }
                )

            require(generated == ordinary, "speculative output differs from ordinary greedy decode")
            measured = cycles[min(args.warmup_cycles, len(cycles)) :]
            require(measured, "warmup excluded every speculative cycle")
            measured_tokens = sum(cycle["produced"] for cycle in measured)
            measured_seconds = sum(cycle["cycle_seconds"] for cycle in measured)
            ordinary_measured = ordinary_seconds[min(2, len(ordinary_seconds)) :]
            ordinary_rate = len(ordinary_measured) / sum(ordinary_measured)
            speculative_rate = measured_tokens / measured_seconds
            accepted_drafts = sum(cycle["accepted_drafts"] for cycle in measured)
            drafted = sum(cycle["drafted"] for cycle in measured)
            acceptance = accepted_drafts / drafted
            lookup_cycles = [cycle for cycle in measured if cycle["draft_source"] == "lookup"]
            lookup_candidates = [cycle for cycle in measured if cycle["lookup_candidate"]]
            mtp_cycles = [cycle for cycle in measured if cycle["draft_source"] == "mtp"]
            lookup_drafted = sum(cycle["drafted"] for cycle in lookup_cycles)
            lookup_accepted = sum(cycle["accepted_drafts"] for cycle in lookup_cycles)
            lookup_acceptance = (
                lookup_accepted / lookup_drafted if lookup_drafted else 0.0
            )
            second_attempt_rate = sum(
                cycle["second_draft_attempted"] for cycle in measured
            ) / len(measured)
            second_draft_rate = (
                sum(cycle["drafted"] == 2 for cycle in mtp_cycles) / len(mtp_cycles)
                if mtp_cycles
                else 0.0
            )
            second_eligible = [
                cycle
                for cycle in measured
                if cycle["second_draft_attempted"] and cycle["accepted_drafts"] >= 1
            ]
            second_correct = sum(cycle["second_correct"] for cycle in second_eligible)
            second_acceptance = (
                second_correct / len(second_eligible) if second_eligible else 0.0
            )
            third_eligible = [
                cycle
                for cycle in measured
                if cycle["third_draft_attempted"] and cycle["accepted_drafts"] >= 2
            ]
            third_correct = sum(cycle["third_correct"] for cycle in third_eligible)
            third_acceptance = third_correct / len(third_eligible) if third_eligible else 0.0
            correct_second_margins = [
                cycle["second_margin"] for cycle in second_eligible if cycle["second_correct"]
            ]
            rejected_second_margins = [
                cycle["second_margin"] for cycle in second_eligible if not cycle["second_correct"]
            ]
            mtp_ms = [cycle["mtp_seconds"] * 1000 for cycle in measured]
            lookup_ms = [cycle["lookup_seconds"] * 1000 for cycle in measured]
            verify_ms = [cycle["verify_seconds"] * 1000 for cycle in measured]
            cycle_ms = [cycle["cycle_seconds"] * 1000 for cycle in measured]
            replay_ms = [cycle["replay_seconds"] * 1000 for cycle in measured]
            cycles_by_drafts = {
                count: [cycle for cycle in measured if cycle["drafted"] == count]
                for count in range(1, args.max_draft_tokens + 1)
            }
            draft_length_counts = ",".join(
                f"{count}:{len(group)}" for count, group in cycles_by_drafts.items()
            )
            verify_by_drafts = ",".join(
                f"{count}:{statistics.median([cycle['verify_seconds'] * 1000 for cycle in group]):.3f}"
                for count, group in cycles_by_drafts.items()
                if group
            )
            cycle_by_drafts = ",".join(
                f"{count}:{statistics.median([cycle['cycle_seconds'] * 1000 for cycle in group]):.3f}"
                for count, group in cycles_by_drafts.items()
                if group
            )
            accepted_by_depth = ",".join(
                f"{depth}:{sum(cycle['accepted_drafts'] >= depth for cycle in measured if cycle['drafted'] >= depth)}/"
                f"{sum(cycle['drafted'] >= depth for cycle in measured)}"
                for depth in range(1, args.max_draft_tokens + 1)
                if any(cycle["drafted"] >= depth for cycle in measured)
            )
            first_margin_outcomes = margin_outcome_bins(
                [
                    (cycle["first_margin"], cycle["accepted_drafts"] >= 1)
                    for cycle in measured
                ]
            )
            second_margin_outcomes = margin_outcome_bins(
                [
                    (cycle["second_margin"], cycle["second_correct"])
                    for cycle in second_eligible
                ]
            )
            third_margin_outcomes = margin_outcome_bins(
                [
                    (cycle["third_margin"], cycle["third_correct"])
                    for cycle in third_eligible
                ]
            )
            print(
                f"speculative-result prompt_tokens={len(prompt_ids)} generated_tokens={len(generated)} "
                f"load_prefill_seconds={load_prefill_seconds:.3f} cycles={len(cycles)} "
                f"measured_cycles={len(measured)} measured_tokens={measured_tokens} "
                f"drafted={drafted} accepted_drafts={accepted_drafts} "
                f"acceptance={acceptance:.6f} second_attempt_rate={second_attempt_rate:.6f} "
                f"second_draft_rate={second_draft_rate:.6f} second_eligible={len(second_eligible)} "
                f"second_acceptance={second_acceptance:.6f} "
                f"second_correct_margin_median={statistics.median(correct_second_margins) if correct_second_margins else 0.0:.6f} "
                f"second_rejected_margin_median={statistics.median(rejected_second_margins) if rejected_second_margins else 0.0:.6f} "
                f"third_attempt_rate={sum(cycle['third_draft_attempted'] for cycle in measured) / len(measured):.6f} "
                f"third_draft_rate={sum(cycle['drafted'] >= 3 for cycle in mtp_cycles) / len(mtp_cycles) if mtp_cycles else 0.0:.6f} "
                f"third_eligible={len(third_eligible)} third_acceptance={third_acceptance:.6f} "
                f"lookup_candidates={len(lookup_candidates)} "
                f"lookup_candidate_rate={len(lookup_candidates) / len(measured):.6f} "
                f"lookup_hits={len(lookup_cycles)} lookup_hit_rate={len(lookup_cycles) / len(measured):.6f} "
                f"lookup_agreement={len(lookup_cycles) / len(lookup_candidates) if lookup_candidates else 0.0:.6f} "
                f"lookup_drafted={lookup_drafted} lookup_accepted={lookup_accepted} "
                f"lookup_acceptance={lookup_acceptance:.6f} "
                f"lookup_median_ms={statistics.median(lookup_ms):.6f} "
                f"lookup_p95_ms={percentile(lookup_ms, 0.95):.6f} "
                f"ordinary_tok_s={ordinary_rate:.3f} "
                f"speculative_tok_s={speculative_rate:.3f} speedup={speculative_rate / ordinary_rate:.3f} "
                f"mtp_median_ms={statistics.median(mtp_ms) if mtp_ms else 0.0:.3f} "
                f"mtp_p95_ms={percentile(mtp_ms, 0.95) if mtp_ms else 0.0:.3f} "
                f"verify_median_ms={statistics.median(verify_ms):.3f} "
                f"draft_length_counts={draft_length_counts} "
                f"accepted_by_depth={accepted_by_depth} "
                f"first_margin_outcomes={first_margin_outcomes} "
                f"second_margin_outcomes={second_margin_outcomes} "
                f"third_margin_outcomes={third_margin_outcomes} "
                f"verify_by_drafts_ms={verify_by_drafts} "
                f"rollback_count={sum(value > 0 for value in replay_ms)} "
                f"rollback_total_ms={sum(replay_ms):.3f} "
                f"cycle_median_ms={statistics.median(cycle_ms):.3f} "
                f"cycle_by_drafts_ms={cycle_by_drafts} "
                f"embedding_lookups={getattr(model.embeddings, 'lookups', 0)} "
                f"embedding_cache_hits={getattr(model.embeddings, 'cache_hits', 0)} "
                f"embedding_staging_ms={getattr(model.embeddings, 'staging_seconds', 0.0) * 1000:.3f} "
                f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f} integrity=exact",
                flush=True,
            )
            if args.token_timings:
                print(
                    "speculative-cycle-ms "
                    + ",".join(
                        f"{cycle['cycle_seconds'] * 1000:.3f}:"
                        f"{cycle['draft_source']}:{cycle['accepted_drafts']}/{cycle['drafted']}"
                        for cycle in cycles
                    ),
                    flush=True,
                )
            if args.cycle_trace is not None:
                trace = {
                    "format": "nemotron-speculative-cycle-trace-v1",
                    "model_dir": str(args.model_dir.resolve()),
                    "mtp_sidecar": str(args.mtp_sidecar.resolve()),
                    "mtp_lm_head": (
                        str(args.mtp_lm_head.resolve()) if args.mtp_lm_head is not None else None
                    ),
                    "learned_mtp_predictor": (
                        str(args.learned_mtp_predictor.resolve())
                        if args.learned_mtp_predictor is not None
                        else None
                    ),
                    "mtp_final_norm_override": (
                        str(args.mtp_final_norm_override.resolve())
                        if args.mtp_final_norm_override is not None
                        else None
                    ),
                    "mtp_final_norm_damping": args.mtp_final_norm_damping,
                    "max_new_tokens": args.max_new_tokens,
                    "warmup_cycles": args.warmup_cycles,
                    "max_draft_tokens": args.max_draft_tokens,
                    "draft_margin_threshold": args.draft_margin_threshold,
                    "second_draft_margin_threshold": args.second_draft_margin_threshold,
                    "third_attempt_margin_threshold": args.third_attempt_margin_threshold,
                    "third_draft_margin_threshold": args.third_draft_margin_threshold,
                    "integrity": "exact",
                    "cycles": cycles,
                }
                args.cycle_trace.parent.mkdir(parents=True, exist_ok=True)
                trace_part = args.cycle_trace.with_name(args.cycle_trace.name + ".part")
                trace_part.write_text(
                    json.dumps(trace, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                trace_part.replace(args.cycle_trace)
                print(f"speculative-cycle-trace path={args.cycle_trace}", flush=True)
            print(tokenizer.decode(generated), flush=True)
        finally:
            mx.set_wired_limit(result["effective_cap_bytes"])
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron speculative error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

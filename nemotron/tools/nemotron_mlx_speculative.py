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

from nemotron_metadata import MetadataError, require
from nemotron_mlx_resident import ResidentModel, preflight
from nemotron_ngram_lookup import NGramLookup


def timed_eval(callable_):
    started = time.perf_counter()
    result = callable_()
    arrays = [value for value in result if isinstance(value, mx.array)] if isinstance(result, tuple) else [result]
    mx.eval(*arrays)
    mx.synchronize()
    return result, time.perf_counter() - started


def percentile(values: list[float], fraction: float) -> float:
    require(values, "cannot calculate a percentile of no values")
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def draft_margin(logits: mx.array) -> float:
    require(logits.ndim == 1 and logits.size >= 2, "draft logits must contain two classes")
    top_two = mx.partition(logits, logits.size - 2)[-2:]
    return float(mx.max(top_two) - mx.min(top_two))


def accepted_draft_prefix(draft_tokens: list[int], verified_logits: mx.array) -> int:
    require(
        verified_logits.ndim == 2 and verified_logits.shape[0] >= len(draft_tokens),
        "verified logits do not cover every draft",
    )
    accepted = 0
    for index, draft_token in enumerate(draft_tokens):
        if draft_token != int(mx.argmax(verified_logits[index])):
            break
        accepted += 1
    return accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--mtp-sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", type=Path)
    parser.add_argument("--prompt", default="Complete this Python function:\n\ndef binary_search(values, target):\n")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--margin-gib", type=float, default=0.5)
    parser.add_argument("--cache-limit-mib", type=int)
    parser.add_argument("--capture-rollback", action="store_true")
    parser.add_argument("--max-draft-tokens", type=int, choices=(1, 2), default=1)
    parser.add_argument("--draft-margin-threshold", type=float, default=1.5)
    parser.add_argument("--second-draft-margin-threshold", type=float, default=1.0)
    parser.add_argument("--lookup-max-draft-tokens", type=int, default=0)
    parser.add_argument("--lookup-min-key-tokens", type=int, default=3)
    parser.add_argument("--lookup-max-key-tokens", type=int, default=8)
    parser.add_argument("--lookup-table-entries", type=int, default=65_536)
    parser.add_argument("--lookup-positions-per-key", type=int, default=4)
    parser.add_argument("--lookup-min-matches", type=int, default=2)
    parser.add_argument("--lookup-mtp-agreement-tokens", type=int, choices=(1, 2), default=1)
    parser.add_argument("--token-timings", action="store_true")
    parser.add_argument("--paged-embeddings", action="store_true")
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.max_new_tokens >= 4, "speculative benchmark requires at least four tokens")
        require(args.warmup_cycles >= 0, "warmup cycles cannot be negative")
        require(
            math.isfinite(args.draft_margin_threshold)
            and math.isfinite(args.second_draft_margin_threshold),
            "draft margin thresholds must be finite",
        )
        require(
            args.cache_limit_mib is None or args.cache_limit_mib >= 0,
            "cache limit cannot be negative",
        )
        require(args.embedding_cache_rows >= 0, "embedding cache rows cannot be negative")
        require(
            0 <= args.lookup_max_draft_tokens <= 4,
            "lookup draft limit must be between zero and four",
        )
        require(
            1 <= args.lookup_min_key_tokens <= args.lookup_max_key_tokens,
            "invalid lookup key-token range",
        )
        result = preflight(
            args.model_dir,
            args.margin_gib,
            args.mtp_sidecar,
            args.mtp_lm_head,
            paged_embeddings=args.paged_embeddings,
        )
        print("speculative-preflight " + json.dumps(result, separators=(",", ":")), flush=True)
        require(result["safe_to_attempt"], "Metal wired cap is too low for target plus MTP sidecar")
        # Keep the full pre-approved kernel cap available for transient verifier
        # and MTP buffers. The preflight still rejects payloads whose explicit
        # requirement exceeds this cap.
        previous_limit = mx.set_wired_limit(result["effective_cap_bytes"])
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
            )
            require(model.mtp is not None, "resident MTP sidecar did not load")
            print(
                f"speculative-loaded active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
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
                mx.synchronize()
                ordinary_seconds.append(time.perf_counter() - started)
                ordinary.append(int(mx.argmax(ordinary_logits)))
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
                second_result = None
                second_logits = None
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
                (draft_result, first_mtp_seconds) = timed_eval(
                    lambda: model.mtp.draft_step(hidden, base_token)
                )
                draft_logits, draft_hidden, _, _ = draft_result
                first_draft = model.mtp.argmax_token(draft_logits)
                first_margin = draft_margin(draft_logits)
                mtp_seconds = first_mtp_seconds
                second_draft_attempted = False
                second_candidate = None
                second_margin = None
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
                    (second_result, second_mtp_seconds) = timed_eval(
                        lambda: model.mtp.draft_step(draft_hidden, first_draft)
                    )
                    second_logits, _, _, _ = second_result
                    mtp_seconds += second_mtp_seconds
                    second_margin = draft_margin(second_logits)
                    second_candidate = model.mtp.argmax_token(second_logits)
                    lookup_agreed = lookup_draft.token_ids[1] == second_candidate
                draft_source = "lookup" if lookup_agreed else "mtp"
                draft_tokens = (
                    list(lookup_draft.token_ids) if lookup_agreed else [first_draft]
                )
                if not lookup_agreed:
                    if (
                        args.max_draft_tokens >= 2
                        and remaining >= 3
                        and first_margin >= args.draft_margin_threshold
                    ):
                        if not second_draft_attempted:
                            second_draft_attempted = True
                            (second_result, second_mtp_seconds) = timed_eval(
                                lambda: model.mtp.draft_step(draft_hidden, first_draft)
                            )
                            second_logits, _, _, _ = second_result
                            mtp_seconds += second_mtp_seconds
                            second_margin = draft_margin(second_logits)
                            second_candidate = model.mtp.argmax_token(second_logits)
                        if second_margin >= args.second_draft_margin_threshold:
                            draft_tokens.append(second_candidate)
                if not cycles:
                    print(
                        f"speculative-draft-ready source={draft_source} "
                        f"ms={(mtp_seconds + lookup_seconds) * 1000:.3f} "
                        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                        f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                        flush=True,
                    )
                draft_result = None
                draft_logits = None
                draft_hidden = None
                second_result = None
                second_logits = None

                before_verify = (
                    model.snapshot()
                    if not args.capture_rollback or len(draft_tokens) > 1
                    else None
                )
                verify_started = time.perf_counter()
                if args.capture_rollback and draft_source == "lookup":
                    capture_index = len(draft_tokens) - 1
                    verified_logits, verified_hidden, accepted_snapshot = (
                        model.verify_sequence(
                            [base_token, *draft_tokens],
                            capture_index,
                        )
                    )
                    accepted_snapshots = {capture_index: accepted_snapshot}
                elif args.capture_rollback:
                    if len(draft_tokens) == 1:
                        verified_logits, verified_hidden, accepted_snapshot = model.verify_sequence(
                            [base_token, *draft_tokens],
                            0,
                        )
                        accepted_snapshots = {0: accepted_snapshot}
                    else:
                        verified_logits, verified_hidden, accepted_snapshot = (
                            model.verify_sequence(
                                [base_token, *draft_tokens],
                                1,
                            )
                        )
                        accepted_snapshots = {1: accepted_snapshot}
                else:
                    verified_logits, verified_hidden = model.forward_sequence(
                        [base_token, *draft_tokens]
                    )
                    accepted_snapshots = None
                mx.synchronize()
                verify_seconds = time.perf_counter() - verify_started
                accepted_drafts = accepted_draft_prefix(draft_tokens, verified_logits)
                second_correct = (
                    second_draft_attempted
                    and accepted_drafts >= 1
                    and second_candidate == int(mx.argmax(verified_logits[1]))
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
                base_token = int(mx.argmax(logits))
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
            two_token_cycles = [cycle for cycle in measured if cycle["drafted"] == 1]
            three_token_cycles = [cycle for cycle in measured if cycle["drafted"] == 2]
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
                f"verify2_median_ms={statistics.median([cycle['verify_seconds'] * 1000 for cycle in two_token_cycles]) if two_token_cycles else 0.0:.3f} "
                f"verify3_median_ms={statistics.median([cycle['verify_seconds'] * 1000 for cycle in three_token_cycles]) if three_token_cycles else 0.0:.3f} "
                f"rollback_count={sum(value > 0 for value in replay_ms)} "
                f"rollback_total_ms={sum(replay_ms):.3f} "
                f"cycle_median_ms={statistics.median(cycle_ms):.3f} "
                f"cycle2_median_ms={statistics.median([cycle['cycle_seconds'] * 1000 for cycle in two_token_cycles]) if two_token_cycles else 0.0:.3f} "
                f"cycle3_median_ms={statistics.median([cycle['cycle_seconds'] * 1000 for cycle in three_token_cycles]) if three_token_cycles else 0.0:.3f} "
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
            print(tokenizer.decode(generated), flush=True)
        finally:
            mx.set_wired_limit(previous_limit)
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron speculative error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

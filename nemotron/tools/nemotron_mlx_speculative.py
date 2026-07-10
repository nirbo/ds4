#!/usr/bin/env python3
"""Exact one-draft MTP speculative generation for resident Nemotron."""

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
    parser.add_argument("--token-timings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.max_new_tokens >= 4, "speculative benchmark requires at least four tokens")
        require(args.warmup_cycles >= 0, "warmup cycles cannot be negative")
        require(
            args.cache_limit_mib is None or args.cache_limit_mib >= 0,
            "cache limit cannot be negative",
        )
        result = preflight(
            args.model_dir,
            args.margin_gib,
            args.mtp_sidecar,
            args.mtp_lm_head,
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
            model = ResidentModel(args.model_dir, args.mtp_sidecar, args.mtp_lm_head)
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
            while len(generated) < args.max_new_tokens:
                cycle_started = time.perf_counter()
                if not cycles:
                    print(
                        f"speculative-cycle-start active_gib={mx.get_active_memory() / 2**30:.3f} "
                        f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                        flush=True,
                    )
                (draft_result, mtp_seconds) = timed_eval(
                    lambda: model.mtp(hidden, base_token)
                )
                draft_logits, _, _ = draft_result
                draft_token = model.mtp.argmax_token(draft_logits)
                if not cycles:
                    print(
                        f"speculative-mtp-ready ms={mtp_seconds * 1000:.3f} "
                        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                        f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                        flush=True,
                    )
                draft_result = None
                draft_logits = None

                before_verify = None if args.capture_rollback else model.snapshot()
                verify_started = time.perf_counter()
                if args.capture_rollback:
                    verified_logits, verified_hidden, accepted_snapshot = model.verify_sequence(
                        [base_token, draft_token],
                        0,
                    )
                else:
                    verified_logits, verified_hidden = model.forward_sequence(
                        [base_token, draft_token]
                    )
                    accepted_snapshot = None
                mx.synchronize()
                verify_seconds = time.perf_counter() - verify_started
                target_token = int(mx.argmax(verified_logits[0]))
                accepted = draft_token == target_token
                produced = 1
                generated.append(base_token)
                if accepted and len(generated) < args.max_new_tokens:
                    generated.append(draft_token)
                    produced += 1
                    logits = verified_logits[1]
                    hidden = verified_hidden[1]
                else:
                    replay_started = time.perf_counter()
                    if accepted_snapshot is not None:
                        model.restore(accepted_snapshot)
                        logits = verified_logits[0]
                        hidden = verified_hidden[0]
                    else:
                        require(before_verify is not None, "rejection replay has no cache snapshot")
                        model.restore(before_verify)
                        logits, hidden = model.forward(base_token)
                    mx.synchronize()
                    replay_seconds = time.perf_counter() - replay_started
                if accepted:
                    replay_seconds = 0.0
                accepted_snapshot = None
                before_verify = None
                base_token = int(mx.argmax(logits))
                cycles.append(
                    {
                        "accepted": accepted,
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
            acceptance = sum(cycle["accepted"] for cycle in measured) / len(measured)
            mtp_ms = [cycle["mtp_seconds"] * 1000 for cycle in measured]
            verify_ms = [cycle["verify_seconds"] * 1000 for cycle in measured]
            cycle_ms = [cycle["cycle_seconds"] * 1000 for cycle in measured]
            replay_ms = [cycle["replay_seconds"] * 1000 for cycle in measured]
            print(
                f"speculative-result prompt_tokens={len(prompt_ids)} generated_tokens={len(generated)} "
                f"load_prefill_seconds={load_prefill_seconds:.3f} cycles={len(cycles)} "
                f"measured_cycles={len(measured)} measured_tokens={measured_tokens} "
                f"acceptance={acceptance:.6f} ordinary_tok_s={ordinary_rate:.3f} "
                f"speculative_tok_s={speculative_rate:.3f} speedup={speculative_rate / ordinary_rate:.3f} "
                f"mtp_median_ms={statistics.median(mtp_ms):.3f} mtp_p95_ms={percentile(mtp_ms, 0.95):.3f} "
                f"verify_median_ms={statistics.median(verify_ms):.3f} "
                f"rollback_count={sum(value > 0 for value in replay_ms)} "
                f"rollback_total_ms={sum(replay_ms):.3f} "
                f"cycle_median_ms={statistics.median(cycle_ms):.3f} "
                f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f} integrity=exact",
                flush=True,
            )
            if args.token_timings:
                print(
                    "speculative-cycle-ms "
                    + ",".join(
                        f"{cycle['cycle_seconds'] * 1000:.3f}:{int(cycle['accepted'])}"
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

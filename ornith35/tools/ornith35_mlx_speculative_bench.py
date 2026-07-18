#!/usr/bin/env python3
"""Benchmark Ornith-35 target block verification before adding a drafter."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_speculative as speculative
from ornith35_dspark_reference import PRODUCTION_CONFIG as DSPARK_CONFIG
from ornith35_moe_reference import MoEError, require
from ornith35_tokenizer import DEFAULT_ROOT, TokenizerError, load_text_tokenizer, render_text_prompt


def _median_seconds(callback, rounds: int) -> float:
    callback()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _serial_target_block(
    proposals: tuple[int, ...],
    session: model.TextDecodeSession,
) -> None:
    current = session
    for token_id in proposals:
        result, current = model.forward_session_token(token_id, current)
        model.evaluate_result(result)


def _require_exact_cursor(
    actual: speculative.GreedyTargetCursor,
    expected: speculative.GreedyTargetCursor,
) -> None:
    require(actual.state.position == expected.state.position, "rollback position mismatch")
    checks = [
        ("hidden", mx.array_equal(actual.hidden, expected.hidden)),
        ("logits", mx.array_equal(actual.logits, expected.logits)),
    ]
    for index, (actual_state, expected_state) in enumerate(
        zip(actual.state.layers, expected.state.layers)
    ):
        if isinstance(expected_state, gdn.MLXGDNState):
            require(isinstance(actual_state, gdn.MLXGDNState), "rollback GDN type mismatch")
            checks.extend(
                (
                    (f"layer{index}.conv", mx.array_equal(actual_state.conv, expected_state.conv)),
                    (
                        f"layer{index}.recurrent",
                        mx.array_equal(actual_state.recurrent, expected_state.recurrent),
                    ),
                )
            )
        else:
            require(
                isinstance(
                    actual_state,
                    (attention.MLXAttentionState, attention.MLXLinearAttentionState),
                )
                and isinstance(expected_state, attention.MLXAttentionState),
                "rollback attention type mismatch",
            )
            actual_keys = (
                actual_state.keys[:, : actual_state.position]
                if isinstance(actual_state, attention.MLXLinearAttentionState)
                else actual_state.keys
            )
            actual_values = (
                actual_state.values[:, : actual_state.position]
                if isinstance(actual_state, attention.MLXLinearAttentionState)
                else actual_state.values
            )
            checks.extend(
                (
                    (f"layer{index}.keys", mx.array_equal(actual_keys, expected_state.keys)),
                    (
                        f"layer{index}.values",
                        mx.array_equal(actual_values, expected_state.values),
                    ),
                )
            )
    mx.eval(*(check for _, check in checks))
    mismatches = [name for name, check in checks if not bool(check.item())]
    require(not mismatches, f"rollback cursor mismatch: {','.join(mismatches)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompt",
        default="Complete this Python function:\n\ndef binary_search(values, target):\n",
    )
    parser.add_argument("--proposal-tokens", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--trajectory-blocks", type=int, default=0)
    parser.add_argument(
        "--sweep-prefixes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--compiled-prefill-tails",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exact-bf16-block-head",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--capture-dspark-aux",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--linear-target-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="benchmark one advancing fixed-capacity target trajectory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(
            1 <= args.proposal_tokens <= speculative.MAX_PROPOSAL_TOKENS,
            "proposal token count is outside the verifier limit",
        )
        require(args.rounds > 0, "benchmark rounds must be positive")
        require(args.trajectory_blocks >= 0, "trajectory blocks must be nonnegative")
        require(
            not args.linear_target_cache or args.trajectory_blocks > 0,
            "linear target cache requires at least one trajectory block",
        )
        tokenizer = load_text_tokenizer(args.root)
        prompt_ids = list(tokenizer.encode(render_text_prompt(args.prompt)))
        require(bool(prompt_ids), "benchmark prompt produced no tokens")

        started = time.perf_counter()
        weights = model.load_text_model(
            args.root,
            map_embedding=True,
            quantize_lm_head=True,
        )
        state = model.initial_state(weights, model.PRODUCTION_CONFIG)
        result, schedule = generate.prefill_prompt(
            prompt_ids,
            state,
            weights,
            max_chunk=args.prefill_chunk,
        )
        exact_block_lm_head = (
            model.load_exact_block_lm_head(args.root)
            if args.exact_bf16_block_head
            else None
        )
        cursor = speculative.cursor_from_result(result)
        serial_session = model.start_decode_session(
            weights,
            cursor.state,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=True,
            compile_attention_tails=True,
        )

        proposals = []
        proposal_cursor = cursor
        proposal_session = serial_session
        for _ in range(args.proposal_tokens):
            token_id = speculative.greedy_token(
                proposal_cursor.logits,
                proposal_cursor.hidden,
                weights.lm_head,
            )
            proposals.append(token_id)
            next_result, proposal_session = model.forward_session_token(
                token_id,
                proposal_session,
            )
            model.evaluate_result(next_result)
            proposal_cursor = speculative.cursor_from_result(next_result)
        proposal_ids = tuple(proposals)
        verifier = speculative.start_greedy_verifier(
            weights,
            cursor,
            block_tokens=args.proposal_tokens,
            compile_prefill_tails=args.compiled_prefill_tails,
            exact_block_lm_head=exact_block_lm_head,
            auxiliary_hidden_state_indices=(
                DSPARK_CONFIG.aux_hidden_state_indices
                if args.capture_dspark_aux
                else ()
            ),
        )
        reference, _ = speculative.verify_greedy_block(proposal_ids, verifier)
        require(reference.all_accepted, "target chunk did not accept its serial greedy trajectory")

        print(
            "speculative-bench-ready "
            f"prompt_tokens={len(prompt_ids)} chunks={generate.format_prefill_schedule(schedule)} "
            f"proposal_tokens={len(proposal_ids)} setup_s={time.perf_counter() - started:.3f} "
            f"compiled_prefill_tails={str(args.compiled_prefill_tails).lower()} "
            f"exact_bf16_block_head={str(args.exact_bf16_block_head).lower()} "
            f"capture_dspark_aux={str(args.capture_dspark_aux).lower()} "
            f"linear_target_cache={str(args.linear_target_cache).lower()} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )

        if args.trajectory_blocks:
            required_tokens = args.trajectory_blocks * len(proposal_ids) + 1
            serial_tokens = list(proposal_ids)
            while len(serial_tokens) < required_tokens:
                token_id = speculative.greedy_token(
                    proposal_cursor.logits,
                    proposal_cursor.hidden,
                    weights.lm_head,
                )
                serial_tokens.append(token_id)
                next_result, proposal_session = model.forward_session_token(
                    token_id,
                    proposal_session,
                )
                model.evaluate_result(next_result)
                proposal_cursor = speculative.cursor_from_result(next_result)
            linear_session = None
            if args.linear_target_cache:
                linear_session = model.start_linear_decode_session(
                    weights,
                    cursor.state,
                    cursor.state.position + args.trajectory_blocks * len(proposal_ids),
                    model.PRODUCTION_CONFIG,
                    compile_gdn_layers=True,
                    compile_attention_tails=True,
                )
                linear_cursor = speculative.GreedyTargetCursor(
                    state=linear_session.state,
                    hidden=cursor.hidden,
                    logits=cursor.logits,
                )
                trajectory_session = speculative.start_greedy_verifier(
                    weights,
                    linear_cursor,
                    block_tokens=args.proposal_tokens,
                    compile_prefill_tails=args.compiled_prefill_tails,
                    exact_block_lm_head=exact_block_lm_head,
                    auxiliary_hidden_state_indices=(
                        DSPARK_CONFIG.aux_hidden_state_indices
                        if args.capture_dspark_aux
                        else ()
                    ),
                    linear_session=linear_session,
                )
            else:
                trajectory_session = verifier
            trajectory_blocks = [
                tuple(serial_tokens[offset : offset + len(proposal_ids)])
                for offset in range(
                    0,
                    args.trajectory_blocks * len(proposal_ids),
                    len(proposal_ids),
                )
            ]
            block_seconds = []
            for block_index, block in enumerate(trajectory_blocks):
                offset = block_index * len(proposal_ids)
                block_started = time.perf_counter()
                checked, trajectory_session = speculative.verify_greedy_block(
                    block,
                    trajectory_session,
                )
                block_seconds.append(time.perf_counter() - block_started)
                expected = tuple(
                    serial_tokens[offset : offset + len(proposal_ids) + 1]
                )
                require(
                    checked.all_accepted and checked.emitted_tokens == expected,
                    f"greedy trajectory diverged at block {block_index}",
                )
            trajectory_seconds = sum(block_seconds)
            if linear_session is not None:
                expected_session = verifier
                for block in trajectory_blocks:
                    expected_checked, expected_session = speculative.verify_greedy_block(
                        block,
                        expected_session,
                    )
                    require(
                        expected_checked.all_accepted,
                        "immutable target schedule rejected its greedy trajectory",
                    )
                _require_exact_cursor(
                    trajectory_session.cursor,
                    expected_session.cursor,
                )
            steady_block_ms = (
                f"{statistics.median(block_seconds[1:]) * 1000.0:.3f}"
                if len(block_seconds) > 1
                else "unavailable"
            )
            print(
                "speculative-bench-trajectory "
                f"blocks={args.trajectory_blocks} block_tokens={len(proposal_ids)} "
                f"matched_tokens={required_tokens} exact=true "
                f"linear_target_cache={str(args.linear_target_cache).lower()} "
                f"elapsed_ms={trajectory_seconds * 1000.0:.3f} "
                f"first_block_ms={block_seconds[0] * 1000.0:.3f} "
                f"steady_block_ms={steady_block_ms} "
                f"target_ceiling_tokens_s={required_tokens / trajectory_seconds:.3f} "
                f"active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
                flush=True,
            )
            if args.linear_target_cache:
                return 0

        serial_seconds = _median_seconds(
            lambda: _serial_target_block(proposal_ids, serial_session),
            args.rounds,
        )
        verify_seconds = _median_seconds(
            lambda: speculative.verify_greedy_block(proposal_ids, verifier),
            args.rounds,
        )
        print(
            "speculative-bench-result kind=all-accepted "
            f"serial_ms={serial_seconds * 1000.0:.3f} "
            f"verify_ms={verify_seconds * 1000.0:.3f} "
            f"target_speedup={serial_seconds / verify_seconds:.3f} "
            f"emitted_tokens={len(reference.emitted_tokens)} "
            f"target_ceiling_tokens_s={len(reference.emitted_tokens) / verify_seconds:.3f}",
            flush=True,
        )

        if args.sweep_prefixes:
            for length in range(1, len(proposal_ids)):
                prefix = proposal_ids[:length]
                checked, _ = speculative.verify_greedy_block(prefix, verifier)
                require(checked.all_accepted, "target prefix unexpectedly rejected")
                prefix_serial = _median_seconds(
                    lambda prefix=prefix: _serial_target_block(prefix, serial_session),
                    args.rounds,
                )
                prefix_verify = _median_seconds(
                    lambda prefix=prefix: speculative.verify_greedy_block(prefix, verifier),
                    args.rounds,
                )
                print(
                    "speculative-bench-result kind=prefix-sweep "
                    f"proposal_tokens={length} serial_ms={prefix_serial * 1000.0:.3f} "
                    f"verify_ms={prefix_verify * 1000.0:.3f} "
                    f"target_speedup={prefix_serial / prefix_verify:.3f} "
                    f"emitted_tokens={len(checked.emitted_tokens)}",
                    flush=True,
                )

        for mismatch in range(1, len(proposal_ids)):
            changed = list(proposal_ids)
            changed[mismatch] = (changed[mismatch] + 1) % model.PRODUCTION_CONFIG.vocab_size
            changed_ids = tuple(changed)
            checked, _ = speculative.verify_greedy_block(changed_ids, verifier)
            require(checked.accepted_count == mismatch, "forced mismatch landed at wrong position")
            expected_transition = model.prefill_hidden_chunk(
                proposal_ids[:mismatch],
                cursor.state,
                weights,
                use_steel=False,
                _validated=True,
            )
            expected_logits = model.project_lm_head(
                weights.lm_head,
                expected_transition.hidden[-1],
            )
            model.evaluate_chunk_transition(expected_transition)
            mx.eval(expected_logits)
            _require_exact_cursor(
                checked.cursor,
                speculative.GreedyTargetCursor(
                    state=expected_transition.state,
                    hidden=expected_transition.hidden[-1],
                    logits=expected_logits,
                ),
            )
            elapsed = _median_seconds(
                lambda changed_ids=changed_ids: speculative.verify_greedy_block(
                    changed_ids,
                    verifier,
                ),
                args.rounds,
            )
            print(
                "speculative-bench-result kind=mismatch "
                f"accepted={mismatch} verify_replay_ms={elapsed * 1000.0:.3f} "
                f"emitted_tokens={len(checked.emitted_tokens)} "
                f"target_ceiling_tokens_s={len(checked.emitted_tokens) / elapsed:.3f} "
                "rollback_exact_tensors=82",
                flush=True,
            )
        return 0
    except (MoEError, TokenizerError, OSError, ValueError) as exc:
        print(f"speculative-bench-error: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

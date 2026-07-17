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
                isinstance(actual_state, attention.MLXAttentionState)
                and isinstance(expected_state, attention.MLXAttentionState),
                "rollback attention type mismatch",
            )
            checks.extend(
                (
                    (f"layer{index}.keys", mx.array_equal(actual_state.keys, expected_state.keys)),
                    (
                        f"layer{index}.values",
                        mx.array_equal(actual_state.values, expected_state.values),
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
        verifier = speculative.start_greedy_verifier(weights, cursor)
        reference, _ = speculative.verify_greedy_block(proposal_ids, verifier)
        require(reference.all_accepted, "target chunk did not accept its serial greedy trajectory")

        print(
            "speculative-bench-ready "
            f"prompt_tokens={len(prompt_ids)} chunks={generate.format_prefill_schedule(schedule)} "
            f"proposal_tokens={len(proposal_ids)} setup_s={time.perf_counter() - started:.3f} "
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
            trajectory_session = verifier
            for block_index in range(args.trajectory_blocks):
                offset = block_index * len(proposal_ids)
                block = tuple(serial_tokens[offset : offset + len(proposal_ids)])
                checked, trajectory_session = speculative.verify_greedy_block(
                    block,
                    trajectory_session,
                )
                expected = tuple(
                    serial_tokens[offset : offset + len(proposal_ids) + 1]
                )
                require(
                    checked.all_accepted and checked.emitted_tokens == expected,
                    f"greedy trajectory diverged at block {block_index}",
                )
            print(
                "speculative-bench-trajectory "
                f"blocks={args.trajectory_blocks} block_tokens={len(proposal_ids)} "
                f"matched_tokens={required_tokens} exact=true",
                flush=True,
            )

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

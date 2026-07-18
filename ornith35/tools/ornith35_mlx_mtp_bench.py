#!/usr/bin/env python3
"""Measure real Qwen3.5 MTP acceptance against the exact Ornith-35 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx

import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_runtime as runtime
import ornith35_mlx_speculative as speculative
from ornith35_moe_reference import MoEError, require
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


NATIVE_CONTEXT_TOKENS = 262_144


def _prefill_target(
    prompt_ids: list[int],
    target_weights: model.TextModelWeights,
    max_chunk: int,
) -> tuple[
    speculative.GreedyTargetCursor,
    mx.array,
    tuple[int, ...],
]:
    schedule = generate.prefill_schedule(len(prompt_ids), max_chunk)
    state = model.initial_state(target_weights, model.PRODUCTION_CONFIG)
    hidden_chunks = []
    offset = 0
    for size in schedule:
        token_slice = prompt_ids[offset : offset + size]
        if size == 1:
            transition = model.forward_hidden_token(
                token_slice[0],
                state,
                target_weights,
            )
            model.evaluate_transition(transition)
            hidden_chunks.append(transition.hidden.reshape(1, -1))
        else:
            transition = model.prefill_hidden_chunk(
                token_slice,
                state,
                target_weights,
                use_steel=False,
            )
            model.evaluate_chunk_transition(transition)
            hidden_chunks.append(transition.hidden)
        state = transition.state
        offset += size
    require(offset == len(prompt_ids), "MTP target prefill is incomplete")
    target_hidden = mx.concatenate(hidden_chunks, axis=0)
    final_hidden = target_hidden[-1]
    logits = model.project_lm_head(target_weights.lm_head, final_hidden)
    mx.eval(target_hidden, logits)
    return (
        speculative.GreedyTargetCursor(
            state=state,
            hidden=final_hidden,
            logits=logits,
        ),
        target_hidden,
        schedule,
    )


def _append_unique_tokens(generated: list[int], emitted: tuple[int, ...]) -> int:
    require(emitted, "MTP verifier emitted no target token")
    if not generated:
        generated.extend(emitted)
        return len(emitted)
    require(generated[-1] == emitted[0], "MTP anchor does not continue prior output")
    generated.extend(emitted[1:])
    return len(emitted) - 1


def _serial_greedy_tokens(
    count: int,
    cursor: speculative.GreedyTargetCursor,
    session: model.TextLinearDecodeSession,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    tokens = []
    times = []
    current = cursor
    for index in range(count):
        started = time.perf_counter()
        token_id = speculative.greedy_token(
            current.logits,
            current.hidden,
            session.weights.lm_head,
        )
        tokens.append(token_id)
        if index + 1 == count:
            break
        result = model.forward_linear_session_token(token_id, session)
        mx.synchronize()
        times.append(time.perf_counter() - started)
        current = speculative.cursor_from_result(result)
    return tuple(tokens), tuple(times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompt",
        default="Complete this Python function and explain edge cases:\n\n"
        "def binary_search(values, target):\n",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--block-tokens", type=int, default=3)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exact-bf16-block-head",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verify-mtp-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--draft-exact-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--adaptation-dir", type=Path)
    parser.add_argument(
        "--allow-diagnostic-adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.steps > 0, "MTP benchmark steps must be positive")
        require(
            2 <= args.block_tokens <= speculative.MAX_PROPOSAL_TOKENS,
            "MTP block token count must be between 2 and 8",
        )
        require(args.log_every > 0, "MTP log interval must be positive")
        tokenizer = load_text_tokenizer(args.root)
        prompt_ids = list(
            tokenizer.encode(
                render_text_prompt(
                    args.prompt,
                    enable_thinking=args.enable_thinking,
                )
            )
        )
        require(prompt_ids, "MTP benchmark prompt produced no tokens")
        capacity = len(prompt_ids) + args.steps * args.block_tokens
        require(
            capacity <= NATIVE_CONTEXT_TOKENS,
            "MTP benchmark exceeds native context",
        )

        setup_started = time.perf_counter()
        target_weights = model.load_text_model(
            args.root,
            map_embedding=True,
            quantize_lm_head=not args.exact_bf16_block_head,
        )
        exact_block_lm_head = (
            target_weights.lm_head if args.exact_bf16_block_head else None
        )
        if exact_block_lm_head is not None:
            require(
                isinstance(exact_block_lm_head, mx.array),
                "exact MTP target head is not BF16",
            )
        mtp_weights = mtp.load_weights(
            args.root,
            verify_hash=args.verify_mtp_hash,
            adaptation_dir=args.adaptation_dir,
            allow_diagnostic_adaptation=args.allow_diagnostic_adaptation,
        )
        cursor, target_hidden, schedule = _prefill_target(
            prompt_ids,
            target_weights,
            args.prefill_chunk,
        )
        anchor = speculative.greedy_token(
            cursor.logits,
            cursor.hidden,
            target_weights.lm_head,
        )
        context_started = time.perf_counter()
        mtp_context = runtime.build_prompt_context(
            prompt_ids,
            target_hidden,
            anchor,
            target_weights.embedding,
            mtp_weights,
            mtp.PRODUCTION_CONFIG,
        )
        context_seconds = time.perf_counter() - context_started
        serial_linear = model.start_linear_decode_session(
            target_weights,
            cursor.state,
            capacity,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=True,
            compile_attention_tails=True,
        )
        target_linear = model.start_linear_decode_session(
            target_weights,
            cursor.state,
            capacity,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=True,
            compile_attention_tails=True,
        )
        linear_cursor = speculative.GreedyTargetCursor(
            state=target_linear.state,
            hidden=cursor.hidden,
            logits=cursor.logits,
        )
        session = runtime.start_greedy_session(
            target_weights,
            linear_cursor,
            mtp_weights,
            mtp_context,
            model.PRODUCTION_CONFIG,
            mtp.PRODUCTION_CONFIG,
            block_tokens=args.block_tokens,
            exact_block_lm_head=exact_block_lm_head,
            target_linear_session=target_linear,
            draft_exact_rerank=args.draft_exact_rerank,
        )

        serial_count = args.steps * args.block_tokens + 1
        serial_tokens, serial_times = _serial_greedy_tokens(
            serial_count,
            cursor,
            serial_linear,
        )
        draft_started = time.perf_counter()
        warm_proposal = runtime.propose(session)
        mx.synchronize()
        draft_seconds = time.perf_counter() - draft_started
        require(
            len(warm_proposal.target_token_ids) == args.block_tokens,
            "MTP warm proposal length mismatch",
        )

        print(
            "mtp-bench-ready "
            f"prompt_tokens={len(prompt_ids)} "
            f"chunks={generate.format_prefill_schedule(schedule)} "
            f"steps={args.steps} block_tokens={args.block_tokens} "
            f"capacity={capacity} setup_s={time.perf_counter() - setup_started:.3f} "
            f"mtp_context_s={context_seconds:.3f} "
            f"warm_proposal_ms={draft_seconds * 1000.0:.3f} "
            f"exact_block_head={str(exact_block_lm_head is not None).lower()} "
            f"draft_exact_rerank={str(args.draft_exact_rerank).lower()} "
            f"adaptation={json.dumps(str(args.adaptation_dir.resolve()) if args.adaptation_dir else None)} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )

        generated: list[int] = []
        step_seconds = []
        step_unique_tokens = []
        future_accepted = []
        all_accepted_blocks = 0
        for step_index in range(args.steps):
            started = time.perf_counter()
            step, session = runtime.step_greedy(session)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            verification = step.verification
            serial_offset = max(len(generated) - 1, 0)
            expected_emitted = serial_tokens[
                serial_offset : serial_offset + len(verification.emitted_tokens)
            ]
            require(
                verification.emitted_tokens == expected_emitted,
                "MTP target verifier diverged from serial greedy target "
                f"at step {step_index}: emitted={verification.emitted_tokens} "
                f"expected={expected_emitted}",
            )
            unique_tokens = _append_unique_tokens(generated, verification.emitted_tokens)
            accepted = max(0, verification.accepted_count - 1)
            step_seconds.append(elapsed)
            step_unique_tokens.append(unique_tokens)
            future_accepted.append(accepted)
            all_accepted_blocks += int(verification.all_accepted)
            if step_index % args.log_every == 0 or step_index + 1 == args.steps:
                print(
                    "mtp-bench-step "
                    f"index={step_index} "
                    f"accepted_future={accepted}/{args.block_tokens - 1} "
                    f"all_accepted={str(verification.all_accepted).lower()} "
                    f"unique_tokens={unique_tokens} "
                    f"elapsed_ms={elapsed * 1000.0:.3f} "
                    f"position={session.verifier.cursor.state.position}",
                    flush=True,
                )

        expected_generated = session.verifier.cursor.state.position - len(prompt_ids) + 1
        require(len(generated) == expected_generated, "MTP output/state length mismatch")
        require(
            tuple(generated) == serial_tokens[: len(generated)],
            "MTP output diverged from serial greedy target",
        )
        transition_count = max(len(generated) - 1, 0)
        serial_seconds = sum(serial_times[:transition_count])
        elapsed_seconds = sum(step_seconds)
        decode_tokens_s = transition_count / elapsed_seconds
        base_tokens_s = transition_count / serial_seconds
        steady_seconds = sum(step_seconds[1:])
        steady_tokens = sum(step_unique_tokens[1:])
        serial_steady_times = serial_times[2:transition_count]
        base_steady_tokens_s = (
            len(serial_steady_times) / sum(serial_steady_times)
            if serial_steady_times
            else base_tokens_s
        )
        mtp_steady_tokens_s = (
            steady_tokens / steady_seconds if steady_seconds else decode_tokens_s
        )
        proposed_future = args.steps * (args.block_tokens - 1)
        histogram = [
            future_accepted.count(count)
            for count in range(args.block_tokens)
        ]
        position_acceptance = [
            sum(accepted >= position for accepted in future_accepted) / args.steps
            for position in range(1, args.block_tokens)
        ]
        print(
            "mtp-bench-result "
            f"exact=true generated_tokens={len(generated)} "
            f"accepted_future={sum(future_accepted)}/{proposed_future} "
            f"acceptance={sum(future_accepted) / proposed_future:.6f} "
            f"mean_accepted_future={statistics.mean(future_accepted):.3f} "
            f"all_accepted_blocks={all_accepted_blocks}/{args.steps} "
            f"elapsed_ms={elapsed_seconds * 1000.0:.3f} "
            f"decode_tokens_s={decode_tokens_s:.3f} "
            f"base_tokens_s={base_tokens_s:.3f} "
            f"speedup={decode_tokens_s / base_tokens_s:.3f} "
            f"steady_tokens_s={mtp_steady_tokens_s:.3f} "
            f"base_steady_tokens_s={base_steady_tokens_s:.3f} "
            f"steady_speedup={mtp_steady_tokens_s / base_steady_tokens_s:.3f} "
            f"median_step_ms={statistics.median(step_seconds) * 1000.0:.3f} "
            f"accepted_histogram={json.dumps(histogram, separators=(',', ':'))} "
            f"position_acceptance={json.dumps(position_acceptance, separators=(',', ':'))} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
        print(
            "mtp-bench-output "
            f"token_ids={json.dumps(generated)} "
            f"text={json.dumps(tokenizer.decode(generated), ensure_ascii=True)}",
            flush=True,
        )
        return 0
    except (MoEError, TokenizerError, OSError, RuntimeError, ValueError) as exc:
        print(f"mtp-bench-error: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

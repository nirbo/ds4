#!/usr/bin/env python3
"""Run the public Ornith-35 DSpark draft against the exact greedy target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx

import ornith35_mlx_dspark as dspark
import ornith35_mlx_dspark_runtime as runtime
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


def _evaluate_aux_transition(
    result: model.TextModelAuxTransition | model.TextModelAuxChunkTransition,
) -> None:
    if isinstance(result, model.TextModelAuxChunkTransition):
        model.evaluate_chunk_transition(result)
    else:
        model.evaluate_transition(result)
    mx.eval(*result.auxiliary_hidden_states)


def _prefill_target_and_draft(
    prompt_ids: list[int],
    target_weights: model.TextModelWeights,
    draft_weights: dspark.MLXDSparkWeights,
    capacity: int,
    max_chunk: int,
) -> tuple[
    speculative.GreedyTargetCursor,
    dspark.MLXDSparkLinearContextState,
    tuple[int, ...],
]:
    schedule = generate.prefill_schedule(len(prompt_ids), max_chunk)
    target_state = model.initial_state(target_weights, model.PRODUCTION_CONFIG)
    draft_context = dspark.initial_linear_context(DSPARK_CONFIG, capacity)
    offset = 0
    final_hidden = None
    for size in schedule:
        token_slice = prompt_ids[offset : offset + size]
        if size == 1:
            result = model.forward_hidden_token_with_aux(
                token_slice[0],
                target_state,
                target_weights,
                DSPARK_CONFIG.aux_hidden_state_indices,
            )
        else:
            result = model.prefill_hidden_chunk_with_aux(
                token_slice,
                target_state,
                target_weights,
                DSPARK_CONFIG.aux_hidden_state_indices,
                use_steel=False,
            )
        _evaluate_aux_transition(result)
        draft_context = runtime.append_target_auxiliary(
            draft_context,
            result,
            draft_weights,
            _validated=True,
        )
        target_state = result.state
        final_hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
        offset += size
    require(final_hidden is not None and offset == len(prompt_ids), "DSpark prefill is incomplete")
    mx.eval(*draft_context.keys, *draft_context.values)
    logits = model.project_lm_head(target_weights.lm_head, final_hidden)
    mx.eval(logits)
    return (
        speculative.GreedyTargetCursor(
            state=target_state,
            hidden=final_hidden,
            logits=logits,
        ),
        draft_context,
        schedule,
    )


def _append_unique_tokens(generated: list[int], emitted: tuple[int, ...]) -> int:
    require(emitted, "DSpark verifier emitted no target token")
    if not generated:
        generated.extend(emitted)
        return len(emitted)
    require(generated[-1] == emitted[0], "DSpark anchor does not continue prior output")
    generated.extend(emitted[1:])
    return len(emitted) - 1


def _serial_greedy_tokens(
    count: int,
    cursor: speculative.GreedyTargetCursor,
    session: model.TextLinearDecodeSession,
) -> tuple[int, ...]:
    tokens = []
    current_cursor = cursor
    for index in range(count):
        token_id = speculative.greedy_token(
            current_cursor.logits,
            current_cursor.hidden,
            session.weights.lm_head,
        )
        tokens.append(token_id)
        if index + 1 == count:
            break
        result = model.forward_linear_session_token(token_id, session)
        current_cursor = speculative.cursor_from_result(result)
    return tuple(tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompt",
        default="Complete this Python function:\n\ndef binary_search(values, target):\n",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--draft-rounds", type=int, default=3)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--target-stage-tokens", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.steps > 0, "DSpark benchmark steps must be positive")
        require(args.draft_rounds > 0, "DSpark draft rounds must be positive")
        require(args.log_every > 0, "DSpark log interval must be positive")
        require(
            args.target_stage_tokens == 0
            or 1 <= args.target_stage_tokens < DSPARK_CONFIG.block_size,
            "DSpark target-stage token count is invalid",
        )
        tokenizer = load_text_tokenizer(args.root)
        prompt_ids = list(tokenizer.encode(render_text_prompt(args.prompt)))
        require(prompt_ids, "DSpark benchmark prompt produced no tokens")
        capacity = len(prompt_ids) + args.steps * DSPARK_CONFIG.block_size
        require(
            capacity <= DSPARK_CONFIG.max_position_embeddings,
            "DSpark benchmark exceeds native context",
        )

        started = time.perf_counter()
        target_weights = model.load_text_model(
            args.root,
            map_embedding=True,
            quantize_lm_head=True,
        )
        draft_weights = dspark.load_weights(dspark.require_verified_source(args.root))
        exact_block_lm_head = (
            None
            if args.target_stage_tokens == 1
            else model.load_exact_block_lm_head(args.root)
        )
        cursor, draft_context, schedule = _prefill_target_and_draft(
            prompt_ids,
            target_weights,
            draft_weights,
            capacity,
            args.prefill_chunk,
        )
        serial_session = model.start_linear_decode_session(
            target_weights,
            cursor.state,
            capacity,
            compile_gdn_layers=True,
            compile_attention_tails=True,
        )
        target_linear = model.start_linear_decode_session(
            target_weights,
            cursor.state,
            capacity,
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
            draft_weights,
            draft_context,
            exact_block_lm_head=exact_block_lm_head,
            target_linear_session=target_linear,
            target_stage_tokens=args.target_stage_tokens,
            compile_prefill_tails=args.target_stage_tokens != 1,
        )
        anchor = speculative.greedy_token(
            cursor.logits,
            cursor.hidden,
            target_weights.lm_head,
        )

        def draft_once() -> None:
            proposal = dspark.propose(
                anchor,
                draft_context,
                draft_weights,
                _validated=True,
            )
            mx.eval(proposal.target_token_ids, proposal.confidence)

        draft_seconds = _median_seconds(draft_once, args.draft_rounds)
        setup_seconds = time.perf_counter() - started
        print(
            "dspark-bench-ready "
            f"prompt_tokens={len(prompt_ids)} "
            f"chunks={generate.format_prefill_schedule(schedule)} "
            f"steps={args.steps} capacity={capacity} "
            f"target_stage_tokens={args.target_stage_tokens} "
            f"exact_block_head={str(exact_block_lm_head is not None).lower()} "
            f"setup_s={setup_seconds:.3f} draft_ms={draft_seconds * 1000.0:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )

        generated: list[int] = []
        step_seconds = []
        step_unique_tokens = []
        future_accepted = []
        all_accepted_blocks = 0
        confidences = []
        first_confidences = []
        for step_index in range(args.steps):
            step_started = time.perf_counter()
            step, session = runtime.step_greedy(session)
            mx.eval(
                step.proposal.confidence,
                *session.draft_context.keys,
                *session.draft_context.values,
            )
            mx.synchronize()
            elapsed = time.perf_counter() - step_started
            verification = step.verification
            unique_tokens = _append_unique_tokens(generated, verification.emitted_tokens)
            accepted = max(0, verification.accepted_count - 1)
            step_seconds.append(elapsed)
            step_unique_tokens.append(unique_tokens)
            future_accepted.append(accepted)
            all_accepted_blocks += int(verification.all_accepted)
            proposal_confidences = [
                float(value) for value in step.proposal.confidence.tolist()
            ]
            confidences.extend(proposal_confidences)
            first_confidences.append(proposal_confidences[0])
            if step_index % args.log_every == 0 or step_index + 1 == args.steps:
                print(
                    "dspark-bench-step "
                    f"index={step_index} accepted_future={accepted}/7 "
                    f"all_accepted={str(verification.all_accepted).lower()} "
                    f"unique_tokens={unique_tokens} elapsed_ms={elapsed * 1000.0:.3f} "
                    f"position={session.verifier.cursor.state.position}",
                    flush=True,
                )

        expected_generated = session.verifier.cursor.state.position - len(prompt_ids) + 1
        require(len(generated) == expected_generated, "DSpark output/state length mismatch")
        serial_started = time.perf_counter()
        serial = _serial_greedy_tokens(len(generated), cursor, serial_session)
        serial_seconds = time.perf_counter() - serial_started
        require(tuple(generated) == serial, "DSpark output diverged from serial greedy target")

        elapsed_seconds = sum(step_seconds)
        measured_tokens = max(len(generated) - 1, 0)
        decode_tokens_s = measured_tokens / elapsed_seconds
        base_tokens_s = measured_tokens / serial_seconds
        steady_seconds = sum(step_seconds[1:])
        steady_tokens = sum(step_unique_tokens[1:])
        proposed_future = args.steps * (DSPARK_CONFIG.block_size - 1)
        accepted_histogram = [
            future_accepted.count(count)
            for count in range(DSPARK_CONFIG.block_size)
        ]
        position_acceptance = [
            sum(accepted >= position for accepted in future_accepted) / args.steps
            for position in range(1, DSPARK_CONFIG.block_size)
        ]
        accepted_first_confidences = [
            confidence
            for confidence, accepted in zip(first_confidences, future_accepted)
            if accepted > 0
        ]
        rejected_first_confidences = [
            confidence
            for confidence, accepted in zip(first_confidences, future_accepted)
            if accepted == 0
        ]
        accepted_confidence_mean = (
            f"{statistics.mean(accepted_first_confidences):.6f}"
            if accepted_first_confidences
            else "unavailable"
        )
        rejected_confidence_mean = (
            f"{statistics.mean(rejected_first_confidences):.6f}"
            if rejected_first_confidences
            else "unavailable"
        )
        profile_rows = [
            [round(confidence, 6), accepted, round(elapsed * 1000.0, 3)]
            for confidence, accepted, elapsed in zip(
                first_confidences,
                future_accepted,
                step_seconds,
            )
        ]
        print(
            "dspark-bench-result "
            f"exact=true generated_tokens={len(generated)} "
            f"accepted_future={sum(future_accepted)}/{proposed_future} "
            f"acceptance={sum(future_accepted) / proposed_future:.6f} "
            f"mean_accepted_future={statistics.mean(future_accepted):.3f} "
            f"all_accepted_blocks={all_accepted_blocks}/{args.steps} "
            f"elapsed_ms={elapsed_seconds * 1000.0:.3f} "
            f"decode_tokens_s={decode_tokens_s:.3f} "
            f"base_tokens_s={base_tokens_s:.3f} "
            f"speedup={decode_tokens_s / base_tokens_s:.3f} "
            f"steady_tokens_s={steady_tokens / steady_seconds if steady_seconds else 0.0:.3f} "
            f"median_step_ms={statistics.median(step_seconds) * 1000.0:.3f} "
            f"mean_confidence={statistics.mean(confidences):.6f} "
            f"first_confidence_accepted_mean={accepted_confidence_mean} "
            f"first_confidence_rejected_mean={rejected_confidence_mean} "
            f"accepted_histogram={json.dumps(accepted_histogram, separators=(',', ':'))} "
            f"position_acceptance={json.dumps(position_acceptance, separators=(',', ':'))} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
        print(
            "dspark-bench-profile "
            f"confidence_acceptance_elapsed={json.dumps(profile_rows, separators=(',', ':'))}",
            flush=True,
        )
        print(
            "dspark-bench-output "
            f"token_ids={json.dumps(generated)} "
            f"text={json.dumps(tokenizer.decode(generated), ensure_ascii=True)}",
            flush=True,
        )
        return 0
    except (dspark.MLXDSparkError, MoEError, TokenizerError, OSError, ValueError) as exc:
        print(f"dspark-bench-error: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exact greedy target verification for Ornith-35 speculative blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_model as model
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import require


MAX_PROPOSAL_TOKENS: Final = 8
_SESSION_SEAL = object()


@dataclass(frozen=True)
class GreedyTargetCursor:
    """Target state whose hidden/logits predict the next unconsumed token."""

    state: model.TextModelState
    hidden: mx.array
    logits: mx.array


@dataclass(frozen=True)
class GreedyVerifierSession:
    """Deeply validated immutable verifier state."""

    weights: model.TextModelWeights
    cursor: GreedyTargetCursor
    config: model.TextModelConfig
    block_tokens: int
    _compiled_prefill_tails: model.CompiledPrefillTails | None
    _exact_block_lm_head: mx.array | None
    _seal: object


@dataclass(frozen=True)
class GreedyBlockVerification:
    """One exact target decision over a proposed greedy token block.

    The returned cursor has consumed only ``committed_tokens``. The final
    emitted replacement or bonus token is still predicted by that cursor and
    must be consumed before a later draft proposal is built from its state.
    """

    proposal_ids: tuple[int, ...]
    verified_target_ids: tuple[int, ...]
    committed_tokens: tuple[int, ...]
    emitted_tokens: tuple[int, ...]
    accepted_count: int
    all_accepted: bool
    target_forward_tokens: int
    rollback_replay_tokens: int
    rollback_recurrent_tokens: int
    cursor: GreedyTargetCursor


def _validate_cursor(
    cursor: GreedyTargetCursor,
    config: model.TextModelConfig,
) -> None:
    require(isinstance(cursor, GreedyTargetCursor), "invalid greedy target cursor")
    model.validate_state(cursor.state, config)
    require(
        cursor.hidden.shape == (config.hidden_size,),
        "greedy cursor hidden shape mismatch",
    )
    require(
        cursor.logits.shape == (config.vocab_size,),
        "greedy cursor logits shape mismatch",
    )
    require(cursor.hidden.dtype == cursor.logits.dtype, "greedy cursor dtype mismatch")


def cursor_from_result(
    result: model.TextModelResult | model.TextModelChunkResult,
) -> GreedyTargetCursor:
    """Create a next-token cursor from an evaluated target result."""
    require(
        isinstance(result, (model.TextModelResult, model.TextModelChunkResult)),
        "invalid target result",
    )
    hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
    require(result.logits.ndim == 1, "target result logits must be a vector")
    return GreedyTargetCursor(state=result.state, hidden=hidden, logits=result.logits)


def start_greedy_verifier(
    weights: model.TextModelWeights,
    cursor: GreedyTargetCursor,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    *,
    block_tokens: int = 8,
    compile_prefill_tails: bool = True,
    exact_block_lm_head: mx.array | None = None,
) -> GreedyVerifierSession:
    """Validate model ownership once before repeated speculative verification."""
    model.validate_weights(weights, config)
    require(
        isinstance(cursor, GreedyTargetCursor)
        and len(cursor.state.layers) == len(config.layer_types)
        and all(
            kind != model.LAYER_ATTENTION
            or isinstance(layer_state, attention.MLXAttentionState)
            for kind, layer_state in zip(config.layer_types, cursor.state.layers)
        ),
        "greedy verifier requires immutable attention state",
    )
    _validate_cursor(cursor, config)
    require(
        cursor.hidden.dtype == model.matrix_dtype(weights.embedding),
        "greedy cursor/model dtype mismatch",
    )
    require(
        1 <= block_tokens <= MAX_PROPOSAL_TOKENS,
        "greedy verifier block token count is invalid",
    )
    compiled_prefill_tails = model.build_compiled_prefill_tails(
        weights,
        config,
        block_tokens,
        enabled=compile_prefill_tails,
    )
    if exact_block_lm_head is not None:
        require(
            exact_block_lm_head.dtype == mx.bfloat16
            and exact_block_lm_head.shape == (config.vocab_size, config.hidden_size),
            "exact block LM-head mismatch",
        )
    return GreedyVerifierSession(
        weights=weights,
        cursor=cursor,
        config=config,
        block_tokens=block_tokens,
        _compiled_prefill_tails=compiled_prefill_tails,
        _exact_block_lm_head=exact_block_lm_head,
        _seal=_SESSION_SEAL,
    )


def greedy_token(
    logits: mx.array,
    hidden: mx.array,
    lm_head: mx.array | vocab.MLXAffineQuantizedMatrix,
) -> int:
    """Select the same deterministic greedy token as the target generator."""
    require(logits.ndim == 1, "greedy target logits must be a vector")
    if isinstance(lm_head, vocab.MLXAffineQuantizedMatrix) and lm_head.reference is not None:
        token_ids, values = vocab.exact_candidate_scores(
            lm_head,
            logits,
            hidden,
            candidate_count=64,
        )
        return min(
            zip(token_ids, values),
            key=lambda item: (-item[1], item[0]),
        )[0]
    return int(mx.argmax(logits).item())


def _project_block_logits(
    lm_head: mx.array | vocab.MLXAffineQuantizedMatrix,
    hidden: mx.array,
    exact_block_lm_head: mx.array | None = None,
) -> mx.array:
    require(hidden.ndim == 2 and hidden.shape[0] > 0, "invalid verifier hidden block")
    if exact_block_lm_head is not None:
        return vocab.project_bf16_block_exact(exact_block_lm_head, hidden)
    # Preserve the generator's single-token GEMV reduction order. MLX's batch
    # GEMM differs in low bits and can perturb the hybrid Q8 candidate pool.
    return mx.stack([model.project_lm_head(lm_head, row) for row in hidden])


def _session_with_cursor(
    session: GreedyVerifierSession,
    cursor: GreedyTargetCursor,
) -> GreedyVerifierSession:
    return GreedyVerifierSession(
        weights=session.weights,
        cursor=cursor,
        config=session.config,
        block_tokens=session.block_tokens,
        _compiled_prefill_tails=session._compiled_prefill_tails,
        _exact_block_lm_head=session._exact_block_lm_head,
        _seal=_SESSION_SEAL,
    )


def _rollback_from_gdn_inputs(
    accepted: int,
    original: model.TextModelState,
    verified: model.TextModelState,
    weights: model.TextModelWeights,
    config: model.TextModelConfig,
    gdn_inputs: tuple[mx.array, ...],
) -> model.TextModelState:
    """Restore an accepted prefix without replaying attention or MoE layers."""
    verified_tokens = verified.position - original.position
    require(0 < accepted < verified_tokens, "invalid rollback length")
    next_states = []
    gdn_index = 0
    next_position = original.position + accepted
    for kind, original_state, verified_state, layer_weights in zip(
        config.layer_types,
        original.layers,
        verified.layers,
        weights.layers,
    ):
        if kind == model.LAYER_GDN:
            require(isinstance(original_state, gdn.MLXGDNState), "rollback GDN state mismatch")
            require(
                isinstance(layer_weights, layer.GDNLayerWeights),
                "rollback GDN weights mismatch",
            )
            rollback_input = gdn_inputs[gdn_index][:accepted]
            _, next_state = gdn.prefill_chunk(
                rollback_input,
                original_state,
                layer_weights.token_mixer,
                config.gdn,
            )
            next_states.append(next_state)
            gdn_index += 1
            continue
        require(
            isinstance(original_state, attention.MLXAttentionState)
            and isinstance(verified_state, attention.MLXAttentionState),
            "rollback requires immutable attention state",
        )
        next_states.append(
            attention.MLXAttentionState(
                keys=verified_state.keys[:, :next_position],
                values=verified_state.values[:, :next_position],
            )
        )
    require(gdn_index == len(gdn_inputs), "rollback GDN journal mismatch")
    state = model.TextModelState(position=next_position, layers=tuple(next_states))
    model.evaluate_state(state)
    return state


def verify_greedy_block(
    proposal_ids: tuple[int, ...] | list[int],
    session: GreedyVerifierSession,
    *,
    exact_long_attention: bool = True,
) -> tuple[GreedyBlockVerification, GreedyVerifierSession]:
    """Verify one proposal block and return an exact rollback-safe cursor."""
    require(
        isinstance(session, GreedyVerifierSession) and session._seal is _SESSION_SEAL,
        "invalid greedy verifier session",
    )
    proposals = tuple(proposal_ids)
    require(
        1 <= len(proposals) <= MAX_PROPOSAL_TOKENS,
        f"proposal block must contain 1 through {MAX_PROPOSAL_TOKENS} tokens",
    )
    require(
        all(
            isinstance(token_id, int) and 0 <= token_id < session.config.vocab_size
            for token_id in proposals
        ),
        "proposal token ID is out of range",
    )

    anchor_target = greedy_token(
        session.cursor.logits,
        session.cursor.hidden,
        session.weights.lm_head,
    )
    if proposals[0] != anchor_target:
        verification = GreedyBlockVerification(
            proposal_ids=proposals,
            verified_target_ids=(anchor_target,),
            committed_tokens=(),
            emitted_tokens=(anchor_target,),
            accepted_count=0,
            all_accepted=False,
            target_forward_tokens=0,
            rollback_replay_tokens=0,
            rollback_recurrent_tokens=0,
            cursor=session.cursor,
        )
        return verification, session

    transition, gdn_inputs = model.prefill_hidden_chunk_with_gdn_rollback(
        proposals,
        session.cursor.state,
        session.weights,
        session.config,
        use_steel=False,
        exact_long_attention=exact_long_attention,
        compiled_prefill_tails=(
            session._compiled_prefill_tails
            if len(proposals) == session.block_tokens
            else None
        ),
        _validated=True,
    )
    block_logits = _project_block_logits(
        session.weights.lm_head,
        transition.hidden,
        session._exact_block_lm_head,
    )
    model.evaluate_chunk_transition(transition)
    mx.eval(block_logits)

    verified = [anchor_target]
    for index in range(1, len(proposals)):
        target_id = (
            int(mx.argmax(block_logits[index - 1]).item())
            if session._exact_block_lm_head is not None
            else greedy_token(
                block_logits[index - 1],
                transition.hidden[index - 1],
                session.weights.lm_head,
            )
        )
        verified.append(target_id)
        if proposals[index] == target_id:
            continue

        accepted = index
        replay_state = _rollback_from_gdn_inputs(
            accepted,
            session.cursor.state,
            transition.state,
            session.weights,
            session.config,
            gdn_inputs,
        )
        next_logits = (
            model.project_lm_head(
                session.weights.lm_head,
                transition.hidden[accepted - 1],
            )
            if session._exact_block_lm_head is not None
            else block_logits[accepted - 1]
        )
        mx.eval(next_logits)
        next_cursor = GreedyTargetCursor(
            state=replay_state,
            hidden=transition.hidden[accepted - 1],
            logits=next_logits,
        )
        next_session = _session_with_cursor(session, next_cursor)
        verification = GreedyBlockVerification(
            proposal_ids=proposals,
            verified_target_ids=tuple(verified),
            committed_tokens=proposals[:accepted],
            emitted_tokens=proposals[:accepted] + (target_id,),
            accepted_count=accepted,
            all_accepted=False,
            target_forward_tokens=len(proposals),
            rollback_replay_tokens=0,
            rollback_recurrent_tokens=accepted,
            cursor=next_cursor,
        )
        return verification, next_session

    bonus_id = (
        int(mx.argmax(block_logits[-1]).item())
        if session._exact_block_lm_head is not None
        else greedy_token(
            block_logits[-1],
            transition.hidden[-1],
            session.weights.lm_head,
        )
    )
    verified.append(bonus_id)
    next_logits = (
        model.project_lm_head(session.weights.lm_head, transition.hidden[-1])
        if session._exact_block_lm_head is not None
        else block_logits[-1]
    )
    mx.eval(next_logits)
    next_cursor = GreedyTargetCursor(
        state=transition.state,
        hidden=transition.hidden[-1],
        logits=next_logits,
    )
    next_session = _session_with_cursor(session, next_cursor)
    verification = GreedyBlockVerification(
        proposal_ids=proposals,
        verified_target_ids=tuple(verified),
        committed_tokens=proposals,
        emitted_tokens=proposals + (bonus_id,),
        accepted_count=len(proposals),
        all_accepted=True,
        target_forward_tokens=len(proposals),
        rollback_replay_tokens=0,
        rollback_recurrent_tokens=0,
        cursor=next_cursor,
    )
    return verification, next_session

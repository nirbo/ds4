#!/usr/bin/env python3
"""Exact greedy and sampled target verification for Ornith-35 blocks."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Final

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_model as model
import ornith35_mlx_sampling as sampling
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
    """Deeply validated verifier state and optional single-owner K/V binding."""

    weights: model.TextModelWeights
    cursor: GreedyTargetCursor
    config: model.TextModelConfig
    block_tokens: int
    _compiled_prefill_tails: model.CompiledPrefillTails | None
    _exact_block_lm_head: mx.array | None
    _auxiliary_hidden_state_indices: tuple[int, ...]
    _linear_session: model.TextLinearDecodeSession | None
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
    auxiliary_hidden_state_indices: tuple[int, ...] = ()
    committed_auxiliary_hidden_states: tuple[mx.array, ...] = ()
    committed_hidden_states: mx.array | None = None


@dataclass(frozen=True)
class _TargetBlock:
    transition: model.TextModelChunkTransition
    gdn_inputs: tuple[mx.array, ...]
    block_logits: mx.array
    exact_target_ids: tuple[int, ...] | None
    captured_auxiliary: tuple[mx.array, ...]


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
    auxiliary_hidden_state_indices: tuple[int, ...] = (),
    linear_session: model.TextLinearDecodeSession | None = None,
) -> GreedyVerifierSession:
    """Validate model ownership once before repeated speculative verification."""
    model.validate_weights(weights, config)
    _validate_cursor(cursor, config)
    require(
        cursor.hidden.dtype == model.matrix_dtype(weights.embedding),
        "greedy cursor/model dtype mismatch",
    )
    require(
        1 <= block_tokens <= MAX_PROPOSAL_TOKENS,
        "greedy verifier block token count is invalid",
    )
    if linear_session is None:
        require(
            all(
                kind != model.LAYER_ATTENTION
                or isinstance(layer_state, attention.MLXAttentionState)
                for kind, layer_state in zip(config.layer_types, cursor.state.layers)
            ),
            "greedy verifier requires immutable attention state or a linear owner",
        )
    else:
        model.validate_linear_decode_session(linear_session)
        require(linear_session.weights is weights, "linear verifier weights mismatch")
        require(linear_session.config == config, "linear verifier config mismatch")
        require(
            linear_session.state is cursor.state,
            "linear verifier cursor is not the owned session state",
        )
        require(
            all(
                kind != model.LAYER_ATTENTION
                or isinstance(layer_state, attention.MLXLinearAttentionState)
                for kind, layer_state in zip(config.layer_types, cursor.state.layers)
            ),
            "linear verifier requires fixed-capacity attention state",
        )
        require(
            cursor.state.position + block_tokens <= linear_session.capacity,
            "linear verifier capacity cannot hold one target block",
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
    auxiliary_indices = (
        model.validate_aux_hidden_state_indices(auxiliary_hidden_state_indices, config)
        if auxiliary_hidden_state_indices
        else ()
    )
    return GreedyVerifierSession(
        weights=weights,
        cursor=cursor,
        config=config,
        block_tokens=block_tokens,
        _compiled_prefill_tails=compiled_prefill_tails,
        _exact_block_lm_head=exact_block_lm_head,
        _auxiliary_hidden_state_indices=auxiliary_indices,
        _linear_session=linear_session,
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
        _auxiliary_hidden_state_indices=session._auxiliary_hidden_state_indices,
        _linear_session=session._linear_session,
        _seal=_SESSION_SEAL,
    )


def _validate_verifier_session(session: GreedyVerifierSession) -> None:
    require(
        isinstance(session, GreedyVerifierSession) and session._seal is _SESSION_SEAL,
        "invalid greedy verifier session",
    )
    if session._linear_session is not None:
        model.validate_linear_decode_session(session._linear_session)
        require(
            session._linear_session.state is session.cursor.state,
            "stale linear verifier cursor",
        )


def _forward_target_token_validated(
    session: GreedyVerifierSession,
    anchor_target: int,
) -> tuple[GreedyTargetCursor, tuple[mx.array, ...]]:
    captured_auxiliary: tuple[mx.array, ...] = ()
    if session._linear_session is not None:
        if session._auxiliary_hidden_state_indices:
            result = model.forward_linear_session_token_with_aux(
                anchor_target,
                session._linear_session,
                session._auxiliary_hidden_state_indices,
            )
            captured_auxiliary = tuple(
                value.reshape(1, value.shape[0])
                for value in result.auxiliary_hidden_states
            )
        else:
            result = model.forward_linear_session_token(
                anchor_target,
                session._linear_session,
            )
        next_cursor = cursor_from_result(result)
    elif session._auxiliary_hidden_state_indices:
        transition = model.forward_hidden_token_with_aux(
            anchor_target,
            session.cursor.state,
            session.weights,
            session._auxiliary_hidden_state_indices,
            session.config,
        )
        next_logits = model.project_lm_head(
            session.weights.lm_head,
            transition.hidden,
        )
        model.evaluate_transition(
            transition,
            additional_arrays=(next_logits, *transition.auxiliary_hidden_states),
        )
        captured_auxiliary = tuple(
            value.reshape(1, value.shape[0])
            for value in transition.auxiliary_hidden_states
        )
        next_cursor = GreedyTargetCursor(
            state=transition.state,
            hidden=transition.hidden,
            logits=next_logits,
        )
    else:
        result = model.forward_token(
            anchor_target,
            session.cursor.state,
            session.weights,
            session.config,
        )
        model.evaluate_result(result)
        next_cursor = cursor_from_result(result)
    return next_cursor, captured_auxiliary


def _finish_target_token(
    session: GreedyVerifierSession,
    anchor_target: int,
    next_cursor: GreedyTargetCursor,
    bonus_id: int,
    captured_auxiliary: tuple[mx.array, ...],
) -> tuple[GreedyBlockVerification, GreedyVerifierSession]:
    if session._linear_session is not None:
        require(
            session._linear_session.state is next_cursor.state,
            "linear verifier failed to commit its target state",
        )
    next_session = _session_with_cursor(session, next_cursor)
    verification = GreedyBlockVerification(
        proposal_ids=(anchor_target,),
        verified_target_ids=(anchor_target, bonus_id),
        committed_tokens=(anchor_target,),
        emitted_tokens=(anchor_target, bonus_id),
        accepted_count=1,
        all_accepted=True,
        target_forward_tokens=1,
        rollback_replay_tokens=0,
        rollback_recurrent_tokens=0,
        cursor=next_cursor,
        auxiliary_hidden_state_indices=session._auxiliary_hidden_state_indices,
        committed_auxiliary_hidden_states=captured_auxiliary,
        committed_hidden_states=next_cursor.hidden.reshape(1, -1),
    )
    return verification, next_session


def _advance_greedy_target_validated(
    session: GreedyVerifierSession,
    anchor_target: int,
) -> tuple[GreedyBlockVerification, GreedyVerifierSession]:
    next_cursor, captured_auxiliary = _forward_target_token_validated(
        session,
        anchor_target,
    )
    bonus_id = greedy_token(
        next_cursor.logits,
        next_cursor.hidden,
        session.weights.lm_head,
    )
    return _finish_target_token(
        session,
        anchor_target,
        next_cursor,
        bonus_id,
        captured_auxiliary,
    )


def advance_greedy_target(
    session: GreedyVerifierSession,
    *,
    _validated: bool = False,
) -> tuple[GreedyBlockVerification, GreedyVerifierSession]:
    """Advance one exact greedy target token without speculative bookkeeping."""
    if not _validated:
        _validate_verifier_session(session)
    anchor_target = greedy_token(
        session.cursor.logits,
        session.cursor.hidden,
        session.weights.lm_head,
    )
    return _advance_greedy_target_validated(session, anchor_target)


def advance_sampled_target(
    session: GreedyVerifierSession,
    pending_token_id: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: random.Random,
    _validated: bool = False,
) -> tuple[GreedyBlockVerification, GreedyVerifierSession]:
    """Consume one target-sampled token and sample its exact target successor."""
    if not _validated:
        _validate_verifier_session(session)
    current_distribution = sampling.target_distribution(
        session.cursor.logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        hidden=session.cursor.hidden,
        lm_head=session.weights.lm_head,
    )
    require(
        current_distribution.probability(pending_token_id) > 0.0,
        "pending sampled token has zero target probability",
    )
    next_cursor, captured_auxiliary = _forward_target_token_validated(
        session,
        pending_token_id,
    )
    bonus_id = sampling.target_distribution(
        next_cursor.logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        hidden=next_cursor.hidden,
        lm_head=session.weights.lm_head,
    ).sample(rng)
    return _finish_target_token(
        session,
        pending_token_id,
        next_cursor,
        bonus_id,
        captured_auxiliary,
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
    require(
        original.context_profile == verified.context_profile,
        "rollback context profiles disagree",
    )
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
            next_state = original_state
            # The small-chunk recurrence is numerically close, but it is not
            # universally bit-exact to decode_step. A rejected target block
            # must restore the authoritative one-token state exactly.
            for token_input in rollback_input:
                _, next_state = gdn.decode_step(
                    token_input,
                    next_state,
                    layer_weights.token_mixer,
                    config.gdn,
                    _validated=True,
                )
            next_states.append(next_state)
            gdn_index += 1
            continue
        if isinstance(original_state, attention.MLXAttentionState):
            require(
                isinstance(verified_state, attention.MLXAttentionState),
                "rollback attention-state kinds disagree",
            )
            next_states.append(
                attention.MLXAttentionState(
                    keys=verified_state.keys[:, :next_position],
                    values=verified_state.values[:, :next_position],
                    context_profile=verified_state.context_profile,
                )
            )
            continue
        require(
            isinstance(original_state, attention.MLXLinearAttentionState)
            and isinstance(verified_state, attention.MLXLinearAttentionState)
            and original_state.capacity == verified_state.capacity,
            "rollback linear attention-state kinds disagree",
        )
        next_states.append(
            attention.MLXLinearAttentionState(
                keys=verified_state.keys,
                values=verified_state.values,
                position=next_position,
                capacity=verified_state.capacity,
                context_profile=verified_state.context_profile,
            )
        )
    require(gdn_index == len(gdn_inputs), "rollback GDN journal mismatch")
    state = model.TextModelState(
        position=next_position,
        layers=tuple(next_states),
        context_profile=original.context_profile,
    )
    model.evaluate_state(state)
    return state


def _forward_target_block(
    proposals: tuple[int, ...],
    session: GreedyVerifierSession,
    *,
    exact_long_attention: bool,
) -> _TargetBlock:
    require(len(proposals) > 1, "target block requires multiple proposal tokens")
    compiled_tails = (
        session._compiled_prefill_tails
        if len(proposals) == session.block_tokens
        else None
    )
    captured_auxiliary: tuple[mx.array, ...] = ()
    if session._linear_session is not None:
        if session._auxiliary_hidden_state_indices:
            transition, gdn_inputs = (
                model.prefill_linear_session_chunk_with_gdn_rollback_and_aux(
                    proposals,
                    session._linear_session,
                    session._auxiliary_hidden_state_indices,
                    use_steel=False,
                    exact_long_attention=exact_long_attention,
                    compiled_prefill_tails=compiled_tails,
                )
            )
            captured_auxiliary = transition.auxiliary_hidden_states
        else:
            rollback_inputs: list[mx.array] = []
            transition = model.prefill_linear_session_chunk(
                proposals,
                session._linear_session,
                project_logits=False,
                use_steel=False,
                exact_long_attention=exact_long_attention,
                _gdn_rollback_inputs=rollback_inputs,
                _compiled_prefill_tails=compiled_tails,
            )
            require(
                type(transition) is model.TextModelChunkTransition,
                "linear verifier transition mismatch",
            )
            require(
                len(rollback_inputs) == session.config.layer_types.count(model.LAYER_GDN),
                "GDN rollback-input count mismatch",
            )
            gdn_inputs = tuple(rollback_inputs)
    elif session._auxiliary_hidden_state_indices:
        transition, gdn_inputs = model.prefill_hidden_chunk_with_gdn_rollback_and_aux(
            proposals,
            session.cursor.state,
            session.weights,
            session._auxiliary_hidden_state_indices,
            session.config,
            use_steel=False,
            exact_long_attention=exact_long_attention,
            compiled_prefill_tails=compiled_tails,
            _validated=True,
        )
        captured_auxiliary = transition.auxiliary_hidden_states
    else:
        transition, gdn_inputs = model.prefill_hidden_chunk_with_gdn_rollback(
            proposals,
            session.cursor.state,
            session.weights,
            session.config,
            use_steel=False,
            exact_long_attention=exact_long_attention,
            compiled_prefill_tails=compiled_tails,
            _validated=True,
        )
    block_logits = _project_block_logits(
        session.weights.lm_head,
        transition.hidden,
        session._exact_block_lm_head,
    )
    exact_target_ids_array = (
        mx.argmax(block_logits, axis=1)
        if session._exact_block_lm_head is not None
        else None
    )
    model.evaluate_chunk_transition(
        transition,
        additional_arrays=(
            (exact_target_ids_array,)
            if exact_target_ids_array is not None
            else (block_logits,)
        ),
    )
    exact_target_ids = (
        tuple(int(value) for value in exact_target_ids_array.tolist())
        if exact_target_ids_array is not None
        else None
    )
    return _TargetBlock(
        transition=transition,
        gdn_inputs=gdn_inputs,
        block_logits=block_logits,
        exact_target_ids=exact_target_ids,
        captured_auxiliary=captured_auxiliary,
    )


def _validated_proposals(
    proposal_ids: tuple[int, ...] | list[int],
    session: GreedyVerifierSession,
) -> tuple[int, ...]:
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
    return proposals


def verify_greedy_block(
    proposal_ids: tuple[int, ...] | list[int],
    session: GreedyVerifierSession,
    *,
    exact_long_attention: bool = True,
) -> tuple[GreedyBlockVerification, GreedyVerifierSession]:
    """Verify one proposal block and return an exact rollback-safe cursor."""
    _validate_verifier_session(session)
    proposals = _validated_proposals(proposal_ids, session)

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
            auxiliary_hidden_state_indices=session._auxiliary_hidden_state_indices,
            committed_hidden_states=None,
        )
        return verification, session

    if len(proposals) == 1:
        return _advance_greedy_target_validated(session, anchor_target)

    target = _forward_target_block(
        proposals,
        session,
        exact_long_attention=exact_long_attention,
    )
    transition = target.transition
    gdn_inputs = target.gdn_inputs
    block_logits = target.block_logits
    exact_target_ids = target.exact_target_ids
    captured_auxiliary = target.captured_auxiliary

    verified = [anchor_target]
    for index in range(1, len(proposals)):
        target_id = (
            exact_target_ids[index - 1]
            if exact_target_ids is not None
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
        if session._linear_session is not None:
            model.restore_linear_session_state(session._linear_session, replay_state)
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
            auxiliary_hidden_state_indices=session._auxiliary_hidden_state_indices,
            committed_auxiliary_hidden_states=tuple(
                value[:accepted] for value in captured_auxiliary
            ),
            committed_hidden_states=transition.hidden[:accepted],
        )
        return verification, next_session

    bonus_id = (
        exact_target_ids[-1]
        if exact_target_ids is not None
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
    if session._linear_session is not None:
        require(
            session._linear_session.state is transition.state,
            "linear verifier failed to commit its target state",
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
        auxiliary_hidden_state_indices=session._auxiliary_hidden_state_indices,
        committed_auxiliary_hidden_states=captured_auxiliary,
        committed_hidden_states=transition.hidden,
    )
    return verification, next_session


def verify_sampled_block(
    proposal_ids: tuple[int, ...] | list[int],
    session: GreedyVerifierSession,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: random.Random,
    draft_distributions: tuple[sampling.TokenDistribution, ...] | None = None,
    exact_long_attention: bool = True,
) -> tuple[GreedyBlockVerification, GreedyVerifierSession]:
    """Verify a draft distribution against the exact sampled target distribution.

    Future tokens are accepted with ``min(1, p(x) / q(x))``. Rejection samples
    normalized ``max(p - q, 0)``. A missing draft distribution is interpreted
    as a delta for compatibility. Both forms reproduce the configured target
    top-k/top-p distribution exactly.
    """
    _validate_verifier_session(session)
    require(temperature > 0.0, "sampled verification requires positive temperature")
    proposals = _validated_proposals(proposal_ids, session)
    distributions = (
        draft_distributions
        if draft_distributions is not None
        else tuple(
            sampling.TokenDistribution((token_id,), (1.0,))
            for token_id in proposals[1:]
        )
    )
    require(
        len(distributions) == len(proposals) - 1,
        "sampled draft distribution count mismatch",
    )
    for index, distribution in enumerate(distributions):
        sampling.validate_distribution(distribution)
        require(
            all(token_id < session.config.vocab_size for token_id in distribution.token_ids),
            "sampled draft distribution token is out of range",
        )
        require(
            distribution.probability(proposals[index + 1]) > 0.0,
            "sampled token has zero draft probability",
        )
    anchor_distribution = sampling.target_distribution(
        session.cursor.logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        hidden=session.cursor.hidden,
        lm_head=session.weights.lm_head,
    )
    require(
        anchor_distribution.probability(proposals[0]) > 0.0,
        "sampled proposal anchor has zero target probability",
    )
    if len(proposals) == 1:
        return advance_sampled_target(
            session,
            proposals[0],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
            _validated=True,
        )

    target = _forward_target_block(
        proposals,
        session,
        exact_long_attention=exact_long_attention,
    )
    transition = target.transition
    distribution_head = (
        session._exact_block_lm_head
        if session._exact_block_lm_head is not None
        else session.weights.lm_head
    )
    verified = [proposals[0]]
    for index in range(1, len(proposals)):
        distribution = sampling.target_distribution(
            target.block_logits[index - 1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            hidden=transition.hidden[index - 1],
            lm_head=distribution_head,
        )
        draft_id = proposals[index]
        draft_distribution = distributions[index - 1]
        acceptance = sampling.speculative_acceptance_probability(
            distribution,
            draft_distribution,
            draft_id,
        )
        if rng.random() < acceptance:
            verified.append(draft_id)
            continue

        replacement_id = sampling.residual_distribution(
            distribution,
            draft_distribution,
        ).sample(rng)
        verified.append(replacement_id)
        accepted = index
        replay_state = _rollback_from_gdn_inputs(
            accepted,
            session.cursor.state,
            transition.state,
            session.weights,
            session.config,
            target.gdn_inputs,
        )
        if session._linear_session is not None:
            model.restore_linear_session_state(session._linear_session, replay_state)
        next_logits = (
            model.project_lm_head(
                session.weights.lm_head,
                transition.hidden[accepted - 1],
            )
            if session._exact_block_lm_head is not None
            else target.block_logits[accepted - 1]
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
            emitted_tokens=proposals[:accepted] + (replacement_id,),
            accepted_count=accepted,
            all_accepted=False,
            target_forward_tokens=len(proposals),
            rollback_replay_tokens=0,
            rollback_recurrent_tokens=accepted,
            cursor=next_cursor,
            auxiliary_hidden_state_indices=session._auxiliary_hidden_state_indices,
            committed_auxiliary_hidden_states=tuple(
                value[:accepted] for value in target.captured_auxiliary
            ),
            committed_hidden_states=transition.hidden[:accepted],
        )
        return verification, next_session

    bonus_id = sampling.target_distribution(
        target.block_logits[-1],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        hidden=transition.hidden[-1],
        lm_head=distribution_head,
    ).sample(rng)
    verified.append(bonus_id)
    next_logits = (
        model.project_lm_head(session.weights.lm_head, transition.hidden[-1])
        if session._exact_block_lm_head is not None
        else target.block_logits[-1]
    )
    mx.eval(next_logits)
    next_cursor = GreedyTargetCursor(
        state=transition.state,
        hidden=transition.hidden[-1],
        logits=next_logits,
    )
    if session._linear_session is not None:
        require(
            session._linear_session.state is transition.state,
            "linear sampled verifier failed to commit its target state",
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
        auxiliary_hidden_state_indices=session._auxiliary_hidden_state_indices,
        committed_auxiliary_hidden_states=target.captured_auxiliary,
        committed_hidden_states=transition.hidden,
    )
    return verification, next_session

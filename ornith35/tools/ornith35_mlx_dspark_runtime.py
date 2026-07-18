#!/usr/bin/env python3
"""Exact greedy target integration for the Ornith-35 DSpark draft."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

import ornith35_mlx_dspark as dspark
import ornith35_mlx_model as model
import ornith35_mlx_speculative as speculative
from ornith35_dspark_reference import DSparkConfig, PRODUCTION_CONFIG as DSPARK_CONFIG
from ornith35_moe_reference import require


_SESSION_SEAL = object()
_AUX_RESULT_TYPES = (
    model.TextModelAuxTransition,
    model.TextModelAuxChunkTransition,
    model.TextModelAuxResult,
    model.TextModelAuxChunkResult,
)


@dataclass(frozen=True)
class DSparkGreedySession:
    verifier: speculative.GreedyVerifierSession
    draft_weights: dspark.MLXDSparkWeights
    draft_context: dspark.MLXDSparkContextState | dspark.MLXDSparkLinearContextState
    draft_config: DSparkConfig
    target_stage_tokens: int
    _seal: object


@dataclass(frozen=True)
class DSparkGreedyStep:
    anchor_token_id: int
    proposal: dspark.MLXDSparkProposal
    verification: speculative.GreedyBlockVerification


def append_target_auxiliary(
    context: dspark.MLXDSparkContextState | dspark.MLXDSparkLinearContextState,
    target_result: (
        model.TextModelAuxTransition
        | model.TextModelAuxChunkTransition
        | model.TextModelAuxResult
        | model.TextModelAuxChunkResult
    ),
    draft_weights: dspark.MLXDSparkWeights,
    draft_config: DSparkConfig = DSPARK_CONFIG,
    *,
    _validated: bool = False,
) -> dspark.MLXDSparkContextState | dspark.MLXDSparkLinearContextState:
    """Append exactly the target token states represented by one evaluated result."""
    require(isinstance(target_result, _AUX_RESULT_TYPES), "invalid target auxiliary result")
    require(
        target_result.auxiliary_hidden_state_indices
        == draft_config.aux_hidden_state_indices,
        "target/DSpark auxiliary hidden-state indices disagree",
    )
    auxiliary = tuple(
        value.reshape(1, value.shape[0]) if value.ndim == 1 else value
        for value in target_result.auxiliary_hidden_states
    )
    require(auxiliary and auxiliary[0].ndim == 2, "invalid target auxiliary state rank")
    tokens = auxiliary[0].shape[0]
    require(
        tokens > 0 and all(value.shape == auxiliary[0].shape for value in auxiliary),
        "target auxiliary state shapes disagree",
    )
    require(
        context.position + tokens == target_result.state.position,
        "target/DSpark context positions disagree",
    )
    return dspark.append_context(
        context,
        auxiliary,
        draft_weights,
        draft_config,
        _validated=_validated,
    )


def _validate_session(session: DSparkGreedySession) -> None:
    require(
        isinstance(session, DSparkGreedySession) and session._seal is _SESSION_SEAL,
        "invalid DSpark greedy session",
    )
    verifier = session.verifier
    draft_config = session.draft_config
    require(
        verifier.config.vocab_size == draft_config.target_vocab_size,
        "target/DSpark vocabulary mismatch",
    )
    require(
        verifier.config.hidden_size == draft_config.hidden_size,
        "target/DSpark hidden width mismatch",
    )
    expected_verifier_tokens = session.target_stage_tokens or draft_config.block_size
    require(
        0 <= session.target_stage_tokens < draft_config.block_size
        and verifier.block_tokens == expected_verifier_tokens,
        "target/DSpark verifier stage mismatch",
    )
    require(
        verifier._auxiliary_hidden_state_indices
        == draft_config.aux_hidden_state_indices,
        "target/DSpark auxiliary hidden-state indices disagree",
    )
    require(
        verifier.cursor.state.position == session.draft_context.position,
        "target/DSpark cursor positions disagree",
    )
    if isinstance(session.draft_context, dspark.MLXDSparkLinearContextState):
        dspark.validate_linear_context(
            session.draft_context,
            draft_config,
            session.draft_weights.embedding.dtype,
        )
    else:
        dspark.validate_context(
            session.draft_context,
            draft_config,
            session.draft_weights.embedding.dtype,
        )


def start_greedy_session(
    target_weights: model.TextModelWeights,
    target_cursor: speculative.GreedyTargetCursor,
    draft_weights: dspark.MLXDSparkWeights,
    draft_context: dspark.MLXDSparkContextState | dspark.MLXDSparkLinearContextState,
    target_config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    draft_config: DSparkConfig = DSPARK_CONFIG,
    *,
    compile_prefill_tails: bool = True,
    exact_block_lm_head: mx.array | None = None,
    target_linear_session: model.TextLinearDecodeSession | None = None,
    target_stage_tokens: int = 0,
) -> DSparkGreedySession:
    """Validate target/draft ownership once and create an aligned session."""
    dspark.validate_weights(draft_weights, draft_config)
    require(
        model.matrix_dtype(target_weights.embedding) == draft_weights.embedding.dtype,
        "target/DSpark model dtypes disagree",
    )
    require(
        target_stage_tokens == 0
        or 1 <= target_stage_tokens < draft_config.block_size,
        "DSpark target-stage token count is invalid",
    )
    verifier = speculative.start_greedy_verifier(
        target_weights,
        target_cursor,
        target_config,
        block_tokens=target_stage_tokens or draft_config.block_size,
        compile_prefill_tails=compile_prefill_tails,
        exact_block_lm_head=exact_block_lm_head,
        auxiliary_hidden_state_indices=draft_config.aux_hidden_state_indices,
        linear_session=target_linear_session,
    )
    session = DSparkGreedySession(
        verifier=verifier,
        draft_weights=draft_weights,
        draft_context=draft_context,
        draft_config=draft_config,
        target_stage_tokens=target_stage_tokens,
        _seal=_SESSION_SEAL,
    )
    _validate_session(session)
    return session


def _append_committed_target_states(
    session: DSparkGreedySession,
    verification: speculative.GreedyBlockVerification,
) -> dspark.MLXDSparkContextState | dspark.MLXDSparkLinearContextState:
    committed = len(verification.committed_tokens)
    captured = verification.committed_auxiliary_hidden_states
    require(
        verification.auxiliary_hidden_state_indices
        == session.draft_config.aux_hidden_state_indices,
        "verifier returned the wrong auxiliary hidden states",
    )
    if committed == 0:
        require(not captured, "verifier returned uncommitted auxiliary states")
        return session.draft_context
    require(
        len(captured) == len(session.draft_config.aux_hidden_state_indices)
        and all(
            value.shape == (committed, session.draft_config.hidden_size)
            for value in captured
        ),
        "verifier committed auxiliary-state shape mismatch",
    )
    return dspark.append_context(
        session.draft_context,
        captured,
        session.draft_weights,
        session.draft_config,
        _validated=True,
    )


def _concatenate_staged_auxiliary(
    first: speculative.GreedyBlockVerification,
    second: speculative.GreedyBlockVerification,
) -> tuple[mx.array, ...]:
    require(
        first.auxiliary_hidden_state_indices
        == second.auxiliary_hidden_state_indices,
        "staged verifier auxiliary indices disagree",
    )
    if not first.committed_auxiliary_hidden_states:
        return second.committed_auxiliary_hidden_states
    if not second.committed_auxiliary_hidden_states:
        return first.committed_auxiliary_hidden_states
    require(
        len(first.committed_auxiliary_hidden_states)
        == len(second.committed_auxiliary_hidden_states),
        "staged verifier auxiliary counts disagree",
    )
    return tuple(
        mx.concatenate((left, right), axis=0)
        for left, right in zip(
            first.committed_auxiliary_hidden_states,
            second.committed_auxiliary_hidden_states,
        )
    )


def _verify_staged_greedy_block(
    proposal_ids: tuple[int, ...],
    verifier: speculative.GreedyVerifierSession,
    stage_tokens: int,
    *,
    exact_long_attention: bool,
) -> tuple[
    speculative.GreedyBlockVerification,
    speculative.GreedyVerifierSession,
]:
    """Verify causal chunks until rejection instead of paying for the full block."""
    require(
        1 <= stage_tokens < len(proposal_ids),
        "DSpark target-stage token count is invalid",
    )
    aggregate: speculative.GreedyBlockVerification | None = None
    next_verifier = verifier
    offset = 0
    while offset < len(proposal_ids):
        end = min(offset + stage_tokens, len(proposal_ids))
        stage, next_verifier = speculative.verify_greedy_block(
            proposal_ids[offset:end],
            next_verifier,
            exact_long_attention=exact_long_attention,
        )
        if aggregate is None:
            verified_target_ids = stage.verified_target_ids
            committed_tokens = stage.committed_tokens
            emitted_tokens = stage.emitted_tokens
            captured = stage.committed_auxiliary_hidden_states
            target_forward_tokens = stage.target_forward_tokens
            rollback_replay_tokens = stage.rollback_replay_tokens
            rollback_recurrent_tokens = stage.rollback_recurrent_tokens
        else:
            verified_target_ids = (
                aggregate.verified_target_ids[:-1] + stage.verified_target_ids
            )
            committed_tokens = aggregate.committed_tokens + stage.committed_tokens
            emitted_tokens = aggregate.committed_tokens + stage.emitted_tokens
            captured = _concatenate_staged_auxiliary(aggregate, stage)
            target_forward_tokens = (
                aggregate.target_forward_tokens + stage.target_forward_tokens
            )
            rollback_replay_tokens = (
                aggregate.rollback_replay_tokens + stage.rollback_replay_tokens
            )
            rollback_recurrent_tokens = (
                aggregate.rollback_recurrent_tokens + stage.rollback_recurrent_tokens
            )
        aggregate = speculative.GreedyBlockVerification(
            proposal_ids=proposal_ids,
            verified_target_ids=verified_target_ids,
            committed_tokens=committed_tokens,
            emitted_tokens=emitted_tokens,
            accepted_count=len(committed_tokens),
            all_accepted=(stage.all_accepted and end == len(proposal_ids)),
            target_forward_tokens=target_forward_tokens,
            rollback_replay_tokens=rollback_replay_tokens,
            rollback_recurrent_tokens=rollback_recurrent_tokens,
            cursor=stage.cursor,
            auxiliary_hidden_state_indices=stage.auxiliary_hidden_state_indices,
            committed_auxiliary_hidden_states=captured,
        )
        if not stage.all_accepted or end == len(proposal_ids):
            return aggregate, next_verifier
        if proposal_ids[end] != stage.verified_target_ids[-1]:
            return aggregate, next_verifier
        offset = end
    raise AssertionError("unreachable staged verifier state")


def step_greedy(
    session: DSparkGreedySession,
    *,
    exact_long_attention: bool = True,
) -> tuple[DSparkGreedyStep, DSparkGreedySession]:
    """Propose seven tokens, verify exactly, and commit only the accepted prefix."""
    _validate_session(session)
    anchor = speculative.greedy_token(
        session.verifier.cursor.logits,
        session.verifier.cursor.hidden,
        session.verifier.weights.lm_head,
    )
    proposal = dspark.propose(
        anchor,
        session.draft_context,
        session.draft_weights,
        session.draft_config,
        _validated=True,
    )
    mx.eval(proposal.target_token_ids)
    future_tokens = tuple(int(value) for value in proposal.target_token_ids.tolist())
    require(
        len(future_tokens) == session.draft_config.block_size - 1,
        "DSpark proposal length mismatch",
    )
    proposal_ids = (anchor, *future_tokens)
    if session.target_stage_tokens:
        verification, next_verifier = _verify_staged_greedy_block(
            proposal_ids,
            session.verifier,
            session.target_stage_tokens,
            exact_long_attention=exact_long_attention,
        )
    else:
        verification, next_verifier = speculative.verify_greedy_block(
            proposal_ids,
            session.verifier,
            exact_long_attention=exact_long_attention,
        )
    next_context = _append_committed_target_states(session, verification)
    require(
        next_context.position == next_verifier.cursor.state.position,
        "target/DSpark commit positions diverged",
    )
    next_session = DSparkGreedySession(
        verifier=next_verifier,
        draft_weights=session.draft_weights,
        draft_context=next_context,
        draft_config=session.draft_config,
        target_stage_tokens=session.target_stage_tokens,
        _seal=_SESSION_SEAL,
    )
    return (
        DSparkGreedyStep(
            anchor_token_id=anchor,
            proposal=proposal,
            verification=verification,
        ),
        next_session,
    )

#!/usr/bin/env python3
"""Exact greedy target integration for the Ornith-35 Qwen3.5 MTP sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_model as model
import ornith35_mlx_mtp as mtp
import ornith35_mlx_speculative as speculative
import ornith35_mtp_reference as mtp_reference
from ornith35_moe_reference import require


_SESSION_SEAL = object()
_TARGET_SESSION_SEAL = object()
_ADAPTIVE_SESSION_SEAL = object()


@dataclass(frozen=True)
class MTPAuthoritativeContext:
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState
    hidden: mx.array
    conditioned_token_id: int


@dataclass(frozen=True)
class MTPGreedySession:
    verifier: speculative.GreedyVerifierSession
    mtp_weights: mtp.MLXMTPWeights
    mtp_context: MTPAuthoritativeContext
    mtp_config: mtp_reference.MTPConfig
    draft_exact_rerank: bool
    _seal: object


@dataclass(frozen=True)
class MTPProposal:
    anchor_token_id: int
    future_token_ids: tuple[int, ...]

    @property
    def target_token_ids(self) -> tuple[int, ...]:
        return (self.anchor_token_id, *self.future_token_ids)


@dataclass(frozen=True)
class MTPGreedyStep:
    proposal: MTPProposal
    verification: speculative.GreedyBlockVerification


@dataclass(frozen=True)
class TargetGreedySession:
    """Exact target-only continuation detached from an MTP session."""

    verifier: speculative.GreedyVerifierSession
    _seal: object


@dataclass(frozen=True)
class MTPAdaptivePolicy:
    """One-way MTP fallback policy based on recent accepted future tokens."""

    minimum_mtp_blocks: int = 8
    window_blocks: int = 4
    minimum_future_acceptance: float = 0.70


@dataclass(frozen=True)
class MTPAdaptiveSession:
    active: MTPGreedySession | TargetGreedySession
    policy: MTPAdaptivePolicy
    future_accepted: tuple[int, ...]
    detached_after_mtp_blocks: int | None
    _seal: object

    @property
    def mode(self) -> str:
        return "mtp" if isinstance(self.active, MTPGreedySession) else "target"


def _evaluate_result(result: mtp.MLXMTPResult) -> None:
    mx.eval(
        result.hidden,
        result.state.keys,
        result.state.values,
        result.selected_experts,
        result.routing_weights,
    )


def _evaluate_chunk_result(result: mtp.MLXMTPChunkResult) -> None:
    mx.eval(
        result.hidden,
        result.state.keys,
        result.state.values,
        result.selected_experts,
        result.routing_weights,
    )


def append_authoritative_hidden(
    context_state: attention.MLXAttentionState | attention.MLXLinearAttentionState,
    target_hidden_states: mx.array,
    following_token_ids: Sequence[int],
    target_embedding: (
        mx.array
        | model.vocab.MLXAffineQuantizedMatrix
        | model.vocab.MLXMappedBF16Matrix
    ),
    mtp_weights: mtp.MLXMTPWeights,
    mtp_config: mtp_reference.MTPConfig,
    *,
    _validated: bool = False,
) -> MTPAuthoritativeContext:
    """Append target-authoritative hidden/token pairs to advancing MTP KV."""
    tokens = tuple(following_token_ids)
    require(tokens, "MTP authoritative append is empty")
    require(
        target_hidden_states.ndim == 2
        and target_hidden_states.shape == (len(tokens), mtp_config.hidden_size),
        "MTP authoritative hidden-state shape mismatch",
    )
    require(
        all(isinstance(token_id, int) and token_id >= 0 for token_id in tokens),
        "invalid MTP following token ID",
    )
    if not _validated:
        mtp.validate_weights(mtp_weights, mtp_config)
        require(
            attention.state_length(context_state, mtp_config.attention) >= 0,
            "invalid MTP context state",
        )
    if len(tokens) > 1:
        token_embeddings = model.embed_tokens(target_embedding, tokens)
        chunk = mtp.prefill_steps(
            token_embeddings,
            target_hidden_states,
            context_state,
            mtp_weights,
            mtp_config,
            _validated=True,
        )
        _evaluate_chunk_result(chunk)
        return MTPAuthoritativeContext(
            state=chunk.state,
            hidden=chunk.hidden[-1],
            conditioned_token_id=tokens[-1],
        )
    token_embedding = model.embed_token(target_embedding, tokens[0])
    result = mtp.forward_step(
        token_embedding,
        target_hidden_states[0],
        context_state,
        mtp_weights,
        mtp_config,
        _validated=True,
    )
    _evaluate_result(result)
    return MTPAuthoritativeContext(
        state=result.state,
        hidden=result.hidden,
        conditioned_token_id=tokens[-1],
    )


def build_prompt_context(
    prompt_token_ids: Sequence[int],
    target_hidden_states: mx.array,
    pending_token_id: int,
    target_embedding: (
        mx.array
        | model.vocab.MLXAffineQuantizedMatrix
        | model.vocab.MLXMappedBF16Matrix
    ),
    mtp_weights: mtp.MLXMTPWeights,
    mtp_config: mtp_reference.MTPConfig,
) -> MTPAuthoritativeContext:
    """Build the vLLM-compatible shifted MTP context for one complete prompt."""
    prompt = tuple(prompt_token_ids)
    require(prompt, "MTP prompt context is empty")
    require(
        target_hidden_states.shape == (len(prompt), mtp_config.hidden_size),
        "MTP prompt target-hidden shape mismatch",
    )
    following = (*prompt[1:], pending_token_id)
    initial = attention.zeros_state(
        mtp_config.attention,
        dtype=mtp_weights.fc.dtype,
    )
    return append_authoritative_hidden(
        initial,
        target_hidden_states,
        following,
        target_embedding,
        mtp_weights,
        mtp_config,
        _validated=False,
    )


def _validate_context(
    context: MTPAuthoritativeContext,
    cursor: speculative.GreedyTargetCursor,
    mtp_weights: mtp.MLXMTPWeights,
    mtp_config: mtp_reference.MTPConfig,
) -> None:
    require(
        isinstance(context, MTPAuthoritativeContext),
        "invalid authoritative MTP context",
    )
    require(
        context.hidden.shape == (mtp_config.hidden_size,)
        and context.hidden.dtype == mtp_weights.fc.dtype,
        "MTP context hidden mismatch",
    )
    require(
        attention.state_length(context.state, mtp_config.attention)
        == cursor.state.position,
        "target/MTP context positions disagree",
    )
    require(
        0 <= context.conditioned_token_id < cursor.logits.shape[0],
        "MTP conditioned token is out of range",
    )


def _validate_session(session: MTPGreedySession) -> None:
    require(
        isinstance(session, MTPGreedySession) and session._seal is _SESSION_SEAL,
        "invalid MTP greedy session",
    )
    require(
        session.verifier.config.hidden_size == session.mtp_config.hidden_size,
        "target/MTP hidden width mismatch",
    )
    require(
        not session.verifier._auxiliary_hidden_state_indices,
        "MTP verifier must use final target hidden states",
    )
    require(
        isinstance(session.draft_exact_rerank, bool),
        "invalid MTP draft rerank policy",
    )
    _validate_context(
        session.mtp_context,
        session.verifier.cursor,
        session.mtp_weights,
        session.mtp_config,
    )


def _validate_target_session(session: TargetGreedySession) -> None:
    require(
        isinstance(session, TargetGreedySession)
        and session._seal is _TARGET_SESSION_SEAL,
        "invalid detached target session",
    )


def _validate_adaptive_policy(policy: MTPAdaptivePolicy) -> None:
    require(isinstance(policy, MTPAdaptivePolicy), "invalid MTP adaptive policy")
    require(policy.minimum_mtp_blocks > 0, "adaptive MTP minimum block count must be positive")
    require(policy.window_blocks > 0, "adaptive MTP window must be positive")
    require(
        policy.minimum_mtp_blocks >= policy.window_blocks,
        "adaptive MTP minimum block count cannot be shorter than its window",
    )
    require(
        0.0 <= policy.minimum_future_acceptance <= 1.0,
        "adaptive MTP acceptance threshold is out of range",
    )


def _validate_adaptive_session(session: MTPAdaptiveSession) -> None:
    require(
        isinstance(session, MTPAdaptiveSession)
        and session._seal is _ADAPTIVE_SESSION_SEAL,
        "invalid adaptive MTP session",
    )
    _validate_adaptive_policy(session.policy)
    require(
        all(isinstance(value, int) and value >= 0 for value in session.future_accepted),
        "invalid adaptive MTP acceptance history",
    )
    if isinstance(session.active, MTPGreedySession):
        _validate_session(session.active)
        require(
            session.detached_after_mtp_blocks is None,
            "active MTP session has a target fallback marker",
        )
    else:
        _validate_target_session(session.active)
        require(
            session.detached_after_mtp_blocks == len(session.future_accepted),
            "detached target session block marker mismatch",
        )
    future_slots = session.active.verifier.block_tokens - 1
    require(
        all(value <= future_slots for value in session.future_accepted),
        "adaptive MTP acceptance count exceeds the proposal width",
    )


def _require_adaptive_session(session: MTPAdaptiveSession) -> None:
    require(
        isinstance(session, MTPAdaptiveSession)
        and session._seal is _ADAPTIVE_SESSION_SEAL,
        "invalid adaptive MTP session",
    )


def _require_pending_target(session: MTPGreedySession) -> int:
    pending = speculative.greedy_token(
        session.verifier.cursor.logits,
        session.verifier.cursor.hidden,
        session.verifier.weights.lm_head,
    )
    require(
        session.mtp_context.conditioned_token_id == pending,
        "MTP context is not conditioned on the target pending token",
    )
    return pending


def start_greedy_session(
    target_weights: model.TextModelWeights,
    target_cursor: speculative.GreedyTargetCursor,
    mtp_weights: mtp.MLXMTPWeights,
    mtp_context: MTPAuthoritativeContext,
    target_config: model.TextModelConfig,
    mtp_config: mtp_reference.MTPConfig,
    *,
    block_tokens: int = 3,
    compile_prefill_tails: bool = True,
    exact_block_lm_head: mx.array | None = None,
    target_linear_session: model.TextLinearDecodeSession | None = None,
    draft_exact_rerank: bool = True,
) -> MTPGreedySession:
    """Create an exact verifier around a target-aligned MTP context."""
    mtp.validate_weights(mtp_weights, mtp_config)
    require(
        model.matrix_dtype(target_weights.embedding) == mtp_weights.fc.dtype,
        "target/MTP model dtypes disagree",
    )
    verifier = speculative.start_greedy_verifier(
        target_weights,
        target_cursor,
        target_config,
        block_tokens=block_tokens,
        compile_prefill_tails=compile_prefill_tails,
        exact_block_lm_head=exact_block_lm_head,
        linear_session=target_linear_session,
    )
    session = MTPGreedySession(
        verifier=verifier,
        mtp_weights=mtp_weights,
        mtp_context=mtp_context,
        mtp_config=mtp_config,
        draft_exact_rerank=draft_exact_rerank,
        _seal=_SESSION_SEAL,
    )
    _validate_session(session)
    _require_pending_target(session)
    return session


def _draft_token(
    hidden: mx.array,
    target_weights: model.TextModelWeights,
    *,
    exact_rerank: bool,
) -> int:
    logits = model.project_lm_head(target_weights.lm_head, hidden)
    mx.eval(logits)
    if exact_rerank:
        return speculative.greedy_token(logits, hidden, target_weights.lm_head)
    return int(mx.argmax(logits).item())


def _propose_validated(session: MTPGreedySession, anchor: int) -> MTPProposal:
    state = session.mtp_context.state
    hidden = session.mtp_context.hidden
    future: list[int] = []
    for index in range(session.verifier.block_tokens - 1):
        token_id = _draft_token(
            hidden,
            session.verifier.weights,
            exact_rerank=session.draft_exact_rerank,
        )
        future.append(token_id)
        if index + 1 == session.verifier.block_tokens - 1:
            break
        token_embedding = model.embed_token(
            session.verifier.weights.embedding,
            token_id,
        )
        result = mtp.forward_step(
            token_embedding,
            hidden,
            state,
            session.mtp_weights,
            session.mtp_config,
            _validated=True,
        )
        _evaluate_result(result)
        state = result.state
        hidden = result.hidden
    return MTPProposal(
        anchor_token_id=anchor,
        future_token_ids=tuple(future),
    )


def propose(session: MTPGreedySession) -> MTPProposal:
    """Generate future MTP tokens after the target-owned anchor token."""
    _validate_session(session)
    return _propose_validated(session, _require_pending_target(session))


def _reconcile_authoritative_context(
    session: MTPGreedySession,
    verification: speculative.GreedyBlockVerification,
) -> MTPAuthoritativeContext:
    committed = len(verification.committed_tokens)
    require(committed > 0, "target rejected the MTP-owned anchor")
    hidden = verification.committed_hidden_states
    require(
        hidden is not None
        and hidden.shape == (committed, session.mtp_config.hidden_size),
        "verifier did not return committed target hidden states",
    )
    require(
        len(verification.emitted_tokens) == committed + 1,
        "MTP verifier emitted-token alignment mismatch",
    )
    following = (
        *verification.committed_tokens[1:],
        verification.emitted_tokens[-1],
    )
    require(len(following) == committed, "MTP shifted-token alignment mismatch")
    context = append_authoritative_hidden(
        session.mtp_context.state,
        hidden,
        following,
        session.verifier.weights.embedding,
        session.mtp_weights,
        session.mtp_config,
        _validated=True,
    )
    require(
        attention.state_length(context.state, session.mtp_config.attention)
        == verification.cursor.state.position,
        "target/MTP reconciled positions disagree",
    )
    return context


def step_greedy(
    session: MTPGreedySession,
    *,
    exact_long_attention: bool = True,
) -> tuple[MTPGreedyStep, MTPGreedySession]:
    """Propose, verify exactly, then rebuild only target-committed MTP KV."""
    _validate_session(session)
    # Session creation checks this anchor against the target, and every later
    # context is rebuilt from target-committed rows. The verifier repeats the
    # authoritative check before it can commit any proposal token.
    anchor = session.mtp_context.conditioned_token_id
    proposal = _propose_validated(session, anchor)
    verification, next_verifier = speculative.verify_greedy_block(
        proposal.target_token_ids,
        session.verifier,
        exact_long_attention=exact_long_attention,
    )
    next_context = _reconcile_authoritative_context(session, verification)
    next_session = MTPGreedySession(
        verifier=next_verifier,
        mtp_weights=session.mtp_weights,
        mtp_context=next_context,
        mtp_config=session.mtp_config,
        draft_exact_rerank=session.draft_exact_rerank,
        _seal=_SESSION_SEAL,
    )
    _validate_session(next_session)
    return MTPGreedyStep(proposal=proposal, verification=verification), next_session


def step_target_greedy(
    session: MTPGreedySession,
) -> tuple[MTPGreedyStep, MTPGreedySession]:
    """Advance one exact target token while preserving an aligned MTP context."""
    _validate_session(session)
    anchor = _require_pending_target(session)
    proposal = MTPProposal(anchor_token_id=anchor, future_token_ids=())
    verification, next_verifier = speculative.verify_greedy_block(
        proposal.target_token_ids,
        session.verifier,
    )
    require(
        verification.accepted_count == 1
        and verification.all_accepted
        and verification.committed_tokens == (anchor,),
        "single-token target fallback did not commit its anchor",
    )
    next_context = _reconcile_authoritative_context(session, verification)
    next_session = MTPGreedySession(
        verifier=next_verifier,
        mtp_weights=session.mtp_weights,
        mtp_context=next_context,
        mtp_config=session.mtp_config,
        draft_exact_rerank=session.draft_exact_rerank,
        _seal=_SESSION_SEAL,
    )
    _validate_session(next_session)
    return MTPGreedyStep(proposal=proposal, verification=verification), next_session


def detach_target_session(session: MTPGreedySession) -> TargetGreedySession:
    """Permanently retain only the exact target continuation state."""
    _validate_session(session)
    detached = TargetGreedySession(
        verifier=session.verifier,
        _seal=_TARGET_SESSION_SEAL,
    )
    _validate_target_session(detached)
    return detached


def step_detached_target_greedy(
    session: TargetGreedySession,
) -> tuple[MTPGreedyStep, TargetGreedySession]:
    """Advance one exact target token without evaluating or updating MTP."""
    _validate_target_session(session)
    verification, next_verifier = speculative.advance_greedy_target(
        session.verifier,
        _validated=True,
    )
    require(
        len(verification.proposal_ids) == 1
        and verification.accepted_count == 1
        and verification.all_accepted
        and verification.committed_tokens == verification.proposal_ids,
        "detached target step did not commit its greedy token",
    )
    proposal = MTPProposal(
        anchor_token_id=verification.proposal_ids[0],
        future_token_ids=(),
    )
    next_session = TargetGreedySession(
        verifier=next_verifier,
        _seal=_TARGET_SESSION_SEAL,
    )
    _validate_target_session(next_session)
    return MTPGreedyStep(proposal=proposal, verification=verification), next_session


def start_adaptive_session(
    session: MTPGreedySession,
    policy: MTPAdaptivePolicy = MTPAdaptivePolicy(),
) -> MTPAdaptiveSession:
    """Wrap an exact MTP session in a measured one-way fallback policy."""
    _validate_session(session)
    _validate_adaptive_policy(policy)
    adaptive = MTPAdaptiveSession(
        active=session,
        policy=policy,
        future_accepted=(),
        detached_after_mtp_blocks=None,
        _seal=_ADAPTIVE_SESSION_SEAL,
    )
    _validate_adaptive_session(adaptive)
    return adaptive


def adaptive_recent_future_acceptance(session: MTPAdaptiveSession) -> float | None:
    """Return the selected policy window's accepted-future ratio."""
    _validate_adaptive_session(session)
    if not session.future_accepted:
        return None
    recent = session.future_accepted[-session.policy.window_blocks :]
    future_slots = session.active.verifier.block_tokens - 1
    return sum(recent) / (len(recent) * future_slots)


def step_adaptive_greedy(
    session: MTPAdaptiveSession,
    *,
    exact_long_attention: bool = True,
) -> tuple[MTPGreedyStep, MTPAdaptiveSession]:
    """Advance exactly, detaching MTP after a persistently weak yield window."""
    _require_adaptive_session(session)
    if isinstance(session.active, TargetGreedySession):
        step, active = step_detached_target_greedy(session.active)
        next_session = MTPAdaptiveSession(
            active=active,
            policy=session.policy,
            future_accepted=session.future_accepted,
            detached_after_mtp_blocks=session.detached_after_mtp_blocks,
            _seal=_ADAPTIVE_SESSION_SEAL,
        )
        return step, next_session

    step, active_mtp = step_greedy(
        session.active,
        exact_long_attention=exact_long_attention,
    )
    accepted = max(0, step.verification.accepted_count - 1)
    history = (*session.future_accepted, accepted)
    active: MTPGreedySession | TargetGreedySession = active_mtp
    detached_after = None
    if len(history) >= session.policy.minimum_mtp_blocks:
        recent = history[-session.policy.window_blocks :]
        future_slots = active_mtp.verifier.block_tokens - 1
        acceptance = sum(recent) / (len(recent) * future_slots)
        if acceptance < session.policy.minimum_future_acceptance:
            active = detach_target_session(active_mtp)
            detached_after = len(history)
    next_session = MTPAdaptiveSession(
        active=active,
        policy=session.policy,
        future_accepted=history,
        detached_after_mtp_blocks=detached_after,
        _seal=_ADAPTIVE_SESSION_SEAL,
    )
    return step, next_session

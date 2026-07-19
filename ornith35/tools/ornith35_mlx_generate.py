#!/usr/bin/env python3
"""Bounded text generation from the verified Ornith-35 target."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Callable

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_cache as persistent_cache
import ornith35_mlx_model as model
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_runtime as mtp_runtime
import ornith35_mlx_sampling as sampling
import ornith35_mlx_speculative as speculative
import ornith35_mlx_vocab as vocab
import ornith35_mtp_reference as mtp_reference
from ornith35_moe_reference import MoEError, require as require_model
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


DEFAULT_MTP_ADAPTATION = Path(
    "experiments/mtp-distill-coding-v1/adapter-r32-e8-s29-v2"
)
TURBOQUANT_MEASURED_CROSSOVER_TOKENS = 20_000


def sample_candidates(
    token_ids: list[int],
    logits: list[float],
    *,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> int:
    """Sample a preselected top-k set using a deterministic CPU RNG."""
    return sampling.candidate_distribution(
        token_ids,
        logits,
        temperature=temperature,
        top_p=top_p,
    ).sample(rng)


def choose_next_token(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: random.Random,
    hidden: mx.array | None = None,
    lm_head: mx.array | vocab.MLXAffineQuantizedMatrix | None = None,
) -> int:
    return sampling.target_distribution(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        hidden=hidden,
        lm_head=lm_head,
    ).sample(rng)


def split_reasoning_response(response: str) -> tuple[str | None, str]:
    marker = "</think>"
    if marker not in response:
        return None, response
    reasoning, final = response.split(marker, 1)
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :]
    return reasoning.strip(), final.lstrip()


def prefill_schedule(token_count: int, max_chunk: int) -> tuple[int, ...]:
    """Use a bounded set of compiled chunk sizes and a serial tail."""
    require_model(token_count > 0, "prefill token count must be positive")
    require_model(
        max_chunk == 1
        or (8 <= max_chunk <= 128 and max_chunk & (max_chunk - 1) == 0),
        "prefill chunk must be 1 or a power of two from 8 through 128",
    )
    if max_chunk == 1:
        return (1,) * token_count
    remaining = token_count
    schedule = []
    while remaining >= 8:
        chunk = min(max_chunk, 1 << (remaining.bit_length() - 1))
        schedule.append(chunk)
        remaining -= chunk
    schedule.extend((1,) * remaining)
    return tuple(schedule)


def format_prefill_schedule(schedule: tuple[int, ...]) -> str:
    """Compress consecutive equal chunk sizes for bounded human-readable logs."""
    require_model(bool(schedule), "prefill schedule must not be empty")
    groups = []
    current = schedule[0]
    count = 1
    for size in schedule[1:]:
        if size == current:
            count += 1
            continue
        groups.append(f"{current}x{count}" if count > 1 else str(current))
        current = size
        count = 1
    groups.append(f"{current}x{count}" if count > 1 else str(current))
    return ",".join(groups)


def mtp_enabled_for_prompt(
    requested: bool,
    prompt_tokens: int,
    max_prompt_tokens: int,
) -> bool:
    """Select the measured short-prefix MTP regime; zero disables the ceiling."""
    require_model(prompt_tokens > 0, "MTP prompt token count must be positive")
    require_model(max_prompt_tokens >= 0, "MTP prompt ceiling must be nonnegative")
    return requested and (max_prompt_tokens == 0 or prompt_tokens <= max_prompt_tokens)


def mtp_enabled_for_generation(
    requested: bool,
    prompt_tokens: int,
    max_prompt_tokens: int,
    *,
    temperature: float,
    enable_thinking: bool,
) -> bool:
    """Select only the prompt-disjoint measured MTP generation regime."""
    length_enabled = mtp_enabled_for_prompt(
        requested,
        prompt_tokens,
        max_prompt_tokens,
    )
    require_model(temperature >= 0.0, "MTP temperature must be nonnegative")
    require_model(isinstance(enable_thinking, bool), "MTP thinking policy must be boolean")
    if max_prompt_tokens == 0:
        return length_enabled
    return length_enabled and temperature == 0.0 and enable_thinking


def prefill_prompt(
    prompt_ids: list[int],
    state: model.TextModelState,
    weights: model.TextModelWeights,
    *,
    max_chunk: int,
    linear_session: model.TextLinearDecodeSession | None = None,
    exact_long_attention: bool = True,
) -> tuple[model.TextModelResult | model.TextModelChunkResult, tuple[int, ...]]:
    """Materialize exact prompt chunks and project logits only at the end."""
    schedule = prefill_schedule(len(prompt_ids), max_chunk)
    offset = 0
    result = None
    for size in schedule:
        final = offset + size == len(prompt_ids)
        token_slice = prompt_ids[offset : offset + size]
        if not final and size > 1:
            if linear_session is not None:
                state = model.prefill_linear_session_state_chunk(
                    token_slice,
                    linear_session,
                    use_steel=False,
                    exact_long_attention=exact_long_attention,
                )
            else:
                state = model.prefill_state_chunk(
                    token_slice,
                    state,
                    weights,
                    use_steel=False,
                    exact_long_attention=exact_long_attention,
                )
                model.evaluate_state(state)
            offset += size
            continue
        if linear_session is not None and size == 1:
            if final:
                result = model.forward_linear_session_token(
                    token_slice[0],
                    linear_session,
                )
            else:
                result = model.forward_linear_session_hidden_token(
                    token_slice[0],
                    linear_session,
                )
        elif linear_session is not None:
            result = model.prefill_linear_session_final_chunk(
                token_slice,
                linear_session,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
        elif size == 1:
            if final:
                result = model.forward_token(token_slice[0], state, weights)
                model.evaluate_result(result)
            else:
                transition = model.forward_hidden_token(token_slice[0], state, weights)
                model.evaluate_transition(transition)
                result = transition
        elif final:
            result = model.prefill_final_chunk(
                token_slice,
                state,
                weights,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
            model.evaluate_result(result)
        else:
            transition = model.prefill_hidden_chunk(
                token_slice,
                state,
                weights,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
            model.evaluate_chunk_transition(transition)
            result = transition
        state = result.state
        offset += size
    require_model(
        isinstance(result, (model.TextModelResult, model.TextModelChunkResult)),
        "prompt prefill produced no logits",
    )
    return result, schedule


def prefill_turboquant_prompt(
    prompt_ids: list[int],
    session: model.TextTurboQuantDecodeSession,
) -> tuple[model.TextModelResult, tuple[int, ...]]:
    """Resume a persisted packed prefix without reconstructing historical BF16 K/V."""
    require_model(prompt_ids, "TurboQuant prompt suffix is empty")
    result = None
    for offset, token_id in enumerate(prompt_ids):
        if offset + 1 == len(prompt_ids):
            result = model.forward_turboquant_session_token(token_id, session)
        else:
            model.forward_turboquant_session_hidden_token(token_id, session)
    require_model(result is not None, "TurboQuant prompt prefill produced no logits")
    return result, (1,) * len(prompt_ids)


def prefill_state_prompt(
    prompt_ids: list[int] | tuple[int, ...],
    state: model.TextModelState,
    weights: model.TextModelWeights,
    *,
    max_chunk: int,
    linear_session: model.TextLinearDecodeSession | None = None,
    exact_long_attention: bool = True,
) -> tuple[model.TextModelState, tuple[int, ...]]:
    """Advance an unobservable stable prefix without projecting final outputs."""
    schedule = prefill_schedule(len(prompt_ids), max_chunk)
    offset = 0
    for size in schedule:
        token_slice = prompt_ids[offset : offset + size]
        if linear_session is not None and size == 1:
            state = model.forward_linear_session_hidden_token(
                token_slice[0],
                linear_session,
            ).state
        elif linear_session is not None:
            state = model.prefill_linear_session_state_chunk(
                token_slice,
                linear_session,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
        elif size == 1:
            transition = model.forward_hidden_token(token_slice[0], state, weights)
            model.evaluate_transition(transition)
            state = transition.state
        else:
            state = model.prefill_state_chunk(
                token_slice,
                state,
                weights,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
            model.evaluate_state(state)
        offset += size
    return state, schedule


def prefill_state_prompt_with_mtp(
    prompt_ids: list[int] | tuple[int, ...],
    state: model.TextModelState,
    weights: model.TextModelWeights,
    mtp_weights: mtp.MLXMTPWeights,
    *,
    max_chunk: int,
    mtp_capacity: int,
    mtp_prefix: mtp_runtime.MTPPrefixState | None = None,
    linear_session: model.TextLinearDecodeSession | None = None,
    exact_long_attention: bool = True,
    target_config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    mtp_config: mtp_reference.MTPConfig = mtp.PRODUCTION_CONFIG,
) -> tuple[
    model.TextModelState,
    tuple[int, ...],
    mtp_runtime.MTPPrefixState,
]:
    """Advance a stable target prefix and retain suffix-independent MTP state."""
    require_model(prompt_ids, "MTP state prefill is empty")
    require_model(
        mtp_capacity >= state.position + len(prompt_ids),
        "MTP context capacity is shorter than the stable prefix",
    )
    schedule = prefill_schedule(len(prompt_ids), max_chunk)
    if mtp_prefix is None:
        require_model(state.position == 0, "nonempty target state requires an MTP prefix")
        mtp_state = mtp_runtime.initial_context_state(
            mtp_weights,
            mtp_config,
            capacity=mtp_capacity,
        )
    else:
        resumed = mtp_runtime.resume_prefix_state(
            mtp_prefix,
            state.position,
            prompt_ids[0],
            mtp_capacity,
            weights.embedding,
            mtp_weights,
            mtp_config,
        )
        mtp_state = resumed.state
    offset = 0
    final_hidden = None
    for size in schedule:
        token_slice = prompt_ids[offset : offset + size]
        if linear_session is not None and size == 1:
            outcome = model.forward_linear_session_hidden_token(
                token_slice[0],
                linear_session,
            )
        elif linear_session is not None:
            outcome = model.prefill_linear_session_chunk(
                token_slice,
                linear_session,
                project_logits=False,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
        elif size == 1:
            outcome = model.forward_hidden_token(
                token_slice[0],
                state,
                weights,
                target_config,
            )
            model.evaluate_transition(outcome)
        else:
            outcome = model.prefill_hidden_chunk(
                token_slice,
                state,
                weights,
                target_config,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
            model.evaluate_chunk_transition(outcome)
        state = outcome.state
        hidden_rows = outcome.hidden.reshape(1, -1) if size == 1 else outcome.hidden
        known_rows = min(size, len(prompt_ids) - 1 - offset)
        if known_rows > 0:
            following = prompt_ids[offset + 1 : offset + 1 + known_rows]
            context = mtp_runtime.append_authoritative_hidden(
                mtp_state,
                hidden_rows[:known_rows],
                following,
                weights.embedding,
                mtp_weights,
                mtp_config,
                _validated=True,
            )
            mtp_state = context.state
        if offset + size == len(prompt_ids):
            final_hidden = hidden_rows[-1]
        offset += size
    require_model(final_hidden is not None, "MTP state prefill produced no boundary hidden")
    prefix = mtp_runtime.MTPPrefixState(
        state=mtp_state,
        boundary_hidden=final_hidden,
    )
    mtp_runtime.validate_prefix_state(
        prefix,
        state.position,
        mtp_config,
        dtype=mtp_weights.fc.dtype,
    )
    return state, schedule, prefix


def prefill_prompt_with_mtp(
    prompt_ids: list[int],
    state: model.TextModelState,
    weights: model.TextModelWeights,
    mtp_weights: mtp.MLXMTPWeights,
    *,
    max_chunk: int,
    mtp_capacity: int,
    select_pending: Callable[[mx.array, mx.array], int],
    mtp_prefix: mtp_runtime.MTPPrefixState | None = None,
    linear_session: model.TextLinearDecodeSession | None = None,
    exact_long_attention: bool = True,
    target_config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    mtp_config: mtp_reference.MTPConfig = mtp.PRODUCTION_CONFIG,
) -> tuple[
    model.TextModelResult | model.TextModelChunkResult,
    tuple[int, ...],
    mtp_runtime.MTPAuthoritativeContext,
]:
    """Stream exact target hidden rows into fixed-capacity MTP state."""
    require_model(prompt_ids, "MTP prompt prefill is empty")
    require_model(
        mtp_capacity >= state.position + len(prompt_ids),
        "MTP context capacity is shorter than the completed prompt",
    )
    schedule = prefill_schedule(len(prompt_ids), max_chunk)
    if mtp_prefix is None:
        require_model(state.position == 0, "nonempty target state requires an MTP prefix")
        mtp_state = mtp_runtime.initial_context_state(
            mtp_weights,
            mtp_config,
            capacity=mtp_capacity,
        )
    else:
        resumed = mtp_runtime.resume_prefix_state(
            mtp_prefix,
            state.position,
            prompt_ids[0],
            mtp_capacity,
            weights.embedding,
            mtp_weights,
            mtp_config,
        )
        mtp_state = resumed.state
    offset = 0
    final_result: model.TextModelResult | model.TextModelChunkResult | None = None
    for size in schedule:
        final = offset + size == len(prompt_ids)
        token_slice = prompt_ids[offset : offset + size]
        if linear_session is not None and size == 1:
            if final:
                outcome = model.forward_linear_session_token(
                    token_slice[0],
                    linear_session,
                )
            else:
                outcome = model.forward_linear_session_hidden_token(
                    token_slice[0],
                    linear_session,
                )
        elif linear_session is not None:
            outcome = model.prefill_linear_session_chunk(
                token_slice,
                linear_session,
                project_logits=final,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
        elif size == 1:
            if final:
                outcome = model.forward_token(
                    token_slice[0], state, weights, target_config
                )
                model.evaluate_result(outcome)
            else:
                outcome = model.forward_hidden_token(
                    token_slice[0], state, weights, target_config
                )
                model.evaluate_transition(outcome)
        elif final:
            outcome = model.prefill_chunk(
                token_slice,
                state,
                weights,
                target_config,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
            model.evaluate_chunk_result(outcome)
        else:
            outcome = model.prefill_hidden_chunk(
                token_slice,
                state,
                weights,
                target_config,
                use_steel=False,
                exact_long_attention=exact_long_attention,
            )
            model.evaluate_chunk_transition(outcome)
        state = outcome.state
        hidden_rows = outcome.hidden.reshape(1, -1) if size == 1 else outcome.hidden
        known_rows = min(size, len(prompt_ids) - 1 - offset)
        if known_rows > 0:
            following = prompt_ids[offset + 1 : offset + 1 + known_rows]
            context = mtp_runtime.append_authoritative_hidden(
                mtp_state,
                hidden_rows[:known_rows],
                following,
                weights.embedding,
                mtp_weights,
                mtp_config,
                _validated=True,
            )
            mtp_state = context.state
        if final:
            require_model(
                isinstance(outcome, (model.TextModelResult, model.TextModelChunkResult)),
                "final MTP prompt chunk produced no logits",
            )
            final_result = outcome
        offset += size

    require_model(final_result is not None, "MTP prompt prefill produced no target result")
    final_hidden = (
        final_result.hidden[-1]
        if final_result.hidden.ndim == 2
        else final_result.hidden
    )
    pending = select_pending(final_result.logits, final_hidden)
    context = mtp_runtime.append_authoritative_hidden(
        mtp_state,
        final_hidden.reshape(1, -1),
        (pending,),
        weights.embedding,
        mtp_weights,
        mtp_config,
        _validated=True,
    )
    require_model(
        attention.state_length(context.state, mtp_config.attention)
        == final_result.state.position,
        "target and streamed MTP prompt positions disagree",
    )
    return final_result, schedule, context


def generate(
    root: Path,
    prompt: str,
    *,
    system: str | None,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    prefill_chunk: int,
    linear_kv_cache: bool,
    turboquant_kv: bool,
    compiled_gdn_layers: bool,
    compiled_attention_tails: bool,
    mapped_embedding: bool,
    quantized_lm_head: bool,
    exact_long_attention: bool,
    context_profile: str,
    load_cache: Path | None,
    save_cache: bool,
    cache_root: Path | None,
    cache_system_prefix: bool,
    cache_longest_prefix: bool,
    cache_max_gib: float,
    use_mtp: bool,
    mtp_adaptation_dir: Path | None,
    mtp_block_tokens: int,
    mtp_max_prompt_tokens: int,
    mtp_adaptive_fallback: bool,
    mtp_adaptive_minimum_blocks: int,
    mtp_adaptive_window_blocks: int,
    mtp_adaptive_minimum_acceptance: float,
) -> str:
    selected_context = context.resolve_profile(context_profile)
    require_model(0 < max_tokens <= 4096, "max tokens must be between 1 and 4096")
    require_model(temperature >= 0.0, "temperature must be nonnegative")
    require_model(0 < top_k <= model.PRODUCTION_CONFIG.vocab_size, "invalid top-k")
    require_model(0.0 < top_p <= 1.0, "invalid top-p")
    require_model(
        not quantized_lm_head or temperature == 0.0 or top_k <= 256,
        "hybrid LM head sampling supports top-k at most 256",
    )
    require_model(cache_max_gib > 0.0, "cache size budget must be positive")
    require_model(
        not turboquant_kv or selected_context.profile_id == context.NATIVE_PROFILE_ID,
        "TurboQuant K/V is quality-gated only for the native context profile",
    )
    require_model(
        not turboquant_kv or not use_mtp,
        "TurboQuant K/V cannot be combined with MTP",
    )
    require_model(
        not turboquant_kv or not cache_system_prefix,
        "TurboQuant K/V does not support system-prefix warming",
    )
    require_model(
        not use_mtp or selected_context.profile_id == context.NATIVE_PROFILE_ID,
        "MTP is validated only for the native context profile",
    )
    if use_mtp:
        require_model(
            2 <= mtp_block_tokens <= speculative.MAX_PROPOSAL_TOKENS,
            "MTP block token count must be between 2 and 8",
        )
        require_model(
            mtp_max_prompt_tokens >= 0,
            "MTP maximum prompt token count must be nonnegative",
        )
        if mtp_adaptive_fallback:
            require_model(
                mtp_adaptive_minimum_blocks > 0,
                "adaptive MTP minimum block count must be positive",
            )
            require_model(
                0 < mtp_adaptive_window_blocks <= mtp_adaptive_minimum_blocks,
                "adaptive MTP window must be positive and no longer than its minimum",
            )
            require_model(
                0.0 <= mtp_adaptive_minimum_acceptance <= 1.0,
                "adaptive MTP acceptance threshold must be in [0, 1]",
            )
    prefill_schedule(1, prefill_chunk)
    tokenizer = load_text_tokenizer(root)
    rendered = render_text_prompt(
        prompt,
        system=system,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(rendered)
    require_model(prompt_ids, "rendered prompt produced no tokens")
    if turboquant_kv and len(prompt_ids) < TURBOQUANT_MEASURED_CROSSOVER_TOKENS:
        print(
            "generate-turboquant-short-prefix "
            f"tokens={len(prompt_ids)} measured_crossover_tokens="
            f"{TURBOQUANT_MEASURED_CROSSOVER_TOKENS}",
            flush=True,
        )
    mtp_requested = use_mtp
    use_mtp = mtp_enabled_for_generation(
        use_mtp,
        len(prompt_ids),
        mtp_max_prompt_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    if mtp_requested and not use_mtp:
        reasons = []
        if len(prompt_ids) > mtp_max_prompt_tokens:
            reasons.append("prompt_limit")
        if temperature > 0.0:
            reasons.append("sampled_decode")
        if not enable_thinking:
            reasons.append("thinking_disabled")
        print(
            "generate-mtp-skipped "
            f"reason={'+'.join(reasons)} prompt_tokens={len(prompt_ids)} "
            f"limit={mtp_max_prompt_tokens} temperature={temperature:.6f} "
            f"thinking={str(enable_thinking).lower()}",
            flush=True,
        )
    speculative_capacity = mtp_block_tokens if use_mtp else 0
    decode_capacity = len(prompt_ids) + max_tokens + speculative_capacity
    require_model(
        decode_capacity <= selected_context.max_position_embeddings,
        f"prompt, generation, and speculative reserve exceed {selected_context.profile_id}",
    )
    selected_mtp_adaptation = (
        mtp_adaptation_dir
        if mtp_adaptation_dir is not None
        else root / DEFAULT_MTP_ADAPTATION
    )
    cache_enabled = (
        load_cache is not None
        or save_cache
        or cache_system_prefix
        or cache_longest_prefix
    )
    cache_identity = (
        persistent_cache.production_identity(
            root,
            Path(__file__).resolve().parents[2],
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            chat_template_sha256=tokenizer.template_sha256,
            mapped_embedding=mapped_embedding,
            quantized_lm_head=quantized_lm_head,
            mtp_adaptation_dir=selected_mtp_adaptation if use_mtp else None,
            rope_profile=selected_context.profile_id,
            turboquant_kv=turboquant_kv,
        )
        if cache_enabled
        else None
    )
    cache_destination = cache_root if cache_root is not None else root / "cache"
    system_prefix_ids: tuple[int, ...] = ()
    if cache_system_prefix:
        require_model(cache_identity is not None, "cache identity is missing")
        require_model(system is not None and system.strip(), "system-prefix caching requires --system")
        system_rendered = f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
        system_prefix_ids = tokenizer.encode(system_rendered)
        require_model(
            tuple(prompt_ids[: len(system_prefix_ids)]) == system_prefix_ids,
            "rendered system tokens are not an exact prompt prefix",
        )
    if load_cache is None and cache_longest_prefix:
        require_model(cache_identity is not None, "cache identity is missing")
        lookup = persistent_cache.find_longest_prefix(
            cache_destination,
            prompt_ids,
            cache_identity,
            model.PRODUCTION_CONFIG,
        )
        load_cache = lookup.path
        print(
            "generate-cache-discovery "
            f"selected_tokens={lookup.token_count} elapsed_s={lookup.elapsed_s:.3f} "
            f"scanned_entries={lookup.scanned_entries} "
            f"compatible_entries={lookup.compatible_entries} "
            f"matching_entries={lookup.matching_entries} "
            f"path={lookup.path if lookup.path is not None else 'none'}",
            flush=True,
        )
    if cache_system_prefix and load_cache is None:
        require_model(cache_identity is not None, "cache identity is missing")
        if system_prefix_ids:
            candidate = cache_destination / persistent_cache.cache_key(
                system_prefix_ids,
                cache_identity,
                model.PRODUCTION_CONFIG,
            )
            if candidate.is_dir():
                load_cache = candidate
    restored = None
    if load_cache is not None:
        require_model(cache_identity is not None, "cache identity is missing")
        cache_started = time.perf_counter()
        restored = persistent_cache.load_cache(
            load_cache,
            cache_identity,
            model.PRODUCTION_CONFIG,
        )
        require_model(
            len(restored.token_ids) < len(prompt_ids),
            "loaded cache must leave at least one prompt token for final logits",
        )
        require_model(
            tuple(prompt_ids[: len(restored.token_ids)]) == restored.token_ids,
            "loaded cache is not an exact prompt prefix",
        )
        print(
            "generate-cache-restored "
            f"tokens={len(restored.token_ids)} elapsed_s={time.perf_counter() - cache_started:.3f} "
            f"manifest_s={restored.load_timing.manifest_s:.3f} "
            f"tokens_s={restored.load_timing.tokens_s:.3f} "
            f"verify_s={restored.load_timing.payload_verify_s:.3f} "
            f"materialize_s={restored.load_timing.payload_materialize_s:.3f} "
            f"payload_gib={restored.load_timing.payload_bytes / 2**30:.3f} "
            f"mtp_prefix={str(restored.mtp_prefix is not None).lower()} "
            f"path={restored.path}",
            flush=True,
        )
    cache_restored = restored is not None
    restored_mtp_prefix = restored.mtp_prefix if restored is not None else None
    protected_cache_keys = {restored.key} if restored is not None else set()
    print(
        "generate-start "
        f"prompt_tokens={len(prompt_ids)} max_tokens={max_tokens} "
        f"thinking={str(enable_thinking).lower()} temperature={temperature:.6g} "
        f"top_k={top_k} top_p={top_p:.6g} seed={seed} "
        f"prefill_chunk={prefill_chunk} "
        f"linear_kv_cache={str(linear_kv_cache).lower()} "
        f"turboquant_kv={str(turboquant_kv).lower()} "
        f"compiled_gdn_layers={str(compiled_gdn_layers).lower()} "
        f"compiled_attention_tails={str(compiled_attention_tails).lower()} "
        f"mapped_embedding={str(mapped_embedding).lower()} "
        f"quantized_lm_head={str(quantized_lm_head).lower()} "
        f"exact_long_attention={str(exact_long_attention).lower()} "
        f"context_profile={selected_context.profile_id} "
        f"cache_system_prefix={str(cache_system_prefix).lower()} "
        f"cache_longest_prefix={str(cache_longest_prefix).lower()} "
        f"mtp_requested={str(mtp_requested).lower()} "
        f"mtp_effective={str(use_mtp).lower()} "
        f"mtp_block_tokens={mtp_block_tokens} "
        f"mtp_max_prompt_tokens={mtp_max_prompt_tokens} "
        f"mtp_adaptive_fallback={str(mtp_adaptive_fallback).lower()}",
        flush=True,
    )

    load_started = time.perf_counter()
    weights = model.load_text_model(
        root,
        map_embedding=mapped_embedding,
        quantize_lm_head=quantized_lm_head,
    )
    load_elapsed = time.perf_counter() - load_started
    print(
        f"generate-model-ready load_s={load_elapsed:.3f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f}",
        flush=True,
    )
    loaded_mtp_weights = None
    if use_mtp:
        mtp_load_started = time.perf_counter()
        loaded_mtp_weights = mtp.load_weights(
            root,
            verify_hash=True,
            adaptation_dir=selected_mtp_adaptation,
        )
        print(
            "generate-mtp-ready "
            f"load_s={time.perf_counter() - mtp_load_started:.3f} "
            f"adaptation={selected_mtp_adaptation.resolve()} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )

    state = (
        restored.state
        if restored is not None
        else model.initial_state(
            weights,
            model.PRODUCTION_CONFIG,
            selected_context.profile_id,
        )
    )
    restored_turboquant = turboquant_kv and restored is not None
    linear_session = (
        model.start_linear_decode_session(
            weights,
            state,
            decode_capacity,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=compiled_gdn_layers,
            compile_attention_tails=compiled_attention_tails,
        )
        if linear_kv_cache and not restored_turboquant
        else None
    )
    turboquant_session = (
        model.start_turboquant_decode_session(
            weights,
            state,
            decode_capacity,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=compiled_gdn_layers,
            compile_attention_tails=compiled_attention_tails,
        )
        if restored_turboquant
        else None
    )
    if linear_session is not None:
        state = linear_session.state
        restored = None
    elif turboquant_session is not None:
        state = turboquant_session.state
        restored = None
    active_mtp_prefix = restored_mtp_prefix
    if not cache_restored and system_prefix_ids:
        warm_started = time.perf_counter()
        if use_mtp:
            require_model(loaded_mtp_weights is not None, "MTP weights are missing")
            state, warm_schedule, active_mtp_prefix = prefill_state_prompt_with_mtp(
                system_prefix_ids,
                state,
                weights,
                loaded_mtp_weights,
                max_chunk=prefill_chunk,
                mtp_capacity=decode_capacity,
                linear_session=linear_session,
                exact_long_attention=exact_long_attention,
            )
        else:
            state, warm_schedule = prefill_state_prompt(
                system_prefix_ids,
                state,
                weights,
                max_chunk=prefill_chunk,
                linear_session=linear_session,
                exact_long_attention=exact_long_attention,
            )
        warm_elapsed = time.perf_counter() - warm_started
        cache_started = time.perf_counter()
        require_model(cache_identity is not None, "cache identity is missing")
        saved_system = persistent_cache.save_cache(
            cache_destination,
            system_prefix_ids,
            state,
            cache_identity,
            model.PRODUCTION_CONFIG,
            mtp_prefix=active_mtp_prefix,
        )
        protected_cache_keys.add(saved_system.name)
        print(
            "generate-cache-warmed "
            f"tokens={len(system_prefix_ids)} prefill_s={warm_elapsed:.3f} "
            f"save_s={time.perf_counter() - cache_started:.3f} "
            f"mtp_prefix={str(active_mtp_prefix is not None).lower()} "
            f"chunks={format_prefill_schedule(warm_schedule)} path={saved_system}",
            flush=True,
        )
    cached_tokens = state.position
    suffix_ids = list(prompt_ids[cached_tokens:])
    require_model(suffix_ids, "prompt cache left no suffix to evaluate")
    rng = random.Random(seed)
    prefill_started = time.perf_counter()
    mtp_context = None
    if turboquant_session is not None:
        result, schedule = prefill_turboquant_prompt(
            suffix_ids,
            turboquant_session,
        )
    elif use_mtp:
        require_model(loaded_mtp_weights is not None, "MTP weights are missing")
        result, schedule, mtp_context = prefill_prompt_with_mtp(
            suffix_ids,
            state,
            weights,
            loaded_mtp_weights,
            max_chunk=prefill_chunk,
            mtp_capacity=decode_capacity,
            select_pending=lambda logits, hidden: choose_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
                hidden=hidden,
                lm_head=weights.lm_head,
            ),
            mtp_prefix=active_mtp_prefix,
            linear_session=linear_session,
            exact_long_attention=exact_long_attention,
        )
    else:
        result, schedule = prefill_prompt(
            suffix_ids,
            state,
            weights,
            max_chunk=prefill_chunk,
            linear_session=linear_session,
            exact_long_attention=exact_long_attention,
        )
    state = result.state
    cache_mtp_prefix = None
    if use_mtp:
        require_model(mtp_context is not None, "MTP prompt context is missing")
        final_hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
        cache_mtp_prefix = mtp_runtime.prefix_from_context(
            mtp_context,
            final_hidden,
            state.position,
            mtp.PRODUCTION_CONFIG,
            dtype=loaded_mtp_weights.fc.dtype,
        )
    prefill_elapsed = time.perf_counter() - prefill_started
    print(
        "generate-prefill-done "
        f"tokens={len(suffix_ids)} cached_tokens={cached_tokens} "
        f"total_tokens={len(prompt_ids)} elapsed_s={prefill_elapsed:.3f} "
        f"tokens_s={len(suffix_ids) / prefill_elapsed:.3f} "
        f"chunks={format_prefill_schedule(schedule)} "
        f"mtp_context={str(use_mtp).lower()} "
        "steel=false logit_projections=1",
        flush=True,
    )
    if turboquant_kv and turboquant_session is None:
        conversion_started = time.perf_counter()
        turboquant_session = model.start_turboquant_decode_session(
            weights,
            state,
            decode_capacity,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=compiled_gdn_layers,
            compile_attention_tails=compiled_attention_tails,
        )
        state = turboquant_session.state
        result = replace(result, state=state)
        linear_session = None
        print(
            "generate-turboquant-ready "
            f"tokens={state.position} elapsed_s={time.perf_counter() - conversion_started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
    if save_cache:
        cache_started = time.perf_counter()
        require_model(cache_identity is not None, "cache identity is missing")
        saved_path = persistent_cache.save_cache(
            cache_destination,
            prompt_ids,
            state,
            cache_identity,
            model.PRODUCTION_CONFIG,
            mtp_prefix=cache_mtp_prefix,
        )
        protected_cache_keys.add(saved_path.name)
        print(
            "generate-cache-saved "
            f"tokens={len(prompt_ids)} elapsed_s={time.perf_counter() - cache_started:.3f} "
            f"mtp_prefix={str(cache_mtp_prefix is not None).lower()} "
            f"path={saved_path}",
            flush=True,
        )
    if save_cache or cache_system_prefix:
        pruned = persistent_cache.prune_cache(
            cache_destination,
            max_bytes=int(cache_max_gib * 2**30),
            max_entries=persistent_cache.DEFAULT_MAX_ENTRIES,
            protect=tuple(protected_cache_keys),
        )
        print(
            "generate-cache-retention "
            f"removed_entries={pruned.removed_entries} "
            f"removed_gib={pruned.removed_bytes / 2**30:.3f} "
            f"retained_entries={pruned.retained_entries} "
            f"retained_gib={pruned.retained_bytes / 2**30:.3f} "
            f"over_budget={str(pruned.over_budget).lower()}",
            flush=True,
        )
    decode_session = None
    mtp_session = None
    mtp_adaptive_session = None
    if use_mtp:
        require_model(mtp_context is not None, "MTP prompt context is missing")
        require_model(loaded_mtp_weights is not None, "MTP weights are missing")
        cursor = speculative.cursor_from_result(result)
        if temperature == 0.0:
            mtp_session = mtp_runtime.start_greedy_session(
                weights,
                cursor,
                loaded_mtp_weights,
                mtp_context,
                model.PRODUCTION_CONFIG,
                mtp.PRODUCTION_CONFIG,
                block_tokens=mtp_block_tokens,
                target_linear_session=linear_session,
            )
        else:
            mtp_session = mtp_runtime.start_sampled_session(
                weights,
                cursor,
                loaded_mtp_weights,
                mtp_context,
                model.PRODUCTION_CONFIG,
                mtp.PRODUCTION_CONFIG,
                block_tokens=mtp_block_tokens,
                target_linear_session=linear_session,
            )
        if mtp_adaptive_fallback:
            mtp_adaptive_session = mtp_runtime.start_adaptive_session(
                mtp_session,
                mtp_runtime.MTPAdaptivePolicy(
                    minimum_mtp_blocks=mtp_adaptive_minimum_blocks,
                    window_blocks=mtp_adaptive_window_blocks,
                    minimum_future_acceptance=mtp_adaptive_minimum_acceptance,
                ),
            )
    elif turboquant_session is None and linear_session is None:
        decode_session = model.start_decode_session(
            weights,
            state,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=compiled_gdn_layers,
            compile_attention_tails=compiled_attention_tails,
        )
    logits = result.logits
    final_hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
    del result, state

    generated: list[int] = (
        [mtp_context.conditioned_token_id]
        if mtp_context is not None
        else []
    )
    transition_elapsed = 0.0
    stop = "eos" if generated and generated[-1] in tokenizer.eos_token_ids else "length"
    mtp_blocks = 0
    mtp_target_only_steps = 0
    mtp_future_proposed = 0
    mtp_future_accepted = 0
    if use_mtp:
        while stop != "eos" and len(generated) < max_tokens:
            started = time.perf_counter()
            if mtp_adaptive_session is not None:
                if temperature == 0.0:
                    mtp_step, mtp_adaptive_session = mtp_runtime.step_adaptive_greedy(
                        mtp_adaptive_session,
                        exact_long_attention=exact_long_attention,
                    )
                else:
                    mtp_step, mtp_adaptive_session = mtp_runtime.step_adaptive_sampled(
                        mtp_adaptive_session,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        rng=rng,
                        exact_long_attention=exact_long_attention,
                    )
            else:
                require_model(mtp_session is not None, "MTP decode session is missing")
                if temperature == 0.0:
                    mtp_step, mtp_session = mtp_runtime.step_greedy(
                        mtp_session,
                        exact_long_attention=exact_long_attention,
                    )
                else:
                    mtp_step, mtp_session = mtp_runtime.step_sampled(
                        mtp_session,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        rng=rng,
                        exact_long_attention=exact_long_attention,
                    )
            mx.synchronize()
            transition_elapsed += time.perf_counter() - started
            emitted = mtp_step.verification.emitted_tokens
            require_model(emitted, "MTP verifier emitted no target token")
            require_model(
                generated[-1] == emitted[0],
                "MTP anchor does not continue prior output",
            )
            if mtp_step.proposal.future_token_ids:
                mtp_blocks += 1
                mtp_future_proposed += len(mtp_step.proposal.future_token_ids)
                mtp_future_accepted += max(
                    0,
                    mtp_step.verification.accepted_count - 1,
                )
            else:
                mtp_target_only_steps += 1
            for token_id in emitted[1:]:
                generated.append(token_id)
                if token_id in tokenizer.eos_token_ids:
                    stop = "eos"
                    break
                if len(generated) == max_tokens:
                    break
        final_mtp_mode = (
            mtp_adaptive_session.mode
            if mtp_adaptive_session is not None
            else "mtp"
        )
        detached_after = (
            mtp_adaptive_session.detached_after_mtp_blocks
            if mtp_adaptive_session is not None
            else None
        )
        recent_acceptance = (
            mtp_runtime.adaptive_recent_future_acceptance(mtp_adaptive_session)
            if mtp_adaptive_session is not None
            else None
        )
        print(
            "generate-mtp-done "
            f"blocks={mtp_blocks} target_only_steps={mtp_target_only_steps} "
            f"future_accepted={mtp_future_accepted}/{mtp_future_proposed} "
            f"final_mode={final_mtp_mode} detached_after={detached_after} "
            f"recent_acceptance={recent_acceptance}",
            flush=True,
        )
    else:
        for step in range(max_tokens):
            started = time.perf_counter()
            next_id = choose_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
                hidden=final_hidden,
                lm_head=weights.lm_head,
            )
            generated.append(next_id)
            if next_id in tokenizer.eos_token_ids:
                stop = "eos"
                break
            if step + 1 == max_tokens:
                break
            if turboquant_session is not None:
                result = model.forward_turboquant_session_token(next_id, turboquant_session)
            elif linear_session is not None:
                result = model.forward_linear_session_token(next_id, linear_session)
            else:
                require_model(decode_session is not None, "decode session is missing")
                result, decode_session = model.forward_session_token(next_id, decode_session)
                model.evaluate_result(result)
            logits = result.logits
            final_hidden = result.hidden
            transition_elapsed += time.perf_counter() - started

    measured = max(len(generated) - 1, 0)
    speed = measured / transition_elapsed if transition_elapsed > 0.0 else 0.0
    response_ids = (
        generated[:-1]
        if generated and generated[-1] in tokenizer.eos_token_ids
        else generated
    )
    response = tokenizer.decode(response_ids)
    print(
        "generate-done "
        f"tokens={len(generated)} stop={stop} decode_tokens_s={speed:.3f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    reasoning, final = split_reasoning_response(response)
    if reasoning is not None:
        print("reasoning-begin", flush=True)
        print(reasoning, flush=True)
        print("reasoning-end", flush=True)
        print("response-begin", flush=True)
        print(final, flush=True)
        print("response-end", flush=True)
    else:
        print("response-begin", flush=True)
        print(response, flush=True)
        print("response-end", flush=True)
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=128,
        help="exact prefill chunk cap; use 1 for the token-serial authority",
    )
    parser.add_argument(
        "--linear-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use exact fixed-capacity, single-owner K/V buffers during decode",
    )
    parser.add_argument(
        "--turboquant-kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimentally compress native-context attention K/V after exact prefill",
    )
    parser.add_argument(
        "--compiled-gdn-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile exact fixed-shape GatedDeltaNet layers for decode",
    )
    parser.add_argument(
        "--compiled-attention-tails",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile exact fixed-shape residual/MoE tails after attention",
    )
    parser.add_argument(
        "--mapped-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read exact BF16 input rows from the verified source mapping",
    )
    parser.add_argument(
        "--quantized-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use Q8/32 vocabulary scoring with exact mapped BF16 top-64 reranking",
    )
    parser.add_argument(
        "--exact-long-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="batch exact BF16 attention after its measured long-prefix crossover",
    )
    parser.add_argument(
        "--context-profile",
        choices=tuple(context.SUPPORTED_CONTEXT_PROFILES),
        default=context.NATIVE_PROFILE_ID,
        help="immutable RoPE and capacity profile for this state and its caches",
    )
    parser.add_argument(
        "--load-cache",
        type=Path,
        help="verify and restore an explicit exact prefix-cache entry",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="atomically save the completed prompt state for a future longer prefix",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="cache storage root; defaults to MODEL_ROOT/cache",
    )
    parser.add_argument(
        "--cache-system-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="restore or atomically warm the exact rendered system prefix",
    )
    parser.add_argument(
        "--cache-longest-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="discover and strictly restore the longest cached exact prompt prefix",
    )
    parser.add_argument(
        "--cache-max-gib",
        type=float,
        default=24.0,
        help="bounded LRU cache budget when cache writing is enabled",
    )
    parser.add_argument(
        "--mtp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use exact target-verified MTP in the measured greedy thinking regime",
    )
    parser.add_argument(
        "--mtp-adaptation-dir",
        type=Path,
        help="folded MTP adaptation; defaults to the accepted artifact under MODEL_ROOT",
    )
    parser.add_argument("--mtp-block-tokens", type=int, default=3)
    parser.add_argument(
        "--mtp-max-prompt-tokens",
        type=int,
        default=256,
        help="MTP prompt ceiling; zero forces unmeasured length/sampling/thinking regimes",
    )
    parser.add_argument(
        "--mtp-adaptive-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="detach permanently to target-only decode after persistently weak MTP yield",
    )
    parser.add_argument("--mtp-adaptive-minimum-blocks", type=int, default=8)
    parser.add_argument("--mtp-adaptive-window-blocks", type=int, default=4)
    parser.add_argument(
        "--mtp-adaptive-minimum-acceptance",
        type=float,
        default=0.70,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        generate(
            args.root,
            args.prompt,
            system=args.system,
            enable_thinking=args.enable_thinking,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            prefill_chunk=args.prefill_chunk,
            linear_kv_cache=args.linear_kv_cache,
            turboquant_kv=args.turboquant_kv,
            compiled_gdn_layers=args.compiled_gdn_layers,
            compiled_attention_tails=args.compiled_attention_tails,
            mapped_embedding=args.mapped_embedding,
            quantized_lm_head=args.quantized_lm_head,
            exact_long_attention=args.exact_long_attention,
            context_profile=args.context_profile,
            load_cache=args.load_cache,
            save_cache=args.save_cache,
            cache_root=args.cache_root,
            cache_system_prefix=args.cache_system_prefix,
            cache_longest_prefix=args.cache_longest_prefix,
            cache_max_gib=args.cache_max_gib,
            use_mtp=args.mtp,
            mtp_adaptation_dir=args.mtp_adaptation_dir,
            mtp_block_tokens=args.mtp_block_tokens,
            mtp_max_prompt_tokens=args.mtp_max_prompt_tokens,
            mtp_adaptive_fallback=args.mtp_adaptive_fallback,
            mtp_adaptive_minimum_blocks=args.mtp_adaptive_minimum_blocks,
            mtp_adaptive_window_blocks=args.mtp_adaptive_window_blocks,
            mtp_adaptive_minimum_acceptance=args.mtp_adaptive_minimum_acceptance,
        )
    except (
        context.ContextError,
        MoEError,
        TokenizerError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ornith35 generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bounded text generation from the verified Ornith-35 target."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import mlx.core as mx

import ornith35_mlx_cache as persistent_cache
import ornith35_mlx_model as model
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError, require as require_model
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


def sample_candidates(
    token_ids: list[int],
    logits: list[float],
    *,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> int:
    """Sample a preselected top-k set using a deterministic CPU RNG."""
    require_model(len(token_ids) == len(logits) > 0, "candidate shape mismatch")
    require_model(temperature > 0.0, "sampling temperature must be positive")
    require_model(0.0 < top_p <= 1.0, "top-p must be in (0, 1]")
    ranked = sorted(zip(token_ids, logits), key=lambda item: (-item[1], item[0]))
    maximum = ranked[0][1]
    weights = [math.exp((value - maximum) / temperature) for _, value in ranked]
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]

    retained = 1
    cumulative = probabilities[0]
    while retained < len(probabilities) and cumulative < top_p:
        cumulative += probabilities[retained]
        retained += 1
    threshold = rng.random() * math.fsum(probabilities[:retained])
    cumulative = 0.0
    for (token_id, _), probability in zip(ranked[:retained], probabilities[:retained]):
        cumulative += probability
        if threshold <= cumulative:
            return token_id
    return ranked[retained - 1][0]


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
    require_model(logits.ndim == 1, "target logits must be a vector")
    require_model(temperature >= 0.0, "temperature must be nonnegative")
    if (
        isinstance(lm_head, vocab.MLXAffineQuantizedMatrix)
        and lm_head.reference is not None
    ):
        require_model(hidden is not None, "hybrid LM head requires final hidden state")
        require_model(
            temperature == 0.0 or top_k <= 256,
            "hybrid LM head supports sampled top-k at most 256",
        )
        candidate_count = 64 if temperature == 0.0 else max(64, top_k)
        token_ids, values = vocab.exact_candidate_scores(
            lm_head,
            logits,
            hidden,
            candidate_count=candidate_count,
        )
        ranked = sorted(zip(token_ids, values), key=lambda item: (-item[1], item[0]))
        if temperature == 0.0:
            return ranked[0][0]
        selected = ranked[:top_k]
        return sample_candidates(
            [token_id for token_id, _ in selected],
            [value for _, value in selected],
            temperature=temperature,
            top_p=top_p,
            rng=rng,
        )
    if temperature == 0.0:
        return int(mx.argmax(logits).item())
    require_model(0 < top_k <= logits.size, "top-k is outside the vocabulary")
    require_model(0.0 < top_p <= 1.0, "top-p must be in (0, 1]")
    indices = mx.argpartition(logits, logits.size - top_k)[-top_k:]
    values = mx.take(logits, indices).astype(mx.float32)
    mx.eval(indices, values)
    return sample_candidates(
        [int(value) for value in indices.tolist()],
        [float(value) for value in values.tolist()],
        temperature=temperature,
        top_p=top_p,
        rng=rng,
    )


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
    compiled_gdn_layers: bool,
    mapped_embedding: bool,
    quantized_lm_head: bool,
    exact_long_attention: bool,
    load_cache: Path | None,
    save_cache: bool,
    cache_root: Path | None,
    cache_system_prefix: bool,
    cache_max_gib: float,
) -> str:
    require_model(0 < max_tokens <= 4096, "max tokens must be between 1 and 4096")
    require_model(temperature >= 0.0, "temperature must be nonnegative")
    require_model(0 < top_k <= model.PRODUCTION_CONFIG.vocab_size, "invalid top-k")
    require_model(0.0 < top_p <= 1.0, "invalid top-p")
    require_model(
        not quantized_lm_head or temperature == 0.0 or top_k <= 256,
        "hybrid LM head sampling supports top-k at most 256",
    )
    require_model(cache_max_gib > 0.0, "cache size budget must be positive")
    prefill_schedule(1, prefill_chunk)
    tokenizer = load_text_tokenizer(root)
    rendered = render_text_prompt(
        prompt,
        system=system,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(rendered)
    require_model(prompt_ids, "rendered prompt produced no tokens")
    cache_enabled = load_cache is not None or save_cache or cache_system_prefix
    cache_identity = (
        persistent_cache.production_identity(
            root,
            Path(__file__).resolve().parents[2],
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            chat_template_sha256=tokenizer.template_sha256,
            mapped_embedding=mapped_embedding,
            quantized_lm_head=quantized_lm_head,
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
        if load_cache is None:
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
            f"path={restored.path}",
            flush=True,
        )
    cache_restored = restored is not None
    protected_cache_keys = {restored.key} if restored is not None else set()
    print(
        "generate-start "
        f"prompt_tokens={len(prompt_ids)} max_tokens={max_tokens} "
        f"thinking={str(enable_thinking).lower()} temperature={temperature:.6g} "
        f"top_k={top_k} top_p={top_p:.6g} seed={seed} "
        f"prefill_chunk={prefill_chunk} "
        f"linear_kv_cache={str(linear_kv_cache).lower()} "
        f"compiled_gdn_layers={str(compiled_gdn_layers).lower()} "
        f"mapped_embedding={str(mapped_embedding).lower()} "
        f"quantized_lm_head={str(quantized_lm_head).lower()} "
        f"exact_long_attention={str(exact_long_attention).lower()} "
        f"cache_system_prefix={str(cache_system_prefix).lower()}",
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

    state = (
        restored.state
        if restored is not None
        else model.initial_state(weights, model.PRODUCTION_CONFIG)
    )
    linear_session = (
        model.start_linear_decode_session(
            weights,
            state,
            len(prompt_ids) + max_tokens,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=compiled_gdn_layers,
        )
        if linear_kv_cache
        else None
    )
    if linear_session is not None:
        state = linear_session.state
        restored = None
    if not cache_restored and system_prefix_ids:
        warm_started = time.perf_counter()
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
        )
        protected_cache_keys.add(saved_system.name)
        print(
            "generate-cache-warmed "
            f"tokens={len(system_prefix_ids)} prefill_s={warm_elapsed:.3f} "
            f"save_s={time.perf_counter() - cache_started:.3f} "
            f"chunks={format_prefill_schedule(warm_schedule)} path={saved_system}",
            flush=True,
        )
    cached_tokens = state.position
    suffix_ids = list(prompt_ids[cached_tokens:])
    require_model(suffix_ids, "prompt cache left no suffix to evaluate")
    prefill_started = time.perf_counter()
    result, schedule = prefill_prompt(
        suffix_ids,
        state,
        weights,
        max_chunk=prefill_chunk,
        linear_session=linear_session,
        exact_long_attention=exact_long_attention,
    )
    state = result.state
    prefill_elapsed = time.perf_counter() - prefill_started
    print(
        "generate-prefill-done "
        f"tokens={len(suffix_ids)} cached_tokens={cached_tokens} "
        f"total_tokens={len(prompt_ids)} elapsed_s={prefill_elapsed:.3f} "
        f"tokens_s={len(suffix_ids) / prefill_elapsed:.3f} "
        f"chunks={format_prefill_schedule(schedule)} "
        "steel=false logit_projections=1",
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
        )
        protected_cache_keys.add(saved_path.name)
        print(
            "generate-cache-saved "
            f"tokens={len(prompt_ids)} elapsed_s={time.perf_counter() - cache_started:.3f} "
            f"path={saved_path}",
            flush=True,
        )
    if save_cache or cache_system_prefix:
        pruned = persistent_cache.prune_cache(
            cache_destination,
            max_bytes=int(cache_max_gib * 2**30),
            max_entries=64,
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
    if linear_session is not None:
        decode_session = None
    else:
        decode_session = model.start_decode_session(
            weights,
            state,
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=compiled_gdn_layers,
        )
        linear_session = None
    logits = result.logits
    final_hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
    del result, state

    generated: list[int] = []
    transition_elapsed = 0.0
    stop = "length"
    rng = random.Random(seed)
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
        if linear_session is not None:
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
        "--compiled-gdn-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile exact fixed-shape GatedDeltaNet layers for decode",
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
        "--cache-max-gib",
        type=float,
        default=24.0,
        help="bounded LRU cache budget when cache writing is enabled",
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
            compiled_gdn_layers=args.compiled_gdn_layers,
            mapped_embedding=args.mapped_embedding,
            quantized_lm_head=args.quantized_lm_head,
            exact_long_attention=args.exact_long_attention,
            load_cache=args.load_cache,
            save_cache=args.save_cache,
            cache_root=args.cache_root,
            cache_system_prefix=args.cache_system_prefix,
            cache_max_gib=args.cache_max_gib,
        )
    except (MoEError, TokenizerError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ornith35 generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

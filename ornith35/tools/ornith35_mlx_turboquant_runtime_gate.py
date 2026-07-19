#!/usr/bin/env python3
"""Real-checkpoint quality and timing gate for direct packed K4-MSE decode."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_cache as persistent_cache
import ornith35_mlx_generate as generate
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
import ornith35_mlx_turboquant_cache as turboquant_cache
from ornith35_mlx_turboquant_characterize import (
    PromptSpec,
    encode_bounded_prompt,
    token_sha256,
)
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT
from ornith35_tokenizer import TextTokenizer, load_text_tokenizer, render_text_prompt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = REPOSITORY_ROOT / "tests" / "long_context_security_prompt.txt"
HEAD_TAIL_MARKER = "\n\n[... bounded evaluation omitted middle context ...]\n\n"
FINAL_TASK_MARKER = "\nFinal task:"


def encode_head_tail_prompt(
    tokenizer: TextTokenizer,
    path: Path,
    max_tokens: int,
    suffix_characters: int,
) -> tuple[int, ...]:
    require(path.is_file(), f"missing prompt source: {path}")
    require(suffix_characters > 0, "preserved suffix must be positive")
    text = path.read_text(encoding="utf-8").strip()
    require(text, f"empty prompt source: {path}")
    full = tokenizer.encode(render_text_prompt(text, enable_thinking=True))
    if len(full) <= max_tokens:
        return full
    suffix_characters = min(suffix_characters, len(text))
    suffix = text[-suffix_characters:]
    boundary = len(text) - suffix_characters

    def encode(prefix_characters: int) -> tuple[int, ...]:
        content = text[:prefix_characters] + HEAD_TAIL_MARKER + suffix
        return tokenizer.encode(render_text_prompt(content, enable_thinking=True))

    suffix_only = encode(0)
    require(
        len(suffix_only) <= max_tokens,
        "preserved prompt suffix exceeds the token bound",
    )
    low = 0
    high = boundary
    best = suffix_only
    while low <= high:
        middle = (low + high) // 2
        candidate = encode(middle)
        if len(candidate) <= max_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def encode_padded_before_final_prompt(
    tokenizer: TextTokenizer,
    path: Path,
    target_tokens: int,
) -> tuple[int, ...]:
    require(path.is_file(), f"missing prompt source: {path}")
    text = path.read_text(encoding="utf-8").strip()
    require(text, f"empty prompt source: {path}")
    insertion = text.rfind(FINAL_TASK_MARKER)
    require(insertion >= 0, "prompt has no final-task insertion boundary")
    head = text[:insertion]
    tail = text[insertion:]

    def encode(records: int) -> tuple[int, ...]:
        filler = "\n".join(
            f"Archive filler record {index:06d}: ordinary harbor weather, repairs, "
            "and errands; this record contains no assignment fact."
            for index in range(records)
        )
        content = head + "\n\n" + filler + "\n" + tail
        return tokenizer.encode(render_text_prompt(content, enable_thinking=True))

    unpadded = encode(0)
    require(len(unpadded) < target_tokens, "prompt already meets or exceeds the padding target")
    low = 0
    high = 1
    best = unpadded
    while len(candidate := encode(high)) <= target_tokens:
        best = candidate
        low = high
        high *= 2
        require(high <= 1_048_576, "prompt padding search exceeded its bound")
    while low + 1 < high:
        middle = (low + high) // 2
        candidate = encode(middle)
        if len(candidate) <= target_tokens:
            best = candidate
            low = middle
        else:
            high = middle
    return best


def compare_logits(source: mx.array, candidate: mx.array) -> dict[str, float | int | bool]:
    source32 = source.astype(mx.float32)
    candidate32 = candidate.astype(mx.float32)
    source_top = mx.argmax(source32)
    candidate_top = mx.argmax(candidate32)
    source_top8 = mx.argpartition(source32, source32.size - 8)[-8:]
    candidate_top8 = mx.argpartition(candidate32, candidate32.size - 8)[-8:]
    source_log = source32 - mx.logsumexp(source32)
    candidate_log = candidate32 - mx.logsumexp(candidate32)
    divergence = mx.sum(mx.exp(source_log) * (source_log - candidate_log))
    maximum = mx.max(mx.abs(candidate32 - source32))
    mx.eval(
        source_top,
        candidate_top,
        source_top8,
        candidate_top8,
        divergence,
        maximum,
    )
    source_ids = set(source_top8.tolist())
    candidate_ids = set(candidate_top8.tolist())
    source_id = int(source_top.item())
    candidate_id = int(candidate_top.item())
    source_top8_ids = source_top8.tolist()
    source_top8_values = source32[source_top8].tolist()
    ranked_source = sorted(
        zip(source_top8_ids, source_top8_values),
        key=lambda item: (-item[1], item[0]),
    )
    choice_ids = mx.array((source_id, candidate_id), dtype=mx.int32)
    source_choice_values = source32[choice_ids]
    candidate_choice_values = candidate32[choice_ids]
    mx.eval(source_choice_values, candidate_choice_values)
    source_choice_scores = source_choice_values.tolist()
    candidate_choice_scores = candidate_choice_values.tolist()
    return {
        "source_top": source_id,
        "candidate_top": candidate_id,
        "top1": source_id == candidate_id,
        "top8_recall": len(source_ids & candidate_ids) / 8,
        "kl": max(0.0, float(divergence.item())),
        "max_abs": float(maximum.item()),
        "source_margin": float(ranked_source[0][1] - ranked_source[1][1]),
        "source_choice_gap": float(source_choice_scores[0] - source_choice_scores[1]),
        "candidate_choice_gap": float(
            candidate_choice_scores[1] - candidate_choice_scores[0]
        ),
    }


def packed_bytes(state: model.TextModelState) -> int:
    total = 0
    for layer_state in state.layers:
        if isinstance(layer_state, attention.MLXTurboQuantAttentionState):
            total += turboquant_cache.stored_bytes(layer_state)
    return total


def run_independent_greedy(
    first_token: int,
    token_limit: int,
    eos_token_ids: frozenset[int],
    session: model.TextLinearDecodeSession | model.TextTurboQuantDecodeSession,
) -> tuple[tuple[int, ...], float]:
    tokens: list[int] = []
    elapsed = 0.0
    token_id = first_token
    for _ in range(token_limit):
        tokens.append(token_id)
        if token_id in eos_token_ids or len(tokens) == token_limit:
            break
        started = time.perf_counter()
        if isinstance(session, model.TextLinearDecodeSession):
            result = model.forward_linear_session_token(token_id, session)
        else:
            result = model.forward_turboquant_session_token(token_id, session)
        elapsed += time.perf_counter() - started
        token_id = int(mx.argmax(result.logits).item())
    return tuple(tokens), elapsed


def run_independent_sampled(
    initial_logits: mx.array,
    initial_hidden: mx.array,
    token_limit: int,
    eos_token_ids: frozenset[int],
    session: model.TextLinearDecodeSession | model.TextTurboQuantDecodeSession,
    lm_head: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> tuple[tuple[int, ...], float]:
    tokens: list[int] = []
    elapsed = 0.0
    logits = initial_logits
    hidden = initial_hidden
    rng = random.Random(seed)
    for _ in range(token_limit):
        token_id = generate.choose_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
            hidden=hidden,
            lm_head=lm_head,
        )
        tokens.append(token_id)
        if token_id in eos_token_ids or len(tokens) == token_limit:
            break
        started = time.perf_counter()
        if isinstance(session, model.TextLinearDecodeSession):
            result = model.forward_linear_session_token(token_id, session)
        else:
            result = model.forward_turboquant_session_token(token_id, session)
        logits = result.logits
        hidden = result.hidden
        elapsed += time.perf_counter() - started
    return tuple(tokens), elapsed


def required_line_coverage(
    response: str,
    required_lines: tuple[str, ...],
) -> tuple[int, tuple[str, ...]]:
    missing = tuple(line for line in required_lines if line not in response)
    return len(required_lines) - len(missing), missing


def require_persisted_state_equal(
    source: model.TextModelState,
    restored: model.TextModelState,
) -> None:
    require(source.position == restored.position, "persisted state position changed")
    require(
        source.context_profile == restored.context_profile,
        "persisted context profile changed",
    )
    require(len(source.layers) == len(restored.layers), "persisted layer count changed")
    for index, (left, right) in enumerate(zip(source.layers, restored.layers)):
        if isinstance(left, gdn.MLXGDNState):
            require(isinstance(right, gdn.MLXGDNState), f"persisted GDN type changed at {index}")
            require(
                bool(mx.array_equal(left.conv, right.conv).item()),
                f"persisted convolution state changed at {index}",
            )
            require(
                bool(mx.array_equal(left.recurrent, right.recurrent).item()),
                f"persisted recurrent state changed at {index}",
            )
            continue
        require(
            isinstance(left, attention.MLXTurboQuantAttentionState)
            and isinstance(right, attention.MLXTurboQuantImmutableAttentionState),
            f"persisted packed state type changed at {index}",
        )
        history = turboquant_cache.packed_history(left)
        for name in ("packed_keys", "key_norms", "packed_values", "value_norms"):
            require(
                bool(
                    mx.array_equal(
                        getattr(left, name)[:, :history],
                        getattr(right, name),
                    ).item()
                ),
                f"persisted packed payload changed at {index}:{name}",
            )
        for name in (
            "exact_head_keys",
            "exact_head_values",
            "exact_keys",
            "exact_values",
        ):
            require(
                bool(mx.array_equal(getattr(left, name), getattr(right, name)).item()),
                f"persisted exact boundary changed at {index}:{name}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--greedy-tokens", type=int, default=0)
    parser.add_argument("--sampled-tokens", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--required-lines",
        type=Path,
        help="nonempty lines that each independent response must contain",
    )
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument(
        "--preserve-suffix-characters",
        type=int,
        default=0,
        help="retain this many final source characters while bounding the prompt",
    )
    parser.add_argument(
        "--pad-before-final",
        action="store_true",
        help="insert deterministic assignment-free records before the final task",
    )
    parser.add_argument(
        "--persistence-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.prompt_tokens >= 32, "prompt token bound is too small")
        require(1 <= args.steps <= 64, "trajectory steps must be in [1, 64]")
        require(0 <= args.greedy_tokens <= 1024, "greedy token count must be in [0, 1024]")
        require(0 <= args.sampled_tokens <= 1024, "sampled token count must be in [0, 1024]")
        require(
            not args.greedy_tokens or not args.sampled_tokens,
            "greedy and sampled independent generation are mutually exclusive",
        )
        require(
            not args.sampled_tokens or args.temperature > 0.0,
            "sampled generation temperature must be positive",
        )
        require(0 < args.top_k <= model.PRODUCTION_CONFIG.vocab_size, "invalid sampled top-k")
        require(0.0 < args.top_p <= 1.0, "invalid sampled top-p")
        require(
            args.preserve_suffix_characters >= 0,
            "preserved suffix character count must be nonnegative",
        )
        require(
            not args.pad_before_final or args.preserve_suffix_characters == 0,
            "prompt suffix preservation and pre-final padding are mutually exclusive",
        )
        require(args.chunk in (8, 16, 32, 64, 128), "invalid prefill chunk")
        required_lines: tuple[str, ...] = ()
        if args.required_lines is not None:
            require(args.required_lines.is_file(), "required-line file is missing")
            required_lines = tuple(
                line.strip()
                for line in args.required_lines.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            require(required_lines, "required-line file is empty")
            require(
                len(set(required_lines)) == len(required_lines),
                "required-line file contains duplicates",
            )
            require(
                args.greedy_tokens > 0 or args.sampled_tokens > 0,
                "required-line scoring needs independent generation",
            )
        tokenizer = load_text_tokenizer(args.root)
        if args.pad_before_final:
            prompt_ids = encode_padded_before_final_prompt(
                tokenizer,
                args.prompt,
                args.prompt_tokens,
            )
        elif args.preserve_suffix_characters:
            prompt_ids = encode_head_tail_prompt(
                tokenizer,
                args.prompt,
                args.prompt_tokens,
                args.preserve_suffix_characters,
            )
        else:
            prompt_ids = encode_bounded_prompt(
                tokenizer,
                PromptSpec("runtime-gate", "holdout", args.prompt),
                args.prompt_tokens,
            ).token_ids
        prompt_mode = (
            "pad-before-final"
            if args.pad_before_final
            else "head-tail"
            if args.preserve_suffix_characters
            else "prefix"
        )
        print(
            "turboquant-runtime-prompt "
            f"mode={prompt_mode} tokens={len(prompt_ids)} "
            f"sha256={token_sha256(prompt_ids)}",
            flush=True,
        )
        load_started = time.perf_counter()
        weights = model.load_text_model(args.root)
        print(
            "turboquant-runtime-model-ready "
            f"load_s={time.perf_counter() - load_started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        generation_tokens = max(args.greedy_tokens, args.sampled_tokens)
        generation_mode = "sampled" if args.sampled_tokens else "greedy"
        capacity = len(prompt_ids) + max(args.steps, generation_tokens)
        state = model.initial_state(weights, model.PRODUCTION_CONFIG)
        cache_init_started = time.perf_counter()
        exact = model.start_linear_decode_session(weights, state, capacity)
        cache_init_elapsed = time.perf_counter() - cache_init_started
        prefill_started = time.perf_counter()
        prefill, schedule = generate.prefill_prompt(
            list(prompt_ids),
            state,
            weights,
            max_chunk=args.chunk,
            linear_session=exact,
        )
        prefill_elapsed = time.perf_counter() - prefill_started
        require(prefill.state is exact.state, "production prefill session/state mismatch")
        prefix_state = prefill.state
        prefix_checkpoint = model.checkpoint_linear_session_state(exact)
        first_token = int(mx.argmax(prefill.logits).item())
        if generation_tokens:
            greedy_packed = model.start_turboquant_decode_session(
                weights,
                prefix_state,
                capacity,
            )
            if args.sampled_tokens:
                initial_hidden = (
                    prefill.hidden[-1]
                    if prefill.hidden.ndim == 2
                    else prefill.hidden
                )
                sample_kwargs = {
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "seed": args.sample_seed,
                }
                exact_tokens, exact_elapsed = run_independent_sampled(
                    prefill.logits,
                    initial_hidden,
                    generation_tokens,
                    tokenizer.eos_token_ids,
                    exact,
                    weights.lm_head,
                    **sample_kwargs,
                )
                packed_tokens, packed_elapsed = run_independent_sampled(
                    prefill.logits,
                    initial_hidden,
                    generation_tokens,
                    tokenizer.eos_token_ids,
                    greedy_packed,
                    weights.lm_head,
                    **sample_kwargs,
                )
            else:
                exact_tokens, exact_elapsed = run_independent_greedy(
                    first_token,
                    generation_tokens,
                    tokenizer.eos_token_ids,
                    exact,
                )
                packed_tokens, packed_elapsed = run_independent_greedy(
                    first_token,
                    generation_tokens,
                    tokenizer.eos_token_ids,
                    greedy_packed,
                )
            compared = min(len(exact_tokens), len(packed_tokens))
            common_prefix = 0
            while (
                common_prefix < compared
                and exact_tokens[common_prefix] == packed_tokens[common_prefix]
            ):
                common_prefix += 1
            position_matches = sum(
                left == right
                for left, right in zip(exact_tokens, packed_tokens)
            )
            exact_response = tokenizer.decode(
                exact_tokens[:-1]
                if exact_tokens and exact_tokens[-1] in tokenizer.eos_token_ids
                else exact_tokens
            )
            packed_response = tokenizer.decode(
                packed_tokens[:-1]
                if packed_tokens and packed_tokens[-1] in tokenizer.eos_token_ids
                else packed_tokens
            )
            exact_rate = (
                (len(exact_tokens) - 1) / exact_elapsed
                if exact_elapsed > 0.0
                else 0.0
            )
            packed_rate = (
                (len(packed_tokens) - 1) / packed_elapsed
                if packed_elapsed > 0.0
                else 0.0
            )
            print(
                "turboquant-runtime-generation "
                f"mode={generation_mode} "
                f"exact_tokens={len(exact_tokens)} packed_tokens={len(packed_tokens)} "
                f"common_prefix={common_prefix} position_matches={position_matches}/{compared} "
                f"exact_sha256={hashlib.sha256(exact_response.encode('utf-8')).hexdigest()} "
                f"packed_sha256={hashlib.sha256(packed_response.encode('utf-8')).hexdigest()} "
                f"exact_tokens_s={exact_rate:.3f} "
                f"packed_tokens_s={packed_rate:.3f}",
                flush=True,
            )
            print(f"turboquant-runtime-{generation_mode}-exact-begin", flush=True)
            print(exact_response, flush=True)
            print(f"turboquant-runtime-{generation_mode}-exact-end", flush=True)
            print(f"turboquant-runtime-{generation_mode}-packed-begin", flush=True)
            print(packed_response, flush=True)
            print(f"turboquant-runtime-{generation_mode}-packed-end", flush=True)
            if required_lines:
                exact_coverage, exact_missing = required_line_coverage(
                    exact_response,
                    required_lines,
                )
                packed_coverage, packed_missing = required_line_coverage(
                    packed_response,
                    required_lines,
                )
                print(
                    "turboquant-runtime-required-lines "
                    f"exact={exact_coverage}/{len(required_lines)} "
                    f"packed={packed_coverage}/{len(required_lines)} "
                    f"exact_missing={','.join(exact_missing) if exact_missing else 'none'} "
                    f"packed_missing={','.join(packed_missing) if packed_missing else 'none'}",
                    flush=True,
                )
                require(
                    exact_coverage == len(required_lines),
                    "BF16 authority failed required-line coverage",
                )
                require(
                    packed_coverage == len(required_lines),
                    "TurboQuant failed required-line coverage",
                )
            prefix_state = model.restore_linear_session_checkpoint(exact, prefix_checkpoint)
            del greedy_packed
            gc.collect()
            mx.clear_cache()
        conversion_started = time.perf_counter()
        packed = model.start_turboquant_decode_session(weights, prefix_state, capacity)
        conversion_elapsed = time.perf_counter() - conversion_started
        model.validate_turboquant_decode_session(packed)
        if args.persistence_check:
            persistence_started = time.perf_counter()
            cache_identity = persistent_cache.production_identity(
                args.root,
                REPOSITORY_ROOT,
                tokenizer_sha256=tokenizer.tokenizer_sha256,
                chat_template_sha256=tokenizer.template_sha256,
                mapped_embedding=False,
                quantized_lm_head=False,
                turboquant_kv=True,
            )
            with tempfile.TemporaryDirectory(prefix="ornith35-turboquant-gate-") as temporary:
                cache_path = persistent_cache.save_cache(
                    Path(temporary),
                    prompt_ids,
                    packed.state,
                    cache_identity,
                    model.PRODUCTION_CONFIG,
                )
                persisted_bytes = sum(
                    path.stat().st_size
                    for path in cache_path.rglob("*")
                    if path.is_file()
                )
                restored = persistent_cache.load_cache(
                    cache_path,
                    cache_identity,
                    model.PRODUCTION_CONFIG,
                    expected_tokens=prompt_ids,
                )
                require_persisted_state_equal(packed.state, restored.state)
                packed = model.start_turboquant_decode_session(
                    weights,
                    restored.state,
                    capacity,
                )
                model.validate_turboquant_decode_session(packed)
            print(
                "turboquant-runtime-persistence "
                f"bytes={persisted_bytes} "
                f"elapsed_s={time.perf_counter() - persistence_started:.3f} "
                "verified=true resumed=true",
                flush=True,
            )
        print(
            "turboquant-runtime-ready "
            f"tokens={prefill.state.position} "
            f"schedule={generate.format_prefill_schedule(schedule)} "
            f"cache_init_s={cache_init_elapsed:.3f} "
            f"prefill_s={prefill_elapsed:.3f} "
            f"conversion_s={conversion_elapsed:.3f} "
            f"packed_mib={packed_bytes(packed.state) / 2**20:.6f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )

        token_id = first_token
        exact_times: list[float] = []
        packed_times: list[float] = []
        reports = []
        for step in range(args.steps):
            operations = (
                ("exact", lambda: model.forward_linear_session_token(token_id, exact)),
                ("packed", lambda: model.forward_turboquant_session_token(token_id, packed)),
            )
            if step % 2:
                operations = tuple(reversed(operations))
            samples = {}
            for name, operation in operations:
                started = time.perf_counter()
                samples[name] = operation()
                elapsed = time.perf_counter() - started
                (exact_times if name == "exact" else packed_times).append(elapsed)
            report = compare_logits(samples["exact"].logits, samples["packed"].logits)
            reports.append(report)
            print(
                "turboquant-runtime-step "
                f"step={step + 1}/{args.steps} token={token_id} "
                f"source={report['source_top']} candidate={report['candidate_top']} "
                f"top1={str(report['top1']).lower()} "
                f"top8_recall={report['top8_recall']:.3f} "
                f"kl={report['kl']:.9g} max_abs={report['max_abs']:.9g} "
                f"source_margin={report['source_margin']:.9g} "
                f"source_choice_gap={report['source_choice_gap']:.9g} "
                f"candidate_choice_gap={report['candidate_choice_gap']:.9g}",
                flush=True,
            )
            token_id = int(mx.argmax(samples["exact"].logits).item())

        agreements = sum(int(report["top1"]) for report in reports)
        mean_recall = statistics.mean(float(report["top8_recall"]) for report in reports)
        mean_kl = statistics.mean(float(report["kl"]) for report in reports)
        max_kl = max(float(report["kl"]) for report in reports)
        exact_median = statistics.median(exact_times)
        packed_median = statistics.median(packed_times)
        print(
            "turboquant-runtime-result "
            f"top1={agreements}/{args.steps} mean_top8_recall={mean_recall:.6f} "
            f"mean_kl={mean_kl:.9g} max_kl={max_kl:.9g} "
            f"exact_ms={exact_median * 1000:.3f} "
            f"packed_ms={packed_median * 1000:.3f} "
            f"speedup={exact_median / packed_median:.4f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
    except (MoEError, OSError, ValueError) as exc:
        print(f"TurboQuant runtime gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

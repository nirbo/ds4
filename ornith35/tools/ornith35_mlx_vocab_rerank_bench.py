#!/usr/bin/env python3
"""Balanced production-path benchmark for the hybrid Q8/BF16 LM head."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import require_verified_source
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(retained)


def compare_linear_states(
    source: model.TextLinearDecodeSession,
    candidate: model.TextLinearDecodeSession,
) -> int:
    require(source.state.position == candidate.state.position, "session position mismatch")
    checks = []
    for left, right in zip(source.state.layers, candidate.state.layers):
        if isinstance(left, gdn.MLXGDNState):
            require(isinstance(right, gdn.MLXGDNState), "candidate GDN state mismatch")
            checks.extend(
                (
                    mx.array_equal(left.conv, right.conv),
                    mx.array_equal(left.recurrent, right.recurrent),
                )
            )
        else:
            require(
                isinstance(left, attention.MLXLinearAttentionState)
                and isinstance(right, attention.MLXLinearAttentionState),
                "candidate attention state mismatch",
            )
            checks.extend(
                (
                    mx.array_equal(left.keys, right.keys),
                    mx.array_equal(left.values, right.values),
                )
            )
    mx.eval(*checks)
    require(all(bool(value.item()) for value in checks), "candidate trajectory state drift")
    return len(checks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompt",
        default="Implement an LRU cache in Python with O(1) get and put operations.",
    )
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.warmup >= 4, "benchmark warmup must be at least four")
        require(32 <= args.rounds <= 1024, "benchmark rounds must be in [32, 1024]")
        require(args.temperature >= 0.0, "benchmark temperature must be nonnegative")
        require(0 < args.top_k <= 256, "benchmark top-k must be in [1, 256]")
        require(0.0 < args.top_p <= 1.0, "benchmark top-p must be in (0, 1]")
        load_started = time.perf_counter()
        source = model.load_text_model(args.root, map_embedding=True)
        require(isinstance(source.lm_head, mx.array), "source LM head must be BF16")
        reference = vocab.MLXMappedBF16Matrix(
            require_verified_source(args.root),
            "lm_head.weight",
            (model.PRODUCTION_CONFIG.vocab_size, model.PRODUCTION_CONFIG.hidden_size),
        )
        quantized = vocab.quantize_affine(
            source.lm_head,
            bits=8,
            group_size=32,
            reference=reference,
        )
        mx.eval(quantized.packed, quantized.scales, quantized.biases)
        mx.synchronize()
        candidate = model.TextModelWeights(
            embedding=source.embedding,
            layers=source.layers,
            final_norm=source.final_norm,
            lm_head=quantized,
        )
        model.validate_weights(candidate, model.PRODUCTION_CONFIG)
        tokenizer = load_text_tokenizer(args.root)
        prompt_ids = tokenizer.encode(
            render_text_prompt(args.prompt, enable_thinking=True)
        )
        initial = model.initial_state(source, model.PRODUCTION_CONFIG)
        source_result, _ = generate.prefill_prompt(
            list(prompt_ids),
            initial,
            source,
            max_chunk=128,
        )
        hidden = (
            source_result.hidden[-1]
            if source_result.hidden.ndim == 2
            else source_result.hidden
        )
        candidate_logits = model.project_lm_head(quantized, hidden)
        mx.eval(candidate_logits)
        capacity = source_result.state.position + args.warmup + args.rounds + 1
        source_session = model.start_linear_decode_session(
            source,
            source_result.state,
            capacity,
            model.PRODUCTION_CONFIG,
        )
        candidate_session = model.start_linear_decode_session(
            candidate,
            source_result.state,
            capacity,
            model.PRODUCTION_CONFIG,
        )
        source_logits = source_result.logits
        source_hidden = hidden
        candidate_hidden = hidden
        source_rng = random.Random(args.seed)
        candidate_rng = random.Random(args.seed)
        source_times: list[float] = []
        candidate_times: list[float] = []
        choices = 0
        print(
            "vocab-rerank-ready "
            f"prompt_tokens={len(prompt_ids)} load_s={time.perf_counter() - load_started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )

        def advance_source() -> tuple[int, float]:
            nonlocal source_logits, source_hidden
            started = time.perf_counter()
            token_id = generate.choose_next_token(
                source_logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                rng=source_rng,
                hidden=source_hidden,
                lm_head=source.lm_head,
            )
            result = model.forward_linear_session_token(token_id, source_session)
            source_logits = result.logits
            source_hidden = result.hidden
            return token_id, time.perf_counter() - started

        def advance_candidate() -> tuple[int, float]:
            nonlocal candidate_logits, candidate_hidden
            started = time.perf_counter()
            token_id = generate.choose_next_token(
                candidate_logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                rng=candidate_rng,
                hidden=candidate_hidden,
                lm_head=candidate.lm_head,
            )
            result = model.forward_linear_session_token(token_id, candidate_session)
            candidate_logits = result.logits
            candidate_hidden = result.hidden
            return token_id, time.perf_counter() - started

        for step in range(args.warmup + args.rounds):
            if step % 2:
                candidate_token, candidate_time = advance_candidate()
                source_token, source_time = advance_source()
            else:
                source_token, source_time = advance_source()
                candidate_token, candidate_time = advance_candidate()
            require(source_token == candidate_token, f"candidate choice drift at step {step}")
            choices += 1
            if step >= args.warmup:
                source_times.append(source_time)
                candidate_times.append(candidate_time)
        exact_states = compare_linear_states(source_session, candidate_session)
        require(
            bool(mx.array_equal(source_hidden, candidate_hidden).item()),
            "candidate hidden-state drift",
        )
        source_mean = trimmed_mean(source_times)
        candidate_mean = trimmed_mean(candidate_times)
        print(
            "vocab-rerank-done "
            f"choices={choices}/{choices} state_exact={exact_states} "
            f"source_ms={source_mean * 1000:.3f} "
            f"candidate_ms={candidate_mean * 1000:.3f} "
            f"source_tok_s={1.0 / source_mean:.3f} "
            f"candidate_tok_s={1.0 / candidate_mean:.3f} "
            f"speedup={source_mean / candidate_mean:.4f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
    except (MoEError, OSError, TokenizerError, ValueError) as exc:
        print(f"vocab rerank benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

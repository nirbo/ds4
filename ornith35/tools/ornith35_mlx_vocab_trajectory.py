#!/usr/bin/env python3
"""Teacher-forced source comparison for quantized Ornith-35 vocabulary weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

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


DEFAULT_PROMPTS = (
    "Implement an LRU cache in Python and explain its complexity.",
    "Debug a concurrent queue, identify its races, and provide corrected code.",
    "Prove that every finite tree with at least two vertices has two leaves.",
)


def final_hidden(result: model.TextModelResult | model.TextModelChunkResult) -> mx.array:
    return result.hidden[-1] if result.hidden.ndim == 2 else result.hidden


def final_routes(
    result: model.TextModelResult | model.TextModelChunkResult,
) -> tuple[mx.array, ...]:
    return tuple(route[-1] if route.ndim == 2 else route for route in result.selected_experts)


def compare_step(
    source: model.TextModelResult | model.TextModelChunkResult,
    candidate: model.TextModelResult | model.TextModelChunkResult,
) -> tuple[bool, float, float, float, int]:
    source_logits = source.logits.astype(mx.float32)
    candidate_logits = candidate.logits.astype(mx.float32)
    logit_difference = candidate_logits - source_logits
    logit_l2 = mx.sqrt(
        mx.sum(logit_difference * logit_difference)
        / mx.sum(source_logits * source_logits)
    )
    logit_max = mx.max(mx.abs(logit_difference))
    source_hidden = final_hidden(source).astype(mx.float32)
    candidate_hidden = final_hidden(candidate).astype(mx.float32)
    hidden_difference = candidate_hidden - source_hidden
    hidden_l2 = mx.sqrt(
        mx.sum(hidden_difference * hidden_difference)
        / mx.sum(source_hidden * source_hidden)
    )
    source_top = mx.argmax(source.logits)
    candidate_top = mx.argmax(candidate.logits)
    route_differences = [
        mx.sum(left != right)
        for left, right in zip(final_routes(source), final_routes(candidate))
    ]
    mx.eval(
        logit_l2,
        logit_max,
        hidden_l2,
        source_top,
        candidate_top,
        *route_differences,
    )
    return (
        int(source_top.item()) == int(candidate_top.item()),
        float(logit_l2.item()),
        float(logit_max.item()),
        float(hidden_l2.item()),
        sum(int(value.item()) for value in route_differences),
    )


def make_candidate(
    source: model.TextModelWeights,
    root: Path,
    *,
    mapped_embedding: bool,
    quantized_embedding: bool,
    quantized_lm_head: bool,
) -> model.TextModelWeights:
    if mapped_embedding:
        embedding = vocab.MLXMappedBF16Matrix(
            require_verified_source(root),
            "model.language_model.embed_tokens.weight",
            (248_320, 2048),
        )
    elif quantized_embedding:
        embedding = vocab.quantize_affine(source.embedding, bits=8, group_size=32)
    else:
        embedding = source.embedding
    lm_head = (
        vocab.quantize_affine(source.lm_head, bits=8, group_size=32)
        if quantized_lm_head
        else source.lm_head
    )
    arrays = []
    for matrix in (embedding, lm_head):
        if isinstance(matrix, vocab.MLXAffineQuantizedMatrix):
            arrays.extend((matrix.packed, matrix.scales, matrix.biases))
    mx.eval(*arrays)
    mx.synchronize()
    candidate = model.TextModelWeights(
        embedding=embedding,
        layers=source.layers,
        final_norm=source.final_norm,
        lm_head=lm_head,
    )
    model.validate_weights(candidate, model.PRODUCTION_CONFIG)
    return candidate


def run_prompt(
    source_weights: model.TextModelWeights,
    candidate_weights: model.TextModelWeights,
    prompt_ids: tuple[int, ...],
    steps: int,
) -> None:
    source_state = model.initial_state(source_weights, model.PRODUCTION_CONFIG)
    candidate_state = model.initial_state(candidate_weights, model.PRODUCTION_CONFIG)
    source, _ = generate.prefill_prompt(
        list(prompt_ids),
        source_state,
        source_weights,
        max_chunk=128,
    )
    candidate, _ = generate.prefill_prompt(
        list(prompt_ids),
        candidate_state,
        candidate_weights,
        max_chunk=128,
    )
    source_session = model.start_decode_session(
        source_weights,
        source.state,
        model.PRODUCTION_CONFIG,
    )
    candidate_session = model.start_decode_session(
        candidate_weights,
        candidate.state,
        model.PRODUCTION_CONFIG,
    )
    top_matches = 0
    first_mismatch = None
    logit_l2_values = []
    logit_max_values = []
    hidden_l2_values = []
    route_differences = 0
    source_times = []
    candidate_times = []
    for step in range(steps):
        matched, logit_l2, logit_max, hidden_l2, route_difference = compare_step(
            source,
            candidate,
        )
        top_matches += int(matched)
        if not matched and first_mismatch is None:
            first_mismatch = step
        logit_l2_values.append(logit_l2)
        logit_max_values.append(logit_max)
        hidden_l2_values.append(hidden_l2)
        route_differences += route_difference
        source_token = int(mx.argmax(source.logits).item())
        if step % 2:
            started = time.perf_counter()
            candidate, candidate_session = model.forward_session_token(
                source_token,
                candidate_session,
            )
            model.evaluate_result(candidate)
            candidate_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            source, source_session = model.forward_session_token(source_token, source_session)
            model.evaluate_result(source)
            source_times.append(time.perf_counter() - started)
        else:
            started = time.perf_counter()
            source, source_session = model.forward_session_token(source_token, source_session)
            model.evaluate_result(source)
            source_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            candidate, candidate_session = model.forward_session_token(
                source_token,
                candidate_session,
            )
            model.evaluate_result(candidate)
            candidate_times.append(time.perf_counter() - started)
    source_mean = statistics.fmean(source_times[2:])
    candidate_mean = statistics.fmean(candidate_times[2:])
    print(
        "vocab-trajectory "
        f"prompt_tokens={len(prompt_ids)} steps={steps} "
        f"top1={top_matches}/{steps} first_mismatch={first_mismatch} "
        f"logit_l2_mean={statistics.fmean(logit_l2_values):.8g} "
        f"logit_l2_max={max(logit_l2_values):.8g} "
        f"logit_max_abs={max(logit_max_values):.8g} "
        f"hidden_l2_mean={statistics.fmean(hidden_l2_values):.8g} "
        f"hidden_l2_max={max(hidden_l2_values):.8g} "
        f"route_id_differences={route_differences}/{steps * 40 * 8} "
        f"source_tok_s={1.0 / source_mean:.3f} "
        f"candidate_tok_s={1.0 / candidate_mean:.3f} "
        f"speedup={source_mean / candidate_mean:.4f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--mapped-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--quantized-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--quantized-lm-head",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(8 <= args.steps <= 512, "trajectory steps must be in [8, 512]")
        require(
            args.mapped_embedding or args.quantized_embedding or args.quantized_lm_head,
            "at least one quantized vocabulary matrix is required",
        )
        require(
            not (args.mapped_embedding and args.quantized_embedding),
            "embedding cannot be both mapped and quantized",
        )
        started = time.perf_counter()
        source = model.load_text_model(args.root)
        candidate = make_candidate(
            source,
            args.root,
            mapped_embedding=args.mapped_embedding,
            quantized_embedding=args.quantized_embedding,
            quantized_lm_head=args.quantized_lm_head,
        )
        tokenizer = load_text_tokenizer(args.root)
        print(
            "vocab-trajectory-ready "
            f"load_s={time.perf_counter() - started:.3f} "
            f"mapped_embedding={str(args.mapped_embedding).lower()} "
            f"quantized_embedding={str(args.quantized_embedding).lower()} "
            f"quantized_lm_head={str(args.quantized_lm_head).lower()} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS
        for prompt in prompts:
            prompt_ids = tokenizer.encode(render_text_prompt(prompt, enable_thinking=True))
            run_prompt(source, candidate, prompt_ids, args.steps)
    except (MoEError, OSError, TokenizerError, ValueError) as exc:
        print(f"vocab trajectory failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

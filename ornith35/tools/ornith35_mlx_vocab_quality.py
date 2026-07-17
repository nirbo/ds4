#!/usr/bin/env python3
"""Quality-gate the Q8/32 LM head on source-model coding trajectories."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError, require
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


DEFAULT_PROMPTS = (
    "Implement an LRU cache in Python with O(1) get and put operations.",
    "Write a correct iterative binary search in Rust and include edge-case tests.",
    "Debug this Python race and explain the fix: two threads increment a shared counter without a lock.",
    "Implement a TypeScript debounce function that preserves this and argument types.",
    "Write SQL that returns the three highest-paid employees per department, including ties.",
    "Implement topological sort with cycle detection in C++20.",
    "Review a function that parses untrusted JSON and list concrete security hardening changes.",
    "Write a Go worker pool with context cancellation and no goroutine leaks.",
    "Implement a persistent immutable trie and explain the asymptotic costs.",
    "Find the bug in a recursive tree serializer that loses null children, then correct it.",
    "Design idempotent retry handling for a payment API and provide pseudocode.",
    "Implement a streaming UTF-8 line reader that handles split multibyte sequences.",
    "Write property-based tests for a union-find implementation.",
    "Optimize a matrix transpose for cache locality without changing its result.",
    "Implement Dijkstra's algorithm and reject negative edge weights.",
    "Explain and fix an ABA bug in a lock-free stack design.",
)


def _final_hidden(
    result: model.TextModelResult | model.TextModelChunkResult,
) -> mx.array:
    return result.hidden[-1] if result.hidden.ndim == 2 else result.hidden


def _array_bytes(value: mx.array) -> int:
    return value.size * value.itemsize


def _head_bytes(head: mx.array | vocab.MLXAffineQuantizedMatrix) -> int:
    if isinstance(head, vocab.MLXAffineQuantizedMatrix):
        return vocab.stored_bytes(head)
    return _array_bytes(head)


def _distribution_metrics(
    source_logits: mx.array,
    candidate_logits: mx.array,
    top_k: int,
) -> tuple[bool, float, float, float, int]:
    source32 = source_logits.astype(mx.float32)
    candidate32 = candidate_logits.astype(mx.float32)
    difference = candidate32 - source32
    relative_l2 = mx.sqrt(mx.sum(difference * difference) / mx.sum(source32 * source32))
    max_abs = mx.max(mx.abs(difference))
    source_indices = mx.argpartition(source32, source32.size - top_k)[-top_k:]
    candidate_indices = mx.argpartition(candidate32, candidate32.size - top_k)[-top_k:]
    source_top = mx.argmax(source32)
    candidate_top = mx.argmax(candidate32)
    source_pair = mx.partition(source32, source32.size - 2)[-2:]
    margin = mx.max(source_pair) - mx.min(source_pair)
    mx.eval(
        relative_l2,
        max_abs,
        source_indices,
        candidate_indices,
        source_top,
        candidate_top,
        margin,
    )
    overlap = len(set(source_indices.tolist()) & set(candidate_indices.tolist()))
    return (
        int(source_top.item()) == int(candidate_top.item()),
        float(relative_l2.item()),
        float(max_abs.item()),
        float(margin.item()),
        overlap,
    )


def _exact_rerank(
    source_head: mx.array,
    hidden: mx.array,
    source_logits: mx.array,
    candidate_logits: mx.array,
    *,
    candidate_count: int,
    source_top_k: int,
) -> tuple[bool, int, int, int]:
    candidate_indices = mx.argpartition(
        candidate_logits,
        candidate_logits.size - candidate_count,
    )[-candidate_count:]
    source_indices = mx.argpartition(
        source_logits,
        source_logits.size - source_top_k,
    )[-source_top_k:]
    exact_scores = vocab.project_bf16_rows_exact(
        mx.take(source_head, candidate_indices, axis=0),
        hidden,
    )
    source_scores = mx.take(source_logits, candidate_indices)
    row_exact = mx.array_equal(exact_scores, source_scores)
    maximum = mx.max(exact_scores)
    corrected = mx.min(
        mx.where(
            exact_scores == maximum,
            candidate_indices,
            mx.array(source_logits.size, dtype=candidate_indices.dtype),
        )
    )
    source_top = mx.argmax(source_logits)
    raw_top = mx.argmax(candidate_logits)
    mx.eval(
        candidate_indices,
        source_indices,
        row_exact,
        corrected,
        source_top,
        raw_top,
    )
    require(bool(row_exact.item()), "candidate-row projection did not reproduce source logits")
    candidate_set = set(candidate_indices.tolist())
    recall = len(candidate_set & set(source_indices.tolist()))
    return (
        int(corrected.item()) == int(source_top.item()),
        recall,
        int(source_top.item()),
        int(raw_top.item()),
    )


def _kl_divergence(source_logits: mx.array, candidate_logits: mx.array) -> float:
    source32 = source_logits.astype(mx.float32)
    candidate32 = candidate_logits.astype(mx.float32)
    source_log = source32 - mx.logsumexp(source32)
    candidate_log = candidate32 - mx.logsumexp(candidate32)
    divergence = mx.sum(mx.exp(source_log) * (source_log - candidate_log))
    mx.eval(divergence)
    return float(divergence.item())


def evaluate_prompt(
    weights: model.TextModelWeights,
    quantized_head: vocab.MLXAffineQuantizedMatrix,
    prompt_ids: tuple[int, ...],
    eos_token_ids: frozenset[int],
    *,
    steps: int,
    top_k: int,
    kl_stride: int,
    rerank_candidates: int,
) -> dict[str, object]:
    state = model.initial_state(weights, model.PRODUCTION_CONFIG)
    result, _ = generate.prefill_prompt(
        list(prompt_ids),
        state,
        weights,
        max_chunk=128,
    )
    session = model.start_decode_session(
        weights,
        result.state,
        model.PRODUCTION_CONFIG,
    )
    matches = 0
    first_mismatch = None
    corrected_matches = 0
    first_corrected_mismatch = None
    rerank_recalls: list[int] = []
    mismatch_records: list[str] = []
    relative_l2: list[float] = []
    max_abs: list[float] = []
    margins: list[float] = []
    overlaps: list[int] = []
    divergences: list[float] = []
    generated: list[int] = []
    started = time.perf_counter()
    for step in range(steps):
        hidden = _final_hidden(result)
        candidate_logits = model.project_lm_head(quantized_head, hidden)
        mx.eval(candidate_logits)
        matched, l2, maximum, margin, overlap = _distribution_metrics(
            result.logits,
            candidate_logits,
            top_k,
        )
        corrected, recall, source_top, raw_top = _exact_rerank(
            weights.lm_head,
            hidden,
            result.logits,
            candidate_logits,
            candidate_count=rerank_candidates,
            source_top_k=top_k,
        )
        matches += int(matched)
        if not matched and first_mismatch is None:
            first_mismatch = step
        corrected_matches += int(corrected)
        if not corrected and first_corrected_mismatch is None:
            first_corrected_mismatch = step
        rerank_recalls.append(recall)
        if not matched and len(mismatch_records) < 16:
            mismatch_records.append(
                f"{step}:{source_top}:{raw_top}:margin={margin:.8g}:recall={recall}"
            )
        relative_l2.append(l2)
        max_abs.append(maximum)
        margins.append(margin)
        overlaps.append(overlap)
        if step % kl_stride == 0:
            divergences.append(_kl_divergence(result.logits, candidate_logits))
        token_id = int(mx.argmax(result.logits).item())
        generated.append(token_id)
        if token_id in eos_token_ids:
            break
        result, session = model.forward_session_token(token_id, session)
        model.evaluate_result(result)
    elapsed = time.perf_counter() - started
    token_bytes = b"".join(token.to_bytes(4, "little") for token in generated)
    return {
        "steps": len(generated),
        "matches": matches,
        "first_mismatch": first_mismatch,
        "corrected_matches": corrected_matches,
        "first_corrected_mismatch": first_corrected_mismatch,
        "rerank_recall_min": min(rerank_recalls),
        "mismatches": ",".join(mismatch_records) if mismatch_records else "none",
        "l2_mean": statistics.fmean(relative_l2),
        "l2_max": max(relative_l2),
        "max_abs": max(max_abs),
        "margin_min": min(margins),
        "top_k_mean": statistics.fmean(overlaps),
        "kl_mean": statistics.fmean(divergences),
        "kl_max": max(divergences),
        "sha256": hashlib.sha256(token_bytes).hexdigest(),
        "elapsed": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--kl-stride", type=int, default=8)
    parser.add_argument("--rerank-candidates", type=int, default=64)
    parser.add_argument(
        "--allow-top1-mismatch",
        action="store_true",
        help="report rather than reject a changed greedy choice",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(32 <= args.steps <= 1024, "quality steps must be in [32, 1024]")
        require(2 <= args.top_k <= 100, "quality top-k must be in [2, 100]")
        require(1 <= args.kl_stride <= args.steps, "invalid KL sampling stride")
        require(
            args.top_k <= args.rerank_candidates <= 256,
            "rerank candidates must cover top-k and be at most 256",
        )
        started = time.perf_counter()
        weights = model.load_text_model(args.root, map_embedding=True)
        require(isinstance(weights.lm_head, mx.array), "source LM head must be BF16")
        quantized_head = vocab.quantize_affine(
            weights.lm_head,
            bits=8,
            group_size=32,
        )
        mx.eval(quantized_head.packed, quantized_head.scales, quantized_head.biases)
        mx.synchronize()
        tokenizer = load_text_tokenizer(args.root)
        source_bytes = _head_bytes(weights.lm_head)
        candidate_bytes = _head_bytes(quantized_head)
        print(
            "vocab-quality-ready "
            f"load_s={time.perf_counter() - started:.3f} "
            f"source_gib={source_bytes / 2**30:.4f} "
            f"candidate_gib={candidate_bytes / 2**30:.4f} "
            f"saved_gib={(source_bytes - candidate_bytes) / 2**30:.4f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS
        total_steps = 0
        total_matches = 0
        total_corrected_matches = 0
        first_failure = None
        first_corrected_failure = None
        l2_means = []
        kl_means = []
        top_k_means = []
        for index, prompt in enumerate(prompts):
            prompt_ids = tokenizer.encode(render_text_prompt(prompt, enable_thinking=True))
            report = evaluate_prompt(
                weights,
                quantized_head,
                prompt_ids,
                tokenizer.eos_token_ids,
                steps=args.steps,
                top_k=args.top_k,
                kl_stride=args.kl_stride,
                rerank_candidates=args.rerank_candidates,
            )
            total_steps += int(report["steps"])
            total_matches += int(report["matches"])
            total_corrected_matches += int(report["corrected_matches"])
            if report["first_mismatch"] is not None and first_failure is None:
                first_failure = (index, report["first_mismatch"])
            if (
                report["first_corrected_mismatch"] is not None
                and first_corrected_failure is None
            ):
                first_corrected_failure = (index, report["first_corrected_mismatch"])
            l2_means.append(float(report["l2_mean"]))
            kl_means.append(float(report["kl_mean"]))
            top_k_means.append(float(report["top_k_mean"]))
            print(
                "vocab-quality-prompt "
                f"index={index} prompt_tokens={len(prompt_ids)} "
                f"steps={report['steps']} top1={report['matches']}/{report['steps']} "
                f"first_mismatch={report['first_mismatch']} "
                f"reranked_top1={report['corrected_matches']}/{report['steps']} "
                f"first_reranked_mismatch={report['first_corrected_mismatch']} "
                f"top{args.top_k}_recall_min={report['rerank_recall_min']}/{args.top_k} "
                f"mismatches={report['mismatches']} "
                f"logit_l2_mean={report['l2_mean']:.8g} "
                f"logit_l2_max={report['l2_max']:.8g} "
                f"max_abs={report['max_abs']:.8g} "
                f"source_margin_min={report['margin_min']:.8g} "
                f"top{args.top_k}_overlap_mean={report['top_k_mean']:.4f} "
                f"kl_mean={report['kl_mean']:.8g} kl_max={report['kl_max']:.8g} "
                f"token_sha256={report['sha256']} elapsed_s={report['elapsed']:.3f}",
                flush=True,
            )
        print(
            "vocab-quality-done "
            f"prompts={len(prompts)} steps={total_steps} "
            f"top1={total_matches}/{total_steps} first_failure={first_failure} "
            f"reranked_top1={total_corrected_matches}/{total_steps} "
            f"first_reranked_failure={first_corrected_failure} "
            f"prompt_l2_mean={statistics.fmean(l2_means):.8g} "
            f"prompt_kl_mean={statistics.fmean(kl_means):.8g} "
            f"top{args.top_k}_overlap_mean={statistics.fmean(top_k_means):.4f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
        if first_corrected_failure is not None and not args.allow_top1_mismatch:
            raise MoEError(
                f"Q8/32 exact candidate rerank changed a greedy choice at "
                f"{first_corrected_failure}"
            )
    except (MoEError, OSError, TokenizerError, ValueError) as exc:
        print(f"vocab quality gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

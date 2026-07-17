#!/usr/bin/env python3
"""Quality, size, and latency sweep for a quantized Ornith-35 LM head."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
from ornith35_moe_reference import MoEError, require
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


def parse_candidates(text: str) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    try:
        for item in text.split(","):
            bits_text, group_text = item.strip().split(":", 1)
            candidate = (int(bits_text), int(group_text))
            require(candidate[0] in (2, 3, 4, 5, 6, 8), "unsupported affine bit width")
            require(candidate[1] in (32, 64, 128), "unsupported affine group size")
            if candidate not in values:
                values.append(candidate)
    except (ValueError, AttributeError) as exc:
        raise MoEError("candidates must use comma-separated bits:group values") from exc
    require(bool(values), "at least one quantization candidate is required")
    return tuple(values)


def array_bytes(value: mx.array) -> int:
    return value.size * value.itemsize


def collect_hidden(
    weights: model.TextModelWeights,
    root: Path,
    prompt: str,
    samples: int,
) -> mx.array:
    tokenizer = load_text_tokenizer(root)
    prompt_ids = list(tokenizer.encode(render_text_prompt(prompt, enable_thinking=True)))
    state = model.initial_state(weights, model.PRODUCTION_CONFIG)
    result, _ = generate.prefill_prompt(prompt_ids, state, weights, max_chunk=128)
    session = model.start_decode_session(weights, result.state, model.PRODUCTION_CONFIG)
    hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
    logits = result.logits
    rows: list[mx.array] = []
    for _ in range(samples):
        rows.append(hidden)
        token = int(mx.argmax(logits).item())
        result, session = model.forward_session_token(token, session)
        model.evaluate_result(result)
        hidden = result.hidden
        logits = result.logits
    batch = mx.stack(rows)
    mx.eval(batch)
    mx.synchronize()
    return batch


def measure(operation, warmup: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        mx.eval(operation())
        mx.synchronize()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter()
        mx.eval(operation())
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return samples


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.mean(retained)


def run_candidate(
    head: mx.array,
    hidden: mx.array,
    reference: mx.array,
    bits: int,
    group_size: int,
    warmup: int,
    rounds: int,
) -> None:
    started = time.perf_counter()
    packed, scales, biases = mx.quantize(
        head,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    mx.eval(packed, scales, biases)
    mx.synchronize()
    quantize_seconds = time.perf_counter() - started
    quantized = mx.quantized_matmul(
        hidden,
        packed,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    difference = quantized.astype(mx.float32) - reference.astype(mx.float32)
    relative_l2 = mx.sqrt(mx.sum(mx.square(difference)) / mx.sum(mx.square(reference)))
    max_abs = mx.max(mx.abs(difference))
    mean_abs = mx.mean(mx.abs(difference))
    reference_top = mx.argmax(reference, axis=1)
    quantized_top = mx.argmax(quantized, axis=1)
    top_matches = mx.sum(reference_top == quantized_top)
    mx.eval(
        quantized,
        relative_l2,
        max_abs,
        mean_abs,
        reference_top,
        quantized_top,
        top_matches,
    )
    probe = hidden[hidden.shape[0] // 2]
    source_times: list[float] = []
    quantized_times: list[float] = []
    for index in range(warmup + rounds):
        quantized_first = index % 2 == 1
        if quantized_first:
            quantized_sample = measure(
                lambda: mx.quantized_matmul(
                    probe,
                    packed,
                    scales,
                    biases,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                ),
                0,
                1,
            )[0]
            source_sample = measure(lambda: mx.matmul(head, probe), 0, 1)[0]
        else:
            source_sample = measure(lambda: mx.matmul(head, probe), 0, 1)[0]
            quantized_sample = measure(
                lambda: mx.quantized_matmul(
                    probe,
                    packed,
                    scales,
                    biases,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                ),
                0,
                1,
            )[0]
        if index >= warmup:
            source_times.append(source_sample)
            quantized_times.append(quantized_sample)
    source_mean = trimmed_mean(source_times)
    quantized_mean = trimmed_mean(quantized_times)
    quantized_bytes = sum(array_bytes(value) for value in (packed, scales, biases))
    source_bytes = array_bytes(head)
    mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(reference_top.tolist(), quantized_top.tolist())
        )
        if left != right
    ]
    print(
        "vocab-quant-candidate "
        f"bits={bits} group={group_size} samples={hidden.shape[0]} "
        f"source_gib={source_bytes / 2**30:.4f} "
        f"quantized_gib={quantized_bytes / 2**30:.4f} "
        f"saved_gib={(source_bytes - quantized_bytes) / 2**30:.4f} "
        f"quantize_s={quantize_seconds:.3f} "
        f"relative_l2={float(relative_l2.item()):.8g} "
        f"mean_abs={float(mean_abs.item()):.8g} "
        f"max_abs={float(max_abs.item()):.8g} "
        f"top1={int(top_matches.item())}/{hidden.shape[0]} "
        f"mismatch_rows={mismatches} "
        f"source_ms={source_mean * 1000:.3f} "
        f"quantized_ms={quantized_mean * 1000:.3f} "
        f"speedup={source_mean / quantized_mean:.4f}",
        flush=True,
    )
    del packed, scales, biases, quantized, difference
    gc.collect()
    mx.clear_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompt",
        default="Implement an LRU cache in Python and explain its complexity.",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--candidates", default="8:64,6:64,5:64,4:64")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(8 <= args.samples <= 512, "samples must be in [8, 512]")
        require(args.warmup >= 1, "warmup must be positive")
        require(args.rounds >= 4, "rounds must be at least four")
        candidates = parse_candidates(args.candidates)
        started = time.perf_counter()
        weights = model.load_text_model(args.root)
        print(
            "vocab-quant-model-ready "
            f"load_s={time.perf_counter() - started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        hidden = collect_hidden(weights, args.root, args.prompt, args.samples)
        reference = mx.matmul(hidden, mx.transpose(weights.lm_head))
        mx.eval(reference)
        mx.synchronize()
        print(
            "vocab-quant-hidden-ready "
            f"samples={hidden.shape[0]} active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        for bits, group_size in candidates:
            run_candidate(
                weights.lm_head,
                hidden,
                reference,
                bits,
                group_size,
                args.warmup,
                args.rounds,
            )
    except (MoEError, OSError, TokenizerError, ValueError) as exc:
        print(f"vocab quantization benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

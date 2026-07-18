#!/usr/bin/env python3
"""Real-checkpoint quality and timing gate for direct packed K4-MSE decode."""

from __future__ import annotations

import argparse
from pathlib import Path
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
)
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT
from ornith35_tokenizer import load_text_tokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = REPOSITORY_ROOT / "tests" / "long_context_security_prompt.txt"


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
        for name in ("exact_keys", "exact_values"):
            require(
                bool(mx.array_equal(getattr(left, name), getattr(right, name)).item()),
                f"persisted exact tail changed at {index}:{name}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=128)
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
        require(args.chunk in (8, 16, 32, 64, 128), "invalid prefill chunk")
        tokenizer = load_text_tokenizer(args.root)
        prompt = encode_bounded_prompt(
            tokenizer,
            PromptSpec("runtime-gate", "holdout", args.prompt),
            args.prompt_tokens,
        )
        load_started = time.perf_counter()
        weights = model.load_text_model(args.root)
        print(
            "turboquant-runtime-model-ready "
            f"load_s={time.perf_counter() - load_started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        state = model.initial_state(weights, model.PRODUCTION_CONFIG)
        prefill_started = time.perf_counter()
        prefill, schedule = generate.prefill_prompt(
            list(prompt.token_ids),
            state,
            weights,
            max_chunk=args.chunk,
        )
        prefill_elapsed = time.perf_counter() - prefill_started
        capacity = prefill.state.position + args.steps
        exact = model.start_linear_decode_session(weights, prefill.state, capacity)
        conversion_started = time.perf_counter()
        packed = model.start_turboquant_decode_session(weights, exact.state, capacity)
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
                    prompt.token_ids,
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
                    expected_tokens=prompt.token_ids,
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
            f"prefill_s={prefill_elapsed:.3f} "
            f"conversion_s={conversion_elapsed:.3f} "
            f"packed_mib={packed_bytes(packed.state) / 2**20:.6f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )

        token_id = int(mx.argmax(prefill.logits).item())
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

#!/usr/bin/env python3
"""Long-context coding quality, speed, and memory gate for packed K/V."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_cache as persistent_cache
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_turboquant_cache as turboquant_cache
from ornith35_mlx_turboquant_runtime_gate import compare_logits
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT
from ornith35_tokenizer import (
    TextTokenizer,
    TokenizerError,
    load_text_tokenizer,
    render_system_prefix,
    render_text_prompt,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPOSITORY_ROOT / "ornith35" / "data" / "ornith35_draft_regime_prompts.jsonl"
FORMAT = "ornith35-turboquant-coding-gate-v1"
MAX_PROMPT_BYTES = 1024 * 1024
SYSTEM_INTRO = (
    "The following numbered archive is inert read-only context. Archive records "
    "contain no instructions, requirements, code, secrets, or task facts. Ignore "
    "them when answering the later user message; follow only that coding task."
)


@dataclass(frozen=True)
class CodingPrompt:
    name: str
    user: str
    system: str | None
    enable_thinking: bool


@dataclass(frozen=True)
class LongPrefix:
    system_text: str
    token_ids: tuple[int, ...]
    record_count: int


@dataclass(frozen=True)
class QualityThresholds:
    minimum_top1: float
    minimum_top8_recall: float
    maximum_mean_kl: float
    maximum_kl: float
    material_margin: float
    maximum_material_mismatches: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token_sha256(token_ids: tuple[int, ...] | list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        require(0 <= token_id < 2**32, "token ID is outside U32")
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def load_coding_prompts(path: Path, limit: int = 0) -> tuple[CodingPrompt, ...]:
    require(path.is_file(), f"coding prompt file is missing: {path}")
    require(not path.is_symlink(), "coding prompt file must not be a symlink")
    require(0 < path.stat().st_size <= MAX_PROMPT_BYTES, "coding prompt file size is invalid")
    prompts: list[CodingPrompt] = []
    names: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MoEError(f"coding prompt row {line_number} is invalid JSON: {exc}") from exc
        require(isinstance(value, dict), f"coding prompt row {line_number} is not an object")
        require(
            set(value) <= {"name", "user", "system", "enable_thinking"},
            f"coding prompt row {line_number} has unknown fields",
        )
        name = value.get("name")
        user = value.get("user")
        system = value.get("system")
        thinking = value.get("enable_thinking", True)
        require(isinstance(name, str) and name.strip(), f"coding prompt row {line_number} has no name")
        require(isinstance(user, str) and user.strip(), f"coding prompt row {line_number} has no user text")
        require(
            system is None or (isinstance(system, str) and system.strip()),
            f"coding prompt row {line_number} has an invalid system message",
        )
        require(isinstance(thinking, bool), f"coding prompt row {line_number} has an invalid thinking flag")
        name = name.strip()
        require(name not in names, f"duplicate coding prompt name: {name}")
        names.add(name)
        prompts.append(
            CodingPrompt(
                name=name,
                user=user.strip(),
                system=system.strip() if system is not None else None,
                enable_thinking=thinking,
            )
        )
        if limit and len(prompts) == limit:
            break
    require(prompts, "coding prompt file contains no prompts")
    return tuple(prompts)


def _archive_record(index: int) -> str:
    return (
        f"Archive record {index:06d}: status nominal; maintenance window closed; "
        f"inventory batch {index % 997:03d}; reference value {(index * 37) % 1009:04d}."
    )


def _archive_text(record_count: int) -> str:
    records = "\n".join(_archive_record(index) for index in range(record_count))
    return SYSTEM_INTRO if not records else f"{SYSTEM_INTRO}\n\n{records}"


def build_long_system_prefix(tokenizer: TextTokenizer, target_tokens: int) -> LongPrefix:
    require(target_tokens >= 128, "long-prefix token target is too small")

    def encode(record_count: int) -> tuple[str, tuple[int, ...]]:
        text = _archive_text(record_count)
        return text, tokenizer.encode(render_system_prefix(text))

    base_text, base_ids = encode(0)
    require(len(base_ids) <= target_tokens, "long-prefix target cannot hold the archive header")
    low = 0
    high = 1
    best_text = base_text
    best_ids = base_ids
    while True:
        text, token_ids = encode(high)
        if len(token_ids) > target_tokens:
            break
        low = high
        high *= 2
        best_text = text
        best_ids = token_ids
        require(high <= 1_048_576, "long-prefix record search exceeded its bound")
    while low + 1 < high:
        middle = (low + high) // 2
        text, token_ids = encode(middle)
        if len(token_ids) <= target_tokens:
            low = middle
            best_text = text
            best_ids = token_ids
        else:
            high = middle
    return LongPrefix(best_text, best_ids, low)


def prompt_user_text(prompt: CodingPrompt) -> str:
    if prompt.system is None:
        return prompt.user
    return f"Response requirements:\n{prompt.system}\n\nCoding task:\n{prompt.user}"


def prompt_tail_ids(
    tokenizer: TextTokenizer,
    prefix: LongPrefix,
    prompt: CodingPrompt,
) -> tuple[int, ...]:
    user = prompt_user_text(prompt)
    tail = tokenizer.encode(
        render_text_prompt(user, enable_thinking=prompt.enable_thinking)
    )
    complete = tokenizer.encode(
        render_text_prompt(
            user,
            system=prefix.system_text,
            enable_thinking=prompt.enable_thinking,
        )
    )
    require(
        complete == prefix.token_ids + tail,
        f"tokenizer boundary changed across the shared prefix for {prompt.name}",
    )
    return tail


def parse_sample_seeds(values: list[int]) -> tuple[int, ...]:
    require(all(0 <= seed < 2**32 for seed in values), "sample seed is outside U32")
    require(len(set(values)) == len(values), "sample seeds must be unique")
    return tuple(values)


def _final_hidden(result: model.TextModelResult | model.TextModelChunkResult) -> mx.array:
    return result.hidden[-1] if result.hidden.ndim == 2 else result.hidden


def select_source_token(
    result: model.TextModelResult | model.TextModelChunkResult,
    mode: str,
    rng: random.Random | None,
    weights: model.TextModelWeights,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if mode == "greedy":
        return int(mx.argmax(result.logits).item())
    require(mode.startswith("seed-") and rng is not None, "invalid sampled trajectory")
    return generate.choose_next_token(
        result.logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        rng=rng,
        hidden=_final_hidden(result),
        lm_head=weights.lm_head,
    )


def evaluate_trajectory(
    initial: model.TextModelResult | model.TextModelChunkResult,
    exact: model.TextLinearDecodeSession,
    packed: model.TextTurboQuantDecodeSession,
    weights: model.TextModelWeights,
    eos_token_ids: frozenset[int],
    *,
    mode: str,
    seed: int | None,
    steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    material_margin: float,
) -> dict[str, Any]:
    exact_result = initial
    rng = random.Random(seed) if seed is not None else None
    reports: list[dict[str, float | int | bool]] = []
    exact_times: list[float] = []
    packed_times: list[float] = []
    generated: list[int] = []
    mismatches: list[dict[str, float | int]] = []
    material_mismatches = 0
    for step in range(steps):
        token_id = select_source_token(
            exact_result,
            mode,
            rng,
            weights,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(token_id)
        if token_id in eos_token_ids:
            break
        operations = (
            ("exact", lambda: model.forward_linear_session_token(token_id, exact)),
            ("packed", lambda: model.forward_turboquant_session_token(token_id, packed)),
        )
        if step % 2:
            operations = tuple(reversed(operations))
        samples: dict[str, model.TextModelResult] = {}
        for name, operation in operations:
            started = time.perf_counter()
            samples[name] = operation()
            elapsed = time.perf_counter() - started
            (exact_times if name == "exact" else packed_times).append(elapsed)
        report = compare_logits(samples["exact"].logits, samples["packed"].logits)
        reports.append(report)
        if not bool(report["top1"]):
            mismatch = {
                "step": step + 1,
                "source_top": int(report["source_top"]),
                "candidate_top": int(report["candidate_top"]),
                "source_margin": float(report["source_margin"]),
                "kl": float(report["kl"]),
            }
            if len(mismatches) < 32:
                mismatches.append(mismatch)
            if float(report["source_margin"]) >= material_margin:
                material_mismatches += 1
        exact_result = samples["exact"]
    require(reports, f"trajectory {mode} ended before producing a comparable transition")
    compared = len(reports)
    exact_elapsed = sum(exact_times)
    packed_elapsed = sum(packed_times)
    return {
        "mode": mode,
        "seed": seed,
        "steps": compared,
        "top1": sum(int(bool(report["top1"])) for report in reports),
        "top8_recall_mean": statistics.fmean(float(report["top8_recall"]) for report in reports),
        "kl_mean": statistics.fmean(float(report["kl"]) for report in reports),
        "kl_max": max(float(report["kl"]) for report in reports),
        "max_abs": max(float(report["max_abs"]) for report in reports),
        "material_mismatches": material_mismatches,
        "mismatches": mismatches,
        "exact_s": exact_elapsed,
        "packed_s": packed_elapsed,
        "speedup": exact_elapsed / packed_elapsed,
        "source_token_sha256": token_sha256(generated),
    }


def summarize_trajectories(reports: list[dict[str, Any]]) -> dict[str, Any]:
    require(reports, "no coding trajectories were evaluated")
    steps = sum(int(report["steps"]) for report in reports)
    exact_s = sum(float(report["exact_s"]) for report in reports)
    packed_s = sum(float(report["packed_s"]) for report in reports)
    return {
        "trajectories": len(reports),
        "steps": steps,
        "top1": sum(int(report["top1"]) for report in reports),
        "top8_recall_mean": sum(
            float(report["top8_recall_mean"]) * int(report["steps"])
            for report in reports
        ) / steps,
        "kl_mean": sum(
            float(report["kl_mean"]) * int(report["steps"])
            for report in reports
        ) / steps,
        "kl_max": max(float(report["kl_max"]) for report in reports),
        "max_abs": max(float(report["max_abs"]) for report in reports),
        "material_mismatches": sum(int(report["material_mismatches"]) for report in reports),
        "exact_s": exact_s,
        "packed_s": packed_s,
        "exact_tokens_s": steps / exact_s,
        "packed_tokens_s": steps / packed_s,
        "speedup": exact_s / packed_s,
    }


def quality_failures(
    summary: dict[str, Any],
    thresholds: QualityThresholds,
) -> tuple[str, ...]:
    failures: list[str] = []
    top1_rate = int(summary["top1"]) / int(summary["steps"])
    if top1_rate < thresholds.minimum_top1:
        failures.append(f"top1 {top1_rate:.6f} < {thresholds.minimum_top1:.6f}")
    if float(summary["top8_recall_mean"]) < thresholds.minimum_top8_recall:
        failures.append(
            f"top8 recall {summary['top8_recall_mean']:.6f} < "
            f"{thresholds.minimum_top8_recall:.6f}"
        )
    if float(summary["kl_mean"]) > thresholds.maximum_mean_kl:
        failures.append(
            f"mean KL {summary['kl_mean']:.9g} > {thresholds.maximum_mean_kl:.9g}"
        )
    if float(summary["kl_max"]) > thresholds.maximum_kl:
        failures.append(f"max KL {summary['kl_max']:.9g} > {thresholds.maximum_kl:.9g}")
    if int(summary["material_mismatches"]) > thresholds.maximum_material_mismatches:
        failures.append(
            f"material mismatches {summary['material_mismatches']} > "
            f"{thresholds.maximum_material_mismatches}"
        )
    return tuple(failures)


def exact_kv_bytes(state: model.TextModelState) -> int:
    total = 0
    for layer_state in state.layers:
        if isinstance(layer_state, attention.MLXLinearAttentionState):
            total += layer_state.keys.size * layer_state.keys.itemsize
            total += layer_state.values.size * layer_state.values.itemsize
    return total


def packed_kv_bytes(state: model.TextModelState) -> int:
    return sum(
        turboquant_cache.stored_bytes(layer_state)
        for layer_state in state.layers
        if isinstance(layer_state, attention.MLXTurboQuantAttentionState)
    )


def prefill_shared_prefix(
    prefix_ids: tuple[int, ...],
    session: model.TextLinearDecodeSession,
    weights: model.TextModelWeights,
    *,
    chunk: int,
    progress_tokens: int,
) -> tuple[model.TextModelResult | model.TextModelChunkResult, float]:
    result: model.TextModelResult | model.TextModelChunkResult | None = None
    started = time.perf_counter()
    for offset in range(0, len(prefix_ids), progress_tokens):
        segment = list(prefix_ids[offset : offset + progress_tokens])
        segment_started = time.perf_counter()
        result, _ = generate.prefill_prompt(
            segment,
            session.state,
            weights,
            max_chunk=chunk,
            linear_session=session,
        )
        elapsed = time.perf_counter() - started
        completed = offset + len(segment)
        print(
            "turboquant-coding-prefix-progress "
            f"tokens={completed}/{len(prefix_ids)} "
            f"segment_s={time.perf_counter() - segment_started:.3f} "
            f"tokens_s={completed / elapsed:.3f}",
            flush=True,
        )
    require(result is not None, "shared prefix prefill produced no result")
    return result, time.perf_counter() - started


def atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--prefix-tokens", type=int, default=65_536)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, action="append", default=[])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--progress-tokens", type=int, default=4096)
    parser.add_argument("--minimum-top1", type=float, default=0.99)
    parser.add_argument("--minimum-top8-recall", type=float, default=0.95)
    parser.add_argument("--maximum-mean-kl", type=float, default=0.01)
    parser.add_argument("--maximum-kl", type=float, default=0.1)
    parser.add_argument("--material-margin", type=float, default=0.5)
    parser.add_argument("--maximum-material-mismatches", type=int, default=0)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(0 <= args.prompt_limit <= 128, "prompt limit must be in [0, 128]")
        require(4 <= args.steps <= 256, "trajectory steps must be in [4, 256]")
        require(128 <= args.prefix_tokens < context.NATIVE_CONTEXT_TOKENS, "invalid native prefix size")
        require(args.chunk in (8, 16, 32, 64, 128), "invalid prefill chunk")
        require(
            args.progress_tokens >= args.chunk and args.progress_tokens % args.chunk == 0,
            "progress interval must be a positive multiple of the chunk",
        )
        require(args.temperature > 0.0, "sampling temperature must be positive")
        require(0 < args.top_k <= model.PRODUCTION_CONFIG.vocab_size, "invalid top-k")
        require(0.0 < args.top_p <= 1.0, "invalid top-p")
        thresholds = QualityThresholds(
            minimum_top1=args.minimum_top1,
            minimum_top8_recall=args.minimum_top8_recall,
            maximum_mean_kl=args.maximum_mean_kl,
            maximum_kl=args.maximum_kl,
            material_margin=args.material_margin,
            maximum_material_mismatches=args.maximum_material_mismatches,
        )
        require(0.0 <= thresholds.minimum_top1 <= 1.0, "invalid top-1 threshold")
        require(0.0 <= thresholds.minimum_top8_recall <= 1.0, "invalid top-8 threshold")
        require(thresholds.maximum_mean_kl >= 0.0, "invalid mean-KL threshold")
        require(thresholds.maximum_kl >= thresholds.maximum_mean_kl, "invalid max-KL threshold")
        require(thresholds.material_margin >= 0.0, "invalid material margin")
        require(thresholds.maximum_material_mismatches >= 0, "invalid material mismatch limit")
        seeds = parse_sample_seeds(args.sample_seed)
        prompts = load_coding_prompts(args.prompts, args.prompt_limit)
        tokenizer = load_text_tokenizer(args.root)
        prefix = build_long_system_prefix(tokenizer, args.prefix_tokens)
        tails = tuple(prompt_tail_ids(tokenizer, prefix, prompt) for prompt in prompts)
        maximum_tail = max(len(tail) for tail in tails)
        capacity = len(prefix.token_ids) + maximum_tail + args.steps
        context.validate_range(context.NATIVE_PROFILE_ID, 0, capacity)
        identity = persistent_cache.production_identity(
            args.root,
            REPOSITORY_ROOT,
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            chat_template_sha256=tokenizer.template_sha256,
            mapped_embedding=False,
            quantized_lm_head=False,
            turboquant_kv=True,
        )
        print(
            "turboquant-coding-plan "
            f"prompts={len(prompts)} trajectories_per_prompt={1 + len(seeds)} "
            f"steps={args.steps} target_prefix_tokens={args.prefix_tokens} "
            f"actual_prefix_tokens={len(prefix.token_ids)} records={prefix.record_count} "
            f"capacity={capacity} prefix_sha256={token_sha256(prefix.token_ids)} "
            f"runtime_sha256={identity.runtime_sha256}",
            flush=True,
        )
        load_started = time.perf_counter()
        weights = model.load_text_model(args.root)
        print(
            "turboquant-coding-model-ready "
            f"load_s={time.perf_counter() - load_started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )
        state = model.initial_state(weights, model.PRODUCTION_CONFIG)
        exact = model.start_linear_decode_session(weights, state, capacity)
        prefix_result, prefix_prefill_s = prefill_shared_prefix(
            prefix.token_ids,
            exact,
            weights,
            chunk=args.chunk,
            progress_tokens=args.progress_tokens,
        )
        base_checkpoint = model.checkpoint_linear_session_state(exact)
        trajectory_reports: list[dict[str, Any]] = []
        production_packed: model.TextTurboQuantDecodeSession | None = None
        modes = (("greedy", None),) + tuple((f"seed-{seed}", seed) for seed in seeds)
        for prompt_index, (prompt, tail) in enumerate(zip(prompts, tails)):
            base_state = model.restore_linear_session_checkpoint(exact, base_checkpoint)
            prompt_started = time.perf_counter()
            prompt_result, _ = generate.prefill_prompt(
                list(tail),
                base_state,
                weights,
                max_chunk=args.chunk,
                linear_session=exact,
            )
            prompt_prefill_s = time.perf_counter() - prompt_started
            prompt_checkpoint = model.checkpoint_linear_session_state(exact)
            print(
                "turboquant-coding-prompt-ready "
                f"index={prompt_index + 1}/{len(prompts)} name={prompt.name} "
                f"tail_tokens={len(tail)} position={exact.state.position} "
                f"prefill_s={prompt_prefill_s:.3f}",
                flush=True,
            )
            for mode_index, (mode, seed) in enumerate(modes):
                if mode_index:
                    model.restore_linear_session_checkpoint(exact, prompt_checkpoint)
                conversion_started = time.perf_counter()
                packed = model.start_turboquant_decode_session(weights, exact.state, capacity)
                conversion_s = time.perf_counter() - conversion_started
                report = evaluate_trajectory(
                    prompt_result,
                    exact,
                    packed,
                    weights,
                    tokenizer.eos_token_ids,
                    mode=mode,
                    seed=seed,
                    steps=args.steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    material_margin=thresholds.material_margin,
                )
                report.update(
                    {
                        "prompt_index": prompt_index,
                        "prompt_name": prompt.name,
                        "prompt_tail_tokens": len(tail),
                        "prompt_tail_sha256": token_sha256(tail),
                        "conversion_s": conversion_s,
                    }
                )
                trajectory_reports.append(report)
                print(
                    "turboquant-coding-trajectory "
                    f"prompt={prompt.name} mode={mode} steps={report['steps']} "
                    f"top1={report['top1']}/{report['steps']} "
                    f"top8_recall={report['top8_recall_mean']:.6f} "
                    f"mean_kl={report['kl_mean']:.9g} max_kl={report['kl_max']:.9g} "
                    f"material_mismatches={report['material_mismatches']} "
                    f"exact_tokens_s={report['steps'] / report['exact_s']:.3f} "
                    f"packed_tokens_s={report['steps'] / report['packed_s']:.3f} "
                    f"speedup={report['speedup']:.4f} conversion_s={conversion_s:.3f}",
                    flush=True,
                )
                final_trajectory = prompt_index + 1 == len(prompts) and mode_index + 1 == len(modes)
                if final_trajectory:
                    production_packed = packed
                else:
                    del packed
                    gc.collect()
                    mx.clear_cache()
        require(production_packed is not None, "coding gate retained no packed production state")
        summary = summarize_trajectories(trajectory_reports)
        exact_bytes = exact_kv_bytes(exact.state)
        packed_bytes = packed_kv_bytes(production_packed.state)
        paired_active = mx.get_active_memory()
        workload_peak = mx.get_peak_memory()
        del prompt_result
        del prompt_checkpoint
        del base_state
        del base_checkpoint
        del prefix_result
        del exact
        del state
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        packed_only_active = mx.get_active_memory()
        released = paired_active - packed_only_active
        release_ok = released >= int(exact_bytes * 0.90)
        failures = list(quality_failures(summary, thresholds))
        if not release_ok:
            failures.append(
                f"BF16 release {released / 2**20:.3f} MiB is below 90% of "
                f"{exact_bytes / 2**20:.3f} MiB"
            )
        status = "pass" if not failures else "fail"
        print(
            "turboquant-coding-result "
            f"status={status} prompts={len(prompts)} trajectories={summary['trajectories']} "
            f"steps={summary['steps']} top1={summary['top1']}/{summary['steps']} "
            f"top1_rate={summary['top1'] / summary['steps']:.6f} "
            f"top8_recall={summary['top8_recall_mean']:.6f} "
            f"mean_kl={summary['kl_mean']:.9g} max_kl={summary['kl_max']:.9g} "
            f"material_mismatches={summary['material_mismatches']} "
            f"exact_tokens_s={summary['exact_tokens_s']:.3f} "
            f"packed_tokens_s={summary['packed_tokens_s']:.3f} "
            f"speedup={summary['speedup']:.4f} prefix_prefill_s={prefix_prefill_s:.3f} "
            f"exact_kv_mib={exact_bytes / 2**20:.3f} packed_kv_mib={packed_bytes / 2**20:.3f} "
            f"paired_active_gib={paired_active / 2**30:.3f} "
            f"packed_only_active_gib={packed_only_active / 2**30:.3f} "
            f"released_gib={released / 2**30:.3f} peak_gib={workload_peak / 2**30:.3f} "
            f"failures={';'.join(failures) if failures else 'none'}",
            flush=True,
        )
        report_value = {
            "format": FORMAT,
            "status": status,
            "identity": asdict(identity),
            "gate_sha256": sha256_file(Path(__file__)),
            "prompt_file": str(args.prompts.resolve()),
            "prompt_file_sha256": sha256_file(args.prompts),
            "prefix": {
                "target_tokens": args.prefix_tokens,
                "tokens": len(prefix.token_ids),
                "records": prefix.record_count,
                "token_sha256": token_sha256(prefix.token_ids),
                "text_sha256": sha256_bytes(prefix.system_text.encode("utf-8")),
            },
            "configuration": {
                "steps": args.steps,
                "sample_seeds": list(seeds),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "chunk": args.chunk,
                "thresholds": asdict(thresholds),
            },
            "summary": summary,
            "memory": {
                "exact_kv_bytes": exact_bytes,
                "packed_kv_bytes": packed_bytes,
                "paired_active_bytes": paired_active,
                "packed_only_active_bytes": packed_only_active,
                "released_bytes": released,
                "release_ok": release_ok,
                "peak_bytes": workload_peak,
            },
            "prefix_prefill_s": prefix_prefill_s,
            "failures": failures,
            "trajectories": trajectory_reports,
        }
        if args.report is not None:
            report_sha256 = atomic_json(args.report, report_value)
            print(
                f"turboquant-coding-report path={args.report} sha256={report_sha256}",
                flush=True,
            )
        if failures:
            raise MoEError("TurboQuant coding gate did not meet acceptance thresholds")
    except (MoEError, OSError, TokenizerError, ValueError) as exc:
        print(f"TurboQuant coding gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

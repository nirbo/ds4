#!/usr/bin/env python3
"""Gate Ornith-35 MTP and DSpark on prompt-disjoint target-authoritative paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
from typing import Any, Sequence

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_dspark as dspark
import ornith35_mlx_dspark_runtime as dspark_runtime
import ornith35_mlx_gdn as gdn
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_mtp as mtp
import ornith35_mlx_mtp_runtime as mtp_runtime
import ornith35_mlx_sampling as sampling
import ornith35_mlx_speculative as speculative
from ornith35_dspark_reference import PRODUCTION_CONFIG as DSPARK_CONFIG
from ornith35_moe_reference import MoEError, require
import ornith35_nvfp4 as nvfp4
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TextTokenizer,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


FORMAT = "ornith35-draft-regime-gate-v1"
ENGINES = ("mtp-sampled", "mtp-greedy", "dspark-greedy")
PRODUCTION_MODEL_ID = "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
PRODUCTION_MODEL_REVISION = "85ffd2d0629ae5fa4f860dda356ec33161806c9b"
DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "data" / "ornith35_draft_regime_prompts.jsonl"
DEFAULT_ADAPTATION = Path("experiments/mtp-distill-coding-v1/adapter-r32-e8-s29-v2")
DEFAULT_CAPTURE_STATE = Path("experiments/mtp-distill-coding-v1/capture/state.json")
NATIVE_CONTEXT_TOKENS = 262_144
_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True)
class GatePrompt:
    name: str
    user: str
    system: str | None
    enable_thinking: bool

    def canonical(self) -> dict[str, object]:
        return {
            "enable_thinking": self.enable_thinking,
            "name": self.name,
            "system": self.system,
            "user": self.user,
        }


@dataclass(frozen=True)
class EncodedPrompt:
    prompt: GatePrompt
    rendered_sha256: str
    token_ids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON value is not an object: {path}")
    return value


def load_prompts(path: Path) -> list[GatePrompt]:
    """Load the exact named JSONL gate schema; aliases and extra fields are invalid."""
    prompts: list[GatePrompt] = []
    names: set[str] = set()
    required = {"name", "user", "enable_thinking"}
    allowed = required | {"system"}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read gate prompts {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid gate prompt JSON at line {line_number}: {exc}") from exc
        require(isinstance(value, dict), f"gate prompt line {line_number} is not an object")
        require(required <= set(value), f"gate prompt line {line_number} is missing required fields")
        require(set(value) <= allowed, f"gate prompt line {line_number} has unknown fields")
        name = value["name"]
        user = value["user"]
        system = value.get("system")
        thinking = value["enable_thinking"]
        require(
            isinstance(name, str) and _NAME_PATTERN.fullmatch(name) is not None,
            f"gate prompt line {line_number} has an invalid name",
        )
        require(name not in names, f"duplicate gate prompt name: {name}")
        require(isinstance(user, str) and user.strip(), f"gate prompt {name} has empty user text")
        require(
            system is None or (isinstance(system, str) and system.strip()),
            f"gate prompt {name} has an invalid system message",
        )
        require(isinstance(thinking, bool), f"gate prompt {name} has an invalid thinking flag")
        names.add(name)
        prompts.append(GatePrompt(name, user, system, thinking))
    require(prompts, "gate prompt corpus is empty")
    return prompts


def render_prompt(prompt: GatePrompt) -> str:
    return render_text_prompt(
        prompt.user,
        system=prompt.system,
        enable_thinking=prompt.enable_thinking,
    )


def encode_prompts(
    prompts: Sequence[GatePrompt],
    tokenizer: TextTokenizer,
    *,
    max_prompt_tokens: int,
) -> list[EncodedPrompt]:
    encoded: list[EncodedPrompt] = []
    for prompt in prompts:
        rendered = render_prompt(prompt)
        token_ids = tuple(tokenizer.encode(rendered))
        require(token_ids, f"gate prompt encoded to no tokens: {prompt.name}")
        require(
            len(token_ids) <= max_prompt_tokens,
            f"gate prompt exceeds {max_prompt_tokens} tokens: {prompt.name}={len(token_ids)}",
        )
        encoded.append(
            EncodedPrompt(
                prompt=prompt,
                rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                token_ids=token_ids,
            )
        )
    return encoded


def capture_prompt_hashes(state: dict[str, Any]) -> set[str]:
    require(state.get("format") == "ornith35-mtp-teacher-capture-v2", "capture format mismatch")
    completed = state.get("completed")
    require(isinstance(completed, dict) and completed, "capture has no completed prompts")
    hashes = set()
    for key, entry in completed.items():
        require(isinstance(key, str) and isinstance(entry, dict), "capture prompt record is invalid")
        value = entry.get("prompt_sha256")
        require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"capture prompt hash is invalid: {key}",
        )
        hashes.add(value)
    return hashes


def require_prompt_disjoint(encoded: Sequence[EncodedPrompt], capture_state: dict[str, Any]) -> None:
    training = capture_prompt_hashes(capture_state)
    overlap = [prompt.prompt.name for prompt in encoded if prompt.rendered_sha256 in training]
    require(not overlap, f"gate prompts overlap MTP teacher capture: {','.join(overlap)}")


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be unique nonnegative integers")
    return seeds


def repository_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect repository state: {exc}") from exc
    require(
        len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
        "repository revision is invalid",
    )
    return revision, dirty


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_unique_tokens(generated: list[int], emitted: tuple[int, ...]) -> None:
    require(emitted, "draft verifier emitted no target token")
    if not generated:
        generated.extend(emitted)
        return
    require(generated[-1] == emitted[0], "draft anchor does not continue prior output")
    generated.extend(emitted[1:])


def _evaluate_aux_transition(
    result: model.TextModelAuxTransition | model.TextModelAuxChunkTransition,
) -> None:
    if isinstance(result, model.TextModelAuxChunkTransition):
        model.evaluate_chunk_transition(result)
    else:
        model.evaluate_transition(result)
    mx.eval(*result.auxiliary_hidden_states)


def prefill_target(
    prompt_ids: Sequence[int],
    target_weights: model.TextModelWeights,
    max_chunk: int,
) -> tuple[speculative.GreedyTargetCursor, mx.array, tuple[int, ...]]:
    schedule = generate.prefill_schedule(len(prompt_ids), max_chunk)
    state = model.initial_state(target_weights, model.PRODUCTION_CONFIG)
    hidden_chunks = []
    offset = 0
    for size in schedule:
        token_slice = prompt_ids[offset : offset + size]
        if size == 1:
            transition = model.forward_hidden_token(token_slice[0], state, target_weights)
            model.evaluate_transition(transition)
            hidden_chunks.append(transition.hidden.reshape(1, -1))
        else:
            transition = model.prefill_hidden_chunk(
                token_slice,
                state,
                target_weights,
                use_steel=False,
            )
            model.evaluate_chunk_transition(transition)
            hidden_chunks.append(transition.hidden)
        state = transition.state
        offset += size
    require(offset == len(prompt_ids), "target prefill is incomplete")
    target_hidden = mx.concatenate(hidden_chunks, axis=0)
    final_hidden = target_hidden[-1]
    logits = model.project_lm_head(target_weights.lm_head, final_hidden)
    mx.eval(target_hidden, logits)
    return speculative.GreedyTargetCursor(state, final_hidden, logits), target_hidden, schedule


def prefill_target_and_dspark(
    prompt_ids: Sequence[int],
    target_weights: model.TextModelWeights,
    draft_weights: dspark.MLXDSparkWeights,
    capacity: int,
    max_chunk: int,
) -> tuple[
    speculative.GreedyTargetCursor,
    dspark.MLXDSparkLinearContextState,
    tuple[int, ...],
]:
    schedule = generate.prefill_schedule(len(prompt_ids), max_chunk)
    target_state = model.initial_state(target_weights, model.PRODUCTION_CONFIG)
    draft_context = dspark.initial_linear_context(DSPARK_CONFIG, capacity)
    offset = 0
    final_hidden = None
    for size in schedule:
        token_slice = prompt_ids[offset : offset + size]
        if size == 1:
            result = model.forward_hidden_token_with_aux(
                token_slice[0],
                target_state,
                target_weights,
                DSPARK_CONFIG.aux_hidden_state_indices,
            )
        else:
            result = model.prefill_hidden_chunk_with_aux(
                token_slice,
                target_state,
                target_weights,
                DSPARK_CONFIG.aux_hidden_state_indices,
                use_steel=False,
            )
        _evaluate_aux_transition(result)
        draft_context = dspark_runtime.append_target_auxiliary(
            draft_context,
            result,
            draft_weights,
            _validated=True,
        )
        target_state = result.state
        final_hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
        offset += size
    require(final_hidden is not None and offset == len(prompt_ids), "DSpark prefill is incomplete")
    mx.eval(*draft_context.keys, *draft_context.values)
    logits = model.project_lm_head(target_weights.lm_head, final_hidden)
    mx.eval(logits)
    return speculative.GreedyTargetCursor(target_state, final_hidden, logits), draft_context, schedule


def _active_attention_arrays(
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState,
) -> tuple[mx.array, mx.array]:
    if isinstance(state, attention.MLXLinearAttentionState):
        return state.keys[:, : state.position], state.values[:, : state.position]
    return state.keys, state.values


def exact_cursor_mismatches(
    actual: speculative.GreedyTargetCursor,
    expected: speculative.GreedyTargetCursor,
) -> list[str]:
    mismatches = []
    if actual.state.position != expected.state.position:
        mismatches.append("position")
    if len(actual.state.layers) != len(expected.state.layers):
        mismatches.append("layer-count")
        return mismatches
    comparisons: list[tuple[str, mx.array]] = [
        ("hidden", mx.array_equal(actual.hidden, expected.hidden)),
        ("logits", mx.array_equal(actual.logits, expected.logits)),
    ]
    for index, (left, right) in enumerate(zip(actual.state.layers, expected.state.layers)):
        if isinstance(left, gdn.MLXGDNState) and isinstance(right, gdn.MLXGDNState):
            comparisons.extend(
                (
                    (f"layer-{index}-conv", mx.array_equal(left.conv, right.conv)),
                    (f"layer-{index}-recurrent", mx.array_equal(left.recurrent, right.recurrent)),
                )
            )
            continue
        if isinstance(left, (attention.MLXAttentionState, attention.MLXLinearAttentionState)) and isinstance(
            right,
            (attention.MLXAttentionState, attention.MLXLinearAttentionState),
        ):
            left_keys, left_values = _active_attention_arrays(left)
            right_keys, right_values = _active_attention_arrays(right)
            comparisons.extend(
                (
                    (f"layer-{index}-keys", mx.array_equal(left_keys, right_keys)),
                    (f"layer-{index}-values", mx.array_equal(left_values, right_values)),
                )
            )
            continue
        mismatches.append(f"layer-{index}-type")
    mx.eval(*(comparison for _, comparison in comparisons))
    mismatches.extend(
        name for name, comparison in comparisons if not bool(comparison.item())
    )
    return mismatches


def replay_target_path(
    generated: Sequence[int],
    cursor: speculative.GreedyTargetCursor,
    target_weights: model.TextModelWeights,
    capacity: int,
    *,
    sampled: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> tuple[speculative.GreedyTargetCursor, list[float]]:
    require(len(generated) >= 2, "target replay path is too short")
    session = model.start_linear_decode_session(
        target_weights,
        cursor.state,
        capacity,
        compile_gdn_layers=True,
        compile_attention_tails=True,
    )
    replay = speculative.GreedyTargetCursor(session.state, cursor.hidden, cursor.logits)
    rng = random.Random(seed ^ 0x5DEECE66D)
    if sampled:
        initial = sampling.target_distribution(
            replay.logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            hidden=replay.hidden,
            lm_head=target_weights.lm_head,
        )
        require(initial.probability(generated[0]) > 0.0, "sampled anchor has no target mass")
        initial.sample(rng)
    else:
        require(
            generated[0]
            == speculative.greedy_token(replay.logits, replay.hidden, target_weights.lm_head),
            "greedy replay anchor mismatch",
        )
    elapsed = []
    for index, token_id in enumerate(generated[:-1]):
        started = time.perf_counter()
        result = model.forward_linear_session_token(token_id, session)
        replay = speculative.cursor_from_result(result)
        if sampled:
            distribution = sampling.target_distribution(
                replay.logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                hidden=replay.hidden,
                lm_head=target_weights.lm_head,
            )
            require(
                distribution.probability(generated[index + 1]) > 0.0,
                f"sampled emitted token has no target mass at transition {index}",
            )
            distribution.sample(rng)
        else:
            require(
                generated[index + 1]
                == speculative.greedy_token(
                    replay.logits,
                    replay.hidden,
                    target_weights.lm_head,
                ),
                f"greedy replay diverged at transition {index}",
            )
        mx.synchronize()
        elapsed.append(time.perf_counter() - started)
    return replay, elapsed


def _trajectory_record(
    *,
    engine: str,
    prompt: EncodedPrompt,
    seed: int | None,
    generated: Sequence[int],
    step_seconds: Sequence[float],
    step_transitions: Sequence[int],
    target_seconds: Sequence[float],
    accepted_future: int,
    proposed_future: int,
    draft_blocks: int,
    target_only_steps: int,
    detached_after: int | None,
    exact_replay: bool,
    state_mismatches: Sequence[str],
) -> dict[str, Any]:
    transitions = len(generated) - 1
    require(transitions == sum(step_transitions), "step transition accounting mismatch")
    require(transitions == len(target_seconds), "target timing accounting mismatch")
    first_transitions = step_transitions[0]
    draft_elapsed = math.fsum(step_seconds)
    target_elapsed = math.fsum(target_seconds)
    steady_transitions = transitions - first_transitions
    draft_steady = math.fsum(step_seconds[1:])
    target_steady = math.fsum(target_seconds[first_transitions:])
    require(draft_elapsed > 0.0 and target_elapsed > 0.0, "trajectory timing is empty")
    require(steady_transitions > 0 and draft_steady > 0.0 and target_steady > 0.0, "steady timing is empty")
    return {
        "accepted_future": accepted_future,
        "acceptance": accepted_future / proposed_future if proposed_future else None,
        "decode_tokens_per_second": transitions / draft_elapsed,
        "detached_after_mtp_blocks": detached_after,
        "draft_blocks": draft_blocks,
        "draft_seconds": draft_elapsed,
        "draft_steady_seconds": draft_steady,
        "engine": engine,
        "exact_replay": exact_replay,
        "generated_token_ids": list(generated),
        "peak_gib": mx.get_peak_memory() / 2**30,
        "prompt_name": prompt.prompt.name,
        "prompt_sha256": prompt.rendered_sha256,
        "prompt_tokens": len(prompt.token_ids),
        "proposed_future": proposed_future,
        "seed": seed,
        "speedup": target_elapsed / draft_elapsed,
        "state_mismatches": list(state_mismatches),
        "steady_speedup": target_steady / draft_steady,
        "steady_transitions": steady_transitions,
        "target_only_steps": target_only_steps,
        "target_seconds": target_elapsed,
        "target_steady_seconds": target_steady,
        "target_tokens_per_second": transitions / target_elapsed,
        "transitions": transitions,
    }


def run_mtp_trajectory(
    engine: str,
    prompt: EncodedPrompt,
    cursor: speculative.GreedyTargetCursor,
    target_hidden: mx.array,
    target_weights: model.TextModelWeights,
    mtp_weights: mtp.MLXMTPWeights,
    *,
    steps: int,
    block_tokens: int,
    capacity: int,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
    adaptation_policy: mtp_runtime.MTPAdaptivePolicy,
) -> dict[str, Any]:
    sampled = engine == "mtp-sampled"
    rng = random.Random(seed)
    if sampled:
        anchor = generate.choose_next_token(
            cursor.logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
            hidden=cursor.hidden,
            lm_head=target_weights.lm_head,
        )
    else:
        anchor = speculative.greedy_token(cursor.logits, cursor.hidden, target_weights.lm_head)
    context = mtp_runtime.build_prompt_context(
        prompt.token_ids,
        target_hidden,
        anchor,
        target_weights.embedding,
        mtp_weights,
        mtp.PRODUCTION_CONFIG,
    )
    target_linear = model.start_linear_decode_session(
        target_weights,
        cursor.state,
        capacity,
        compile_gdn_layers=True,
        compile_attention_tails=True,
    )
    linear_cursor = speculative.GreedyTargetCursor(target_linear.state, cursor.hidden, cursor.logits)
    start_session = (
        mtp_runtime.start_sampled_session
        if sampled
        else mtp_runtime.start_greedy_session
    )
    session = start_session(
        target_weights,
        linear_cursor,
        mtp_weights,
        context,
        model.PRODUCTION_CONFIG,
        mtp.PRODUCTION_CONFIG,
        block_tokens=block_tokens,
        compile_prefill_tails=True,
        target_linear_session=target_linear,
        draft_exact_rerank=True,
    )
    adaptive = mtp_runtime.start_adaptive_session(session, adaptation_policy)
    generated: list[int] = []
    step_seconds = []
    step_transitions = []
    accepted_future = 0
    proposed_future = 0
    draft_blocks = 0
    target_only_steps = 0
    for _ in range(steps):
        started = time.perf_counter()
        if sampled:
            step, adaptive = mtp_runtime.step_adaptive_sampled(
                adaptive,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
        else:
            step, adaptive = mtp_runtime.step_adaptive_greedy(adaptive)
        mx.synchronize()
        step_seconds.append(time.perf_counter() - started)
        verification = step.verification
        step_transitions.append(len(verification.committed_tokens))
        append_unique_tokens(generated, verification.emitted_tokens)
        future_slots = len(step.proposal.future_token_ids)
        if future_slots:
            draft_blocks += 1
            proposed_future += future_slots
            accepted_future += max(0, verification.accepted_count - 1)
        else:
            target_only_steps += 1
    verifier = adaptive.active.verifier
    target_cursor, target_seconds = replay_target_path(
        generated,
        cursor,
        target_weights,
        capacity,
        sampled=sampled,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )
    mismatches = exact_cursor_mismatches(verifier.cursor, target_cursor)
    return _trajectory_record(
        engine=engine,
        prompt=prompt,
        seed=seed if sampled else None,
        generated=generated,
        step_seconds=step_seconds,
        step_transitions=step_transitions,
        target_seconds=target_seconds,
        accepted_future=accepted_future,
        proposed_future=proposed_future,
        draft_blocks=draft_blocks,
        target_only_steps=target_only_steps,
        detached_after=adaptive.detached_after_mtp_blocks,
        exact_replay=not mismatches,
        state_mismatches=mismatches,
    )


def run_dspark_trajectory(
    prompt: EncodedPrompt,
    cursor: speculative.GreedyTargetCursor,
    draft_context: dspark.MLXDSparkLinearContextState,
    target_weights: model.TextModelWeights,
    draft_weights: dspark.MLXDSparkWeights,
    *,
    steps: int,
    capacity: int,
    target_stage_tokens: int,
) -> dict[str, Any]:
    target_linear = model.start_linear_decode_session(
        target_weights,
        cursor.state,
        capacity,
        compile_gdn_layers=True,
        compile_attention_tails=True,
    )
    linear_cursor = speculative.GreedyTargetCursor(target_linear.state, cursor.hidden, cursor.logits)
    session = dspark_runtime.start_greedy_session(
        target_weights,
        linear_cursor,
        draft_weights,
        draft_context,
        exact_block_lm_head=None,
        target_linear_session=target_linear,
        target_stage_tokens=target_stage_tokens,
        compile_prefill_tails=target_stage_tokens != 1,
    )
    generated: list[int] = []
    step_seconds = []
    step_transitions = []
    accepted_future = 0
    proposed_future = 0
    for _ in range(steps):
        started = time.perf_counter()
        step, session = dspark_runtime.step_greedy(session)
        mx.eval(
            step.proposal.confidence,
            *session.draft_context.keys,
            *session.draft_context.values,
        )
        mx.synchronize()
        step_seconds.append(time.perf_counter() - started)
        verification = step.verification
        step_transitions.append(len(verification.committed_tokens))
        append_unique_tokens(generated, verification.emitted_tokens)
        proposed_future += DSPARK_CONFIG.block_size - 1
        accepted_future += max(0, verification.accepted_count - 1)
    target_cursor, target_seconds = replay_target_path(
        generated,
        cursor,
        target_weights,
        capacity,
        sampled=False,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=0,
    )
    mismatches = exact_cursor_mismatches(session.verifier.cursor, target_cursor)
    return _trajectory_record(
        engine="dspark-greedy",
        prompt=prompt,
        seed=None,
        generated=generated,
        step_seconds=step_seconds,
        step_transitions=step_transitions,
        target_seconds=target_seconds,
        accepted_future=accepted_future,
        proposed_future=proposed_future,
        draft_blocks=steps,
        target_only_steps=0,
        detached_after=None,
        exact_replay=not mismatches,
        state_mismatches=mismatches,
    )


def aggregate_runs(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(records, "cannot aggregate an empty gate")
    transitions = sum(record["transitions"] for record in records)
    steady_transitions = sum(record["steady_transitions"] for record in records)
    draft_seconds = math.fsum(record["draft_seconds"] for record in records)
    target_seconds = math.fsum(record["target_seconds"] for record in records)
    draft_steady = math.fsum(record["draft_steady_seconds"] for record in records)
    target_steady = math.fsum(record["target_steady_seconds"] for record in records)
    exact_runs = sum(bool(record["exact_replay"]) for record in records)
    proposed = sum(record["proposed_future"] for record in records)
    accepted = sum(record["accepted_future"] for record in records)
    prompt_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        prompt_groups.setdefault(record["prompt_name"], []).append(record)
    prompt_speedups = {
        name: math.fsum(record["target_steady_seconds"] for record in group)
        / math.fsum(record["draft_steady_seconds"] for record in group)
        for name, group in prompt_groups.items()
    }
    aggregate_speedup = target_seconds / draft_seconds
    steady_speedup = target_steady / draft_steady
    faster_prompts = sum(speedup > 1.0 for speedup in prompt_speedups.values())
    exact = exact_runs == len(records)
    eligible = (
        exact
        and steady_speedup >= 1.02
        and min(prompt_speedups.values()) >= 0.90
        and faster_prompts / len(prompt_speedups) >= 0.75
    )
    classification = (
        "production-beneficial"
        if eligible
        else "exact-but-not-production-beneficial"
        if exact
        else "invalid-target-replay"
    )
    return {
        "accepted_future": accepted,
        "acceptance": accepted / proposed if proposed else None,
        "classification": classification,
        "decode_tokens_per_second": transitions / draft_seconds,
        "detached_runs": sum(record["detached_after_mtp_blocks"] is not None for record in records),
        "draft_seconds": draft_seconds,
        "draft_steady_seconds": draft_steady,
        "exact_runs": exact_runs,
        "faster_prompt_fraction": faster_prompts / len(prompt_speedups),
        "faster_prompts": faster_prompts,
        "maximum_peak_gib": max(record["peak_gib"] for record in records),
        "median_run_steady_speedup": statistics.median(
            record["steady_speedup"] for record in records
        ),
        "minimum_prompt_steady_speedup": min(prompt_speedups.values()),
        "prompt_count": len(prompt_speedups),
        "prompt_steady_speedups": prompt_speedups,
        "proposed_future": proposed,
        "run_count": len(records),
        "speedup": aggregate_speedup,
        "steady_speedup": steady_speedup,
        "steady_transitions": steady_transitions,
        "target_seconds": target_seconds,
        "target_steady_seconds": target_steady,
        "target_tokens_per_second": transitions / target_seconds,
        "thresholds": {
            "aggregate_steady_speedup_minimum": 1.02,
            "faster_prompt_fraction_minimum": 0.75,
            "prompt_steady_speedup_floor": 0.90,
        },
        "transitions": transitions,
    }


def expected_run_keys(engine: str, prompts: Sequence[EncodedPrompt], seeds: Sequence[int]) -> list[str]:
    if engine == "mtp-sampled":
        return [f"{prompt.prompt.name}:seed-{seed}" for prompt in prompts for seed in seeds]
    return [f"{prompt.prompt.name}:greedy" for prompt in prompts]


def build_identity(
    args: argparse.Namespace,
    prompts_path: Path,
    prompts: Sequence[GatePrompt],
    encoded: Sequence[EncodedPrompt],
    tokenizer: TextTokenizer,
    capture_state_path: Path,
    revision: str,
) -> dict[str, Any]:
    target_state = load_json_object(args.root / "source-nvfp4-state.json")
    target_weight = target_state.get("weight")
    require(isinstance(target_weight, dict), "target source identity is absent")
    require(target_state.get("repository") == PRODUCTION_MODEL_ID, "target repository mismatch")
    require(target_state.get("revision") == PRODUCTION_MODEL_REVISION, "target revision mismatch")
    require(
        target_weight.get("sha256") == nvfp4.EXPECTED_WEIGHT_SHA256,
        "target weight identity mismatch",
    )
    identity: dict[str, Any] = {
        "capture_state_sha256": sha256_file(capture_state_path),
        "chat_template_sha256": tokenizer.template_sha256,
        "corpus_canonical_sha256": canonical_sha256([prompt.canonical() for prompt in prompts]),
        "corpus_file_sha256": sha256_file(prompts_path),
        "engine": args.engine,
        "model_repository": PRODUCTION_MODEL_ID,
        "model_revision": PRODUCTION_MODEL_REVISION,
        "prompt_count": len(encoded),
        "repository_revision": revision,
        "settings": {
            "adaptive_minimum_future_acceptance": args.adaptive_minimum_future_acceptance,
            "adaptive_minimum_mtp_blocks": args.adaptive_minimum_mtp_blocks,
            "adaptive_window_blocks": args.adaptive_window_blocks,
            "block_tokens": args.block_tokens if args.engine.startswith("mtp-") else DSPARK_CONFIG.block_size,
            "ignore_eos": True,
            "max_prompt_tokens": args.max_prompt_tokens,
            "prefill_chunk": args.prefill_chunk,
            "seeds": list(args.seeds) if args.engine == "mtp-sampled" else [],
            "steps": args.steps,
            "target_stage_tokens": args.target_stage_tokens if args.engine == "dspark-greedy" else None,
            "temperature": args.temperature if args.engine == "mtp-sampled" else 0.0,
            "top_k": args.top_k if args.engine == "mtp-sampled" else 1,
            "top_p": args.top_p if args.engine == "mtp-sampled" else 1.0,
        },
        "target_source_state_sha256": sha256_file(args.root / "source-nvfp4-state.json"),
        "target_weight_sha256": target_weight.get("sha256"),
        "tokenizer_sha256": tokenizer.tokenizer_sha256,
        "tool_sha256": sha256_file(Path(__file__)),
    }
    if args.engine.startswith("mtp-"):
        adaptation_state = load_json_object(args.adaptation_dir / "state.json")
        artifact = adaptation_state.get("artifact")
        require(isinstance(artifact, dict), "MTP adaptation artifact identity is absent")
        identity.update(
            {
                "adaptation_artifact_sha256": artifact.get("sha256"),
                "adaptation_state_sha256": sha256_file(args.adaptation_dir / "state.json"),
                "mtp_sidecar_sha256": mtp.EXPECTED_SIDECAR_SHA256,
                "mtp_source_state_sha256": sha256_file(args.root / "source-mtp-state.json"),
            }
        )
    else:
        source_state = load_json_object(args.root / "source-dspark-state.json")
        weight = source_state.get("weight")
        require(isinstance(weight, dict), "DSpark source identity is absent")
        identity.update(
            {
                "dspark_source_state_sha256": sha256_file(args.root / "source-dspark-state.json"),
                "dspark_weight_sha256": weight.get("sha256"),
            }
        )
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seeds", type=parse_seeds, default=(11, 29, 47))
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--block-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--target-stage-tokens", type=int, default=1)
    parser.add_argument("--adaptive-minimum-mtp-blocks", type=int, default=8)
    parser.add_argument("--adaptive-window-blocks", type=int, default=4)
    parser.add_argument("--adaptive-minimum-future-acceptance", type=float, default=0.70)
    parser.add_argument("--adaptation-dir", type=Path)
    parser.add_argument("--capture-state", type=Path)
    parser.add_argument(
        "--verify-mtp-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.steps >= 2, "gate steps must be at least two")
        require(args.max_prompt_tokens > 0, "maximum prompt tokens must be positive")
        require(args.max_prompts is None or args.max_prompts > 0, "max prompts must be positive")
        require(args.max_runs is None or args.max_runs > 0, "max runs must be positive")
        require(
            2 <= args.block_tokens <= speculative.MAX_PROPOSAL_TOKENS,
            "MTP block tokens must be between 2 and 8",
        )
        require(args.temperature > 0.0, "sampling temperature must be positive")
        require(args.top_k > 0, "sampling top-k must be positive")
        require(0.0 < args.top_p <= 1.0, "sampling top-p must be in (0, 1]")
        require(
            args.target_stage_tokens == 1,
            "the accepted DSpark gate regime requires target-stage-tokens=1",
        )
        require(
            args.adaptive_minimum_mtp_blocks >= 2,
            "adaptive MTP minimum blocks must be at least two",
        )
        require(
            1 <= args.adaptive_window_blocks <= args.adaptive_minimum_mtp_blocks,
            "adaptive MTP window is invalid",
        )
        require(
            0.0 <= args.adaptive_minimum_future_acceptance <= 1.0,
            "adaptive MTP acceptance threshold is invalid",
        )
        args.root = args.root.resolve()
        args.prompts = args.prompts.resolve()
        args.output = args.output.resolve()
        args.adaptation_dir = (
            args.adaptation_dir.resolve()
            if args.adaptation_dir is not None
            else args.root / DEFAULT_ADAPTATION
        )
        capture_state_path = (
            args.capture_state.resolve()
            if args.capture_state is not None
            else args.root / DEFAULT_CAPTURE_STATE
        )
        revision, dirty = repository_state()
        require(not dirty, "repository is dirty; commit the gate implementation before measurement")
        prompts = load_prompts(args.prompts)
        if args.max_prompts is not None:
            prompts = prompts[: args.max_prompts]
        tokenizer = load_text_tokenizer(args.root)
        encoded = encode_prompts(
            prompts,
            tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        capture_state = load_json_object(capture_state_path)
        require_prompt_disjoint(encoded, capture_state)
        max_block = args.block_tokens if args.engine.startswith("mtp-") else DSPARK_CONFIG.block_size
        require(
            max(len(prompt.token_ids) for prompt in encoded) + args.steps * max_block
            <= NATIVE_CONTEXT_TOKENS,
            "gate trajectory exceeds native context",
        )
        identity = build_identity(
            args,
            args.prompts,
            prompts,
            encoded,
            tokenizer,
            capture_state_path,
            revision,
        )
        expected_keys = expected_run_keys(args.engine, encoded, args.seeds)
        if args.output.exists():
            state = load_json_object(args.output)
            require(state.get("format") == FORMAT, "gate output format mismatch")
            require(state.get("identity") == identity, "gate output identity mismatch")
            require(isinstance(state.get("runs"), dict), "gate output run state is invalid")
            require(set(state["runs"]) <= set(expected_keys), "gate output contains an unknown run")
            require(
                all(record.get("exact_replay") is True for record in state["runs"].values()),
                "gate output contains a failed target replay",
            )
            if state.get("status") == "complete":
                require(set(state["runs"]) == set(expected_keys), "complete gate is missing runs")
                print(
                    f"draft-gate-complete-resume engine={args.engine} output={args.output} "
                    f"classification={state['aggregate']['classification']}",
                    flush=True,
                )
                return 0
            require(state.get("status") == "running", "gate output status is invalid")
        else:
            state = {
                "aggregate": None,
                "format": FORMAT,
                "identity": identity,
                "runs": {},
                "status": "running",
            }
            atomic_json(args.output, state)
        pending = [key for key in expected_keys if key not in state["runs"]]
        print(
            "draft-gate-start "
            f"engine={args.engine} prompts={len(encoded)} runs={len(expected_keys)} "
            f"completed={len(state['runs'])} pending={len(pending)} steps={args.steps} "
            f"output={args.output} free_gib={os.statvfs(args.output.parent).f_bavail * os.statvfs(args.output.parent).f_frsize / 2**30:.3f}",
            flush=True,
        )
        if not pending:
            state["aggregate"] = aggregate_runs(list(state["runs"].values()))
            state["status"] = "complete"
            atomic_json(args.output, state)
            return 0

        nvfp4.require_verified_source(args.root)
        mx.set_cache_limit(128 * 2**20)
        target_weights = model.load_text_model(
            args.root,
            map_embedding=True,
            quantize_lm_head=True,
        )
        mtp_weights = None
        draft_weights = None
        if args.engine.startswith("mtp-"):
            mtp_weights = mtp.load_weights(
                args.root,
                verify_hash=args.verify_mtp_hash,
                adaptation_dir=args.adaptation_dir,
            )
        else:
            draft_weights = dspark.load_weights(dspark.require_verified_source(args.root))

        policy = mtp_runtime.MTPAdaptivePolicy(
            minimum_mtp_blocks=args.adaptive_minimum_mtp_blocks,
            window_blocks=args.adaptive_window_blocks,
            minimum_future_acceptance=args.adaptive_minimum_future_acceptance,
        )
        processed = 0
        warmed = False
        for prompt in encoded:
            prompt_pending = [key for key in pending if key.startswith(f"{prompt.prompt.name}:")]
            if not prompt_pending:
                continue
            capacity = len(prompt.token_ids) + args.steps * max_block
            if args.engine.startswith("mtp-"):
                require(mtp_weights is not None, "MTP weights are not loaded")
                cursor, target_hidden, schedule = prefill_target(
                    prompt.token_ids,
                    target_weights,
                    args.prefill_chunk,
                )
                run_seeds = args.seeds if args.engine == "mtp-sampled" else (0,)
                for seed in run_seeds:
                    key = (
                        f"{prompt.prompt.name}:seed-{seed}"
                        if args.engine == "mtp-sampled"
                        else f"{prompt.prompt.name}:greedy"
                    )
                    if key not in pending:
                        continue
                    if not warmed:
                        print(f"draft-gate-warm engine={args.engine} prompt={prompt.prompt.name}", flush=True)
                        run_mtp_trajectory(
                            args.engine,
                            prompt,
                            cursor,
                            target_hidden,
                            target_weights,
                            mtp_weights,
                            steps=2,
                            block_tokens=args.block_tokens,
                            capacity=capacity,
                            seed=0,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            adaptation_policy=policy,
                        )
                        warmed = True
                    record = run_mtp_trajectory(
                        args.engine,
                        prompt,
                        cursor,
                        target_hidden,
                        target_weights,
                        mtp_weights,
                        steps=args.steps,
                        block_tokens=args.block_tokens,
                        capacity=capacity,
                        seed=seed,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        adaptation_policy=policy,
                    )
                    state["runs"][key] = record
                    state["aggregate"] = None
                    atomic_json(args.output, state)
                    require(record["exact_replay"], f"target replay mismatch: {record['state_mismatches']}")
                    processed += 1
                    print(
                        "draft-gate-run "
                        f"key={key} prompt_tokens={len(prompt.token_ids)} "
                        f"transitions={record['transitions']} "
                        f"acceptance={record['acceptance']} "
                        f"speedup={record['speedup']:.4f} "
                        f"steady_speedup={record['steady_speedup']:.4f} "
                        f"detached_after={record['detached_after_mtp_blocks']} "
                        f"peak_gib={record['peak_gib']:.3f}",
                        flush=True,
                    )
                    if args.max_runs is not None and processed >= args.max_runs:
                        print(f"draft-gate-stop max_runs={args.max_runs}", flush=True)
                        return 0
            else:
                require(draft_weights is not None, "DSpark weights are not loaded")
                cursor, draft_context, schedule = prefill_target_and_dspark(
                    prompt.token_ids,
                    target_weights,
                    draft_weights,
                    capacity,
                    args.prefill_chunk,
                )
                key = f"{prompt.prompt.name}:greedy"
                if not warmed:
                    print(f"draft-gate-warm engine={args.engine} prompt={prompt.prompt.name}", flush=True)
                    warm_cursor, warm_context, _ = prefill_target_and_dspark(
                        prompt.token_ids,
                        target_weights,
                        draft_weights,
                        capacity,
                        args.prefill_chunk,
                    )
                    run_dspark_trajectory(
                        prompt,
                        warm_cursor,
                        warm_context,
                        target_weights,
                        draft_weights,
                        steps=2,
                        capacity=capacity,
                        target_stage_tokens=args.target_stage_tokens,
                    )
                    warmed = True
                record = run_dspark_trajectory(
                    prompt,
                    cursor,
                    draft_context,
                    target_weights,
                    draft_weights,
                    steps=args.steps,
                    capacity=capacity,
                    target_stage_tokens=args.target_stage_tokens,
                )
                state["runs"][key] = record
                state["aggregate"] = None
                atomic_json(args.output, state)
                require(record["exact_replay"], f"target replay mismatch: {record['state_mismatches']}")
                processed += 1
                print(
                    "draft-gate-run "
                    f"key={key} prompt_tokens={len(prompt.token_ids)} "
                    f"transitions={record['transitions']} "
                    f"acceptance={record['acceptance']} "
                    f"speedup={record['speedup']:.4f} "
                    f"steady_speedup={record['steady_speedup']:.4f} "
                    f"peak_gib={record['peak_gib']:.3f}",
                    flush=True,
                )
                if args.max_runs is not None and processed >= args.max_runs:
                    print(f"draft-gate-stop max_runs={args.max_runs}", flush=True)
                    return 0
            del schedule
            gc.collect()

        require(set(state["runs"]) == set(expected_keys), "gate did not complete every expected run")
        state["aggregate"] = aggregate_runs(list(state["runs"].values()))
        state["status"] = "complete"
        atomic_json(args.output, state)
        aggregate = state["aggregate"]
        print(
            "draft-gate-result "
            f"engine={args.engine} exact={aggregate['exact_runs']}/{aggregate['run_count']} "
            f"acceptance={aggregate['acceptance']} "
            f"decode_tokens_s={aggregate['decode_tokens_per_second']:.3f} "
            f"target_tokens_s={aggregate['target_tokens_per_second']:.3f} "
            f"speedup={aggregate['speedup']:.4f} "
            f"steady_speedup={aggregate['steady_speedup']:.4f} "
            f"minimum_prompt_steady_speedup={aggregate['minimum_prompt_steady_speedup']:.4f} "
            f"faster_prompts={aggregate['faster_prompts']}/{aggregate['prompt_count']} "
            f"classification={aggregate['classification']} "
            f"peak_gib={aggregate['maximum_peak_gib']:.3f}",
            flush=True,
        )
        return 0
    except (
        argparse.ArgumentTypeError,
        dspark.MLXDSparkError,
        MoEError,
        TokenizerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"draft-gate-error: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

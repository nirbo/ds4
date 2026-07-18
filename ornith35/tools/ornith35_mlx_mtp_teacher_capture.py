#!/usr/bin/env python3
"""Capture bounded target-authoritative trajectories for Ornith-35 MTP tuning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import mlx.core as mx

import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_mtp as mtp
import ornith35_mlx_speculative as speculative
import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError, require
import ornith35_nvfp4 as nvfp4
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TextTokenizer,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


FORMAT = "ornith35-mtp-teacher-capture-v2"
CANDIDATE_POLICY = "target-mtp-union-v1"
DEFAULT_CANDIDATES = 64
DEFAULT_PROMPTS = (
    "Implement an LRU cache in Rust with O(1) get and put operations.",
    "Complete this Python binary search and explain every edge case.",
    "Find and fix the race condition in a bounded Go worker pool.",
    "Review a C++ lock-free queue for memory-ordering defects.",
    "Write a TypeScript parser for a small expression grammar.",
    "Design resumable shard processing with atomic verification and cleanup.",
    "Optimize a Metal reduction kernel without changing BF16 rounding.",
    "Diagnose a use-after-free in an async C++ request pipeline.",
)


@dataclass(frozen=True)
class PromptRecord:
    user: str
    system: str | None = None
    enable_thinking: bool = True

    def canonical(self) -> dict[str, object]:
        return {
            "user": self.user,
            "system": self.system,
            "enable_thinking": self.enable_thinking,
        }


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


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read teacher state {path}: {exc}") from exc
    require(isinstance(value, dict), f"teacher state is not an object: {path}")
    return value


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


def repository_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve repository revision: {exc}") from exc
    revision = result.stdout.strip()
    require(
        len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
        "invalid repository revision",
    )
    return revision


def load_prompts(
    path: Path | None,
    *,
    default_thinking: bool = True,
) -> list[PromptRecord]:
    if path is None:
        return [
            PromptRecord(user=prompt, enable_thinking=default_thinking)
            for prompt in DEFAULT_PROMPTS
        ]
    prompts: list[PromptRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            value = json.loads(line)
            require(isinstance(value, dict), f"prompt row {line_number} is not an object")
            user = value.get("user", value.get("prompt", value.get("text")))
            system = value.get("system")
            thinking = value.get("enable_thinking", default_thinking)
            require(isinstance(user, str), f"prompt row {line_number} has no user text")
            require(
                system is None or isinstance(system, str),
                f"prompt row {line_number} has an invalid system message",
            )
            require(
                isinstance(thinking, bool),
                f"prompt row {line_number} has an invalid thinking flag",
            )
            record = PromptRecord(user=user, system=system, enable_thinking=thinking)
        else:
            record = PromptRecord(user=line, enable_thinking=default_thinking)
        require(record.user.strip(), f"prompt row {line_number} is empty")
        prompts.append(record)
    require(prompts, "prompt file contains no prompts")
    return prompts


def render_prompt(record: PromptRecord) -> str:
    return render_text_prompt(
        record.user,
        system=record.system,
        enable_thinking=record.enable_thinking,
    )


def _metadata_int(metadata: dict[str, str], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"teacher metadata has invalid {name}") from exc
    return value


def validate_trace_arrays(
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
    *,
    prompt_index: int,
    expected_prompt_sha256: str | None = None,
) -> None:
    required = {
        "target_hidden",
        "token_ids",
        "expected_token_ids",
        "candidate_token_ids",
        "candidate_logits",
        "scored",
        "prompt_indices",
        "positions",
    }
    require(set(arrays) == required, "teacher trace tensor set mismatch")
    require(metadata.get("format") == FORMAT, "teacher trace format mismatch")
    require(
        metadata.get("candidate_policy") == CANDIDATE_POLICY,
        "teacher candidate policy mismatch",
    )
    require(
        metadata.get("prompt_index") == str(prompt_index),
        "teacher trace prompt index mismatch",
    )
    if expected_prompt_sha256 is not None:
        require(
            metadata.get("prompt_sha256") == expected_prompt_sha256,
            "teacher trace prompt identity mismatch",
        )
    prompt_tokens = _metadata_int(metadata, "prompt_tokens")
    generated_tokens = _metadata_int(metadata, "generated_tokens")
    candidates = _metadata_int(metadata, "candidate_count")
    token_count = prompt_tokens + generated_tokens
    row_count = token_count - 1
    require(prompt_tokens > 0 and generated_tokens > 0, "teacher trace is empty")
    require(1 <= candidates <= 256, "teacher candidate count is invalid")
    require(
        arrays["target_hidden"].dtype == mx.bfloat16
        and arrays["target_hidden"].shape == (token_count, model.PRODUCTION_CONFIG.hidden_size),
        "teacher target-hidden tensor mismatch",
    )
    require(
        arrays["token_ids"].dtype == mx.int32
        and arrays["token_ids"].shape == (token_count,),
        "teacher token tensor mismatch",
    )
    for name in ("expected_token_ids", "scored", "prompt_indices", "positions"):
        require(
            arrays[name].dtype == mx.int32 and arrays[name].shape == (row_count,),
            f"teacher row tensor mismatch: {name}",
        )
    require(
        arrays["candidate_token_ids"].dtype == mx.int32
        and arrays["candidate_token_ids"].shape == (row_count, candidates),
        "teacher candidate-ID tensor mismatch",
    )
    require(
        arrays["candidate_logits"].dtype == mx.float32
        and arrays["candidate_logits"].shape == (row_count, candidates),
        "teacher candidate-logit tensor mismatch",
    )
    tokens = arrays["token_ids"].tolist()
    require(
        all(0 <= token < model.PRODUCTION_CONFIG.vocab_size for token in tokens),
        "teacher token is out of range",
    )
    require(
        arrays["prompt_indices"].tolist() == [prompt_index] * row_count,
        "teacher prompt indices are inconsistent",
    )
    require(
        arrays["positions"].tolist() == list(range(row_count)),
        "teacher positions are not contiguous",
    )
    scored = arrays["scored"].tolist()
    first_scored = prompt_tokens - 1
    expected_scored = [0] * first_scored + [1] * generated_tokens
    require(scored == expected_scored, "teacher scored mask is not the target continuation")
    expected_ids = arrays["expected_token_ids"].tolist()
    candidate_ids = arrays["candidate_token_ids"].tolist()
    candidate_logits = arrays["candidate_logits"].tolist()
    for row in range(row_count):
        if not scored[row]:
            require(expected_ids[row] == -1, "unscored teacher row has a hard label")
            require(
                candidate_ids[row] == [-1] * candidates,
                "unscored teacher row has candidate IDs",
            )
            require(
                candidate_logits[row] == [0.0] * candidates,
                "unscored teacher row has candidate logits",
            )
            continue
        require(
            0 <= expected_ids[row] < model.PRODUCTION_CONFIG.vocab_size,
            "teacher expected token is out of range",
        )
        require(
            len(set(candidate_ids[row])) == candidates
            and all(0 <= token < model.PRODUCTION_CONFIG.vocab_size for token in candidate_ids[row]),
            "teacher candidate IDs are invalid",
        )
        require(
            candidate_ids[row][0] == expected_ids[row],
            "teacher expected token is not the exact candidate winner",
        )
        pairs = list(zip(candidate_ids[row], candidate_logits[row]))
        require(
            pairs[1:] == sorted(pairs[1:], key=lambda item: (-item[1], item[0])),
            "teacher candidates are not deterministically ranked",
        )
        if row + 2 < token_count:
            require(
                expected_ids[row] == tokens[row + 2],
                "teacher hard label does not continue the target trajectory",
            )


def validate_completed(output_dir: Path, state: dict[str, Any]) -> None:
    require(state.get("format") == FORMAT, "teacher state format mismatch")
    completed = state.get("completed")
    require(isinstance(completed, dict), "teacher state has no completed map")
    total_rows = 0
    total_scored = 0
    for key, entry in sorted(completed.items(), key=lambda item: int(item[0])):
        prompt_index = int(key)
        require(isinstance(entry, dict), f"teacher state entry {key} is malformed")
        path = output_dir / entry.get("file", "")
        require(path.is_file() and not path.is_symlink(), f"teacher shard is missing: {path}")
        require(path.stat().st_size == entry.get("bytes"), f"teacher shard size mismatch: {path}")
        require(sha256_file(path) == entry.get("sha256"), f"teacher shard hash mismatch: {path}")
        arrays, metadata = mx.load(str(path), return_metadata=True)
        validate_trace_arrays(
            arrays,
            metadata,
            prompt_index=prompt_index,
            expected_prompt_sha256=entry.get("prompt_sha256"),
        )
        rows = arrays["expected_token_ids"].size
        scored = int(mx.sum(arrays["scored"]).item())
        require(rows == entry.get("rows"), f"teacher shard row count mismatch: {path}")
        require(scored == entry.get("scored_rows"), f"teacher shard scored count mismatch: {path}")
        total_rows += rows
        total_scored += scored
    require(total_rows == state.get("rows"), "teacher state total row count mismatch")
    require(total_scored == state.get("scored_rows"), "teacher state scored count mismatch")


def _prefill_target(
    prompt_ids: Sequence[int],
    weights: model.TextModelWeights,
    max_chunk: int,
) -> tuple[speculative.GreedyTargetCursor, mx.array, tuple[int, ...]]:
    schedule = generate.prefill_schedule(len(prompt_ids), max_chunk)
    state = model.initial_state(weights, model.PRODUCTION_CONFIG)
    hidden_chunks: list[mx.array] = []
    offset = 0
    for size in schedule:
        token_slice = prompt_ids[offset : offset + size]
        if size == 1:
            transition = model.forward_hidden_token(token_slice[0], state, weights)
            model.evaluate_transition(transition)
            hidden_chunks.append(transition.hidden.reshape(1, -1))
        else:
            transition = model.prefill_hidden_chunk(
                token_slice,
                state,
                weights,
                use_steel=False,
            )
            model.evaluate_chunk_transition(transition)
            hidden_chunks.append(transition.hidden)
        state = transition.state
        offset += size
    require(offset == len(prompt_ids), "teacher target prefill is incomplete")
    hidden = mx.concatenate(hidden_chunks, axis=0)
    logits = model.project_lm_head(weights.lm_head, hidden[-1])
    mx.eval(hidden, logits)
    return (
        speculative.GreedyTargetCursor(state=state, hidden=hidden[-1], logits=logits),
        hidden,
        schedule,
    )


def _rank_exact_logits(
    weights: model.TextModelWeights,
    logits: mx.array,
    hidden: mx.array,
    candidate_count: int,
) -> tuple[int, list[int], list[float]]:
    require(
        isinstance(weights.lm_head, vocab.MLXAffineQuantizedMatrix),
        "teacher capture requires the hybrid Q8/BF16 target head",
    )
    token_ids, scores = vocab.exact_candidate_scores(
        weights.lm_head,
        logits,
        hidden,
        candidate_count=candidate_count,
    )
    ranked = sorted(zip(token_ids, scores), key=lambda item: (-item[1], item[0]))
    expected = speculative.greedy_token(logits, hidden, weights.lm_head)
    require(ranked[0][0] == expected, "teacher exact candidate ranking disagrees with target")
    return expected, [item[0] for item in ranked], [item[1] for item in ranked]


def _rank_exact_candidates(
    weights: model.TextModelWeights,
    cursor: speculative.GreedyTargetCursor,
    candidate_count: int,
) -> tuple[int, list[int], list[float]]:
    return _rank_exact_logits(weights, cursor.logits, cursor.hidden, candidate_count)


def _bootstrap_mtp_hidden(
    target_hidden: mx.array,
    token_ids: Sequence[int],
    target_weights: model.TextModelWeights,
    mtp_weights: mtp.MLXMTPWeights,
    *,
    chunk_size: int = 4,
) -> mx.array:
    require(
        target_hidden.shape == (len(token_ids), model.PRODUCTION_CONFIG.hidden_size),
        "teacher MTP replay sequence mismatch",
    )
    state = mtp.initial_state(mtp_weights, mtp.PRODUCTION_CONFIG)
    outputs: list[mx.array] = []
    rows = len(token_ids) - 1
    for offset in range(0, rows, chunk_size):
        end = min(offset + chunk_size, rows)
        embeddings = model.embed_tokens(target_weights.embedding, token_ids[offset + 1 : end + 1])
        result = mtp.prefill_steps(
            embeddings,
            target_hidden[offset:end],
            state,
            mtp_weights,
            mtp.PRODUCTION_CONFIG,
            exact_long_attention=False,
            _validated=True,
        )
        mx.eval(
            result.hidden,
            result.state.keys,
            result.state.values,
            result.selected_experts,
            result.routing_weights,
        )
        outputs.append(result.hidden)
        state = result.state
    require(outputs, "teacher MTP replay produced no rows")
    return mx.concatenate(outputs, axis=0)


def select_candidate_union(
    expected: int,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    candidate_count: int,
) -> list[int]:
    """Prioritize source hard negatives while retaining target distribution support."""
    require(candidate_count > 1, "teacher candidate union is too small")
    require(expected >= 0, "teacher candidate union has an invalid label")
    require(
        len(set(source_ids)) == len(source_ids)
        and len(set(target_ids)) == len(target_ids),
        "teacher candidate source contains duplicates",
    )
    selected = [expected]
    source_quota = max(1, candidate_count // 2)
    for token_id in source_ids:
        if token_id not in selected:
            selected.append(token_id)
        if len(selected) >= source_quota + 1:
            break
    for token_id in target_ids:
        if token_id not in selected:
            selected.append(token_id)
        if len(selected) == candidate_count:
            break
    for token_id in source_ids:
        if token_id not in selected:
            selected.append(token_id)
        if len(selected) == candidate_count:
            break
    require(len(selected) == candidate_count, "teacher candidate union is incomplete")
    return selected


def _union_hard_negative_candidates(
    target_hidden: mx.array,
    token_ids: Sequence[int],
    expected_ids: list[int],
    scored: list[int],
    target_candidate_ids: list[list[int]],
    target_weights: model.TextModelWeights,
    mtp_weights: mtp.MLXMTPWeights,
    candidate_count: int,
) -> tuple[list[list[int]], list[list[float]], int]:
    require(
        isinstance(target_weights.lm_head, vocab.MLXAffineQuantizedMatrix)
        and target_weights.lm_head.reference is not None,
        "teacher hard-negative mining requires the hybrid target head",
    )
    bootstrap_hidden = _bootstrap_mtp_hidden(
        target_hidden,
        token_ids,
        target_weights,
        mtp_weights,
    )
    candidate_ids = [[-1] * candidate_count for _ in expected_ids]
    candidate_logits = [[0.0] * candidate_count for _ in expected_ids]
    bootstrap_matches = 0
    for row, is_scored in enumerate(scored):
        if not is_scored:
            continue
        logits = model.project_lm_head(target_weights.lm_head, bootstrap_hidden[row])
        mx.eval(logits)
        proposal, source_ids, _ = _rank_exact_logits(
            target_weights,
            logits,
            bootstrap_hidden[row],
            candidate_count,
        )
        expected = expected_ids[row]
        bootstrap_matches += int(proposal == expected)
        selected = select_candidate_union(
            expected,
            source_ids,
            target_candidate_ids[row],
            candidate_count,
        )
        rows = target_weights.lm_head.reference.rows(selected)
        scores = vocab.project_bf16_rows_exact(rows, target_hidden[row + 1])
        mx.eval(scores)
        score_values = [float(value) for value in scores.tolist()]
        tail = sorted(
            zip(selected[1:], score_values[1:]),
            key=lambda item: (-item[1], item[0]),
        )
        candidate_ids[row] = [expected, *[item[0] for item in tail]]
        candidate_logits[row] = [score_values[0], *[item[1] for item in tail]]
    return candidate_ids, candidate_logits, bootstrap_matches


def capture_prompt(
    prompt_index: int,
    prompt_ids: Sequence[int],
    prompt_sha256: str,
    prompt: PromptRecord,
    weights: model.TextModelWeights,
    mtp_weights: mtp.MLXMTPWeights,
    tokenizer: TextTokenizer,
    *,
    generated_tokens: int,
    candidate_count: int,
    prefill_chunk: int,
) -> tuple[dict[str, mx.array], dict[str, str], dict[str, object]]:
    cursor, prompt_hidden, schedule = _prefill_target(prompt_ids, weights, prefill_chunk)
    linear = model.start_linear_decode_session(
        weights,
        cursor.state,
        len(prompt_ids) + generated_tokens,
        model.PRODUCTION_CONFIG,
        compile_gdn_layers=True,
        compile_attention_tails=True,
    )
    token_ids = list(prompt_ids)
    generated_hidden: list[mx.array] = []
    rows = len(prompt_ids) + generated_tokens - 1
    expected_ids = [-1] * rows
    scored = [0] * rows
    candidate_ids = [[-1] * candidate_count for _ in range(rows)]
    candidate_logits = [[0.0] * candidate_count for _ in range(rows)]
    generated = []
    current = cursor
    started = time.perf_counter()
    for generated_index in range(generated_tokens):
        token_id = speculative.greedy_token(
            current.logits,
            current.hidden,
            weights.lm_head,
        )
        result = model.forward_linear_session_token(token_id, linear)
        next_cursor = speculative.cursor_from_result(result)
        expected, ranked_ids, ranked_logits = _rank_exact_candidates(
            weights,
            next_cursor,
            candidate_count,
        )
        row = len(prompt_ids) - 1 + generated_index
        expected_ids[row] = expected
        scored[row] = 1
        candidate_ids[row] = ranked_ids
        candidate_logits[row] = ranked_logits
        token_ids.append(token_id)
        generated.append(token_id)
        generated_hidden.append(result.hidden.reshape(1, -1))
        current = next_cursor
        if (generated_index + 1) % 16 == 0 or generated_index + 1 == generated_tokens:
            print(
                "mtp-teacher-token "
                f"prompt={prompt_index} generated={generated_index + 1}/{generated_tokens} "
                f"elapsed_s={time.perf_counter() - started:.3f} "
                f"position={current.state.position}",
                flush=True,
            )
    hidden = mx.concatenate((prompt_hidden, *generated_hidden), axis=0)
    candidate_ids, candidate_logits, bootstrap_matches = _union_hard_negative_candidates(
        hidden,
        token_ids,
        expected_ids,
        scored,
        candidate_ids,
        weights,
        mtp_weights,
        candidate_count,
    )
    arrays = {
        "target_hidden": hidden.astype(mx.bfloat16),
        "token_ids": mx.array(token_ids, dtype=mx.int32),
        "expected_token_ids": mx.array(expected_ids, dtype=mx.int32),
        "candidate_token_ids": mx.array(candidate_ids, dtype=mx.int32),
        "candidate_logits": mx.array(candidate_logits, dtype=mx.float32),
        "scored": mx.array(scored, dtype=mx.int32),
        "prompt_indices": mx.full((rows,), prompt_index, dtype=mx.int32),
        "positions": mx.arange(rows, dtype=mx.int32),
    }
    mx.eval(*arrays.values())
    metadata = {
        "format": FORMAT,
        "prompt_index": str(prompt_index),
        "prompt_sha256": prompt_sha256,
        "prompt_tokens": str(len(prompt_ids)),
        "generated_tokens": str(generated_tokens),
        "candidate_count": str(candidate_count),
        "candidate_policy": CANDIDATE_POLICY,
        "enable_thinking": str(prompt.enable_thinking).lower(),
        "tokenizer_sha256": tokenizer.tokenizer_sha256,
        "template_sha256": tokenizer.template_sha256,
        "source_weight_sha256": nvfp4.EXPECTED_WEIGHT_SHA256,
        "mtp_sidecar_sha256": mtp.EXPECTED_SIDECAR_SHA256,
    }
    validate_trace_arrays(
        arrays,
        metadata,
        prompt_index=prompt_index,
        expected_prompt_sha256=prompt_sha256,
    )
    report = {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated_tokens,
        "rows": rows,
        "scored_rows": sum(scored),
        "prefill_schedule": list(schedule),
        "generated_sha256": canonical_sha256(generated),
        "bootstrap_matches": bootstrap_matches,
        "bootstrap_acceptance": bootstrap_matches / generated_tokens,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return arrays, metadata, report


def save_trace(
    output_dir: Path,
    prompt_index: int,
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
) -> Path:
    final = output_dir / f"prompt-{prompt_index:05d}.safetensors"
    temporary = output_dir / f".prompt-{prompt_index:05d}.part.safetensors"
    require(not final.exists(), f"teacher shard already exists without state: {final}")
    temporary.unlink(missing_ok=True)
    try:
        mx.save_safetensors(str(temporary), arrays, metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        loaded, loaded_metadata = mx.load(str(temporary), return_metadata=True)
        validate_trace_arrays(
            loaded,
            loaded_metadata,
            prompt_index=prompt_index,
            expected_prompt_sha256=metadata["prompt_sha256"],
        )
        os.replace(temporary, final)
        fsync_directory(output_dir)
    finally:
        temporary.unlink(missing_ok=True)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--tokens-per-prompt", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verify-mtp-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.tokens_per_prompt > 0, "tokens per prompt must be positive")
        require(1 <= args.candidate_count <= 256, "candidate count must be between 1 and 256")
        require(args.prefill_chunk > 0, "prefill chunk must be positive")
        require(args.max_prompt_tokens > 0, "maximum prompt length must be positive")
        require(args.max_prompts is None or args.max_prompts > 0, "max prompts must be positive")
        prompts = load_prompts(
            args.prompts_file,
            default_thinking=args.enable_thinking,
        )
        source_path = nvfp4.require_verified_source(args.root)
        mtp_path = mtp.require_verified_mtp_sidecar(
            args.root,
            verify_hash=args.verify_mtp_hash,
        )
        tokenizer = load_text_tokenizer(args.root)
        rendered = [render_prompt(prompt) for prompt in prompts]
        encoded = [tokenizer.encode(value) for value in rendered]
        require(all(encoded), "a rendered teacher prompt encoded to no tokens")
        require(
            max(len(tokens) for tokens in encoded) <= args.max_prompt_tokens,
            "a teacher prompt exceeds the configured token bound",
        )
        require(
            max(len(tokens) for tokens in encoded) + args.tokens_per_prompt <= 262_144,
            "teacher capture exceeds native context",
        )
        prompt_hashes = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in rendered]
        identity = {
            "root": str(args.root.resolve()),
            "source_path": str(source_path.resolve()),
            "source_weight_sha256": nvfp4.EXPECTED_WEIGHT_SHA256,
            "source_state_sha256": sha256_file(args.root / "source-nvfp4-state.json"),
            "mtp_path": str(mtp_path.resolve()),
            "mtp_sidecar_sha256": mtp.EXPECTED_SIDECAR_SHA256,
            "mtp_state_sha256": sha256_file(args.root / "source-mtp-state.json"),
            "tokenizer_sha256": tokenizer.tokenizer_sha256,
            "template_sha256": tokenizer.template_sha256,
            "prompts_sha256": canonical_sha256([prompt.canonical() for prompt in prompts]),
            "prompt_count": len(prompts),
            "tokens_per_prompt": args.tokens_per_prompt,
            "candidate_count": args.candidate_count,
            "candidate_policy": CANDIDATE_POLICY,
            "prefill_chunk": args.prefill_chunk,
            "max_prompt_tokens": args.max_prompt_tokens,
            "repository_revision": repository_revision(),
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            require(state.get("format") == FORMAT, "teacher resume format mismatch")
            require(state.get("identity") == identity, "teacher capture resume identity mismatch")
        else:
            state = {
                "format": FORMAT,
                "status": "running",
                "identity": identity,
                "completed": {},
                "rows": 0,
                "scored_rows": 0,
            }
            atomic_json(state_path, state)
        validate_completed(args.output_dir, state)
        if args.validate_only:
            print(
                "mtp-teacher-validated "
                f"status={state['status']} completed={len(state['completed'])}/{len(prompts)} "
                f"rows={state['rows']} scored={state['scored_rows']}",
                flush=True,
            )
            return 0

        estimated_bytes = sum(
            (len(tokens) + args.tokens_per_prompt) * model.PRODUCTION_CONFIG.hidden_size * 2
            + (len(tokens) + args.tokens_per_prompt - 1) * args.candidate_count * 8
            + 64 * 1024
            for tokens in encoded
        )
        free_bytes = shutil.disk_usage(args.output_dir).free
        require(
            free_bytes >= estimated_bytes + 512 * 2**20,
            "insufficient disk for bounded teacher capture plus safety margin",
        )
        print(
            "mtp-teacher-plan "
            f"prompts={len(prompts)} tokens_per_prompt={args.tokens_per_prompt} "
            f"candidate_count={args.candidate_count} estimated_mib={estimated_bytes / 2**20:.3f} "
            f"free_gib={free_bytes / 2**30:.3f}",
            flush=True,
        )

        mx.set_cache_limit(128 * 2**20)
        weights = model.load_text_model(
            args.root,
            map_embedding=True,
            quantize_lm_head=True,
        )
        mtp_weights = mtp.load_weights(args.root, verify_hash=False)
        processed = 0
        started = time.perf_counter()
        for prompt_index, (prompt, prompt_ids, prompt_sha256) in enumerate(
            zip(prompts, encoded, prompt_hashes)
        ):
            key = str(prompt_index)
            if key in state["completed"]:
                entry = state["completed"][key]
                path = args.output_dir / entry["file"]
                arrays, metadata = mx.load(str(path), return_metadata=True)
                validate_trace_arrays(
                    arrays,
                    metadata,
                    prompt_index=prompt_index,
                    expected_prompt_sha256=prompt_sha256,
                )
                print(
                    f"mtp-teacher-resume prompt={prompt_index} rows={entry['rows']} path={path}",
                    flush=True,
                )
                continue
            if args.max_prompts is not None and processed >= args.max_prompts:
                break
            prompt_started = time.perf_counter()
            arrays, metadata, report = capture_prompt(
                prompt_index,
                prompt_ids,
                prompt_sha256,
                prompt,
                weights,
                mtp_weights,
                tokenizer,
                generated_tokens=args.tokens_per_prompt,
                candidate_count=args.candidate_count,
                prefill_chunk=args.prefill_chunk,
            )
            path = save_trace(args.output_dir, prompt_index, arrays, metadata)
            entry = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "prompt_sha256": prompt_sha256,
                **report,
            }
            state["completed"][key] = entry
            state["rows"] += entry["rows"]
            state["scored_rows"] += entry["scored_rows"]
            state["status"] = (
                "complete" if len(state["completed"]) == len(prompts) else "running"
            )
            atomic_json(state_path, state)
            processed += 1
            print(
                "mtp-teacher-commit "
                f"prompt={prompt_index} rows={entry['rows']} scored={entry['scored_rows']} "
                f"bytes={entry['bytes']} elapsed_s={time.perf_counter() - prompt_started:.3f} "
                f"path={path}",
                flush=True,
            )
            mx.clear_cache()
        validate_completed(args.output_dir, state)
        print(
            "mtp-teacher-done "
            f"status={state['status']} completed={len(state['completed'])}/{len(prompts)} "
            f"rows={state['rows']} scored={state['scored_rows']} "
            f"elapsed_s={time.perf_counter() - started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
        return 0
    except (
        MoEError,
        TokenizerError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"mtp-teacher-error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

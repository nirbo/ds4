#!/usr/bin/env python3
"""Cooperative, resumable background system-prefix warming for Ornith-35."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_cache as cache
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_runtime_coordination as coordination
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT
from ornith35_tokenizer import (
    TokenizerError,
    load_prompt_text_file,
    load_text_tokenizer,
    render_system_prefix,
)


SPEC_FORMAT = "ornith35-background-cache-warm-spec-v1"
STATE_FORMAT = "ornith35-background-cache-warm-state-v1"
JOBS_DIRECTORY = ".warm-jobs"
SPEC_NAME = "spec.json"
STATE_NAME = "state.json"
LOG_NAME = "run.log"
ACTIVE_DIRECTORY = "active"
OWNER_NAME = "owner.json"
CANCEL_NAME = "cancel.request"
DEFAULT_CHECKPOINT_TOKENS = 32_768
DEFAULT_CHECKPOINT_MAX_TOKENS = 65_536
DEFAULT_PROGRESS_TOKENS = 4_096
DEFAULT_POLL_SECONDS = 0.5
DEFAULT_MAX_WAIT_SECONDS = 86_400.0


class WarmError(RuntimeError):
    pass


class WarmPreempted(WarmError):
    def __init__(self, position: int):
        super().__init__(f"foreground requested the model at token {position}")
        self.position = position


class WarmCancelled(WarmError):
    pass


class _Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class WarmSpec:
    root: Path
    repo_root: Path
    system_file: Path
    cache_root: Path
    system_sha256: str
    system_bytes: int
    token_ids: tuple[int, ...]
    token_sha256: str
    identity: cache.CacheIdentity
    final_key: str
    context_profile: str
    mapped_embedding: bool
    quantized_lm_head: bool
    prefill_chunk: int
    checkpoint_tokens: int
    checkpoint_max_tokens: int
    progress_tokens: int
    cache_max_gib: float
    cache_max_bytes: int
    poll_seconds: float
    max_wait_seconds: float
    spec_sha256: str
    job_dir: Path


class StateWriter:
    def __init__(self, spec: WarmSpec):
        self.spec = spec
        existing = _read_json_if_present(spec.job_dir / STATE_NAME)
        if existing is not None:
            require(existing.get("format") == STATE_FORMAT, "warm state format mismatch")
            require(existing.get("job_key") == spec.final_key, "warm state job mismatch")
            require(
                existing.get("spec_sha256") == spec.spec_sha256,
                "warm state spec mismatch",
            )
            require(
                existing.get("system_sha256") == spec.system_sha256,
                "warm state system mismatch",
            )
            require(
                existing.get("token_count") == len(spec.token_ids),
                "warm state token-count mismatch",
            )
            require(
                type(existing.get("created_ns")) is int
                and existing["created_ns"] > 0,
                "warm state creation time is invalid",
            )
        self.created_ns = (
            existing.get("created_ns")
            if isinstance(existing, dict) and type(existing.get("created_ns")) is int
            else time.time_ns()
        )
        self.progress_tokens = (
            existing.get("progress_tokens", 0) if isinstance(existing, dict) else 0
        )
        self.checkpoint_tokens = (
            existing.get("checkpoint_tokens", 0) if isinstance(existing, dict) else 0
        )
        self.checkpoint_key = (
            existing.get("checkpoint_key") if isinstance(existing, dict) else None
        )
        self._validate_progress()

    def _validate_progress(self) -> None:
        require(
            type(self.progress_tokens) is int
            and 0 <= self.progress_tokens <= len(self.spec.token_ids),
            "warm state progress is invalid",
        )
        require(
            type(self.checkpoint_tokens) is int
            and 0 <= self.checkpoint_tokens <= self.progress_tokens,
            "warm state checkpoint is invalid",
        )
        require(
            self.checkpoint_key is None
            or (
                isinstance(self.checkpoint_key, str)
                and len(self.checkpoint_key) == 64
                and all(c in "0123456789abcdef" for c in self.checkpoint_key)
            ),
            "warm state checkpoint key is invalid",
        )
        require(
            (self.checkpoint_tokens == 0) == (self.checkpoint_key is None),
            "warm state checkpoint identity is inconsistent",
        )

    def update(self, status: str, **values: Any) -> dict[str, Any]:
        require(isinstance(status, str) and status, "warm state status is invalid")
        self.progress_tokens = values.pop("progress_tokens", self.progress_tokens)
        self.checkpoint_tokens = values.pop(
            "checkpoint_tokens",
            self.checkpoint_tokens,
        )
        self.checkpoint_key = values.pop("checkpoint_key", self.checkpoint_key)
        self._validate_progress()
        state = {
            "format": STATE_FORMAT,
            "status": status,
            "job_key": self.spec.final_key,
            "spec_sha256": self.spec.spec_sha256,
            "system_sha256": self.spec.system_sha256,
            "token_count": len(self.spec.token_ids),
            "created_ns": self.created_ns,
            "updated_ns": time.time_ns(),
            "pid": os.getpid(),
            "progress_tokens": self.progress_tokens,
            "checkpoint_tokens": self.checkpoint_tokens,
            **values,
        }
        if self.checkpoint_key is not None:
            state["checkpoint_key"] = self.checkpoint_key
        _atomic_write_json(self.spec.job_dir / STATE_NAME, state)
        return state


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _token_sha256(token_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for offset in range(0, len(token_ids), 8192):
        values = token_ids[offset : offset + 8192]
        digest.update(struct.pack(f"<{len(values)}I", *values))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(path.is_dir() and not path.is_symlink(), f"unsafe warm directory: {path}")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _ensure_directory(path.parent)
    temporary = path.parent / f".{path.name}.part-{os.getpid()}-{uuid4().hex}"
    payload = _canonical_json(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"warm JSON is missing or unsafe: {path}")
    require(path.stat().st_size <= 1024 * 1024, f"warm JSON is too large: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WarmError(f"cannot read warm JSON: {path}") from exc
    require(isinstance(value, dict), f"warm JSON is not an object: {path}")
    return value


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _spec_payload(spec: WarmSpec) -> dict[str, Any]:
    return {
        "format": SPEC_FORMAT,
        "root": str(spec.root),
        "repo_root": str(spec.repo_root),
        "system_file": str(spec.system_file),
        "cache_root": str(spec.cache_root),
        "system_sha256": spec.system_sha256,
        "system_bytes": spec.system_bytes,
        "token_sha256": spec.token_sha256,
        "token_count": len(spec.token_ids),
        "identity": asdict(spec.identity),
        "final_key": spec.final_key,
        "context_profile": spec.context_profile,
        "mapped_embedding": spec.mapped_embedding,
        "quantized_lm_head": spec.quantized_lm_head,
        "prefill_chunk": spec.prefill_chunk,
        "checkpoint_tokens": spec.checkpoint_tokens,
        "checkpoint_max_tokens": spec.checkpoint_max_tokens,
        "progress_tokens": spec.progress_tokens,
        "cache_max_gib": spec.cache_max_gib,
        "cache_max_bytes": spec.cache_max_bytes,
        "poll_seconds": spec.poll_seconds,
        "max_wait_seconds": spec.max_wait_seconds,
        "spec_sha256": spec.spec_sha256,
    }


def build_spec(args: argparse.Namespace) -> WarmSpec:
    root = args.root.expanduser().absolute()
    repo_root = args.repo_root.expanduser().absolute()
    system_file = args.system_file.expanduser().absolute()
    cache_root = (
        args.cache_root.expanduser().absolute()
        if args.cache_root is not None
        else root / "cache"
    )
    require(root.is_dir() and not root.is_symlink(), "model root is missing or unsafe")
    require(repo_root.is_dir() and not repo_root.is_symlink(), "repository root is missing or unsafe")
    if cache_root.exists():
        require(cache_root.is_dir() and not cache_root.is_symlink(), "cache root is unsafe")
    prompt_file = load_prompt_text_file(system_file)
    system_sha256 = prompt_file.sha256
    system = prompt_file.text
    generate.prefill_schedule(1, args.prefill_chunk)
    require(args.checkpoint_tokens >= 0, "checkpoint interval must be nonnegative")
    require(args.checkpoint_max_tokens >= 0, "checkpoint ceiling must be nonnegative")
    require(args.progress_tokens > 0, "progress interval must be positive")
    require(args.cache_max_gib > 0.0 and math.isfinite(args.cache_max_gib), "cache budget is invalid")
    require(0.05 <= args.poll_seconds <= 60.0, "background poll interval is invalid")
    require(
        args.max_wait_seconds > 0.0 and math.isfinite(args.max_wait_seconds),
        "background wait timeout must be positive and finite",
    )
    cache_max_bytes = int(args.cache_max_gib * 2**30)
    require(cache_max_bytes > 0, "cache budget is too small")
    selected_context = context.resolve_profile(args.context_profile)
    tokenizer = load_text_tokenizer(root)
    token_ids = tokenizer.encode(render_system_prefix(system))
    require(token_ids, "rendered system prefix produced no tokens")
    require(
        len(token_ids) <= selected_context.max_position_embeddings,
        "system prefix exceeds the selected context profile",
    )
    identity = cache.production_identity(
        root,
        repo_root,
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        chat_template_sha256=tokenizer.template_sha256,
        mapped_embedding=args.mapped_embedding,
        quantized_lm_head=args.quantized_lm_head,
        rope_profile=selected_context.profile_id,
    )
    token_ids = tuple(token_ids)
    token_sha256 = _token_sha256(token_ids)
    final_key = cache.cache_key(token_ids, identity, model.PRODUCTION_CONFIG)
    unsigned = {
        "format": SPEC_FORMAT,
        "root": str(root),
        "repo_root": str(repo_root),
        "system_file": str(system_file),
        "cache_root": str(cache_root),
        "system_sha256": system_sha256,
        "system_bytes": prompt_file.byte_count,
        "token_sha256": token_sha256,
        "token_count": len(token_ids),
        "identity": asdict(identity),
        "final_key": final_key,
        "context_profile": selected_context.profile_id,
        "mapped_embedding": args.mapped_embedding,
        "quantized_lm_head": args.quantized_lm_head,
        "prefill_chunk": args.prefill_chunk,
        "checkpoint_tokens": args.checkpoint_tokens,
        "checkpoint_max_tokens": args.checkpoint_max_tokens,
        "progress_tokens": args.progress_tokens,
        "cache_max_gib": args.cache_max_gib,
        "cache_max_bytes": cache_max_bytes,
        "poll_seconds": args.poll_seconds,
        "max_wait_seconds": args.max_wait_seconds,
    }
    spec_sha256 = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    job_dir = cache_root / JOBS_DIRECTORY / final_key
    return WarmSpec(
        root=root,
        repo_root=repo_root,
        system_file=system_file,
        cache_root=cache_root,
        system_sha256=system_sha256,
        system_bytes=prompt_file.byte_count,
        token_ids=token_ids,
        token_sha256=token_sha256,
        identity=identity,
        final_key=final_key,
        context_profile=selected_context.profile_id,
        mapped_embedding=args.mapped_embedding,
        quantized_lm_head=args.quantized_lm_head,
        prefill_chunk=args.prefill_chunk,
        checkpoint_tokens=args.checkpoint_tokens,
        checkpoint_max_tokens=args.checkpoint_max_tokens,
        progress_tokens=args.progress_tokens,
        cache_max_gib=args.cache_max_gib,
        cache_max_bytes=cache_max_bytes,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        spec_sha256=spec_sha256,
        job_dir=job_dir,
    )


def _write_or_verify_spec(spec: WarmSpec) -> None:
    _ensure_directory(spec.job_dir)
    path = spec.job_dir / SPEC_NAME
    expected = _spec_payload(spec)
    if path.exists():
        require(_read_json(path) == expected, "warm job spec changed for the same final key")
        return
    _atomic_write_json(path, expected)


def _pid_alive(pid: object) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _claim_job(job_dir: Path, token: str, pid: int, phase: str) -> None:
    _ensure_directory(job_dir)
    active = job_dir / ACTIVE_DIRECTORY
    while True:
        try:
            active.mkdir(mode=0o700)
            break
        except FileExistsError:
            require(active.is_dir() and not active.is_symlink(), "warm active claim is unsafe")
            owner = _read_json_if_present(active / OWNER_NAME)
            if owner is None and time.time_ns() - active.stat().st_mtime_ns < 30_000_000_000:
                raise WarmError("warm job claim is still initializing")
            if owner is not None and _pid_alive(owner.get("pid")):
                raise WarmError(
                    f"warm job is already active: pid={owner.get('pid')} phase={owner.get('phase')}"
                )
            hidden = job_dir / f".active.stale-{uuid4().hex}"
            os.rename(active, hidden)
            _fsync_directory(job_dir)
            shutil.rmtree(hidden)
    _atomic_write_json(
        active / OWNER_NAME,
        {"token": token, "pid": pid, "phase": phase, "updated_ns": time.time_ns()},
    )
    _fsync_directory(job_dir)


def _adopt_claim(job_dir: Path, token: str, pid: int, phase: str) -> None:
    owner_path = job_dir / ACTIVE_DIRECTORY / OWNER_NAME
    owner = _read_json(owner_path)
    require(owner.get("token") == token, "warm active claim token mismatch")
    _atomic_write_json(
        owner_path,
        {"token": token, "pid": pid, "phase": phase, "updated_ns": time.time_ns()},
    )


def _release_claim(job_dir: Path, token: str) -> None:
    active = job_dir / ACTIVE_DIRECTORY
    if not os.path.lexists(active):
        return
    require(active.is_dir() and not active.is_symlink(), "warm active claim is unsafe")
    owner = _read_json(active / OWNER_NAME)
    require(owner.get("token") == token, "cannot release another warm worker's claim")
    hidden = job_dir / f".active.done-{uuid4().hex}"
    os.rename(active, hidden)
    _fsync_directory(job_dir)
    shutil.rmtree(hidden)
    _fsync_directory(job_dir)


def _log(event: str, **values: Any) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    detail = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"{timestamp} {event}{' ' if detail else ''}{detail}", flush=True)


def _cancel_requested(spec: WarmSpec) -> bool:
    marker = spec.job_dir / CANCEL_NAME
    if not os.path.lexists(marker):
        return False
    require(marker.is_file() and not marker.is_symlink(), "warm cancellation marker is unsafe")
    return True


def _check_control(spec: WarmSpec, position: int) -> None:
    if _cancel_requested(spec):
        raise WarmCancelled("warm job cancellation requested")
    if coordination.foreground_requested(spec.root):
        raise WarmPreempted(position)


def _release_mlx() -> None:
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _next_checkpoint(spec: WarmSpec, position: int) -> int | None:
    if spec.checkpoint_tokens == 0:
        return None
    candidate = ((position // spec.checkpoint_tokens) + 1) * spec.checkpoint_tokens
    if spec.checkpoint_max_tokens and candidate > spec.checkpoint_max_tokens:
        return None
    if candidate >= len(spec.token_ids):
        return None
    return candidate


def _prune(spec: WarmSpec, protect: tuple[str, ...]) -> cache.CachePruneResult:
    return cache.prune_cache(
        spec.cache_root,
        max_bytes=spec.cache_max_bytes,
        max_entries=cache.DEFAULT_MAX_ENTRIES,
        protect=protect,
    )


def _warm_attempt(spec: WarmSpec, writer: StateWriter) -> Path:
    _check_control(spec, 0)
    lookup = cache.find_longest_prefix(
        spec.cache_root,
        spec.token_ids,
        spec.identity,
        model.PRODUCTION_CONFIG,
        min_suffix_tokens=0,
    )
    checkpoint_position = lookup.token_count
    if lookup.path is not None and checkpoint_position == len(spec.token_ids):
        writer.update(
            "running",
            phase="verify-existing",
            progress_tokens=checkpoint_position,
            checkpoint_tokens=checkpoint_position,
            checkpoint_key=lookup.path.name,
        )
        verify_started = time.perf_counter()
        restored = None
        try:
            restored = cache.load_cache(
                lookup.path,
                spec.identity,
                model.PRODUCTION_CONFIG,
                expected_tokens=spec.token_ids,
            )
        finally:
            restored = None
            _release_mlx()
        _log(
            "warm-cache-already-complete",
            tokens=checkpoint_position,
            verify_s=f"{time.perf_counter() - verify_started:.3f}",
            path=lookup.path,
        )
        return lookup.path

    restored = None
    weights = None
    session = None
    state = None
    saved = None
    next_checkpoint = _next_checkpoint(spec, checkpoint_position)
    last_report = checkpoint_position
    try:
        if lookup.path is not None:
            writer.update(
                "running",
                phase="restore",
                progress_tokens=checkpoint_position,
                checkpoint_tokens=checkpoint_position,
                checkpoint_key=lookup.path.name,
            )
            restore_started = time.perf_counter()
            restored = cache.load_cache(
                lookup.path,
                spec.identity,
                model.PRODUCTION_CONFIG,
                expected_tokens=spec.token_ids[:checkpoint_position],
            )
            state = restored.state
            _log(
                "warm-cache-restored",
                tokens=checkpoint_position,
                elapsed_s=f"{time.perf_counter() - restore_started:.3f}",
                path=lookup.path,
            )
        _check_control(spec, checkpoint_position)

        def model_progress(completed: int, total: int) -> None:
            _check_control(spec, checkpoint_position)
            if completed == 0 or completed == total or completed % 5 == 0:
                writer.update(
                    "running",
                    phase="model-load",
                    loaded_layers=completed,
                    total_layers=total,
                    progress_tokens=checkpoint_position,
                    checkpoint_tokens=checkpoint_position,
                    checkpoint_key=(lookup.path.name if lookup.path is not None else None),
                )
                _log("warm-model-load", layers=f"{completed}/{total}")

        load_started = time.perf_counter()
        weights = model.load_text_model(
            spec.root,
            map_embedding=spec.mapped_embedding,
            quantize_lm_head=spec.quantized_lm_head,
            progress_callback=model_progress,
        )
        _log(
            "warm-model-ready",
            elapsed_s=f"{time.perf_counter() - load_started:.3f}",
            active_gib=f"{mx.get_active_memory() / 2**30:.3f}",
        )
        if state is None:
            state = model.initial_state(
                weights,
                model.PRODUCTION_CONFIG,
                spec.context_profile,
            )
        session = model.start_linear_decode_session(
            weights,
            state,
            len(spec.token_ids),
            model.PRODUCTION_CONFIG,
            compile_gdn_layers=False,
            compile_attention_tails=False,
        )
        restored = None
        state = session.state
        suffix = spec.token_ids[checkpoint_position:]

        def prefill_progress(
            completed: int,
            total: int,
            current_state: model.TextModelState,
        ) -> None:
            nonlocal checkpoint_position, next_checkpoint, last_report
            position = len(spec.token_ids) - total + completed
            _check_control(spec, position)
            if position - last_report >= spec.progress_tokens or position == len(spec.token_ids):
                writer.update(
                    "running",
                    phase="prefill",
                    progress_tokens=position,
                    checkpoint_tokens=checkpoint_position,
                    total_tokens=len(spec.token_ids),
                )
                _log(
                    "warm-prefill-progress",
                    tokens=f"{position}/{len(spec.token_ids)}",
                    active_gib=f"{mx.get_active_memory() / 2**30:.3f}",
                )
                last_report = position
            if next_checkpoint is None or position < next_checkpoint:
                return
            _check_control(spec, position)
            checkpoint_started = time.perf_counter()
            checkpoint_path = cache.save_cache(
                spec.cache_root,
                spec.token_ids[:position],
                current_state,
                spec.identity,
                model.PRODUCTION_CONFIG,
            )
            checkpoint_position = position
            next_checkpoint = _next_checkpoint(spec, checkpoint_position)
            pruned = _prune(spec, (checkpoint_path.name,))
            writer.update(
                "running",
                phase="checkpoint",
                progress_tokens=position,
                checkpoint_tokens=position,
                checkpoint_key=checkpoint_path.name,
                checkpoint_s=time.perf_counter() - checkpoint_started,
                retained_bytes=pruned.retained_bytes,
            )
            _log(
                "warm-checkpoint-saved",
                tokens=position,
                elapsed_s=f"{time.perf_counter() - checkpoint_started:.3f}",
                path=checkpoint_path,
            )

        prefill_started = time.perf_counter()
        state, schedule = generate.prefill_state_prompt(
            suffix,
            session.state,
            weights,
            max_chunk=spec.prefill_chunk,
            linear_session=session,
            exact_long_attention=True,
            progress_callback=prefill_progress,
        )
        prefill_elapsed = time.perf_counter() - prefill_started
        require(state is session.state, "background prefill lost linear-session ownership")
        _check_control(spec, state.position)
        save_started = time.perf_counter()
        saved = cache.save_cache(
            spec.cache_root,
            spec.token_ids,
            state,
            spec.identity,
            model.PRODUCTION_CONFIG,
        )
        save_elapsed = time.perf_counter() - save_started
        writer.update(
            "running",
            phase="verify-final",
            progress_tokens=len(spec.token_ids),
            checkpoint_tokens=len(spec.token_ids),
            checkpoint_key=saved.name,
            final_key=saved.name,
        )
        verified = cache.load_cache(
            saved,
            spec.identity,
            model.PRODUCTION_CONFIG,
            expected_tokens=spec.token_ids,
        )
        require(verified.state.position == len(spec.token_ids), "verified warm position mismatch")
        verify_elapsed = verified.load_timing.total_s
        del verified
        pruned = _prune(spec, (saved.name,))
        _log(
            "warm-cache-complete",
            tokens=len(spec.token_ids),
            prefill_s=f"{prefill_elapsed:.3f}",
            save_s=f"{save_elapsed:.3f}",
            verify_s=f"{verify_elapsed:.3f}",
            schedule=generate.format_prefill_schedule(schedule),
            path=saved,
            retained_gib=f"{pruned.retained_bytes / 2**30:.3f}",
        )
        return saved
    finally:
        restored = None
        state = None
        session = None
        weights = None
        _release_mlx()


def run_worker(spec: WarmSpec, writer: StateWriter) -> int:
    deferred_started = None
    last_wait_reason = None
    while True:
        if _cancel_requested(spec):
            writer.update("cancelled", phase="idle")
            _log("warm-cancelled", job=spec.final_key)
            return 0
        try:
            with coordination.background_lease(spec.root):
                deferred_started = None
                last_wait_reason = None
                writer.update(
                    "running",
                    phase="acquired",
                )
                _log("warm-background-acquired", job=spec.final_key)
                path = _warm_attempt(spec, writer)
            writer.update(
                "complete",
                phase="done",
                progress_tokens=len(spec.token_ids),
                checkpoint_tokens=len(spec.token_ids),
                checkpoint_key=path.name,
                final_key=path.name,
                final_path=str(path),
            )
            return 0
        except coordination.BackgroundDeferred as exc:
            if deferred_started is None:
                deferred_started = time.monotonic()
            waited_s = time.monotonic() - deferred_started
            if waited_s > spec.max_wait_seconds:
                writer.update(
                    "deferred-timeout",
                    phase="idle",
                    waited_s=waited_s,
                )
                _log("warm-deferred-timeout", waited_s=f"{waited_s:.3f}")
                return 75
            reason = str(exc)
            if reason != last_wait_reason:
                writer.update("waiting", phase="foreground", reason=reason)
                _log("warm-waiting", reason=reason)
                last_wait_reason = reason
            time.sleep(spec.poll_seconds)
        except WarmPreempted as exc:
            writer.update(
                "preempted",
                phase="foreground",
                progress_tokens=exc.position,
                reason=str(exc),
            )
            _log("warm-preempted", tokens=exc.position)
            last_wait_reason = None
        except WarmCancelled:
            writer.update("cancelled", phase="cleanup")
            _log("warm-cancelled", job=spec.final_key)
            return 0


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--context-profile",
        choices=tuple(context.SUPPORTED_CONTEXT_PROFILES),
        default=context.NATIVE_PROFILE_ID,
    )
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--checkpoint-tokens", type=int, default=DEFAULT_CHECKPOINT_TOKENS)
    parser.add_argument(
        "--checkpoint-max-tokens",
        type=int,
        default=DEFAULT_CHECKPOINT_MAX_TOKENS,
    )
    parser.add_argument("--progress-tokens", type=int, default=DEFAULT_PROGRESS_TOKENS)
    parser.add_argument("--cache-max-gib", type=float, default=24.0)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-wait-seconds", type=float, default=DEFAULT_MAX_WAIT_SECONDS)
    parser.add_argument(
        "--mapped-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quantized-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="launch a detached low-priority warmer")
    _common_arguments(start)
    run = subparsers.add_parser("run", help="run cooperatively in the current terminal")
    _common_arguments(run)
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    _common_arguments(worker)
    worker.add_argument("--claim-token", required=True, help=argparse.SUPPRESS)
    worker.add_argument("--expected-job-key", required=True, help=argparse.SUPPRESS)
    worker.add_argument("--expected-spec-sha256", required=True, help=argparse.SUPPRESS)
    for name in ("status", "cancel"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        command.add_argument("--cache-root", type=Path)
        command.add_argument("--job-key", required=True)
    return parser.parse_args()


def _job_dir(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().absolute()
    cache_root = (
        args.cache_root.expanduser().absolute()
        if args.cache_root is not None
        else root / "cache"
    )
    require(len(args.job_key) == 64 and all(c in "0123456789abcdef" for c in args.job_key), "invalid job key")
    return cache_root / JOBS_DIRECTORY / args.job_key


def _expected_worker_job_dir(args: argparse.Namespace) -> Path:
    key = args.expected_job_key
    require(len(key) == 64 and all(c in "0123456789abcdef" for c in key), "invalid expected job key")
    root = args.root.expanduser().absolute()
    cache_root = (
        args.cache_root.expanduser().absolute()
        if args.cache_root is not None
        else root / "cache"
    )
    return cache_root / JOBS_DIRECTORY / key


def _fail_worker_bootstrap(job_dir: Path, token: str, exc: Exception) -> None:
    try:
        state_path = job_dir / STATE_NAME
        state = _read_json_if_present(state_path)
        if state is not None:
            state.update(
                status="failed",
                phase="bootstrap",
                error=f"{type(exc).__name__}: {exc}",
                pid=os.getpid(),
                updated_ns=time.time_ns(),
            )
            _atomic_write_json(state_path, state)
    finally:
        _release_claim(job_dir, token)


def _status(args: argparse.Namespace) -> int:
    job_dir = _job_dir(args)
    require(job_dir.is_dir() and not job_dir.is_symlink(), "warm job is missing or unsafe")
    state = _read_json(job_dir / STATE_NAME)
    require(state.get("format") == STATE_FORMAT, "warm state format mismatch")
    require(state.get("job_key") == args.job_key, "warm state job mismatch")
    observed = dict(state)
    active = job_dir / ACTIVE_DIRECTORY
    initializing = False
    if os.path.lexists(active):
        require(active.is_dir() and not active.is_symlink(), "warm active claim is unsafe")
        owner = _read_json_if_present(active / OWNER_NAME)
        initializing = (
            owner is None
            and time.time_ns() - active.stat().st_mtime_ns < 30_000_000_000
        )
    else:
        owner = None
    if owner is None:
        worker = None
    else:
        worker = {
            "alive": _pid_alive(owner.get("pid")),
            "phase": owner.get("phase"),
            "pid": owner.get("pid"),
        }
    observed["worker"] = worker
    active_statuses = {"queued", "starting", "running", "waiting", "preempted"}
    if state.get("status") in active_statuses and initializing:
        observed["observed_status"] = "initializing"
    elif state.get("status") in active_statuses and not (worker and worker["alive"]):
        observed["observed_status"] = "interrupted"
    else:
        observed["observed_status"] = state.get("status")
    print(json.dumps(observed, indent=2, sort_keys=True))
    print(f"warm-status state={job_dir / STATE_NAME} log={job_dir / LOG_NAME}")
    return 0


def _cancel(args: argparse.Namespace) -> int:
    job_dir = _job_dir(args)
    require(job_dir.is_dir() and not job_dir.is_symlink(), "warm job is missing or unsafe")
    marker = job_dir / CANCEL_NAME
    if os.path.lexists(marker):
        require(marker.is_file() and not marker.is_symlink(), "warm cancellation marker is unsafe")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(marker, flags, 0o600)
        try:
            os.write(descriptor, f"requested_ns={time.time_ns()}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(job_dir)
    print(f"warm-cancel-requested job={args.job_key} marker={marker}")
    return 0


def _worker_command(spec: WarmSpec, claim_token: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--root",
        str(spec.root),
        "--repo-root",
        str(spec.repo_root),
        "--system-file",
        str(spec.system_file),
        "--cache-root",
        str(spec.cache_root),
        "--context-profile",
        spec.context_profile,
        "--prefill-chunk",
        str(spec.prefill_chunk),
        "--checkpoint-tokens",
        str(spec.checkpoint_tokens),
        "--checkpoint-max-tokens",
        str(spec.checkpoint_max_tokens),
        "--progress-tokens",
        str(spec.progress_tokens),
        "--cache-max-gib",
        repr(spec.cache_max_gib),
        "--poll-seconds",
        str(spec.poll_seconds),
        "--max-wait-seconds",
        str(spec.max_wait_seconds),
        "--claim-token",
        claim_token,
        "--expected-job-key",
        spec.final_key,
        "--expected-spec-sha256",
        spec.spec_sha256,
    ]
    command.append("--mapped-embedding" if spec.mapped_embedding else "--no-mapped-embedding")
    command.append(
        "--quantized-lm-head" if spec.quantized_lm_head else "--no-quantized-lm-head"
    )
    taskpolicy = Path("/usr/sbin/taskpolicy")
    return [str(taskpolicy), "-b", *command] if taskpolicy.is_file() else command


def _start(spec: WarmSpec) -> int:
    _write_or_verify_spec(spec)
    token = uuid4().hex
    _claim_job(spec.job_dir, token, os.getpid(), "launching")
    writer = None
    try:
        cancel = spec.job_dir / CANCEL_NAME
        cancel.unlink(missing_ok=True)
        writer = StateWriter(spec)
        writer.update("queued", phase="launch")
        log_path = spec.job_dir / LOG_NAME
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                _worker_command(spec, token),
                cwd=spec.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        _adopt_claim(spec.job_dir, token, process.pid, "worker")
    except Exception:
        _release_claim(spec.job_dir, token)
        if writer is not None:
            writer.update("failed", phase="launch", error="worker launch failed")
        raise
    print(
        "warm-started "
        f"job={spec.final_key} pid={process.pid} tokens={len(spec.token_ids)} "
        f"state={spec.job_dir / STATE_NAME} log={log_path}"
    )
    return 0


def _run_claimed(spec: WarmSpec, token: str) -> int:
    writer = None
    try:
        writer = StateWriter(spec)
        try:
            os.nice(10)
        except OSError:
            pass
        writer.update("starting", phase="validate")
        _log(
            "warm-start",
            job=spec.final_key,
            pid=os.getpid(),
            tokens=len(spec.token_ids),
            cache_root=spec.cache_root,
        )
        return run_worker(spec, writer)
    except Exception as exc:
        if writer is not None:
            writer.update("failed", phase="exception", error=f"{type(exc).__name__}: {exc}")
        _log("warm-failed", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        _release_claim(spec.job_dir, token)


def main() -> int:
    args = parse_args()
    worker_job_dir = None
    worker_started = False
    try:
        if args.command == "status":
            return _status(args)
        if args.command == "cancel":
            return _cancel(args)
        if args.command == "_worker":
            worker_job_dir = _expected_worker_job_dir(args)
        spec = build_spec(args)
        if args.command == "_worker":
            require(spec.final_key == args.expected_job_key, "warm source changed before worker start")
            require(
                spec.spec_sha256 == args.expected_spec_sha256,
                "warm job options changed before worker start",
            )
        _write_or_verify_spec(spec)
        if args.command == "start":
            return _start(spec)
        if args.command == "run":
            token = uuid4().hex
            _claim_job(spec.job_dir, token, os.getpid(), "worker")
            try:
                (spec.job_dir / CANCEL_NAME).unlink(missing_ok=True)
                log_path = spec.job_dir / LOG_NAME
                with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                    stdout = _Tee(sys.stdout, log_handle)
                    stderr = _Tee(sys.stderr, log_handle)
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        return _run_claimed(spec, token)
            except Exception:
                _release_claim(spec.job_dir, token)
                raise
        require(args.command == "_worker", "invalid warm command")
        _adopt_claim(spec.job_dir, args.claim_token, os.getpid(), "worker")
        worker_started = True
        return _run_claimed(spec, args.claim_token)
    except (
        WarmError,
        coordination.RuntimeCoordinationError,
        context.ContextError,
        MoEError,
        TokenizerError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        if worker_job_dir is not None and not worker_started:
            try:
                _fail_worker_bootstrap(worker_job_dir, args.claim_token, exc)
            except (OSError, ValueError, MoEError, WarmError) as cleanup_exc:
                print(
                    f"ornith35 cache warm bootstrap cleanup failed: {cleanup_exc}",
                    file=sys.stderr,
                )
        print(f"ornith35 cache warm failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

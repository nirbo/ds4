#!/usr/bin/env python3
"""Process-safe foreground priority for Ornith-35 Metal ownership."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import stat
import time
from typing import Iterator
from uuid import uuid4


RUNTIME_DIRECTORY = ".ornith35-runtime"
FOREGROUND_DIRECTORY = "foreground"
MODEL_LOCK_NAME = "model.lock"


class RuntimeCoordinationError(RuntimeError):
    """Raised when safe model-process coordination cannot be established."""


class BackgroundDeferred(RuntimeCoordinationError):
    """Raised when foreground work owns or requests the model runtime."""


@dataclass(frozen=True)
class ForegroundLease:
    marker: Path
    waited_s: float


@dataclass(frozen=True)
class BackgroundLease:
    lock_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeCoordinationError(message)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeCoordinationError(f"cannot create runtime directory: {path}") from exc
    _require(path.is_dir() and not path.is_symlink(), f"runtime path is unsafe: {path}")


def _runtime_paths(model_root: Path) -> tuple[Path, Path, Path]:
    _require(model_root.is_dir() and not model_root.is_symlink(), "model root is missing or unsafe")
    runtime_root = model_root / RUNTIME_DIRECTORY
    foreground_root = runtime_root / FOREGROUND_DIRECTORY
    _ensure_directory(runtime_root)
    _ensure_directory(foreground_root)
    return runtime_root, foreground_root, runtime_root / MODEL_LOCK_NAME


def _open_lock(path: Path, *, create: bool = True) -> int:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeCoordinationError(f"cannot open runtime lock: {path}") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeCoordinationError(f"runtime lock is not a regular file: {path}")
    return descriptor


def _try_exclusive(descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise RuntimeCoordinationError("runtime lock acquisition failed") from exc


def _publish_foreground_marker(foreground_root: Path) -> tuple[Path, int]:
    token = uuid4().hex
    temporary = foreground_root / f".{os.getpid()}-{token}.part"
    marker = foreground_root / f"{os.getpid()}-{token}.request"
    descriptor = _open_lock(temporary)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        payload = json.dumps(
            {"pid": os.getpid(), "created_ns": time.time_ns()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii") + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.rename(temporary, marker)
        _fsync_directory(foreground_root)
        return marker, descriptor
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        finally:
            os.close(descriptor)
        raise


def active_foreground_requests(model_root: Path) -> int:
    """Count live request markers and remove unlocked crash residue."""
    _, foreground_root, _ = _runtime_paths(model_root)
    active = 0
    removed = False
    for marker in sorted(foreground_root.iterdir(), key=lambda path: path.name):
        if marker.name.startswith("."):
            if not marker.name.endswith(".part"):
                continue
            _require(not marker.is_symlink(), f"unsafe foreground partial: {marker.name}")
            try:
                descriptor = _open_lock(marker, create=False)
            except RuntimeCoordinationError:
                if not marker.exists():
                    continue
                raise
            if not _try_exclusive(descriptor):
                os.close(descriptor)
                continue
            try:
                marker.unlink(missing_ok=True)
                removed = True
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            continue
        _require(
            marker.name.endswith(".request") and not marker.is_symlink(),
            f"unexpected foreground marker: {marker.name}",
        )
        try:
            descriptor = _open_lock(marker, create=False)
        except RuntimeCoordinationError:
            if not marker.exists():
                continue
            raise
        if not _try_exclusive(descriptor):
            active += 1
            os.close(descriptor)
            continue
        try:
            marker.unlink(missing_ok=True)
            removed = True
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    if removed:
        _fsync_directory(foreground_root)
    return active


def foreground_requested(model_root: Path) -> bool:
    return active_foreground_requests(model_root) > 0


@contextmanager
def foreground_lease(
    model_root: Path,
    *,
    timeout_s: float = 60.0,
    poll_s: float = 0.05,
) -> Iterator[ForegroundLease]:
    """Publish priority, wait for background release, then own model residency."""
    _require(timeout_s > 0.0, "foreground lease timeout must be positive")
    _require(0.0 < poll_s <= 1.0, "foreground lease poll interval is invalid")
    _, foreground_root, lock_path = _runtime_paths(model_root)
    marker = None
    marker_descriptor = None
    lock_descriptor = None
    acquired = False
    try:
        marker, marker_descriptor = _publish_foreground_marker(foreground_root)
        lock_descriptor = _open_lock(lock_path)
        started = time.perf_counter()
        while not acquired:
            acquired = _try_exclusive(lock_descriptor)
            if acquired:
                break
            if time.perf_counter() - started >= timeout_s:
                raise RuntimeCoordinationError(
                    f"foreground model lease timed out after {timeout_s:.3f}s"
                )
            time.sleep(poll_s)
        yield ForegroundLease(marker=marker, waited_s=time.perf_counter() - started)
    finally:
        try:
            if lock_descriptor is not None:
                if acquired:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
        finally:
            try:
                if marker is not None:
                    marker.unlink(missing_ok=True)
                    _fsync_directory(foreground_root)
            finally:
                if marker_descriptor is not None:
                    fcntl.flock(marker_descriptor, fcntl.LOCK_UN)
                    os.close(marker_descriptor)


@contextmanager
def background_lease(model_root: Path) -> Iterator[BackgroundLease]:
    """Own model residency only while no foreground work is active or pending."""
    _, _, lock_path = _runtime_paths(model_root)
    if foreground_requested(model_root):
        raise BackgroundDeferred("foreground generation is pending")
    descriptor = _open_lock(lock_path)
    acquired = _try_exclusive(descriptor)
    if not acquired:
        os.close(descriptor)
        raise BackgroundDeferred("model runtime is already occupied")
    try:
        if foreground_requested(model_root):
            raise BackgroundDeferred("foreground generation arrived during background handoff")
        yield BackgroundLease(lock_path=lock_path)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

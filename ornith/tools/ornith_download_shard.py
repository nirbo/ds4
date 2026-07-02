#!/usr/bin/env python3
"""Download one Ornith shard with .part progress logging."""

from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
from pathlib import Path


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(path: Path | None, message: str) -> None:
    line = f"{stamp()} {message}"
    print(line, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


def progress_log(path: Path | None, label: str, done: int, total: int | None, started: float) -> None:
    elapsed = max(time.time() - started, 0.001)
    rate = done / elapsed
    if total:
        pct = done * 100.0 / total
        remain = max(total - done, 0) / rate if rate else 0
        log(path, f"{label} bytes={done}/{total} pct={pct:.1f} rate={rate / 1024**2:.1f}MiB/s eta={remain:.0f}s")
    else:
        log(path, f"{label} bytes={done} rate={rate / 1024**2:.1f}MiB/s")


def copy_stream(src, dst, start: int, total: int | None, log_path: Path | None, label: str, interval: float) -> int:
    done = start
    last = 0.0
    started = time.time()
    while True:
        chunk = src.read(1024 * 1024)
        if not chunk:
            break
        dst.write(chunk)
        done += len(chunk)
        now = time.time()
        if interval <= 0 or now - last >= interval:
            progress_log(log_path, label, done, total, started)
            last = now
    progress_log(log_path, label, done, total, started)
    return done


def download(url: str, dst: Path, expected_size: int | None = None, log_path: Path | None = None, interval: float = 5.0) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    log(log_path, f"download-start url={url} dst={dst} part={part} resume_bytes={offset}")
    started = time.time()

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("", "file"):
        src_path = Path(urllib.request.url2pathname(parsed.path if parsed.scheme else url))
        total = src_path.stat().st_size
        with src_path.open("rb") as src, part.open("ab") as out:
            src.seek(offset)
            size = copy_stream(src, out, offset, total, log_path, "download", interval)
    else:
        req = urllib.request.Request(url)
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        with urllib.request.urlopen(req) as resp:
            code = getattr(resp, "status", 200)
            if offset and code == 200:
                log(log_path, "download-server-ignored-range restarting")
                offset = 0
                mode = "wb"
            else:
                mode = "ab"
            length = resp.headers.get("Content-Length")
            total = offset + int(length) if length else expected_size
            with part.open(mode) as out:
                size = copy_stream(resp, out, offset, total, log_path, "download", interval)

    if expected_size is not None and size != expected_size:
        raise ValueError(f"download size mismatch: got {size}, expected {expected_size}")
    part.replace(dst)
    elapsed = max(time.time() - started, 0.001)
    log(log_path, f"download-done dst={dst} bytes={size} elapsed={elapsed:.2f}s rate={size / elapsed / 1024**2:.1f}MiB/s")
    return {"bytes": size}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--expected-size", type=int)
    p.add_argument("--log", type=Path)
    p.add_argument("--progress-interval", type=float, default=5.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    download(args.url, args.dst, args.expected_size, args.log, args.progress_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

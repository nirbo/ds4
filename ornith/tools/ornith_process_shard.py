#!/usr/bin/env python3
"""Process one already-downloaded Ornith shard."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ornith_ornq_validate import check_offsets, compare_source, read_ornq
from ornith_quantize_safetensors import quantize
from ornith_safetensors_filter import filter_safetensors, load_allowlist


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(path: Path | None, message: str) -> None:
    line = f"{stamp()} {message}"
    print(line, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


def copy_with_progress(src: Path, dst: Path, log_path: Path | None, interval: float) -> int:
    total = src.stat().st_size
    done = 0
    last = 0.0
    started = time.time()
    with src.open("rb") as inp, dst.open("wb") as out:
        while True:
            chunk = inp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            now = time.time()
            if interval <= 0 or now - last >= interval:
                rate = done / max(now - started, 0.001)
                log(log_path, f"copy bytes={done}/{total} pct={done * 100 / total:.1f} rate={rate / 1024**2:.1f}MiB/s")
                last = now
    shutil.copystat(src, dst)
    return done


def validate_ornq(dst: Path, src: Path, log_path: Path | None) -> None:
    header, data_start = read_ornq(dst)
    errors = check_offsets(dst, header, data_start)
    if errors:
        raise ValueError("; ".join(errors))
    for report in compare_source(dst, src, samples=256):
        log(
            log_path,
            f"quant-validate tensor={report['name']} mode={report['quant']} "
            f"samples={report['samples']} mse={report['mse']:.6g} max_abs={report['max_abs']:.6g}",
        )


def process(
    action: str,
    src: Path,
    dst: Path,
    allowlist: Path | None = None,
    log_path: Path | None = None,
    interval: float = 5.0,
    processor: str = "safetensors",
) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        tmp.unlink()
    log(log_path, f"process-start processor={processor} action={action} src={src} dst={dst}")
    started = time.time()
    if processor == "quantize":
        stats = quantize(src, tmp, log_path=log_path)
        validate_ornq(tmp, src, log_path)
    elif action == "copy":
        copied = copy_with_progress(src, tmp, log_path, interval)
        stats = {"bytes": copied, "selected": 0}
    elif action == "filter":
        def progress(name: str, i: int, total: int) -> None:
            log(log_path, f"filter tensor={i}/{total} name={name}")
        stats = filter_safetensors(src, tmp, allowlist=load_allowlist(allowlist), progress=progress)
    else:
        raise ValueError(f"unsupported action: {action}")
    tmp.replace(dst)
    size = dst.stat().st_size
    elapsed = max(time.time() - started, 0.001)
    log(log_path, f"process-done processor={processor} action={action} dst={dst} bytes={size} elapsed={elapsed:.2f}s rate={size / elapsed / 1024**2:.1f}MiB/s")
    return stats


def benchmark(action: str, src: Path, dst: Path, allowlist: Path | None = None, log_path: Path | None = None, interval: float = 5.0, processor: str = "safetensors") -> dict:
    stats = process(action, src, dst, allowlist, log_path, interval, processor)
    dst.unlink()
    log(log_path, f"benchmark-cleanup deleted={dst}")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--action", required=True, choices=("copy", "filter"))
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--allowlist", type=Path)
    p.add_argument("--log", type=Path)
    p.add_argument("--progress-interval", type=float, default=5.0)
    p.add_argument("--processor", choices=("safetensors", "quantize"), default="safetensors")
    p.add_argument("--benchmark-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.benchmark_only:
        benchmark(args.action, args.src, args.dst, args.allowlist, args.log, args.progress_interval, args.processor)
    else:
        process(args.action, args.src, args.dst, args.allowlist, args.log, args.progress_interval, args.processor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

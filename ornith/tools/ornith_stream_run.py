#!/usr/bin/env python3
"""Run the Ornith one-shard prefetch/process loop."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from ornith_download_shard import download, download_hf
from ornith_process_shard import process
from ornith_stream_state import (
    load_json,
    log_event,
    mark_done,
    mark_downloaded,
    mark_failed,
    new_state,
    start_download,
    start_process,
    write_json,
)


def shard_urls(manifest: dict) -> dict[str, str]:
    return {shard["file"]: shard["url"] for shard in manifest["shards"]}


def allowlist_path(root: Path, shard_name: str) -> Path:
    return root / f"{Path(shard_name).stem}.text.allowlist"


def output_path(root: Path, shard_name: str, processor: str = "safetensors") -> Path:
    path = root / shard_name
    return path.with_suffix(".ornq") if processor == "quantize" else path


def raw_path(root: Path, shard_name: str) -> Path:
    return root / shard_name


def start_download_thread(
    state: dict,
    state_path: Path,
    state_lock: threading.Lock,
    urls: dict[str, str],
    raw_dir: Path,
    log: Path,
    interval: float,
    method: str,
    repo: str | None,
) -> threading.Thread | None:
    with state_lock:
        shard = start_download(state)
        write_json(state_path, state)
    if not shard:
        return None

    name = shard["file"]

    def worker() -> None:
        try:
            dst = raw_path(raw_dir, name)
            if method == "hf":
                if not repo:
                    raise ValueError("manifest missing repo for hf download")
                download_hf(repo, name, dst, log_path=log)
            else:
                download(urls[name], dst, log_path=log, interval=interval)
            with state_lock:
                mark_downloaded(state, name, dst)
                write_json(state_path, state)
            log_event(log, f"download-ready shard={name} raw={dst}")
        except Exception as exc:
            with state_lock:
                mark_failed(state, name, f"download: {exc}")
                write_json(state_path, state)
            log_event(log, f"download-failed shard={name} error={exc}")

    thread = threading.Thread(target=worker, name=f"download:{name}", daemon=False)
    thread.start()
    return thread


def run(args: argparse.Namespace) -> int:
    log = args.log or args.state.with_suffix(args.state.suffix + ".log")
    plan = load_json(args.plan)
    manifest = load_json(args.manifest)
    urls = shard_urls(manifest)
    repo = manifest.get("repo")
    state = load_json(args.state) if args.state.exists() else new_state(plan)
    state_lock = threading.Lock()
    write_json(args.state, state)
    log_event(log, f"run-start state={args.state} raw_dir={args.raw_dir} out_dir={args.out_dir}")

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    download_thread = start_download_thread(state, args.state, state_lock, urls, raw_dir, log, args.progress_interval, args.download_method, repo)
    processed = 0
    while True:
        if download_thread:
            download_thread.join()
            download_thread = None

        with state_lock:
            shard = start_process(state)
            write_json(args.state, state)
        if not shard:
            break

        name = shard["file"]
        action = shard["action"]
        src = raw_path(raw_dir, name)
        dst = output_path(out_dir, name, args.processor)
        allowlist = allowlist_path(args.allowlist_dir, name) if args.processor == "safetensors" and action == "filter" else None
        if not args.max_shards or processed + 1 < args.max_shards:
            download_thread = start_download_thread(state, args.state, state_lock, urls, raw_dir, log, args.progress_interval, args.download_method, repo)

        try:
            process(action, src, dst, allowlist=allowlist, log_path=log, interval=args.progress_interval, processor=args.processor)
            with state_lock:
                mark_done(state, name, dst, delete_raw=not args.keep_raw)
                write_json(args.state, state)
            log_event(log, f"process-verified shard={name} output={dst}")
            processed += 1
        except Exception as exc:
            with state_lock:
                mark_failed(state, name, f"process: {exc}")
                write_json(args.state, state)
            log_event(log, f"process-failed shard={name} error={exc}")
            return 1

        if args.max_shards and processed >= args.max_shards:
            log_event(log, f"run-stop max_shards={args.max_shards}")
            break

    if download_thread:
        download_thread.join()
    log_event(log, f"run-done processed={processed}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--allowlist-dir", required=True, type=Path)
    p.add_argument("--log", type=Path)
    p.add_argument("--progress-interval", type=float, default=5.0)
    p.add_argument("--download-method", choices=("urllib", "hf"), default="urllib")
    p.add_argument("--processor", choices=("safetensors", "quantize"), default="safetensors")
    p.add_argument("--max-shards", type=int)
    p.add_argument("--keep-raw", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

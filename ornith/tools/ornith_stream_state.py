#!/usr/bin/env python3
"""Track resumable Ornith one-shard-at-a-time processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def new_state(plan: dict) -> dict:
    return {
        "created_at": int(time.time()),
        "shards": [
            {
                "file": shard["file"],
                "action": shard["action"],
                "text_tensor_count": shard["text_tensor_count"],
                "skipped_tensor_count": shard["skipped_tensor_count"],
                "status": "pending",
                "attempts": 0,
            }
            for shard in plan["shards"]
            if shard["action"] != "skip"
        ],
    }


STATUSES = ("pending", "downloading", "downloaded", "processing", "done", "failed")


def counts(state: dict) -> dict[str, int]:
    out = {name: 0 for name in STATUSES}
    for shard in state["shards"]:
        out[shard["status"]] = out.get(shard["status"], 0) + 1
    return out


def find_shard(state: dict, name: str) -> dict:
    for shard in state["shards"]:
        if shard["file"] == name:
            return shard
    raise ValueError(f"unknown shard: {name}")


def start_download(state: dict) -> dict | None:
    if any(shard["status"] in ("downloading", "downloaded") for shard in state["shards"]):
        return None
    for shard in state["shards"]:
        if shard["status"] in ("pending", "failed"):
            shard["status"] = "downloading"
            shard["download_attempts"] = shard.get("download_attempts", 0) + 1
            shard["download_started_at"] = int(time.time())
            shard.pop("error", None)
            return shard
    return None


def mark_downloaded(state: dict, shard_name: str, raw: Path) -> dict:
    if not raw.is_file():
        raise ValueError(f"missing raw shard: {raw}")
    shard = find_shard(state, shard_name)
    shard["status"] = "downloaded"
    shard["raw"] = str(raw)
    shard["raw_size"] = raw.stat().st_size
    shard["downloaded_at"] = int(time.time())
    return shard


def start_process(state: dict) -> dict | None:
    if any(shard["status"] == "processing" for shard in state["shards"]):
        return None
    for shard in state["shards"]:
        if shard["status"] == "downloaded":
            raw = Path(shard.get("raw", ""))
            if not raw.is_file():
                shard["status"] = "failed"
                shard["error"] = "downloaded raw shard is missing"
                continue
            shard["status"] = "processing"
            shard["process_attempts"] = shard.get("process_attempts", 0) + 1
            shard["process_started_at"] = int(time.time())
            shard.pop("error", None)
            return shard
    return None


def mark_done(state: dict, shard_name: str, output: Path, delete_raw: bool = False) -> dict:
    if not output.is_file():
        raise ValueError(f"missing output: {output}")
    shard = find_shard(state, shard_name)
    shard["status"] = "done"
    shard["output"] = str(output)
    shard["output_size"] = output.stat().st_size
    shard["sha256"] = sha256_file(output)
    shard["done_at"] = int(time.time())
    raw = Path(shard.get("raw", ""))
    if delete_raw and raw.is_file():
        raw.unlink()
        shard["raw_deleted_at"] = int(time.time())
    return shard


def mark_failed(state: dict, shard_name: str, error: str) -> dict:
    shard = find_shard(state, shard_name)
    shard["status"] = "failed"
    shard["error"] = error
    shard["failed_at"] = int(time.time())
    return shard


def verify_done(state: dict) -> list[str]:
    errors = []
    for shard in state["shards"]:
        if shard["status"] != "done":
            continue
        output = Path(shard.get("output", ""))
        if not output.is_file():
            errors.append(f"{shard['file']}: missing output")
            continue
        if output.stat().st_size != shard.get("output_size"):
            errors.append(f"{shard['file']}: size changed")
        if sha256_file(output) != shard.get("sha256"):
            errors.append(f"{shard['file']}: sha256 changed")
    return errors


def print_status(state: dict) -> None:
    c = counts(state)
    for name in STATUSES:
        print(f"{name}: {c[name]}")
    for status in ("downloading", "processing"):
        active = [shard["file"] for shard in state["shards"] if shard["status"] == status]
        if active:
            print(f"{status}: {active[0]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True, type=Path)
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init")
    init.add_argument("--plan", required=True, type=Path)
    sub.add_parser("status")
    sub.add_parser("start-download")
    downloaded = sub.add_parser("downloaded")
    downloaded.add_argument("--shard", required=True)
    downloaded.add_argument("--raw", required=True, type=Path)
    sub.add_parser("start-process")
    sub.add_parser("start-next")
    done = sub.add_parser("done")
    done.add_argument("--shard", required=True)
    done.add_argument("--output", required=True, type=Path)
    done.add_argument("--delete-raw", action="store_true")
    fail = sub.add_parser("fail")
    fail.add_argument("--shard", required=True)
    fail.add_argument("--error", required=True)
    sub.add_parser("verify")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "init":
        write_json(args.state, new_state(load_json(args.plan)))
        print(f"wrote: {args.state}")
        return 0

    state = load_json(args.state)
    if args.cmd == "status":
        print_status(state)
    elif args.cmd == "start-download":
        shard = start_download(state)
        write_json(args.state, state)
        print(shard["file"] if shard else "complete")
    elif args.cmd == "downloaded":
        shard = mark_downloaded(state, args.shard, args.raw)
        write_json(args.state, state)
        print(f"downloaded: {shard['file']}")
    elif args.cmd in ("start-process", "start-next"):
        shard = start_process(state)
        write_json(args.state, state)
        print(shard["file"] if shard else "complete")
    elif args.cmd == "done":
        shard = mark_done(state, args.shard, args.output, args.delete_raw)
        write_json(args.state, state)
        print(f"done: {shard['file']}")
    elif args.cmd == "fail":
        shard = mark_failed(state, args.shard, args.error)
        write_json(args.state, state)
        print(f"failed: {shard['file']}")
    elif args.cmd == "verify":
        errors = verify_done(state)
        for error in errors:
            print(error)
        return 1 if errors else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch small Ornith metadata files.

This refuses model weights and caps every downloaded file. It is intended for
config/tokenizer/planning files only.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


REPO = "deepreinforce-ai/Ornith-1.0-397B"
DEFAULT_FILES = {
    "README.md",
    "LICENSE",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
BLOCKED_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".onnx")


def read_json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def download_file(url: str, max_bytes: int) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"download exceeded {max_bytes} bytes")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=REPO)
    p.add_argument("--out", type=Path, default=Path("../models/Ornith-1.0-397B"))
    p.add_argument("--max-mib", type=int, default=64)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    max_bytes = args.max_mib * 1024 * 1024
    model = read_json_url(f"https://huggingface.co/api/models/{args.repo}")
    names = []
    for sibling in model.get("siblings", []):
        name = sibling.get("rfilename", "")
        if not name or "/" in name:
            continue
        if name.endswith(BLOCKED_SUFFIXES):
            continue
        if name in DEFAULT_FILES:
            names.append(name)

    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)

    for name in sorted(names):
        url = f"https://huggingface.co/{args.repo}/resolve/main/{name}"
        size = content_length(url)
        if size and size > max_bytes:
            print(f"skip {name}: {size} bytes")
            continue
        if args.dry_run:
            print(f"would fetch {name}: {size or 'unknown'} bytes")
            continue
        data = download_file(url, max_bytes)
        (args.out / name).write_bytes(data)
        print(f"fetched {name}: {len(data)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

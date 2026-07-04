#!/usr/bin/env python3
"""Run native REAP observation over prompt-token lines and merge metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import ornith_decode_tokens


ROOT = Path(__file__).resolve().parents[2]
OBSERVE_C = ROOT / "ornith" / "ornith_reap_observe.c"
ORNITH_C = ROOT / "ornith" / "ornith.c"


def ensure_binary(path: Path) -> Path:
    latest = max(OBSERVE_C.stat().st_mtime, ORNITH_C.stat().st_mtime)
    if path.exists() and path.stat().st_mtime >= latest:
        return path
    subprocess.run([
        "cc", "-O2", "-std=c11", "-Iornith",
        "ornith/ornith.c", "ornith/ornith_reap_observe.c", "-lm", "-o", str(path),
    ], cwd=ROOT, check=True)
    return path


def prompt_lines(path: Path, tokenizer: Path | None, text: bool) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if not text:
        return lines
    if tokenizer is None:
        raise ValueError("--text-prompts requires --tokenizer")
    tok = ornith_decode_tokens.load_tokenizer(str(tokenizer))
    return [",".join(str(i) for i in ornith_decode_tokens.encode(line, tok)) for line in lines]


def cap_prompt_tokens(prompts: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return prompts
    out = []
    for prompt in prompts:
        ids = [part for part in prompt.split(",") if part]
        out.append(",".join(ids[:limit]))
    return [prompt for prompt in out if prompt]


def run(args: argparse.Namespace) -> dict:
    binary = ensure_binary(args.binary)
    prompts = cap_prompt_tokens(prompt_lines(args.prompts, args.tokenizer, args.text_prompts), args.max_prompt_tokens)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        prompt_file = tmp / "prompts.tokenids.txt"
        one = tmp / "observations.json"
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        cmd = [
            str(binary), str(args.catalog), str(args.shards), "--prompts", str(prompt_file),
            str(args.max_new), str(args.layers), str(args.expert_top_k),
            str(args.vocab_limit), str(one),
        ]
        print(f"observe prompts={len(prompts)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        return json.loads(one.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--shards", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path, help="line-based token ids, or text with --text-prompts")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--binary", type=Path, default=Path("/tmp/ornith_reap_observe"))
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--text-prompts", action="store_true")
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--max-prompt-tokens", type=int, default=0)
    p.add_argument("--max-new", type=int, default=1)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--expert-top-k", type=int, default=1)
    p.add_argument("--vocab-limit", type=int, default=32)
    args = p.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

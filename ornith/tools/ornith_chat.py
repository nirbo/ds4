#!/usr/bin/env python3
"""One-shot text CLI over the native Ornith generator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import ornith_decode_tokens
import ornith_prompt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT.parent / "models" / "Ornith-1.0-397B"


def default_binary(backend: str) -> Path:
    return Path("/tmp/ornith_generate_metal" if backend == "metal" else "/tmp/ornith_generate")


def ensure_binary(path: Path, backend: str, explicit: bool) -> None:
    if path.exists():
        return
    if explicit:
        raise SystemExit(f"generator binary not found: {path}")
    if backend == "metal":
        cmd = [
            "clang", "-DORNITH_WITH_METAL", "-O2", "-std=c11", "-Iornith",
            "ornith/ornith.c", "ornith/ornith_metal.m", "ornith/ornith_generate.c",
            "-framework", "Foundation", "-framework", "Metal", "-lm", "-o", str(path),
        ]
    else:
        cmd = [
            "cc", "-O2", "-std=c11", "-Iornith",
            "ornith/ornith.c", "ornith/ornith_generate.c", "-lm", "-o", str(path),
        ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def render_prompt(args: argparse.Namespace) -> str:
    if args.messages:
        messages = json.loads(Path(args.messages).read_text(encoding="utf-8"))
        return ornith_prompt.render_text_chat(
            messages,
            add_generation_prompt=True,
            enable_thinking=not args.nothink,
        )
    if args.raw:
        return args.prompt
    return ornith_prompt.render_text_chat(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        enable_thinking=not args.nothink,
    )


def parse_generator_output(text: str) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    scores: list[float] = []
    for line in text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            ids.append(int(parts[1]))
            scores.append(float(parts[2]))
    return ids, scores


def trim_completion(text: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        pos = text.find(stop)
        if pos >= 0:
            text = text[:pos]
    return text


def run(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else model_dir / "tokenizer.json"
    catalog = Path(args.catalog) if args.catalog else model_dir / "ornith-runtime-catalog.tsv"
    shards = Path(args.shards) if args.shards else model_dir / "quant-full" / "out"
    binary = Path(args.binary) if args.binary else default_binary(args.backend)
    ensure_binary(binary, args.backend, args.binary is not None)

    prompt_text = render_prompt(args)
    tokenizer = ornith_decode_tokens.load_tokenizer(str(tokenizer_path))
    prompt_ids = ornith_decode_tokens.encode(prompt_text, tokenizer)
    if not prompt_ids:
        raise SystemExit("empty prompt")

    cmd = [
        str(binary),
        str(catalog),
        str(shards),
        ",".join(str(i) for i in prompt_ids),
        str(args.max_new),
        str(args.layers),
        str(args.expert_top_k),
        str(args.vocab_limit),
    ]
    if args.backend == "metal":
        cmd.append("metal")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    ids, scores = parse_generator_output(proc.stdout)
    decoded = ornith_decode_tokens.decode(ids, ornith_decode_tokens.load_id_to_token(str(tokenizer_path)))
    print(trim_completion(decoded), end="" if decoded.endswith("\n") else "\n")
    if args.show_tokens:
        print(proc.stdout, end="", file=sys.stderr)
        if scores:
            print(f"tokens={len(ids)} best_score={scores[0]:.6g}", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="", help="User prompt text")
    p.add_argument("--messages", help="JSON messages file; overrides prompt")
    p.add_argument("--raw", action="store_true", help="Use prompt text as already-rendered prompt")
    p.add_argument("--nothink", action="store_true", help="Render chat prompt with thinking disabled")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--tokenizer")
    p.add_argument("--catalog")
    p.add_argument("--shards")
    p.add_argument("--binary")
    p.add_argument("--backend", choices=("metal", "cpu"), default="metal")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--layers", type=int, default=60)
    p.add_argument("--expert-top-k", type=int, default=10)
    p.add_argument("--vocab-limit", type=int, default=0)
    p.add_argument("--show-tokens", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

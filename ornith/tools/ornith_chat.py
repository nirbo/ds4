#!/usr/bin/env python3
"""Text CLI over the native Ornith generator."""

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


class TokenCodec:
    def __init__(self, path: Path):
        self.path = path
        self.fallback = ornith_decode_tokens.load_tokenizer(str(path))
        self.id_to_token = ornith_decode_tokens.load_id_to_token(str(path))
        self.fast = None
        try:
            from tokenizers import Tokenizer  # type: ignore
            self.fast = Tokenizer.from_file(str(path))
        except Exception:
            self.fast = None

    def encode(self, text: str) -> list[int]:
        if self.fast is not None:
            return self.fast.encode(text, add_special_tokens=False).ids
        return ornith_decode_tokens.encode(text, self.fallback)

    def decode(self, ids: list[int]) -> str:
        if self.fast is not None:
            return self.fast.decode(ids, skip_special_tokens=False)
        return ornith_decode_tokens.decode(ids, self.id_to_token)


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


def load_messages(path: str | None) -> list[dict]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("--messages must point to a JSON message list")
    return data


def render_prompt(args: argparse.Namespace, messages: list[dict] | None = None) -> str:
    if args.messages:
        return ornith_prompt.render_text_chat(
            messages if messages is not None else load_messages(args.messages),
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


def visible_completion(text: str) -> str:
    text = trim_completion(text)
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].lstrip("\n")
    return text


def generator_config(args: argparse.Namespace) -> tuple[Path, Path, Path, TokenCodec]:
    model_dir = Path(args.model_dir)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else model_dir / "tokenizer.json"
    catalog = Path(args.catalog) if args.catalog else model_dir / "ornith-runtime-catalog.tsv"
    shards = Path(args.shards) if args.shards else model_dir / "quant-full" / "out"
    binary = Path(args.binary) if args.binary else default_binary(args.backend)
    ensure_binary(binary, args.backend, args.binary is not None)
    return binary, catalog, shards, TokenCodec(tokenizer_path)


def generate_once(args: argparse.Namespace, prompt_text: str, config) -> tuple[str, str]:
    binary, catalog, shards, codec = config
    prompt_ids = codec.encode(prompt_text)
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
        raise SystemExit(proc.returncode)

    ids, scores = parse_generator_output(proc.stdout)
    decoded = codec.decode(ids)
    if args.show_tokens:
        print(proc.stdout, end="", file=sys.stderr)
        if scores:
            print(f"tokens={len(ids)} best_score={scores[0]:.6g}", file=sys.stderr)
    return decoded, proc.stdout


def run_interactive(args: argparse.Namespace, config) -> int:
    messages = load_messages(args.messages)
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    while True:
        if not messages or messages[-1].get("role") != "user":
            try:
                text = input("user> ")
            except EOFError:
                print()
                return 0
            if text.strip() in {"/q", "/quit", "exit", "quit"}:
                return 0
            if not text.strip():
                continue
            messages.append({"role": "user", "content": text})
        prompt_text = ornith_prompt.render_text_chat(
            messages,
            add_generation_prompt=True,
            enable_thinking=not args.nothink,
        )
        decoded, _raw = generate_once(args, prompt_text, config)
        visible = visible_completion(decoded)
        print(f"assistant> {visible}", end="" if visible.endswith("\n") else "\n")
        messages.append({"role": "assistant", "content": visible})
    return 0


def run(args: argparse.Namespace) -> int:
    config = generator_config(args)
    if args.interactive:
        return run_interactive(args, config)

    prompt_text = render_prompt(args)
    decoded, _raw = generate_once(args, prompt_text, config)
    text = visible_completion(decoded)
    print(text, end="" if text.endswith("\n") else "\n")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="", help="User prompt text")
    p.add_argument("--messages", help="JSON messages file; overrides prompt")
    p.add_argument("--interactive", "-i", action="store_true")
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

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


def zeros(n: int) -> list[float]:
    return [0.0] * n


def merge_layer(dst: dict, src: dict) -> None:
    n = len(src["expert_frequency"])
    if not dst:
        dst.update({
            "total_tokens": 0,
            "expert_frequency": [0] * n,
            "weighted_expert_frequency_sum": zeros(n),
            "ean_weighted_sum": zeros(n),
            "reap_weighted_sum": zeros(n),
            "max_activations": zeros(n),
        })
    for i in range(n):
        freq = int(src["expert_frequency"][i])
        dst["expert_frequency"][i] += freq
        dst["weighted_expert_frequency_sum"][i] += float(src["weighted_expert_frequency_sum"][i])
        dst["ean_weighted_sum"][i] += float(src["ean_mean"][i]) * freq
        dst["reap_weighted_sum"][i] += float(src["reap"][i]) * freq
        dst["max_activations"][i] = max(dst["max_activations"][i], float(src["max_activations"][i]))
    dst["total_tokens"] += int(src["total_tokens"])


def finalize(report: dict) -> dict:
    out_layers = {}
    for layer, data in report["layers"].items():
        freq = data["expert_frequency"]
        n = len(freq)
        ean = zeros(n)
        reap = zeros(n)
        for i, count in enumerate(freq):
            if count:
                ean[i] = data["ean_weighted_sum"][i] / count
                reap[i] = data["reap_weighted_sum"][i] / count
        out_layers[layer] = {
            "total_tokens": data["total_tokens"],
            "expert_frequency": freq,
            "weighted_expert_frequency_sum": data["weighted_expert_frequency_sum"],
            "ean_mean": ean,
            "reap": reap,
            "max_activations": data["max_activations"],
        }
    return {"format": "ornith-reap-observer-v1", "layers": out_layers}


def run(args: argparse.Namespace) -> dict:
    binary = ensure_binary(args.binary)
    prompts = prompt_lines(args.prompts, args.tokenizer, args.text_prompts)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    merged: dict = {"layers": {}}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, prompt in enumerate(prompts):
            one = tmp / f"obs-{i}.json"
            cmd = [
                str(binary), str(args.catalog), str(args.shards), prompt,
                str(args.max_new), str(args.layers), str(args.expert_top_k),
                str(args.vocab_limit), str(one),
            ]
            print(f"observe {i + 1}/{len(prompts)} tokens={prompt}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
            data = json.loads(one.read_text(encoding="utf-8"))
            for layer, src in data["layers"].items():
                merge_layer(merged["layers"].setdefault(layer, {}), src)
    return finalize(merged)


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

#!/usr/bin/env python3
"""Build a deterministic, coding-heavy calibration corpus for Ornith."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


TASKS = [
    "Review this code for correctness, numerical drift, and memory-lifetime bugs. Give concrete fixes.",
    "Explain this implementation, identify its invariants, and propose one safe performance optimization.",
    "Write focused tests that would catch subtle regressions in this code.",
    "Find concurrency, resumption, and partial-failure risks in this implementation.",
    "Analyze the data layout and point out any shape, stride, alignment, or quantization hazards.",
]

GENERAL = [
    "Implement a resumable producer-consumer pipeline that downloads one shard ahead, verifies output, and deletes input only after an atomic state commit.",
    "Debug a Metal compute pipeline whose GPU implementation is slower than its CPU reference. Describe the measurements you would take.",
    "Explain how per-expert activation importance should be collected for gate/up and down projections in a routed MoE.",
    "Design a property-based test for a binary quantized tensor format with blockwise scales.",
    "Review a lock-free ring buffer used by an inference server and identify ABA and visibility hazards.",
    "Write a Python parser for a binary file with a JSON header, checked contiguous offsets, and memory-mapped payloads.",
    "Compare uniform expert pruning, activation-aware pruning, and expert merging for a coding-focused MoE.",
    "Derive the memory needed for a 300B MoE under mixed 2-bit routed weights and 8-bit dense weights.",
    "Implement top-k softmax routing with stable normalization and explain numerical corner cases.",
    "Plan a zero-copy CPU/GPU inference path on Apple silicon while preserving a scalar correctness reference.",
    "Explain why quantizing router weights can cause much larger behavior changes than the same relative error in a routed expert.",
    "Create a benchmark plan that distinguishes quantization loss from pruning loss using NLL, KL divergence, and task continuations.",
    "Fix a C program that can delete its only raw input before its durable state file records the verified output.",
    "Design an SSD expert cache for a sparse MoE that minimizes page faults and CPU/GPU synchronization.",
    "Review an autoregressive decoder with recurrent linear-attention state and periodic full-attention KV state.",
    "Implement a deterministic weighted rate-distortion optimizer that chooses prune, 2-bit, 3-bit, or 4-bit per expert.",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file from the repository",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a focused test command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
]


def chunks(text: str, size: int = 6000, overlap: int = 400) -> list[str]:
    text = text.strip()
    out = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            newline = text.rfind("\n", pos + size // 2, end)
            if newline > pos:
                end = newline
        if text[pos:end].strip():
            out.append(text[pos:end].strip())
        if end == len(text):
            break
        pos = max(pos + 1, end - overlap)
    return out


def record(category: str, source: str, prompt: str, context: str | None = None, tools: bool = False) -> dict:
    content = prompt if context is None else f"{prompt}\n\nSource: {source}\n\n```\n{context}\n```"
    value = {
        "category": category,
        "source": source,
        "messages": [
            {"role": "system", "content": "You are Ornith, an agentic coding model. Be precise, verify assumptions, and preserve technical detail."},
            {"role": "user", "content": content},
        ],
    }
    if tools:
        value["tools"] = TOOLS
    return value


def build(root: Path, limit: int, seed: int) -> list[dict]:
    candidates = []
    paths = sorted(
        list((root / "ornith").glob("*.c"))
        + list((root / "ornith").glob("*.h"))
        + list((root / "ornith").glob("*.m"))
        + list((root / "ornith" / "tools").glob("*.py"))
        + list((root / "tests").glob("ornith_*"))
        + [root / "ORNITH.md", root / "AGENTS.md"]
    )
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        for index, chunk in enumerate(chunks(text)):
            for task_index, task in enumerate(TASKS):
                candidates.append(record("source", f"{rel}:{index}", task, chunk, tools=(task_index == 3)))
    for repetition in range(24):
        for index, prompt in enumerate(GENERAL):
            suffix = " Return a concise implementation sketch." if repetition % 3 == 0 else " State the failure modes explicitly."
            candidates.append(record("coding", f"general:{index}:{repetition}", prompt + suffix, tools=(repetition % 4 == 0)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    unique = []
    seen = set()
    for item in candidates:
        key = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) == limit:
            break
    if len(unique) < limit:
        raise ValueError(f"only generated {len(unique)} unique prompts, requested {limit}")
    return unique


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=768)
    p.add_argument("--seed", type=int, default=397)
    args = p.parse_args()
    values = build(args.root, args.limit, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
    counts = Counter(v["category"] for v in values)
    manifest = {
        "format": "ornith-calibration-dataset-v1",
        "records": len(values),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "categories": dict(sorted(counts.items())),
        "output": str(args.out),
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(values)} sha256={manifest['sha256']} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

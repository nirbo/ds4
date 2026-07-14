#!/usr/bin/env python3
"""Build a deterministic balanced prompt set for MTP teacher capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-teacher-prompts-v1"


def balanced_prompts(
    corpus: dict[str, list[str]], tokenizer, per_category: int, max_prompt_tokens: int
) -> list[dict]:
    selected = {}
    for category, rows in sorted(corpus.items()):
        eligible = []
        for text in rows:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            if 0 < token_count <= max_prompt_tokens:
                eligible.append((token_count, hashlib.sha256(text.encode()).hexdigest(), text))
        eligible.sort()
        require(
            len(eligible) >= per_category,
            f"category {category} has only {len(eligible)} eligible prompts",
        )
        selected[category] = eligible[:per_category]
    output = []
    for offset in range(per_category):
        for category in sorted(selected):
            token_count, _, prompt = selected[category][offset]
            output.append({"category": category, "prompt": prompt, "prompt_tokens": token_count})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.per_category > 0 and args.max_prompt_tokens > 0, "invalid prompt limits")
        corpus = load_json(args.corpus)
        require(
            isinstance(corpus, dict)
            and corpus
            and all(isinstance(rows, list) and all(isinstance(row, str) for row in rows) for rows in corpus.values()),
            "invalid prompt corpus",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        prompts = balanced_prompts(corpus, tokenizer, args.per_category, args.max_prompt_tokens)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".part")
        with temporary.open("w") as handle:
            for row in prompts:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(args.output)
        report = {
            "format": FORMAT,
            "status": "complete",
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": sha256_file(args.corpus),
            "tokenizer": str(args.tokenizer.resolve()),
            "tokenizer_sha256": sha256_file(args.tokenizer / "tokenizer.json"),
            "tool_sha256": sha256_file(Path(__file__)),
            "per_category": args.per_category,
            "max_prompt_tokens": args.max_prompt_tokens,
            "categories": sorted(corpus),
            "prompts": len(prompts),
            "minimum_prompt_tokens": min(row["prompt_tokens"] for row in prompts),
            "maximum_prompt_tokens_observed": max(row["prompt_tokens"] for row in prompts),
            "mean_prompt_tokens": sum(row["prompt_tokens"] for row in prompts) / len(prompts),
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
        }
        atomic_json(args.report, report)
        print("mtp-teacher-prompts " + json.dumps(report, separators=(",", ":")), flush=True)
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"nemotron MTP teacher prompts error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

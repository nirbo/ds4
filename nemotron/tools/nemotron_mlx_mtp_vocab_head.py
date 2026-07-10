#!/usr/bin/env python3
"""Build an exact BF16 reduced-vocabulary head used only for MTP drafts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-mlx-mtp-vocab-head-v1"
SELECTIONS = ("balanced-frequency", "raw-frequency")
STORAGE_MODES = ("copied-bf16", "shared-target-bf16")


def load_corpus(path: Path) -> dict[str, list[str]]:
    data = load_json(path)
    require(isinstance(data, dict) and data, "vocabulary corpus must be a non-empty object")
    result = {}
    for category, rows in data.items():
        require(
            isinstance(category, str)
            and category
            and isinstance(rows, list)
            and len(rows) >= 2
            and all(isinstance(row, str) and row for row in rows),
            f"invalid vocabulary corpus category: {category!r}",
        )
        result[category] = rows
    return result


def added_token_ids(tokenizer_path: Path, vocab_size: int) -> set[int]:
    tokenizer = load_json(tokenizer_path)
    result = {
        token["id"]
        for token in tokenizer.get("added_tokens", [])
        if isinstance(token, dict) and isinstance(token.get("id"), int)
    }
    require(result and all(0 <= token_id < vocab_size for token_id in result), "invalid added tokens")
    return result


def category_counts(
    corpus: dict[str, list[str]],
    tokenizer,
) -> tuple[dict[str, Counter[int]], dict[str, Counter[int]]]:
    training = {}
    heldout = {}
    for category, rows in sorted(corpus.items()):
        training_count: Counter[int] = Counter()
        heldout_count: Counter[int] = Counter()
        for index, text in enumerate(rows):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            require(token_ids, f"vocabulary corpus row encoded to no tokens: {category}/{index}")
            (training_count if index % 2 == 0 else heldout_count).update(token_ids)
        require(training_count and heldout_count, f"empty vocabulary split: {category}")
        training[category] = training_count
        heldout[category] = heldout_count
    return training, heldout


def rank_tokens(training: dict[str, Counter[int]], selection: str) -> list[int]:
    require(selection in SELECTIONS, f"unsupported vocabulary selection: {selection}")
    raw: Counter[int] = Counter()
    category_hits: Counter[int] = Counter()
    balanced: dict[int, float] = defaultdict(float)
    for counts in training.values():
        raw.update(counts)
        total = sum(counts.values())
        for token_id, count in counts.items():
            category_hits[token_id] += 1
            balanced[token_id] += count / total
    if selection == "raw-frequency":
        return sorted(raw, key=lambda token_id: (-raw[token_id], token_id))
    return sorted(
        balanced,
        key=lambda token_id: (
            -balanced[token_id],
            -category_hits[token_id],
            -raw[token_id],
            token_id,
        ),
    )


def select_token_ids(
    vocab_size: int,
    budget: int,
    required_ids: set[int],
    ranked_ids: list[int],
) -> list[int]:
    require(0 < len(required_ids) <= budget <= vocab_size, "invalid vocabulary budget")
    selected = set(required_ids)
    for token_id in ranked_ids:
        require(0 <= token_id < vocab_size, f"ranked token is out of range: {token_id}")
        if len(selected) >= budget:
            break
        selected.add(token_id)
    for token_id in range(vocab_size):
        if len(selected) >= budget:
            break
        selected.add(token_id)
    require(len(selected) == budget, "failed to fill reduced vocabulary budget")
    return sorted(selected)


def coverage(counts: dict[str, Counter[int]], selected: set[int]) -> dict:
    categories = {}
    covered_total = 0
    token_total = 0
    for category, values in sorted(counts.items()):
        total = sum(values.values())
        covered = sum(count for token_id, count in values.items() if token_id in selected)
        categories[category] = {
            "tokens": total,
            "covered_tokens": covered,
            "coverage": covered / total,
        }
        covered_total += covered
        token_total += total
    return {
        "tokens": token_total,
        "covered_tokens": covered_total,
        "coverage": covered_total / token_total,
        "minimum_category_coverage": min(value["coverage"] for value in categories.values()),
        "categories": categories,
    }


def locate_lm_head(model_dir: Path) -> tuple[Path, mx.array]:
    index = load_json(model_dir / "model.safetensors.index.json")
    shard_name = index.get("weight_map", {}).get("lm_head.weight")
    require(isinstance(shard_name, str), "target runtime has no lm_head.weight")
    shard_path = model_dir / shard_name
    tensors = mx.load(str(shard_path))
    require("lm_head.weight" in tensors, "lm_head.weight is absent from its shard")
    weight = tensors["lm_head.weight"]
    require(weight.dtype == mx.bfloat16 and weight.ndim == 2, "MTP source head must be BF16")
    return shard_path, weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--selection", choices=SELECTIONS, default="balanced-frequency")
    parser.add_argument("--storage", choices=STORAGE_MODES, default="copied-bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        report_path = args.model_dir / "nemotron_mlx_pack_report.json"
        source_report = load_json(report_path)
        require(
            source_report.get("format") == "nemotron-mlx-runtime-v1"
            and source_report.get("status") == "complete",
            "source target runtime is incomplete",
        )
        require(args.model_dir.resolve() != args.output_dir.resolve(), "source and output are identical")
        tokenizer_dir = args.tokenizer_dir or args.model_dir
        tokenizer_path = tokenizer_dir / "tokenizer.json"
        tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        corpus = load_corpus(args.corpus)
        training, heldout = category_counts(corpus, tokenizer)
        source_path, original = locate_lm_head(args.model_dir)
        vocab_size, hidden_size = original.shape
        require(tokenizer.vocab_size == vocab_size, "tokenizer/head vocabulary mismatch")
        required_ids = added_token_ids(tokenizer_path, vocab_size)
        ranked_ids = rank_tokens(training, args.selection)
        target_token_ids = select_token_ids(vocab_size, args.budget, required_ids, ranked_ids)
        selected = set(target_token_ids)
        evidence = {
            "training": coverage(training, selected),
            "heldout": coverage(heldout, selected),
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "materialize.log")
        source_hash = sha256_file(source_path)
        operation_log.write(
            f"mtp-vocab-head-start budget={args.budget} selection={args.selection} storage={args.storage} "
            f"source={source_path} source_sha256={source_hash}"
        )
        id_array = mx.array(target_token_ids, dtype=mx.int32)
        output = {"target_token_ids": id_array}
        if args.storage == "copied-bf16":
            output["weight"] = original[id_array]
        mx.eval(*output.values())
        output_path = args.output_dir / "lm_head.safetensors"
        temporary = output_path.with_name(output_path.stem + ".part" + output_path.suffix)
        temporary.unlink(missing_ok=True)
        mx.save_safetensors(
            str(temporary),
            output,
            metadata={
                "format": FORMAT,
                "selection": args.selection,
                "storage": args.storage,
            },
        )
        temporary.replace(output_path)
        loaded, metadata = mx.load(str(output_path), return_metadata=True)
        require(set(loaded) == set(output), "reduced MTP head tensor mismatch")
        require(
            metadata
            == {
                "format": FORMAT,
                "selection": args.selection,
                "storage": args.storage,
            },
            "reduced MTP head metadata mismatch",
        )
        require(
            loaded["target_token_ids"].dtype == mx.int32
            and loaded["target_token_ids"].tolist() == target_token_ids,
            "reduced MTP head token map mismatch",
        )
        if args.storage == "copied-bf16":
            require(
                loaded["weight"].dtype == mx.bfloat16
                and loaded["weight"].shape == (args.budget, hidden_size),
                "reduced MTP head weight mismatch",
            )
            exact = mx.all(loaded["weight"] == original[loaded["target_token_ids"]])
            mx.eval(exact)
            require(bool(exact), "reduced MTP head changed retained BF16 rows")
        payload_bytes = sum(value.nbytes for value in loaded.values())
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_report["source_revision"],
            "source_model_dir": str(args.model_dir.resolve()),
            "source_report_sha256": sha256_file(report_path),
            "source_shard": source_path.name,
            "source_shard_sha256": source_hash,
            "source_shape": list(original.shape),
            "source_dtype": str(original.dtype),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "tokenizer_config_sha256": sha256_file(tokenizer_config_path),
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": sha256_file(args.corpus),
            "split": "even-training-odd-heldout-per-category",
            "selection": args.selection,
            "storage": args.storage,
            "budget": args.budget,
            "required_added_tokens": len(required_ids),
            "ranked_training_tokens": len(ranked_ids),
            "coverage": evidence,
            "payload_bytes": payload_bytes,
            "payload_gib": payload_bytes / 2**30,
            "artifact": output_path.name,
            "artifact_sha256": sha256_file(output_path),
        }
        atomic_json(args.output_dir / "nemotron_mtp_vocab_head_report.json", report)
        operation_log.write(
            f"mtp-vocab-head-complete payload_gib={report['payload_gib']:.6f} "
            f"heldout_coverage={evidence['heldout']['coverage']:.6f} "
            f"artifact_sha256={report['artifact_sha256']}"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError) as exc:
        if operation_log is not None:
            operation_log.write(f"mtp-vocab-head-failed error={exc}")
        print(f"nemotron MTP vocabulary head error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

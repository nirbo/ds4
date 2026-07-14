#!/usr/bin/env python3
"""Capture resumable exact target trajectories with resident MTP speculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_resident import ResidentModel, preflight
from nemotron_mlx_speculative import matching_draft_prefix, timed_draft
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-teacher-capture-v2"
DEFAULT_PROMPTS = [
    "Complete this Python function:\n\ndef binary_search(values, target):\n",
    "Write a Rust function that returns the longest common prefix of a list of strings.\n",
    "Implement an LRU cache in Python with O(1) get and put operations.\n",
    "Explain and fix a race condition in a Go worker pool.\n",
]


def prompt_digest(prompts: list[str]) -> str:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            value = json.loads(line)
            require(isinstance(value, dict), f"prompt JSONL row {line_number} is not an object")
            prompt = value.get("prompt", value.get("text"))
            require(isinstance(prompt, str), f"prompt JSONL row {line_number} has no prompt")
        else:
            prompt = line
        require(prompt.strip(), f"prompt row {line_number} is empty")
        prompts.append(prompt)
    require(prompts, "prompt file contains no prompts")
    return prompts


def mtp_head_report_path(head_dir: Path) -> Path:
    candidates = [
        head_dir / "nemotron_mtp_head_report.json",
        head_dir / "nemotron_mtp_vocab_head_report.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    require(len(existing) == 1, "alternate MTP head must contain exactly one supported report")
    return existing[0]


def authoritative_cycle_rows(
    current_hidden: mx.array,
    base_token: int,
    verified_hidden: mx.array,
    verified_tokens: list[int],
    accepted_drafts: int,
    row_limit: int,
) -> list[tuple[mx.array, int, int]]:
    """Return only rows whose conditioning prefix is on the target trajectory."""

    require(row_limit >= 0, "teacher row limit cannot be negative")
    require(
        verified_hidden.ndim == 2
        and verified_hidden.shape[0] == len(verified_tokens)
        and len(verified_tokens) >= accepted_drafts + 1,
        "verifier rows do not cover the accepted trajectory",
    )
    require(0 <= accepted_drafts < len(verified_tokens), "invalid accepted draft count")
    rows = [(current_hidden, base_token, verified_tokens[0])]
    for index in range(accepted_drafts):
        rows.append(
            (verified_hidden[index], verified_tokens[index], verified_tokens[index + 1])
        )
    return rows[:row_limit]


def validate_trace_arrays(
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
    prompt_index: int,
    expected_rows: int,
) -> None:
    required = {
        "target_hidden",
        "accepted_token_ids",
        "expected_token_ids",
        "prompt_indices",
        "scored",
    }
    require(set(arrays) == required, "teacher trace tensor set mismatch")
    require(metadata.get("format") == FORMAT, "teacher trace format mismatch")
    require(metadata.get("prompt_index") == str(prompt_index), "teacher trace prompt mismatch")
    require(
        arrays["target_hidden"].dtype == mx.bfloat16
        and arrays["target_hidden"].ndim == 2
        and arrays["target_hidden"].shape == (expected_rows, 4096),
        "teacher hidden tensor mismatch",
    )
    for name in required - {"target_hidden"}:
        require(
            arrays[name].dtype == mx.int32 and arrays[name].shape == (expected_rows,),
            f"teacher trace tensor mismatch: {name}",
        )
    require(
        arrays["prompt_indices"].tolist() == [prompt_index] * expected_rows,
        "teacher prompt indices are inconsistent",
    )
    require(arrays["scored"].tolist() == [1] * expected_rows, "teacher scored mask mismatch")
    accepted = arrays["accepted_token_ids"].tolist()
    expected = arrays["expected_token_ids"].tolist()
    require(accepted[1:] == expected[:-1], "teacher trace is not a contiguous target trajectory")


def validate_completed(output_dir: Path, state: dict) -> None:
    completed = state.get("completed")
    require(isinstance(completed, dict), "teacher capture state has no completed map")
    total_rows = 0
    for key, entry in sorted(completed.items(), key=lambda item: int(item[0])):
        prompt_index = int(key)
        require(isinstance(entry, dict), f"teacher state entry {key} is malformed")
        shard = output_dir / entry.get("file", "")
        require(shard.is_file(), f"completed teacher shard is missing: {shard}")
        require(shard.stat().st_size == entry.get("bytes"), f"teacher shard size mismatch: {shard}")
        require(sha256_file(shard) == entry.get("sha256"), f"teacher shard hash mismatch: {shard}")
        rows = entry.get("rows")
        require(isinstance(rows, int) and rows > 0, f"teacher shard row count is invalid: {shard}")
        arrays, metadata = mx.load(str(shard), return_metadata=True)
        validate_trace_arrays(arrays, metadata, prompt_index, rows)
        total_rows += rows
    require(total_rows == state.get("rows"), "teacher state total row count mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--mtp-sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--tokens-per-prompt", type=int, default=128)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--margin-gib", type=float, default=0.5)
    parser.add_argument("--cache-limit-mib", type=int, default=128)
    parser.add_argument("--paged-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    parser.add_argument("--compile-mamba", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.tokens_per_prompt > 0, "tokens per prompt must be positive")
        require(args.max_prompts is None or args.max_prompts > 0, "max prompts must be positive")
        require(args.cache_limit_mib >= 0, "cache limit cannot be negative")
        prompts = load_prompts(args.prompts_file)
        model_report = args.model_dir / "nemotron_mlx_pack_report.json"
        sidecar_report = args.mtp_sidecar / "nemotron_mtp_pack_report.json"
        head_report = mtp_head_report_path(args.mtp_lm_head)
        identity = {
            "format": FORMAT,
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(model_report),
            "mtp_sidecar": str(args.mtp_sidecar.resolve()),
            "mtp_sidecar_report_sha256": sha256_file(sidecar_report),
            "mtp_lm_head": str(args.mtp_lm_head.resolve()),
            "mtp_lm_head_report_sha256": sha256_file(head_report),
            "prompts_sha256": prompt_digest(prompts),
            "prompt_count": len(prompts),
            "tokens_per_prompt": args.tokens_per_prompt,
            "paged_embeddings": args.paged_embeddings,
            "embedding_cache_rows": args.embedding_cache_rows,
            "compile_mamba": args.compile_mamba,
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            require(state.get("identity") == identity, "teacher capture resume identity mismatch")
            require(isinstance(state.get("completed"), dict), "teacher capture state is malformed")
        else:
            state = {
                "format": FORMAT,
                "status": "running",
                "identity": identity,
                "completed": {},
                "rows": 0,
            }
            atomic_json(state_path, state)

        if args.validate_only:
            validate_completed(args.output_dir, state)
            print(
                f"mtp-teacher-validated status={state['status']} "
                f"completed={len(state['completed'])}/{len(prompts)} rows={state['rows']}",
                flush=True,
            )
            return 0

        result = preflight(
            args.model_dir,
            args.margin_gib,
            args.mtp_sidecar,
            args.mtp_lm_head,
            paged_embeddings=args.paged_embeddings,
        )
        print("mtp-teacher-preflight " + json.dumps(result, separators=(",", ":")), flush=True)
        require(result["safe_to_attempt"], "Metal wired cap is too low for target plus MTP sidecar")
        mx.set_wired_limit(result["effective_cap_bytes"])
        mx.set_cache_limit(args.cache_limit_mib * 2**20)
        started = time.perf_counter()
        model = ResidentModel(
            args.model_dir,
            args.mtp_sidecar,
            args.mtp_lm_head,
            paged_embeddings=args.paged_embeddings,
            embedding_cache_rows=args.embedding_cache_rows,
            compile_mamba=args.compile_mamba,
        )
        require(model.mtp is not None, "resident MTP sidecar did not load")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        empty_snapshot = model.snapshot()
        processed = 0
        for prompt_index, prompt in enumerate(prompts):
            key = str(prompt_index)
            if key in state["completed"]:
                entry = state["completed"][key]
                shard = args.output_dir / entry["file"]
                arrays, metadata = mx.load(str(shard), return_metadata=True)
                validate_trace_arrays(arrays, metadata, prompt_index, entry["rows"])
                require(sha256_file(shard) == entry["sha256"], f"teacher shard hash mismatch: {shard}")
                print(f"mtp-teacher-resume index={prompt_index} rows={entry['rows']} path={shard}", flush=True)
                continue
            if args.max_prompts is not None and processed >= args.max_prompts:
                break
            prompt_started = time.perf_counter()
            model.restore(empty_snapshot)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            require(prompt_ids, f"prompt {prompt_index} encoded to no tokens")
            logits, current_hidden = model.prefill(prompt_ids, 1)
            base_token = int(mx.argmax(logits))
            hidden_rows = []
            accepted_tokens = []
            expected_tokens = []
            generated = []
            cycles = 0
            accepted_total = 0
            while len(hidden_rows) < args.tokens_per_prompt:
                draft_token, _, _, _ = timed_draft(model.mtp, current_hidden, base_token)
                verified_logits, verified_hidden, base_snapshot = model.verify_sequence(
                    [base_token, draft_token], 0
                )
                verified_token_array = mx.argmax(verified_logits, axis=-1)
                mx.eval(verified_token_array)
                verified_tokens = [int(token) for token in verified_token_array.tolist()]
                accepted = matching_draft_prefix([draft_token], verified_tokens)
                rows = authoritative_cycle_rows(
                    current_hidden,
                    base_token,
                    verified_hidden,
                    verified_tokens,
                    accepted,
                    args.tokens_per_prompt - len(hidden_rows),
                )
                for row_hidden, accepted_token, expected_token in rows:
                    hidden_rows.append(row_hidden.astype(mx.bfloat16))
                    accepted_tokens.append(accepted_token)
                    expected_tokens.append(expected_token)
                generated.append(base_token)
                if accepted:
                    generated.append(draft_token)
                    logits = verified_logits[1]
                    current_hidden = verified_hidden[1]
                else:
                    model.restore(base_snapshot)
                    logits = verified_logits[0]
                    current_hidden = verified_hidden[0]
                base_token = verified_tokens[accepted]
                accepted_total += accepted
                cycles += 1

            arrays = {
                "target_hidden": mx.stack(hidden_rows),
                "accepted_token_ids": mx.array(accepted_tokens, dtype=mx.int32),
                "expected_token_ids": mx.array(expected_tokens, dtype=mx.int32),
                "prompt_indices": mx.full((len(hidden_rows),), prompt_index, dtype=mx.int32),
                "scored": mx.ones((len(hidden_rows),), dtype=mx.int32),
            }
            mx.eval(*arrays.values())
            shard_name = f"trace-{prompt_index:06d}.safetensors"
            shard = args.output_dir / shard_name
            temporary = shard.with_name(shard.stem + ".part" + shard.suffix)
            temporary.unlink(missing_ok=True)
            mx.save_safetensors(
                str(temporary),
                arrays,
                metadata={
                    "format": FORMAT,
                    "prompt_index": str(prompt_index),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "generated_token_ids_json": json.dumps(generated, separators=(",", ":")),
                },
            )
            temporary.replace(shard)
            entry = {
                "file": shard_name,
                "sha256": sha256_file(shard),
                "bytes": shard.stat().st_size,
                "rows": len(hidden_rows),
                "cycles": cycles,
                "accepted_drafts": accepted_total,
                "elapsed_seconds": time.perf_counter() - prompt_started,
            }
            saved_arrays, saved_metadata = mx.load(str(shard), return_metadata=True)
            validate_trace_arrays(saved_arrays, saved_metadata, prompt_index, len(hidden_rows))
            state["completed"][key] = entry
            state["rows"] = sum(item["rows"] for item in state["completed"].values())
            state["status"] = "complete" if len(state["completed"]) == len(prompts) else "running"
            state["peak_gib"] = mx.get_peak_memory() / 2**30
            atomic_json(state_path, state)
            processed += 1
            print(
                f"mtp-teacher-commit index={prompt_index} rows={entry['rows']} "
                f"cycles={cycles} accepted={accepted_total} bytes={entry['bytes']} "
                f"elapsed={entry['elapsed_seconds']:.3f}s path={shard}",
                flush=True,
            )
        print(
            f"mtp-teacher-stop status={state['status']} processed={processed} "
            f"completed={len(state['completed'])}/{len(prompts)} rows={state['rows']} "
            f"elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron MTP teacher capture error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

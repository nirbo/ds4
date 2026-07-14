#!/usr/bin/env python3
"""Materialize exact recursive-MTP teacher states for compact student distillation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import NemotronMTPSidecar
from nemotron_mlx_mtp_predictor import TRACE_FORMAT
from nemotron_paged_embeddings import PagedBF16Embedding
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-distill-features-v1"
HIDDEN_SIZE = 4096


def top_logits(logits: mx.array, count: int) -> tuple[mx.array, mx.array]:
    require(0 < count <= logits.size, "invalid MTP distillation top-logit count")
    indices = mx.argpartition(-logits, kth=count - 1)[:count]
    values = logits[indices]
    order = mx.argsort(-values)
    return indices[order].astype(mx.int32), values[order].astype(mx.float32)


def recursive_teacher_rows(
    mtp,
    target_hidden: mx.array,
    accepted_token_ids: list[int],
    max_depth: int,
    top_k: int,
) -> dict[str, mx.array]:
    hidden_rows = []
    token_rows = []
    top_index_rows = []
    top_logit_rows = []
    for source_hidden, accepted_token in zip(target_hidden, accepted_token_ids):
        hidden = source_hidden
        token = accepted_token
        depth_hidden = []
        depth_tokens = []
        depth_top_indices = []
        depth_top_logits = []
        for _ in range(max_depth):
            logits, hidden, _, _ = mtp.draft_step(hidden, token)
            indices, values = top_logits(logits, top_k)
            token = mtp.argmax_token(logits)
            mx.eval(hidden, indices, values)
            depth_hidden.append(hidden.astype(mx.float32))
            depth_tokens.append(token)
            depth_top_indices.append(indices)
            depth_top_logits.append(values)
        hidden_rows.append(mx.stack(depth_hidden))
        token_rows.append(depth_tokens)
        top_index_rows.append(mx.stack(depth_top_indices))
        top_logit_rows.append(mx.stack(depth_top_logits))
    require(hidden_rows, "MTP distillation source shard has no rows")
    result = {
        "teacher_hidden": mx.stack(hidden_rows).astype(mx.float32),
        "teacher_token_ids": mx.array(token_rows, dtype=mx.int32),
        "teacher_top_indices": mx.stack(top_index_rows).astype(mx.int32),
        "teacher_top_logits": mx.stack(top_logit_rows).astype(mx.float32),
    }
    mx.eval(*result.values())
    return result


def validate_feature_shard(
    path: Path,
    entry: dict,
    prompt_index: int,
    max_depth: int,
    top_k: int,
) -> None:
    require(path.is_file(), f"MTP distillation feature shard is missing: {path}")
    require(path.stat().st_size == entry.get("bytes"), f"feature size mismatch: {path}")
    require(sha256_file(path) == entry.get("sha256"), f"feature hash mismatch: {path}")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    rows = entry.get("rows")
    require(metadata.get("format") == FORMAT, "MTP distillation feature format mismatch")
    require(metadata.get("prompt_index") == str(prompt_index), "feature prompt mismatch")
    require(metadata.get("max_depth") == str(max_depth), "feature depth mismatch")
    require(metadata.get("top_k") == str(top_k), "feature top-k mismatch")
    require(
        set(arrays)
        == {
            "teacher_hidden",
            "teacher_token_ids",
            "teacher_top_indices",
            "teacher_top_logits",
        },
        "MTP distillation feature tensor set mismatch",
    )
    require(
        arrays["teacher_hidden"].shape == (rows, max_depth, HIDDEN_SIZE)
        and arrays["teacher_hidden"].dtype == mx.float32,
        "MTP distillation hidden tensor mismatch",
    )
    require(
        arrays["teacher_token_ids"].shape == (rows, max_depth)
        and arrays["teacher_token_ids"].dtype == mx.int32,
        "MTP distillation token tensor mismatch",
    )
    require(
        arrays["teacher_top_indices"].shape == (rows, max_depth, top_k)
        and arrays["teacher_top_indices"].dtype == mx.int32
        and arrays["teacher_top_logits"].shape == (rows, max_depth, top_k)
        and arrays["teacher_top_logits"].dtype == mx.float32,
        "MTP distillation top-logit tensor mismatch",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--mtp-sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-prompts", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(2 <= args.max_depth <= 4, "MTP distillation depth must be between 2 and 4")
        require(2 <= args.top_k <= 256, "MTP distillation top-k must be between 2 and 256")
        from nemotron_mlx_mtp_teacher_capture import validate_completed

        capture_state_path = args.capture_dir / "state.json"
        capture_state = load_json(capture_state_path)
        require(
            capture_state.get("format") == TRACE_FORMAT
            and capture_state.get("status") == "complete",
            "MTP distillation source capture is incomplete",
        )
        validate_completed(args.capture_dir, capture_state)
        identity = {
            "format": FORMAT,
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(
                args.model_dir / "nemotron_mlx_pack_report.json"
            ),
            "capture_dir": str(args.capture_dir.resolve()),
            "capture_state_sha256": sha256_file(capture_state_path),
            "mtp_sidecar": str(args.mtp_sidecar.resolve()),
            "mtp_sidecar_report_sha256": sha256_file(
                args.mtp_sidecar / "nemotron_mtp_pack_report.json"
            ),
            "mtp_lm_head": str(args.mtp_lm_head.resolve()),
            "mtp_lm_head_report_sha256": sha256_file(
                args.mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
            ),
            "max_depth": args.max_depth,
            "top_k": args.top_k,
            "mlx_version": version("mlx"),
            "tool_sha256": sha256_file(Path(__file__)),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            require(state.get("identity") == identity, "MTP distillation resume identity mismatch")
        else:
            state = {
                "format": FORMAT,
                "status": "running",
                "identity": identity,
                "completed": {},
            }
            atomic_json(state_path, state)

        global_tensors = mx.load(str(args.model_dir / "global.safetensors"))
        embeddings = PagedBF16Embedding(args.model_dir, cache_rows=256)
        mtp = NemotronMTPSidecar(
            args.mtp_sidecar,
            embeddings,
            ModelOptBF16Linear(global_tensors["lm_head.weight"]),
            alternate_lm_head=args.mtp_lm_head,
        )
        processed = 0
        started = time.perf_counter()
        for key, source_entry in sorted(
            capture_state["completed"].items(), key=lambda item: int(item[0])
        ):
            prompt_index = int(key)
            if key in state["completed"]:
                validate_feature_shard(
                    args.output_dir / state["completed"][key]["file"],
                    state["completed"][key],
                    prompt_index,
                    args.max_depth,
                    args.top_k,
                )
                continue
            if args.max_prompts is not None and processed >= args.max_prompts:
                break
            source_path = args.capture_dir / source_entry["file"]
            require(
                sha256_file(source_path) == source_entry["sha256"],
                "MTP distillation source trace hash mismatch",
            )
            source = mx.load(str(source_path))
            prompt_started = time.perf_counter()
            arrays = recursive_teacher_rows(
                mtp,
                source["target_hidden"],
                source["accepted_token_ids"].tolist(),
                args.max_depth,
                args.top_k,
            )
            name = f"features-{prompt_index:06d}.safetensors"
            path = args.output_dir / name
            temporary = path.with_name(path.stem + ".part" + path.suffix)
            mx.save_safetensors(
                str(temporary),
                arrays,
                metadata={
                    "format": FORMAT,
                    "prompt_index": str(prompt_index),
                    "max_depth": str(args.max_depth),
                    "top_k": str(args.top_k),
                },
            )
            temporary.replace(path)
            entry = {
                "file": name,
                "rows": arrays["teacher_hidden"].shape[0],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "elapsed_seconds": time.perf_counter() - prompt_started,
            }
            validate_feature_shard(
                path, entry, prompt_index, args.max_depth, args.top_k
            )
            state["completed"][key] = entry
            state["status"] = (
                "complete"
                if len(state["completed"]) == len(capture_state["completed"])
                else "running"
            )
            state["rows"] = sum(item["rows"] for item in state["completed"].values())
            state["payload_bytes"] = sum(
                item["bytes"] for item in state["completed"].values()
            )
            state["peak_gib"] = mx.get_peak_memory() / 2**30
            atomic_json(state_path, state)
            processed += 1
            print(
                f"mtp-distill-feature-commit index={prompt_index} rows={entry['rows']} "
                f"elapsed={entry['elapsed_seconds']:.3f}s",
                flush=True,
            )
        embeddings.close()
        print(
            f"mtp-distill-feature-stop status={state['status']} processed={processed} "
            f"completed={len(state['completed'])}/{len(capture_state['completed'])} "
            f"rows={state.get('rows', 0)} elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron MTP distillation feature error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

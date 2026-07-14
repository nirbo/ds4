#!/usr/bin/env python3
"""Materialize resumable official-MTP features for learned recursive drafts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import NemotronMTPSidecar
from nemotron_mlx_mtp_predictor import TRACE_FORMAT
from nemotron_paged_embeddings import PagedBF16Embedding
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-recursive-features-v1"


def validate_feature_shard(path: Path, entry: dict, prompt_index: int) -> None:
    require(path.is_file(), f"recursive feature shard is missing: {path}")
    require(path.stat().st_size == entry.get("bytes"), f"recursive feature size mismatch: {path}")
    require(sha256_file(path) == entry.get("sha256"), f"recursive feature hash mismatch: {path}")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    rows = entry.get("rows")
    require(metadata.get("format") == FORMAT, "recursive feature format mismatch")
    require(metadata.get("prompt_index") == str(prompt_index), "recursive feature prompt mismatch")
    require(
        set(arrays) == {"mtp_hidden", "mtp_token_ids"}
        and arrays["mtp_hidden"].shape == (rows, 4096)
        and arrays["mtp_hidden"].dtype == mx.bfloat16
        and arrays["mtp_token_ids"].shape == (rows,)
        and arrays["mtp_token_ids"].dtype == mx.int32,
        "recursive feature tensors mismatch",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--mtp-sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-prompts", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        capture_state_path = args.capture_dir / "state.json"
        capture_state = load_json(capture_state_path)
        require(
            capture_state.get("format") == TRACE_FORMAT
            and capture_state.get("status") == "complete",
            "recursive feature source capture is incomplete",
        )
        identity = {
            "format": FORMAT,
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(args.model_dir / "nemotron_mlx_pack_report.json"),
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
            "tool_sha256": sha256_file(Path(__file__)),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            require(state.get("identity") == identity, "recursive feature resume identity mismatch")
        else:
            state = {"format": FORMAT, "status": "running", "identity": identity, "completed": {}}
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
                )
                continue
            if args.max_prompts is not None and processed >= args.max_prompts:
                break
            source_path = args.capture_dir / source_entry["file"]
            require(sha256_file(source_path) == source_entry["sha256"], "source trace hash mismatch")
            arrays = mx.load(str(source_path))
            hidden_rows = []
            token_ids = []
            prompt_started = time.perf_counter()
            for hidden, accepted_token in zip(
                arrays["target_hidden"], arrays["accepted_token_ids"].tolist()
            ):
                logits, next_hidden, _, _ = mtp.draft_step(hidden, accepted_token)
                token = mtp.argmax_token(logits)
                mx.eval(next_hidden)
                hidden_rows.append(next_hidden.astype(mx.bfloat16))
                token_ids.append(token)
            output_arrays = {
                "mtp_hidden": mx.stack(hidden_rows),
                "mtp_token_ids": mx.array(token_ids, dtype=mx.int32),
            }
            mx.eval(*output_arrays.values())
            name = f"features-{prompt_index:06d}.safetensors"
            path = args.output_dir / name
            temporary = path.with_name(path.stem + ".part" + path.suffix)
            mx.save_safetensors(
                str(temporary),
                output_arrays,
                metadata={"format": FORMAT, "prompt_index": str(prompt_index)},
            )
            temporary.replace(path)
            entry = {
                "file": name,
                "rows": len(token_ids),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "elapsed_seconds": time.perf_counter() - prompt_started,
            }
            validate_feature_shard(path, entry, prompt_index)
            state["completed"][key] = entry
            state["status"] = (
                "complete"
                if len(state["completed"]) == len(capture_state["completed"])
                else "running"
            )
            state["rows"] = sum(item["rows"] for item in state["completed"].values())
            state["peak_gib"] = mx.get_peak_memory() / 2**30
            atomic_json(state_path, state)
            processed += 1
            print(
                f"mtp-recursive-feature-commit index={prompt_index} rows={entry['rows']} "
                f"elapsed={entry['elapsed_seconds']:.3f}s",
                flush=True,
            )
        embeddings.close()
        print(
            f"mtp-recursive-feature-stop status={state['status']} processed={processed} "
            f"completed={len(state['completed'])}/{len(capture_state['completed'])} "
            f"rows={state.get('rows', 0)} elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron recursive MTP feature error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

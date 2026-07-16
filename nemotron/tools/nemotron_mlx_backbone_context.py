#!/usr/bin/env python3
"""Capture resumable NVFP4-proxy contexts for BF16 backbone expert fitting."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.nemotron_h import group_expert_select
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_calibrate import build_batches, corpus_samples
from nemotron_mlx_mamba import layer_tensors
from nemotron_mlx_moe_layer import load_linear
from nemotron_mlx_stream_forward import StreamingForward
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-backbone-lowbit-context-v1"
STATE_FORMAT = "nemotron-backbone-lowbit-context-state-v1"


def atomic_safetensors(path: Path, tensors: dict[str, mx.array], metadata: dict[str, str]) -> None:
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(str(temporary), tensors, metadata=metadata)
    temporary.replace(path)


def split_for_sample(sample_sha256: str, validation_modulus: int) -> str:
    require(
        isinstance(sample_sha256, str) and len(sample_sha256) == 64,
        "context sample hash is invalid",
    )
    require(validation_modulus >= 2, "validation modulus must be at least two")
    return "validation" if int(sample_sha256, 16) % validation_modulus == 0 else "train"


def context_corpus_samples(path: Path) -> list[tuple[str, str]]:
    """Read either the established object corpus or balanced prompt JSONL."""

    if path.suffix != ".jsonl":
        return corpus_samples(path)
    result = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                require(isinstance(row, dict), f"context JSONL row {line_number} is not an object")
                category = row.get("category")
                prompt = row.get("prompt")
                require(
                    isinstance(category, str) and category and isinstance(prompt, str) and prompt,
                    f"context JSONL row {line_number} has no category/prompt",
                )
                result.append((category, prompt))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"invalid context JSONL {path}: {exc}") from exc
    require(result, "context JSONL has no prompts")
    return result


def validate_context_arrays(
    arrays: dict[str, mx.array],
    *,
    hidden_size: int,
    latent_size: int,
    top_k: int,
) -> int:
    require(
        set(arrays) == {"layer_input", "latent", "indices", "scores"},
        "context tensor set mismatch",
    )
    rows = arrays["layer_input"].shape[0]
    require(rows > 0, "context shard has no rows")
    require(arrays["layer_input"].shape == (rows, hidden_size), "context layer-input shape mismatch")
    require(arrays["latent"].shape == (rows, latent_size), "context latent shape mismatch")
    require(arrays["indices"].shape == arrays["scores"].shape == (rows, top_k), "context route shape mismatch")
    require(arrays["layer_input"].dtype == mx.float32, "context layer inputs must be float32")
    require(arrays["latent"].dtype == mx.float32, "context latent values must be float32")
    require(arrays["indices"].dtype == mx.int32, "context expert IDs must be int32")
    require(arrays["scores"].dtype == mx.float32, "context route scores must be float32")
    require(bool(mx.all(arrays["scores"] >= 0)), "context route score is negative")
    return rows


def validate_context_shard(path: Path, entry: dict, state: dict) -> dict[str, mx.array]:
    require(path.is_file(), f"missing context shard: {path}")
    require(path.stat().st_size == entry.get("bytes"), f"context shard size mismatch: {path}")
    require(sha256_file(path) == entry.get("sha256"), f"context shard hash mismatch: {path}")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    require(metadata.get("format") == FORMAT, f"unsupported context shard: {path}")
    require(int(metadata.get("layer", -1)) == state["layer"], f"context layer mismatch: {path}")
    require(int(metadata.get("batch", -1)) == entry["batch"], f"context batch mismatch: {path}")
    require(metadata.get("split") == entry["split"], f"context split mismatch: {path}")
    require(metadata.get("source_revision") == state["source_revision"], f"context revision mismatch: {path}")
    rows = validate_context_arrays(
        arrays,
        hidden_size=state["architecture"]["hidden_size"],
        latent_size=state["architecture"]["latent_size"],
        top_k=state["architecture"]["top_k"],
    )
    require(rows == entry["rows"], f"context row count mismatch: {path}")
    return arrays


def load_context_rows(output_dir: Path, split: str) -> dict[str, mx.array]:
    """Load one provenance-checked split for a bounded layer fit."""

    arrays, _ = load_context_rows_with_provenance(output_dir, split, require_provenance=False)
    return arrays


def load_context_rows_with_provenance(
    output_dir: Path,
    split: str,
    *,
    require_provenance: bool = True,
) -> tuple[dict[str, mx.array], dict]:
    """Load a split and preserve prompt/category ownership for every row."""

    require(split in ("train", "validation"), "unsupported context split")
    state_path = output_dir / "state.json"
    state = load_json(state_path)
    require(state.get("format") == STATE_FORMAT, "unsupported context state")
    require(state.get("status") in ("running", "complete"), "context state is not usable")
    selected = [entry for entry in state.get("completed", []) if entry.get("split") == split]
    require(selected, f"context state has no {split} rows")
    require(
        len({entry.get("batch") for entry in selected}) == len(selected),
        f"context state has duplicate {split} batches",
    )
    selected.sort(key=lambda entry: entry["batch"])
    payloads = [validate_context_shard(output_dir / entry["file"], entry, state) for entry in selected]
    result = {
        name: mx.concatenate([payload[name] for payload in payloads], axis=0)
        for name in ("layer_input", "latent", "indices", "scores")
    }
    mx.eval(*result.values())

    provenance = {
        "split": split,
        "entries": selected,
        "row_prompt_indices": None,
        "prompt_sample_sha256": [],
        "prompt_categories": [],
        "prompt_category_memberships": [],
        "prompt_rows": np.empty(0, dtype=np.int64),
    }
    if not require_provenance:
        return result, provenance

    prompt_indices: dict[str, int] = {}
    prompt_hashes: list[str] = []
    prompt_category_rows: list[dict[str, int]] = []
    row_prompt_indices = []
    for entry in selected:
        sample_sha256 = entry.get("sample_sha256")
        category = entry.get("category")
        require(
            isinstance(sample_sha256, str) and len(sample_sha256) == 64,
            f"context batch {entry['batch']} has invalid prompt provenance",
        )
        require(
            isinstance(category, str) and category,
            f"context batch {entry['batch']} has invalid category provenance",
        )
        if sample_sha256 not in prompt_indices:
            prompt_indices[sample_sha256] = len(prompt_hashes)
            prompt_hashes.append(sample_sha256)
            prompt_category_rows.append({})
        prompt_index = prompt_indices[sample_sha256]
        category_rows = prompt_category_rows[prompt_index]
        category_rows[category] = category_rows.get(category, 0) + entry["rows"]
        row_prompt_indices.append(np.full(entry["rows"], prompt_index, dtype=np.int32))

    row_prompt_array = np.concatenate(row_prompt_indices)
    require(
        row_prompt_array.shape == (result["latent"].shape[0],),
        "context prompt provenance row count mismatch",
    )
    prompt_rows = np.bincount(row_prompt_array, minlength=len(prompt_hashes)).astype(np.int64)
    require(np.all(prompt_rows > 0), "context prompt provenance has an empty prompt")
    prompt_categories = [
        min(rows, key=lambda category: (-rows[category], category))
        for rows in prompt_category_rows
    ]
    provenance.update(
        {
            "row_prompt_indices": row_prompt_array,
            "prompt_sample_sha256": prompt_hashes,
            "prompt_categories": prompt_categories,
            "prompt_category_memberships": [
                sorted(rows) for rows in prompt_category_rows
            ],
            "prompt_rows": prompt_rows,
        }
    )
    return result, provenance


class ProxyContextProjector:
    """The exact official router and latent projection, without loading experts."""

    def __init__(self, source_dir: Path, layer: int, config: dict):
        tensors = layer_tensors(source_dir, layer)
        base = f"backbone.layers.{layer}"
        mixer = f"{base}.mixer"
        required = [
            f"{base}.norm.weight",
            f"{mixer}.gate.weight",
            f"{mixer}.gate.e_score_correction_bias",
        ]
        for name in required:
            require(name in tensors, f"missing proxy context tensor: {name}")
        self.norm_weight = tensors[required[0]]
        self.gate_weight = tensors[required[1]]
        self.correction_bias = tensors[required[2]]
        self.fc1_latent = load_linear(tensors, f"{mixer}.fc1_latent_proj")
        self.epsilon = config["layer_norm_epsilon"]
        self.top_k = config["num_experts_per_tok"]
        self.n_group = config["n_group"]
        self.topk_group = config["topk_group"]
        self.routed_scaling_factor = config["routed_scaling_factor"]
        self.norm_topk_prob = config["norm_topk_prob"]

    def __call__(self, layer_input: mx.array) -> dict[str, mx.array]:
        hidden = mx.fast.rms_norm(layer_input, self.norm_weight, self.epsilon)
        indices, scores = group_expert_select(
            hidden @ self.gate_weight.T,
            self.correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        latent = self.fc1_latent(hidden)
        result = {
            "layer_input": layer_input.reshape(-1, layer_input.shape[-1]).astype(mx.float32),
            "latent": latent.reshape(-1, latent.shape[-1]).astype(mx.float32),
            "indices": indices.reshape(-1, indices.shape[-1]).astype(mx.int32),
            "scores": scores.reshape(-1, scores.shape[-1]).astype(mx.float32),
        }
        mx.eval(*result.values())
        return result


def coverage(completed: list[dict], expert_counts: list[int]) -> dict:
    observed = sum(count > 0 for count in expert_counts)
    return {
        "rows": sum(entry["rows"] for entry in completed),
        "train_rows": sum(entry["rows"] for entry in completed if entry["split"] == "train"),
        "validation_rows": sum(
            entry["rows"] for entry in completed if entry["split"] == "validation"
        ),
        "observed_experts": observed,
        "expert_coverage": observed / len(expert_counts),
        "min_routes": min(expert_counts),
        "max_routes": max(expert_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--max-sample-tokens", type=int, default=512)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.batch_tokens > 0 and args.max_sample_tokens > 0, "context token limits must be positive")
        require(args.validation_modulus >= 2, "validation modulus must be at least two")
        require(args.max_batches is None or args.max_batches > 0, "max batches must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        pattern = config["hybrid_override_pattern"]
        require(0 <= args.layer < len(pattern) and pattern[args.layer] == "E", "selected layer is not LatentMoE")
        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        batches = build_batches(
            tokenizer,
            context_corpus_samples(args.corpus),
            args.batch_tokens,
            args.max_sample_tokens,
        )
        identity = {
            "format": STATE_FORMAT,
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": sha256_file(args.corpus),
            "tool_sha256": sha256_file(Path(__file__)),
            "layer": args.layer,
            "batch_tokens": args.batch_tokens,
            "max_sample_tokens": args.max_sample_tokens,
            "validation_modulus": args.validation_modulus,
            "split_strategy": "prompt-sha256-modulus-v1",
            "total_batches": len(batches),
            "architecture": {
                "hidden_size": config["hidden_size"],
                "latent_size": config["moe_latent_size"],
                "top_k": config["num_experts_per_tok"],
                "experts": config["n_routed_experts"],
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            for key, value in identity.items():
                require(state.get(key) == value, f"context state identity mismatch: {key}")
        else:
            state = {
                **identity,
                "status": "running",
                "completed": [],
                "expert_route_counts": [0] * config["n_routed_experts"],
                "coverage": {},
            }
            atomic_json(state_path, state)
        operation_log = OperationLog(args.output_dir / "run.log")
        operation_log.write(
            f"backbone-context-start layer={args.layer} completed={len(state['completed'])}/{len(batches)}"
        )
        for entry in state["completed"]:
            validate_context_shard(args.output_dir / entry["file"], entry, state)
        if args.validate_only:
            operation_log.write("backbone-context-validate-complete")
            return 0

        projector = ProxyContextProjector(args.source_dir, args.layer, config)
        completed_batches = {entry["batch"] for entry in state["completed"]}
        processed = 0
        for batch_index, batch in enumerate(batches):
            if batch_index in completed_batches:
                continue
            if args.max_batches is not None and processed >= args.max_batches:
                break
            split = split_for_sample(batch["sample_sha256"], args.validation_modulus)
            operation_log.write(
                f"backbone-context-batch-start batch={batch_index} split={split} "
                f"category={batch['category']} rows={len(batch['token_ids'])}"
            )
            started = time.perf_counter()
            runner = StreamingForward(args.source_dir)
            layer_input = runner.forward_sequence(
                batch["token_ids"],
                max_layers=args.layer,
                score_head=False,
            )
            arrays = projector(layer_input)
            rows = validate_context_arrays(
                arrays,
                hidden_size=config["hidden_size"],
                latent_size=config["moe_latent_size"],
                top_k=config["num_experts_per_tok"],
            )
            file_name = f"batch-{batch_index:05d}-{split}.safetensors"
            path = args.output_dir / file_name
            atomic_safetensors(
                path,
                arrays,
                {
                    "format": FORMAT,
                    "source_revision": source_state["revision"],
                    "layer": str(args.layer),
                    "batch": str(batch_index),
                    "split": split,
                },
            )
            entry = {
                "batch": batch_index,
                "split": split,
                "category": batch["category"],
                "sample_sha256": batch["sample_sha256"],
                "token_offset": batch["offset"],
                "token_ids_sha256": hashlib.sha256(
                    np.asarray(batch["token_ids"], dtype=np.uint32).tobytes()
                ).hexdigest(),
                "rows": rows,
                "file": file_name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            validate_context_shard(path, entry, state)
            routed = np.asarray(arrays["indices"]).reshape(-1)
            counts = np.bincount(routed, minlength=config["n_routed_experts"])
            state["expert_route_counts"] = (
                np.asarray(state["expert_route_counts"], dtype=np.int64) + counts
            ).tolist()
            state["completed"].append(entry)
            state["coverage"] = coverage(state["completed"], state["expert_route_counts"])
            atomic_json(state_path, state)
            operation_log.write(
                f"backbone-context-batch-done batch={batch_index} split={split} rows={rows} "
                f"elapsed={time.perf_counter() - started:.2f}s "
                f"coverage={state['coverage']['expert_coverage']:.3%}"
            )
            processed += 1
            del runner, layer_input, arrays
            gc.collect()
            mx.clear_cache()
        if len(state["completed"]) == len(batches):
            state["status"] = "complete"
            atomic_json(state_path, state)
        operation_log.write(
            f"backbone-context-stop status={state['status']} processed={processed} "
            f"completed={len(state['completed'])}/{len(batches)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"backbone-context-failed error={exc}")
        print(f"nemotron backbone context error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

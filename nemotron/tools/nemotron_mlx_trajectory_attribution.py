#!/usr/bin/env python3
"""Attribute expert-pruning error on a complete stored reasoning trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_layer_sensitivity import baseline_components, parse_plan, pruned_routed_output
from nemotron_mlx_humaneval import prompt_for as humaneval_prompt_for
from nemotron_mlx_livecodebench import load_items, prompt_for as livecodebench_prompt_for
from nemotron_mlx_mbpp import chat_token_ids, prompt_for as mbpp_prompt_for
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_proxy_compare import error_metrics
from nemotron_mlx_stream_forward import StreamingForward, validate_virtual_plan
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-trajectory-attribution-v1"
CAPTURE_FORMAT = "nemotron-trajectory-capture-v1"


def parse_ints(value: str, *, positive: bool = False) -> list[int]:
    try:
        result = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise MetadataError(f"invalid integer list: {value}") from exc
    require(result and result == sorted(set(result)), "integer list must be sorted and unique")
    require(not positive or result[0] > 0, "integer list values must be positive")
    return result


def sampled_positions(prompt_tokens: int, total_tokens: int, stride: int, limit: int) -> list[int]:
    require(0 < prompt_tokens <= total_tokens and stride > 0 and limit > 0, "invalid trajectory sampling")
    start = prompt_tokens - 1
    positions = list(range(start, total_tokens, stride))
    if positions[-1] != total_tokens - 1:
        positions.append(total_tokens - 1)
    if len(positions) > limit:
        indexes = np.linspace(0, len(positions) - 1, num=limit, dtype=np.int64)
        positions = [positions[int(index)] for index in indexes]
    return sorted(set(positions))


def rank_removed_experts(
    indices: np.ndarray,
    scores: np.ndarray,
    output_norms: np.ndarray,
    retained: list[int],
    expert_count: int | None = None,
) -> tuple[list[int], np.ndarray]:
    experts = max(int(indices.max()) + 1, max(retained) + 1)
    if expert_count is not None:
        require(expert_count >= experts, "expert count is smaller than observed expert index")
        experts = expert_count
    importance = np.zeros(experts, dtype=np.float64)
    np.add.at(importance, indices.reshape(-1), (scores * output_norms).reshape(-1))
    retained_mask = np.zeros(experts, dtype=bool)
    retained_mask[retained] = True
    ranking = [
        int(expert)
        for expert in np.argsort(-importance, kind="stable")
        if not retained_mask[expert] and importance[expert] > 0
    ]
    return ranking, importance


def token_digest(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def trajectory_tokens(
    tokenizer,
    dataset: Path,
    report: dict,
    task_id: str,
    repeat: int,
    max_generated_tokens: int,
    trajectory_format: str = "livecodebench",
) -> tuple[list[int], dict]:
    if trajectory_format == "livecodebench":
        items = {str(item["question_id"]): item for item in load_items(dataset)}
        prompt = livecodebench_prompt_for
        expected_report_format = "nemotron-livecodebench-v1"
    else:
        require(trajectory_format in ("mbpp", "humaneval"), "unsupported trajectory format")
        items = {
            str(item["task_id"]): item
            for line in dataset.read_text().splitlines()
            if line.strip()
            for item in [json.loads(line)]
        }
        prompt = mbpp_prompt_for if trajectory_format == "mbpp" else humaneval_prompt_for
        expected_report_format = (
            "nemotron-mbpp-eval-v1"
            if trajectory_format == "mbpp"
            else "nemotron-humaneval-v1"
        )
        require(repeat == 0, f"{trajectory_format} trajectories do not have repeats")
    require(report.get("format") == expected_report_format, "trajectory report format mismatch")
    require(task_id in items, f"trajectory task is absent from dataset: {task_id}")
    rows = [
        row
        for row in report.get("results", [])
        if str(row.get("task_id")) == task_id and int(row.get("repeat", 0)) == repeat
    ]
    require(len(rows) == 1, "trajectory report row is missing or duplicated")
    row = rows[0]
    generation = report.get("generation", {})
    prompt_ids = chat_token_ids(
        tokenizer,
        prompt(items[task_id]),
        None,
        bool(generation.get("enable_thinking")),
        bool(generation.get("low_effort")),
    )
    if trajectory_format == "livecodebench":
        generated_text = str(row.get("reasoning", ""))
        if row.get("response"):
            generated_text += "</think>" + str(row["response"])
    else:
        generated_text = str(row.get("response", ""))
    generated_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    reported_tokens = int(row.get("generated_tokens", 0))
    require(abs(len(generated_ids) - reported_tokens) <= 2, "stored trajectory does not round-trip to token IDs")
    if max_generated_tokens:
        generated_ids = generated_ids[:max_generated_tokens]
    require(generated_ids, "stored trajectory encoded to no generated tokens")
    trajectory = {
        "task_id": task_id,
        "repeat": repeat,
        "seed": row.get("seed"),
        "passed": bool(row.get("passed")),
        "truncated": bool(row.get("truncated")),
        "prompt_tokens": len(prompt_ids),
        "reported_generated_tokens": reported_tokens,
        "reencoded_generated_tokens": len(tokenizer.encode(generated_text, add_special_tokens=False)),
        "used_generated_tokens": len(generated_ids),
    }
    if trajectory_format != "livecodebench":
        trajectory["trajectory_format"] = trajectory_format
    return prompt_ids + generated_ids, trajectory


def route_observation(block, x: mx.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = block.norm(x)
    indices, scores = block.route(hidden)
    latent = block.fc1_latent(hidden)
    selected = expert_outputs(latent, block.experts, indices)
    norms = mx.linalg.norm(selected.astype(mx.float32), axis=-1)
    mx.eval(indices, scores, norms)
    return (
        np.asarray(indices, dtype=np.int32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(norms, dtype=np.float32),
    )


def capture_inputs(
    source_dir: Path,
    capture_dir: Path,
    identity: dict,
    token_ids: list[int],
    layers: list[int],
    positions: list[int],
    operation_log: OperationLog,
) -> dict:
    state_path = capture_dir / "state.json"
    if state_path.exists():
        state = load_json(state_path)
        require(
            state.get("format") == CAPTURE_FORMAT and state.get("identity") == identity,
            "capture identity mismatch",
        )
        require(state.get("status") == "complete", "trajectory capture is incomplete")
        for layer in layers:
            path = capture_dir / f"layer-{layer:03d}.npy"
            require(
                path.is_file() and state["files"][str(layer)] == sha256_file(path),
                f"capture layer {layer} mismatch",
            )
        operation_log.write(f"capture-resume path={capture_dir} layers={len(layers)}")
        return state

    capture_dir.mkdir(parents=True, exist_ok=True)
    operation_log.write(
        f"capture-start tokens={len(token_ids)} positions={len(positions)} "
        f"last_layer={max(layers)}"
    )
    runner = StreamingForward(source_dir)
    started = time.perf_counter()
    runner.forward_sequence(
        token_ids,
        max_layers=max(layers) + 1,
        trace=True,
        score_head=False,
        capture_layer_inputs=set(layers),
        capture_positions=positions,
    )
    require(set(runner.layer_inputs) == set(layers), "trajectory layer capture is incomplete")
    files = {}
    for layer in layers:
        path = capture_dir / f"layer-{layer:03d}.npy"
        temporary = path.with_suffix(".npy.part")
        with temporary.open("wb") as handle:
            np.save(handle, runner.layer_inputs[layer], allow_pickle=False)
        temporary.replace(path)
        files[str(layer)] = sha256_file(path)
    state = {
        "format": CAPTURE_FORMAT,
        "status": "complete",
        "identity": identity,
        "files": files,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gib": mx.get_peak_memory() / 2**30,
    }
    atomic_json(state_path, state)
    operation_log.write(
        f"capture-done elapsed={state['elapsed_seconds']:.2f}s peak={state['peak_gib']:.3f}GiB"
    )
    del runner
    gc.collect()
    mx.clear_cache()
    return state


def analyze_layer(block, x: mx.array, retained: list[int], addback_counts: list[int]) -> dict:
    baseline_routed, shared = baseline_components(block, x)
    mx.eval(baseline_routed, shared)
    baseline_routed_np = np.asarray(baseline_routed, dtype=np.float32)
    shared_np = np.asarray(shared, dtype=np.float32)
    x_np = np.asarray(x, dtype=np.float32)
    baseline_update = baseline_routed_np + shared_np
    baseline_output = x_np + baseline_update
    indices, scores, norms = route_observation(block, x)
    ranking, importance = rank_removed_experts(
        indices,
        scores,
        norms,
        retained,
        int(block.gate_weight.shape[0]),
    )
    retained_mask = np.zeros(block.gate_weight.shape[0], dtype=bool)
    retained_mask[retained] = True
    lost = ~retained_mask[indices]
    curves = []
    for count in [0, *addback_counts]:
        restored = sorted(set(retained).union(ranking[:count]))
        candidate = pruned_routed_output(block, x, restored)
        mx.eval(candidate)
        candidate_np = np.asarray(candidate, dtype=np.float32)
        curves.append(
            {
                "requested_addback": count,
                "actual_addback": len(restored) - len(retained),
                "retained_experts": len(restored),
                "added_experts": ranking[:count],
                "routed": error_metrics(candidate_np, baseline_routed_np),
                "update": error_metrics(candidate_np + shared_np, baseline_update),
                "output": error_metrics(x_np + candidate_np + shared_np, baseline_output),
            }
        )
    return {
        "base_retained_experts": len(retained),
        "lost_route_slots": int(lost.sum()),
        "total_route_slots": int(lost.size),
        "mean_lost_score_mass": float((scores * lost).sum(axis=-1).mean()),
        "max_lost_score_mass": float((scores * lost).sum(axis=-1).max()),
        "lost_weighted_output_norm": float((scores * norms * lost).sum()),
        "selected_expert_importance": {
            str(expert): float(value)
            for expert, value in enumerate(importance)
            if value > 0.0
        },
        "removed_expert_ranking": ranking,
        "removed_expert_importance": {str(expert): float(importance[expert]) for expert in ranking},
        "curves": curves,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--trajectory-format",
        choices=("livecodebench", "mbpp", "humaneval"),
        default="livecodebench",
    )
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--max-generated-tokens", type=int, default=0)
    parser.add_argument("--capture-stride", type=int, default=16)
    parser.add_argument("--max-capture-positions", type=int, default=256)
    parser.add_argument("--addback-counts", default="4,8,16,32")
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        operation_log = OperationLog(args.log or args.output.with_suffix(".log"))
        require(args.repeat >= 0 and args.max_generated_tokens >= 0, "invalid trajectory limits")
        layers = parse_ints(args.layers)
        addback_counts = parse_ints(args.addback_counts, positive=True)
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        moe_layers = [layer for layer, kind in enumerate(config["hybrid_override_pattern"]) if kind == "E"]
        require(all(layer in moe_layers for layer in layers), "trajectory layer is not an MoE layer")
        plan_specs = [parse_plan(value) for value in args.plan]
        require(len({label for label, _ in plan_specs}) == len(plan_specs), "duplicate plan label")
        plans = {}
        plan_hashes = {}
        for label, path in plan_specs:
            plan = load_json(path)
            plans[label] = validate_virtual_plan(plan, config, source_state["revision"])
            plan_hashes[label] = sha256_file(path)

        tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
        report = load_json(args.report)
        token_ids, trajectory = trajectory_tokens(
            tokenizer,
            args.dataset,
            report,
            args.task_id,
            args.repeat,
            args.max_generated_tokens,
            args.trajectory_format,
        )
        positions = sampled_positions(
            trajectory["prompt_tokens"],
            len(token_ids),
            args.capture_stride,
            args.max_capture_positions,
        )
        capture_identity = {
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "dataset_sha256": sha256_file(args.dataset),
            "report_sha256": sha256_file(args.report),
            "trajectory": trajectory,
            "token_ids_sha256": token_digest(token_ids),
            "total_tokens": len(token_ids),
            "positions": positions,
            "layers": layers,
        }
        capture = capture_inputs(
            args.source_dir,
            args.capture_dir,
            capture_identity,
            token_ids,
            layers,
            positions,
            operation_log,
        )
        identity = {
            "format": FORMAT,
            "tool_sha256": sha256_file(Path(__file__)),
            "capture_state_sha256": sha256_file(args.capture_dir / "state.json"),
            "plan_sha256": plan_hashes,
            "addback_counts": addback_counts,
        }
        if args.output.exists():
            output = load_json(args.output)
            for key, value in identity.items():
                require(output.get(key) == value, f"attribution report identity mismatch: {key}")
        else:
            output = {**identity, "status": "running", "capture": capture, "layers": []}
            atomic_json(args.output, output)
        completed = {int(item["layer"]) for item in output["layers"]}
        for layer in layers:
            if layer in completed:
                operation_log.write(f"layer-resume layer={layer} status=complete")
                continue
            started = time.perf_counter()
            operation_log.write(f"layer-start layer={layer} positions={len(positions)}")
            x = mx.array(np.load(args.capture_dir / f"layer-{layer:03d}.npy", allow_pickle=False))
            block = load_moe_layer(args.source_dir, layer)
            plan_results = {
                label: analyze_layer(block, x, plans[label][str(layer)], addback_counts)
                for label, _ in plan_specs
            }
            output["layers"].append(
                {
                    "layer": layer,
                    "positions": len(positions),
                    "plans": plan_results,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            atomic_json(args.output, output)
            operation_log.write(
                f"layer-done layer={layer} elapsed={time.perf_counter() - started:.2f}s"
            )
            del block, x
            gc.collect()
            mx.clear_cache()
        output["status"] = "complete"
        atomic_json(args.output, output)
        operation_log.write(
            f"run-done path={args.output} sha256={sha256_file(args.output)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"nemotron trajectory attribution error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

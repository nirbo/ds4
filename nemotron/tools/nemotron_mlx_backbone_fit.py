#!/usr/bin/env python3
"""Resumably fit one Nemotron BF16 expert layer into binary candidates."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT
from nemotron_bf16_source import validate_local_shards
from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_backbone_context import STATE_FORMAT as CONTEXT_STATE_FORMAT
from nemotron_mlx_backbone_context import load_context_rows
from nemotron_mlx_backbone_lowbit import FORMAT as EXPERT_FORMAT
from nemotron_mlx_backbone_lowbit import (
    BinaryExpert,
    affine_weight,
    dequantize_affine,
    expert_output,
    fit_binary_expert,
    output_error,
    precision_metrics,
    refine_binary_expert,
)
from nemotron_prune_materialize import (
    OperationLog,
    atomic_json,
    load_source_state,
    sha256_file,
)


STATE_FORMAT = "nemotron-backbone-lowbit-fit-state-v1"


def parse_experts(value: str | None, expert_count: int) -> list[int]:
    if value is None or value == "all":
        return list(range(expert_count))
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise MetadataError(f"invalid expert list: {value}") from exc
    require(result and result == sorted(set(result)), "expert list must be sorted and unique")
    require(result[0] >= 0 and result[-1] < expert_count, "expert list is out of range")
    return result


def expert_context_rows(
    contexts: dict[str, mx.array], expert: int, route_weight_power: float
) -> tuple[mx.array, mx.array, np.ndarray, mx.array]:
    require(route_weight_power > 0.0, "route-weight power must be positive")
    indices = np.asarray(contexts["indices"])
    scores = np.asarray(contexts["scores"], dtype=np.float32)
    rows, slots = np.where(indices == expert)
    require(np.unique(rows).size == rows.size, "expert is routed more than once in one context row")
    weights = np.power(scores[rows, slots], route_weight_power).astype(np.float32)
    return (
        contexts["latent"][mx.array(rows)],
        mx.array(weights),
        rows,
        mx.array(scores[rows, slots], dtype=mx.float32),
    )


class BF16LayerWeights:
    """Lazy shard-backed access to one strict BF16 expert layer."""

    def __init__(self, raw_dir: Path, contract: dict):
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        self.raw_dir = raw_dir
        self.contract = contract
        self.layer = contract["layer"]
        self.experts = contract["architecture"]["experts"]
        self.latent = contract["architecture"]["latent_width"]
        self.hidden = contract["architecture"]["hidden_width"]
        self.tensors = {
            entry["name"]: mx.load(str(raw_dir / entry["name"]))
            for entry in contract["required_shards"]
        }

    def expert(self, expert: int) -> tuple[mx.array, mx.array]:
        require(0 <= expert < self.experts, "BF16 expert ID is out of range")
        base = f"backbone.layers.{self.layer}.mixer.experts.{expert}"
        names = (f"{base}.up_proj.weight", f"{base}.down_proj.weight")
        values = []
        for name, expected_shape in zip(
            names,
            ((self.hidden, self.latent), (self.latent, self.hidden)),
            strict=True,
        ):
            shard = self.contract["tensor_map"].get(name)
            require(isinstance(shard, str), f"BF16 layer contract has no tensor: {name}")
            require(name in self.tensors[shard], f"BF16 source shard has no tensor: {name}")
            value = self.tensors[shard][name]
            require(value.dtype == mx.bfloat16, f"BF16 expert tensor has wrong dtype: {name}")
            require(value.shape == expected_shape, f"BF16 expert tensor has wrong shape: {name}")
            values.append(value)
        mx.eval(*values)
        return values[0], values[1]


class NVFP4TargetWeights:
    """Lazy exact dequantization of the official QAT target expert function."""

    def __init__(self, source_dir: Path, layer: int, experts: int, latent: int, hidden: int):
        self.layer = layer
        self.experts = experts
        self.latent = latent
        self.hidden = hidden
        index = load_json(source_dir / "model.safetensors.index.json")
        base = f"backbone.layers.{layer}.mixer.experts"
        shard_names = sorted(
            {
                shard
                for name, shard in index.get("weight_map", {}).items()
                if name.startswith(base + ".")
            }
        )
        require(shard_names, f"no native NVFP4 target tensors for layer {layer}")
        self.index = index["weight_map"]
        self.tensors = {shard: mx.load(str(source_dir / shard)) for shard in shard_names}

    def _projection(self, expert: int, projection: str, shape: tuple[int, int]) -> mx.array:
        prefix = f"backbone.layers.{self.layer}.mixer.experts.{expert}.{projection}"
        names = (f"{prefix}.weight", f"{prefix}.weight_scale", f"{prefix}.weight_scale_2")
        values = []
        for name in names:
            shard = self.index.get(name)
            require(isinstance(shard, str) and shard in self.tensors, f"missing native target tensor: {name}")
            require(name in self.tensors[shard], f"native target tensor absent from shard: {name}")
            values.append(self.tensors[shard][name])
        packed, scales, global_scale = values
        require(packed.dtype == mx.uint8 and packed.shape == (shape[0], shape[1] // 2), f"native target weight mismatch: {prefix}")
        require(scales.dtype == mx.uint8 and scales.shape == (shape[0], shape[1] // 16), f"native target scale mismatch: {prefix}")
        require(global_scale.dtype == mx.float32 and global_scale.size == 1, f"native target global scale mismatch: {prefix}")
        dense = mx.dequantize(
            packed.view(mx.uint32),
            scales,
            group_size=16,
            bits=4,
            mode="nvfp4",
            dtype=mx.float32,
        ) * global_scale.reshape(())
        mx.eval(dense)
        return dense

    def expert(self, expert: int) -> tuple[mx.array, mx.array]:
        require(0 <= expert < self.experts, "native target expert ID is out of range")
        return (
            self._projection(expert, "up_proj", (self.hidden, self.latent)),
            self._projection(expert, "down_proj", (self.latent, self.hidden)),
        )


def artifact_tensors(
    expert: BinaryExpert,
    validation_rows: mx.array,
    validation_weighted_residual: mx.array,
) -> dict[str, mx.array]:
    expert.validate()
    require(validation_rows.dtype == mx.int32 and validation_rows.ndim == 1, "invalid evidence rows")
    require(
        validation_weighted_residual.dtype == mx.float32
        and validation_weighted_residual.ndim == 2
        and validation_weighted_residual.shape[0] == validation_rows.size
        and validation_weighted_residual.shape[1] == expert.up.columns,
        "invalid weighted validation residual",
    )
    return {
        "up.weight": expert.up.weight,
        "up.scales": expert.up.scales,
        "up.biases": expert.up.biases,
        "down.weight": expert.down.weight,
        "down.scales": expert.down.scales,
        "down.biases": expert.down.biases,
        "validation.rows": validation_rows,
        "validation.weighted_residual": validation_weighted_residual,
    }


def atomic_expert_artifact(
    path: Path,
    expert: BinaryExpert,
    *,
    layer: int,
    expert_id: int,
    source_revision: str,
    contract_sha256: str,
    context_state_sha256: str,
    fit_strategy: str = "bf16-endpoint",
    validation_rows: mx.array,
    validation_weighted_residual: mx.array,
) -> None:
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(
        str(temporary),
        artifact_tensors(expert, validation_rows, validation_weighted_residual),
        metadata={
            "format": EXPERT_FORMAT,
            "source_revision": source_revision,
            "layer": str(layer),
            "expert": str(expert_id),
            "bits": "1",
            "group_size": str(expert.up.group_size),
            "latent_width": str(expert.up.columns),
            "hidden_width": str(expert.up.rows),
            "contract_sha256": contract_sha256,
            "context_state_sha256": context_state_sha256,
            "fit_strategy": fit_strategy,
        },
    )
    temporary.replace(path)


def validate_expert_artifact(path: Path, entry: dict, identity: dict) -> dict[str, mx.array]:
    require(path.is_file(), f"missing binary expert artifact: {path}")
    require(path.stat().st_size == entry.get("bytes"), f"binary expert artifact size mismatch: {path}")
    require(sha256_file(path) == entry.get("sha256"), f"binary expert artifact hash mismatch: {path}")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    require(metadata.get("format") == EXPERT_FORMAT, f"unsupported binary expert artifact: {path}")
    require(metadata.get("source_revision") == identity["source_revision"], "binary expert revision mismatch")
    require(int(metadata.get("layer", -1)) == identity["layer"], "binary expert layer mismatch")
    require(int(metadata.get("expert", -1)) == entry["expert"], "binary expert ID mismatch")
    require(metadata.get("contract_sha256") == identity["contract_sha256"], "binary contract mismatch")
    require(
        metadata.get("fit_strategy", "bf16-endpoint")
        == identity.get("fit_strategy", "bf16-endpoint"),
        "binary fit strategy mismatch",
    )
    require(
        metadata.get("context_state_sha256") == identity["context_state_sha256"],
        "binary context mismatch",
    )
    architecture = identity.get("architecture")
    require(isinstance(architecture, dict), "binary artifact architecture is absent")
    latent = architecture.get("latent_width")
    hidden = architecture.get("hidden_width")
    group_size = identity.get("group_size")
    validation_rows = identity.get("validation_context_rows")
    require(
        isinstance(latent, int)
        and isinstance(hidden, int)
        and isinstance(group_size, int)
        and latent > 0
        and hidden > 0
        and group_size > 0
        and latent % group_size == 0
        and hidden % group_size == 0,
        "binary artifact architecture is invalid",
    )
    require(
        isinstance(validation_rows, int) and validation_rows > 0,
        "binary artifact validation extent is invalid",
    )
    require(metadata.get("bits") == "1", "binary artifact bit width mismatch")
    require(int(metadata.get("group_size", -1)) == group_size, "binary artifact group size mismatch")
    require(int(metadata.get("latent_width", -1)) == latent, "binary artifact latent width mismatch")
    require(int(metadata.get("hidden_width", -1)) == hidden, "binary artifact hidden width mismatch")
    require(
        set(arrays)
        == {
            "up.weight",
            "up.scales",
            "up.biases",
            "down.weight",
            "down.scales",
            "down.biases",
            "validation.rows",
            "validation.weighted_residual",
        },
        "binary expert tensor set mismatch",
    )
    expected = {
        "up.weight": (mx.uint32, (hidden, latent // 32)),
        "up.scales": (mx.bfloat16, (hidden, latent // group_size)),
        "up.biases": (mx.bfloat16, (hidden, latent // group_size)),
        "down.weight": (mx.uint32, (latent, hidden // 32)),
        "down.scales": (mx.bfloat16, (latent, hidden // group_size)),
        "down.biases": (mx.bfloat16, (latent, hidden // group_size)),
    }
    for name, (dtype, shape) in expected.items():
        require(arrays[name].dtype == dtype, f"binary artifact dtype mismatch: {name}")
        require(arrays[name].shape == shape, f"binary artifact shape mismatch: {name}")
    evidence_rows = arrays["validation.rows"]
    require(
        evidence_rows.dtype == mx.int32 and evidence_rows.ndim == 1 and evidence_rows.size > 0,
        "binary evidence rows have wrong dtype or shape",
    )
    require(
        bool(mx.all((evidence_rows >= 0) & (evidence_rows < validation_rows))),
        "binary evidence row is out of range",
    )
    require(
        len(set(evidence_rows.tolist())) == evidence_rows.size,
        "binary evidence rows are not unique",
    )
    require(
        arrays["validation.weighted_residual"].dtype == mx.float32
        and arrays["validation.weighted_residual"].shape == (evidence_rows.size, latent),
        "binary weighted residual is invalid",
    )
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--proxy-source-dir", required=True, type=Path)
    parser.add_argument("--proxy-source-state", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experts", default="all")
    parser.add_argument("--max-experts", type=int)
    parser.add_argument("--min-train-routes", type=int, default=8)
    parser.add_argument("--min-validation-routes", type=int, default=4)
    parser.add_argument("--route-weight-power", type=float, default=2.0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--fit-strategy",
        choices=("bf16-endpoint", "native-target-rtn"),
        default="bf16-endpoint",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--endpoint-margin", type=float, default=0.25)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--refine-batch-size", type=int, default=16)
    parser.add_argument("--refine-endpoint-learning-rate", type=float, default=1e-3)
    parser.add_argument("--refine-code-learning-rate", type=float, default=3e-3)
    parser.add_argument("--refine-code-warmup-steps", type=int, default=8)
    parser.add_argument("--refine-evaluate-every", type=int, default=4)
    parser.add_argument("--refine-temperature", type=float, default=1.0)
    parser.add_argument("--refine-logit-margin", type=float, default=0.25)
    parser.add_argument("--refine-code-anchor", type=float, default=1e-4)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.max_experts is None or args.max_experts > 0, "max experts must be positive")
        require(args.min_train_routes >= 2, "minimum train routes must be at least two")
        require(args.min_validation_routes >= 1, "minimum validation routes must be positive")
        require(args.route_weight_power > 0.0, "route-weight power must be positive")
        require(args.group_size > 0, "group size must be positive")
        require(args.ridge >= 0.0 and args.endpoint_margin >= 0.0, "fit regularizers must be nonnegative")
        require(args.refine_steps >= 0, "refinement steps must be nonnegative")
        require(
            args.fit_strategy == "bf16-endpoint" or args.refine_steps == 0,
            "native-target-rtn does not support endpoint/code refinement",
        )
        require(args.refine_batch_size > 0, "refinement batch size must be positive")
        require(
            args.refine_endpoint_learning_rate > 0.0
            and args.refine_code_learning_rate > 0.0,
            "refinement learning rates must be positive",
        )
        require(
            args.refine_code_warmup_steps >= 0 and args.refine_evaluate_every > 0,
            "invalid refinement schedule",
        )
        require(
            args.refine_temperature > 0.0
            and args.refine_logit_margin > 0.0
            and args.refine_code_anchor >= 0.0,
            "invalid refinement quantizer settings",
        )
        contract = load_json(args.contract)
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        validate_local_shards(contract, args.raw_dir, hash_payloads=True)
        context_state_path = args.contexts / "state.json"
        context_state = load_json(context_state_path)
        require(context_state.get("format") == CONTEXT_STATE_FORMAT, "unsupported context state")
        proxy_state = load_source_state(args.proxy_source_state, args.proxy_source_dir)
        require(
            context_state.get("source_revision") == proxy_state["revision"],
            "context/native target revision mismatch",
        )
        require(
            context_state.get("source_state_sha256") == sha256_file(args.proxy_source_state),
            "context/native target state mismatch",
        )
        require(context_state.get("layer") == contract["layer"], "context/BF16 layer mismatch")
        require(
            context_state.get("architecture", {}).get("latent_size")
            == contract["architecture"]["latent_width"],
            "context/BF16 latent width mismatch",
        )
        train = load_context_rows(args.contexts, "train")
        validation = load_context_rows(args.contexts, "validation")
        expert_count = contract["architecture"]["experts"]
        selected_experts = parse_experts(args.experts, expert_count)
        metric_labels = (
            {
                "initial": "bf16-source-rtn",
                "fitted": "bf16-endpoint-or-refined",
            }
            if args.fit_strategy == "bf16-endpoint"
            else {
                "initial": "bf16-source-rtn",
                "fitted": "native-target-rtn",
            }
        )
        identity = {
            "format": STATE_FORMAT,
            "source_repository": contract["repository"],
            "source_revision": contract["source_revision"],
            "proxy_source_revision": context_state["source_revision"],
            "proxy_source_state_sha256": sha256_file(args.proxy_source_state),
            "layer": contract["layer"],
            "architecture": {
                "experts": expert_count,
                "latent_width": contract["architecture"]["latent_width"],
                "hidden_width": contract["architecture"]["hidden_width"],
            },
            "validation_context_rows": validation["latent"].shape[0],
            "contract": str(args.contract.resolve()),
            "contract_sha256": sha256_file(args.contract),
            "context_state": str(context_state_path.resolve()),
            "context_state_sha256": sha256_file(context_state_path),
            "tool_sha256": sha256_file(Path(__file__)),
            "experts": selected_experts,
            "min_train_routes": args.min_train_routes,
            "min_validation_routes": args.min_validation_routes,
            "route_weight_power": args.route_weight_power,
            "group_size": args.group_size,
            "fit_strategy": args.fit_strategy,
            "metric_labels": metric_labels,
            "ridge": args.ridge,
            "endpoint_margin": args.endpoint_margin,
            "refinement": {
                "steps": args.refine_steps,
                "batch_size": args.refine_batch_size,
                "endpoint_learning_rate": args.refine_endpoint_learning_rate,
                "code_learning_rate": args.refine_code_learning_rate,
                "code_warmup_steps": min(args.refine_code_warmup_steps, args.refine_steps),
                "evaluate_every": args.refine_evaluate_every,
                "temperature": args.refine_temperature,
                "logit_margin": args.refine_logit_margin,
                "code_anchor": args.refine_code_anchor,
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = args.output_dir / "experts"
        artifacts_dir.mkdir(exist_ok=True)
        state_path = args.output_dir / "state.json"
        if state_path.exists():
            state = load_json(state_path)
            for key, value in identity.items():
                require(state.get(key) == value, f"binary fit state identity mismatch: {key}")
        else:
            state = {**identity, "status": "running", "completed": [], "skipped": []}
            atomic_json(state_path, state)
        operation_log = OperationLog(args.output_dir / "run.log")
        operation_log.write(
            f"backbone-fit-start layer={contract['layer']} completed={len(state['completed'])} "
            f"skipped={len(state['skipped'])} selected={len(selected_experts)}"
        )
        for entry in state["completed"]:
            validate_expert_artifact(artifacts_dir / entry["file"], entry, identity)
        completed = {entry["expert"] for entry in state["completed"]}
        skipped = {entry["expert"] for entry in state["skipped"]}
        if args.validate_only:
            operation_log.write("backbone-fit-validate-complete")
            return 0

        source = BF16LayerWeights(args.raw_dir, contract)
        target = NVFP4TargetWeights(
            args.proxy_source_dir,
            contract["layer"],
            expert_count,
            contract["architecture"]["latent_width"],
            contract["architecture"]["hidden_width"],
        )
        processed = 0
        for expert in selected_experts:
            if expert in completed or expert in skipped:
                continue
            if args.max_experts is not None and processed >= args.max_experts:
                break
            train_latent, train_weights, _, _ = expert_context_rows(
                train, expert, args.route_weight_power
            )
            validation_latent, validation_weights, validation_rows_np, validation_scores = expert_context_rows(
                validation, expert, args.route_weight_power
            )
            if (
                train_latent.shape[0] < args.min_train_routes
                or validation_latent.shape[0] < args.min_validation_routes
            ):
                reason = {
                    "expert": expert,
                    "train_routes": train_latent.shape[0],
                    "validation_routes": validation_latent.shape[0],
                    "reason": "insufficient-routes",
                }
                state["skipped"].append(reason)
                atomic_json(state_path, state)
                operation_log.write(
                    f"backbone-fit-skip expert={expert} train={train_latent.shape[0]} "
                    f"validation={validation_latent.shape[0]} reason=insufficient-routes"
                )
                continue
            operation_log.write(
                f"backbone-fit-expert-start expert={expert} train={train_latent.shape[0]} "
                f"validation={validation_latent.shape[0]}"
            )
            started = time.perf_counter()
            teacher_up, teacher_down = source.expert(expert)
            target_up, target_down = target.expert(expert)
            source_binary = source_up = source_down = candidate_up = candidate_down = None
            if args.fit_strategy == "bf16-endpoint":
                binary, metrics = fit_binary_expert(
                    teacher_up,
                    teacher_down,
                    train_latent,
                    validation_latent,
                    train_weights,
                    validation_weights,
                    args.group_size,
                    args.ridge,
                    args.endpoint_margin,
                    target_up=target_up,
                    target_down=target_down,
                )
                tier_up = teacher_up
                tier_down = teacher_down
            else:
                source_binary = BinaryExpert(
                    affine_weight(teacher_up, 1, args.group_size),
                    affine_weight(teacher_down, 1, args.group_size),
                )
                binary = BinaryExpert(
                    affine_weight(target_up.astype(mx.bfloat16), 1, args.group_size),
                    affine_weight(target_down.astype(mx.bfloat16), 1, args.group_size),
                )
                source_binary.validate()
                binary.validate()
                source_up = dequantize_affine(source_binary.up)
                source_down = dequantize_affine(source_binary.down)
                candidate_up = dequantize_affine(binary.up)
                candidate_down = dequantize_affine(binary.down)
                metrics = {}
                for label, latent_rows, sample_weights in (
                    ("train", train_latent, train_weights),
                    ("validation", validation_latent, validation_weights),
                ):
                    teacher_output = expert_output(latent_rows, target_up, target_down)
                    source_error = output_error(
                        expert_output(latent_rows, source_up, source_down),
                        teacher_output,
                        sample_weights,
                    )
                    candidate_error = output_error(
                        expert_output(latent_rows, candidate_up, candidate_down),
                        teacher_output,
                        sample_weights,
                    )
                    metrics[label] = {
                        **{f"initial_{key}": value for key, value in source_error.items()},
                        **{f"fitted_{key}": value for key, value in candidate_error.items()},
                    }
                tier_up = target_up.astype(mx.bfloat16)
                tier_down = target_down.astype(mx.bfloat16)
            refinement = None
            final_up = None
            final_down = None
            if args.refine_steps:
                binary, refinement = refine_binary_expert(
                    binary,
                    teacher_up,
                    teacher_down,
                    train_latent,
                    validation_latent,
                    train_weights,
                    validation_weights,
                    target_up=target_up,
                    target_down=target_down,
                    steps=args.refine_steps,
                    batch_size=args.refine_batch_size,
                    learning_rate=args.refine_endpoint_learning_rate,
                    code_learning_rate=args.refine_code_learning_rate,
                    temperature=args.refine_temperature,
                    logit_margin=args.refine_logit_margin,
                    code_anchor=args.refine_code_anchor,
                    endpoint_margin=args.endpoint_margin,
                    evaluate_every=args.refine_evaluate_every,
                    code_warmup_steps=args.refine_code_warmup_steps,
                )
                final_up = dequantize_affine(binary.up)
                final_down = dequantize_affine(binary.down)
                for label, latent_rows, sample_weights in (
                    ("train", train_latent, train_weights),
                    ("validation", validation_latent, validation_weights),
                ):
                    current = metrics[label]
                    current.update(
                        {
                            f"endpoint_{key.removeprefix('fitted_')}": value
                            for key, value in list(current.items())
                            if key.startswith("fitted_")
                        }
                    )
                    final_error = output_error(
                        expert_output(latent_rows, final_up, final_down),
                        expert_output(latent_rows, target_up, target_down),
                        sample_weights,
                    )
                    current.update({f"fitted_{key}": value for key, value in final_error.items()})
                operation_log.write(
                    f"backbone-fit-refine expert={expert} best_step={refinement['best_step']} "
                    f"code_flips={refinement['best_code_flips']} "
                    f"initial={refinement['initial']['relative_l2']:.6g} "
                    f"best={refinement['best']['relative_l2']:.6g}"
                )
            tiers = precision_metrics(
                tier_up,
                tier_down,
                validation_latent,
                validation_weights,
                group_size=args.group_size,
                target_up=target_up,
                target_down=target_down,
            )
            candidate_output = expert_output(
                validation_latent,
                dequantize_affine(binary.up),
                dequantize_affine(binary.down),
            )
            target_output = expert_output(validation_latent, target_up, target_down)
            validation_weighted_residual = (
                candidate_output - target_output
            ) * validation_scores[:, None]
            validation_rows = mx.array(validation_rows_np, dtype=mx.int32)
            mx.eval(validation_rows, validation_weighted_residual)
            file_name = f"expert-{expert:03d}.safetensors"
            path = artifacts_dir / file_name
            atomic_expert_artifact(
                path,
                binary,
                layer=contract["layer"],
                expert_id=expert,
                source_revision=contract["source_revision"],
                contract_sha256=identity["contract_sha256"],
                context_state_sha256=identity["context_state_sha256"],
                fit_strategy=args.fit_strategy,
                validation_rows=validation_rows,
                validation_weighted_residual=validation_weighted_residual.astype(mx.float32),
            )
            entry = {
                "expert": expert,
                "fit_strategy": args.fit_strategy,
                "candidate_plan_eligible": (
                    args.fit_strategy == "native-target-rtn"
                    or metrics["validation"]["fitted_error2"]
                    < metrics["validation"]["initial_error2"]
                ),
                "train_routes": train_latent.shape[0],
                "validation_routes": validation_latent.shape[0],
                "file": file_name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "payload_bytes": binary.payload_bytes,
                "metrics": metrics,
                "refinement": refinement,
                "precision_tiers": tiers,
                "validation_weighted_residual_error2": float(
                    mx.sum(mx.square(validation_weighted_residual.astype(mx.float32)))
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            validate_expert_artifact(path, entry, identity)
            state["completed"].append(entry)
            atomic_json(state_path, state)
            operation_log.write(
                f"backbone-fit-expert-done expert={expert} elapsed={entry['elapsed_seconds']:.2f}s "
                f"initial={metrics['validation']['initial_relative_l2']:.6g} "
                f"fitted={metrics['validation']['fitted_relative_l2']:.6g}"
            )
            processed += 1
            del (
                teacher_up,
                teacher_down,
                target_up,
                target_down,
                binary,
                final_up,
                final_down,
                source_binary,
                source_up,
                source_down,
                candidate_up,
                candidate_down,
                tier_up,
                tier_down,
                candidate_output,
                target_output,
                validation_weighted_residual,
            )
            gc.collect()
            mx.clear_cache()
        if len(state["completed"]) + len(state["skipped"]) == len(selected_experts):
            state["status"] = "complete"
            atomic_json(state_path, state)
        operation_log.write(
            f"backbone-fit-stop status={state['status']} processed={processed} "
            f"completed={len(state['completed'])} skipped={len(state['skipped'])}"
        )
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"backbone-fit-failed error={exc}")
        print(f"nemotron backbone fit error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

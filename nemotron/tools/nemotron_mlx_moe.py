#!/usr/bin/env python3
"""GPU-owned Nemotron LatentMoE execution over native ModelOpt NVFP4 tensors."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require


@dataclass
class NVFP4SwitchWeight:
    weight: mx.array
    scales: mx.array
    global_scales: mx.array

    def validate(self) -> None:
        require(self.weight.dtype == mx.uint8 and self.weight.ndim == 3, "invalid packed switch weight")
        require(self.scales.dtype == mx.uint8 and self.scales.ndim == 3, "invalid switch block scales")
        require(
            self.global_scales.dtype == mx.float32 and self.global_scales.ndim == 1,
            "invalid switch global scales",
        )
        experts, rows, packed_columns = self.weight.shape
        require(self.scales.shape == (experts, rows, packed_columns // 8), "switch scale shape mismatch")
        require(self.global_scales.shape == (experts,), "switch global-scale shape mismatch")
        require(packed_columns % 4 == 0, "packed switch columns must be uint32 aligned")

    @property
    def experts(self) -> int:
        return self.weight.shape[0]

    @property
    def input_dims(self) -> int:
        return self.weight.shape[2] * 2

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]


@dataclass
class NVFP4ExpertMLP:
    up: NVFP4SwitchWeight
    down: NVFP4SwitchWeight

    def validate(self) -> None:
        self.up.validate()
        self.down.validate()
        require(self.up.experts == self.down.experts, "up/down expert count mismatch")
        require(self.up.output_dims == self.down.input_dims, "up/down intermediate size mismatch")
        require(self.down.output_dims == self.up.input_dims, "up/down latent size mismatch")


def slice_expert_blocks(
    experts: NVFP4ExpertMLP,
    kept_blocks: np.ndarray,
    block_size: int = 16,
) -> NVFP4ExpertMLP:
    """Slice aligned expert hidden blocks without changing retained NVFP4 bytes."""

    experts.validate()
    count, blocks = kept_blocks.shape
    require(count == experts.up.experts, "block plan expert count mismatch")
    require(experts.up.output_dims % block_size == 0, "expert width is not block aligned")
    source_blocks = experts.up.output_dims // block_size
    require(np.all((0 <= kept_blocks) & (kept_blocks < source_blocks)), "block ID out of range")
    require(np.all(np.diff(kept_blocks, axis=1) > 0), "block IDs must be sorted and unique")
    down_bytes_per_block = block_size // 2
    up_weight = []
    up_scales = []
    down_weight = []
    down_scales = []
    for expert, block_ids_np in enumerate(kept_blocks):
        block_ids = mx.array(block_ids_np, dtype=mx.uint32)
        row_ids = (
            block_ids[:, None] * block_size
            + mx.arange(block_size, dtype=mx.uint32)[None, :]
        ).reshape(-1)
        up_weight.append(experts.up.weight[expert, row_ids])
        up_scales.append(experts.up.scales[expert, row_ids])
        down_weight.append(
            experts.down.weight[expert]
            .reshape(experts.down.output_dims, source_blocks, down_bytes_per_block)[:, block_ids]
            .reshape(experts.down.output_dims, blocks * down_bytes_per_block)
        )
        down_scales.append(experts.down.scales[expert][:, block_ids])
    result = NVFP4ExpertMLP(
        up=NVFP4SwitchWeight(mx.stack(up_weight), mx.stack(up_scales), experts.up.global_scales),
        down=NVFP4SwitchWeight(
            mx.stack(down_weight), mx.stack(down_scales), experts.down.global_scales
        ),
    )
    result.validate()
    return result


def switch_matmul(x: mx.array, weights: NVFP4SwitchWeight, indices: mx.array) -> mx.array:
    """Apply selected expert matrices while keeping routing and scaling on GPU."""

    weights.validate()
    require(x.ndim in (3, 4) and x.shape[-1] == weights.input_dims, "switch input shape mismatch")
    require(indices.ndim == 3 and indices.shape[:2] == x.shape[:2], "switch index shape mismatch")
    if x.ndim == 4:
        require(x.shape[2] == indices.shape[2], "per-expert switch input shape mismatch")
    require(indices.dtype in (mx.int32, mx.uint32), "switch indices must be 32-bit integers")

    packed_u32 = weights.weight.view(mx.uint32)
    selected_global = weights.global_scales[indices][..., None, None]
    expert_inputs = (
        mx.expand_dims(x, (-2, -3))
        if x.ndim == 3
        else mx.expand_dims(x, -2)
    ) * selected_global
    return mx.gather_qmm(
        expert_inputs,
        packed_u32,
        weights.scales,
        rhs_indices=indices,
        transpose=True,
        group_size=16,
        bits=4,
        mode="nvfp4",
    )


def expert_mlp(x: mx.array, weights: NVFP4ExpertMLP, indices: mx.array, scores: mx.array) -> mx.array:
    """Run selected up/ReLU-squared/down experts and weighted reduction."""

    down = expert_outputs(x, weights, indices)
    return (down * scores[..., None]).sum(axis=-2)


def expert_outputs(x: mx.array, weights: NVFP4ExpertMLP, indices: mx.array) -> mx.array:
    """Return each selected expert output before router-weighted reduction."""

    weights.validate()
    up = switch_matmul(x, weights.up, indices)
    hidden = mx.square(mx.maximum(up, mx.array(0.0, dtype=up.dtype)))
    down = switch_matmul(hidden.squeeze(-2), weights.down, indices).squeeze(-2)
    return down


def _direct_selected_mlp(
    x: mx.array, weights: NVFP4ExpertMLP, indices: mx.array, scores: mx.array
) -> mx.array:
    """Selected-tensor qmm reference used to verify gather routing."""

    require(x.shape[:2] == (1, 1), "direct selected reference requires one token")
    flat_indices = indices.reshape(-1)
    up_global = weights.up.global_scales[flat_indices]
    up = mx.quantized_matmul(
        x.reshape(-1)[None, None, :] * up_global[:, None, None],
        weights.up.weight.view(mx.uint32)[flat_indices],
        weights.up.scales[flat_indices],
        transpose=True,
        group_size=16,
        bits=4,
        mode="nvfp4",
    )
    hidden = mx.square(mx.maximum(up, mx.array(0.0, dtype=up.dtype)))
    down_global = weights.down.global_scales[flat_indices]
    down = mx.quantized_matmul(
        hidden * down_global[:, None, None],
        weights.down.weight.view(mx.uint32)[flat_indices],
        weights.down.scales[flat_indices],
        transpose=True,
        group_size=16,
        bits=4,
        mode="nvfp4",
    )
    return (down[:, 0, :] * scores.reshape(-1)[:, None]).sum(axis=0).reshape(1, 1, -1)


def load_expert_layer(source_dir: Path, layer: int) -> NVFP4ExpertMLP:
    config = load_json(source_dir / "config.json")
    pattern = config.get("hybrid_override_pattern")
    require(isinstance(pattern, str) and 0 <= layer < len(pattern), "layer index out of range")
    require(pattern[layer] == "E", f"layer {layer} is not a routed-expert layer")
    experts = config.get("n_routed_experts")
    require(isinstance(experts, int) and experts > 0, "invalid routed expert count")

    if config.get("nemotron_runtime", {}).get("format") == "nemotron-mlx-runtime-v1":
        index = load_json(source_dir / "model.safetensors.index.json")
        prefix = f"backbone.layers.{layer}.mixer.switch_mlp"
        shard_names = {
            shard for name, shard in index.get("weight_map", {}).items() if name.startswith(prefix + ".")
        }
        require(len(shard_names) == 1, f"runtime expert layer {layer} must occupy one shard")
        return load_packed_expert_file(source_dir / next(iter(shard_names)), layer)

    index = load_json(source_dir / "model.safetensors.index.json")
    base = f"backbone.layers.{layer}.mixer.experts"
    shard_names = sorted(
        {
            shard
            for name, shard in index.get("weight_map", {}).items()
            if name.startswith(base + ".")
        }
    )
    require(shard_names, f"no expert tensors found for layer {layer}")
    tensors: dict[str, mx.array] = {}
    for shard_name in shard_names:
        shard_tensors = mx.load(str(source_dir / shard_name))
        tensors.update(
            (name, value)
            for name, value in shard_tensors.items()
            if name.startswith(base + ".")
        )

    result = NVFP4ExpertMLP(
        up=_load_expert_named_projection(tensors, base, "up_proj", experts),
        down=_load_expert_named_projection(tensors, base, "down_proj", experts),
    )
    result.validate()
    return result


def load_packed_expert_file(path: Path, layer: int) -> NVFP4ExpertMLP:
    tensors = mx.load(str(path))
    base = f"backbone.layers.{layer}.mixer.switch_mlp"

    def projection(name: str) -> NVFP4SwitchWeight:
        prefix = f"{base}.{name}"
        required = [f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.global_scales"]
        for tensor_name in required:
            require(tensor_name in tensors, f"missing packed runtime tensor: {tensor_name}")
        return NVFP4SwitchWeight(
            tensors[required[0]],
            tensors[required[1]],
            tensors[required[2]].reshape(-1).astype(mx.float32),
        )

    result = NVFP4ExpertMLP(up=projection("fc1"), down=projection("fc2"))
    result.validate()
    return result


def _load_expert_named_projection(
    weights: dict[str, mx.array], base: str, projection: str, experts: int
) -> NVFP4SwitchWeight:
    packed = []
    scales = []
    global_scales = []
    for expert in range(experts):
        prefix = f"{base}.{expert}.{projection}"
        for suffix in ("weight", "weight_scale", "weight_scale_2"):
            require(f"{prefix}.{suffix}" in weights, f"missing tensor: {prefix}.{suffix}")
        packed.append(weights[f"{prefix}.weight"])
        scales.append(weights[f"{prefix}.weight_scale"])
        global_scales.append(weights[f"{prefix}.weight_scale_2"].reshape(()))
    return NVFP4SwitchWeight(mx.stack(packed), mx.stack(scales), mx.stack(global_scales))


def benchmark(source_dir: Path, layer: int, repeats: int) -> dict[str, float]:
    started = time.perf_counter()
    experts = load_expert_layer(source_dir, layer)
    experts.validate()
    mx.eval(
        experts.up.weight,
        experts.up.scales,
        experts.up.global_scales,
        experts.down.weight,
        experts.down.scales,
        experts.down.global_scales,
    )
    mx.synchronize()
    load_seconds = time.perf_counter() - started

    top_k = min(22, experts.up.experts)
    indices = mx.array([[[index * 17 % experts.up.experts for index in range(top_k)]]], dtype=mx.uint32)
    scores = mx.ones((1, 1, top_k), dtype=mx.float32) / top_k
    values = [math.sin(index * 0.013) + math.cos(index * 0.007) * 0.25 for index in range(experts.up.input_dims)]
    x = mx.array(values, dtype=mx.float32).reshape(1, 1, -1)
    output = expert_mlp(x, experts, indices, scores)
    reference = _direct_selected_mlp(x, experts, indices, scores)
    mx.eval(output, reference)
    mx.synchronize()
    difference = output - reference
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(reference)))
    relative_l2 = math.sqrt(error2 / max(reference2, 1e-30))
    max_abs = float(mx.max(mx.abs(difference)))

    started = time.perf_counter()
    outputs = [expert_mlp(x, experts, indices, scores) for _ in range(repeats)]
    mx.eval(*outputs)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "load_seconds": load_seconds,
        "active_gib": mx.get_active_memory() / 2**30,
        "peak_gib": mx.get_peak_memory() / 2**30,
        "ms": elapsed * 1000 / repeats,
        "checksum": float(outputs[-1].sum()),
        "relative_l2": relative_l2,
        "max_abs": max_abs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        result = benchmark(args.source_dir, args.layer, args.repeats)
        print(
            f"mlx moe: layer={args.layer} load={result['load_seconds']:.3f}s "
            f"active={result['active_gib']:.2f}GiB peak={result['peak_gib']:.2f}GiB "
            f"ms={result['ms']:.6f} checksum={result['checksum']:.9g} "
            f"relative_l2={result['relative_l2']:.9g} max_abs={result['max_abs']:.9g}"
        )
        require(
            result["relative_l2"] <= 2e-6 and result["max_abs"] <= 2e-5,
            "gathered MoE differs from direct selected reference",
        )
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron MLX MoE error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

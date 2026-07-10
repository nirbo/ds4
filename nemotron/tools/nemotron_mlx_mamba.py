#!/usr/bin/env python3
"""Load and execute one official Nemotron Mamba2 layer through MLX."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import fp8_matvec, fp8_matvec_custom


class ModelOptFP8Linear(nn.Module):
    def __init__(
        self,
        weight: mx.array,
        scale: mx.array,
        implementation: Callable[[mx.array, mx.array, mx.array], mx.array] = fp8_matvec,
    ):
        super().__init__()
        require(weight.dtype == mx.uint8 and weight.ndim == 2, "invalid FP8 linear weight")
        require(scale.dtype == mx.float32 and scale.size == 1, "invalid FP8 linear scale")
        self.weight = weight
        self.scale = scale.reshape(1)
        self.implementation = implementation

    def __call__(self, x: mx.array) -> mx.array:
        require(x.shape[-1] == self.weight.shape[1], "FP8 linear input shape mismatch")
        leading = math.prod(x.shape[:-1])
        require(leading == 1, "FP8 decode linear currently requires one token")
        output = self.implementation(self.weight, self.scale, x.reshape(-1).astype(mx.float32))
        return output.reshape(*x.shape[:-1], self.weight.shape[0])


def layer_tensors(source_dir: Path, layer: int) -> dict[str, mx.array]:
    index = load_json(source_dir / "model.safetensors.index.json")
    prefix = f"backbone.layers.{layer}."
    shard_names = sorted(
        {shard for name, shard in index.get("weight_map", {}).items() if name.startswith(prefix)}
    )
    require(shard_names, f"no tensors found for layer {layer}")
    result: dict[str, mx.array] = {}
    for shard_name in shard_names:
        result.update(
            (name, value)
            for name, value in mx.load(str(source_dir / shard_name)).items()
            if name.startswith(prefix) and ".experts." not in name
        )
    return result


def load_mamba_layer(
    source_dir: Path,
    layer: int,
    implementation: Callable[[mx.array, mx.array, mx.array], mx.array] = fp8_matvec,
) -> NemotronHBlock:
    config = load_json(source_dir / "config.json")
    args = ModelArgs.from_dict(config)
    require(args.hybrid_override_pattern[layer] == "M", f"layer {layer} is not Mamba2")
    tensors = layer_tensors(source_dir, layer)
    base = f"backbone.layers.{layer}"
    mixer = f"{base}.mixer"
    required = [
        f"{base}.norm.weight",
        f"{mixer}.A_log",
        f"{mixer}.D",
        f"{mixer}.dt_bias",
        f"{mixer}.conv1d.weight",
        f"{mixer}.conv1d.bias",
        f"{mixer}.norm.weight",
        f"{mixer}.in_proj.weight",
        f"{mixer}.in_proj.weight_scale",
        f"{mixer}.out_proj.weight",
        f"{mixer}.out_proj.weight_scale",
    ]
    for name in required:
        require(name in tensors, f"missing Mamba tensor: {name}")

    block = NemotronHBlock(args, "M")
    block.norm.weight = tensors[f"{base}.norm.weight"]
    block.mixer.A_log = tensors[f"{mixer}.A_log"]
    block.mixer.D = tensors[f"{mixer}.D"]
    block.mixer.dt_bias = tensors[f"{mixer}.dt_bias"]
    block.mixer.conv1d.weight = tensors[f"{mixer}.conv1d.weight"].moveaxis(2, 1)
    block.mixer.conv1d.bias = tensors[f"{mixer}.conv1d.bias"]
    block.mixer.norm.weight = tensors[f"{mixer}.norm.weight"]
    block.mixer.in_proj = ModelOptFP8Linear(
        tensors[f"{mixer}.in_proj.weight"],
        tensors[f"{mixer}.in_proj.weight_scale"],
        implementation,
    )
    block.mixer.out_proj = ModelOptFP8Linear(
        tensors[f"{mixer}.out_proj.weight"],
        tensors[f"{mixer}.out_proj.weight_scale"],
        implementation,
    )
    block.eval()
    return block


def compare_implementations(source_dir: Path, layer: int) -> dict[str, float]:
    native = load_mamba_layer(source_dir, layer, fp8_matvec)
    reference = load_mamba_layer(source_dir, layer, fp8_matvec_custom)
    hidden_size = native.norm.weight.size
    native_cache = ArraysCache(size=2)
    reference_cache = ArraysCache(size=2)
    native_output = reference_output = None
    for step in range(4):
        x = mx.array(
            [
                math.sin(index * 0.009 + step * 0.13) * 0.3
                + math.cos(index * 0.017 - step * 0.07) * 0.1
                for index in range(hidden_size)
            ],
            dtype=mx.float32,
        ).reshape(1, 1, hidden_size)
        native_output = native(x, mask=None, cache=native_cache)
        reference_output = reference(x, mask=None, cache=reference_cache)
    require(native_output is not None and reference_output is not None, "Mamba comparison produced no output")
    mx.eval(native_output, reference_output, native_cache.state, reference_cache.state)
    difference = native_output - reference_output
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(reference_output)))
    state_error = max(
        float(mx.max(mx.abs(left - right)))
        for left, right in zip(native_cache.state, reference_cache.state)
    )
    return {
        "relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "max_abs": float(mx.max(mx.abs(difference))),
        "state_max_abs": state_error,
    }


def benchmark(source_dir: Path, layer: int, repeats: int) -> dict[str, float]:
    block = load_mamba_layer(source_dir, layer)
    hidden_size = block.norm.weight.size
    x = mx.array([math.sin(index * 0.01) * 0.2 for index in range(hidden_size)], dtype=mx.float32).reshape(
        1, 1, hidden_size
    )
    cache = ArraysCache(size=2)
    warm = block(x, mask=None, cache=cache)
    mx.eval(warm, cache.state)
    mx.synchronize()

    started = time.perf_counter()
    outputs = []
    for _ in range(repeats):
        output = block(x, mask=None, cache=cache)
        outputs.append(output)
    mx.eval(*outputs, cache.state)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "ms": elapsed * 1000 / repeats,
        "checksum": float(outputs[-1].sum()),
        "active_mib": mx.get_active_memory() / 2**20,
        "peak_mib": mx.get_peak_memory() / 2**20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        comparison = compare_implementations(args.source_dir, args.layer)
        performance = benchmark(args.source_dir, args.layer, args.repeats)
        print(
            f"mlx mamba: layer={args.layer} ms={performance['ms']:.6f} "
            f"active={performance['active_mib']:.1f}MiB peak={performance['peak_mib']:.1f}MiB "
            f"relative_l2={comparison['relative_l2']:.9g} max_abs={comparison['max_abs']:.9g} "
            f"state_max_abs={comparison['state_max_abs']:.9g} checksum={performance['checksum']:.9g}"
        )
        require(
            comparison["relative_l2"] <= 2e-5
            and comparison["max_abs"] <= 2e-4
            and comparison["state_max_abs"] <= 2e-4,
            "Mamba optimized/reference drift exceeds tolerance",
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        print(f"nemotron MLX Mamba error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

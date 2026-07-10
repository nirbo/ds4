#!/usr/bin/env python3
"""Load and execute one official Nemotron full-attention layer through MLX."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mamba import layer_tensors


def load_attention_layer(source_dir: Path, layer: int) -> NemotronHBlock:
    config = load_json(source_dir / "config.json")
    args = ModelArgs.from_dict(config)
    require(args.hybrid_override_pattern[layer] == "*", f"layer {layer} is not full attention")
    tensors = layer_tensors(source_dir, layer)
    base = f"backbone.layers.{layer}"
    mixer = f"{base}.mixer"
    required = [
        f"{base}.norm.weight",
        f"{mixer}.q_proj.weight",
        f"{mixer}.k_proj.weight",
        f"{mixer}.v_proj.weight",
        f"{mixer}.o_proj.weight",
        f"{mixer}.k_proj.k_scale",
        f"{mixer}.v_proj.v_scale",
    ]
    for name in required:
        require(name in tensors, f"missing attention tensor: {name}")
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        require(tensors[f"{mixer}.{projection}.weight"].dtype == mx.bfloat16, f"attention {projection} is not BF16")

    block = NemotronHBlock(args, "*")
    block.norm.weight = tensors[f"{base}.norm.weight"]
    block.mixer.q_proj = ModelOptBF16Linear(tensors[f"{mixer}.q_proj.weight"])
    block.mixer.k_proj = ModelOptBF16Linear(tensors[f"{mixer}.k_proj.weight"])
    block.mixer.v_proj = ModelOptBF16Linear(tensors[f"{mixer}.v_proj.weight"])
    block.mixer.o_proj = ModelOptBF16Linear(tensors[f"{mixer}.o_proj.weight"])
    # k_scale/v_scale calibrate optional quantized KV caches. Official BF16
    # attention does not apply them to ordinary key/value projections.
    block.k_scale = tensors[f"{mixer}.k_proj.k_scale"]
    block.v_scale = tensors[f"{mixer}.v_proj.v_scale"]
    block.eval()
    return block


def cache_parity(source_dir: Path, layer: int, tokens: int = 4) -> dict[str, float]:
    block = load_attention_layer(source_dir, layer)
    hidden_size = block.norm.weight.size
    values = [
        [
            math.sin(index * 0.011 + token * 0.17) * 0.2
            + math.cos(index * 0.007 - token * 0.09) * 0.05
            for index in range(hidden_size)
        ]
        for token in range(tokens)
    ]
    sequence = mx.array(values, dtype=mx.float32).reshape(1, tokens, hidden_size)
    full = block(sequence, mask=create_attention_mask(sequence, None), cache=None)

    cache = KVCache()
    incremental = []
    for token in range(tokens):
        incremental.append(block(sequence[:, token : token + 1, :], mask=None, cache=cache))
    decoded = mx.concatenate(incremental, axis=1)
    mx.eval(full, decoded, cache.keys, cache.values)
    difference = full - decoded
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(full)))
    return {
        "relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "max_abs": float(mx.max(mx.abs(difference))),
    }


def benchmark(source_dir: Path, layer: int, context: int, repeats: int) -> dict[str, float]:
    block = load_attention_layer(source_dir, layer)
    hidden_size = block.norm.weight.size
    cache = KVCache()
    for token in range(context):
        x = mx.array(
            [math.sin(index * 0.01 + token * 0.03) * 0.2 for index in range(hidden_size)],
            dtype=mx.float32,
        ).reshape(1, 1, hidden_size)
        warm = block(x, mask=None, cache=cache)
    mx.eval(warm, cache.keys, cache.values)
    mx.synchronize()

    x = mx.array([math.cos(index * 0.013) * 0.2 for index in range(hidden_size)], dtype=mx.float32).reshape(
        1, 1, hidden_size
    )
    started = time.perf_counter()
    outputs = [block(x, mask=None, cache=cache) for _ in range(repeats)]
    mx.eval(*outputs, cache.keys, cache.values)
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
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.context > 0 and args.repeats > 0, "context and repeats must be positive")
        parity = cache_parity(args.source_dir, args.layer)
        performance = benchmark(args.source_dir, args.layer, args.context, args.repeats)
        print(
            f"mlx attention: layer={args.layer} context={args.context} ms={performance['ms']:.6f} "
            f"active={performance['active_mib']:.1f}MiB peak={performance['peak_mib']:.1f}MiB "
            f"relative_l2={parity['relative_l2']:.9g} max_abs={parity['max_abs']:.9g} "
            f"checksum={performance['checksum']:.9g}"
        )
        require(parity["relative_l2"] <= 2e-5 and parity["max_abs"] <= 2e-4, "attention cache parity failed")
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        print(f"nemotron MLX attention error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit input gradients through representative quantized NemotronH blocks."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_attention import load_attention_layer
from nemotron_mlx_mamba import load_mamba_layer
from nemotron_mlx_moe import expert_outputs
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_prune_materialize import atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-kd-gradient-audit-v1"


class NativeBF16Linear:
    """Gradient-capable reference for frozen BF16 inference projections."""

    def __init__(self, source):
        require(source.weight.dtype == mx.bfloat16, "native fallback requires BF16 weight")
        self.weight = source.weight

    def __call__(self, x: mx.array) -> mx.array:
        require(x.shape[-1] == self.weight.shape[1], "native BF16 input shape mismatch")
        return x.astype(mx.float32) @ self.weight.T.astype(mx.float32)


def gradient_safe_attention(block) -> None:
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(block.mixer, name, NativeBF16Linear(getattr(block.mixer, name)))


def gradient_safe_moe(block, x: mx.array) -> mx.array:
    hidden = block.norm(x)
    indices, scores = block.route(hidden)
    latent = block.fc1_latent(hidden)
    selected = expert_outputs(latent, block.experts, mx.stop_gradient(indices))
    aggregate = (selected * scores[..., None]).sum(axis=-2)
    routed = NativeBF16Linear(block.fc2_latent)(aggregate)
    shared_hidden = mx.square(mx.maximum(block.shared_up(hidden), mx.array(0.0, hidden.dtype)))
    shared = block.shared_down(shared_hidden)
    return x + routed + shared


def relative_l2(left: mx.array, right: mx.array) -> float:
    difference = left.astype(mx.float32) - right.astype(mx.float32)
    return math.sqrt(
        float(mx.sum(mx.square(difference)))
        / max(float(mx.sum(mx.square(right.astype(mx.float32)))), 1e-30)
    )


def gradient_row(label: str, production, differentiable, x: mx.array) -> dict:
    reference = production(x)
    candidate = differentiable(x)

    def loss(value):
        output = differentiable(value)
        return mx.mean(mx.square(output.astype(mx.float32)))

    value, gradient = mx.value_and_grad(loss)(x)
    mx.eval(reference, candidate, value, gradient)
    row = {
        "kind": label,
        "loss": float(value),
        "gradient_norm": float(mx.linalg.norm(gradient)),
        "gradient_finite": bool(np.isfinite(np.asarray(gradient)).all()),
        "forward_relative_l2": relative_l2(candidate, reference),
        "forward_max_abs": float(mx.max(mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32)))),
    }
    require(row["gradient_finite"] and row["gradient_norm"] > 0, f"{label} gradient failed")
    require(row["forward_relative_l2"] <= 2e-5, f"{label} fallback parity failed")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.tokens >= 2, "gradient audit requires at least two tokens")
        state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        hidden = config["hidden_size"]
        values = mx.arange(args.tokens * hidden, dtype=mx.float32).reshape(1, args.tokens, hidden)
        x = mx.sin(values * 0.0013) * 0.02 + mx.cos(values * 0.0007) * 0.01
        rows = []

        mamba = load_mamba_layer(args.source_dir, 0)
        rows.append(
            gradient_row(
                "mamba",
                lambda value: mamba(value, mask=None, cache=None),
                lambda value: mamba(value, mask=None, cache=None),
                x,
            )
        )
        del mamba
        gc.collect()
        mx.clear_cache()

        attention = load_attention_layer(args.source_dir, 7)
        production_attention = load_attention_layer(args.source_dir, 7)
        gradient_safe_attention(attention)
        rows.append(
            gradient_row(
                "attention",
                lambda value: production_attention(
                    value, mask=create_attention_mask(value, None), cache=None
                ),
                lambda value: attention(value, mask=create_attention_mask(value, None), cache=None),
                x,
            )
        )
        del attention, production_attention
        gc.collect()
        mx.clear_cache()

        moe = load_moe_layer(args.source_dir, 1)
        production_moe = load_moe_layer(args.source_dir, 1)
        rows.append(
            gradient_row(
                "moe",
                production_moe,
                lambda value: gradient_safe_moe(moe, value),
                x,
            )
        )
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": state["revision"],
            "tool_sha256": sha256_file(Path(__file__)),
            "tokens": args.tokens,
            "rows": rows,
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"kd-gradient-audit path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron KD gradient audit error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Execute a complete official Nemotron LatentMoE layer through MLX."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nemotron_h import ModelArgs, group_expert_select

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import (
    ModelOptBF16Linear,
    ModelOptFP8Linear,
    ModelOptNVFP4Linear,
    fp8_matvec,
    fp8_matvec_custom,
    nvfp4_matvec,
    nvfp4_matvec_custom,
)
from nemotron_mlx_mamba import layer_tensors
from nemotron_mlx_moe import NVFP4ExpertMLP, expert_outputs, load_expert_layer


def load_linear(tensors: dict[str, mx.array], prefix: str, fp8_impl=fp8_matvec, nvfp4_impl=nvfp4_matvec):
    weight_name = f"{prefix}.weight"
    require(weight_name in tensors, f"missing linear weight: {weight_name}")
    weight = tensors[weight_name]
    if weight.dtype == mx.bfloat16:
        return ModelOptBF16Linear(weight)
    require(weight.dtype == mx.uint8, f"unsupported linear dtype for {weight_name}: {weight.dtype}")
    scale_name = f"{prefix}.weight_scale"
    require(scale_name in tensors, f"missing linear scale: {scale_name}")
    if f"{prefix}.weight_scale_2" in tensors:
        return ModelOptNVFP4Linear(
            weight,
            tensors[scale_name],
            tensors[f"{prefix}.weight_scale_2"],
            nvfp4_impl,
        )
    return ModelOptFP8Linear(weight, tensors[scale_name], fp8_impl)


class NemotronLatentMoELayer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer: int,
        tensors: dict[str, mx.array],
        experts: NVFP4ExpertMLP,
        fp8_impl=fp8_matvec,
        nvfp4_impl=nvfp4_matvec,
    ):
        super().__init__()
        base = f"backbone.layers.{layer}"
        mixer = f"{base}.mixer"
        required = [
            f"{base}.norm.weight",
            f"{mixer}.gate.weight",
            f"{mixer}.gate.e_score_correction_bias",
        ]
        for name in required:
            require(name in tensors, f"missing MoE tensor: {name}")
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.norm.weight = tensors[f"{base}.norm.weight"]
        self.gate_weight = tensors[f"{mixer}.gate.weight"]
        self.correction_bias = tensors[f"{mixer}.gate.e_score_correction_bias"]
        require(
            experts.up.experts == self.gate_weight.shape[0] == self.correction_bias.shape[0],
            "MoE expert/router count mismatch",
        )
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.layer = layer
        self.fc1_latent = load_linear(tensors, f"{mixer}.fc1_latent_proj", fp8_impl, nvfp4_impl)
        self.fc2_latent = load_linear(tensors, f"{mixer}.fc2_latent_proj", fp8_impl, nvfp4_impl)
        self.shared_up = load_linear(tensors, f"{mixer}.shared_experts.up_proj", fp8_impl, nvfp4_impl)
        self.shared_down = load_linear(tensors, f"{mixer}.shared_experts.down_proj", fp8_impl, nvfp4_impl)
        self.experts = experts

    def route(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        return group_expert_select(
            hidden @ self.gate_weight.T,
            self.correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )

    def route_retained(
        self, hidden: mx.array, retained: list[int]
    ) -> tuple[mx.array, mx.array]:
        require(self.n_group == 1 and self.topk_group == 1, "retained routing requires one group")
        require(self.top_k <= len(retained) <= self.experts.up.experts, "invalid retained expert set")
        require(retained == sorted(set(retained)), "retained experts must be sorted and unique")
        retained_indices = mx.array(retained, dtype=mx.uint32)
        local_indices, scores = group_expert_select(
            hidden @ self.gate_weight[retained_indices].T,
            self.correction_bias[retained_indices],
            self.top_k,
            1,
            1,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        return retained_indices[local_indices], scores

    def forward_with_route(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        output, indices, scores, _ = self.forward_with_observation(x)
        return output, indices, scores

    def forward_with_observation(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        output, indices, scores, output_norms, _ = self.forward_with_expert_outputs(x)
        return output, indices, scores, output_norms

    def forward_with_expert_outputs(
        self,
        x: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        hidden = self.norm(x)
        indices, scores = self.route(hidden)
        return self._forward_with_selected(x, hidden, indices, scores)

    def forward_with_retained(
        self,
        x: mx.array,
        retained: list[int],
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        hidden = self.norm(x)
        indices, scores = self.route_retained(hidden, retained)
        output, indices, scores, output_norms, _ = self._forward_with_selected(
            x, hidden, indices, scores
        )
        return output, indices, scores, output_norms

    def _forward_with_selected(
        self,
        x: mx.array,
        hidden: mx.array,
        indices: mx.array,
        scores: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        latent = self.fc1_latent(hidden)
        selected_outputs = expert_outputs(latent, self.experts, indices)
        output_norms = mx.sqrt(mx.sum(mx.square(selected_outputs.astype(mx.float32)), axis=-1))
        routed = self.fc2_latent((selected_outputs * scores[..., None]).sum(axis=-2))
        shared_hidden = mx.square(mx.maximum(self.shared_up(hidden), mx.array(0.0, dtype=hidden.dtype)))
        shared = self.shared_down(shared_hidden)
        return x + routed + shared, indices, scores, output_norms, selected_outputs

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward_with_route(x)[0]


def load_moe_layer(
    source_dir: Path,
    layer: int,
    *,
    experts: NVFP4ExpertMLP | None = None,
    fp8_impl=fp8_matvec,
    nvfp4_impl=nvfp4_matvec,
) -> NemotronLatentMoELayer:
    config = load_json(source_dir / "config.json")
    args = ModelArgs.from_dict(config)
    require(args.hybrid_override_pattern[layer] == "E", f"layer {layer} is not LatentMoE")
    tensors = layer_tensors(source_dir, layer)
    experts = experts or load_expert_layer(source_dir, layer)
    result = NemotronLatentMoELayer(args, layer, tensors, experts, fp8_impl, nvfp4_impl)
    result.eval()
    return result


def compare_implementations(source_dir: Path, layer: int) -> dict[str, float]:
    experts = load_expert_layer(source_dir, layer)
    native = load_moe_layer(source_dir, layer, experts=experts)
    reference = load_moe_layer(
        source_dir,
        layer,
        experts=experts,
        fp8_impl=fp8_matvec_custom,
        nvfp4_impl=nvfp4_matvec_custom,
    )
    hidden_size = native.norm.weight.size
    x = mx.array(
        [math.sin(index * 0.013) * 0.25 + math.cos(index * 0.019) * 0.08 for index in range(hidden_size)],
        dtype=mx.float32,
    ).reshape(1, 1, hidden_size)
    native_output = native(x)
    reference_output = reference(x)
    mx.eval(native_output, reference_output)
    difference = native_output - reference_output
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(reference_output)))
    return {
        "relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "max_abs": float(mx.max(mx.abs(difference))),
    }


def benchmark(source_dir: Path, layer: int, repeats: int) -> dict[str, float]:
    block = load_moe_layer(source_dir, layer)
    hidden_size = block.norm.weight.size
    x = mx.array([math.sin(index * 0.01) * 0.2 for index in range(hidden_size)], dtype=mx.float32).reshape(
        1, 1, hidden_size
    )
    warm = block(x)
    mx.eval(warm)
    mx.synchronize()
    started = time.perf_counter()
    outputs = [block(x) for _ in range(repeats)]
    mx.eval(*outputs)
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
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        comparison = compare_implementations(args.source_dir, args.layer)
        performance = benchmark(args.source_dir, args.layer, args.repeats)
        print(
            f"mlx moe layer: layer={args.layer} ms={performance['ms']:.6f} "
            f"active={performance['active_mib']:.1f}MiB peak={performance['peak_mib']:.1f}MiB "
            f"relative_l2={comparison['relative_l2']:.9g} max_abs={comparison['max_abs']:.9g} "
            f"checksum={performance['checksum']:.9g}"
        )
        require(
            comparison["relative_l2"] <= 3e-5 and comparison["max_abs"] <= 5e-4,
            "MoE layer optimized/reference drift exceeds tolerance",
        )
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        print(f"nemotron MLX MoE layer error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

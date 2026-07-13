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
    bf16_batch_matmul,
    bf16_matvec,
    fp8_matvec,
    fp8_matvec_custom,
    nvfp4_matvec,
    nvfp4_matvec_custom,
)
from nemotron_mlx_mamba import layer_tensors
from nemotron_mlx_moe import (
    NVFP4ExpertMLP,
    NVFP4SwitchWeight,
    expert_outputs,
    load_expert_layer,
    switch_matmul,
)


def _compiled_bf16_linear(weight: mx.array, x: mx.array) -> mx.array:
    leading = math.prod(x.shape[:-1])
    if leading == 1:
        output = bf16_matvec(weight, x.reshape(-1).astype(mx.float32))[None, :]
    else:
        output = mx.concatenate(
            [
                bf16_batch_matmul(
                    weight,
                    x.reshape(leading, weight.shape[1]).astype(mx.float32)[start : start + 32],
                )
                for start in range(0, leading, 32)
            ],
            axis=0,
        )
    return output.reshape(*x.shape[:-1], weight.shape[0])


def _compiled_fp8_linear(weight: mx.array, scale: mx.array, x: mx.array) -> mx.array:
    leading = math.prod(x.shape[:-1])
    rows, columns = weight.shape
    unity_scales = mx.full((rows, columns // 32), 127, dtype=mx.uint8)
    output = mx.quantized_matmul(
        x.reshape(leading, columns).astype(mx.float32) * scale.reshape(()),
        weight.view(mx.uint32),
        unity_scales,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )
    return output.reshape(*x.shape[:-1], rows)


def _compiled_nvfp4_linear(
    weight: mx.array,
    scales: mx.array,
    global_scale: mx.array,
    x: mx.array,
) -> mx.array:
    leading = math.prod(x.shape[:-1])
    rows = weight.shape[0]
    columns = weight.shape[1] * 2
    output = mx.quantized_matmul(
        x.reshape(leading, columns).astype(mx.float32) * global_scale.reshape(()),
        weight.view(mx.uint32),
        scales,
        transpose=True,
        group_size=16,
        bits=4,
        mode="nvfp4",
    )
    return output.reshape(*x.shape[:-1], rows)


@mx.compile
def _compiled_bf16_bf16_fp8_fp8_tail(
    x: mx.array,
    hidden: mx.array,
    indices: mx.array,
    scores: mx.array,
    fc1_weight: mx.array,
    fc2_weight: mx.array,
    shared_up_weight: mx.array,
    shared_up_scale: mx.array,
    shared_down_weight: mx.array,
    shared_down_scale: mx.array,
    expert_up_weight: mx.array,
    expert_up_scales: mx.array,
    expert_up_global_scales: mx.array,
    expert_down_weight: mx.array,
    expert_down_scales: mx.array,
    expert_down_global_scales: mx.array,
) -> mx.array:
    latent = _compiled_bf16_linear(fc1_weight, hidden)
    up = switch_matmul(
        latent,
        NVFP4SwitchWeight(expert_up_weight, expert_up_scales, expert_up_global_scales),
        indices,
    )
    expert_hidden = mx.square(mx.maximum(up, mx.array(0.0, dtype=up.dtype)))
    selected = switch_matmul(
        expert_hidden.squeeze(-2),
        NVFP4SwitchWeight(
            expert_down_weight,
            expert_down_scales,
            expert_down_global_scales,
        ),
        indices,
    ).squeeze(-2)
    routed = _compiled_bf16_linear(
        fc2_weight,
        (selected * scores[..., None]).sum(axis=-2),
    )
    shared_hidden = mx.square(
        mx.maximum(
            _compiled_fp8_linear(shared_up_weight, shared_up_scale, hidden),
            mx.array(0.0, dtype=hidden.dtype),
        )
    )
    shared = _compiled_fp8_linear(shared_down_weight, shared_down_scale, shared_hidden)
    return x + routed + shared


_BF16 = 0
_FP8 = 1
_NVFP4 = 2
_DUMMY_BLOCK_SCALES = mx.array([0], dtype=mx.uint8)
_DUMMY_GLOBAL_SCALE = mx.array([1.0], dtype=mx.float32)


def _compiled_mixed_linear(
    kind: int,
    weight: mx.array,
    scales: mx.array,
    global_scale: mx.array,
    x: mx.array,
) -> mx.array:
    if kind == _BF16:
        return _compiled_bf16_linear(weight, x)
    if kind == _FP8:
        return _compiled_fp8_linear(weight, global_scale, x)
    return _compiled_nvfp4_linear(weight, scales, global_scale, x)


def _make_compiled_mixed_tail(signature: tuple[int, int, int, int]):
    def tail(
        x: mx.array,
        hidden: mx.array,
        indices: mx.array,
        scores: mx.array,
        fc1_weight: mx.array,
        fc1_scales: mx.array,
        fc1_global_scale: mx.array,
        fc2_weight: mx.array,
        fc2_scales: mx.array,
        fc2_global_scale: mx.array,
        shared_up_weight: mx.array,
        shared_up_scales: mx.array,
        shared_up_global_scale: mx.array,
        shared_down_weight: mx.array,
        shared_down_scales: mx.array,
        shared_down_global_scale: mx.array,
        expert_up_weight: mx.array,
        expert_up_scales: mx.array,
        expert_up_global_scales: mx.array,
        expert_down_weight: mx.array,
        expert_down_scales: mx.array,
        expert_down_global_scales: mx.array,
    ) -> mx.array:
        latent = _compiled_mixed_linear(
            signature[0],
            fc1_weight,
            fc1_scales,
            fc1_global_scale,
            hidden,
        )
        up = switch_matmul(
            latent,
            NVFP4SwitchWeight(
                expert_up_weight,
                expert_up_scales,
                expert_up_global_scales,
            ),
            indices,
        )
        expert_hidden = mx.square(mx.maximum(up, mx.array(0.0, dtype=up.dtype)))
        selected = switch_matmul(
            expert_hidden.squeeze(-2),
            NVFP4SwitchWeight(
                expert_down_weight,
                expert_down_scales,
                expert_down_global_scales,
            ),
            indices,
        ).squeeze(-2)
        routed = _compiled_mixed_linear(
            signature[1],
            fc2_weight,
            fc2_scales,
            fc2_global_scale,
            (selected * scores[..., None]).sum(axis=-2),
        )
        shared_hidden = mx.square(
            mx.maximum(
                _compiled_mixed_linear(
                    signature[2],
                    shared_up_weight,
                    shared_up_scales,
                    shared_up_global_scale,
                    hidden,
                ),
                mx.array(0.0, dtype=hidden.dtype),
            )
        )
        shared = _compiled_mixed_linear(
            signature[3],
            shared_down_weight,
            shared_down_scales,
            shared_down_global_scale,
            shared_hidden,
        )
        return x + routed + shared

    return mx.compile(tail)


_COMPILED_MIXED_TAILS = {
    signature: _make_compiled_mixed_tail(signature)
    for signature in (
        (_FP8, _BF16, _FP8, _NVFP4),
        (_FP8, _FP8, _FP8, _FP8),
        (_FP8, _BF16, _FP8, _FP8),
        (_BF16, _BF16, _FP8, _BF16),
        (_BF16, _BF16, _BF16, _FP8),
    )
}


def _linear_kind(linear: nn.Module) -> int:
    if isinstance(linear, ModelOptBF16Linear):
        return _BF16
    if isinstance(linear, ModelOptFP8Linear):
        return _FP8
    require(isinstance(linear, ModelOptNVFP4Linear), "unsupported compiled linear type")
    return _NVFP4


def _linear_arguments(linear: nn.Module) -> tuple[mx.array, mx.array, mx.array]:
    if isinstance(linear, ModelOptBF16Linear):
        return linear.weight, _DUMMY_BLOCK_SCALES, _DUMMY_GLOBAL_SCALE
    if isinstance(linear, ModelOptFP8Linear):
        return linear.weight, _DUMMY_BLOCK_SCALES, linear.scale
    require(isinstance(linear, ModelOptNVFP4Linear), "unsupported compiled linear type")
    return linear.weight, linear.scales, linear.global_scale


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
        precision_signature = tuple(
            _linear_kind(linear)
            for linear in (
                self.fc1_latent,
                self.fc2_latent,
                self.shared_up,
                self.shared_down,
            )
        )
        dominant_compiled = (
            isinstance(self.fc1_latent, ModelOptBF16Linear)
            and isinstance(self.fc2_latent, ModelOptBF16Linear)
            and isinstance(self.shared_up, ModelOptFP8Linear)
            and self.shared_up.implementation is fp8_matvec
            and isinstance(self.shared_down, ModelOptFP8Linear)
            and self.shared_down.implementation is fp8_matvec
        )
        native_implementations = all(
            not isinstance(linear, (ModelOptFP8Linear, ModelOptNVFP4Linear))
            or linear.implementation in (fp8_matvec, nvfp4_matvec)
            for linear in (
                self.fc1_latent,
                self.fc2_latent,
                self.shared_up,
                self.shared_down,
            )
        )
        if dominant_compiled:
            self.compiled_tail = _compiled_bf16_bf16_fp8_fp8_tail
        elif native_implementations:
            self.compiled_tail = _COMPILED_MIXED_TAILS.get(precision_signature)
        else:
            self.compiled_tail = None

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

    def route_retained_gate(
        self,
        hidden: mx.array,
        retained: list[int],
        gate_weight: mx.array,
    ) -> tuple[mx.array, mx.array]:
        require(self.n_group == 1 and self.topk_group == 1, "retained routing requires one group")
        require(self.top_k <= len(retained) <= self.experts.up.experts, "invalid retained expert set")
        require(retained == sorted(set(retained)), "retained experts must be sorted and unique")
        require(
            gate_weight.shape == (len(retained), self.gate_weight.shape[1]),
            "retained router override shape mismatch",
        )
        retained_indices = mx.array(retained, dtype=mx.uint32)
        local_indices, scores = group_expert_select(
            hidden @ gate_weight.T,
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

    def forward_with_retained_gate(
        self,
        x: mx.array,
        retained: list[int],
        gate_weight: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        hidden = self.norm(x)
        indices, scores = self.route_retained_gate(hidden, retained, gate_weight)
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
        if self.compiled_tail is not None:
            hidden = self.norm(x)
            indices, scores = self.route(hidden)
            if self.compiled_tail is _compiled_bf16_bf16_fp8_fp8_tail:
                return self.compiled_tail(
                    x,
                    hidden,
                    indices,
                    scores,
                    self.fc1_latent.weight,
                    self.fc2_latent.weight,
                    self.shared_up.weight,
                    self.shared_up.scale,
                    self.shared_down.weight,
                    self.shared_down.scale,
                    self.experts.up.weight,
                    self.experts.up.scales,
                    self.experts.up.global_scales,
                    self.experts.down.weight,
                    self.experts.down.scales,
                    self.experts.down.global_scales,
                )
            return self.compiled_tail(
                x,
                hidden,
                indices,
                scores,
                *_linear_arguments(self.fc1_latent),
                *_linear_arguments(self.fc2_latent),
                *_linear_arguments(self.shared_up),
                *_linear_arguments(self.shared_down),
                self.experts.up.weight,
                self.experts.up.scales,
                self.experts.up.global_scales,
                self.experts.down.weight,
                self.experts.down.scales,
                self.experts.down.global_scales,
            )
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


def comparison_input(hidden_size: int, tokens: int) -> mx.array:
    return mx.array(
        [
            [
                math.sin(index * 0.013 + token * 0.17) * 0.25
                + math.cos(index * 0.019 - token * 0.11) * 0.08
                for index in range(hidden_size)
            ]
            for token in range(tokens)
        ],
        dtype=mx.float32,
    ).reshape(1, tokens, hidden_size)


def compare_implementations(source_dir: Path, layer: int, tokens: int = 1) -> dict[str, float]:
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
    x = comparison_input(hidden_size, tokens)
    native_output = native(x)
    reference_output = mx.concatenate(
        [reference(x[:, token : token + 1]) for token in range(tokens)],
        axis=1,
    )
    mx.eval(native_output, reference_output)
    difference = native_output - reference_output
    error2 = float(mx.sum(mx.square(difference)))
    reference2 = float(mx.sum(mx.square(reference_output)))
    return {
        "relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "max_abs": float(mx.max(mx.abs(difference))),
    }


def benchmark(source_dir: Path, layer: int, repeats: int, tokens: int = 1) -> dict[str, float]:
    block = load_moe_layer(source_dir, layer)
    hidden_size = block.norm.weight.size
    x = comparison_input(hidden_size, tokens)
    warm = block(x)
    mx.eval(warm)
    mx.synchronize()
    hidden = block.norm(x)
    warm_route = block.route(hidden)
    mx.eval(*warm_route)
    mx.synchronize()
    started = time.perf_counter()
    routes = [block.route(hidden) for _ in range(repeats)]
    mx.eval(*(value for route in routes for value in route))
    mx.synchronize()
    route_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    outputs = [block(x) for _ in range(repeats)]
    mx.eval(*outputs)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "ms": elapsed * 1000 / repeats,
        "route_ms": route_elapsed * 1000 / repeats,
        "checksum": float(outputs[-1].sum()),
        "active_mib": mx.get_active_memory() / 2**20,
        "peak_mib": mx.get_peak_memory() / 2**20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "repeats must be positive")
        require(1 <= args.tokens <= 8, "token count must be between one and eight")
        comparison = compare_implementations(args.source_dir, args.layer, args.tokens)
        performance = benchmark(args.source_dir, args.layer, args.repeats, args.tokens)
        print(
            f"mlx moe layer: layer={args.layer} tokens={args.tokens} "
            f"ms={performance['ms']:.6f} route_ms={performance['route_ms']:.6f} "
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

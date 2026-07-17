#!/usr/bin/env python3
"""GPU-owned top-k packed-NVFP4 MoE block for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ornith35_mlx_dense import token_tiled_matvec
from ornith35_mlx_nvfp4 import (
    nvfp4_batched_matvec,
    nvfp4_batched_paired_matvec,
    nvfp4_batched_selected_paired_matvec,
    nvfp4_batched_selected_weighted_matvec,
    nvfp4_matvec,
    nvfp4_paired_matvec,
    nvfp4_selected_matvec,
    nvfp4_selected_paired_matvec,
    nvfp4_selected_shared_gate_up_silu,
    nvfp4_selected_shared_weighted_rows4_matvec,
    nvfp4_selected_weighted_rows4_matvec,
    nvfp4_selected_weighted_matvec,
)
from ornith35_moe_reference import MoEConfig, require
from ornith35_nvfp4 import NVFP4Weight, SafetensorsFile


PRODUCTION_CONFIG = MoEConfig(
    hidden_size=2048,
    intermediate_size=512,
    num_experts=256,
    top_k=8,
)


@dataclass(frozen=True)
class NVFP4Arrays:
    packed: mx.array
    scales: mx.array
    global_scale: mx.array


@dataclass(frozen=True)
class NVFP4Stack:
    packed: mx.array
    scales: mx.array
    global_scale: mx.array


@dataclass(frozen=True)
class ExpertArrays:
    gate: NVFP4Arrays
    up: NVFP4Arrays
    down: NVFP4Arrays


@dataclass(frozen=True)
class ExpertStack:
    gate: NVFP4Stack
    up: NVFP4Stack
    down: NVFP4Stack


@dataclass(frozen=True)
class MLXMoEWeights:
    router_shared: mx.array
    experts: ExpertStack
    shared_expert: ExpertArrays

    @property
    def router(self) -> mx.array:
        return self.router_shared[:-1]

    @property
    def shared_gate(self) -> mx.array:
        return self.router_shared[-1:]


@dataclass(frozen=True)
class MLXMoEResult:
    output: mx.array
    selected_experts: mx.array
    routing_weights: mx.array


def _validate_single(weight: NVFP4Arrays, rows: int, columns: int, name: str) -> None:
    require(weight.packed.dtype == mx.uint8, f"{name} packed dtype mismatch")
    require(weight.scales.dtype == mx.uint8, f"{name} scale dtype mismatch")
    require(weight.global_scale.dtype == mx.float32, f"{name} global dtype mismatch")
    require(weight.packed.shape == (rows, columns // 2), f"{name} packed shape mismatch")
    require(weight.scales.shape == (rows, columns // 16), f"{name} scale shape mismatch")
    require(weight.global_scale.shape == (1,), f"{name} global shape mismatch")


def _validate_stack(
    weight: NVFP4Stack,
    experts: int,
    rows: int,
    columns: int,
    name: str,
) -> None:
    require(weight.packed.dtype == mx.uint8, f"{name} packed dtype mismatch")
    require(weight.scales.dtype == mx.uint8, f"{name} scale dtype mismatch")
    require(weight.global_scale.dtype == mx.float32, f"{name} global dtype mismatch")
    require(
        weight.packed.shape == (experts, rows, columns // 2),
        f"{name} packed shape mismatch",
    )
    require(
        weight.scales.shape == (experts, rows, columns // 16),
        f"{name} scale shape mismatch",
    )
    require(weight.global_scale.shape == (experts,), f"{name} global shape mismatch")


def validate_weights(weights: MLXMoEWeights, config: MoEConfig) -> None:
    require(weights.router.shape == (config.num_experts, config.hidden_size), "router shape mismatch")
    require(weights.shared_gate.shape == (1, config.hidden_size), "shared gate shape mismatch")
    require(weights.router.dtype in (mx.bfloat16, mx.float32), "invalid router dtype")
    require(weights.shared_gate.dtype == weights.router.dtype, "shared gate dtype mismatch")
    require(
        weights.router_shared.shape
        == (config.num_experts + 1, config.hidden_size),
        "combined router/shared-gate shape mismatch",
    )
    require(
        weights.router_shared.dtype == weights.router.dtype,
        "combined router/shared-gate dtype mismatch",
    )
    _validate_stack(
        weights.experts.gate,
        config.num_experts,
        config.intermediate_size,
        config.hidden_size,
        "expert gate",
    )
    _validate_stack(
        weights.experts.up,
        config.num_experts,
        config.intermediate_size,
        config.hidden_size,
        "expert up",
    )
    _validate_stack(
        weights.experts.down,
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
        "expert down",
    )
    _validate_single(
        weights.shared_expert.gate,
        config.intermediate_size,
        config.hidden_size,
        "shared gate projection",
    )
    _validate_single(
        weights.shared_expert.up,
        config.intermediate_size,
        config.hidden_size,
        "shared up projection",
    )
    _validate_single(
        weights.shared_expert.down,
        config.hidden_size,
        config.intermediate_size,
        "shared down projection",
    )


def _silu(value: mx.array) -> mx.array:
    return value * mx.sigmoid(value)


def _route_token(logits: mx.array, top_k: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    selected = mx.argsort(probabilities)[-top_k:][::-1]
    routing = mx.take(probabilities, selected)
    return selected, (routing / mx.sum(routing)).astype(dtype)


def _route_batch(logits: mx.array, top_k: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    selected = mx.argsort(probabilities, axis=-1)[:, -top_k:][:, ::-1]
    routing = mx.take_along_axis(probabilities, selected, axis=-1)
    return selected, (routing / mx.sum(routing, axis=-1, keepdims=True)).astype(dtype)


def forward(
    hidden: mx.array,
    weights: MLXMoEWeights,
    config: MoEConfig = PRODUCTION_CONFIG,
    *,
    paired_gate_up: bool = True,
    fused_shared_gate: bool = True,
    fused_routed_down: bool = True,
    _validated: bool = False,
) -> MLXMoEResult:
    """Route and evaluate one token without a CPU expert-selection boundary."""
    require(hidden.ndim == 1 and hidden.shape == (config.hidden_size,), "hidden-state shape mismatch")
    if not _validated:
        validate_weights(weights, config)
    model_dtype = weights.router.dtype
    hidden = hidden.astype(model_dtype)
    combined_route = fused_shared_gate
    if combined_route:
        router_shared = mx.matmul(weights.router_shared, hidden)
        logits = router_shared[: config.num_experts]
        shared_multiplier = mx.sigmoid(router_shared[config.num_experts]).reshape(())
    else:
        logits = mx.matmul(weights.router, hidden)
    selected, routing = _route_token(logits, config.top_k, model_dtype)
    hidden32 = hidden.astype(mx.float32)

    fused_gate_up = paired_gate_up and model_dtype == mx.bfloat16
    if fused_gate_up:
        intermediate, shared_intermediate = nvfp4_selected_shared_gate_up_silu(
            weights.experts.gate.packed,
            weights.experts.gate.scales,
            weights.experts.gate.global_scale,
            weights.experts.up.packed,
            weights.experts.up.scales,
            weights.experts.up.global_scale,
            weights.shared_expert.gate.packed,
            weights.shared_expert.gate.scales,
            weights.shared_expert.gate.global_scale,
            weights.shared_expert.up.packed,
            weights.shared_expert.up.scales,
            weights.shared_expert.up.global_scale,
            selected,
            hidden32,
        )
    elif paired_gate_up:
        gate_up = nvfp4_selected_paired_matvec(
            weights.experts.gate.packed,
            weights.experts.gate.scales,
            weights.experts.gate.global_scale,
            weights.experts.up.packed,
            weights.experts.up.scales,
            weights.experts.up.global_scale,
            selected,
            hidden32,
        ).astype(model_dtype)
        gate = gate_up[:, 0]
        up = gate_up[:, 1]
    else:
        gate = nvfp4_selected_matvec(
            weights.experts.gate.packed,
            weights.experts.gate.scales,
            weights.experts.gate.global_scale,
            selected,
            hidden32,
            batched_input=False,
        ).astype(model_dtype)
        up = nvfp4_selected_matvec(
            weights.experts.up.packed,
            weights.experts.up.scales,
            weights.experts.up.global_scale,
            selected,
            hidden32,
            batched_input=False,
        ).astype(model_dtype)
    if not fused_gate_up:
        intermediate = _silu(gate) * up
        if paired_gate_up:
            shared_gate_up = nvfp4_paired_matvec(
                weights.shared_expert.gate.packed,
                weights.shared_expert.gate.scales,
                weights.shared_expert.gate.global_scale,
                weights.shared_expert.up.packed,
                weights.shared_expert.up.scales,
                weights.shared_expert.up.global_scale,
                hidden32,
            ).astype(model_dtype)
            shared_gate = shared_gate_up[0]
            shared_up = shared_gate_up[1]
        else:
            shared_gate = nvfp4_matvec(
                weights.shared_expert.gate.packed,
                weights.shared_expert.gate.scales,
                weights.shared_expert.gate.global_scale,
                hidden32,
            ).astype(model_dtype)
            shared_up = nvfp4_matvec(
                weights.shared_expert.up.packed,
                weights.shared_expert.up.scales,
                weights.shared_expert.up.global_scale,
                hidden32,
            ).astype(model_dtype)
        shared_intermediate = _silu(shared_gate) * shared_up
    if not combined_route:
        shared_multiplier = mx.sigmoid(mx.matmul(weights.shared_gate, hidden)).reshape(())
    if fused_routed_down and model_dtype == mx.bfloat16:
        output = nvfp4_selected_shared_weighted_rows4_matvec(
            weights.experts.down.packed,
            weights.experts.down.scales,
            weights.experts.down.global_scale,
            weights.shared_expert.down.packed,
            weights.shared_expert.down.scales,
            weights.shared_expert.down.global_scale,
            selected,
            intermediate.astype(mx.float32),
            shared_intermediate.astype(mx.float32),
            routing,
            shared_multiplier,
        )
    else:
        if fused_routed_down:
            routed = nvfp4_selected_weighted_rows4_matvec(
                weights.experts.down.packed,
                weights.experts.down.scales,
                weights.experts.down.global_scale,
                selected,
                intermediate.astype(mx.float32),
                routing,
            )
        else:
            down = nvfp4_selected_matvec(
                weights.experts.down.packed,
                weights.experts.down.scales,
                weights.experts.down.global_scale,
                selected,
                intermediate.astype(mx.float32),
                batched_input=True,
            ).astype(model_dtype)
            routed = mx.sum(
                down.astype(mx.float32) * routing.astype(mx.float32)[:, None],
                axis=0,
            ).astype(model_dtype)
        shared = nvfp4_matvec(
            weights.shared_expert.down.packed,
            weights.shared_expert.down.scales,
            weights.shared_expert.down.global_scale,
            shared_intermediate.astype(mx.float32),
        ).astype(model_dtype)
        output = (routed + shared * shared_multiplier).astype(model_dtype)
    return MLXMoEResult(
        output=output,
        selected_experts=selected,
        routing_weights=routing,
    )


def forward_batch(
    hidden: mx.array,
    weights: MLXMoEWeights,
    config: MoEConfig = PRODUCTION_CONFIG,
    *,
    fused_shared_gate: bool = True,
    token_tiled_shared: bool = True,
    direct_bf16_inputs: bool = True,
) -> MLXMoEResult:
    """Route and evaluate a nonempty token matrix entirely on the GPU."""
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0 and hidden.shape[1] == config.hidden_size,
        "batched hidden-state shape mismatch",
    )
    validate_weights(weights, config)
    model_dtype = weights.router.dtype
    hidden = hidden.astype(model_dtype)
    tiled = (
        token_tiled_shared
        and model_dtype == mx.bfloat16
        and hidden.shape[0] >= 8
        and config.hidden_size % 128 == 0
    )
    if fused_shared_gate:
        if tiled:
            router_shared = token_tiled_matvec(
                weights.router_shared,
                hidden,
                token_tile=8,
                simdgroups_per_threadgroup=16,
            )
        else:
            router_shared = mx.vmap(
                lambda token: mx.matmul(weights.router_shared, token)
            )(hidden)
        logits = router_shared[:, : config.num_experts]
        shared_multiplier = mx.sigmoid(
            router_shared[:, config.num_experts : config.num_experts + 1]
        )
    else:
        logits = mx.vmap(lambda token: mx.matmul(weights.router, token))(hidden)
    selected, routing = _route_batch(logits, config.top_k, model_dtype)
    direct_bf16 = direct_bf16_inputs and model_dtype == mx.bfloat16
    projection_input = hidden if direct_bf16 else hidden.astype(mx.float32)

    gate_up = nvfp4_batched_selected_paired_matvec(
        weights.experts.gate.packed,
        weights.experts.gate.scales,
        weights.experts.gate.global_scale,
        weights.experts.up.packed,
        weights.experts.up.scales,
        weights.experts.up.global_scale,
        selected,
        projection_input,
    ).astype(model_dtype)
    intermediate = _silu(gate_up[:, :, 0]) * gate_up[:, :, 1]
    routed = nvfp4_batched_selected_weighted_matvec(
        weights.experts.down.packed,
        weights.experts.down.scales,
        weights.experts.down.global_scale,
        selected,
        intermediate if direct_bf16 else intermediate.astype(mx.float32),
        routing,
    )

    shared_gate_up_kwargs = (
        {
            "token_tile": 4,
            "simdgroups_per_threadgroup": 8,
        }
        if tiled
        else {}
    )
    shared_gate_up = nvfp4_batched_paired_matvec(
        weights.shared_expert.gate.packed,
        weights.shared_expert.gate.scales,
        weights.shared_expert.gate.global_scale,
        weights.shared_expert.up.packed,
        weights.shared_expert.up.scales,
        weights.shared_expert.up.global_scale,
        projection_input,
        **shared_gate_up_kwargs,
    ).astype(model_dtype)
    shared_intermediate = _silu(shared_gate_up[:, 0]) * shared_gate_up[:, 1]
    shared_down_kwargs = (
        {
            "token_tile": 4,
            "simdgroups_per_threadgroup": 8,
        }
        if tiled
        else {}
    )
    shared = nvfp4_batched_matvec(
        weights.shared_expert.down.packed,
        weights.shared_expert.down.scales,
        weights.shared_expert.down.global_scale,
        (
            shared_intermediate
            if direct_bf16
            else shared_intermediate.astype(mx.float32)
        ),
        **shared_down_kwargs,
    ).astype(model_dtype)
    if not fused_shared_gate:
        if tiled:
            shared_gate = token_tiled_matvec(
                weights.shared_gate,
                hidden,
                token_tile=8,
                simdgroups_per_threadgroup=16,
            )
        else:
            shared_gate = mx.vmap(
                lambda token: mx.matmul(weights.shared_gate, token)
            )(hidden)
        shared_multiplier = mx.sigmoid(shared_gate)
    output = (routed + shared * shared_multiplier).astype(model_dtype)
    return MLXMoEResult(
        output=output,
        selected_experts=selected,
        routing_weights=routing,
    )


def _load_bf16(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "BF16", f"expected BF16 tensor: {name}")
    require(entry.get("shape") == list(shape), f"tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    expected_bytes = 2
    for size in shape:
        expected_bytes *= size
    require(len(payload) == expected_bytes, f"tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.bfloat16).reshape(shape)


def _load_nvfp4(source: SafetensorsFile, prefix: str) -> NVFP4Arrays:
    reference = NVFP4Weight(source, prefix)
    return NVFP4Arrays(
        packed=mx.array(memoryview(source.tensor_bytes(reference.weight_name)), dtype=mx.uint8).reshape(
            reference.rows, reference.packed_columns
        ),
        scales=mx.array(memoryview(source.tensor_bytes(reference.scale_name)), dtype=mx.uint8).reshape(
            reference.rows, reference.blocks_per_row
        ),
        global_scale=mx.array([reference.global_scale], dtype=mx.float32),
    )


def _stack(weights: list[NVFP4Arrays]) -> NVFP4Stack:
    return NVFP4Stack(
        packed=mx.stack([weight.packed for weight in weights]),
        scales=mx.stack([weight.scales for weight in weights]),
        global_scale=mx.concatenate([weight.global_scale for weight in weights]),
    )


def load_layer(source_path: Path, layer: int) -> MLXMoEWeights:
    require(0 <= layer < 40, "layer is outside the Ornith text model")
    prefix = f"model.language_model.layers.{layer}.mlp"
    with SafetensorsFile(source_path) as source:
        experts = []
        for expert in range(PRODUCTION_CONFIG.num_experts):
            expert_prefix = f"{prefix}.experts.{expert}"
            experts.append(
                ExpertArrays(
                    gate=_load_nvfp4(source, f"{expert_prefix}.gate_proj"),
                    up=_load_nvfp4(source, f"{expert_prefix}.up_proj"),
                    down=_load_nvfp4(source, f"{expert_prefix}.down_proj"),
                )
            )
        shared_prefix = f"{prefix}.shared_expert"
        router = _load_bf16(source, f"{prefix}.gate.weight", (256, 2048))
        shared_gate = _load_bf16(
            source,
            f"{prefix}.shared_expert_gate.weight",
            (1, 2048),
        )
        router_shared = mx.concatenate((router, shared_gate), axis=0)
        weights = MLXMoEWeights(
            router_shared=router_shared,
            experts=ExpertStack(
                gate=_stack([expert.gate for expert in experts]),
                up=_stack([expert.up for expert in experts]),
                down=_stack([expert.down for expert in experts]),
            ),
            shared_expert=ExpertArrays(
                gate=_load_nvfp4(source, f"{shared_prefix}.gate_proj"),
                up=_load_nvfp4(source, f"{shared_prefix}.up_proj"),
                down=_load_nvfp4(source, f"{shared_prefix}.down_proj"),
            ),
        )
        arrays = [weights.router_shared]
        for stack in (weights.experts.gate, weights.experts.up, weights.experts.down):
            arrays.extend((stack.packed, stack.scales, stack.global_scale))
        for single in (
            weights.shared_expert.gate,
            weights.shared_expert.up,
            weights.shared_expert.down,
        ):
            arrays.extend((single.packed, single.scales, single.global_scale))
        mx.eval(*arrays)
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights

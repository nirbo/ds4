#!/usr/bin/env python3
"""Exact fixed-shape GatedDeltaNet layer compilation for Ornith-35 decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_moe as moe
from ornith35_moe_reference import require


CompiledFunction = Callable[..., tuple[mx.array, ...]]


@dataclass(frozen=True)
class CompiledGDNLayer:
    """One weight-bound compiled GDN layer with dynamic hidden and state inputs."""

    index: int
    function: CompiledFunction

    def __call__(
        self,
        hidden: mx.array,
        state: gdn.MLXGDNState,
        normalized_input: mx.array | None,
    ) -> layer.LayerResult:
        if self.index == 0:
            require(normalized_input is None, "compiled first-layer input is already normalized")
            values = self.function(hidden, state.conv, state.recurrent)
        else:
            require(
                normalized_input is not None,
                "compiled GDN layer is missing its normalized input",
            )
            values = self.function(
                hidden,
                state.conv,
                state.recurrent,
                normalized_input,
            )
        require(len(values) == 6, "compiled GDN output count mismatch")
        output, conv, recurrent, selected, routing, normalized = values
        return layer.LayerResult(
            output=output,
            state=gdn.MLXGDNState(conv=conv, recurrent=recurrent),
            selected_experts=selected,
            routing_weights=routing,
            normalized_output=normalized,
        )


def compile_gdn_layer(
    index: int,
    weights: layer.GDNLayerWeights,
    next_input_norm: mx.array,
    gdn_config: gdn.GDNConfig = gdn.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
) -> CompiledGDNLayer:
    """Bind immutable weights while retaining all dynamic token and state arrays."""
    require(index >= 0, "compiled GDN layer index must be nonnegative")

    if index == 0:

        def step(
            hidden: mx.array,
            conv: mx.array,
            recurrent: mx.array,
        ) -> tuple[mx.array, ...]:
            result = layer.forward_gdn(
                hidden,
                gdn.MLXGDNState(conv=conv, recurrent=recurrent),
                weights,
                gdn_config,
                moe_config,
                normalized_input=None,
                next_input_norm=next_input_norm,
                _validated=True,
            )
            return _result_arrays(result)

    else:

        def step(
            hidden: mx.array,
            conv: mx.array,
            recurrent: mx.array,
            normalized_input: mx.array,
        ) -> tuple[mx.array, ...]:
            result = layer.forward_gdn(
                hidden,
                gdn.MLXGDNState(conv=conv, recurrent=recurrent),
                weights,
                gdn_config,
                moe_config,
                normalized_input=normalized_input,
                next_input_norm=next_input_norm,
                _validated=True,
            )
            return _result_arrays(result)

    return CompiledGDNLayer(index=index, function=mx.compile(step))


def warm_compiled_gdn_layers(
    compiled_layers: tuple[CompiledGDNLayer | None, ...],
    states: tuple[gdn.MLXGDNState | object, ...],
    dtype: mx.Dtype,
) -> None:
    """Compile all bound graphs before the first measured decode transition."""
    require(len(compiled_layers) == len(states), "compiled GDN warmup layer mismatch")
    hidden = mx.zeros((gdn.PRODUCTION_CONFIG.hidden_size,), dtype=dtype)
    normalized = mx.zeros((gdn.PRODUCTION_CONFIG.hidden_size,), dtype=dtype)
    outputs: list[mx.array] = []
    for compiled_layer, state in zip(compiled_layers, states):
        if compiled_layer is None:
            continue
        require(isinstance(state, gdn.MLXGDNState), "compiled GDN warmup state mismatch")
        result = compiled_layer(
            hidden,
            state,
            None if compiled_layer.index == 0 else normalized,
        )
        outputs.extend(_result_arrays(result))
    require(bool(outputs), "compiled GDN warmup has no layers")
    mx.eval(*outputs)
    mx.synchronize()


def _result_arrays(result: layer.LayerResult) -> tuple[mx.array, ...]:
    require(isinstance(result.state, gdn.MLXGDNState), "compiled GDN state mismatch")
    require(result.normalized_output is not None, "compiled GDN normalized output is missing")
    return (
        result.output,
        result.state.conv,
        result.state.recurrent,
        result.selected_experts,
        result.routing_weights,
        result.normalized_output,
    )

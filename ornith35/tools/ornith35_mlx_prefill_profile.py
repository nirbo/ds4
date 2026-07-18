#!/usr/bin/env python3
"""Attribute exact Ornith-35 state-prefill cost without a Metal trace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_layer as layer
import ornith35_mlx_model as model
import ornith35_mlx_moe as moe
from ornith35_moe_reference import MoEError, require
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


@dataclass(frozen=True)
class LayerTiming:
    index: int
    kind: str
    input_norm: float
    mixer: float
    post_norm: float
    moe: float
    residual: float

    @property
    def total(self) -> float:
        return self.input_norm + self.mixer + self.post_norm + self.moe + self.residual


@dataclass(frozen=True)
class ComponentProfile:
    embedding: float
    rope: float
    layers: tuple[LayerTiming, ...]
    final_kv: float
    state: model.TextModelState

    @property
    def total(self) -> float:
        return self.embedding + self.rope + sum(item.total for item in self.layers) + self.final_kv


@dataclass(frozen=True)
class TargetTiming:
    build: float
    execute: float
    state: model.TextModelState

    @property
    def total(self) -> float:
        return self.build + self.execute


def _evaluate(*arrays: mx.array) -> float:
    started = time.perf_counter()
    mx.eval(*arrays)
    mx.synchronize()
    return time.perf_counter() - started


def _layer_state_arrays(value: model.LayerState) -> tuple[mx.array, ...]:
    if isinstance(value, gdn.MLXGDNState):
        return value.conv, value.recurrent
    require(
        isinstance(value, (attention.MLXAttentionState, attention.MLXLinearAttentionState)),
        "invalid prefill profile state",
    )
    return value.keys, value.values


def _active_state_arrays(state: model.TextModelState) -> tuple[mx.array, ...]:
    arrays: list[mx.array] = []
    for value in state.layers:
        if isinstance(value, gdn.MLXGDNState):
            arrays.extend((value.conv, value.recurrent))
        elif isinstance(value, attention.MLXLinearAttentionState):
            arrays.extend(
                (
                    value.keys[:, : value.position, :],
                    value.values[:, : value.position, :],
                )
            )
        else:
            require(isinstance(value, attention.MLXAttentionState), "invalid state")
            arrays.extend((value.keys, value.values))
    return tuple(arrays)


def synthetic_prefix_state(
    initial: model.TextModelState,
    prefix: int,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
) -> model.TextModelState:
    """Build a zero-K/V timing prefix without changing recurrent state."""
    require(initial.position == 0, "synthetic prefix source must be empty")
    context.validate_range(initial.context_profile, prefix)
    states: list[model.LayerState] = []
    arrays: list[mx.array] = []
    for kind, layer_state in zip(config.layer_types, initial.layers):
        if kind == model.LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), "invalid GDN prefix state")
            states.append(layer_state)
            continue
        require(
            isinstance(layer_state, attention.MLXAttentionState)
            and attention.state_length(layer_state, config.attention) == 0,
            "synthetic attention prefix source must be empty",
        )
        shape = (
            config.attention.num_kv_heads,
            prefix,
            config.attention.head_dim,
        )
        next_state = attention.MLXAttentionState(
            keys=mx.zeros(shape, dtype=mx.bfloat16),
            values=mx.zeros(shape, dtype=mx.bfloat16),
            context_profile=initial.context_profile,
        )
        states.append(next_state)
        arrays.extend((next_state.keys, next_state.values))
    if arrays:
        mx.eval(*arrays)
        mx.synchronize()
    return model.TextModelState(
        position=prefix,
        layers=tuple(states),
        context_profile=initial.context_profile,
    )


def require_exact_state(
    expected: model.TextModelState,
    actual: model.TextModelState,
) -> None:
    require(expected.position == actual.position, "prefill profile position mismatch")
    expected_arrays = _active_state_arrays(expected)
    actual_arrays = _active_state_arrays(actual)
    require(len(expected_arrays) == len(actual_arrays) == 80, "unexpected state count")
    checks = [mx.array_equal(left, right) for left, right in zip(expected_arrays, actual_arrays)]
    mx.eval(*checks)
    mismatches = [index for index, check in enumerate(checks) if not bool(check.item())]
    require(not mismatches, f"prefill profile state mismatch at tensors {mismatches}")


def profile_components_once(
    token_ids: tuple[int, ...],
    state: model.TextModelState,
    weights: model.TextModelWeights,
) -> ComponentProfile:
    """Run state-only prefill with synchronization boundaries for attribution."""
    config = model.PRODUCTION_CONFIG
    hidden = model.embed_tokens(weights.embedding, token_ids)
    embedding_seconds = _evaluate(hidden)
    rope = attention.make_text_rope(
        state.position,
        len(token_ids),
        config.attention,
        model.matrix_dtype(weights.embedding),
    )
    rope_seconds = _evaluate(rope.cosine, rope.sine)
    normalized_input = None
    next_states: list[model.LayerState] = []
    timings: list[LayerTiming] = []
    final_index = len(weights.layers) - 1

    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(
            config.layer_types[:final_index],
            weights.layers[:final_index],
            state.layers[:final_index],
        )
    ):
        dtype = (
            layer_weights.token_mixer.in_proj_qkv.dtype
            if kind == model.LAYER_GDN
            else layer_weights.token_mixer.q_proj.dtype
        )
        hidden = hidden.astype(dtype)
        if normalized_input is None:
            mixed_input = layer.qwen_rms_norm_batch(
                hidden,
                layer_weights.norms.input_layernorm,
                config.rms_norm_eps,
            )
            input_norm_seconds = _evaluate(mixed_input)
        else:
            mixed_input = normalized_input
            input_norm_seconds = 0.0

        if kind == model.LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), "GDN weights mismatch")
            require(isinstance(layer_state, gdn.MLXGDNState), "GDN state mismatch")
            mixed, next_state = gdn.prefill_chunk(
                mixed_input,
                layer_state,
                layer_weights.token_mixer,
                config.gdn,
            )
        else:
            require(
                isinstance(layer_weights, layer.AttentionLayerWeights),
                "attention weights mismatch",
            )
            require(
                isinstance(
                    layer_state,
                    (attention.MLXAttentionState, attention.MLXLinearAttentionState),
                ),
                "attention state mismatch",
            )
            mixed, next_state = attention.prefill_chunk(
                mixed_input,
                layer_state,
                layer_weights.token_mixer,
                config.attention,
                use_steel=False,
                rope=rope,
                grouped_gqa=True,
                exact_long_prefill=True,
            )
        mixer_seconds = _evaluate(mixed, *_layer_state_arrays(next_state))
        hidden, moe_input = layer.residual_and_rms_norm_batch(
            hidden,
            mixed,
            layer_weights.norms.post_attention_layernorm,
            config.rms_norm_eps,
        )
        post_norm_seconds = _evaluate(hidden, moe_input)
        moe_result = moe.forward_batch(
            moe_input,
            layer_weights.moe,
            config.moe,
            fused_shared_gate=True,
        )
        moe_seconds = _evaluate(
            moe_result.output,
            moe_result.selected_experts,
            moe_result.routing_weights,
        )
        next_input_norm = weights.layers[index + 1].norms.input_layernorm
        hidden, normalized_input = layer.residual_and_rms_norm_batch(
            hidden,
            moe_result.output,
            next_input_norm,
            config.rms_norm_eps,
        )
        residual_seconds = _evaluate(hidden, normalized_input)
        next_states.append(next_state)
        timings.append(
            LayerTiming(
                index=index,
                kind=kind,
                input_norm=input_norm_seconds,
                mixer=mixer_seconds,
                post_norm=post_norm_seconds,
                moe=moe_seconds,
                residual=residual_seconds,
            )
        )

    require(normalized_input is not None, "final-layer normalized input is missing")
    final_weights = weights.layers[final_index]
    final_state = state.layers[final_index]
    require(isinstance(final_weights, layer.AttentionLayerWeights), "final layer mismatch")
    require(
        isinstance(final_state, (attention.MLXAttentionState, attention.MLXLinearAttentionState)),
        "final state mismatch",
    )
    next_final_state = attention.prefill_kv_chunk(
        normalized_input,
        final_state,
        final_weights.token_mixer,
        config.attention,
        rope=rope,
    )
    final_kv_seconds = _evaluate(*_layer_state_arrays(next_final_state))
    next_states.append(next_final_state)
    return ComponentProfile(
        embedding=embedding_seconds,
        rope=rope_seconds,
        layers=tuple(timings),
        final_kv=final_kv_seconds,
        state=model.TextModelState(
            position=state.position + len(token_ids),
            layers=tuple(next_states),
        ),
    )


def profile_target_once(
    token_ids: tuple[int, ...],
    state: model.TextModelState,
    weights: model.TextModelWeights,
) -> TargetTiming:
    started = time.perf_counter()
    result = model.prefill_state_chunk(
        token_ids,
        state,
        weights,
        use_steel=False,
        _validated=True,
    )
    built = time.perf_counter()
    model.evaluate_state(result)
    return TargetTiming(build=built - started, execute=time.perf_counter() - built, state=result)


def _median_layer(samples: list[ComponentProfile], position: int) -> LayerTiming:
    sample = samples[0].layers[position]
    return LayerTiming(
        index=sample.index,
        kind=sample.kind,
        input_norm=statistics.median(value.layers[position].input_norm for value in samples),
        mixer=statistics.median(value.layers[position].mixer for value in samples),
        post_norm=statistics.median(value.layers[position].post_norm for value in samples),
        moe=statistics.median(value.layers[position].moe for value in samples),
        residual=statistics.median(value.layers[position].residual for value in samples),
    )


def median_profile(samples: list[ComponentProfile]) -> ComponentProfile:
    require(bool(samples), "prefill profile has no samples")
    count = len(samples[0].layers)
    require(all(len(sample.layers) == count for sample in samples), "layer count changed")
    return ComponentProfile(
        embedding=statistics.median(sample.embedding for sample in samples),
        rope=statistics.median(sample.rope for sample in samples),
        layers=tuple(_median_layer(samples, index) for index in range(count)),
        final_kv=statistics.median(sample.final_kv for sample in samples),
        state=samples[-1].state,
    )


def _sum_component(profile: ComponentProfile, name: str) -> float:
    return sum(getattr(item, name) for item in profile.layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompt",
        default="Complete this Python function:\n\ndef binary_search(values, target):\n",
    )
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-layers", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.chunk in (8, 16, 32, 64, 128), "invalid profile chunk")
        require(args.prefix >= 0, "profile prefix must be nonnegative")
        context.validate_range(context.NATIVE_PROFILE_ID, args.prefix + args.chunk)
        require(args.repeats >= 3, "profile repeats must be at least three")
        require(args.top_layers > 0, "top-layer count must be positive")
        tokenizer = load_text_tokenizer(args.root)
        prompt_ids = tokenizer.encode(render_text_prompt(args.prompt))
        require(bool(prompt_ids), "profile prompt produced no tokens")
        token_ids = tuple(prompt_ids[index % len(prompt_ids)] for index in range(args.chunk))

        started = time.perf_counter()
        weights = model.load_text_model(args.root, map_embedding=True)
        initial = model.initial_state(weights, model.PRODUCTION_CONFIG)
        source = synthetic_prefix_state(initial, args.prefix)
        target_session = model.start_linear_decode_session(
            weights,
            source,
            args.prefix + args.chunk,
            model.PRODUCTION_CONFIG,
        )
        component_session = model.start_linear_decode_session(
            weights,
            source,
            args.prefix + args.chunk,
            model.PRODUCTION_CONFIG,
        )
        del source, initial
        gc.collect()
        mx.clear_cache()
        print(
            "prefill-profile-ready "
            f"prefix={args.prefix} chunk={args.chunk} "
            f"setup_s={time.perf_counter() - started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f}",
            flush=True,
        )

        profile_target_once(token_ids, target_session.state, weights)
        target_samples = [
            profile_target_once(token_ids, target_session.state, weights)
            for _ in range(args.repeats)
        ]
        profile_components_once(token_ids, component_session.state, weights)
        component_samples = [
            profile_components_once(token_ids, component_session.state, weights)
            for _ in range(args.repeats)
        ]
        require_exact_state(target_samples[-1].state, component_samples[-1].state)
        profile = median_profile(component_samples)
        target_median = statistics.median(sample.total for sample in target_samples)
        target_mean = statistics.fmean(sample.total for sample in target_samples)
        build_median = statistics.median(sample.build for sample in target_samples)
        execute_median = statistics.median(sample.execute for sample in target_samples)
        print(
            "prefill-profile-target "
            f"prefix={args.prefix} mean_ms={target_mean * 1000:.3f} "
            f"median_ms={target_median * 1000:.3f} "
            f"build_median_ms={build_median * 1000:.3f} "
            f"execute_median_ms={execute_median * 1000:.3f} "
            f"tokens_s={args.chunk / target_mean:.3f} samples={args.repeats} "
            "exact_tensors=80 linear_cache=true",
            flush=True,
        )
        print(
            "prefill-profile-components "
            f"synchronized_ms={profile.total * 1000:.3f} "
            f"sync_inflation={profile.total / target_median:.3f} "
            f"embedding_ms={profile.embedding * 1000:.3f} "
            f"rope_ms={profile.rope * 1000:.3f} "
            f"input_norm_ms={_sum_component(profile, 'input_norm') * 1000:.3f} "
            f"gdn_mixer_ms={sum(item.mixer for item in profile.layers if item.kind == model.LAYER_GDN) * 1000:.3f} "
            f"attention_mixer_ms={sum(item.mixer for item in profile.layers if item.kind == model.LAYER_ATTENTION) * 1000:.3f} "
            f"post_norm_ms={_sum_component(profile, 'post_norm') * 1000:.3f} "
            f"moe_ms={_sum_component(profile, 'moe') * 1000:.3f} "
            f"residual_ms={_sum_component(profile, 'residual') * 1000:.3f} "
            f"final_kv_ms={profile.final_kv * 1000:.3f}",
            flush=True,
        )
        top = sorted(profile.layers, key=lambda item: item.total, reverse=True)[: args.top_layers]
        print(
            "prefill-profile-top "
            + ",".join(
                f"{item.index}:{item.kind}:{item.total * 1000:.3f}:"
                f"mix={item.mixer * 1000:.3f}:moe={item.moe * 1000:.3f}"
                for item in top
            ),
            flush=True,
        )
        print(
            "prefill-profile-done "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
    except (MoEError, OSError, TokenizerError, ValueError) as exc:
        print(f"prefill profile failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

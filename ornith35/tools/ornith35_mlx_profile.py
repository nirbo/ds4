#!/usr/bin/env python3
"""Profile the verified Ornith-35 one-token target path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

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
    layers: tuple[LayerTiming, ...]
    final_norm: float
    lm_head: float
    logits: mx.array

    @property
    def total(self) -> float:
        return (
            self.embedding
            + sum(item.total for item in self.layers)
            + self.final_norm
            + self.lm_head
        )


@dataclass(frozen=True)
class TargetTiming:
    build: float
    execute: float

    @property
    def total(self) -> float:
        return self.build + self.execute


def _evaluate(*arrays: mx.array) -> float:
    started = time.perf_counter()
    mx.eval(*arrays)
    mx.synchronize()
    return time.perf_counter() - started


def _state_arrays(state: model.LayerState) -> tuple[mx.array, ...]:
    if isinstance(state, gdn.MLXGDNState):
        return state.conv, state.recurrent
    require(isinstance(state, attention.MLXAttentionState), "invalid profile state")
    return state.keys, state.values


def profile_components_once(
    token_id: int,
    state: model.TextModelState,
    weights: model.TextModelWeights,
    config: model.TextModelConfig = model.PRODUCTION_CONFIG,
    *,
    fused_residual_mean_square: bool,
    fused_residual_rmsnorm: bool,
    fused_gdn_convolution: bool,
    fused_gdn_recurrence: bool,
    fused_gdn_core_gate: bool,
    paired_moe_gate_up: bool,
    fused_moe_routed_down: bool,
) -> ComponentProfile:
    """Run exact composition with forced boundaries for attribution only."""
    model.validate_weights(weights, config)
    model.validate_state(state, config)
    hidden = weights.embedding[token_id]
    normalized_input = None
    embedding_seconds = _evaluate(hidden)
    timings = []

    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(config.layer_types, weights.layers, state.layers)
    ):
        model_dtype = (
            layer_weights.token_mixer.in_proj_qkv.dtype
            if kind == model.LAYER_GDN
            else layer_weights.token_mixer.q_proj.dtype
        )
        hidden = hidden.astype(model_dtype)
        if normalized_input is None:
            mixed_input = layer.qwen_rms_norm(
                hidden,
                layer_weights.norms.input_layernorm,
                config.rms_norm_eps,
            )
            input_norm_seconds = _evaluate(mixed_input)
        else:
            mixed_input = normalized_input
            input_norm_seconds = 0.0

        if kind == model.LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), "GDN profile weight mismatch")
            require(isinstance(layer_state, gdn.MLXGDNState), "GDN profile state mismatch")
            mixed, next_state = gdn.decode_step(
                mixed_input,
                layer_state,
                layer_weights.token_mixer,
                config.gdn,
                fused_convolution=fused_gdn_convolution,
                fused_recurrence=fused_gdn_recurrence,
                fused_core_gate_output=fused_gdn_core_gate,
            )
        else:
            require(
                isinstance(layer_weights, layer.AttentionLayerWeights),
                "attention profile weight mismatch",
            )
            require(
                isinstance(layer_state, attention.MLXAttentionState),
                "attention profile state mismatch",
            )
            mixed, next_state = attention.decode_step(
                mixed_input,
                layer_state,
                layer_weights.token_mixer,
                config.attention,
            )
        mixer_seconds = _evaluate(mixed, *_state_arrays(next_state))
        hidden, moe_input = layer.residual_and_rms_norm(
            hidden,
            mixed,
            layer_weights.norms.post_attention_layernorm,
            config.rms_norm_eps,
            fused_rmsnorm=fused_residual_rmsnorm,
            fused_mean_square=fused_residual_mean_square,
        )
        post_norm_seconds = _evaluate(hidden, moe_input)
        moe_result = moe.forward(
            moe_input,
            layer_weights.moe,
            config.moe,
            paired_gate_up=paired_moe_gate_up,
            fused_routed_down=fused_moe_routed_down,
        )
        moe_seconds = _evaluate(
            moe_result.output,
            moe_result.selected_experts,
            moe_result.routing_weights,
        )
        next_input_norm = (
            weights.layers[index + 1].norms.input_layernorm
            if index + 1 < len(weights.layers)
            else weights.final_norm
        )
        hidden, normalized_input = layer.residual_and_rms_norm(
            hidden,
            moe_result.output,
            next_input_norm,
            config.rms_norm_eps,
            fused_rmsnorm=fused_residual_rmsnorm,
            fused_mean_square=fused_residual_mean_square,
        )
        output_seconds = _evaluate(hidden, normalized_input)
        timings.append(
            LayerTiming(
                index=index,
                kind=kind,
                input_norm=input_norm_seconds,
                mixer=mixer_seconds,
                post_norm=post_norm_seconds,
                moe=moe_seconds,
                residual=output_seconds,
            )
        )

    require(normalized_input is not None, "profile final norm is missing")
    normalized = normalized_input
    final_norm_seconds = 0.0
    logits = mx.matmul(weights.lm_head, normalized)
    lm_head_seconds = _evaluate(logits)
    return ComponentProfile(
        embedding=embedding_seconds,
        layers=tuple(timings),
        final_norm=final_norm_seconds,
        lm_head=lm_head_seconds,
        logits=logits,
    )


def profile_target_once(
    token_id: int,
    state: model.TextModelState,
    weights: model.TextModelWeights,
    *,
    fused_residual_mean_square: bool,
    fused_residual_rmsnorm: bool,
    fused_gdn_convolution: bool,
    fused_gdn_recurrence: bool,
    fused_gdn_core_gate: bool,
    paired_moe_gate_up: bool,
    fused_moe_routed_down: bool,
) -> tuple[TargetTiming, model.TextModelResult]:
    started = time.perf_counter()
    result = model.forward_token(
        token_id,
        state,
        weights,
        fused_residual_mean_square=fused_residual_mean_square,
        fused_residual_rmsnorm=fused_residual_rmsnorm,
        fused_gdn_convolution=fused_gdn_convolution,
        fused_gdn_recurrence=fused_gdn_recurrence,
        fused_gdn_core_gate=fused_gdn_core_gate,
        paired_moe_gate_up=paired_moe_gate_up,
        fused_moe_routed_down=fused_moe_routed_down,
    )
    built = time.perf_counter()
    model.evaluate_result(result)
    finished = time.perf_counter()
    return TargetTiming(build=built - started, execute=finished - built), result


def _median_layer(samples: list[ComponentProfile], position: int) -> LayerTiming:
    identities = {
        (sample.layers[position].index, sample.layers[position].kind)
        for sample in samples
    }
    require(len(identities) == 1, "profile layer identity changed")
    index, kind = identities.pop()
    return LayerTiming(
        index=index,
        kind=kind,
        input_norm=statistics.median(sample.layers[position].input_norm for sample in samples),
        mixer=statistics.median(sample.layers[position].mixer for sample in samples),
        post_norm=statistics.median(sample.layers[position].post_norm for sample in samples),
        moe=statistics.median(sample.layers[position].moe for sample in samples),
        residual=statistics.median(sample.layers[position].residual for sample in samples),
    )


def median_profile(samples: list[ComponentProfile]) -> ComponentProfile:
    require(bool(samples), "profile has no samples")
    layer_count = len(samples[0].layers)
    require(
        all(len(sample.layers) == layer_count for sample in samples),
        "profile layer count changed",
    )
    return ComponentProfile(
        embedding=statistics.median(sample.embedding for sample in samples),
        layers=tuple(_median_layer(samples, index) for index in range(layer_count)),
        final_norm=statistics.median(sample.final_norm for sample in samples),
        lm_head=statistics.median(sample.lm_head for sample in samples),
        logits=samples[-1].logits,
    )


def _sum_component(profile: ComponentProfile, name: str) -> float:
    return sum(getattr(item, name) for item in profile.layers)


def _run_capture(
    path: Path,
    repeats: int,
    token_id: int,
    state: model.TextModelState,
    weights: model.TextModelWeights,
    fused_residual_mean_square: bool,
    fused_residual_rmsnorm: bool,
    fused_gdn_convolution: bool,
    fused_gdn_recurrence: bool,
    fused_gdn_core_gate: bool,
    paired_moe_gate_up: bool,
    fused_moe_routed_down: bool,
) -> None:
    require(path.suffix == ".gputrace", "Metal capture must use .gputrace")
    require(not path.exists(), f"refusing to replace Metal capture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(path))
    try:
        for _ in range(repeats):
            profile_target_once(
                token_id,
                state,
                weights,
                fused_residual_mean_square=fused_residual_mean_square,
                fused_residual_rmsnorm=fused_residual_rmsnorm,
                fused_gdn_convolution=fused_gdn_convolution,
                fused_gdn_recurrence=fused_gdn_recurrence,
                fused_gdn_core_gate=fused_gdn_core_gate,
                paired_moe_gate_up=paired_moe_gate_up,
                fused_moe_routed_down=fused_moe_routed_down,
            )
    finally:
        mx.metal.stop_capture()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--prompt",
        default="Complete this Python function:\n\ndef binary_search(values, target):\n",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-layers", type=int, default=10)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--capture-repeats", type=int, default=8)
    parser.add_argument(
        "--fused-residual-mean-square",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-residual-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-gdn-convolution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-gdn-recurrence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-gdn-core-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--paired-moe-gate-up",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-moe-routed-down",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.repeats > 0, "profile repeats must be positive")
        require(args.top_layers > 0, "top-layer count must be positive")
        require(args.capture_repeats > 0, "capture repeats must be positive")
        tokenizer = load_text_tokenizer(args.root)
        prompt_ids = tokenizer.encode(render_text_prompt(args.prompt))
        require(bool(prompt_ids), "profile prompt produced no tokens")

        load_started = time.perf_counter()
        weights = model.load_text_model(args.root)
        state = model.initial_state(weights, model.PRODUCTION_CONFIG)
        result = None
        for token_id in prompt_ids:
            result = model.forward_token(
                token_id,
                state,
                weights,
                fused_residual_mean_square=args.fused_residual_mean_square,
                fused_residual_rmsnorm=args.fused_residual_rmsnorm,
                fused_gdn_convolution=args.fused_gdn_convolution,
                fused_gdn_recurrence=args.fused_gdn_recurrence,
                fused_gdn_core_gate=args.fused_gdn_core_gate,
                paired_moe_gate_up=args.paired_moe_gate_up,
                fused_moe_routed_down=args.fused_moe_routed_down,
            )
            model.evaluate_result(result)
            state = result.state
        require(result is not None, "profile prefill produced no logits")
        token_id = int(mx.argmax(result.logits).item())
        print(
            "profile-ready "
            f"prompt_tokens={len(prompt_ids)} next_token={token_id} "
            f"setup_s={time.perf_counter() - load_started:.3f} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )

        profile_target_once(
            token_id,
            state,
            weights,
            fused_residual_mean_square=args.fused_residual_mean_square,
            fused_residual_rmsnorm=args.fused_residual_rmsnorm,
            fused_gdn_convolution=args.fused_gdn_convolution,
            fused_gdn_recurrence=args.fused_gdn_recurrence,
            fused_gdn_core_gate=args.fused_gdn_core_gate,
            paired_moe_gate_up=args.paired_moe_gate_up,
            fused_moe_routed_down=args.fused_moe_routed_down,
        )
        target_samples = [
            profile_target_once(
                token_id,
                state,
                weights,
                fused_residual_mean_square=args.fused_residual_mean_square,
                fused_residual_rmsnorm=args.fused_residual_rmsnorm,
                fused_gdn_convolution=args.fused_gdn_convolution,
                fused_gdn_recurrence=args.fused_gdn_recurrence,
                fused_gdn_core_gate=args.fused_gdn_core_gate,
                paired_moe_gate_up=args.paired_moe_gate_up,
                fused_moe_routed_down=args.fused_moe_routed_down,
            )[0]
            for _ in range(args.repeats)
        ]
        component_reference = profile_components_once(
            token_id,
            state,
            weights,
            fused_residual_mean_square=False,
            fused_residual_rmsnorm=False,
            fused_gdn_convolution=False,
            fused_gdn_recurrence=False,
            fused_gdn_core_gate=False,
            paired_moe_gate_up=False,
            fused_moe_routed_down=False,
        )
        target_reference = model.forward_token(
            token_id,
            state,
            weights,
            fused_residual_mean_square=args.fused_residual_mean_square,
            fused_residual_rmsnorm=args.fused_residual_rmsnorm,
            fused_gdn_convolution=args.fused_gdn_convolution,
            fused_gdn_recurrence=args.fused_gdn_recurrence,
            fused_gdn_core_gate=args.fused_gdn_core_gate,
            paired_moe_gate_up=args.paired_moe_gate_up,
            fused_moe_routed_down=args.fused_moe_routed_down,
        )
        model.evaluate_result(target_reference)
        reference32 = component_reference.logits.astype(mx.float32)
        target32 = target_reference.logits.astype(mx.float32)
        difference = target32 - reference32
        max_abs = float(mx.max(mx.abs(difference)).item())
        relative_l2 = float(
            (mx.sqrt(mx.sum(difference * difference)) / mx.sqrt(mx.sum(reference32 * reference32))).item()
        )
        top_match = int(mx.argmax(reference32).item()) == int(mx.argmax(target32).item())
        profile = median_profile(
            [
                profile_components_once(
                    token_id,
                    state,
                    weights,
                    fused_residual_mean_square=args.fused_residual_mean_square,
                    fused_residual_rmsnorm=args.fused_residual_rmsnorm,
                    fused_gdn_convolution=args.fused_gdn_convolution,
                    fused_gdn_recurrence=args.fused_gdn_recurrence,
                    fused_gdn_core_gate=args.fused_gdn_core_gate,
                    paired_moe_gate_up=args.paired_moe_gate_up,
                    fused_moe_routed_down=args.fused_moe_routed_down,
                )
                for _ in range(args.repeats)
            ]
        )
        target_median = statistics.median(sample.total for sample in target_samples)
        target_mean = statistics.fmean(sample.total for sample in target_samples)
        build_median = statistics.median(sample.build for sample in target_samples)
        execute_median = statistics.median(sample.execute for sample in target_samples)
        print(
            "profile-target "
            f"mean_ms={target_mean * 1000:.3f} median_ms={target_median * 1000:.3f} "
            f"build_median_ms={build_median * 1000:.3f} "
            f"execute_median_ms={execute_median * 1000:.3f} "
            f"tokens_s={1.0 / target_mean:.3f} samples={args.repeats} "
            f"fused_residual_mean_square={str(args.fused_residual_mean_square).lower()} "
            f"fused_residual_rmsnorm={str(args.fused_residual_rmsnorm).lower()} "
            f"fused_gdn_convolution={str(args.fused_gdn_convolution).lower()} "
            f"fused_gdn_recurrence={str(args.fused_gdn_recurrence).lower()} "
            f"fused_gdn_core_gate={str(args.fused_gdn_core_gate).lower()} "
            f"paired_moe_gate_up={str(args.paired_moe_gate_up).lower()} "
            f"fused_moe_routed_down={str(args.fused_moe_routed_down).lower()}",
            flush=True,
        )
        print(
            "profile-components "
            f"synchronized_ms={profile.total * 1000:.3f} "
            f"sync_inflation={profile.total / target_median:.3f} "
            f"embedding_ms={profile.embedding * 1000:.3f} "
            f"input_norm_ms={_sum_component(profile, 'input_norm') * 1000:.3f} "
            f"gdn_mixer_ms={sum(item.mixer for item in profile.layers if item.kind == model.LAYER_GDN) * 1000:.3f} "
            "attention_mixer_ms="
            f"{sum(item.mixer for item in profile.layers if item.kind == model.LAYER_ATTENTION) * 1000:.3f} "
            f"post_norm_ms={_sum_component(profile, 'post_norm') * 1000:.3f} "
            f"moe_ms={_sum_component(profile, 'moe') * 1000:.3f} "
            f"residual_ms={_sum_component(profile, 'residual') * 1000:.3f} "
            f"final_norm_ms={profile.final_norm * 1000:.3f} "
            f"lm_head_ms={profile.lm_head * 1000:.3f} "
            f"logits_max_abs={max_abs:.9g} logits_relative_l2={relative_l2:.9g} "
            f"top_match={str(top_match).lower()}",
            flush=True,
        )
        top = sorted(profile.layers, key=lambda item: item.total, reverse=True)[: args.top_layers]
        print(
            "profile-top "
            + ",".join(
                f"{item.index}:{item.kind}:{item.total * 1000:.3f}:"
                f"mix={item.mixer * 1000:.3f}:moe={item.moe * 1000:.3f}"
                for item in top
            ),
            flush=True,
        )
        if args.capture is not None:
            _run_capture(
                args.capture,
                args.capture_repeats,
                token_id,
                state,
                weights,
                args.fused_residual_mean_square,
                args.fused_residual_rmsnorm,
                args.fused_gdn_convolution,
                args.fused_gdn_recurrence,
                args.fused_gdn_core_gate,
                args.paired_moe_gate_up,
                args.fused_moe_routed_down,
            )
            print(
                f"profile-capture path={args.capture} repeats={args.capture_repeats}",
                flush=True,
            )
        print(
            "profile-done "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
    except (MoEError, TokenizerError, OSError, RuntimeError, ValueError) as exc:
        print(f"ornith35 profile failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

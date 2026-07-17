#!/usr/bin/env python3
"""Text-only resident one-token model boundary for Ornith-35."""

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
import ornith35_mlx_moe as moe
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT, SafetensorsFile, require_verified_source


LAYER_GDN = "gdn"
LAYER_ATTENTION = "attention"


@dataclass(frozen=True)
class TextModelConfig:
    vocab_size: int
    hidden_size: int
    layer_types: tuple[str, ...]
    gdn: gdn.GDNConfig
    attention: attention.AttentionConfig
    moe: moe.MoEConfig
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        require(self.vocab_size > 0 and self.hidden_size > 0, "model dimensions must be positive")
        require(len(self.layer_types) > 0, "model must contain decoder layers")
        require(
            all(kind in (LAYER_GDN, LAYER_ATTENTION) for kind in self.layer_types),
            "invalid decoder-layer type",
        )
        require(
            self.gdn.hidden_size == self.hidden_size
            and self.attention.hidden_size == self.hidden_size
            and self.moe.hidden_size == self.hidden_size,
            "model component hidden-size mismatch",
        )
        require(self.rms_norm_eps > 0.0, "model RMS epsilon must be positive")


PRODUCTION_CONFIG = TextModelConfig(
    vocab_size=248_320,
    hidden_size=2048,
    layer_types=tuple(
        LAYER_ATTENTION if index % 4 == 3 else LAYER_GDN
        for index in range(40)
    ),
    gdn=gdn.PRODUCTION_CONFIG,
    attention=attention.PRODUCTION_CONFIG,
    moe=moe.PRODUCTION_CONFIG,
    rms_norm_eps=1e-6,
)


LayerWeights = layer.GDNLayerWeights | layer.AttentionLayerWeights
LayerState = gdn.MLXGDNState | attention.MLXAttentionState


@dataclass(frozen=True)
class TextModelWeights:
    embedding: mx.array
    layers: tuple[LayerWeights, ...]
    final_norm: mx.array
    lm_head: mx.array


@dataclass(frozen=True)
class TextModelState:
    position: int
    layers: tuple[LayerState, ...]


@dataclass(frozen=True)
class TextModelResult:
    logits: mx.array
    hidden: mx.array
    state: TextModelState
    selected_experts: tuple[mx.array, ...]
    routing_weights: tuple[mx.array, ...]


def validate_weights(weights: TextModelWeights, config: TextModelConfig) -> None:
    require(
        weights.embedding.shape == (config.vocab_size, config.hidden_size),
        "embedding shape mismatch",
    )
    require(weights.lm_head.shape == (config.vocab_size, config.hidden_size), "LM-head shape mismatch")
    require(weights.final_norm.shape == (config.hidden_size,), "final RMSNorm shape mismatch")
    require(len(weights.layers) == len(config.layer_types), "decoder-layer count mismatch")
    dtype = weights.embedding.dtype
    require(dtype in (mx.bfloat16, mx.float32), "invalid text-model dtype")
    require(weights.lm_head.dtype == dtype, "LM-head dtype mismatch")
    require(weights.final_norm.dtype == dtype, "final RMSNorm dtype mismatch")
    for index, (kind, layer_weights) in enumerate(zip(config.layer_types, weights.layers)):
        expected = layer.GDNLayerWeights if kind == LAYER_GDN else layer.AttentionLayerWeights
        require(isinstance(layer_weights, expected), f"decoder-layer type mismatch at {index}")
        token_dtype = (
            layer_weights.token_mixer.in_proj_qkv.dtype
            if kind == LAYER_GDN
            else layer_weights.token_mixer.q_proj.dtype
        )
        require(token_dtype == dtype, f"decoder-layer dtype mismatch at {index}")


def initial_state(weights: TextModelWeights, config: TextModelConfig) -> TextModelState:
    validate_weights(weights, config)
    dtype = weights.embedding.dtype
    states = tuple(
        gdn.zeros_state(config.gdn, conv_dtype=dtype)
        if kind == LAYER_GDN
        else attention.zeros_state(config.attention, dtype=dtype)
        for kind in config.layer_types
    )
    return TextModelState(position=0, layers=states)


def validate_state(state: TextModelState, config: TextModelConfig) -> None:
    require(state.position >= 0, "model position must be nonnegative")
    require(len(state.layers) == len(config.layer_types), "model-state layer count mismatch")
    for index, (kind, layer_state) in enumerate(zip(config.layer_types, state.layers)):
        if kind == LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            gdn.validate_state(layer_state, config.gdn)
        else:
            require(
                isinstance(layer_state, attention.MLXAttentionState),
                f"attention state mismatch at {index}",
            )
            length = attention.state_length(layer_state, config.attention)
            require(length == state.position, f"attention position mismatch at {index}")


def forward_token(
    token_id: int,
    state: TextModelState,
    weights: TextModelWeights,
    config: TextModelConfig = PRODUCTION_CONFIG,
) -> TextModelResult:
    """Evaluate one token and return full-vocabulary target logits lazily."""
    require(isinstance(token_id, int) and 0 <= token_id < config.vocab_size, "token ID is out of range")
    validate_weights(weights, config)
    validate_state(state, config)
    hidden = weights.embedding[token_id]
    next_states = []
    selected_experts = []
    routing_weights = []
    for index, (kind, layer_weights, layer_state) in enumerate(
        zip(config.layer_types, weights.layers, state.layers)
    ):
        if kind == LAYER_GDN:
            require(isinstance(layer_weights, layer.GDNLayerWeights), f"GDN weights mismatch at {index}")
            require(isinstance(layer_state, gdn.MLXGDNState), f"GDN state mismatch at {index}")
            result = layer.forward_gdn(
                hidden,
                layer_state,
                layer_weights,
                config.gdn,
                config.moe,
            )
        else:
            require(
                isinstance(layer_weights, layer.AttentionLayerWeights),
                f"attention weights mismatch at {index}",
            )
            require(
                isinstance(layer_state, attention.MLXAttentionState),
                f"attention state mismatch at {index}",
            )
            result = layer.forward_attention(
                hidden,
                layer_state,
                layer_weights,
                config.attention,
                config.moe,
            )
        hidden = result.output
        next_states.append(result.state)
        selected_experts.append(result.selected_experts)
        routing_weights.append(result.routing_weights)

    hidden = layer.qwen_rms_norm(hidden, weights.final_norm, config.rms_norm_eps)
    logits = mx.matmul(weights.lm_head, hidden)
    return TextModelResult(
        logits=logits,
        hidden=hidden,
        state=TextModelState(position=state.position + 1, layers=tuple(next_states)),
        selected_experts=tuple(selected_experts),
        routing_weights=tuple(routing_weights),
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


def load_text_model(root: Path) -> TextModelWeights:
    """Load only explicitly cataloged text tensors from a verified source."""
    source_path = require_verified_source(root)
    with SafetensorsFile(source_path) as source:
        embedding = _load_bf16(
            source,
            "model.language_model.embed_tokens.weight",
            (248_320, 2048),
        )
        final_norm = _load_bf16(source, "model.language_model.norm.weight", (2048,))
        lm_head = _load_bf16(source, "lm_head.weight", (248_320, 2048))
        mx.eval(embedding, final_norm, lm_head)
    layers = tuple(layer.load_layer(source_path, index) for index in range(40))
    weights = TextModelWeights(
        embedding=embedding,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights


def parse_token_ids(
    text: str,
    vocab_size: int = PRODUCTION_CONFIG.vocab_size,
) -> tuple[int, ...]:
    try:
        values = tuple(
            int(value.strip()) for value in text.split(",") if value.strip()
        )
    except ValueError as exc:
        raise MoEError("tokens must be comma-separated integers") from exc
    require(values, "at least one token is required")
    require(
        all(0 <= value < vocab_size for value in values),
        "token ID is out of range",
    )
    return values


def _state_arrays(state: TextModelState) -> list[mx.array]:
    arrays: list[mx.array] = []
    for layer_state in state.layers:
        if isinstance(layer_state, gdn.MLXGDNState):
            arrays.extend((layer_state.conv, layer_state.recurrent))
        else:
            require(
                isinstance(layer_state, attention.MLXAttentionState),
                "invalid layer state",
            )
            arrays.extend((layer_state.keys, layer_state.values))
    return arrays


def evaluate_result(result: TextModelResult) -> None:
    """Materialize a target result without exposing cache internals to frontends."""
    mx.eval(
        result.logits,
        result.hidden,
        *result.selected_experts,
        *result.routing_weights,
        *_state_arrays(result.state),
    )
    mx.synchronize()


def run_source_smoke(root: Path, token_ids: tuple[int, ...]) -> None:
    """Run a bounded resident source smoke; this is not a quality acceptance."""
    started = time.perf_counter()
    weights = load_text_model(root)
    loaded = time.perf_counter()
    print(
        "real-model-load "
        f"layers={len(weights.layers)} elapsed_s={loaded - started:.3f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    state = initial_state(weights, PRODUCTION_CONFIG)
    timings = []
    for step, token_id in enumerate(token_ids, start=1):
        before = time.perf_counter()
        result = forward_token(token_id, state, weights, PRODUCTION_CONFIG)
        evaluate_result(result)
        elapsed = time.perf_counter() - before
        timings.append(elapsed)
        require(
            bool(mx.all(mx.isfinite(result.logits)).item()),
            "non-finite target logits",
        )
        require(
            all(
                abs(float(mx.sum(route.astype(mx.float32)).item()) - 1.0) < 2e-3
                for route in result.routing_weights
            ),
            "routing weights are not normalized",
        )
        top_id = int(mx.argmax(result.logits).item())
        top_logit = float(result.logits[top_id].item())
        hidden_l2 = float(
            mx.sqrt(mx.sum(result.hidden.astype(mx.float32) ** 2)).item()
        )
        print(
            "real-model-token "
            f"step={step} input={token_id} output={top_id} "
            f"top_logit={top_logit:.9g} hidden_l2={hidden_l2:.9g} "
            f"elapsed_s={elapsed:.3f}",
            flush=True,
        )
        state = result.state
    if len(timings) > 2:
        measured = timings[2:]
        mean = sum(measured) / len(measured)
        print(
            "real-model-post-warmup "
            f"tokens={len(measured)} mean_ms={mean * 1000:.3f} "
            f"median_ms={statistics.median(measured) * 1000:.3f} "
            f"tokens_s={1.0 / mean:.3f}",
            flush=True,
        )
    print(
        "real-model-done "
        f"position={state.position} active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--tokens",
        default="248044,9707",
        help="comma-separated token IDs for a bounded sequential smoke",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_source_smoke(args.root, parse_token_ids(args.tokens))
    except (MoEError, OSError, ValueError) as exc:
        print(f"ornith35 real model smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

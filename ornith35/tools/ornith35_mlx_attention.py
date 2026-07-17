#!/usr/bin/env python3
"""One-token MLX full-attention composition for text-only Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

import ornith35_mlx_linear_cache as linear_cache
from ornith35_attention_reference import AttentionConfig, require
from ornith35_nvfp4 import SafetensorsFile


PRODUCTION_CONFIG = AttentionConfig(
    hidden_size=2048,
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=256,
    rotary_dim=64,
    rope_theta=10_000_000.0,
    rms_norm_eps=1e-6,
)


GROUPED_GQA_PREFILL_MIN_PREFIX = 1280


QK_NORM_ROPE_KERNEL_SOURCE = r"""
uint head = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
threadgroup float inverse_mean[1];
threadgroup bfloat16_t normalized[256];
bool is_query = head < 16u;
uint local_head = is_query ? head : head - 16u;
uint input_base = is_query ? local_head * 512u : local_head * 256u;
float total = 0.0f;
for (uint block = 0u; block < 2u; ++block) {
    uint local_base = lid * 4u + block * 128u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint index = local_base + offset;
        float value = float(
            is_query ? query_gate[input_base + index] : key[input_base + index]
        );
        volatile float square = value * value;
        total = square + total;
    }
}
total = simd_sum(total);
if (lane == 0u) {
    volatile float mean = total / 256.0f;
    volatile float adjusted = mean + 1.0e-6f;
    inverse_mean[0] = metal::precise::rsqrt(adjusted);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint block = 0u; block < 2u; ++block) {
    uint local_base = lid * 4u + block * 128u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint index = local_base + offset;
        float value = float(
            is_query ? query_gate[input_base + index] : key[input_base + index]
        );
        float weight = float(is_query ? q_norm[index] : k_norm[index]);
        volatile float scaled = value * inverse_mean[0];
        volatile float centered = 1.0f + weight;
        volatile float weighted = scaled * centered;
        normalized[index] = bfloat16_t(weighted);
        if (is_query) {
            output_gate[local_head * 256u + index] =
                query_gate[input_base + 256u + index];
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint block = 0u; block < 2u; ++block) {
    uint local_base = lid * 4u + block * 128u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint index = local_base + offset;
        bfloat16_t output_value = normalized[index];
        if (index < 64u) {
            uint rotated_index = index < 32u ? index + 32u : index - 32u;
            bfloat16_t rotated = index < 32u
                ? -normalized[rotated_index]
                : normalized[rotated_index];
            bfloat16_t first = normalized[index] * cosine[index];
            bfloat16_t second = rotated * sine[index];
            output_value = first + second;
        }
        if (is_query) {
            output_query[local_head * 256u + index] = output_value;
        } else {
            output_key[local_head * 256u + index] = output_value;
        }
    }
}
"""


_qk_norm_rope_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_qk_norm_rope_bf16",
    input_names=["query_gate", "key", "q_norm", "k_norm", "cosine", "sine"],
    output_names=["output_query", "output_gate", "output_key"],
    source=QK_NORM_ROPE_KERNEL_SOURCE,
)


@dataclass(frozen=True)
class MLXAttentionWeights:
    q_proj: mx.array
    k_proj: mx.array
    v_proj: mx.array
    o_proj: mx.array
    q_norm: mx.array
    k_norm: mx.array


@dataclass(frozen=True)
class MLXAttentionState:
    keys: mx.array
    values: mx.array


@dataclass(frozen=True)
class MLXLinearAttentionState:
    """Fixed-capacity K/V storage owned by one advancing decode session."""

    keys: mx.array
    values: mx.array
    position: int
    capacity: int


@dataclass(frozen=True)
class MLXTextRoPE:
    position: int
    tokens: int
    cosine: mx.array
    sine: mx.array


def validate_weights(weights: MLXAttentionWeights, config: AttentionConfig) -> None:
    expected = {
        "q_proj": (config.query_dim * 2, config.hidden_size),
        "k_proj": (config.kv_dim, config.hidden_size),
        "v_proj": (config.kv_dim, config.hidden_size),
        "o_proj": (config.hidden_size, config.query_dim),
        "q_norm": (config.head_dim,),
        "k_norm": (config.head_dim,),
    }
    for name, shape in expected.items():
        require(getattr(weights, name).shape == shape, f"{name} shape mismatch")
    model_dtype = weights.q_proj.dtype
    require(
        model_dtype in (mx.bfloat16, mx.float32),
        "attention model dtype must be BF16 or reference FP32",
    )
    require(
        all(array.dtype == model_dtype for array in weights.__dict__.values()),
        "attention weight dtype mismatch",
    )


def state_length(
    state: MLXAttentionState | MLXLinearAttentionState,
    config: AttentionConfig,
) -> int:
    require(state.keys.ndim == 3, "key state rank mismatch")
    require(state.values.ndim == 3, "value state rank mismatch")
    require(state.keys.shape[0] == config.num_kv_heads, "key state head mismatch")
    require(state.values.shape[0] == config.num_kv_heads, "value state head mismatch")
    require(state.keys.shape[2] == config.head_dim, "key state width mismatch")
    require(state.values.shape[2] == config.head_dim, "value state width mismatch")
    require(state.keys.shape[1] == state.values.shape[1], "KV state length mismatch")
    require(state.keys.dtype == state.values.dtype, "KV state dtype mismatch")
    if isinstance(state, MLXLinearAttentionState):
        require(state.capacity == state.keys.shape[1], "linear KV capacity mismatch")
        require(
            0 <= state.position <= state.capacity,
            "linear KV position is outside capacity",
        )
        return state.position
    return state.keys.shape[1]


def zeros_state(
    config: AttentionConfig,
    dtype: mx.Dtype = mx.bfloat16,
) -> MLXAttentionState:
    shape = (config.num_kv_heads, 0, config.head_dim)
    return MLXAttentionState(keys=mx.zeros(shape, dtype=dtype), values=mx.zeros(shape, dtype=dtype))


def linearize_state(
    state: MLXAttentionState,
    capacity: int,
    config: AttentionConfig = PRODUCTION_CONFIG,
) -> MLXLinearAttentionState:
    """Copy one immutable prefix into append-only, fixed-capacity buffers."""
    require(isinstance(state, MLXAttentionState), "linear source state must be immutable")
    position = state_length(state, config)
    require(capacity >= position, "linear KV capacity is shorter than the prefix")
    require(state.keys.dtype == mx.bfloat16, "linear K/V cache requires BF16 state")
    shape = (config.num_kv_heads, capacity, config.head_dim)
    keys = mx.zeros(shape, dtype=mx.bfloat16)
    values = mx.zeros(shape, dtype=mx.bfloat16)
    if position:
        keys, values = linear_cache.append_kv_bf16(
            keys,
            values,
            state.keys,
            state.values,
            0,
        )
    return MLXLinearAttentionState(
        keys=keys,
        values=values,
        position=position,
        capacity=capacity,
    )


def _rms_norm(value: mx.array, weight: mx.array, eps: float, dtype: mx.Dtype) -> mx.array:
    value32 = value.astype(mx.float32)
    normalized = value32 * mx.rsqrt(mx.mean(value32 * value32, axis=-1, keepdims=True) + eps)
    return (normalized * (1.0 + weight.astype(mx.float32))).astype(dtype)


def _linear_batch(weight: mx.array, vectors: mx.array) -> mx.array:
    return mx.vmap(lambda vector: mx.matmul(weight, vector))(vectors)


def make_text_rope(
    position: int,
    tokens: int,
    config: AttentionConfig,
    dtype: mx.Dtype,
) -> MLXTextRoPE:
    """Build one exact partial-RoPE table for reuse across attention layers."""
    require(position >= 0 and tokens > 0, "invalid text RoPE range")
    indices = mx.arange(0, config.rotary_dim, 2, dtype=mx.float32)
    inverse_frequencies = mx.power(
        config.rope_theta,
        -indices / config.rotary_dim,
    )
    if tokens == 1:
        frequencies = inverse_frequencies * position
        angles = mx.concatenate([frequencies, frequencies])
    else:
        positions = mx.arange(position, position + tokens, dtype=mx.float32)
        frequencies = positions[:, None] * inverse_frequencies[None, :]
        angles = mx.concatenate([frequencies, frequencies], axis=-1)
    return MLXTextRoPE(
        position=position,
        tokens=tokens,
        cosine=mx.cos(angles).astype(dtype),
        sine=mx.sin(angles).astype(dtype),
    )


def _validate_rope(
    rope: MLXTextRoPE,
    position: int,
    tokens: int,
    config: AttentionConfig,
    dtype: mx.Dtype,
) -> None:
    require(
        rope.position == position and rope.tokens == tokens,
        "text RoPE range mismatch",
    )
    shape = (config.rotary_dim,) if tokens == 1 else (tokens, config.rotary_dim)
    require(
        rope.cosine.dtype == dtype and rope.cosine.shape == shape,
        "text RoPE cosine mismatch",
    )
    require(
        rope.sine.dtype == dtype and rope.sine.shape == shape,
        "text RoPE sine mismatch",
    )


def _apply_text_rope(
    value: mx.array,
    position: int,
    config: AttentionConfig,
    dtype: mx.Dtype,
    rope: MLXTextRoPE | None = None,
) -> mx.array:
    half = config.rotary_dim // 2
    if rope is None:
        rope = make_text_rope(position, 1, config, dtype)
    _validate_rope(rope, position, 1, config, dtype)
    rotary = value[:, : config.rotary_dim]
    rotated = mx.concatenate([-rotary[:, half:], rotary[:, :half]], axis=-1)
    embedded = rotary * rope.cosine + rotated * rope.sine
    return mx.concatenate([embedded, value[:, config.rotary_dim :]], axis=-1)


def _apply_text_rope_chunk(
    value: mx.array,
    start_position: int,
    config: AttentionConfig,
    dtype: mx.Dtype,
    rope: MLXTextRoPE | None = None,
) -> mx.array:
    require(value.ndim == 3 and value.shape[0] > 0, "RoPE chunk shape mismatch")
    half = config.rotary_dim // 2
    if rope is None:
        rope = make_text_rope(start_position, value.shape[0], config, dtype)
    _validate_rope(rope, start_position, value.shape[0], config, dtype)
    cosine = rope.cosine[:, None, :]
    sine = rope.sine[:, None, :]
    rotary = value[:, :, : config.rotary_dim]
    rotated = mx.concatenate([-rotary[:, :, half:], rotary[:, :, :half]], axis=-1)
    embedded = rotary * cosine + rotated * sine
    return mx.concatenate([embedded, value[:, :, config.rotary_dim :]], axis=-1)


def fused_qk_norm_rope_step(
    query_gate: mx.array,
    key: mx.array,
    q_norm: mx.array,
    k_norm: mx.array,
    rope: MLXTextRoPE,
) -> tuple[mx.array, mx.array, mx.array]:
    """Apply exact production Q/K norms, partial RoPE, and gate splitting."""
    expected = {
        "query_gate": (query_gate, (8192,)),
        "key": (key, (512,)),
        "q_norm": (q_norm, (256,)),
        "k_norm": (k_norm, (256,)),
    }
    for name, (array, shape) in expected.items():
        require(
            array.dtype == mx.bfloat16 and array.shape == shape,
            f"fused attention {name} mismatch",
        )
    _validate_rope(rope, rope.position, 1, PRODUCTION_CONFIG, mx.bfloat16)
    query, gate, output_key = _qk_norm_rope_kernel(
        inputs=[query_gate, key, q_norm, k_norm, rope.cosine, rope.sine],
        grid=(18 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16, 256), (16, 256), (2, 256)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )
    return query, gate, output_key


def decode_step(
    hidden: mx.array,
    state: MLXAttentionState | MLXLinearAttentionState,
    weights: MLXAttentionWeights,
    config: AttentionConfig = PRODUCTION_CONFIG,
    *,
    rope: MLXTextRoPE | None = None,
    fused_qk_norm_rope: bool = True,
    grouped_gqa: bool = True,
    _validated: bool = False,
) -> tuple[mx.array, MLXAttentionState | MLXLinearAttentionState]:
    """Append one causal text token under the state's ownership contract."""
    require(hidden.ndim == 1 and hidden.shape == (config.hidden_size,), "hidden-state shape mismatch")
    position = state_length(state, config)
    if not _validated:
        validate_weights(weights, config)
    model_dtype = weights.q_proj.dtype
    require(state.keys.dtype == model_dtype, "KV state dtype mismatch")
    hidden = hidden.astype(model_dtype)

    query_gate = mx.matmul(weights.q_proj, hidden)
    key = mx.matmul(weights.k_proj, hidden)
    value = mx.matmul(weights.v_proj, hidden).reshape(config.num_kv_heads, config.head_dim)
    production = config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16
    if fused_qk_norm_rope and production:
        if rope is None:
            rope = make_text_rope(position, 1, config, model_dtype)
        _validate_rope(rope, position, 1, config, model_dtype)
        query, gate, key = fused_qk_norm_rope_step(
            query_gate,
            key,
            weights.q_norm,
            weights.k_norm,
            rope,
        )
    else:
        query_gate = query_gate.reshape(config.num_q_heads, config.head_dim * 2)
        query = query_gate[:, : config.head_dim]
        gate = query_gate[:, config.head_dim :]
        key = key.reshape(config.num_kv_heads, config.head_dim)
        query = _rms_norm(query, weights.q_norm, config.rms_norm_eps, model_dtype)
        key = _rms_norm(key, weights.k_norm, config.rms_norm_eps, model_dtype)
        query = _apply_text_rope(query, position, config, model_dtype, rope)
        key = _apply_text_rope(key, position, config, model_dtype, rope)

    if isinstance(state, MLXLinearAttentionState):
        require(position < state.capacity, "linear KV cache capacity exhausted")
        key_buffer, value_buffer = linear_cache.append_kv_bf16(
            state.keys,
            state.values,
            key[:, None, :],
            value[:, None, :],
            position,
        )
        next_keys = key_buffer[:, : position + 1, :]
        next_values = value_buffer[:, : position + 1, :]
        next_state = MLXLinearAttentionState(
            keys=key_buffer,
            values=value_buffer,
            position=position + 1,
            capacity=state.capacity,
        )
    else:
        next_keys = mx.concatenate([state.keys, key[:, None, :]], axis=1)
        next_values = mx.concatenate([state.values, value[:, None, :]], axis=1)
        next_state = MLXAttentionState(keys=next_keys, values=next_values)
    groups = config.num_q_heads // config.num_kv_heads
    if grouped_gqa:
        grouped_query = query.reshape(
            config.num_kv_heads,
            groups,
            1,
            config.head_dim,
        )
        grouped_keys = mx.swapaxes(next_keys, 1, 2)[:, None, :, :]
        scores = mx.matmul(grouped_query, grouped_keys).reshape(
            config.num_q_heads,
            position + 1,
        )
    else:
        repeated_keys = mx.repeat(next_keys, groups, axis=0)
        scores = mx.matmul(
            query[:, None, :],
            mx.swapaxes(repeated_keys, 1, 2),
        ).reshape(config.num_q_heads, position + 1)
    scores = scores * (config.head_dim**-0.5)
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(model_dtype)
    if grouped_gqa:
        attended = mx.matmul(
            probabilities.reshape(
                config.num_kv_heads,
                groups,
                1,
                position + 1,
            ),
            next_values[:, None, :, :],
        ).reshape(config.num_q_heads, config.head_dim)
    else:
        repeated_values = mx.repeat(next_values, groups, axis=0)
        attended = mx.matmul(
            probabilities[:, None, :],
            repeated_values,
        ).reshape(config.num_q_heads, config.head_dim)
    attended = attended * mx.sigmoid(gate)
    output = mx.matmul(weights.o_proj, attended.reshape(config.query_dim))
    return output, next_state


def prefill_chunk(
    hidden: mx.array,
    state: MLXAttentionState | MLXLinearAttentionState,
    weights: MLXAttentionWeights,
    config: AttentionConfig = PRODUCTION_CONFIG,
    *,
    use_steel: bool = True,
    rope: MLXTextRoPE | None = None,
    grouped_gqa: bool = True,
) -> tuple[mx.array, MLXAttentionState | MLXLinearAttentionState]:
    """Append a causal token chunk and return outputs plus the complete K/V state."""
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0 and hidden.shape[1] == config.hidden_size,
        "attention prefill hidden-state shape mismatch",
    )
    position = state_length(state, config)
    validate_weights(weights, config)
    model_dtype = weights.q_proj.dtype
    require(state.keys.dtype == model_dtype, "KV state dtype mismatch")
    hidden = hidden.astype(model_dtype)
    tokens = hidden.shape[0]
    grouped_gqa = grouped_gqa and (
        config != PRODUCTION_CONFIG
        or position >= GROUPED_GQA_PREFILL_MIN_PREFIX
    )
    query_gate = _linear_batch(weights.q_proj, hidden).reshape(
        tokens,
        config.num_q_heads,
        config.head_dim * 2,
    )
    query = query_gate[:, :, : config.head_dim]
    gate = query_gate[:, :, config.head_dim :]
    key = _linear_batch(weights.k_proj, hidden).reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    value = _linear_batch(weights.v_proj, hidden).reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    query = _rms_norm(query, weights.q_norm, config.rms_norm_eps, model_dtype)
    key = _rms_norm(key, weights.k_norm, config.rms_norm_eps, model_dtype)
    query = _apply_text_rope_chunk(query, position, config, model_dtype, rope)
    key = _apply_text_rope_chunk(key, position, config, model_dtype, rope)

    key_update = mx.transpose(key, (1, 0, 2))
    value_update = mx.transpose(value, (1, 0, 2))
    if isinstance(state, MLXLinearAttentionState):
        require(
            position + tokens <= state.capacity,
            "linear KV cache capacity exhausted",
        )
        key_buffer, value_buffer = linear_cache.append_kv_transposed_bf16(
            state.keys,
            state.values,
            key,
            value,
            position,
        )
        next_keys = key_buffer[:, : position + tokens, :]
        next_values = value_buffer[:, : position + tokens, :]
        next_state = MLXLinearAttentionState(
            keys=key_buffer,
            values=value_buffer,
            position=position + tokens,
            capacity=state.capacity,
        )
    else:
        next_keys = mx.concatenate([state.keys, key_update], axis=1)
        next_values = mx.concatenate([state.values, value_update], axis=1)
        next_state = MLXAttentionState(keys=next_keys, values=next_values)
    if not use_steel or config != PRODUCTION_CONFIG:
        groups = config.num_q_heads // config.num_kv_heads
        if not grouped_gqa:
            repeated_keys = mx.repeat(next_keys, groups, axis=0)
            repeated_values = mx.repeat(next_values, groups, axis=0)
        attended_tokens = []
        for offset, (token_query, token_gate) in enumerate(zip(query, gate)):
            length = position + offset + 1
            if grouped_gqa:
                scores = mx.matmul(
                    token_query.reshape(
                        config.num_kv_heads,
                        groups,
                        1,
                        config.head_dim,
                    ),
                    mx.swapaxes(next_keys[:, :length, :], 1, 2)[:, None, :, :],
                ).reshape(config.num_q_heads, length)
            else:
                token_keys = repeated_keys[:, :length, :]
                scores = mx.matmul(
                    token_query[:, None, :],
                    mx.swapaxes(token_keys, 1, 2),
                ).reshape(config.num_q_heads, length)
            scores = scores * (config.head_dim**-0.5)
            probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
                model_dtype
            )
            if grouped_gqa:
                attended = mx.matmul(
                    probabilities.reshape(
                        config.num_kv_heads,
                        groups,
                        1,
                        length,
                    ),
                    next_values[:, None, :length, :],
                ).reshape(config.num_q_heads, config.head_dim)
            else:
                token_values = repeated_values[:, :length, :]
                attended = mx.matmul(
                    probabilities[:, None, :],
                    token_values,
                ).reshape(config.num_q_heads, config.head_dim)
            attended_tokens.append(attended * mx.sigmoid(token_gate))
        attended = mx.stack(attended_tokens)
        output = _linear_batch(
            weights.o_proj,
            attended.reshape(tokens, config.query_dim),
        )
        return output, next_state

    attended = mx.fast.scaled_dot_product_attention(
        mx.transpose(query, (1, 0, 2))[None, :],
        next_keys[None, :],
        next_values[None, :],
        scale=config.head_dim**-0.5,
        mask="causal",
    )
    attended = mx.transpose(attended[0], (1, 0, 2))
    attended = attended * mx.sigmoid(gate)
    output = _linear_batch(weights.o_proj, attended.reshape(tokens, config.query_dim))
    return output, next_state


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


def load_layer(source_path: Path, layer: int) -> MLXAttentionWeights:
    require(layer >= 0 and layer % 4 == 3 and layer < 40, "layer is not an Ornith attention layer")
    prefix = f"model.language_model.layers.{layer}.self_attn"
    with SafetensorsFile(source_path) as source:
        weights = MLXAttentionWeights(
            q_proj=_load_bf16(source, f"{prefix}.q_proj.weight", (8192, 2048)),
            k_proj=_load_bf16(source, f"{prefix}.k_proj.weight", (512, 2048)),
            v_proj=_load_bf16(source, f"{prefix}.v_proj.weight", (512, 2048)),
            o_proj=_load_bf16(source, f"{prefix}.o_proj.weight", (2048, 4096)),
            q_norm=_load_bf16(source, f"{prefix}.q_norm.weight", (256,)),
            k_norm=_load_bf16(source, f"{prefix}.k_norm.weight", (256,)),
        )
        mx.eval(*weights.__dict__.values())
    validate_weights(weights, PRODUCTION_CONFIG)
    return weights

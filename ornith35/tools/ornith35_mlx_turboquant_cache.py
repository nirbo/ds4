#!/usr/bin/env python3
"""Direct packed K4-MSE K/V primitives for the Ornith-35 Metal runtime."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_linear_cache as linear_cache
import ornith35_mlx_turboquant as turboquant
import ornith35_turboquant_reference as reference


HEAD_DIM = 256
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
GQA_GROUPS = 8
PACKED_DIM = HEAD_DIM // 2
PRODUCTION_EXACT_TAIL_TOKENS = 256
KEY_ROTATION_SEED = 202_607_180_101
VALUE_ROTATION_SEED = 202_607_180_103


PACKED_SCORE_KERNEL_SOURCE = r"""
constexpr uint ornith_simdgroups = 8u;
uint work = threadgroup_position_in_grid.x * ornith_simdgroups
    + simdgroup_index_in_threadgroup;
uint history = history_length;
if (work >= 2u * history) return;
uint kv_head = work / history;
uint key_index = work - kv_head * history;
uint lane = thread_index_in_simdgroup;
float totals[8] = {0.0f};
uint capacity = cache_capacity;
uint packed_base = (kv_head * capacity + key_index) * 128u;
for (uint pair = lane; pair < 128u; pair += 32u) {
    uchar packed = packed_keys[packed_base + pair];
    uint dimension = pair * 2u;
    float low = centroids[packed & 15u];
    float high = centroids[packed >> 4u];
    for (uint query = 0u; query < 8u; ++query) {
        uint query_base = (kv_head * 8u + query) * 256u;
        totals[query] += rotated_queries[query_base + dimension] * low;
        totals[query] += rotated_queries[query_base + dimension + 1u] * high;
    }
}
float norm = float(norms[kv_head * capacity + key_index]);
for (uint query = 0u; query < 8u; ++query) {
    for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {
        totals[query] += simd_shuffle_down(totals[query], offset);
    }
    if (lane == 0u) {
        uint head = kv_head * 8u + query;
        scores[head * history + key_index] = totals[query] * norm;
    }
}
"""


_packed_score_kernel = mx.fast.metal_kernel(
    name="ornith35_turboquant_k4_packed_scores_f32",
    input_names=[
        "rotated_queries",
        "packed_keys",
        "norms",
        "centroids",
        "history_length",
        "cache_capacity",
    ],
    output_names=["scores"],
    source=PACKED_SCORE_KERNEL_SOURCE,
)


PACKED_VALUE_AGGREGATE_KERNEL_SOURCE = r"""
uint work = threadgroup_position_in_grid.x;
if (work >= 2u * 64u) return;
uint kv_head = work / 64u;
uint block = work - kv_head * 64u;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint simdgroup = simdgroup_index_in_threadgroup;
uint history = history_length;
uint capacity = cache_capacity;
uint packed_column = block * 2u;
float totals[32] = {0.0f};
threadgroup float partials[8 * 32];
for (uint token = lid; token < history; token += 256u) {
    float norm = float(norms[kv_head * capacity + token]);
    uint packed_base = (kv_head * capacity + token) * 128u + packed_column;
    uchar first = packed_values[packed_base];
    uchar second = packed_values[packed_base + 1u];
    float decoded[4] = {
        centroids[first & 15u],
        centroids[first >> 4u],
        centroids[second & 15u],
        centroids[second >> 4u],
    };
    for (uint query = 0u; query < 8u; ++query) {
        uint head = kv_head * 8u + query;
        float coefficient = probabilities[head * history + token] * norm;
        for (uint item = 0u; item < 4u; ++item) {
            totals[query * 4u + item] += coefficient * decoded[item];
        }
    }
}
for (uint item = 0u; item < 32u; ++item) {
    totals[item] = simd_sum(totals[item]);
    if (lane == 0u) {
        partials[simdgroup * 32u + item] = totals[item];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 32u) {
    float total = 0.0f;
    for (uint group = 0u; group < 8u; ++group) {
        total += partials[group * 32u + lid];
    }
    uint query = lid / 4u;
    uint item = lid - query * 4u;
    uint head = kv_head * 8u + query;
    rotated_values[head * 256u + block * 4u + item] = total;
}
"""


_packed_value_aggregate_kernel = mx.fast.metal_kernel(
    name="ornith35_turboquant_k4_packed_value_aggregate_f32",
    input_names=[
        "probabilities",
        "packed_values",
        "norms",
        "centroids",
        "history_length",
        "cache_capacity",
    ],
    output_names=["rotated_values"],
    source=PACKED_VALUE_AGGREGATE_KERNEL_SOURCE,
)


PACKED_ENCODE_KERNEL_SOURCE = r"""
uint vector = threadgroup_position_in_grid.x;
uint vectors_count = vector_count;
if (vector >= vectors_count) return;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint simdgroup = simdgroup_index_in_threadgroup;
threadgroup float source[256];
threadgroup float norm_partials[8];
threadgroup float inverse_norm[1];
float value = float(vectors[vector * 256u + lid]);
source[lid] = value;
float square_sum = simd_sum(value * value);
if (lane == 0u) {
    norm_partials[simdgroup] = square_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid == 0u) {
    float total = 0.0f;
    for (uint group = 0u; group < 8u; ++group) {
        total += norm_partials[group];
    }
    float norm = metal::precise::sqrt(total);
    norms[vector] = bfloat16_t(norm);
    inverse_norm[0] = norm > 0.0f ? 1.0f / norm : 1.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 128u) {
    uint low_dimension = lid * 2u;
    uint high_dimension = low_dimension + 1u;
    float low = 0.0f;
    float high = 0.0f;
    for (uint input = 0u; input < 256u; ++input) {
        float normalized = source[input] * inverse_norm[0];
        low += normalized * rotation[low_dimension * 256u + input];
        high += normalized * rotation[high_dimension * 256u + input];
    }
    uint low_index = 0u;
    uint high_index = 0u;
    for (uint boundary = 0u; boundary < 15u; ++boundary) {
        low_index += low >= boundaries[boundary];
        high_index += high >= boundaries[boundary];
    }
    packed[vector * 128u + lid] = uchar(low_index | (high_index << 4u));
}
"""


_packed_encode_kernel = mx.fast.metal_kernel(
    name="ornith35_turboquant_k4_encode_bf16",
    input_names=["vectors", "rotation", "boundaries", "vector_count"],
    output_names=["packed", "norms"],
    source=PACKED_ENCODE_KERNEL_SOURCE,
)


@dataclass(frozen=True)
class MLXPackedMSE4:
    packed: mx.array
    norms: mx.array


@dataclass(frozen=True)
class MLXPackedMSE4Transforms:
    key: turboquant.MLXTransform
    value: turboquant.MLXTransform


@dataclass(frozen=True)
class MLXPackedMSE4State:
    packed_keys: mx.array
    key_norms: mx.array
    packed_values: mx.array
    value_norms: mx.array
    exact_keys: mx.array
    exact_values: mx.array
    exact_tail_capacity: int = PRODUCTION_EXACT_TAIL_TOKENS
    context_profile: str = context.NATIVE_PROFILE_ID


@dataclass(frozen=True)
class MLXLinearPackedMSE4State:
    """Fixed-capacity packed K/V owned by one advancing decode session."""

    packed_keys: mx.array
    key_norms: mx.array
    packed_values: mx.array
    value_norms: mx.array
    exact_keys: mx.array
    exact_values: mx.array
    position: int
    capacity: int
    exact_tail_capacity: int = PRODUCTION_EXACT_TAIL_TOKENS
    context_profile: str = context.NATIVE_PROFILE_ID


PackedMSE4State = MLXPackedMSE4State | MLXLinearPackedMSE4State


def require(condition: bool, message: str) -> None:
    if not condition:
        raise reference.TurboQuantError(message)


@lru_cache(maxsize=1)
def production_transforms() -> MLXPackedMSE4Transforms:
    return MLXPackedMSE4Transforms(
        key=turboquant.haar_rotation(HEAD_DIM, KEY_ROTATION_SEED),
        value=turboquant.haar_rotation(HEAD_DIM, VALUE_ROTATION_SEED),
    )


def _centroids() -> mx.array:
    return mx.array(reference.codebook(HEAD_DIM, 4).centroids, dtype=mx.float32)


def _boundaries() -> mx.array:
    values = reference.codebook(HEAD_DIM, 4).boundaries[1:-1]
    return mx.array(values, dtype=mx.float32)


def encode_mse4_graph(
    vectors: mx.array,
    rotation: turboquant.MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXPackedMSE4:
    """Encode and pack entirely in the MLX graph, including zero vectors."""
    require(vectors.ndim >= 2 and vectors.shape[-1] == HEAD_DIM, "K4 input geometry mismatch")
    require(
        rotation.matrix.shape == (HEAD_DIM, HEAD_DIM)
        and rotation.matrix.dtype == mx.float32,
        "K4 rotation mismatch",
    )
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported K4 norm dtype")
    source = vectors.astype(mx.float32)
    norms = mx.sqrt(mx.sum(source * source, axis=-1, keepdims=True))
    safe_norms = mx.where(norms > 0.0, norms, 1.0)
    rotated = (source / safe_norms) @ mx.swapaxes(rotation.matrix, -2, -1)
    indices = mx.zeros(rotated.shape, dtype=mx.uint8)
    for boundary in _boundaries():
        indices = indices + (rotated >= boundary).astype(mx.uint8)
    packed = (
        indices[..., 0::2]
        + indices[..., 1::2] * mx.array(16, dtype=mx.uint8)
    ).astype(mx.uint8)
    return MLXPackedMSE4(packed=packed, norms=norms.astype(norm_dtype))


def encode_mse4(
    vectors: mx.array,
    rotation: turboquant.MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXPackedMSE4:
    """Encode in one Metal dispatch; retain the graph path as the oracle."""
    require(vectors.ndim >= 2 and vectors.shape[-1] == HEAD_DIM, "K4 input geometry mismatch")
    require(
        rotation.matrix.shape == (HEAD_DIM, HEAD_DIM)
        and rotation.matrix.dtype == mx.float32,
        "K4 rotation mismatch",
    )
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported K4 norm dtype")
    if norm_dtype != mx.bfloat16 or vectors.size == 0:
        return encode_mse4_graph(vectors, rotation, norm_dtype=norm_dtype)
    vectors_count = vectors.size // HEAD_DIM
    packed, norms = _packed_encode_kernel(
        inputs=[
            vectors,
            rotation.matrix,
            _boundaries(),
            mx.array(vectors_count, dtype=mx.uint32),
        ],
        grid=(vectors_count * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*vectors.shape[:-1], PACKED_DIM), (*vectors.shape[:-1], 1)],
        output_dtypes=[mx.uint8, mx.bfloat16],
    )
    return MLXPackedMSE4(packed=packed, norms=norms)


def unpack_indices(encoding: MLXPackedMSE4) -> mx.array:
    validate_encoding(encoding)
    low = encoding.packed & mx.array(15, dtype=mx.uint8)
    high = encoding.packed >> mx.array(4, dtype=mx.uint8)
    return mx.stack((low, high), axis=-1).reshape(*encoding.packed.shape[:-1], HEAD_DIM)


def dequantize_mse4(
    encoding: MLXPackedMSE4,
    rotation: turboquant.MLXTransform,
) -> mx.array:
    indices = unpack_indices(encoding)
    return turboquant.dequantize_mse(
        turboquant.MLXMSEEncoding(
            indices=indices,
            norms=encoding.norms,
            bits=4,
            dimension=HEAD_DIM,
        ),
        rotation,
    )


def validate_encoding(encoding: MLXPackedMSE4) -> None:
    require(isinstance(encoding, MLXPackedMSE4), "invalid packed K4 encoding")
    require(
        encoding.packed.ndim >= 2
        and encoding.packed.shape[-1] == PACKED_DIM
        and encoding.packed.dtype == mx.uint8,
        "packed K4 payload mismatch",
    )
    require(
        encoding.norms.shape == (*encoding.packed.shape[:-1], 1)
        and encoding.norms.dtype in (mx.bfloat16, mx.float32),
        "packed K4 norm mismatch",
    )


def validate_state(state: PackedMSE4State) -> None:
    require(
        isinstance(state, (MLXPackedMSE4State, MLXLinearPackedMSE4State)),
        "invalid packed K/V state",
    )
    physical_capacity = state.packed_keys.shape[1]
    tail = state.exact_keys.shape[1]
    require(
        type(state.exact_tail_capacity) is int
        and 0 <= state.exact_tail_capacity <= PRODUCTION_EXACT_TAIL_TOKENS,
        "packed K/V exact-tail capacity is invalid",
    )
    expected_packed = (NUM_KV_HEADS, physical_capacity, PACKED_DIM)
    expected_norms = (NUM_KV_HEADS, physical_capacity, 1)
    expected_exact = (NUM_KV_HEADS, tail, HEAD_DIM)
    require(
        state.packed_keys.shape == expected_packed
        and state.packed_values.shape == expected_packed
        and state.packed_keys.dtype == state.packed_values.dtype == mx.uint8,
        "packed K/V payload mismatch",
    )
    require(
        state.key_norms.shape == expected_norms
        and state.value_norms.shape == expected_norms
        and state.key_norms.dtype == state.value_norms.dtype == mx.bfloat16,
        "packed K/V norm mismatch",
    )
    require(
        state.exact_keys.shape == expected_exact
        and state.exact_values.shape == expected_exact
        and state.exact_keys.dtype == state.exact_values.dtype == mx.bfloat16,
        "exact K/V tail mismatch",
    )
    if isinstance(state, MLXLinearPackedMSE4State):
        require(
            isinstance(state.position, int)
            and isinstance(state.capacity, int)
            and state.capacity == physical_capacity,
            "linear packed K/V capacity mismatch",
        )
        require(
            0 <= state.position <= state.capacity,
            "linear packed K/V position is outside capacity",
        )
        require(tail == min(state.position, state.exact_tail_capacity), "linear exact tail mismatch")
        context.validate_range(state.context_profile, 0, state.capacity)
        return
    require(
        tail == min(physical_capacity + tail, state.exact_tail_capacity),
        "immutable exact tail mismatch",
    )
    context.validate_range(state.context_profile, 0, physical_capacity + tail)


def state_length(state: PackedMSE4State) -> int:
    validate_state(state)
    if isinstance(state, MLXLinearPackedMSE4State):
        return state.position
    return state.packed_keys.shape[1] + state.exact_keys.shape[1]


def packed_history(state: PackedMSE4State) -> int:
    validate_state(state)
    if isinstance(state, MLXLinearPackedMSE4State):
        return state.position - state.exact_keys.shape[1]
    return state.packed_keys.shape[1]


def packed_capacity(state: PackedMSE4State) -> int:
    validate_state(state)
    return state.packed_keys.shape[1]


def compress_bf16_kv(
    keys: mx.array,
    values: mx.array,
    transforms: MLXPackedMSE4Transforms | None = None,
    *,
    exact_tail: int = PRODUCTION_EXACT_TAIL_TOKENS,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXPackedMSE4State:
    require(
        keys.ndim == values.ndim == 3
        and keys.shape == values.shape
        and keys.shape[0] == NUM_KV_HEADS
        and keys.shape[2] == HEAD_DIM,
        "BF16 K/V source geometry mismatch",
    )
    require(keys.dtype == values.dtype == mx.bfloat16, "BF16 K/V source dtype mismatch")
    require(
        0 <= exact_tail <= PRODUCTION_EXACT_TAIL_TOKENS,
        "invalid exact tail",
    )
    context.validate_range(context_profile, 0, keys.shape[1])
    if transforms is None:
        transforms = production_transforms()
    retained = min(keys.shape[1], exact_tail)
    history = keys.shape[1] - retained
    encoded_keys = encode_mse4(keys[:, :history], transforms.key)
    encoded_values = encode_mse4(values[:, :history], transforms.value)
    state = MLXPackedMSE4State(
        packed_keys=encoded_keys.packed,
        key_norms=encoded_keys.norms,
        packed_values=encoded_values.packed,
        value_norms=encoded_values.norms,
        exact_keys=keys[:, history:],
        exact_values=values[:, history:],
        exact_tail_capacity=exact_tail,
        context_profile=context_profile,
    )
    validate_state(state)
    return state


def linearize_state(
    state: MLXPackedMSE4State,
    capacity: int,
) -> MLXLinearPackedMSE4State:
    """Copy an immutable packed prefix into single-owner fixed-capacity buffers."""
    require(isinstance(state, MLXPackedMSE4State), "linear source state must be immutable")
    validate_state(state)
    position = state_length(state)
    require(isinstance(capacity, int) and capacity >= position, "linear capacity is too short")
    context.validate_range(state.context_profile, 0, capacity)
    packed_shape = (NUM_KV_HEADS, capacity, PACKED_DIM)
    norm_shape = (NUM_KV_HEADS, capacity, 1)
    packed_keys = mx.zeros(packed_shape, dtype=mx.uint8)
    key_norms = mx.zeros(norm_shape, dtype=mx.bfloat16)
    packed_values = mx.zeros(packed_shape, dtype=mx.uint8)
    value_norms = mx.zeros(norm_shape, dtype=mx.bfloat16)
    history = state.packed_keys.shape[1]
    if history:
        packed_keys, key_norms, packed_values, value_norms = (
            linear_cache.append_packed_mse4(
                packed_keys,
                key_norms,
                packed_values,
                value_norms,
                state.packed_keys,
                state.key_norms,
                state.packed_values,
                state.value_norms,
                0,
            )
        )
    linear = MLXLinearPackedMSE4State(
        packed_keys=packed_keys,
        key_norms=key_norms,
        packed_values=packed_values,
        value_norms=value_norms,
        exact_keys=state.exact_keys,
        exact_values=state.exact_values,
        position=position,
        capacity=capacity,
        exact_tail_capacity=state.exact_tail_capacity,
        context_profile=state.context_profile,
    )
    validate_state(linear)
    return linear


def linearize_bf16_kv(
    keys: mx.array,
    values: mx.array,
    capacity: int,
    transforms: MLXPackedMSE4Transforms | None = None,
    *,
    exact_tail: int = PRODUCTION_EXACT_TAIL_TOKENS,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXLinearPackedMSE4State:
    """Encode a BF16 prefix with a bounded exact tail into fixed-capacity storage."""
    immutable = compress_bf16_kv(
        keys,
        values,
        transforms,
        exact_tail=exact_tail,
        context_profile=context_profile,
    )
    return linearize_state(immutable, capacity)


def advance_linear_state(
    state: MLXLinearPackedMSE4State,
    key_update: mx.array,
    value_update: mx.array,
    transforms: MLXPackedMSE4Transforms | None = None,
) -> MLXLinearPackedMSE4State:
    """Append BF16 K/V while moving the prior tail into packed storage."""
    require(isinstance(state, MLXLinearPackedMSE4State), "packed append state must be linear")
    validate_state(state)
    require(
        key_update.ndim == value_update.ndim == 3
        and key_update.shape == value_update.shape
        and key_update.shape[0] == NUM_KV_HEADS
        and key_update.shape[1] > 0
        and key_update.shape[2] == HEAD_DIM,
        "packed BF16 K/V update geometry mismatch",
    )
    require(
        key_update.dtype == value_update.dtype == mx.bfloat16,
        "packed K/V update must be BF16",
    )
    tokens = key_update.shape[1]
    require(state.position + tokens <= state.capacity, "linear packed K/V capacity exhausted")
    context.validate_range(state.context_profile, state.position, tokens)
    if transforms is None:
        transforms = production_transforms()

    combined_keys = (
        key_update
        if not state.exact_keys.shape[1]
        else mx.concatenate((state.exact_keys, key_update), axis=1)
    )
    combined_values = (
        value_update
        if not state.exact_values.shape[1]
        else mx.concatenate((state.exact_values, value_update), axis=1)
    )
    next_tail = min(state.position + tokens, state.exact_tail_capacity)
    pack_count = combined_keys.shape[1] - next_tail
    if pack_count:
        keys_to_pack = combined_keys[:, :pack_count]
        values_to_pack = combined_values[:, :pack_count]
        encoded_keys = encode_mse4(keys_to_pack, transforms.key)
        encoded_values = encode_mse4(values_to_pack, transforms.value)
        write_position = state.position - state.exact_keys.shape[1]
        packed_keys, key_norms, packed_values, value_norms = (
            linear_cache.append_packed_mse4(
                state.packed_keys,
                state.key_norms,
                state.packed_values,
                state.value_norms,
                encoded_keys.packed,
                encoded_keys.norms,
                encoded_values.packed,
                encoded_values.norms,
                write_position,
            )
        )
    else:
        packed_keys = state.packed_keys
        key_norms = state.key_norms
        packed_values = state.packed_values
        value_norms = state.value_norms
    advanced = MLXLinearPackedMSE4State(
        packed_keys=packed_keys,
        key_norms=key_norms,
        packed_values=packed_values,
        value_norms=value_norms,
        exact_keys=combined_keys[:, -next_tail:] if next_tail else combined_keys[:, :0],
        exact_values=combined_values[:, -next_tail:] if next_tail else combined_values[:, :0],
        position=state.position + tokens,
        capacity=state.capacity,
        exact_tail_capacity=state.exact_tail_capacity,
        context_profile=state.context_profile,
    )
    validate_state(advanced)
    return advanced


def dequantize_state(
    state: PackedMSE4State,
    transforms: MLXPackedMSE4Transforms | None = None,
) -> tuple[mx.array, mx.array]:
    """Materialize BF16-shaped history for tests only, never runtime attention."""
    validate_state(state)
    if transforms is None:
        transforms = production_transforms()
    history = packed_history(state)
    keys = dequantize_mse4(
        MLXPackedMSE4(state.packed_keys[:, :history], state.key_norms[:, :history]),
        transforms.key,
    )
    values = dequantize_mse4(
        MLXPackedMSE4(state.packed_values[:, :history], state.value_norms[:, :history]),
        transforms.value,
    )
    return (
        mx.concatenate((keys, state.exact_keys.astype(mx.float32)), axis=1),
        mx.concatenate((values, state.exact_values.astype(mx.float32)), axis=1),
    )


def packed_scores(
    queries: mx.array,
    state: PackedMSE4State,
    transforms: MLXPackedMSE4Transforms | None = None,
) -> mx.array:
    """Score packed historical keys directly and append the exact-tail scores."""
    validate_state(state)
    require(
        queries.shape == (NUM_Q_HEADS, HEAD_DIM)
        and queries.dtype in (mx.bfloat16, mx.float32),
        "packed score query mismatch",
    )
    if transforms is None:
        transforms = production_transforms()
    history = packed_history(state)
    if history:
        rotated_queries = queries.astype(mx.float32) @ mx.swapaxes(
            transforms.key.matrix,
            -2,
            -1,
        )
        work_items = NUM_KV_HEADS * history
        threadgroups = (work_items + 7) // 8
        historical = _packed_score_kernel(
            inputs=[
                rotated_queries,
                state.packed_keys,
                state.key_norms,
                _centroids(),
                mx.array(history, dtype=mx.uint32),
                mx.array(packed_capacity(state), dtype=mx.uint32),
            ],
            grid=(threadgroups * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(NUM_Q_HEADS, history)],
            output_dtypes=[mx.float32],
        )[0]
    else:
        historical = mx.zeros((NUM_Q_HEADS, 0), dtype=mx.float32)
    tail = state.exact_keys.shape[1]
    if not tail:
        return historical
    repeated_keys = mx.repeat(state.exact_keys.astype(mx.float32), GQA_GROUPS, axis=0)
    exact = mx.sum(queries.astype(mx.float32)[:, None, :] * repeated_keys, axis=-1)
    return mx.concatenate((historical, exact), axis=1)


def packed_attend(
    probabilities: mx.array,
    state: PackedMSE4State,
    transforms: MLXPackedMSE4Transforms | None = None,
) -> mx.array:
    """Aggregate packed values directly, inverse-rotating only the final sums."""
    validate_state(state)
    length = state_length(state)
    require(
        probabilities.shape == (NUM_Q_HEADS, length)
        and probabilities.dtype in (mx.bfloat16, mx.float32),
        "packed probability mismatch",
    )
    if transforms is None:
        transforms = production_transforms()
    history = packed_history(state)
    if history:
        rotated = _packed_value_aggregate_kernel(
            inputs=[
                probabilities[:, :history].astype(mx.float32),
                state.packed_values,
                state.value_norms,
                _centroids(),
                mx.array(history, dtype=mx.uint32),
                mx.array(packed_capacity(state), dtype=mx.uint32),
            ],
            grid=(NUM_KV_HEADS * 64 * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(NUM_Q_HEADS, HEAD_DIM)],
            output_dtypes=[mx.float32],
        )[0]
        attended = rotated @ transforms.value.matrix
    else:
        attended = mx.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=mx.float32)
    tail = state.exact_values.shape[1]
    if tail:
        repeated_values = mx.repeat(state.exact_values.astype(mx.float32), GQA_GROUPS, axis=0)
        attended = attended + mx.sum(
            probabilities[:, history:, None].astype(mx.float32) * repeated_values,
            axis=1,
        )
    return attended


def packed_attention(
    queries: mx.array,
    state: PackedMSE4State,
    transforms: MLXPackedMSE4Transforms | None = None,
) -> tuple[mx.array, mx.array]:
    scores = packed_scores(queries, state, transforms) * (HEAD_DIM**-0.5)
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1)
    return packed_attend(probabilities, state, transforms), probabilities


def stored_bytes(state: PackedMSE4State) -> int:
    validate_state(state)
    arrays = (
        state.packed_keys,
        state.key_norms,
        state.packed_values,
        state.value_norms,
        state.exact_keys,
        state.exact_values,
    )
    return sum(array.size * array.itemsize for array in arrays)

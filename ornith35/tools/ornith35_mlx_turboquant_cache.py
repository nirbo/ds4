#!/usr/bin/env python3
"""Direct packed K8-MSE K/V primitives for the Ornith-35 Metal runtime."""

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
PACKED_BITS = 8
PACKED_DIM = (HEAD_DIM * PACKED_BITS + 7) // 8
PRODUCTION_EXACT_HEAD_TOKENS = 256
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
uint packed_base = (kv_head * capacity + key_index) * 256u;
for (uint dimension = lane; dimension < 256u; dimension += 32u) {
    float decoded = centroids[uint(packed_keys[packed_base + dimension])];
    for (uint query = 0u; query < 8u; ++query) {
        uint query_base = (kv_head * 8u + query) * 256u;
        totals[query] += rotated_queries[query_base + dimension] * decoded;
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
    name="ornith35_turboquant_k8_packed_scores_f32",
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
uint packed_column = block * 4u;
float totals[32] = {0.0f};
threadgroup float partials[8 * 32];
for (uint token = lid; token < history; token += 256u) {
    float norm = float(norms[kv_head * capacity + token]);
    uint packed_base = (kv_head * capacity + token) * 256u + packed_column;
    uint word = uint(packed_values[packed_base]);
    word |= uint(packed_values[packed_base + 1u]) << 8u;
    word |= uint(packed_values[packed_base + 2u]) << 16u;
    word |= uint(packed_values[packed_base + 3u]) << 24u;
    float decoded[4] = {
        centroids[word & 255u],
        centroids[(word >> 8u) & 255u],
        centroids[(word >> 16u) & 255u],
        centroids[word >> 24u],
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
    name="ornith35_turboquant_k8_packed_value_aggregate_f32",
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


PACKED_CLASSIFY_KERNEL_SOURCE = r"""
uint coordinate = thread_position_in_grid.x;
if (coordinate >= coordinate_count) return;
float rotated_value = rotated[coordinate];
{
    uint low = 0u;
    uint high = 255u;
    while (low < high) {
        uint middle = (low + high) >> 1u;
        if (rotated_value >= boundaries[middle]) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    packed[coordinate] = uchar(low);
}
"""


_packed_classify_kernel = mx.fast.metal_kernel(
    name="ornith35_turboquant_k8_classify_f32",
    input_names=["rotated", "boundaries", "coordinate_count"],
    output_names=["packed"],
    source=PACKED_CLASSIFY_KERNEL_SOURCE,
)


@dataclass(frozen=True)
class MLXPackedMSE8:
    packed: mx.array
    norms: mx.array


@dataclass(frozen=True)
class MLXPackedMSE8Transforms:
    key: turboquant.MLXTransform
    value: turboquant.MLXTransform


@dataclass(frozen=True)
class MLXPackedMSE8State:
    packed_keys: mx.array
    key_norms: mx.array
    packed_values: mx.array
    value_norms: mx.array
    exact_head_keys: mx.array
    exact_head_values: mx.array
    exact_keys: mx.array
    exact_values: mx.array
    exact_head_capacity: int = PRODUCTION_EXACT_HEAD_TOKENS
    exact_tail_capacity: int = PRODUCTION_EXACT_TAIL_TOKENS
    context_profile: str = context.NATIVE_PROFILE_ID


@dataclass(frozen=True)
class MLXLinearPackedMSE8State:
    """Fixed-capacity packed K/V owned by one advancing decode session."""

    packed_keys: mx.array
    key_norms: mx.array
    packed_values: mx.array
    value_norms: mx.array
    exact_head_keys: mx.array
    exact_head_values: mx.array
    exact_keys: mx.array
    exact_values: mx.array
    position: int
    capacity: int
    exact_head_capacity: int = PRODUCTION_EXACT_HEAD_TOKENS
    exact_tail_capacity: int = PRODUCTION_EXACT_TAIL_TOKENS
    context_profile: str = context.NATIVE_PROFILE_ID


PackedMSE8State = MLXPackedMSE8State | MLXLinearPackedMSE8State


def require(condition: bool, message: str) -> None:
    if not condition:
        raise reference.TurboQuantError(message)


@lru_cache(maxsize=1)
def production_transforms() -> MLXPackedMSE8Transforms:
    return MLXPackedMSE8Transforms(
        key=turboquant.haar_rotation(HEAD_DIM, KEY_ROTATION_SEED),
        value=turboquant.haar_rotation(HEAD_DIM, VALUE_ROTATION_SEED),
    )


def _centroids() -> mx.array:
    return mx.array(reference.codebook(HEAD_DIM, PACKED_BITS).centroids, dtype=mx.float32)


def _boundaries() -> mx.array:
    values = reference.codebook(HEAD_DIM, PACKED_BITS).boundaries[1:-1]
    return mx.array(values, dtype=mx.float32)


def _pack_indices(indices: mx.array) -> mx.array:
    coordinates = mx.array(
        tuple((byte * 8) // PACKED_BITS for byte in range(PACKED_DIM)),
        dtype=mx.uint32,
    )
    shifts = mx.array(
        tuple((byte * 8) % PACKED_BITS for byte in range(PACKED_DIM)),
        dtype=mx.uint32,
    )
    padding = mx.zeros((*indices.shape[:-1], 2), dtype=mx.uint8)
    padded = mx.concatenate((indices, padding), axis=-1).astype(mx.uint32)
    word = mx.take(padded, coordinates, axis=-1)
    word = word | (mx.take(padded, coordinates + 1, axis=-1) << PACKED_BITS)
    word = word | (mx.take(padded, coordinates + 2, axis=-1) << (2 * PACKED_BITS))
    return ((word >> shifts) & 255).astype(mx.uint8)


def encode_mse8_graph(
    vectors: mx.array,
    rotation: turboquant.MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXPackedMSE8:
    """Encode and pack entirely in the MLX graph, including zero vectors."""
    require(vectors.ndim >= 2 and vectors.shape[-1] == HEAD_DIM, "K8 input geometry mismatch")
    require(
        rotation.matrix.shape == (HEAD_DIM, HEAD_DIM)
        and rotation.matrix.dtype == mx.float32,
        "K8 rotation mismatch",
    )
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported K8 norm dtype")
    source = vectors.astype(mx.float32)
    norms = mx.sqrt(mx.sum(source * source, axis=-1, keepdims=True))
    safe_norms = mx.where(norms > 0.0, norms, 1.0)
    rotated = (source / safe_norms) @ mx.swapaxes(rotation.matrix, -2, -1)
    indices = mx.zeros(rotated.shape, dtype=mx.uint8)
    for boundary in _boundaries():
        indices = indices + (rotated >= boundary).astype(mx.uint8)
    packed = _pack_indices(indices)
    return MLXPackedMSE8(packed=packed, norms=norms.astype(norm_dtype))


def encode_mse8(
    vectors: mx.array,
    rotation: turboquant.MLXTransform,
    *,
    norm_dtype: mx.Dtype = mx.bfloat16,
) -> MLXPackedMSE8:
    """Use authoritative MLX rotation followed by direct-byte Metal classification."""
    require(vectors.ndim >= 2 and vectors.shape[-1] == HEAD_DIM, "K8 input geometry mismatch")
    require(
        rotation.matrix.shape == (HEAD_DIM, HEAD_DIM)
        and rotation.matrix.dtype == mx.float32,
        "K8 rotation mismatch",
    )
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported K8 norm dtype")
    if norm_dtype != mx.bfloat16 or vectors.size == 0:
        return encode_mse8_graph(vectors, rotation, norm_dtype=norm_dtype)
    source = vectors.astype(mx.float32)
    norms = mx.sqrt(mx.sum(source * source, axis=-1, keepdims=True))
    safe_norms = mx.where(norms > 0.0, norms, 1.0)
    rotated = (source / safe_norms) @ mx.swapaxes(rotation.matrix, -2, -1)
    (packed,) = _packed_classify_kernel(
        inputs=[
            rotated,
            _boundaries(),
            mx.array(vectors.size, dtype=mx.uint32),
        ],
        grid=(vectors.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[vectors.shape],
        output_dtypes=[mx.uint8],
    )
    return MLXPackedMSE8(packed=packed, norms=norms.astype(mx.bfloat16))


def unpack_indices(encoding: MLXPackedMSE8) -> mx.array:
    validate_encoding(encoding)
    return encoding.packed


def dequantize_mse8(
    encoding: MLXPackedMSE8,
    rotation: turboquant.MLXTransform,
) -> mx.array:
    indices = unpack_indices(encoding)
    return turboquant.dequantize_mse(
        turboquant.MLXMSEEncoding(
            indices=indices,
            norms=encoding.norms,
            bits=PACKED_BITS,
            dimension=HEAD_DIM,
        ),
        rotation,
    )


def validate_encoding(encoding: MLXPackedMSE8) -> None:
    require(isinstance(encoding, MLXPackedMSE8), "invalid packed K8 encoding")
    require(
        encoding.packed.ndim >= 2
        and encoding.packed.shape[-1] == PACKED_DIM
        and encoding.packed.dtype == mx.uint8,
        "packed K8 payload mismatch",
    )
    require(
        encoding.norms.shape == (*encoding.packed.shape[:-1], 1)
        and encoding.norms.dtype in (mx.bfloat16, mx.float32),
        "packed K8 norm mismatch",
    )


def validate_state(state: PackedMSE8State) -> None:
    require(
        isinstance(state, (MLXPackedMSE8State, MLXLinearPackedMSE8State)),
        "invalid packed K/V state",
    )
    physical_capacity = state.packed_keys.shape[1]
    head = state.exact_head_keys.shape[1]
    tail = state.exact_keys.shape[1]
    require(
        type(state.exact_head_capacity) is int
        and 0 <= state.exact_head_capacity <= PRODUCTION_EXACT_HEAD_TOKENS,
        "packed K/V exact-head capacity is invalid",
    )
    require(
        type(state.exact_tail_capacity) is int
        and 0 <= state.exact_tail_capacity <= PRODUCTION_EXACT_TAIL_TOKENS,
        "packed K/V exact-tail capacity is invalid",
    )
    expected_packed = (NUM_KV_HEADS, physical_capacity, PACKED_DIM)
    expected_norms = (NUM_KV_HEADS, physical_capacity, 1)
    expected_head = (NUM_KV_HEADS, head, HEAD_DIM)
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
        state.exact_head_keys.shape == expected_head
        and state.exact_head_values.shape == expected_head
        and state.exact_head_keys.dtype == state.exact_head_values.dtype == mx.bfloat16,
        "exact K/V head mismatch",
    )
    require(
        state.exact_keys.shape == expected_exact
        and state.exact_values.shape == expected_exact
        and state.exact_keys.dtype == state.exact_values.dtype == mx.bfloat16,
        "exact K/V tail mismatch",
    )
    if isinstance(state, MLXLinearPackedMSE8State):
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
        require(
            head == min(state.position, state.exact_head_capacity),
            "linear exact head mismatch",
        )
        remaining = state.position - head
        require(tail == min(remaining, state.exact_tail_capacity), "linear exact tail mismatch")
        require(remaining - tail <= physical_capacity, "linear packed history exceeds capacity")
        context.validate_range(state.context_profile, 0, state.capacity)
        return
    total = head + physical_capacity + tail
    require(
        head == min(total, state.exact_head_capacity),
        "immutable exact head mismatch",
    )
    require(
        tail == min(total - head, state.exact_tail_capacity),
        "immutable exact tail mismatch",
    )
    context.validate_range(state.context_profile, 0, total)


def state_length(state: PackedMSE8State) -> int:
    validate_state(state)
    if isinstance(state, MLXLinearPackedMSE8State):
        return state.position
    return (
        state.exact_head_keys.shape[1]
        + state.packed_keys.shape[1]
        + state.exact_keys.shape[1]
    )


def packed_history(state: PackedMSE8State) -> int:
    validate_state(state)
    if isinstance(state, MLXLinearPackedMSE8State):
        return (
            state.position
            - state.exact_head_keys.shape[1]
            - state.exact_keys.shape[1]
        )
    return state.packed_keys.shape[1]


def packed_capacity(state: PackedMSE8State) -> int:
    validate_state(state)
    return state.packed_keys.shape[1]


def _materialize_boundary(values: mx.array, start: int, stop: int) -> mx.array:
    """Give a retained exact boundary storage independent from a larger source."""
    return mx.contiguous(values[:, start:stop])


def compress_bf16_kv(
    keys: mx.array,
    values: mx.array,
    transforms: MLXPackedMSE8Transforms | None = None,
    *,
    exact_head: int = PRODUCTION_EXACT_HEAD_TOKENS,
    exact_tail: int = PRODUCTION_EXACT_TAIL_TOKENS,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXPackedMSE8State:
    require(
        keys.ndim == values.ndim == 3
        and keys.shape == values.shape
        and keys.shape[0] == NUM_KV_HEADS
        and keys.shape[2] == HEAD_DIM,
        "BF16 K/V source geometry mismatch",
    )
    require(keys.dtype == values.dtype == mx.bfloat16, "BF16 K/V source dtype mismatch")
    require(
        0 <= exact_head <= PRODUCTION_EXACT_HEAD_TOKENS,
        "invalid exact head",
    )
    require(
        0 <= exact_tail <= PRODUCTION_EXACT_TAIL_TOKENS,
        "invalid exact tail",
    )
    context.validate_range(context_profile, 0, keys.shape[1])
    if transforms is None:
        transforms = production_transforms()
    head = min(keys.shape[1], exact_head)
    remaining = keys.shape[1] - head
    tail = min(remaining, exact_tail)
    history = remaining - tail
    packed_end = head + history
    encoded_keys = encode_mse8(keys[:, head:packed_end], transforms.key)
    encoded_values = encode_mse8(values[:, head:packed_end], transforms.value)
    state = MLXPackedMSE8State(
        packed_keys=encoded_keys.packed,
        key_norms=encoded_keys.norms,
        packed_values=encoded_values.packed,
        value_norms=encoded_values.norms,
        exact_head_keys=_materialize_boundary(keys, 0, head),
        exact_head_values=_materialize_boundary(values, 0, head),
        exact_keys=_materialize_boundary(keys, packed_end, keys.shape[1]),
        exact_values=_materialize_boundary(values, packed_end, values.shape[1]),
        exact_head_capacity=exact_head,
        exact_tail_capacity=exact_tail,
        context_profile=context_profile,
    )
    validate_state(state)
    return state


def linearize_state(
    state: MLXPackedMSE8State,
    capacity: int,
) -> MLXLinearPackedMSE8State:
    """Copy an immutable packed prefix into single-owner fixed-capacity buffers."""
    require(isinstance(state, MLXPackedMSE8State), "linear source state must be immutable")
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
            linear_cache.append_packed_mse8(
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
    linear = MLXLinearPackedMSE8State(
        packed_keys=packed_keys,
        key_norms=key_norms,
        packed_values=packed_values,
        value_norms=value_norms,
        exact_head_keys=state.exact_head_keys,
        exact_head_values=state.exact_head_values,
        exact_keys=state.exact_keys,
        exact_values=state.exact_values,
        position=position,
        capacity=capacity,
        exact_head_capacity=state.exact_head_capacity,
        exact_tail_capacity=state.exact_tail_capacity,
        context_profile=state.context_profile,
    )
    validate_state(linear)
    return linear


def linearize_bf16_kv(
    keys: mx.array,
    values: mx.array,
    capacity: int,
    transforms: MLXPackedMSE8Transforms | None = None,
    *,
    exact_head: int = PRODUCTION_EXACT_HEAD_TOKENS,
    exact_tail: int = PRODUCTION_EXACT_TAIL_TOKENS,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXLinearPackedMSE8State:
    """Encode a BF16 prefix with a bounded exact tail into fixed-capacity storage."""
    immutable = compress_bf16_kv(
        keys,
        values,
        transforms,
        exact_head=exact_head,
        exact_tail=exact_tail,
        context_profile=context_profile,
    )
    return linearize_state(immutable, capacity)


def advance_linear_state(
    state: MLXLinearPackedMSE8State,
    key_update: mx.array,
    value_update: mx.array,
    transforms: MLXPackedMSE8Transforms | None = None,
) -> MLXLinearPackedMSE8State:
    """Append BF16 K/V while moving the prior tail into packed storage."""
    require(isinstance(state, MLXLinearPackedMSE8State), "packed append state must be linear")
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

    head_needed = state.exact_head_capacity - state.exact_head_keys.shape[1]
    head_update = min(tokens, head_needed)
    next_head_keys = (
        state.exact_head_keys
        if not head_update
        else mx.contiguous(
            mx.concatenate((state.exact_head_keys, key_update[:, :head_update]), axis=1)
        )
    )
    next_head_values = (
        state.exact_head_values
        if not head_update
        else mx.contiguous(
            mx.concatenate((state.exact_head_values, value_update[:, :head_update]), axis=1)
        )
    )
    remaining_keys = key_update[:, head_update:]
    remaining_values = value_update[:, head_update:]
    combined_keys = (
        remaining_keys
        if not state.exact_keys.shape[1]
        else mx.concatenate((state.exact_keys, remaining_keys), axis=1)
    )
    combined_values = (
        remaining_values
        if not state.exact_values.shape[1]
        else mx.concatenate((state.exact_values, remaining_values), axis=1)
    )
    next_position = state.position + tokens
    next_tail = min(next_position - next_head_keys.shape[1], state.exact_tail_capacity)
    pack_count = combined_keys.shape[1] - next_tail
    if pack_count:
        keys_to_pack = combined_keys[:, :pack_count]
        values_to_pack = combined_values[:, :pack_count]
        encoded_keys = encode_mse8(keys_to_pack, transforms.key)
        encoded_values = encode_mse8(values_to_pack, transforms.value)
        write_position = packed_history(state)
        packed_keys, key_norms, packed_values, value_norms = (
            linear_cache.append_packed_mse8(
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
    advanced = MLXLinearPackedMSE8State(
        packed_keys=packed_keys,
        key_norms=key_norms,
        packed_values=packed_values,
        value_norms=value_norms,
        exact_head_keys=next_head_keys,
        exact_head_values=next_head_values,
        exact_keys=_materialize_boundary(
            combined_keys,
            combined_keys.shape[1] - next_tail,
            combined_keys.shape[1],
        ),
        exact_values=_materialize_boundary(
            combined_values,
            combined_values.shape[1] - next_tail,
            combined_values.shape[1],
        ),
        position=next_position,
        capacity=state.capacity,
        exact_head_capacity=state.exact_head_capacity,
        exact_tail_capacity=state.exact_tail_capacity,
        context_profile=state.context_profile,
    )
    validate_state(advanced)
    return advanced


def dequantize_state(
    state: PackedMSE8State,
    transforms: MLXPackedMSE8Transforms | None = None,
) -> tuple[mx.array, mx.array]:
    """Materialize BF16-shaped history for tests only, never runtime attention."""
    validate_state(state)
    if transforms is None:
        transforms = production_transforms()
    history = packed_history(state)
    keys = dequantize_mse8(
        MLXPackedMSE8(state.packed_keys[:, :history], state.key_norms[:, :history]),
        transforms.key,
    )
    values = dequantize_mse8(
        MLXPackedMSE8(state.packed_values[:, :history], state.value_norms[:, :history]),
        transforms.value,
    )
    return (
        mx.concatenate(
            (state.exact_head_keys.astype(mx.float32), keys, state.exact_keys.astype(mx.float32)),
            axis=1,
        ),
        mx.concatenate(
            (
                state.exact_head_values.astype(mx.float32),
                values,
                state.exact_values.astype(mx.float32),
            ),
            axis=1,
        ),
    )


def packed_scores(
    queries: mx.array,
    state: PackedMSE8State,
    transforms: MLXPackedMSE8Transforms | None = None,
) -> mx.array:
    """Score the exact head, packed middle, and exact tail in token order."""
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
    query = queries.astype(mx.float32)
    head = state.exact_head_keys.shape[1]
    if head:
        repeated_head = mx.repeat(state.exact_head_keys.astype(mx.float32), GQA_GROUPS, axis=0)
        head_scores = mx.sum(query[:, None, :] * repeated_head, axis=-1)
    else:
        head_scores = mx.zeros((NUM_Q_HEADS, 0), dtype=mx.float32)
    tail = state.exact_keys.shape[1]
    if tail:
        repeated_tail = mx.repeat(state.exact_keys.astype(mx.float32), GQA_GROUPS, axis=0)
        tail_scores = mx.sum(query[:, None, :] * repeated_tail, axis=-1)
    else:
        tail_scores = mx.zeros((NUM_Q_HEADS, 0), dtype=mx.float32)
    return mx.concatenate((head_scores, historical, tail_scores), axis=1)


def packed_attend(
    probabilities: mx.array,
    state: PackedMSE8State,
    transforms: MLXPackedMSE8Transforms | None = None,
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
    head = state.exact_head_values.shape[1]
    history = packed_history(state)
    if history:
        rotated = _packed_value_aggregate_kernel(
            inputs=[
                probabilities[:, head : head + history].astype(mx.float32),
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
    if head:
        repeated_head = mx.repeat(state.exact_head_values.astype(mx.float32), GQA_GROUPS, axis=0)
        attended = attended + mx.sum(
            probabilities[:, :head, None].astype(mx.float32) * repeated_head,
            axis=1,
        )
    tail = state.exact_values.shape[1]
    if tail:
        repeated_values = mx.repeat(state.exact_values.astype(mx.float32), GQA_GROUPS, axis=0)
        attended = attended + mx.sum(
            probabilities[:, head + history :, None].astype(mx.float32) * repeated_values,
            axis=1,
        )
    return attended


def packed_attention(
    queries: mx.array,
    state: PackedMSE8State,
    transforms: MLXPackedMSE8Transforms | None = None,
) -> tuple[mx.array, mx.array]:
    scores = packed_scores(queries, state, transforms) * (HEAD_DIM**-0.5)
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1)
    return packed_attend(probabilities, state, transforms), probabilities


def stored_bytes(state: PackedMSE8State) -> int:
    validate_state(state)
    arrays = (
        state.packed_keys,
        state.key_norms,
        state.packed_values,
        state.value_norms,
        state.exact_head_keys,
        state.exact_head_values,
        state.exact_keys,
        state.exact_values,
    )
    return sum(array.size * array.itemsize for array in arrays)

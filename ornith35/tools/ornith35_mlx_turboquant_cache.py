#!/usr/bin/env python3
"""Direct mixed K8/K9-MSE K/V primitives for the Ornith-35 Metal runtime."""

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
SUPPORTED_PACKED_BITS = (8, 9)
PACKED_BITS = 9
PACKED_DIM = (HEAD_DIM * PACKED_BITS + 7) // 8
PRODUCTION_NORM_DTYPE = mx.float32
PRODUCTION_BF16_NORM_LAYERS = frozenset()
PRODUCTION_EXACT_ATTENTION_LAYERS = frozenset((7,))
PRODUCTION_K8_ATTENTION_LAYERS = frozenset((3, 11, 15, 19, 27, 31, 35, 39))
PRODUCTION_EXACT_HEAD_TOKENS = 256
PRODUCTION_EXACT_TAIL_TOKENS = 256
KEY_ROTATION_SEED = 202_607_180_101
VALUE_ROTATION_SEED = 202_607_180_103


K8_PACKED_SCORE_KERNEL_SOURCE = r"""
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


_packed_k8_score_kernel = mx.fast.metal_kernel(
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
    source=K8_PACKED_SCORE_KERNEL_SOURCE,
)


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
uint packed_base = (kv_head * capacity + key_index) * 288u;
for (uint dimension = lane; dimension < 256u; dimension += 32u) {
    uint bit = dimension * 9u;
    uint byte = bit >> 3u;
    uint shift = bit & 7u;
    uint word = uint(packed_keys[packed_base + byte]);
    word |= uint(packed_keys[packed_base + byte + 1u]) << 8u;
    float decoded = centroids[(word >> shift) & 511u];
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
    name="ornith35_turboquant_k9_packed_scores_f32",
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


K8_PACKED_VALUE_AGGREGATE_KERNEL_SOURCE = r"""
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


_packed_k8_value_aggregate_kernel = mx.fast.metal_kernel(
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
    source=K8_PACKED_VALUE_AGGREGATE_KERNEL_SOURCE,
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
uint first_dimension = block * 4u;
float totals[32] = {0.0f};
threadgroup float partials[8 * 32];
for (uint token = lid; token < history; token += 256u) {
    float norm = float(norms[kv_head * capacity + token]);
    uint packed_base = (kv_head * capacity + token) * 288u;
    float decoded[4];
    for (uint item = 0u; item < 4u; ++item) {
        uint bit = (first_dimension + item) * 9u;
        uint byte = bit >> 3u;
        uint shift = bit & 7u;
        uint word = uint(packed_values[packed_base + byte]);
        word |= uint(packed_values[packed_base + byte + 1u]) << 8u;
        decoded[item] = centroids[(word >> shift) & 511u];
    }
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
    name="ornith35_turboquant_k9_packed_value_aggregate_f32",
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


K8_PACKED_CLASSIFY_KERNEL_SOURCE = r"""
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


_packed_k8_classify_kernel = mx.fast.metal_kernel(
    name="ornith35_turboquant_k8_classify_f32",
    input_names=["rotated", "boundaries", "coordinate_count"],
    output_names=["packed"],
    source=K8_PACKED_CLASSIFY_KERNEL_SOURCE,
)


PACKED_CLASSIFY_KERNEL_SOURCE = r"""
uint coordinate = thread_position_in_grid.x;
if (coordinate >= coordinate_count) return;
float rotated_value = rotated[coordinate];
{
    uint low = 0u;
    uint high = 511u;
    while (low < high) {
        uint middle = (low + high) >> 1u;
        if (rotated_value >= boundaries[middle]) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    indices[coordinate] = ushort(low);
}
"""


_packed_classify_kernel = mx.fast.metal_kernel(
    name="ornith35_turboquant_k9_classify_f32",
    input_names=["rotated", "boundaries", "coordinate_count"],
    output_names=["indices"],
    source=PACKED_CLASSIFY_KERNEL_SOURCE,
)


PACK_INDICES_KERNEL_SOURCE = r"""
uint output_index = thread_position_in_grid.x;
if (output_index >= packed_count) return;
uint vector = output_index / 288u;
uint byte = output_index - vector * 288u;
uint bit = byte * 8u;
uint coordinate = bit / 9u;
uint shift = bit - coordinate * 9u;
uint base = vector * 256u;
uint word = uint(indices[base + coordinate]) >> shift;
uint available = 9u - shift;
if (available < 8u && coordinate + 1u < 256u) {
    word |= uint(indices[base + coordinate + 1u]) << available;
}
packed[output_index] = uchar(word & 255u);
"""


_pack_indices_kernel = mx.fast.metal_kernel(
    name="ornith35_turboquant_k9_pack_indices_u16",
    input_names=["indices", "packed_count"],
    output_names=["packed"],
    source=PACK_INDICES_KERNEL_SOURCE,
)


@dataclass(frozen=True)
class MLXPackedMSE:
    packed: mx.array
    norms: mx.array
    bits: int = PACKED_BITS


@dataclass(frozen=True)
class MLXPackedMSETransforms:
    key: turboquant.MLXTransform
    value: turboquant.MLXTransform


@dataclass(frozen=True)
class MLXPackedMSEState:
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
    bits: int = PACKED_BITS


@dataclass(frozen=True)
class MLXLinearPackedMSEState:
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
    bits: int = PACKED_BITS


PackedMSEState = MLXPackedMSEState | MLXLinearPackedMSEState


def require(condition: bool, message: str) -> None:
    if not condition:
        raise reference.TurboQuantError(message)


def production_norm_dtype(layer_index: int) -> mx.Dtype:
    require(type(layer_index) is int and layer_index >= 0, "invalid model layer index")
    return (
        mx.bfloat16
        if layer_index in PRODUCTION_BF16_NORM_LAYERS
        else PRODUCTION_NORM_DTYPE
    )


def production_packed_bits(layer_index: int) -> int:
    require(type(layer_index) is int and layer_index >= 0, "invalid model layer index")
    return 8 if layer_index in PRODUCTION_K8_ATTENTION_LAYERS else PACKED_BITS


def packed_dimension(bits: int) -> int:
    require(type(bits) is int and bits in SUPPORTED_PACKED_BITS, "unsupported packed bit width")
    return (HEAD_DIM * bits + 7) // 8


@lru_cache(maxsize=1)
def production_transforms() -> MLXPackedMSETransforms:
    return MLXPackedMSETransforms(
        key=turboquant.haar_rotation(HEAD_DIM, KEY_ROTATION_SEED),
        value=turboquant.haar_rotation(HEAD_DIM, VALUE_ROTATION_SEED),
    )


def _centroids(bits: int = PACKED_BITS) -> mx.array:
    packed_dimension(bits)
    return mx.array(reference.codebook(HEAD_DIM, bits).centroids, dtype=mx.float32)


def _boundaries(bits: int = PACKED_BITS) -> mx.array:
    packed_dimension(bits)
    values = reference.codebook(HEAD_DIM, bits).boundaries[1:-1]
    return mx.array(values, dtype=mx.float32)


def _pack_indices(indices: mx.array, bits: int = PACKED_BITS) -> mx.array:
    dimension = packed_dimension(bits)
    coordinates = mx.array(
        tuple((byte * 8) // bits for byte in range(dimension)),
        dtype=mx.uint32,
    )
    shifts = mx.array(
        tuple((byte * 8) % bits for byte in range(dimension)),
        dtype=mx.uint32,
    )
    padding = mx.zeros((*indices.shape[:-1], 2), dtype=indices.dtype)
    padded = mx.concatenate((indices, padding), axis=-1).astype(mx.uint32)
    word = mx.take(padded, coordinates, axis=-1)
    word = word | (mx.take(padded, coordinates + 1, axis=-1) << bits)
    word = word | (mx.take(padded, coordinates + 2, axis=-1) << (2 * bits))
    return ((word >> shifts) & 255).astype(mx.uint8)


def encode_mse_graph(
    vectors: mx.array,
    rotation: turboquant.MLXTransform,
    *,
    norm_dtype: mx.Dtype = PRODUCTION_NORM_DTYPE,
    bits: int = PACKED_BITS,
) -> MLXPackedMSE:
    """Encode and pack entirely in the MLX graph, including zero vectors."""
    packed_dimension(bits)
    require(vectors.ndim >= 2 and vectors.shape[-1] == HEAD_DIM, "packed input geometry mismatch")
    require(
        rotation.matrix.shape == (HEAD_DIM, HEAD_DIM)
        and rotation.matrix.dtype == mx.float32,
        "packed rotation mismatch",
    )
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported packed norm dtype")
    source = vectors.astype(mx.float32)
    norms = mx.sqrt(mx.sum(source * source, axis=-1, keepdims=True))
    safe_norms = mx.where(norms > 0.0, norms, 1.0)
    rotated = (source / safe_norms) @ mx.swapaxes(rotation.matrix, -2, -1)
    indices = turboquant.quantize_indices(rotated, _boundaries(bits), bits)
    packed = _pack_indices(indices, bits)
    return MLXPackedMSE(packed=packed, norms=norms.astype(norm_dtype), bits=bits)


def encode_mse(
    vectors: mx.array,
    rotation: turboquant.MLXTransform,
    *,
    norm_dtype: mx.Dtype = PRODUCTION_NORM_DTYPE,
    bits: int = PACKED_BITS,
) -> MLXPackedMSE:
    """Use authoritative MLX rotation followed by Metal classification and packing."""
    dimension = packed_dimension(bits)
    require(vectors.ndim >= 2 and vectors.shape[-1] == HEAD_DIM, "packed input geometry mismatch")
    require(
        rotation.matrix.shape == (HEAD_DIM, HEAD_DIM)
        and rotation.matrix.dtype == mx.float32,
        "packed rotation mismatch",
    )
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported packed norm dtype")
    if vectors.size == 0:
        return encode_mse_graph(vectors, rotation, norm_dtype=norm_dtype, bits=bits)
    source = vectors.astype(mx.float32)
    norms = mx.sqrt(mx.sum(source * source, axis=-1, keepdims=True))
    safe_norms = mx.where(norms > 0.0, norms, 1.0)
    rotated = (source / safe_norms) @ mx.swapaxes(rotation.matrix, -2, -1)
    if bits == 8:
        (packed,) = _packed_k8_classify_kernel(
            inputs=[
                rotated,
                _boundaries(bits),
                mx.array(vectors.size, dtype=mx.uint32),
            ],
            grid=(vectors.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[vectors.shape],
            output_dtypes=[mx.uint8],
        )
    else:
        (indices,) = _packed_classify_kernel(
            inputs=[
                rotated,
                _boundaries(bits),
                mx.array(vectors.size, dtype=mx.uint32),
            ],
            grid=(vectors.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[vectors.shape],
            output_dtypes=[mx.uint16],
        )
        vector_count = vectors.size // HEAD_DIM
        packed_count = vector_count * dimension
        (packed,) = _pack_indices_kernel(
            inputs=[indices, mx.array(packed_count, dtype=mx.uint32)],
            grid=(packed_count, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(*vectors.shape[:-1], dimension)],
            output_dtypes=[mx.uint8],
        )
    return MLXPackedMSE(packed=packed, norms=norms.astype(norm_dtype), bits=bits)


def unpack_indices(encoding: MLXPackedMSE) -> mx.array:
    validate_encoding(encoding)
    if encoding.bits == 8:
        return encoding.packed
    bit_offsets = mx.arange(HEAD_DIM, dtype=mx.uint32) * encoding.bits
    bytes_ = bit_offsets // 8
    shifts = bit_offsets % 8
    padding = mx.zeros((*encoding.packed.shape[:-1], 2), dtype=mx.uint8)
    padded = mx.concatenate((encoding.packed, padding), axis=-1).astype(mx.uint32)
    word = mx.take(padded, bytes_, axis=-1)
    word = word | (mx.take(padded, bytes_ + 1, axis=-1) << 8)
    return ((word >> shifts) & ((1 << encoding.bits) - 1)).astype(mx.uint16)


def dequantize_mse(
    encoding: MLXPackedMSE,
    rotation: turboquant.MLXTransform,
) -> mx.array:
    indices = unpack_indices(encoding)
    return turboquant.dequantize_mse(
        turboquant.MLXMSEEncoding(
            indices=indices,
            norms=encoding.norms,
            bits=encoding.bits,
            dimension=HEAD_DIM,
        ),
        rotation,
    )


def validate_encoding(encoding: MLXPackedMSE) -> None:
    require(isinstance(encoding, MLXPackedMSE), "invalid packed encoding")
    dimension = packed_dimension(encoding.bits)
    require(
        encoding.packed.ndim >= 2
        and encoding.packed.shape[-1] == dimension
        and encoding.packed.dtype == mx.uint8,
        "packed payload mismatch",
    )
    require(
        encoding.norms.shape == (*encoding.packed.shape[:-1], 1)
        and encoding.norms.dtype in (mx.bfloat16, mx.float32),
        "packed norm mismatch",
    )


def validate_state(state: PackedMSEState) -> None:
    require(
        isinstance(state, (MLXPackedMSEState, MLXLinearPackedMSEState)),
        "invalid packed K/V state",
    )
    dimension = packed_dimension(state.bits)
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
    expected_packed = (NUM_KV_HEADS, physical_capacity, dimension)
    expected_norms = (NUM_KV_HEADS, physical_capacity, 1)
    expected_head = (NUM_KV_HEADS, head, HEAD_DIM)
    expected_exact = (NUM_KV_HEADS, tail, HEAD_DIM)
    require(
        state.packed_keys.shape == expected_packed
        and state.packed_values.shape == expected_packed
        and state.packed_keys.dtype == state.packed_values.dtype == mx.uint8,
        "packed K/V payload mismatch",
    )
    norm_dtype = state.key_norms.dtype
    require(
        state.key_norms.shape == expected_norms
        and state.value_norms.shape == expected_norms
        and norm_dtype == state.value_norms.dtype
        and norm_dtype in (mx.bfloat16, mx.float32),
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
    if isinstance(state, MLXLinearPackedMSEState):
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


def state_length(state: PackedMSEState) -> int:
    validate_state(state)
    if isinstance(state, MLXLinearPackedMSEState):
        return state.position
    return (
        state.exact_head_keys.shape[1]
        + state.packed_keys.shape[1]
        + state.exact_keys.shape[1]
    )


def packed_history(state: PackedMSEState) -> int:
    validate_state(state)
    if isinstance(state, MLXLinearPackedMSEState):
        return (
            state.position
            - state.exact_head_keys.shape[1]
            - state.exact_keys.shape[1]
        )
    return state.packed_keys.shape[1]


def packed_capacity(state: PackedMSEState) -> int:
    validate_state(state)
    return state.packed_keys.shape[1]


def _materialize_boundary(values: mx.array, start: int, stop: int) -> mx.array:
    """Give a retained exact boundary storage independent from a larger source."""
    return mx.contiguous(values[:, start:stop])


def compress_bf16_kv(
    keys: mx.array,
    values: mx.array,
    transforms: MLXPackedMSETransforms | None = None,
    *,
    exact_head: int = PRODUCTION_EXACT_HEAD_TOKENS,
    exact_tail: int = PRODUCTION_EXACT_TAIL_TOKENS,
    norm_dtype: mx.Dtype = PRODUCTION_NORM_DTYPE,
    bits: int = PACKED_BITS,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXPackedMSEState:
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
    packed_dimension(bits)
    require(norm_dtype in (mx.bfloat16, mx.float32), "unsupported packed norm dtype")
    context.validate_range(context_profile, 0, keys.shape[1])
    if transforms is None:
        transforms = production_transforms()
    head = min(keys.shape[1], exact_head)
    remaining = keys.shape[1] - head
    tail = min(remaining, exact_tail)
    history = remaining - tail
    packed_end = head + history
    encoded_keys = encode_mse(
        keys[:, head:packed_end],
        transforms.key,
        norm_dtype=norm_dtype,
        bits=bits,
    )
    encoded_values = encode_mse(
        values[:, head:packed_end],
        transforms.value,
        norm_dtype=norm_dtype,
        bits=bits,
    )
    state = MLXPackedMSEState(
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
        bits=bits,
    )
    validate_state(state)
    return state


def linearize_state(
    state: MLXPackedMSEState,
    capacity: int,
) -> MLXLinearPackedMSEState:
    """Copy an immutable packed prefix into single-owner fixed-capacity buffers."""
    require(isinstance(state, MLXPackedMSEState), "linear source state must be immutable")
    validate_state(state)
    position = state_length(state)
    require(isinstance(capacity, int) and capacity >= position, "linear capacity is too short")
    context.validate_range(state.context_profile, 0, capacity)
    packed_shape = (NUM_KV_HEADS, capacity, packed_dimension(state.bits))
    norm_shape = (NUM_KV_HEADS, capacity, 1)
    packed_keys = mx.zeros(packed_shape, dtype=mx.uint8)
    norm_dtype = state.key_norms.dtype
    key_norms = mx.zeros(norm_shape, dtype=norm_dtype)
    packed_values = mx.zeros(packed_shape, dtype=mx.uint8)
    value_norms = mx.zeros(norm_shape, dtype=norm_dtype)
    history = state.packed_keys.shape[1]
    if history:
        packed_keys, key_norms, packed_values, value_norms = (
            linear_cache.append_packed_mse(
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
    linear = MLXLinearPackedMSEState(
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
        bits=state.bits,
    )
    validate_state(linear)
    return linear


def linearize_bf16_kv(
    keys: mx.array,
    values: mx.array,
    capacity: int,
    transforms: MLXPackedMSETransforms | None = None,
    *,
    exact_head: int = PRODUCTION_EXACT_HEAD_TOKENS,
    exact_tail: int = PRODUCTION_EXACT_TAIL_TOKENS,
    norm_dtype: mx.Dtype = PRODUCTION_NORM_DTYPE,
    bits: int = PACKED_BITS,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXLinearPackedMSEState:
    """Encode a BF16 prefix with a bounded exact tail into fixed-capacity storage."""
    immutable = compress_bf16_kv(
        keys,
        values,
        transforms,
        exact_head=exact_head,
        exact_tail=exact_tail,
        norm_dtype=norm_dtype,
        bits=bits,
        context_profile=context_profile,
    )
    return linearize_state(immutable, capacity)


def advance_linear_state(
    state: MLXLinearPackedMSEState,
    key_update: mx.array,
    value_update: mx.array,
    transforms: MLXPackedMSETransforms | None = None,
) -> MLXLinearPackedMSEState:
    """Append BF16 K/V while moving the prior tail into packed storage."""
    require(isinstance(state, MLXLinearPackedMSEState), "packed append state must be linear")
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
        norm_dtype = state.key_norms.dtype
        encoded_keys = encode_mse(
            keys_to_pack,
            transforms.key,
            norm_dtype=norm_dtype,
            bits=state.bits,
        )
        encoded_values = encode_mse(
            values_to_pack,
            transforms.value,
            norm_dtype=norm_dtype,
            bits=state.bits,
        )
        write_position = packed_history(state)
        packed_keys, key_norms, packed_values, value_norms = (
            linear_cache.append_packed_mse(
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
    advanced = MLXLinearPackedMSEState(
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
        bits=state.bits,
    )
    validate_state(advanced)
    return advanced


def dequantize_state(
    state: PackedMSEState,
    transforms: MLXPackedMSETransforms | None = None,
) -> tuple[mx.array, mx.array]:
    """Materialize BF16-shaped history for tests only, never runtime attention."""
    validate_state(state)
    if transforms is None:
        transforms = production_transforms()
    history = packed_history(state)
    keys = dequantize_mse(
        MLXPackedMSE(
            state.packed_keys[:, :history],
            state.key_norms[:, :history],
            state.bits,
        ),
        transforms.key,
    )
    values = dequantize_mse(
        MLXPackedMSE(
            state.packed_values[:, :history],
            state.value_norms[:, :history],
            state.bits,
        ),
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
    state: PackedMSEState,
    transforms: MLXPackedMSETransforms | None = None,
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
        score_kernel = _packed_k8_score_kernel if state.bits == 8 else _packed_score_kernel
        historical = score_kernel(
            inputs=[
                rotated_queries,
                state.packed_keys,
                state.key_norms,
                _centroids(state.bits),
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
    state: PackedMSEState,
    transforms: MLXPackedMSETransforms | None = None,
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
        value_kernel = (
            _packed_k8_value_aggregate_kernel
            if state.bits == 8
            else _packed_value_aggregate_kernel
        )
        rotated = value_kernel(
            inputs=[
                probabilities[:, head : head + history].astype(mx.float32),
                state.packed_values,
                state.value_norms,
                _centroids(state.bits),
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
    state: PackedMSEState,
    transforms: MLXPackedMSETransforms | None = None,
) -> tuple[mx.array, mx.array]:
    scores = packed_scores(queries, state, transforms) * (HEAD_DIM**-0.5)
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1)
    return packed_attend(probabilities, state, transforms), probabilities


def stored_bytes(state: PackedMSEState) -> int:
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

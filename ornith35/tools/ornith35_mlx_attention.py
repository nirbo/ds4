#!/usr/bin/env python3
"""One-token MLX full-attention composition for text-only Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_dense as dense
import ornith35_mlx_linear_cache as linear_cache
import ornith35_mlx_turboquant_cache as turboquant_cache
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
KEY_TILED_PREFILL_MIN_PREFIX = 4_096
EXACT_LONG_PREFILL_MIN_PREFIX = 106_496
EXACT_FUSED_SOFTMAX_VALUE_MAX_PREFIX = 131_072
EXACT_FUSED_SOFTMAX_VALUE_MIN_TOKENS = 64


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


QK_NORM_ROPE_CHUNK_KERNEL_SOURCE = r"""
uint token_head = threadgroup_position_in_grid.x;
uint token = token_head / 18u;
uint head = token_head - token * 18u;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
threadgroup float inverse_mean[1];
threadgroup bfloat16_t normalized[256];
bool is_query = head < 16u;
uint local_head = is_query ? head : head - 16u;
uint token_query_base = token * 8192u;
uint token_key_base = token * 512u;
uint input_base = is_query
    ? token_query_base + local_head * 512u
    : token_key_base + local_head * 256u;
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
            uint gate_base = (token * 16u + local_head) * 256u;
            output_gate[gate_base + index] = query_gate[input_base + 256u + index];
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
            uint rope_index = token * 64u + index;
            bfloat16_t first = normalized[index] * cosine[rope_index];
            bfloat16_t second = rotated * sine[rope_index];
            output_value = first + second;
        }
        if (is_query) {
            uint output_base = (token * 16u + local_head) * 256u;
            output_query[output_base + index] = output_value;
        } else {
            uint output_base = (token * 2u + local_head) * 256u;
            output_key[output_base + index] = output_value;
        }
    }
}
"""


_qk_norm_rope_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_qk_norm_rope_chunk_bf16",
    input_names=["query_gate", "key", "q_norm", "k_norm", "cosine", "sine"],
    output_names=["output_query", "output_gate", "output_key"],
    source=QK_NORM_ROPE_CHUNK_KERNEL_SOURCE,
)


KEY_NORM_ROPE_CHUNK_KERNEL_SOURCE = r"""
uint token_head = threadgroup_position_in_grid.x;
uint token = token_head >> 1;
uint head = token_head - (token << 1);
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint input_base = (token * 2u + head) * 256u;
threadgroup float inverse_mean[1];
threadgroup bfloat16_t normalized[256];
float total = 0.0f;
for (uint block = 0u; block < 2u; ++block) {
    uint local_base = lid * 4u + block * 128u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint index = local_base + offset;
        float value = float(key[input_base + index]);
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
        float value = float(key[input_base + index]);
        float weight = float(k_norm[index]);
        volatile float scaled = value * inverse_mean[0];
        volatile float centered = 1.0f + weight;
        volatile float weighted = scaled * centered;
        normalized[index] = bfloat16_t(weighted);
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
            uint rope_index = token * 64u + index;
            bfloat16_t first = normalized[index] * cosine[rope_index];
            bfloat16_t second = rotated * sine[rope_index];
            output_value = first + second;
        }
        output_key[input_base + index] = output_value;
    }
}
"""


_key_norm_rope_chunk_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_key_norm_rope_chunk_bf16",
    input_names=["key", "k_norm", "cosine", "sine"],
    output_names=["output_key"],
    source=KEY_NORM_ROPE_CHUNK_KERNEL_SOURCE,
)


# Reproduce MLX 0.32.0's normal BF16 GEMV score reduction for K=256 while
# sharing each key load across four causal queries.
EXACT_BATCHED_SCORE_KERNEL_SOURCE = r"""
uint key_block = threadgroup_position_in_grid.x;
uint head = threadgroup_position_in_grid.y;
uint query_block = threadgroup_position_in_grid.z;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint start = start_position;
uint queries_count = query_count;
uint keys_count = key_length;
uint key_index = key_block * 8u + simd_group;
uint kv_head = head / 8u;
float totals[4] = {0.0f};
if (key_index < keys_count) {
    for (uint block = 0u; block < 2u; ++block) {
        uint dimension_base = lane * 4u + block * 128u;
        for (uint offset = 0u; offset < 4u; ++offset) {
            uint dimension = dimension_base + offset;
            float key_value = float(
                keys[(kv_head * keys_count + key_index) * 256u + dimension]
            );
            for (uint local_query = 0u; local_query < 4u; ++local_query) {
                uint query_index = query_block * 4u + local_query;
                if (query_index < queries_count) {
                    float query_value = float(
                        queries[(query_index * 16u + head) * 256u + dimension]
                    );
                    totals[local_query] += key_value * query_value;
                }
            }
        }
    }
    for (uint local_query = 0u; local_query < 4u; ++local_query) {
        for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {
            totals[local_query] += simd_shuffle_down(totals[local_query], offset);
        }
        uint query_index = query_block * 4u + local_query;
        if (lane == 0u && query_index < queries_count) {
            uint valid_length = start + query_index + 1u;
            bfloat16_t value = key_index < valid_length
                ? bfloat16_t(totals[local_query])
                : bfloat16_t(-INFINITY);
            scores[(query_index * 16u + head) * keys_count + key_index] = value;
        }
    }
}
"""


_exact_batched_score_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_exact_batched_scores_bf16_256",
    input_names=["queries", "keys", "start_position", "query_count", "key_length"],
    output_names=["scores"],
    source=EXACT_BATCHED_SCORE_KERNEL_SOURCE,
)


# Preserve the same per-score accumulation and SIMD reduction while amortizing
# each query load over eight adjacent key positions.
EXACT_KEY_TILED_SCORE_KERNEL_SOURCE = r"""
uint key_block = threadgroup_position_in_grid.x;
uint head = threadgroup_position_in_grid.y;
uint query_block = threadgroup_position_in_grid.z;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint start = start_position;
uint queries_count = query_count;
uint keys_count = key_length;
constexpr uint simd_groups = 8u;
constexpr uint key_tile = 8u;
uint key_base = (key_block * simd_groups + simd_group) * key_tile;
uint kv_head = head / 8u;
float totals[key_tile][4];
for (uint local_key = 0u; local_key < key_tile; ++local_key) {
    for (uint local_query = 0u; local_query < 4u; ++local_query) {
        totals[local_key][local_query] = 0.0f;
    }
}
for (uint block = 0u; block < 2u; ++block) {
    uint dimension_base = lane * 4u + block * 128u;
    for (uint offset = 0u; offset < 4u; ++offset) {
        uint dimension = dimension_base + offset;
        float query_values[4];
        for (uint local_query = 0u; local_query < 4u; ++local_query) {
            uint query_index = query_block * 4u + local_query;
            query_values[local_query] = query_index < queries_count
                ? float(queries[(query_index * 16u + head) * 256u + dimension])
                : 0.0f;
        }
        for (uint local_key = 0u; local_key < key_tile; ++local_key) {
            uint key_index = key_base + local_key;
            float key_value = key_index < keys_count
                ? float(keys[(kv_head * keys_count + key_index) * 256u + dimension])
                : 0.0f;
            for (uint local_query = 0u; local_query < 4u; ++local_query) {
                totals[local_key][local_query] += (
                    key_value * query_values[local_query]
                );
            }
        }
    }
}
for (uint local_key = 0u; local_key < key_tile; ++local_key) {
    uint key_index = key_base + local_key;
    if (key_index >= keys_count) continue;
    for (uint local_query = 0u; local_query < 4u; ++local_query) {
        for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {
            totals[local_key][local_query] += simd_shuffle_down(
                totals[local_key][local_query], offset
            );
        }
        uint query_index = query_block * 4u + local_query;
        if (lane == 0u && query_index < queries_count) {
            uint valid_length = start + query_index + 1u;
            bfloat16_t value = key_index < valid_length
                ? bfloat16_t(totals[local_key][local_query])
                : bfloat16_t(-INFINITY);
            scores[(query_index * 16u + head) * keys_count + key_index] = value;
        }
    }
}
"""


_exact_key_tiled_score_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_exact_key_tiled_scores_bf16_256",
    input_names=["queries", "keys", "start_position", "query_count", "key_length"],
    output_names=["scores"],
    source=EXACT_KEY_TILED_SCORE_KERNEL_SOURCE,
)


def _exact_batched_scores(
    queries: mx.array,
    keys: mx.array,
    start_position: mx.array,
    query_count: mx.array,
    key_length: mx.array,
    *,
    queries_count: int,
    keys_count: int,
    key_tiled: bool,
) -> mx.array:
    kernel = _exact_key_tiled_score_kernel if key_tiled else _exact_batched_score_kernel
    keys_per_group = 64 if key_tiled else 8
    return kernel(
        inputs=[queries, keys, start_position, query_count, key_length],
        grid=(
            ((keys_count + keys_per_group - 1) // keys_per_group) * 256,
            16,
            (queries_count + 3) // 4,
        ),
        threadgroup=(256, 1, 1),
        output_shapes=[(queries_count, 16, keys_count)],
        output_dtypes=[mx.bfloat16],
    )[0]


EXACT_LOOPED_SOFTMAX_KERNEL_SOURCE = r"""
uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
uint query_index = row / 16u;
uint valid_length = start_position + query_index + 1u;
uint stride = key_length;
uint rounds = (valid_length + 4095u) / 4096u;
threadgroup float local_max[32];
threadgroup float local_normalizer[32];
float previous_max;
float maximum = Limits<float>::finite_min;
float normalizer = 0.0f;
uint row_base = row * stride;
for (uint round = 0u; round < rounds; ++round) {
    uint offset = round * 4096u + lid * 4u;
    float values[4];
    for (uint item = 0u; item < 4u; ++item) {
        uint index = offset + item;
        values[item] = index < valid_length
            ? float(scaled_scores[row_base + index])
            : Limits<float>::min;
    }
    previous_max = maximum;
    for (uint item = 0u; item < 4u; ++item) {
        maximum = maximum < values[item] ? values[item] : maximum;
    }
    normalizer *= metal::fast::exp(previous_max - maximum);
    for (uint item = 0u; item < 4u; ++item) {
        normalizer += metal::fast::exp(values[item] - maximum);
    }
}
previous_max = maximum;
maximum = simd_max(maximum);
normalizer *= metal::fast::exp(previous_max - maximum);
normalizer = simd_sum(normalizer);
previous_max = maximum;
if (lane == 0u) local_max[group] = maximum;
threadgroup_barrier(mem_flags::mem_threadgroup);
maximum = simd_max(local_max[lane]);
normalizer *= metal::fast::exp(previous_max - maximum);
if (lane == 0u) local_normalizer[group] = normalizer;
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer = simd_sum(local_normalizer[lane]);
normalizer = 1.0f / normalizer;
for (uint round = 0u; round < rounds; ++round) {
    uint offset = round * 4096u + lid * 4u;
    for (uint item = 0u; item < 4u; ++item) {
        uint index = offset + item;
        if (index < valid_length) {
            float probability = metal::fast::exp(
                float(scaled_scores[row_base + index]) - maximum
            ) * normalizer;
            probabilities[row_base + index] = bfloat16_t(probability);
        }
    }
}
"""


_exact_looped_softmax_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_exact_looped_softmax_bf16",
    input_names=["scaled_scores", "start_position", "key_length"],
    output_names=["probabilities"],
    source=EXACT_LOOPED_SOFTMAX_KERNEL_SOURCE,
)


EXACT_BATCHED_VALUE_KERNEL_SOURCE = r"""
uint output_block = threadgroup_position_in_grid.x;
uint row = threadgroup_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint query_index = row / 16u;
uint head = row % 16u;
uint kv_head = head / 8u;
uint valid_length = start_position + query_index + 1u;
uint stride = key_length;
uint lane_row = lane / 4u;
uint lane_column = lane % 4u;
uint output_column = output_block * 32u + simd_group * 16u + lane_column * 4u;
float totals[4] = {0.0f};
uint row_base = row * stride;
uint complete_blocks = valid_length / 32u;
for (uint block = 0u; block < complete_blocks; ++block) {
    threadgroup_barrier(mem_flags::mem_none);
    uint input_index = block * 32u + lane_row * 4u;
    float coefficients[4];
    for (uint item = 0u; item < 4u; ++item) {
        coefficients[item] = float(probabilities[row_base + input_index + item]);
    }
    for (uint item = 0u; item < 4u; ++item) {
        uint value_base = (
            (kv_head * stride + input_index + item) * 256u + output_column
        );
        for (uint column = 0u; column < 4u; ++column) {
            totals[column] += coefficients[item] * float(values[value_base + column]);
        }
    }
}
uint input_index = complete_blocks * 32u + lane_row * 4u;
if (input_index < valid_length) {
    for (uint item = 0u; item < 4u && input_index + item < valid_length; ++item) {
        float coefficient = float(probabilities[row_base + input_index + item]);
        uint value_base = (
            (kv_head * stride + input_index + item) * 256u + output_column
        );
        for (uint column = 0u; column < 4u; ++column) {
            totals[column] += coefficient * float(values[value_base + column]);
        }
    }
}
for (uint column = 0u; column < 4u; ++column) {
    for (ushort offset = 16u; offset >= 4u; offset >>= 1u) {
        totals[column] += simd_shuffle_down(totals[column], offset);
    }
}
if (lane_row == 0u) {
    uint output_base = row * 256u + output_column;
    for (uint column = 0u; column < 4u; ++column) {
        attended[output_base + column] = bfloat16_t(totals[column]);
    }
}
"""


_exact_batched_value_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_exact_batched_values_bf16",
    input_names=["probabilities", "values", "start_position", "key_length"],
    output_names=["attended"],
    source=EXACT_BATCHED_VALUE_KERNEL_SOURCE,
)


EXACT_FUSED_SOFTMAX_VALUE_KERNEL_SOURCE = r"""
uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
uint query_index = row / 16u;
uint head = row % 16u;
uint kv_head = head / 8u;
uint valid_length = start_position + query_index + 1u;
uint stride = key_length;
constexpr uint probability_tile = 15360u;
uint softmax_rounds = (valid_length + 4095u) / 4096u;
uint probability_rounds = (
    valid_length + probability_tile - 1u
) / probability_tile;
uint row_base = row * stride;
threadgroup float local_max[32];
threadgroup float local_normalizer[32];
threadgroup bfloat16_t local_probability[probability_tile];
float previous_max;
float maximum = Limits<float>::finite_min;
float normalizer = 0.0f;
for (uint round = 0u; round < softmax_rounds; ++round) {
    uint offset = round * 4096u + lid * 4u;
    float score_values[4];
    for (uint item = 0u; item < 4u; ++item) {
        uint index = offset + item;
        score_values[item] = index < valid_length
            ? float(scaled_scores[row_base + index])
            : Limits<float>::min;
    }
    previous_max = maximum;
    for (uint item = 0u; item < 4u; ++item) {
        maximum = maximum < score_values[item] ? score_values[item] : maximum;
    }
    normalizer *= metal::fast::exp(previous_max - maximum);
    for (uint item = 0u; item < 4u; ++item) {
        normalizer += metal::fast::exp(score_values[item] - maximum);
    }
}
previous_max = maximum;
maximum = simd_max(maximum);
normalizer *= metal::fast::exp(previous_max - maximum);
normalizer = simd_sum(normalizer);
previous_max = maximum;
if (lane == 0u) local_max[group] = maximum;
threadgroup_barrier(mem_flags::mem_threadgroup);
maximum = simd_max(local_max[lane]);
normalizer *= metal::fast::exp(previous_max - maximum);
if (lane == 0u) local_normalizer[group] = normalizer;
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer = 1.0f / simd_sum(local_normalizer[lane]);

uint lane_row = lane / 4u;
uint lane_column = lane % 4u;
uint output_block = group / 2u;
uint local_value_group = group % 2u;
uint output_column = output_block * 32u
    + local_value_group * 16u + lane_column * 4u;
bool value_lane = group < 16u;
float totals[4] = {0.0f};
for (uint round = 0u; round < probability_rounds; ++round) {
    uint round_base = round * probability_tile;
    for (
        uint local_index = lid;
        local_index < probability_tile;
        local_index += 1024u
    ) {
        uint index = round_base + local_index;
        if (index < valid_length) {
            float probability = metal::fast::exp(
                float(scaled_scores[row_base + index]) - maximum
            ) * normalizer;
            local_probability[local_index] = bfloat16_t(probability);
        } else {
            local_probability[local_index] = bfloat16_t(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (value_lane) {
        uint remaining = min(probability_tile, valid_length - round_base);
        uint complete_blocks = remaining / 32u;
        for (uint block = 0u; block < complete_blocks; ++block) {
            uint local_input = block * 32u + lane_row * 4u;
            uint input_index = round_base + local_input;
            for (uint item = 0u; item < 4u; ++item) {
                float coefficient = float(local_probability[local_input + item]);
                uint value_base = (
                    (kv_head * stride + input_index + item) * 256u + output_column
                );
                for (uint column = 0u; column < 4u; ++column) {
                    totals[column] += coefficient * float(values[value_base + column]);
                }
            }
        }
        uint tail_base = complete_blocks * 32u;
        uint input_index = round_base + tail_base + lane_row * 4u;
        uint local_input = tail_base + lane_row * 4u;
        for (
            uint item = 0u;
            item < 4u && local_input + item < remaining;
            ++item
        ) {
            float coefficient = float(local_probability[local_input + item]);
            uint value_base = (
                (kv_head * stride + input_index + item) * 256u + output_column
            );
            for (uint column = 0u; column < 4u; ++column) {
                totals[column] += coefficient * float(values[value_base + column]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (value_lane) {
    for (uint column = 0u; column < 4u; ++column) {
        for (ushort offset = 16u; offset >= 4u; offset >>= 1u) {
            totals[column] += simd_shuffle_down(totals[column], offset);
        }
    }
    if (lane_row == 0u) {
        uint output_base = row * 256u + output_column;
        for (uint column = 0u; column < 4u; ++column) {
            attended[output_base + column] = bfloat16_t(totals[column]);
        }
    }
}
"""


_exact_fused_softmax_value_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_exact_fused_softmax_value_bf16",
    input_names=["scaled_scores", "values", "start_position", "key_length"],
    output_names=["attended"],
    source=EXACT_FUSED_SOFTMAX_VALUE_KERNEL_SOURCE,
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
    context_profile: str = context.NATIVE_PROFILE_ID


@dataclass(frozen=True)
class MLXLinearAttentionState:
    """Fixed-capacity K/V storage owned by one advancing decode session."""

    keys: mx.array
    values: mx.array
    position: int
    capacity: int
    context_profile: str = context.NATIVE_PROFILE_ID


MLXTurboQuantImmutableAttentionState = turboquant_cache.MLXPackedMSE4State
MLXTurboQuantAttentionState = turboquant_cache.MLXLinearPackedMSE4State
AttentionState = (
    MLXAttentionState
    | MLXLinearAttentionState
    | MLXTurboQuantImmutableAttentionState
    | MLXTurboQuantAttentionState
)


@dataclass(frozen=True)
class MLXTextRoPE:
    position: int
    tokens: int
    cosine: mx.array
    sine: mx.array
    context_profile: str


@dataclass(frozen=True)
class MLXPrefillQKVTrace:
    """Diagnostic Q/K/V after the exact production projection and RoPE path."""

    queries: mx.array
    gates: mx.array
    keys: mx.array
    values: mx.array


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
    state: AttentionState,
    config: AttentionConfig,
) -> int:
    profile = context.resolve_profile(state.context_profile)
    if isinstance(
        state,
        (MLXTurboQuantImmutableAttentionState, MLXTurboQuantAttentionState),
    ):
        require(config == PRODUCTION_CONFIG, "TurboQuant requires production attention geometry")
        return turboquant_cache.state_length(state)
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
            0 <= state.position <= state.capacity <= profile.max_position_embeddings,
            "linear KV position is outside capacity",
        )
        return state.position
    require(
        state.keys.shape[1] <= profile.max_position_embeddings,
        "KV state exceeds its context profile",
    )
    return state.keys.shape[1]


def zeros_state(
    config: AttentionConfig,
    dtype: mx.Dtype = mx.bfloat16,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXAttentionState:
    context.validate_range(context_profile, 0)
    shape = (config.num_kv_heads, 0, config.head_dim)
    return MLXAttentionState(
        keys=mx.zeros(shape, dtype=dtype),
        values=mx.zeros(shape, dtype=dtype),
        context_profile=context_profile,
    )


def linearize_state(
    state: MLXAttentionState,
    capacity: int,
    config: AttentionConfig = PRODUCTION_CONFIG,
) -> MLXLinearAttentionState:
    """Copy one immutable prefix into append-only, fixed-capacity buffers."""
    require(isinstance(state, MLXAttentionState), "linear source state must be immutable")
    position = state_length(state, config)
    require(capacity >= position, "linear KV capacity is shorter than the prefix")
    context.validate_range(state.context_profile, 0, capacity)
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
        context_profile=state.context_profile,
    )


def _rms_norm(value: mx.array, weight: mx.array, eps: float, dtype: mx.Dtype) -> mx.array:
    value32 = value.astype(mx.float32)
    normalized = value32 * mx.rsqrt(mx.mean(value32 * value32, axis=-1, keepdims=True) + eps)
    return (normalized * (1.0 + weight.astype(mx.float32))).astype(dtype)


def _linear_batch(weight: mx.array, vectors: mx.array) -> mx.array:
    return mx.vmap(lambda vector: mx.matmul(weight, vector))(vectors)


def _prefill_linear(
    weight: mx.array,
    vectors: mx.array,
    token_tiled: bool,
) -> mx.array:
    if token_tiled and weight.dtype == mx.bfloat16 and vectors.shape[0] >= 8:
        return dense.token_tiled_matvec(
            weight,
            vectors,
            token_tile=8,
            simdgroups_per_threadgroup=16,
        )
    return _linear_batch(weight, vectors)


QKV_PREFILL_PROJECTION_KERNEL_SOURCE = r"""
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x * SIMDGROUPS + group;
if (row >= 9216u) return;
float sums[8];
for (uint token = 0u; token < 8u; ++token) {
    sums[token] = 0.0f;
}
for (uint column = lane * 4u; column < 2048u; column += 128u) {
    uint local_row;
    if (row < 8192u) {
        local_row = row;
    } else if (row < 8704u) {
        local_row = row - 8192u;
    } else {
        local_row = row - 8704u;
    }
    uint weight_base = local_row * 2048u + column;
    float weight0;
    float weight1;
    float weight2;
    float weight3;
    if (row < 8192u) {
        weight0 = float(q_weight[weight_base]);
        weight1 = float(q_weight[weight_base + 1u]);
        weight2 = float(q_weight[weight_base + 2u]);
        weight3 = float(q_weight[weight_base + 3u]);
    } else if (row < 8704u) {
        weight0 = float(k_weight[weight_base]);
        weight1 = float(k_weight[weight_base + 1u]);
        weight2 = float(k_weight[weight_base + 2u]);
        weight3 = float(k_weight[weight_base + 3u]);
    } else {
        weight0 = float(v_weight[weight_base]);
        weight1 = float(v_weight[weight_base + 1u]);
        weight2 = float(v_weight[weight_base + 2u]);
        weight3 = float(v_weight[weight_base + 3u]);
    }
    for (uint token = 0u; token < 8u; ++token) {
        uint input_base = token * 2048u + column;
        sums[token] += weight0 * float(input[input_base]);
        sums[token] += weight1 * float(input[input_base + 1u]);
        sums[token] += weight2 * float(input[input_base + 2u]);
        sums[token] += weight3 * float(input[input_base + 3u]);
    }
}
for (uint token = 0u; token < 8u; ++token) {
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        sums[token] += simd_shuffle_down(sums[token], offset);
    }
}
if (lane == 0u) {
    if (row < 8192u) {
        for (uint token = 0u; token < 8u; ++token) {
            output_q[token * 8192u + row] = bfloat16_t(sums[token]);
        }
    } else if (row < 8704u) {
        uint output_row = row - 8192u;
        for (uint token = 0u; token < 8u; ++token) {
            output_k[token * 512u + output_row] = bfloat16_t(sums[token]);
        }
    } else {
        uint output_row = row - 8704u;
        for (uint token = 0u; token < 8u; ++token) {
            output_v[token * 512u + output_row] = bfloat16_t(sums[token]);
        }
    }
}
"""


_qkv_prefill_projection_kernel = mx.fast.metal_kernel(
    name="ornith35_attention_qkv_prefill_projection_bf16_block8",
    input_names=["q_weight", "k_weight", "v_weight", "input"],
    output_names=["output_q", "output_k", "output_v"],
    source=QKV_PREFILL_PROJECTION_KERNEL_SOURCE,
)


def fused_qkv_prefill_projection(
    q_weight: mx.array,
    k_weight: mx.array,
    v_weight: mx.array,
    hidden: mx.array,
    *,
    simdgroups: int = 16,
) -> tuple[mx.array, mx.array, mx.array]:
    """Project one exact eight-token production Q/K/V block."""
    require(
        q_weight.dtype == mx.bfloat16 and q_weight.shape == (8192, 2048),
        "fused attention Q projection mismatch",
    )
    require(
        k_weight.dtype == mx.bfloat16 and k_weight.shape == (512, 2048),
        "fused attention K projection mismatch",
    )
    require(
        v_weight.dtype == mx.bfloat16 and v_weight.shape == (512, 2048),
        "fused attention V projection mismatch",
    )
    require(
        hidden.dtype == mx.bfloat16 and hidden.shape == (8, 2048),
        "fused attention hidden block mismatch",
    )
    require(simdgroups in (8, 16, 32), "invalid attention QKV SIMD groups")
    threads = simdgroups * 32
    return _qkv_prefill_projection_kernel(
        inputs=[q_weight, k_weight, v_weight, hidden],
        template=[("SIMDGROUPS", simdgroups)],
        grid=(((9216 + simdgroups - 1) // simdgroups) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(8, 8192), (8, 512), (8, 512)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )


def make_text_rope(
    position: int,
    tokens: int,
    config: AttentionConfig,
    dtype: mx.Dtype,
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> MLXTextRoPE:
    """Build one exact partial-RoPE table for reuse across attention layers."""
    profile = context.validate_range(context_profile, position, tokens)
    require(tokens > 0, "invalid text RoPE range")
    indices = mx.arange(0, config.rotary_dim, 2, dtype=mx.float32)
    inverse_frequencies = mx.power(
        config.rope_theta,
        -indices / config.rotary_dim,
    )
    if profile.rope_type == "yarn":
        low, high = context.yarn_correction_range(
            profile,
            config.rotary_dim,
            config.rope_theta,
        )
        if low == high:
            high += 0.001
        ramp = mx.clip(
            (mx.arange(config.rotary_dim // 2, dtype=mx.float32) - low)
            / (high - low),
            0.0,
            1.0,
        )
        inverse_frequencies = (
            (inverse_frequencies / profile.factor) * ramp
            + inverse_frequencies * (1.0 - ramp)
        )
    if tokens == 1:
        frequencies = inverse_frequencies * position
        angles = mx.concatenate([frequencies, frequencies])
    else:
        positions = mx.arange(position, position + tokens, dtype=mx.float32)
        frequencies = positions[:, None] * inverse_frequencies[None, :]
        angles = mx.concatenate([frequencies, frequencies], axis=-1)
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    attention_factor = context.profile_attention_factor(profile)
    if attention_factor != 1.0:
        cosine = cosine * attention_factor
        sine = sine * attention_factor
    return MLXTextRoPE(
        position=position,
        tokens=tokens,
        cosine=cosine.astype(dtype),
        sine=sine.astype(dtype),
        context_profile=profile.profile_id,
    )


def _validate_rope(
    rope: MLXTextRoPE,
    position: int,
    tokens: int,
    config: AttentionConfig,
    dtype: mx.Dtype,
    context_profile: str | None = None,
) -> None:
    profile = context.validate_range(rope.context_profile, position, tokens)
    if context_profile is not None:
        require(
            profile.profile_id == context.resolve_profile(context_profile).profile_id,
            "text RoPE context profile mismatch",
        )
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
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> mx.array:
    half = config.rotary_dim // 2
    if rope is None:
        rope = make_text_rope(position, 1, config, dtype, context_profile)
    _validate_rope(rope, position, 1, config, dtype, context_profile)
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
    context_profile: str = context.NATIVE_PROFILE_ID,
) -> mx.array:
    require(value.ndim == 3 and value.shape[0] > 0, "RoPE chunk shape mismatch")
    half = config.rotary_dim // 2
    if rope is None:
        rope = make_text_rope(
            start_position,
            value.shape[0],
            config,
            dtype,
            context_profile,
        )
    _validate_rope(
        rope,
        start_position,
        value.shape[0],
        config,
        dtype,
        context_profile,
    )
    if value.shape[0] == 1:
        cosine = rope.cosine[None, None, :]
        sine = rope.sine[None, None, :]
    else:
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


def fused_qk_norm_rope_chunk(
    query_gate: mx.array,
    key: mx.array,
    q_norm: mx.array,
    k_norm: mx.array,
    rope: MLXTextRoPE,
) -> tuple[mx.array, mx.array, mx.array]:
    """Apply exact production Q/K norms, RoPE, and gate split to a chunk."""
    require(
        query_gate.dtype == mx.bfloat16
        and query_gate.ndim == 2
        and query_gate.shape[0] > 0
        and query_gate.shape[1] == 8192,
        "fused attention chunk query/gate mismatch",
    )
    tokens = query_gate.shape[0]
    require(
        key.dtype == mx.bfloat16 and key.shape == (tokens, 2, 256),
        "fused attention chunk key mismatch",
    )
    require(
        q_norm.dtype == mx.bfloat16 and q_norm.shape == (256,),
        "fused attention chunk query norm mismatch",
    )
    require(
        k_norm.dtype == mx.bfloat16 and k_norm.shape == (256,),
        "fused attention chunk key norm mismatch",
    )
    _validate_rope(rope, rope.position, tokens, PRODUCTION_CONFIG, mx.bfloat16)
    query, gate, output_key = _qk_norm_rope_chunk_kernel(
        inputs=[
            query_gate,
            key,
            q_norm,
            k_norm,
            rope.cosine.reshape(tokens, 64),
            rope.sine.reshape(tokens, 64),
        ],
        grid=(tokens * 18 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(tokens, 16, 256), (tokens, 16, 256), (tokens, 2, 256)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )
    return query, gate, output_key


def fused_key_norm_rope_chunk(
    key: mx.array,
    k_norm: mx.array,
    rope: MLXTextRoPE,
) -> mx.array:
    """Apply exact production K norm and RoPE to a key-only chunk."""
    require(
        key.dtype == mx.bfloat16
        and key.ndim == 3
        and key.shape[0] > 0
        and key.shape[1:] == (2, 256),
        "fused attention key-only chunk mismatch",
    )
    tokens = key.shape[0]
    require(
        k_norm.dtype == mx.bfloat16 and k_norm.shape == (256,),
        "fused attention key-only norm mismatch",
    )
    _validate_rope(rope, rope.position, tokens, PRODUCTION_CONFIG, mx.bfloat16)
    return _key_norm_rope_chunk_kernel(
        inputs=[
            key,
            k_norm,
            rope.cosine.reshape(tokens, 64),
            rope.sine.reshape(tokens, 64),
        ],
        grid=(tokens * 2 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[key.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def project_prefill_qkv_for_analysis(
    hidden: mx.array,
    weights: MLXAttentionWeights,
    position: int,
    config: AttentionConfig = PRODUCTION_CONFIG,
    *,
    context_profile: str = context.NATIVE_PROFILE_ID,
    rope: MLXTextRoPE | None = None,
    token_tiled_projections: bool = True,
    fused_prefill_qkv_projection: bool = True,
    fused_prefill_qk_norm_rope: bool = True,
) -> MLXPrefillQKVTrace:
    """Expose exact prefill Q/K/V for bounded numerical characterization only."""
    require(
        hidden.ndim == 2
        and hidden.shape[0] > 0
        and hidden.shape[1] == config.hidden_size,
        "attention analysis hidden-state shape mismatch",
    )
    require(position >= 0, "attention analysis position must be nonnegative")
    validate_weights(weights, config)
    model_dtype = weights.q_proj.dtype
    hidden = hidden.astype(model_dtype)
    tokens = hidden.shape[0]
    fused_projection = (
        fused_prefill_qkv_projection
        and token_tiled_projections
        and config == PRODUCTION_CONFIG
        and model_dtype == mx.bfloat16
        and tokens == 8
    )
    if fused_projection:
        query_gate, key, value = fused_qkv_prefill_projection(
            weights.q_proj,
            weights.k_proj,
            weights.v_proj,
            hidden,
        )
    else:
        query_gate = _prefill_linear(
            weights.q_proj,
            hidden,
            token_tiled_projections,
        )
        key = _prefill_linear(
            weights.k_proj,
            hidden,
            token_tiled_projections,
        )
        value = _prefill_linear(
            weights.v_proj,
            hidden,
            token_tiled_projections,
        )
    key = key.reshape(tokens, config.num_kv_heads, config.head_dim)
    value = value.reshape(tokens, config.num_kv_heads, config.head_dim)
    fused_norm_rope = (
        fused_prefill_qk_norm_rope
        and config == PRODUCTION_CONFIG
        and model_dtype == mx.bfloat16
    )
    if fused_norm_rope:
        if rope is None:
            rope = make_text_rope(
                position,
                tokens,
                config,
                model_dtype,
                context_profile,
            )
        _validate_rope(
            rope,
            position,
            tokens,
            config,
            model_dtype,
            context_profile,
        )
        query, gate, key = fused_qk_norm_rope_chunk(
            query_gate,
            key,
            weights.q_norm,
            weights.k_norm,
            rope,
        )
    else:
        query_gate = query_gate.reshape(
            tokens,
            config.num_q_heads,
            config.head_dim * 2,
        )
        query = query_gate[:, :, : config.head_dim]
        gate = query_gate[:, :, config.head_dim :]
        query = _rms_norm(query, weights.q_norm, config.rms_norm_eps, model_dtype)
        key = _rms_norm(key, weights.k_norm, config.rms_norm_eps, model_dtype)
        query = _apply_text_rope_chunk(
            query,
            position,
            config,
            model_dtype,
            rope,
            context_profile,
        )
        key = _apply_text_rope_chunk(
            key,
            position,
            config,
            model_dtype,
            rope,
            context_profile,
        )
    return MLXPrefillQKVTrace(
        queries=query,
        gates=gate,
        keys=key,
        values=value,
    )


def decode_step(
    hidden: mx.array,
    state: AttentionState,
    weights: MLXAttentionWeights,
    config: AttentionConfig = PRODUCTION_CONFIG,
    *,
    rope: MLXTextRoPE | None = None,
    fused_qk_norm_rope: bool = True,
    grouped_gqa: bool = True,
    _validated: bool = False,
) -> tuple[mx.array, AttentionState]:
    """Append one causal text token under the state's ownership contract."""
    require(hidden.ndim == 1 and hidden.shape == (config.hidden_size,), "hidden-state shape mismatch")
    position = state_length(state, config)
    if not _validated:
        validate_weights(weights, config)
    model_dtype = weights.q_proj.dtype
    require(
        not isinstance(state, MLXTurboQuantImmutableAttentionState),
        "immutable TurboQuant state must be linearized before decode",
    )
    packed_state = isinstance(state, MLXTurboQuantAttentionState)
    if packed_state:
        require(
            config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16,
            "TurboQuant decode requires production BF16 attention",
        )
    else:
        require(state.keys.dtype == model_dtype, "KV state dtype mismatch")
    hidden = hidden.astype(model_dtype)

    query_gate = mx.matmul(weights.q_proj, hidden)
    key = mx.matmul(weights.k_proj, hidden)
    value = mx.matmul(weights.v_proj, hidden).reshape(config.num_kv_heads, config.head_dim)
    production = config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16
    if fused_qk_norm_rope and production:
        if rope is None:
            rope = make_text_rope(
                position,
                1,
                config,
                model_dtype,
                state.context_profile,
            )
        _validate_rope(
            rope,
            position,
            1,
            config,
            model_dtype,
            state.context_profile,
        )
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
        query = _apply_text_rope(
            query,
            position,
            config,
            model_dtype,
            rope,
            state.context_profile,
        )
        key = _apply_text_rope(
            key,
            position,
            config,
            model_dtype,
            rope,
            state.context_profile,
        )

    if packed_state:
        next_state = turboquant_cache.advance_linear_state(
            state,
            key[:, None, :],
            value[:, None, :],
        )
        scores = turboquant_cache.packed_scores(query, next_state)
        probabilities = mx.softmax(
            scores.astype(mx.float32) * (config.head_dim**-0.5),
            axis=-1,
        )
        attended = turboquant_cache.packed_attend(probabilities, next_state)
        attended = attended.astype(model_dtype)
    elif isinstance(state, MLXLinearAttentionState):
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
            context_profile=state.context_profile,
        )
    else:
        next_keys = mx.concatenate([state.keys, key[:, None, :]], axis=1)
        next_values = mx.concatenate([state.values, value[:, None, :]], axis=1)
        next_state = MLXAttentionState(
            keys=next_keys,
            values=next_values,
            context_profile=state.context_profile,
        )
    if not packed_state:
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


def prefill_kv_chunk(
    hidden: mx.array,
    state: MLXAttentionState | MLXLinearAttentionState,
    weights: MLXAttentionWeights,
    config: AttentionConfig = PRODUCTION_CONFIG,
    *,
    rope: MLXTextRoPE | None = None,
    token_tiled_projections: bool = True,
    fused_prefill_qk_norm_rope: bool = True,
) -> MLXAttentionState | MLXLinearAttentionState:
    """Append exact K/V for a chunk whose attention output is not observable."""
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0
        and hidden.shape[1] == config.hidden_size,
        "attention K/V prefill hidden-state shape mismatch",
    )
    position = state_length(state, config)
    validate_weights(weights, config)
    model_dtype = weights.q_proj.dtype
    require(state.keys.dtype == model_dtype, "KV state dtype mismatch")
    hidden = hidden.astype(model_dtype)
    tokens = hidden.shape[0]
    key = _prefill_linear(
        weights.k_proj,
        hidden,
        token_tiled_projections,
    ).reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    value = _prefill_linear(
        weights.v_proj,
        hidden,
        token_tiled_projections,
    ).reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    fused_norm_rope = (
        fused_prefill_qk_norm_rope
        and config == PRODUCTION_CONFIG
        and model_dtype == mx.bfloat16
    )
    if fused_norm_rope:
        if rope is None:
            rope = make_text_rope(
                position,
                tokens,
                config,
                model_dtype,
                state.context_profile,
            )
        _validate_rope(
            rope,
            position,
            tokens,
            config,
            model_dtype,
            state.context_profile,
        )
        key = fused_key_norm_rope_chunk(key, weights.k_norm, rope)
    else:
        key = _rms_norm(key, weights.k_norm, config.rms_norm_eps, model_dtype)
        key = _apply_text_rope_chunk(
            key,
            position,
            config,
            model_dtype,
            rope,
            state.context_profile,
        )
    if isinstance(state, MLXLinearAttentionState):
        require(position + tokens <= state.capacity, "linear KV cache capacity exhausted")
        key_buffer, value_buffer = linear_cache.append_kv_transposed_bf16(
            state.keys,
            state.values,
            key,
            value,
            position,
        )
        return MLXLinearAttentionState(
            keys=key_buffer,
            values=value_buffer,
            position=position + tokens,
            capacity=state.capacity,
            context_profile=state.context_profile,
        )
    return MLXAttentionState(
        keys=mx.concatenate([state.keys, mx.transpose(key, (1, 0, 2))], axis=1),
        values=mx.concatenate(
            [state.values, mx.transpose(value, (1, 0, 2))],
            axis=1,
        ),
        context_profile=state.context_profile,
    )


def prefill_last_query_chunk(
    hidden: mx.array,
    state: MLXAttentionState | MLXLinearAttentionState,
    weights: MLXAttentionWeights,
    config: AttentionConfig = PRODUCTION_CONFIG,
    *,
    rope: MLXTextRoPE | None = None,
    grouped_gqa: bool = True,
    exact_long_prefill: bool = True,
    token_tiled_projections: bool = True,
    fused_prefill_qk_norm_rope: bool = True,
    fused_long_softmax_value: bool | None = None,
) -> tuple[mx.array, MLXAttentionState | MLXLinearAttentionState]:
    """Append a chunk's K/V and evaluate only its observable final query."""
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0
        and hidden.shape[1] == config.hidden_size,
        "attention final-query hidden-state shape mismatch",
    )
    position = state_length(state, config)
    if fused_long_softmax_value is None:
        fused_long_softmax_value = False
    validate_weights(weights, config)
    model_dtype = weights.q_proj.dtype
    require(state.keys.dtype == model_dtype, "KV state dtype mismatch")
    hidden = hidden.astype(model_dtype)
    tokens = hidden.shape[0]
    if rope is None:
        last_rope = make_text_rope(
            position + tokens - 1,
            1,
            config,
            model_dtype,
            state.context_profile,
        )
    else:
        _validate_rope(
            rope,
            position,
            tokens,
            config,
            model_dtype,
            state.context_profile,
        )
        if tokens == 1:
            last_rope = rope
        else:
            last_rope = MLXTextRoPE(
                position=position + tokens - 1,
                tokens=1,
                cosine=rope.cosine[-1],
                sine=rope.sine[-1],
                context_profile=rope.context_profile,
            )
    next_state = prefill_kv_chunk(
        hidden,
        state,
        weights,
        config,
        rope=rope,
        token_tiled_projections=token_tiled_projections,
        fused_prefill_qk_norm_rope=fused_prefill_qk_norm_rope,
    )
    key_length = position + tokens
    if isinstance(next_state, MLXLinearAttentionState):
        next_keys = next_state.keys[:, :key_length, :]
        next_values = next_state.values[:, :key_length, :]
    else:
        next_keys = next_state.keys
        next_values = next_state.values

    query_gate = _linear_batch(weights.q_proj, hidden[-1:]).reshape(
        config.num_q_heads,
        config.head_dim * 2,
    )
    query = query_gate[:, : config.head_dim]
    gate = query_gate[:, config.head_dim :]
    query = _rms_norm(query, weights.q_norm, config.rms_norm_eps, model_dtype)
    query = _apply_text_rope(
        query,
        position + tokens - 1,
        config,
        model_dtype,
        last_rope,
        state.context_profile,
    )
    grouped_gqa = grouped_gqa and (
        config != PRODUCTION_CONFIG
        or position >= GROUPED_GQA_PREFILL_MIN_PREFIX
    )
    if exact_long_prefill and position >= EXACT_LONG_PREFILL_MIN_PREFIX:
        require(
            config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16,
            "exact batched attention requires the production BF16 shape",
        )
        start_scalar = mx.array(position + tokens - 1, dtype=mx.uint32)
        count_scalar = mx.array(1, dtype=mx.uint32)
        length_scalar = mx.array(key_length, dtype=mx.uint32)
        raw_scores = _exact_batched_score_kernel(
            inputs=[
                query[None, :, :],
                next_keys,
                start_scalar,
                count_scalar,
                length_scalar,
            ],
            grid=(((key_length + 7) // 8) * 256, 16, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, config.num_q_heads, key_length)],
            output_dtypes=[model_dtype],
        )[0]
        scaled_scores = raw_scores * (config.head_dim**-0.5)
        if fused_long_softmax_value:
            attended = _exact_fused_softmax_value_kernel(
                inputs=[scaled_scores, next_values, start_scalar, length_scalar],
                grid=(config.num_q_heads * 1024, 1, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(1, config.num_q_heads, config.head_dim)],
                output_dtypes=[model_dtype],
            )[0]
        else:
            probabilities = _exact_looped_softmax_kernel(
                inputs=[scaled_scores, start_scalar, length_scalar],
                grid=(config.num_q_heads * 1024, 1, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[raw_scores.shape],
                output_dtypes=[model_dtype],
            )[0]
            attended = _exact_batched_value_kernel(
                inputs=[probabilities, next_values, start_scalar, length_scalar],
                grid=(8 * 64, config.num_q_heads, 1),
                threadgroup=(64, 1, 1),
                output_shapes=[(1, config.num_q_heads, config.head_dim)],
                output_dtypes=[model_dtype],
            )[0]
    else:
        groups = config.num_q_heads // config.num_kv_heads
        if grouped_gqa:
            scores = mx.matmul(
                query.reshape(
                    config.num_kv_heads,
                    groups,
                    1,
                    config.head_dim,
                ),
                mx.swapaxes(next_keys, 1, 2)[:, None, :, :],
            ).reshape(config.num_q_heads, key_length)
        else:
            repeated_keys = mx.repeat(next_keys, groups, axis=0)
            scores = mx.matmul(
                query[:, None, :],
                mx.swapaxes(repeated_keys, 1, 2),
            ).reshape(config.num_q_heads, key_length)
        probabilities = mx.softmax(
            (scores * (config.head_dim**-0.5)).astype(mx.float32),
            axis=-1,
        ).astype(model_dtype)
        if grouped_gqa:
            attended = mx.matmul(
                probabilities.reshape(
                    config.num_kv_heads,
                    groups,
                    1,
                    key_length,
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
    output = _linear_batch(
        weights.o_proj,
        attended.reshape(1, config.query_dim),
    )[0]
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
    exact_long_prefill: bool = True,
    token_tiled_projections: bool = True,
    fused_prefill_qkv_projection: bool = True,
    fused_prefill_qk_norm_rope: bool = True,
    fused_long_softmax_value: bool | None = None,
    key_tiled_long_scores: bool | None = None,
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
    if fused_long_softmax_value is None:
        fused_long_softmax_value = (
            tokens >= EXACT_FUSED_SOFTMAX_VALUE_MIN_TOKENS
            and position <= EXACT_FUSED_SOFTMAX_VALUE_MAX_PREFIX
        )
    if key_tiled_long_scores is None:
        key_tiled_long_scores = position >= KEY_TILED_PREFILL_MIN_PREFIX
    require(
        not key_tiled_long_scores or exact_long_prefill,
        "key-tiled scores require exact long prefill",
    )
    grouped_gqa = grouped_gqa and (
        config != PRODUCTION_CONFIG
        or position >= GROUPED_GQA_PREFILL_MIN_PREFIX
    )
    fused_projection = (
        fused_prefill_qkv_projection
        and token_tiled_projections
        and config == PRODUCTION_CONFIG
        and model_dtype == mx.bfloat16
        and tokens == 8
    )
    if fused_projection:
        query_gate, key, value = fused_qkv_prefill_projection(
            weights.q_proj,
            weights.k_proj,
            weights.v_proj,
            hidden,
        )
    else:
        query_gate = _prefill_linear(
            weights.q_proj,
            hidden,
            token_tiled_projections,
        )
        key = _prefill_linear(
            weights.k_proj,
            hidden,
            token_tiled_projections,
        )
        value = _prefill_linear(
            weights.v_proj,
            hidden,
            token_tiled_projections,
        )
    key = key.reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    value = value.reshape(
        tokens,
        config.num_kv_heads,
        config.head_dim,
    )
    fused_norm_rope = (
        fused_prefill_qk_norm_rope
        and config == PRODUCTION_CONFIG
        and model_dtype == mx.bfloat16
    )
    if fused_norm_rope:
        if rope is None:
            rope = make_text_rope(
                position,
                tokens,
                config,
                model_dtype,
                state.context_profile,
            )
        _validate_rope(
            rope,
            position,
            tokens,
            config,
            model_dtype,
            state.context_profile,
        )
        query, gate, key = fused_qk_norm_rope_chunk(
            query_gate,
            key,
            weights.q_norm,
            weights.k_norm,
            rope,
        )
    else:
        query_gate = query_gate.reshape(
            tokens,
            config.num_q_heads,
            config.head_dim * 2,
        )
        query = query_gate[:, :, : config.head_dim]
        gate = query_gate[:, :, config.head_dim :]
        query = _rms_norm(query, weights.q_norm, config.rms_norm_eps, model_dtype)
        key = _rms_norm(key, weights.k_norm, config.rms_norm_eps, model_dtype)
        query = _apply_text_rope_chunk(
            query,
            position,
            config,
            model_dtype,
            rope,
            state.context_profile,
        )
        key = _apply_text_rope_chunk(
            key,
            position,
            config,
            model_dtype,
            rope,
            state.context_profile,
        )

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
            context_profile=state.context_profile,
        )
    else:
        next_keys = mx.concatenate([state.keys, key_update], axis=1)
        next_values = mx.concatenate([state.values, value_update], axis=1)
        next_state = MLXAttentionState(
            keys=next_keys,
            values=next_values,
            context_profile=state.context_profile,
        )
    batched_raw_scores = None
    start_scalar = None
    count_scalar = None
    length_scalar = None
    if key_tiled_long_scores:
        require(
            config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16,
            "key-tiled scores require the production BF16 shape",
        )
        key_length = position + tokens
        start_scalar = mx.array(position, dtype=mx.uint32)
        count_scalar = mx.array(tokens, dtype=mx.uint32)
        length_scalar = mx.array(key_length, dtype=mx.uint32)
        batched_raw_scores = _exact_batched_scores(
            query,
            next_keys,
            start_scalar,
            count_scalar,
            length_scalar,
            queries_count=tokens,
            keys_count=key_length,
            key_tiled=True,
        )
    if exact_long_prefill and position >= EXACT_LONG_PREFILL_MIN_PREFIX:
        require(
            config == PRODUCTION_CONFIG and model_dtype == mx.bfloat16,
            "exact batched attention requires the production BF16 shape",
        )
        key_length = position + tokens
        if batched_raw_scores is None:
            start_scalar = mx.array(position, dtype=mx.uint32)
            count_scalar = mx.array(tokens, dtype=mx.uint32)
            length_scalar = mx.array(key_length, dtype=mx.uint32)
            batched_raw_scores = _exact_batched_scores(
                query,
                next_keys,
                start_scalar,
                count_scalar,
                length_scalar,
                queries_count=tokens,
                keys_count=key_length,
                key_tiled=False,
            )
        require(
            start_scalar is not None and length_scalar is not None,
            "exact score scalars are missing",
        )
        raw_scores = batched_raw_scores
        scaled_scores = raw_scores * (config.head_dim**-0.5)
        if fused_long_softmax_value:
            attended = _exact_fused_softmax_value_kernel(
                inputs=[scaled_scores, next_values, start_scalar, length_scalar],
                grid=(tokens * config.num_q_heads * 1024, 1, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(tokens, config.num_q_heads, config.head_dim)],
                output_dtypes=[model_dtype],
            )[0]
        else:
            probabilities = _exact_looped_softmax_kernel(
                inputs=[scaled_scores, start_scalar, length_scalar],
                grid=(tokens * config.num_q_heads * 1024, 1, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[scaled_scores.shape],
                output_dtypes=[model_dtype],
            )[0]
            attended = _exact_batched_value_kernel(
                inputs=[probabilities, next_values, start_scalar, length_scalar],
                grid=(8 * 64, tokens * config.num_q_heads, 1),
                threadgroup=(64, 1, 1),
                output_shapes=[(tokens, config.num_q_heads, config.head_dim)],
                output_dtypes=[model_dtype],
            )[0]
        attended = attended * mx.sigmoid(gate)
        output = _prefill_linear(
            weights.o_proj,
            attended.reshape(tokens, config.query_dim),
            token_tiled_projections,
        )
        return output, next_state
    if not use_steel or config != PRODUCTION_CONFIG or batched_raw_scores is not None:
        groups = config.num_q_heads // config.num_kv_heads
        if not grouped_gqa:
            repeated_keys = mx.repeat(next_keys, groups, axis=0)
            repeated_values = mx.repeat(next_values, groups, axis=0)
        attended_tokens = []
        for offset, (token_query, token_gate) in enumerate(zip(query, gate)):
            length = position + offset + 1
            if batched_raw_scores is not None:
                scores = batched_raw_scores[offset, :, :length]
            elif grouped_gqa:
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
        output = _prefill_linear(
            weights.o_proj,
            attended.reshape(tokens, config.query_dim),
            token_tiled_projections,
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
    output = _prefill_linear(
        weights.o_proj,
        attended.reshape(tokens, config.query_dim),
        token_tiled_projections,
    )
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

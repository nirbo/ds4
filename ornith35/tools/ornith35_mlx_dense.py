#!/usr/bin/env python3
"""Exact token-tiled BF16 projection primitives for Ornith-35 prefill."""

from __future__ import annotations

import mlx.core as mx

from ornith35_moe_reference import require


TOKEN_TILED_MATVEC_SOURCE = r"""
uint work_item = threadgroup_position_in_grid.x * SIMDGROUPS_PER_THREADGROUP
    + simdgroup_index_in_threadgroup;
uint token_tiles = (TOKENS + TOKEN_TILE - 1u) / TOKEN_TILE;
if (work_item >= token_tiles * ROWS) return;
uint token_tile = work_item / ROWS;
uint row = work_item - token_tile * ROWS;
uint token_base = token_tile * TOKEN_TILE;
uint lane = thread_index_in_simdgroup;
float sums[TOKEN_TILE];
for (uint local_token = 0u; local_token < TOKEN_TILE; ++local_token) {
    sums[local_token] = 0.0f;
}
for (uint column = lane * 4u; column < COLUMNS; column += 128u) {
    uint weight_base = row * COLUMNS + column;
    float weight0 = float(weight[weight_base]);
    float weight1 = float(weight[weight_base + 1u]);
    float weight2 = float(weight[weight_base + 2u]);
    float weight3 = float(weight[weight_base + 3u]);
    for (uint local_token = 0u; local_token < TOKEN_TILE; ++local_token) {
        uint token = token_base + local_token;
        if (token < TOKENS) {
            uint input_base = token * COLUMNS + column;
            sums[local_token] += weight0 * float(input[input_base]);
            sums[local_token] += weight1 * float(input[input_base + 1u]);
            sums[local_token] += weight2 * float(input[input_base + 2u]);
            sums[local_token] += weight3 * float(input[input_base + 3u]);
        }
    }
}
for (uint local_token = 0u; local_token < TOKEN_TILE; ++local_token) {
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        sums[local_token] += simd_shuffle_down(sums[local_token], offset);
    }
    uint token = token_base + local_token;
    if (lane == 0u && token < TOKENS) {
        output[token * ROWS + row] = bfloat16_t(sums[local_token]);
    }
}
"""


_token_tiled_matvec_kernel = mx.fast.metal_kernel(
    name="ornith35_bf16_token_tiled_matvec",
    input_names=["weight", "input"],
    output_names=["output"],
    source=TOKEN_TILED_MATVEC_SOURCE,
)


def token_tiled_matvec(
    weight: mx.array,
    vectors: mx.array,
    *,
    token_tile: int = 4,
    simdgroups_per_threadgroup: int = 8,
) -> mx.array:
    """Project a nonempty BF16 token matrix with exact one-token reductions."""
    require(
        weight.dtype == mx.bfloat16 and weight.ndim == 2,
        "token-tiled weight must be a BF16 matrix",
    )
    require(
        vectors.dtype == mx.bfloat16 and vectors.ndim == 2,
        "token-tiled input must be a BF16 matrix",
    )
    rows, columns = weight.shape
    tokens, input_columns = vectors.shape
    require(tokens > 0 and rows > 0, "token-tiled projection must be nonempty")
    require(input_columns == columns, "token-tiled projection shape mismatch")
    require(columns % 128 == 0, "token-tiled columns must be 128-aligned")
    require(token_tile in (1, 2, 4, 8), "invalid token-tile size")
    require(
        simdgroups_per_threadgroup in (8, 16, 32),
        "invalid token-tiled SIMD-group count",
    )
    work_items = ((tokens + token_tile - 1) // token_tile) * rows
    threads = simdgroups_per_threadgroup * 32
    return _token_tiled_matvec_kernel(
        inputs=[weight, vectors],
        template=[
            ("TOKENS", tokens),
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("TOKEN_TILE", token_tile),
            ("SIMDGROUPS_PER_THREADGROUP", simdgroups_per_threadgroup),
        ],
        grid=(
            ((work_items + simdgroups_per_threadgroup - 1)
             // simdgroups_per_threadgroup) * threads,
            1,
            1,
        ),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tokens, rows)],
        output_dtypes=[mx.bfloat16],
    )[0]

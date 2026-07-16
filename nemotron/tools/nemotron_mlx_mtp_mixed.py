#!/usr/bin/env python3
"""Single-dispatch Metal matvec for mixed one-bit/three-bit MTP expert banks."""

from __future__ import annotations

import mlx.core as mx

from nemotron_metadata import require


MIXED_AFFINE_SOURCE = r"""
uint task = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (task >= SELECTED * ROWS) return;
uint slot = task / ROWS;
uint row = task - slot * ROWS;
uint sidecar_expert = uint(indices[slot]);
int low_expert = low_map[sidecar_expert];
bool use_low = low_expert >= 0;
uint local_expert = use_low ? uint(low_expert) : uint(high_map[sidecar_expert]);
uint input_base = PER_EXPERT ? slot * COLUMNS : 0u;
uint scale_base = (local_expert * ROWS + row) * GROUPS;
uint packed_bytes = use_low ? LOW_PACKED_BYTES : HIGH_PACKED_BYTES;
uint weight_base = (local_expert * ROWS + row) * packed_bytes;
const device uchar* low_bytes = reinterpret_cast<const device uchar*>(low_weight);
const device uchar* high_bytes = reinterpret_cast<const device uchar*>(high_weight);

float sum = 0.0f;
for (uint column = thread_index_in_simdgroup; column < COLUMNS; column += 32u) {
    uint code;
    if (use_low) {
        uchar packed = low_bytes[weight_base + (column >> 3u)];
        code = (uint(packed) >> (column & 7u)) & 1u;
    } else {
        uint bit = column * 3u;
        uint byte_index = bit >> 3u;
        uint shift = bit & 7u;
        uint packed = uint(high_bytes[weight_base + byte_index]);
        if (shift > 5u) {
            packed |= uint(high_bytes[weight_base + byte_index + 1u]) << 8u;
        }
        code = (packed >> shift) & 7u;
    }
    uint group = column >> 7u;
    float scale = use_low ? float(low_scales[scale_base + group])
                          : float(high_scales[scale_base + group]);
    float bias = use_low ? float(low_biases[scale_base + group])
                         : float(high_biases[scale_base + group]);
    sum += input[input_base + column] * (float(code) * scale + bias);
}
sum = simd_sum(sum);
if (thread_index_in_simdgroup == 0u) {
    output[slot * ROWS + row] = sum;
}
"""


_mixed_affine_kernel = mx.fast.metal_kernel(
    name="nemotron_mtp_mixed_affine_1b3b_switch_f32",
    input_names=[
        "low_weight",
        "low_scales",
        "low_biases",
        "high_weight",
        "high_scales",
        "high_biases",
        "low_map",
        "high_map",
        "indices",
        "input",
    ],
    output_names=["output"],
    source=MIXED_AFFINE_SOURCE,
)


def mixed_affine_switch(
    x: mx.array,
    indices: mx.array,
    low_bank: dict,
    high_bank: dict,
    projection: str,
) -> mx.array:
    """Dispatch each routed slot to exactly one packed affine expert bank."""

    low = low_bank[projection]
    high = high_bank[projection]
    require(
        (low["settings"]["group_size"], low["settings"]["bits"], low["settings"]["mode"])
        == (128, 1, "affine")
        and (
            high["settings"]["group_size"],
            high["settings"]["bits"],
            high["settings"]["mode"],
        )
        == (128, 3, "affine"),
        "mixed MTP Metal kernel requires affine 1-bit/3-bit group-128 banks",
    )
    require(
        low["biases"] is not None and high["biases"] is not None,
        "mixed MTP Metal kernel requires affine biases",
    )
    rows = int(low["weight"].shape[1])
    require(high["weight"].shape[1] == rows, "mixed MTP bank output widths differ")
    groups = int(low["scales"].shape[-1])
    require(high["scales"].shape[-1] == groups, "mixed MTP bank group counts differ")
    columns = groups * 128
    selected = int(indices.size)
    require(
        x.size in (columns, selected * columns),
        "mixed MTP Metal input shape is unsupported",
    )
    low_packed_bytes = int(low["weight"].shape[-1]) * 4
    high_packed_bytes = int(high["weight"].shape[-1]) * 4
    require(
        low_packed_bytes == columns // 8
        and high_packed_bytes == columns * 3 // 8,
        "mixed MTP packed row width mismatch",
    )
    output = _mixed_affine_kernel(
        inputs=[
            low["weight"],
            low["scales"],
            low["biases"],
            high["weight"],
            high["scales"],
            high["biases"],
            low_bank["original_to_local"],
            high_bank["original_to_local"],
            indices.reshape(-1),
            x.reshape(-1).astype(mx.float32),
        ],
        template=[
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("GROUPS", groups),
            ("SELECTED", selected),
            ("LOW_PACKED_BYTES", low_packed_bytes),
            ("HIGH_PACKED_BYTES", high_packed_bytes),
            ("PER_EXPERT", x.size != columns),
        ],
        grid=((((selected * rows + 7) // 8) * 256), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(selected, 1, rows)],
        output_dtypes=[mx.float32],
    )[0]
    return output.reshape(*indices.shape, 1, rows)

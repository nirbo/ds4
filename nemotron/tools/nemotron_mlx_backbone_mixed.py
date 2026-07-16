#!/usr/bin/env python3
"""Fused fitted-binary/exact-NVFP4 Nemotron backbone expert execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import require
from nemotron_mlx_backbone_lowbit import AffineWeight, dequantize_affine
from nemotron_mlx_moe import NVFP4ExpertMLP, NVFP4SwitchWeight, switch_matmul


FILE_FORMAT = "nemotron-backbone-lowbit-mixed-layer-v1"


KERNEL_HEADER = r"""
constant float nemotron_mixed_e2m1[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

inline float nemotron_mixed_decode_e2m1(uchar nibble) {
    float value = nemotron_mixed_e2m1[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

inline float nemotron_mixed_decode_e4m3fn(uchar bits) {
    uint exponent = (bits >> 3) & 15u;
    uint mantissa = bits & 7u;
    float value;
    if (exponent == 0u) {
        value = float(mantissa) * 0.001953125f;
    } else if (exponent == 15u && mantissa == 7u) {
        value = NAN;
    } else {
        value = (1.0f + float(mantissa) * 0.125f) * exp2(float(int(exponent) - 7));
    }
    return (bits & 128u) ? -value : value;
}
"""


KERNEL_SOURCE = r"""
uint task = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup;
if (task >= SELECTED * ROWS) return;
uint slot = task / ROWS;
uint row = task - slot * ROWS;
uint original_expert = uint(indices[slot]);
int binary_expert = binary_map[original_expert];
bool use_binary = binary_expert >= 0;
uint local_expert = use_binary ? uint(binary_expert) : uint(native_map[original_expert]);
uint input_base = (PER_EXPERT ? slot : slot / TOP_K) * COLUMNS;
const device uchar* binary_bytes = reinterpret_cast<const device uchar*>(binary_weight);

float sum = 0.0f;
for (uint column = thread_index_in_simdgroup; column < COLUMNS; column += 32u) {
    float decoded;
    if (use_binary) {
        uint weight_base = (local_expert * ROWS + row) * BINARY_PACKED_BYTES;
        uchar packed = binary_bytes[weight_base + (column >> 3u)];
        uint code = (uint(packed) >> (column & 7u)) & 1u;
        uint endpoint_base = (local_expert * ROWS + row) * BINARY_GROUPS;
        uint group = column >> 7u;
        decoded = float(code) * float(binary_scales[endpoint_base + group])
            + float(binary_biases[endpoint_base + group]);
    } else {
        uint weight_base = (local_expert * ROWS + row) * NATIVE_PACKED_BYTES;
        uchar packed = native_weight[weight_base + (column >> 1u)];
        uchar nibble = (column & 1u) ? (packed >> 4u) : (packed & 15u);
        uint scale_base = (local_expert * ROWS + row) * NATIVE_BLOCKS;
        decoded = nemotron_mixed_decode_e2m1(nibble)
            * nemotron_mixed_decode_e4m3fn(native_scales[scale_base + (column >> 4u)])
            * native_global_scales[local_expert];
    }
    sum += input[input_base + column] * decoded;
}
sum = simd_sum(sum);
if (thread_index_in_simdgroup == 0u) {
    output[slot * ROWS + row] = sum;
}
"""


_mixed_kernel = mx.fast.metal_kernel(
    name="nemotron_backbone_mixed_binary_nvfp4_switch_f32",
    input_names=[
        "binary_weight",
        "binary_scales",
        "binary_biases",
        "native_weight",
        "native_scales",
        "native_global_scales",
        "binary_map",
        "native_map",
        "indices",
        "input",
    ],
    output_names=["output"],
    header=KERNEL_HEADER,
    source=KERNEL_SOURCE,
)


@dataclass
class BinarySwitchWeight:
    weight: mx.array
    scales: mx.array
    biases: mx.array

    def validate(self) -> None:
        require(self.weight.dtype == mx.uint32 and self.weight.ndim == 3, "invalid binary switch codes")
        require(
            self.scales.dtype == self.biases.dtype == mx.bfloat16
            and self.scales.shape == self.biases.shape
            and self.scales.ndim == 3,
            "invalid binary switch endpoints",
        )
        require(self.weight.shape[:2] == self.scales.shape[:2], "binary switch leading shape mismatch")
        require(self.weight.shape[-1] * 32 == self.scales.shape[-1] * 128, "binary switch width mismatch")

    @property
    def experts(self) -> int:
        return self.weight.shape[0]

    @property
    def input_dims(self) -> int:
        return self.weight.shape[-1] * 32

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]


@dataclass
class BinaryExpertMLP:
    up: BinarySwitchWeight
    down: BinarySwitchWeight

    def validate(self) -> None:
        self.up.validate()
        self.down.validate()
        require(self.up.experts == self.down.experts, "binary expert count mismatch")
        require(self.up.output_dims == self.down.input_dims, "binary expert hidden width mismatch")
        require(self.up.input_dims == self.down.output_dims, "binary expert latent width mismatch")


@dataclass
class MixedExpertMLP:
    binary: BinaryExpertMLP
    native: NVFP4ExpertMLP
    binary_map: mx.array
    native_map: mx.array

    def validate(self) -> None:
        self.binary.validate()
        self.native.validate()
        require(
            self.binary_map.dtype == self.native_map.dtype == mx.int32
            and self.binary_map.ndim == self.native_map.ndim == 1
            and self.binary_map.shape == self.native_map.shape,
            "mixed expert maps are invalid",
        )
        binary_selected = self.binary_map >= 0
        native_selected = self.native_map >= 0
        require(bool(mx.all(binary_selected != native_selected)), "each mixed expert must select one bank")
        require(int(mx.max(self.binary_map)) + 1 == self.binary.up.experts, "binary map is not dense")
        require(int(mx.max(self.native_map)) + 1 == self.native.up.experts, "native map is not dense")
        require(self.binary.up.input_dims == self.native.up.input_dims, "mixed up input width mismatch")
        require(self.binary.up.output_dims == self.native.up.output_dims, "mixed up output width mismatch")


def binary_switch_from_affine(weights: list[AffineWeight]) -> BinarySwitchWeight:
    require(weights, "binary switch bank is empty")
    for weight in weights:
        weight.validate()
        require(weight.bits == 1 and weight.group_size == 128, "binary switch requires affine1-g128")
    first = weights[0]
    require(
        all((weight.rows, weight.columns) == (first.rows, first.columns) for weight in weights),
        "binary switch shapes differ",
    )
    result = BinarySwitchWeight(
        weight=mx.stack([weight.weight for weight in weights]),
        scales=mx.stack([weight.scales for weight in weights]),
        biases=mx.stack([weight.biases for weight in weights]),
    )
    result.validate()
    return result


def compact_native(weights: NVFP4ExpertMLP, expert_ids: list[int]) -> NVFP4ExpertMLP:
    weights.validate()
    require(expert_ids and expert_ids == sorted(set(expert_ids)), "native expert IDs must be sorted")
    require(expert_ids[0] >= 0 and expert_ids[-1] < weights.up.experts, "native expert ID out of range")
    indices = mx.array(expert_ids, dtype=mx.uint32)
    result = NVFP4ExpertMLP(
        up=NVFP4SwitchWeight(
            weights.up.weight[indices],
            weights.up.scales[indices],
            weights.up.global_scales[indices],
        ),
        down=NVFP4SwitchWeight(
            weights.down.weight[indices],
            weights.down.scales[indices],
            weights.down.global_scales[indices],
        ),
    )
    result.validate()
    return result


def build_maps(expert_count: int, binary_ids: list[int], native_ids: list[int]) -> tuple[mx.array, mx.array]:
    require(binary_ids == sorted(set(binary_ids)) and native_ids == sorted(set(native_ids)), "mixed IDs must be sorted")
    require(set(binary_ids).isdisjoint(native_ids), "mixed expert banks overlap")
    require(set(binary_ids).union(native_ids) == set(range(expert_count)), "mixed expert banks are incomplete")
    binary_map = [-1] * expert_count
    native_map = [-1] * expert_count
    for local, original in enumerate(binary_ids):
        binary_map[original] = local
    for local, original in enumerate(native_ids):
        native_map[original] = local
    return mx.array(binary_map, dtype=mx.int32), mx.array(native_map, dtype=mx.int32)


def _binary_reference(x: mx.array, weights: BinarySwitchWeight, indices: mx.array) -> mx.array:
    weights.validate()
    rows = weights.output_dims
    columns = weights.input_dims
    dense = []
    for expert in range(weights.experts):
        affine = AffineWeight(
            weight=weights.weight[expert],
            scales=weights.scales[expert],
            biases=weights.biases[expert],
            bits=1,
            group_size=128,
            rows=rows,
            columns=columns,
        )
        dense.append(dequantize_affine(affine))
    dense = mx.stack(dense)
    selected = dense[indices]
    inputs = mx.expand_dims(x, (-2, -3)) if x.ndim == 3 else mx.expand_dims(x, -2)
    return inputs @ mx.swapaxes(selected, -1, -2)


def mixed_switch_reference(
    x: mx.array,
    weights: MixedExpertMLP,
    indices: mx.array,
    projection: str,
) -> mx.array:
    weights.validate()
    binary = weights.binary.up if projection == "up" else weights.binary.down
    native = weights.native.up if projection == "up" else weights.native.down
    binary_indices = weights.binary_map[indices]
    native_indices = weights.native_map[indices]
    low = _binary_reference(x, binary, mx.maximum(binary_indices, 0))
    high = switch_matmul(x, native, mx.maximum(native_indices, 0))
    return mx.where((binary_indices >= 0)[..., None, None], low, high)


def mixed_switch(
    x: mx.array,
    weights: MixedExpertMLP,
    indices: mx.array,
    projection: str,
) -> mx.array:
    """Decode fitted binary and exact native NVFP4 in one Metal dispatch."""

    weights.validate()
    binary = weights.binary.up if projection == "up" else weights.binary.down
    native = weights.native.up if projection == "up" else weights.native.down
    require(binary.input_dims == native.input_dims, "mixed projection input widths differ")
    require(binary.output_dims == native.output_dims, "mixed projection output widths differ")
    require(x.ndim in (3, 4) and x.shape[-1] == binary.input_dims, "mixed projection input shape mismatch")
    require(indices.ndim == 3 and indices.shape[:2] == x.shape[:2], "mixed projection index shape mismatch")
    if x.ndim == 4:
        require(x.shape[2] == indices.shape[2], "mixed per-expert input shape mismatch")
    selected = indices.size
    rows = binary.output_dims
    columns = binary.input_dims
    output = _mixed_kernel(
        inputs=[
            binary.weight,
            binary.scales,
            binary.biases,
            native.weight,
            native.scales,
            native.global_scales,
            weights.binary_map,
            weights.native_map,
            indices.reshape(-1).astype(mx.int32),
            x.reshape(-1).astype(mx.float32),
        ],
        template=[
            ("ROWS", rows),
            ("COLUMNS", columns),
            ("SELECTED", selected),
            ("BINARY_GROUPS", columns // 128),
            ("BINARY_PACKED_BYTES", columns // 8),
            ("NATIVE_BLOCKS", columns // 16),
            ("NATIVE_PACKED_BYTES", columns // 2),
            ("PER_EXPERT", x.ndim == 4),
            ("TOP_K", indices.shape[-1]),
        ],
        grid=((((selected * rows + 7) // 8) * 256), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*indices.shape, 1, rows)],
        output_dtypes=[mx.float32],
    )[0]
    return output


def mixed_expert_outputs(x: mx.array, weights: MixedExpertMLP, indices: mx.array) -> mx.array:
    up = mixed_switch(x, weights, indices, "up")
    hidden = mx.square(mx.maximum(up, 0.0))
    return mixed_switch(hidden.squeeze(-2), weights, indices, "down").squeeze(-2)


def mixed_expert_mlp(
    x: mx.array,
    weights: MixedExpertMLP,
    indices: mx.array,
    scores: mx.array,
) -> mx.array:
    outputs = mixed_expert_outputs(x, weights, indices)
    require(scores.shape == indices.shape, "mixed expert score shape mismatch")
    return (outputs * scores[..., None]).sum(axis=-2)


def mixed_layer_forward_with_observation(block, weights: MixedExpertMLP, x: mx.array):
    """Compose a mixed LatentMoE layer and retain routed-output norms."""

    hidden = block.norm(x)
    indices, scores = block.route(hidden)
    latent = block.fc1_latent(hidden)
    selected_outputs = mixed_expert_outputs(latent, weights, indices)
    routed_latent = (selected_outputs * scores[..., None]).sum(axis=-2)
    routed = block.fc2_latent(routed_latent)
    shared_hidden = mx.square(mx.maximum(block.shared_up(hidden), 0.0))
    shared = block.shared_down(shared_hidden)
    output_norms = mx.linalg.norm(selected_outputs.astype(mx.float32), axis=-1)
    return x + routed + shared, indices, scores, output_norms


def mixed_layer_forward(block, weights: MixedExpertMLP, x: mx.array):
    """Compose a complete Nemotron LatentMoE layer around mixed experts."""

    output, indices, scores, _ = mixed_layer_forward_with_observation(block, weights, x)
    return output, indices, scores


def mixed_file_tensors(weights: MixedExpertMLP) -> dict[str, mx.array]:
    weights.validate()
    return {
        "binary.up.weight": weights.binary.up.weight,
        "binary.up.scales": weights.binary.up.scales,
        "binary.up.biases": weights.binary.up.biases,
        "binary.down.weight": weights.binary.down.weight,
        "binary.down.scales": weights.binary.down.scales,
        "binary.down.biases": weights.binary.down.biases,
        "native.up.weight": weights.native.up.weight,
        "native.up.scales": weights.native.up.scales,
        "native.up.global_scales": weights.native.up.global_scales,
        "native.down.weight": weights.native.down.weight,
        "native.down.scales": weights.native.down.scales,
        "native.down.global_scales": weights.native.down.global_scales,
        "maps.binary": weights.binary_map,
        "maps.native": weights.native_map,
    }


def mixed_from_tensors(tensors: dict[str, mx.array]) -> MixedExpertMLP:
    expected = {
        "binary.up.weight",
        "binary.up.scales",
        "binary.up.biases",
        "binary.down.weight",
        "binary.down.scales",
        "binary.down.biases",
        "native.up.weight",
        "native.up.scales",
        "native.up.global_scales",
        "native.down.weight",
        "native.down.scales",
        "native.down.global_scales",
        "maps.binary",
        "maps.native",
    }
    require(set(tensors) == expected, "mixed layer tensor set mismatch")
    result = MixedExpertMLP(
        binary=BinaryExpertMLP(
            up=BinarySwitchWeight(
                tensors["binary.up.weight"],
                tensors["binary.up.scales"],
                tensors["binary.up.biases"],
            ),
            down=BinarySwitchWeight(
                tensors["binary.down.weight"],
                tensors["binary.down.scales"],
                tensors["binary.down.biases"],
            ),
        ),
        native=NVFP4ExpertMLP(
            up=NVFP4SwitchWeight(
                tensors["native.up.weight"],
                tensors["native.up.scales"],
                tensors["native.up.global_scales"],
            ),
            down=NVFP4SwitchWeight(
                tensors["native.down.weight"],
                tensors["native.down.scales"],
                tensors["native.down.global_scales"],
            ),
        ),
        binary_map=tensors["maps.binary"],
        native_map=tensors["maps.native"],
    )
    result.validate()
    return result


def load_mixed_file(path: Path) -> tuple[MixedExpertMLP, dict[str, str]]:
    arrays, metadata = mx.load(str(path), return_metadata=True)
    require(metadata.get("format") == FILE_FORMAT, "unsupported mixed layer file")
    return mixed_from_tensors(arrays), metadata

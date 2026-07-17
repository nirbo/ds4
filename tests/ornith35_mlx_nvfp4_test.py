#!/usr/bin/env python3
"""Metal-backed synthetic checks for Ornith-35 NVFP4 composition."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))
import ornith35_nvfp4 as reference

MODULE_PATH = TOOLS / "ornith35_mlx_nvfp4.py"
SPEC = importlib.util.spec_from_file_location("ornith35_mlx_nvfp4", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MLXNVFP4Test(unittest.TestCase):
    def test_fast_decode_header_matches_all_fp4_and_fp8_values(self) -> None:
        probe = mx.fast.metal_kernel(
            name="ornith35_nvfp4_fast_decode_probe",
            input_names=["bits"],
            output_names=["fp4", "fp8"],
            header=MODULE.FAST_DECODE_KERNEL_HEADER,
            source=r"""
uint index = thread_position_in_grid.x;
if (index < 16u) fp4[index] = ornith35_decode_e2m1(bits[index]);
fp8[index] = ornith35_decode_e4m3fn(bits[index]);
""",
        )
        bits = mx.arange(256, dtype=mx.uint32).astype(mx.uint8)
        fp4, fp8 = probe(
            inputs=[bits],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(16,), (256,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        mx.eval(fp4, fp8)
        self.assertEqual(fp4.tolist(), [reference.decode_e2m1(i) for i in range(16)])
        for actual, expected in zip(
            fp8.tolist(),
            (reference.decode_e4m3fn(i) for i in range(256)),
        ):
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, expected)

    def test_production_expert_shapes_compile(self) -> None:
        for rows, columns in ((512, 2048), (2048, 512)):
            packed = mx.zeros((rows, columns // 2), dtype=mx.uint8)
            scales = mx.full((rows, columns // 16), 0x38, dtype=mx.uint8)
            global_scale = mx.ones((1,), dtype=mx.float32)
            vector = mx.ones((columns,), dtype=mx.float32)
            output = MODULE.nvfp4_matvec(packed, scales, global_scale, vector)
            mx.eval(output)
            self.assertEqual(output.shape, (rows,))
            self.assertEqual(mx.max(mx.abs(output)).item(), 0.0)

    def test_matches_scalar_reference(self) -> None:
        packed_values = [
            low | ((low + 1) << 4)
            for low in range(0, 16, 2)
        ] + [0x22] * 8
        packed = mx.array(packed_values, dtype=mx.uint8).reshape(2, 8)
        scales = mx.array([0x38, 0x40], dtype=mx.uint8).reshape(2, 1)
        global_scale = mx.array([2.0], dtype=mx.float32)
        values = [math.sin(index * 0.17) for index in range(16)]
        vector = mx.array(values, dtype=mx.float32)
        output = MODULE.nvfp4_matvec(packed, scales, global_scale, vector)
        mx.eval(output)
        actual = output.tolist()

        expected = []
        for row in range(2):
            total = 0.0
            for column, value in enumerate(values):
                byte = packed_values[row * 8 + column // 2]
                nibble = byte >> 4 if column & 1 else byte & 0xF
                scale = reference.decode_e4m3fn([0x38, 0x40][row])
                total += reference.decode_e2m1(nibble) * scale / 2.0 * value
            expected.append(total)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, delta=2e-6)

    def test_rejects_scale_shape_mismatch(self) -> None:
        packed = mx.zeros((2, 8), dtype=mx.uint8)
        scales = mx.zeros((2, 2), dtype=mx.uint8)
        global_scale = mx.ones((1,), dtype=mx.float32)
        vector = mx.ones((16,), dtype=mx.float32)
        with self.assertRaisesRegex(MODULE.NVFP4Error, "scale shape mismatch"):
            MODULE.nvfp4_matvec(packed, scales, global_scale, vector)

    def test_paired_kernels_match_separate_dispatches(self) -> None:
        gate = mx.array([0x11] * 16 + [0x22] * 16, dtype=mx.uint8).reshape(2, 2, 8)
        up = mx.array([0x34] * 16 + [0x56] * 16, dtype=mx.uint8).reshape(2, 2, 8)
        scales = mx.full((2, 2, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.ones((2,), dtype=mx.float32)
        selected = mx.array([1, 0], dtype=mx.uint32)
        vector = mx.array([math.sin(index * 0.17) for index in range(16)], dtype=mx.float32)

        selected_paired = MODULE.nvfp4_selected_paired_matvec(
            gate,
            scales,
            globals_,
            up,
            scales,
            globals_,
            selected,
            vector,
        )
        selected_separate = mx.stack(
            (
                MODULE.nvfp4_selected_matvec(
                    gate, scales, globals_, selected, vector, batched_input=False
                ),
                MODULE.nvfp4_selected_matvec(
                    up, scales, globals_, selected, vector, batched_input=False
                ),
            ),
            axis=1,
        )
        shared_paired = MODULE.nvfp4_paired_matvec(
            gate[0],
            scales[0],
            globals_[:1],
            up[0],
            scales[0],
            globals_[:1],
            vector,
        )
        shared_separate = mx.stack(
            (
                MODULE.nvfp4_matvec(gate[0], scales[0], globals_[:1], vector),
                MODULE.nvfp4_matvec(up[0], scales[0], globals_[:1], vector),
            )
        )
        mx.eval(selected_paired, selected_separate, shared_paired, shared_separate)
        self.assertEqual(
            float(mx.max(mx.abs(selected_paired - selected_separate)).item()),
            0.0,
        )
        self.assertEqual(
            float(mx.max(mx.abs(shared_paired - shared_separate)).item()),
            0.0,
        )

    def test_selected_weighted_kernel_matches_ordered_reduction(self) -> None:
        packed = mx.array([0x12] * 16 + [0x35] * 16, dtype=mx.uint8).reshape(2, 2, 8)
        scales = mx.full((2, 2, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.array([0.75, 1.25], dtype=mx.float32)
        selected = mx.array([1, 0], dtype=mx.uint32)
        vectors = mx.array(
            [math.sin(index * 0.17) for index in range(32)],
            dtype=mx.float32,
        ).reshape(2, 16)
        down32 = MODULE.nvfp4_selected_matvec(
            packed,
            scales,
            globals_,
            selected,
            vectors,
            batched_input=True,
        )
        for dtype in (mx.float32, mx.bfloat16):
            routing = mx.array([0.375, 0.625], dtype=dtype)
            down = down32.astype(dtype)
            expected = mx.sum(
                down.astype(mx.float32) * routing.astype(mx.float32)[:, None],
                axis=0,
            ).astype(dtype)
            actual = MODULE.nvfp4_selected_weighted_matvec(
                packed,
                scales,
                globals_,
                selected,
                vectors,
                routing,
            )
            mx.eval(expected, actual)
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(float(mx.max(mx.abs(actual - expected)).item()), 0.0)

    def test_rows4_selected_weighted_matches_one_row_kernel(self) -> None:
        packed = mx.array([0x12] * 32 + [0x35] * 32, dtype=mx.uint8).reshape(2, 4, 8)
        scales = mx.full((2, 4, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.array([0.75, 1.25], dtype=mx.float32)
        selected = mx.array([1, 0], dtype=mx.uint32)
        vectors = mx.array(
            [math.sin(index * 0.17) for index in range(32)],
            dtype=mx.float32,
        ).reshape(2, 16)
        for dtype in (mx.float32, mx.bfloat16):
            routing = mx.array([0.375, 0.625], dtype=dtype)
            expected = MODULE.nvfp4_selected_weighted_matvec(
                packed,
                scales,
                globals_,
                selected,
                vectors,
                routing,
            )
            actual = MODULE.nvfp4_selected_weighted_rows4_matvec(
                packed,
                scales,
                globals_,
                selected,
                vectors,
                routing,
            )
            mx.eval(expected, actual)
            self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_rows4_selected_shared_merge_matches_split_path(self) -> None:
        routed = mx.array([0x12] * 32 + [0x35] * 32, dtype=mx.uint8).reshape(2, 4, 8)
        routed_scales = mx.full((2, 4, 1), 0x38, dtype=mx.uint8)
        routed_globals = mx.array([0.75, 1.25], dtype=mx.float32)
        shared = mx.full((4, 8), 0x24, dtype=mx.uint8)
        shared_scales = mx.full((4, 1), 0x38, dtype=mx.uint8)
        shared_global = mx.array([0.875], dtype=mx.float32)
        selected = mx.array([1, 0], dtype=mx.uint32)
        routed_vectors = mx.array(
            [math.sin(index * 0.17) for index in range(32)],
            dtype=mx.float32,
        ).reshape(2, 16)
        shared_vector = mx.array(
            [math.cos(index * 0.13) for index in range(16)],
            dtype=mx.float32,
        )
        routing = mx.array([0.375, 0.625], dtype=mx.bfloat16)
        multiplier = mx.array(0.4375, dtype=mx.bfloat16)
        routed_output = MODULE.nvfp4_selected_weighted_rows4_matvec(
            routed,
            routed_scales,
            routed_globals,
            selected,
            routed_vectors,
            routing,
        )
        shared_output = MODULE.nvfp4_matvec(
            shared,
            shared_scales,
            shared_global,
            shared_vector,
        ).astype(mx.bfloat16)
        expected = (routed_output + shared_output * multiplier).astype(mx.bfloat16)
        actual = MODULE.nvfp4_selected_shared_weighted_rows4_matvec(
            routed,
            routed_scales,
            routed_globals,
            shared,
            shared_scales,
            shared_global,
            selected,
            routed_vectors,
            shared_vector,
            routing,
            multiplier,
        )
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_selected_shared_gate_up_silu_matches_split_path(self) -> None:
        gate = mx.array([0x12] * 32 + [0x35] * 32, dtype=mx.uint8).reshape(2, 4, 8)
        up = mx.array([0x24] * 32 + [0x51] * 32, dtype=mx.uint8).reshape(2, 4, 8)
        scales = mx.full((2, 4, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.array([0.75, 1.25], dtype=mx.float32)
        selected = mx.array([1, 0], dtype=mx.uint32)
        vector = mx.array(
            [math.sin(index * 0.17) for index in range(16)],
            dtype=mx.float32,
        )
        routed_pair = MODULE.nvfp4_selected_paired_matvec(
            gate,
            scales,
            globals_,
            up,
            scales,
            globals_,
            selected,
            vector,
        ).astype(mx.bfloat16)
        shared_pair = MODULE.nvfp4_paired_matvec(
            gate[0],
            scales[0],
            globals_[:1],
            up[0],
            scales[0],
            globals_[:1],
            vector,
        ).astype(mx.bfloat16)
        expected_routed = (
            routed_pair[:, 0] * mx.sigmoid(routed_pair[:, 0]) * routed_pair[:, 1]
        )
        expected_shared = shared_pair[0] * mx.sigmoid(shared_pair[0]) * shared_pair[1]
        actual_routed, actual_shared = MODULE.nvfp4_selected_shared_gate_up_silu(
            gate,
            scales,
            globals_,
            up,
            scales,
            globals_,
            gate[0],
            scales[0],
            globals_[:1],
            up[0],
            scales[0],
            globals_[:1],
            selected,
            vector,
        )
        mx.eval(expected_routed, expected_shared, actual_routed, actual_shared)
        self.assertTrue(bool(mx.array_equal(actual_routed, expected_routed).item()))
        self.assertTrue(bool(mx.array_equal(actual_shared, expected_shared).item()))

    def test_selected_shared_gate_up_uses_precise_bf16_sigmoid(self) -> None:
        gate = mx.array([0x02] + [0x00] * 7, dtype=mx.uint8).reshape(1, 1, 8)
        up = mx.array([0x20] + [0x00] * 7, dtype=mx.uint8).reshape(1, 1, 8)
        scales = mx.full((1, 1, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.array([1.0], dtype=mx.float32)
        selected = mx.array([0], dtype=mx.uint32)
        vector = mx.array(
            [-6.84375, -1.640625] + [0.0] * 14,
            dtype=mx.float32,
        )
        pair = MODULE.nvfp4_selected_paired_matvec(
            gate,
            scales,
            globals_,
            up,
            scales,
            globals_,
            selected,
            vector,
        ).astype(mx.bfloat16)
        expected = pair[:, 0] * mx.sigmoid(pair[:, 0]) * pair[:, 1]
        actual, _ = MODULE.nvfp4_selected_shared_gate_up_silu(
            gate,
            scales,
            globals_,
            up,
            scales,
            globals_,
            gate[0],
            scales[0],
            globals_,
            up[0],
            scales[0],
            globals_,
            selected,
            vector,
        )
        mx.eval(expected, actual)
        self.assertEqual(float(expected[0, 0].item()), 0.01190185546875)
        self.assertTrue(bool(mx.array_equal(actual, expected).item()))

    def test_selected_experts_stay_batched_for_gate_and_down(self) -> None:
        packed_values = [0x11] * 16 + [0x22] * 16
        packed = mx.array(packed_values, dtype=mx.uint8).reshape(2, 2, 8)
        scales = mx.full((2, 2, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.ones((2,), dtype=mx.float32)
        selected = mx.array([1, 0], dtype=mx.uint32)
        shared_vector = mx.array(
            [math.sin(index * 0.17) for index in range(16)],
            dtype=mx.float32,
        )
        gate = MODULE.nvfp4_selected_matvec(
            packed,
            scales,
            globals_,
            selected,
            shared_vector,
            batched_input=False,
        )
        down_vectors = mx.stack([shared_vector, shared_vector * 0.25])
        down = MODULE.nvfp4_selected_matvec(
            packed,
            scales,
            globals_,
            selected,
            down_vectors,
            batched_input=True,
        )
        mx.eval(gate, down)

        vector_sum = math.fsum(shared_vector.tolist())
        for row, expected in zip(gate.tolist(), (vector_sum, vector_sum * 0.5)):
            self.assertAlmostEqual(row[0], expected, delta=2e-6)
            self.assertAlmostEqual(row[1], expected, delta=2e-6)
        for row, expected in zip(down.tolist(), (vector_sum, vector_sum * 0.125)):
            self.assertAlmostEqual(row[0], expected, delta=2e-6)
            self.assertAlmostEqual(row[1], expected, delta=2e-6)

    def test_token_batched_kernels_match_one_token_kernels(self) -> None:
        gate = mx.stack(
            [
                mx.full((16, 8), 0x11, dtype=mx.uint8),
                mx.full((16, 8), 0x23, dtype=mx.uint8),
            ]
        )
        up = mx.stack(
            [
                mx.full((16, 8), 0x34, dtype=mx.uint8),
                mx.full((16, 8), 0x56, dtype=mx.uint8),
            ]
        )
        scales = mx.full((2, 16, 1), 0x38, dtype=mx.uint8)
        globals_ = mx.array([0.75, 1.25], dtype=mx.float32)
        selected = mx.array([[1, 0], [0, 1], [1, 0]], dtype=mx.uint32)
        vectors = mx.array(
            [math.sin((token * 16 + index + 1) * 0.17) for token in range(3) for index in range(16)],
            dtype=mx.float32,
        ).reshape(3, 16)

        shared = MODULE.nvfp4_batched_matvec(
            gate[0], scales[0], globals_[:1], vectors
        )
        shared_expected = mx.stack(
            [
                MODULE.nvfp4_matvec(gate[0], scales[0], globals_[:1], vector)
                for vector in vectors
            ]
        )
        shared_paired = MODULE.nvfp4_batched_paired_matvec(
            gate[0],
            scales[0],
            globals_[:1],
            up[0],
            scales[0],
            globals_[:1],
            vectors,
        )
        shared_paired_expected = mx.stack(
            [
                MODULE.nvfp4_paired_matvec(
                    gate[0],
                    scales[0],
                    globals_[:1],
                    up[0],
                    scales[0],
                    globals_[:1],
                    vector,
                )
                for vector in vectors
            ]
        )
        tiled_shared = [
            MODULE.nvfp4_batched_matvec(
                gate[0],
                scales[0],
                globals_[:1],
                vectors,
                token_tile=token_tile,
            )
            for token_tile in (2, 4, 8)
        ]
        tiled_shared_paired = [
            MODULE.nvfp4_batched_paired_matvec(
                gate[0],
                scales[0],
                globals_[:1],
                up[0],
                scales[0],
                globals_[:1],
                vectors,
                token_tile=token_tile,
            )
            for token_tile in (2, 4, 8)
        ]
        selected_paired = MODULE.nvfp4_batched_selected_paired_matvec(
            gate,
            scales,
            globals_,
            up,
            scales,
            globals_,
            selected,
            vectors,
        )
        selected_paired_expected = mx.stack(
            [
                MODULE.nvfp4_selected_paired_matvec(
                    gate,
                    scales,
                    globals_,
                    up,
                    scales,
                    globals_,
                    token_selected,
                    vector,
                )
                for token_selected, vector in zip(selected, vectors)
            ]
        )
        intermediate = selected_paired[:, :, 0]
        down = mx.stack(
            [
                mx.full((2, 8), 0x12, dtype=mx.uint8),
                mx.full((2, 8), 0x35, dtype=mx.uint8),
            ]
        )
        down_scales = mx.full((2, 2, 1), 0x38, dtype=mx.uint8)
        routing = mx.array(
            [[0.375, 0.625], [0.75, 0.25], [0.5, 0.5]],
            dtype=mx.float32,
        )
        weighted = MODULE.nvfp4_batched_selected_weighted_matvec(
            down,
            down_scales,
            globals_,
            selected,
            intermediate,
            routing,
        )
        minimal_shared = MODULE.nvfp4_batched_matvec(
            gate[0],
            scales[0],
            globals_[:1],
            vectors,
            simdgroups_per_threadgroup=8,
            rows_per_simdgroup=1,
        )
        minimal_shared_paired = MODULE.nvfp4_batched_paired_matvec(
            gate[0],
            scales[0],
            globals_[:1],
            up[0],
            scales[0],
            globals_[:1],
            vectors,
            simdgroups_per_threadgroup=8,
            rows_per_simdgroup=1,
        )
        minimal_selected_paired = MODULE.nvfp4_batched_selected_paired_matvec(
            gate,
            scales,
            globals_,
            up,
            scales,
            globals_,
            selected,
            vectors,
            simdgroups_per_threadgroup=8,
            rows_per_simdgroup=1,
        )
        minimal_weighted = MODULE.nvfp4_batched_selected_weighted_matvec(
            down,
            down_scales,
            globals_,
            selected,
            intermediate,
            routing,
            row_groups_per_threadgroup=1,
            rows_per_simdgroup=1,
        )
        weighted_expected = mx.stack(
            [
                MODULE.nvfp4_selected_weighted_matvec(
                    down,
                    down_scales,
                    globals_,
                    token_selected,
                    token_intermediate,
                    token_routing,
                )
                for token_selected, token_intermediate, token_routing in zip(
                    selected, intermediate, routing
                )
            ]
        )
        mx.eval(
            shared,
            shared_expected,
            shared_paired,
            shared_paired_expected,
            selected_paired,
            selected_paired_expected,
            weighted,
            weighted_expected,
            minimal_shared,
            minimal_shared_paired,
            minimal_selected_paired,
            minimal_weighted,
            *tiled_shared,
            *tiled_shared_paired,
        )
        for actual, expected in (
            (shared, shared_expected),
            (shared_paired, shared_paired_expected),
            (selected_paired, selected_paired_expected),
            (weighted, weighted_expected),
            (shared, minimal_shared),
            (shared_paired, minimal_shared_paired),
            (selected_paired, minimal_selected_paired),
            (weighted, minimal_weighted),
            *((actual, shared_expected) for actual in tiled_shared),
            *((actual, shared_paired_expected) for actual in tiled_shared_paired),
        ):
            self.assertEqual(float(mx.max(mx.abs(actual - expected)).item()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

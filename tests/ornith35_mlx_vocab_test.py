#!/usr/bin/env python3
"""Numerical and validation tests for Ornith-35 vocabulary quantization."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import MoEError


class MLXVocabTest(unittest.TestCase):
    def test_mapped_bf16_rows_preserve_payload_and_order(self) -> None:
        values = [0x3F80] * 32 + [0x4000] * 32 + [0xBF80] * 32
        payload = struct.pack("<96H", *values)
        header = json.dumps(
            {
                "embedding": {
                    "dtype": "BF16",
                    "shape": [3, 32],
                    "data_offsets": [0, len(payload)],
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapped.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            matrix = vocab.MLXMappedBF16Matrix(path, "embedding", (3, 32))
            scalar = matrix.row(1)
            rows = matrix.rows((2, 0, 1))
            mx.eval(scalar, rows)

            self.assertTrue(bool(mx.all(scalar == 2).item()))
            self.assertTrue(bool(mx.all(rows[0] == -1).item()))
            self.assertTrue(bool(mx.all(rows[1] == 1).item()))
            self.assertTrue(bool(mx.all(rows[2] == 2).item()))
            with self.assertRaisesRegex(MoEError, "out of range"):
                matrix.row(3)
            matrix.close()

    def test_affine_q8_projection_and_rows_match_dequantized_weight(self) -> None:
        weight = (
            mx.sin(mx.arange(160, dtype=mx.float32) * 0.17).reshape(5, 32)
            .astype(mx.bfloat16)
        )
        matrix = vocab.quantize_affine(weight, bits=8, group_size=32)
        hidden = mx.cos(mx.arange(64, dtype=mx.float32) * 0.11).reshape(2, 32)
        hidden = hidden.astype(mx.bfloat16)
        actual = vocab.project(matrix, hidden)
        dequantized = vocab.dequantize_rows(matrix, mx.array([0, 1, 2, 3, 4]))
        expected = mx.matmul(hidden, mx.transpose(dequantized))
        selected = vocab.dequantize_rows(matrix, mx.array([1, 4]))
        scalar = vocab.dequantize_rows(matrix, 3)
        mx.eval(actual, expected, selected, scalar)

        difference = actual.astype(mx.float32) - expected.astype(mx.float32)
        relative_l2 = mx.sqrt(mx.sum(difference * difference) / mx.sum(expected * expected))
        self.assertLess(float(relative_l2.item()), 0.02)
        self.assertTrue(
            bool(mx.array_equal(selected, mx.take(dequantized, mx.array([1, 4]), axis=0)).item())
        )
        self.assertTrue(bool(mx.array_equal(scalar, dequantized[3]).item()))
        self.assertEqual(vocab.stored_bytes(matrix), 180)

    def test_exact_selected_rows_match_full_bf16_head_geometry(self) -> None:
        weight = (
            mx.sin(mx.arange(512 * 2048, dtype=mx.float32) * 0.00017)
            .reshape(512, 2048)
            .astype(mx.bfloat16)
        )
        hidden = mx.cos(mx.arange(2048, dtype=mx.float32) * 0.013).astype(mx.bfloat16)
        indices = mx.array(tuple(range(3, 512, 8)), dtype=mx.uint32)
        full = mx.matmul(weight, hidden)
        selected = vocab.project_bf16_rows_exact(
            mx.take(weight, indices, axis=0),
            hidden,
        )
        expected = mx.take(full, indices)
        mx.eval(selected, expected)
        self.assertTrue(bool(mx.array_equal(selected, expected).item()))

    def test_q8_candidates_are_rescored_from_mapped_source_rows(self) -> None:
        weight = (
            mx.sin(mx.arange(80 * 2048, dtype=mx.float32) * 0.00031)
            .reshape(80, 2048)
            .astype(mx.bfloat16)
        )
        hidden = mx.cos(mx.arange(2048, dtype=mx.float32) * 0.019).astype(mx.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.safetensors"
            mx.save_safetensors(path, {"head": weight})
            reference = vocab.MLXMappedBF16Matrix(path, "head", (80, 2048))
            matrix = vocab.quantize_affine(
                weight,
                bits=8,
                group_size=32,
                reference=reference,
            )
            approximate = vocab.project(matrix, hidden)
            token_ids, scores = vocab.exact_candidate_scores(
                matrix,
                approximate,
                hidden,
                candidate_count=64,
            )
            full = mx.matmul(weight, hidden)
            expected = mx.take(full, mx.array(token_ids, dtype=mx.uint32))
            actual = mx.array(scores, dtype=mx.bfloat16)
            mx.eval(expected, actual)
            self.assertTrue(bool(mx.array_equal(actual, expected).item()))
            reference.close()

    def test_rejects_invalid_payload_shape_and_projection_width(self) -> None:
        weight = mx.ones((3, 32), dtype=mx.bfloat16)
        matrix = vocab.quantize_affine(weight, bits=8, group_size=32)
        invalid = vocab.MLXAffineQuantizedMatrix(
            packed=matrix.packed[:, :-1],
            scales=matrix.scales,
            biases=matrix.biases,
            shape=matrix.shape,
            group_size=matrix.group_size,
            bits=matrix.bits,
        )
        with self.assertRaisesRegex(MoEError, "payload shape"):
            vocab.validate(invalid)
        with self.assertRaisesRegex(MoEError, "projection shape"):
            vocab.project(matrix, mx.ones((16,), dtype=mx.bfloat16))


if __name__ == "__main__":
    unittest.main(verbosity=2)

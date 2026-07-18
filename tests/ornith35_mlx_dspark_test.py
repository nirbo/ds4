#!/usr/bin/env python3
"""Scalar-oracle, MLX parity, and loader tests for Ornith-35 DSpark."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_dspark_reference as reference
import ornith35_mlx_dspark as mlx_dspark


CONFIG = reference.DSparkConfig(
    target_vocab_size=11,
    draft_vocab_size=5,
    hidden_size=4,
    aux_hidden_state_indices=(1, 3),
    block_size=4,
    mask_token_id=10,
    num_layers=2,
    intermediate_size=6,
    num_q_heads=2,
    num_kv_heads=1,
    head_dim=2,
    rotary_dim=2,
    rope_theta=1_000.0,
    max_position_embeddings=32,
    rms_norm_eps=1e-6,
    markov_rank=3,
)


def matrix(rows: int, columns: int, phase: float) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            math.sin((row * columns + column + 1) * phase) * 0.17
            for column in range(columns)
        )
        for row in range(rows)
    )


def norm(size: int, phase: float) -> tuple[float, ...]:
    return tuple(0.9 + math.cos((index + 1) * phase) * 0.08 for index in range(size))


def scalar_weights() -> reference.DSparkWeights:
    d2t = (0, 2, 5, 7, 9)
    t2d = tuple(index in d2t for index in range(CONFIG.target_vocab_size))
    layers = []
    for index in range(CONFIG.num_layers):
        phase = 0.011 + index * 0.013
        layers.append(
            reference.DraftLayerWeights(
                attention=reference.DraftAttentionWeights(
                    q_proj=matrix(CONFIG.query_width, CONFIG.hidden_size, phase),
                    k_proj=matrix(CONFIG.kv_width, CONFIG.hidden_size, phase + 0.003),
                    v_proj=matrix(CONFIG.kv_width, CONFIG.hidden_size, phase + 0.005),
                    o_proj=matrix(CONFIG.hidden_size, CONFIG.query_width, phase + 0.007),
                    q_norm=norm(CONFIG.head_dim, phase + 0.009),
                    k_norm=norm(CONFIG.head_dim, phase + 0.011),
                ),
                input_norm=norm(CONFIG.hidden_size, phase + 0.013),
                post_attention_norm=norm(CONFIG.hidden_size, phase + 0.015),
                gate_proj=matrix(
                    CONFIG.intermediate_size,
                    CONFIG.hidden_size,
                    phase + 0.017,
                ),
                up_proj=matrix(
                    CONFIG.intermediate_size,
                    CONFIG.hidden_size,
                    phase + 0.019,
                ),
                down_proj=matrix(
                    CONFIG.hidden_size,
                    CONFIG.intermediate_size,
                    phase + 0.021,
                ),
            )
        )
    return reference.DSparkWeights(
        d2t=d2t,
        t2d=t2d,
        embedding=matrix(CONFIG.target_vocab_size, CONFIG.hidden_size, 0.017),
        fc=matrix(CONFIG.hidden_size, CONFIG.aux_width, 0.019),
        hidden_norm=norm(CONFIG.hidden_size, 0.023),
        layers=tuple(layers),
        norm=norm(CONFIG.hidden_size, 0.029),
        lm_head=matrix(CONFIG.draft_vocab_size, CONFIG.hidden_size, 0.031),
        markov_w1=matrix(CONFIG.target_vocab_size, CONFIG.markov_rank, 0.037),
        markov_w2=matrix(CONFIG.draft_vocab_size, CONFIG.markov_rank, 0.041),
        confidence_weight=tuple(
            math.cos((index + 1) * 0.043) * 0.13
            for index in range(CONFIG.hidden_size + CONFIG.markov_rank)
        ),
        confidence_bias=-0.07,
    )


def mlx_weights(
    weights: reference.DSparkWeights,
    dtype: mx.Dtype = mx.float32,
) -> mlx_dspark.MLXDSparkWeights:
    layers = tuple(
        mlx_dspark.MLXDraftLayerWeights(
            attention=mlx_dspark.MLXDraftAttentionWeights(
                q_proj=mx.array(layer.attention.q_proj, dtype=dtype),
                k_proj=mx.array(layer.attention.k_proj, dtype=dtype),
                v_proj=mx.array(layer.attention.v_proj, dtype=dtype),
                o_proj=mx.array(layer.attention.o_proj, dtype=dtype),
                q_norm=mx.array(layer.attention.q_norm, dtype=dtype),
                k_norm=mx.array(layer.attention.k_norm, dtype=dtype),
            ),
            input_norm=mx.array(layer.input_norm, dtype=dtype),
            post_attention_norm=mx.array(layer.post_attention_norm, dtype=dtype),
            gate_proj=mx.array(layer.gate_proj, dtype=dtype),
            up_proj=mx.array(layer.up_proj, dtype=dtype),
            down_proj=mx.array(layer.down_proj, dtype=dtype),
        )
        for layer in weights.layers
    )
    return mlx_dspark.MLXDSparkWeights(
        d2t=mx.array(weights.d2t, dtype=mx.int64),
        t2d=mx.array(weights.t2d, dtype=mx.bool_),
        embedding=mx.array(weights.embedding, dtype=dtype),
        fc=mx.array(weights.fc, dtype=dtype),
        hidden_norm=mx.array(weights.hidden_norm, dtype=dtype),
        layers=layers,
        norm=mx.array(weights.norm, dtype=dtype),
        lm_head=mx.array(weights.lm_head, dtype=dtype),
        markov_w1=mx.array(weights.markov_w1, dtype=dtype),
        markov_w2=mx.array(weights.markov_w2, dtype=dtype),
        confidence_weight=mx.array(weights.confidence_weight, dtype=dtype),
        confidence_bias=mx.array([weights.confidence_bias], dtype=dtype),
    )


def auxiliary_states() -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                math.sin((layer + 1) * 0.17 + (token * CONFIG.hidden_size + column) * 0.09)
                * 0.31
                for column in range(CONFIG.hidden_size)
            )
            for token in range(3)
        )
        for layer in range(len(CONFIG.aux_hidden_state_indices))
    )


def auxiliary_slice(
    states: tuple[tuple[tuple[float, ...], ...], ...],
    start: int,
    end: int,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(tuple(layer[start:end]) for layer in states)


def product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def write_safetensors(path: Path, *, extra_tensor: bool = False) -> None:
    specs = mlx_dspark.expected_tensor_specs(CONFIG)
    d2t = tuple(range(CONFIG.draft_vocab_size))
    t2d = bytes(
        1 if index < CONFIG.draft_vocab_size else 0
        for index in range(CONFIG.target_vocab_size)
    )
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    chunks = []
    offset = 0
    for name, (dtype, shape) in specs.items():
        elements = product(shape)
        if dtype == "BF16":
            payload = bytes(elements * 2)
        elif dtype == "I64":
            payload = struct.pack(f"<{elements}q", *d2t)
        elif dtype == "BOOL":
            payload = t2d
        else:
            raise AssertionError(dtype)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        chunks.append(payload)
        offset += len(payload)
    if extra_tensor:
        header["unexpected.weight"] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [offset, offset + 2],
        }
        chunks.append(b"\0\0")
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


class MLXDSparkTest(unittest.TestCase):
    def assert_array_close(
        self,
        actual: mx.array,
        expected: object,
        tolerance: float = 3e-5,
    ) -> None:
        expected_array = mx.array(expected, dtype=mx.float32)
        mx.eval(actual, expected_array)
        self.assertEqual(actual.shape, expected_array.shape)
        error = float(mx.max(mx.abs(actual.astype(mx.float32) - expected_array)).item())
        self.assertLessEqual(error, tolerance)

    def test_standard_qwen3_norm_and_full_head_rope(self) -> None:
        actual = mlx_dspark.qwen3_rms_norm(
            mx.array([3.0, 4.0]),
            mx.array([1.0, 2.0]),
            1e-6,
        )
        denominator = math.sqrt(12.5 + 1e-6)
        self.assert_array_close(actual, [3.0 / denominator, 8.0 / denominator])

        config = replace(
            CONFIG,
            num_q_heads=1,
            num_kv_heads=1,
            head_dim=4,
            rotary_dim=4,
        )
        vector = (0.2, -0.3, 0.4, -0.5)
        rotated = reference.apply_rope(vector, 7, config)
        self.assertNotAlmostEqual(rotated[2], vector[2])
        self.assertNotAlmostEqual(rotated[3], vector[3])

    def test_incremental_context_and_proposal_match_scalar_oracle(self) -> None:
        scalar = scalar_weights()
        gpu = mlx_weights(scalar)
        reference.validate_weights(scalar, CONFIG)
        mlx_dspark.validate_weights(gpu, CONFIG)
        auxiliary = auxiliary_states()

        expected_state = reference.initial_context(CONFIG)
        actual_state = mlx_dspark.initial_context(CONFIG, mx.float32)
        for start, end in ((0, 2), (2, 3)):
            expected_state = reference.append_context(
                expected_state,
                auxiliary_slice(auxiliary, start, end),
                scalar,
                CONFIG,
            )
            actual_state = mlx_dspark.append_context(
                actual_state,
                tuple(
                    mx.array(layer[start:end], dtype=mx.float32)
                    for layer in auxiliary
                ),
                gpu,
                CONFIG,
                _validated=True,
            )
        self.assertEqual(actual_state.position, expected_state.position)
        for index in range(CONFIG.num_layers):
            expected_keys = mx.array(expected_state.keys[index], dtype=mx.float32).reshape(
                expected_state.position,
                CONFIG.num_kv_heads,
                CONFIG.head_dim,
            )
            expected_values = mx.array(expected_state.values[index], dtype=mx.float32).reshape(
                expected_state.position,
                CONFIG.num_kv_heads,
                CONFIG.head_dim,
            )
            self.assert_array_close(actual_state.keys[index], expected_keys)
            self.assert_array_close(actual_state.values[index], expected_values)

        anchor = 7
        expected = reference.propose(anchor, expected_state, scalar, CONFIG)
        actual = mlx_dspark.propose(
            mx.array(anchor, dtype=mx.int64),
            actual_state,
            gpu,
            CONFIG,
            _validated=True,
        )
        mx.eval(
            actual.target_token_ids,
            actual.draft_token_ids,
            actual.confidence,
            actual.hidden_states,
            actual.base_logits,
            actual.corrected_logits,
        )
        self.assertEqual(tuple(actual.target_token_ids.tolist()), expected.target_token_ids)
        self.assertEqual(tuple(actual.draft_token_ids.tolist()), expected.draft_token_ids)
        self.assert_array_close(actual.confidence, expected.confidence)
        self.assert_array_close(actual.hidden_states, expected.hidden_states)
        self.assert_array_close(actual.base_logits, expected.base_logits)
        self.assert_array_close(actual.corrected_logits, expected.corrected_logits)

        previous = anchor
        for slot, corrected in enumerate(expected.corrected_logits):
            bias = reference.matvec(scalar.markov_w2, scalar.markov_w1[previous])
            observed = tuple(
                corrected[index] - expected.base_logits[slot][index]
                for index in range(CONFIG.draft_vocab_size)
            )
            for left, right in zip(observed, bias):
                self.assertAlmostEqual(left, right, places=12)
            previous = expected.target_token_ids[slot]

    def test_single_owner_context_matches_immutable_context_and_proposal(self) -> None:
        weights = mlx_weights(scalar_weights(), mx.bfloat16)
        auxiliary = auxiliary_states()
        immutable = mlx_dspark.initial_context(CONFIG, mx.bfloat16)
        linear = mlx_dspark.initial_linear_context(CONFIG, capacity=12)
        original_keys = linear.keys
        original_values = linear.values
        stale_linear = linear
        for start, end in ((0, 2), (2, 3)):
            chunk = tuple(
                mx.array(layer[start:end], dtype=mx.bfloat16)
                for layer in auxiliary
            )
            immutable = mlx_dspark.append_context(
                immutable,
                chunk,
                weights,
                CONFIG,
                _validated=True,
            )
            stale_linear = linear
            linear = mlx_dspark.append_context(
                linear,
                chunk,
                weights,
                CONFIG,
                _validated=True,
            )
        with self.assertRaisesRegex(mlx_dspark.MLXDSparkError, "stale"):
            mlx_dspark.validate_linear_context(stale_linear, CONFIG, mx.bfloat16)
        with self.assertRaisesRegex(mlx_dspark.MLXDSparkError, "stale"):
            mlx_dspark.propose(7, stale_linear, weights, CONFIG, _validated=True)
        with self.assertRaisesRegex(mlx_dspark.MLXDSparkError, "stale"):
            mlx_dspark.append_context(
                stale_linear,
                tuple(mx.array(layer[:1], dtype=mx.bfloat16) for layer in auxiliary),
                weights,
                CONFIG,
                _validated=True,
            )
        self.assertIsInstance(immutable, mlx_dspark.MLXDSparkContextState)
        self.assertIsInstance(linear, mlx_dspark.MLXDSparkLinearContextState)
        mx.eval(*linear.keys, *linear.values)
        for index in range(CONFIG.num_layers):
            active_keys = mx.transpose(linear.keys[index][:, : linear.position], (1, 0, 2))
            active_values = mx.transpose(
                linear.values[index][:, : linear.position],
                (1, 0, 2),
            )
            self.assertTrue(bool(mx.array_equal(active_keys, immutable.keys[index]).item()))
            self.assertTrue(bool(mx.array_equal(active_values, immutable.values[index]).item()))
            self.assertTrue(
                bool(
                    mx.array_equal(
                        original_keys[index][:, : linear.position],
                        linear.keys[index][:, : linear.position],
                    ).item()
                )
            )
            self.assertTrue(
                bool(
                    mx.array_equal(
                        original_values[index][:, : linear.position],
                        linear.values[index][:, : linear.position],
                    ).item()
                )
            )

        expected = mlx_dspark.propose(7, immutable, weights, CONFIG, _validated=True)
        actual = mlx_dspark.propose(7, linear, weights, CONFIG, _validated=True)
        pairs = (
            (actual.target_token_ids, expected.target_token_ids),
            (actual.draft_token_ids, expected.draft_token_ids),
            (actual.confidence, expected.confidence),
            (actual.hidden_states, expected.hidden_states),
            (actual.base_logits, expected.base_logits),
            (actual.corrected_logits, expected.corrected_logits),
        )
        mx.eval(*(value for pair in pairs for value in pair))
        for left, right in pairs:
            self.assertTrue(bool(mx.array_equal(left, right).item()))

        constrained = mlx_dspark.initial_linear_context(CONFIG, capacity=4)
        constrained = mlx_dspark.append_context(
            constrained,
            tuple(mx.array(layer[:1], dtype=mx.bfloat16) for layer in auxiliary),
            weights,
            CONFIG,
            _validated=True,
        )
        with self.assertRaisesRegex(mlx_dspark.MLXDSparkError, "complete proposal"):
            mlx_dspark.propose(7, constrained, weights, CONFIG, _validated=True)

    def test_strict_loader_accepts_exact_schema_and_rejects_extras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "draft.safetensors"
            write_safetensors(path)
            weights = mlx_dspark.load_weights(path, CONFIG)
            self.assertEqual(weights.embedding.shape, (11, 4))
            self.assertEqual(weights.embedding.dtype, mx.bfloat16)
            self.assertEqual(tuple(weights.d2t.tolist()), tuple(range(5)))

            extra = Path(temporary) / "extra.safetensors"
            write_safetensors(extra, extra_tensor=True)
            with self.assertRaisesRegex(mlx_dspark.MLXDSparkError, "schema"):
                mlx_dspark.load_weights(extra, CONFIG)


if __name__ == "__main__":
    unittest.main(verbosity=2)

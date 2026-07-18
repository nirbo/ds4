#!/usr/bin/env python3
"""Independent scalar and MLX parity checks for Ornith-35 Qwen3.5 MTP."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_attention_reference as attention_reference
import ornith35_mlx_attention as mlx_attention
import ornith35_mlx_mtp as mlx_mtp
import ornith35_mtp_reference as reference
from ornith35_moe_reference import MoEConfig, MoEError


def matrix(rows: int, columns: int, phase: float) -> list[list[float]]:
    return [
        [
            math.sin((row * columns + column + 1) * phase) * 0.07
            for column in range(columns)
        ]
        for row in range(rows)
    ]


def norm(size: int, phase: float) -> list[float]:
    return [math.cos((index + 1) * phase) * 0.09 for index in range(size)]


def expert(
    config: MoEConfig,
    phase: float,
) -> reference.DenseExpertWeights:
    return reference.DenseExpertWeights(
        gate=matrix(config.intermediate_size, config.hidden_size, phase),
        up=matrix(config.intermediate_size, config.hidden_size, phase + 0.003),
        down=matrix(config.hidden_size, config.intermediate_size, phase + 0.005),
    )


def make_fixture() -> tuple[reference.MTPConfig, reference.MTPWeights]:
    attention_config = attention_reference.AttentionConfig(
        hidden_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=4,
        rotary_dim=4,
        rope_theta=10_000.0,
    )
    moe_config = MoEConfig(
        hidden_size=16,
        intermediate_size=16,
        num_experts=4,
        top_k=2,
    )
    config = reference.MTPConfig(
        hidden_size=16,
        attention=attention_config,
        moe=moe_config,
    )
    attention_weights = attention_reference.AttentionWeights(
        q_proj=matrix(attention_config.query_dim * 2, 16, 0.011),
        k_proj=matrix(attention_config.kv_dim, 16, 0.013),
        v_proj=matrix(attention_config.kv_dim, 16, 0.017),
        o_proj=matrix(16, attention_config.query_dim, 0.019),
        q_norm=norm(attention_config.head_dim, 0.023),
        k_norm=norm(attention_config.head_dim, 0.029),
    )
    moe_weights = reference.DenseMoEWeights(
        router=matrix(moe_config.num_experts, 16, 0.031),
        experts=tuple(
            expert(moe_config, 0.037 + index * 0.011)
            for index in range(moe_config.num_experts)
        ),
        shared_expert=expert(moe_config, 0.089),
        shared_gate=norm(16, 0.097),
    )
    return config, reference.MTPWeights(
        fc=matrix(16, 32, 0.007),
        pre_fc_norm_embedding=norm(16, 0.101),
        pre_fc_norm_hidden=norm(16, 0.103),
        input_layernorm=norm(16, 0.107),
        attention=attention_weights,
        moe=moe_weights,
        post_attention_layernorm=norm(16, 0.109),
        norm=norm(16, 0.113),
    )


def mlx_expert(
    weights: reference.DenseExpertWeights,
    dtype: mx.Dtype = mx.float32,
) -> mlx_mtp.DenseExpertArrays:
    return mlx_mtp.DenseExpertArrays(
        gate=mx.array(weights.gate, dtype=dtype),
        up=mx.array(weights.up, dtype=dtype),
        down=mx.array(weights.down, dtype=dtype),
    )


def mlx_weights(
    weights: reference.MTPWeights,
    dtype: mx.Dtype = mx.float32,
) -> mlx_mtp.MLXMTPWeights:
    experts = tuple(mlx_expert(value, dtype) for value in weights.moe.experts)
    router = mx.array(weights.moe.router, dtype=dtype)
    shared_gate = mx.array([weights.moe.shared_gate], dtype=dtype)
    return mlx_mtp.MLXMTPWeights(
        fc=mx.array(weights.fc, dtype=dtype),
        pre_fc_norm_embedding=mx.array(
            weights.pre_fc_norm_embedding,
            dtype=dtype,
        ),
        pre_fc_norm_hidden=mx.array(weights.pre_fc_norm_hidden, dtype=dtype),
        input_layernorm=mx.array(weights.input_layernorm, dtype=dtype),
        attention=mlx_attention.MLXAttentionWeights(
            q_proj=mx.array(weights.attention.q_proj, dtype=dtype),
            k_proj=mx.array(weights.attention.k_proj, dtype=dtype),
            v_proj=mx.array(weights.attention.v_proj, dtype=dtype),
            o_proj=mx.array(weights.attention.o_proj, dtype=dtype),
            q_norm=mx.array(weights.attention.q_norm, dtype=dtype),
            k_norm=mx.array(weights.attention.k_norm, dtype=dtype),
        ),
        moe=mlx_mtp.MLXDenseMoEWeights(
            router_shared=mx.concatenate((router, shared_gate), axis=0),
            experts=mlx_mtp.DenseExpertStack(
                gate=mx.stack([value.gate for value in experts]),
                up=mx.stack([value.up for value in experts]),
                down=mx.stack([value.down for value in experts]),
            ),
            shared_expert=mlx_expert(weights.moe.shared_expert, dtype),
        ),
        post_attention_layernorm=mx.array(
            weights.post_attention_layernorm,
            dtype=dtype,
        ),
        norm=mx.array(weights.norm, dtype=dtype),
    )


def flatten(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [value]


def synthetic_extraction_state() -> dict:
    names = set(mlx_mtp.expected_tensor_shapes())
    first = {
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
    }
    second = names - first

    def shard(names_for_shard, count, payload, source_hash):
        return {
            "tensor_count": count,
            "payload_bytes": payload,
            "source_sha256": source_hash,
            "tensor_sha256": {name: "0" * 64 for name in names_for_shard},
        }

    return {
        "format": mlx_mtp.STATE_FORMAT,
        "status": "complete",
        "profile": "mtp-source",
        "repository": mlx_mtp.EXPECTED_REPOSITORY,
        "revision": mlx_mtp.EXPECTED_REVISION,
        "runtime_revision": mlx_mtp.EXPECTED_RUNTIME_REVISION,
        "metadata_state_sha256": mlx_mtp.EXPECTED_METADATA_STATE_SHA256,
        "tool_sha256": mlx_mtp.EXPECTED_TOOL_SHA256,
        "output_sha256": mlx_mtp.EXPECTED_SIDECAR_SHA256,
        "output": {
            "name": "mtp.safetensors",
            "bytes": mlx_mtp.EXPECTED_SIDECAR_BYTES,
            "header_bytes": mlx_mtp.EXPECTED_HEADER_BYTES,
            "header_sha256": mlx_mtp.EXPECTED_HEADER_SHA256,
            "payload_bytes": mlx_mtp.EXPECTED_PAYLOAD_BYTES,
            "tensor_count": mlx_mtp.EXPECTED_TENSOR_COUNT,
        },
        "completed_shards": {
            "model.safetensors-00013-of-00014.safetensors": shard(
                first,
                3,
                67_108_864,
                mlx_mtp.EXPECTED_SHARDS[
                    "model.safetensors-00013-of-00014.safetensors"
                ][2],
            ),
            "model.safetensors-00014-of-00014.safetensors": shard(
                second,
                782,
                1_622_172_672,
                mlx_mtp.EXPECTED_SHARDS[
                    "model.safetensors-00014-of-00014.safetensors"
                ][2],
            ),
        },
    }


class MLXMTPTest(unittest.TestCase):
    def test_batched_dense_moe_matches_independent_tokens(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        hidden = mx.array(
            [
                [
                    math.sin((token * config.hidden_size + index + 1) * 0.137)
                    * 0.2
                    for index in range(config.hidden_size)
                ]
                for token in range(3)
            ],
            dtype=mx.float32,
        )
        actual = mlx_mtp.forward_moe_batch(hidden, weights.moe, config.moe)
        independent = [
            mlx_mtp.forward_moe(row, weights.moe, config.moe)
            for row in hidden
        ]
        expected_output = mx.stack([value.output for value in independent])
        expected_selected = mx.stack(
            [value.selected_experts for value in independent]
        )
        expected_routing = mx.stack(
            [value.routing_weights for value in independent]
        )
        mx.eval(
            actual.output,
            actual.selected_experts,
            actual.routing_weights,
            expected_output,
            expected_selected,
            expected_routing,
        )
        self.assertTrue(
            bool(mx.array_equal(actual.selected_experts, expected_selected).item())
        )
        self.assertTrue(
            bool(mx.allclose(actual.routing_weights, expected_routing, atol=1e-6).item())
        )
        self.assertTrue(
            bool(mx.allclose(actual.output, expected_output, atol=1e-5).item())
        )

    def test_causal_chunk_matches_independent_advancing_steps(self) -> None:
        config, scalar_weights = make_fixture()
        weights = mlx_weights(scalar_weights)
        embeddings = mx.array(
            [
                [
                    math.sin((token * config.hidden_size + index + 1) * 0.127)
                    * 0.3
                    for index in range(config.hidden_size)
                ]
                for token in range(3)
            ],
            dtype=mx.float32,
        )
        target_hidden = mx.array(
            [
                [
                    math.cos((token * config.hidden_size + index + 1) * 0.131)
                    * 0.25
                    for index in range(config.hidden_size)
                ]
                for token in range(3)
            ],
            dtype=mx.float32,
        )
        state = mlx_attention.zeros_state(config.attention, dtype=mx.float32)
        independent = []
        serial_state = state
        for token in range(3):
            result = mlx_mtp.forward_step(
                embeddings[token],
                target_hidden[token],
                serial_state,
                weights,
                config,
            )
            independent.append(result)
            serial_state = result.state
        expected_hidden = mx.stack([value.hidden for value in independent])
        expected_selected = mx.stack(
            [value.selected_experts for value in independent]
        )
        actual = mlx_mtp.prefill_steps(
            embeddings,
            target_hidden,
            state,
            weights,
            config,
        )
        mx.eval(
            actual.hidden,
            actual.state.keys,
            actual.state.values,
            actual.selected_experts,
            expected_hidden,
            serial_state.keys,
            serial_state.values,
            expected_selected,
        )
        self.assertTrue(
            bool(mx.array_equal(actual.selected_experts, expected_selected).item())
        )
        self.assertTrue(bool(mx.allclose(actual.hidden, expected_hidden, atol=1e-4).item()))
        self.assertTrue(
            bool(mx.allclose(actual.state.keys, serial_state.keys, atol=1e-5).item())
        )
        self.assertTrue(
            bool(mx.allclose(actual.state.values, serial_state.values, atol=1e-5).item())
        )

    def test_production_schema_is_exact_and_payload_complete(self) -> None:
        shapes = mlx_mtp.expected_tensor_shapes()
        self.assertEqual(len(shapes), mlx_mtp.EXPECTED_TENSOR_COUNT)
        self.assertEqual(
            sum(2 * math.prod(shape) for shape in shapes.values()),
            mlx_mtp.EXPECTED_PAYLOAD_BYTES,
        )

    def test_extraction_state_rejects_identity_drift(self) -> None:
        state = synthetic_extraction_state()
        mlx_mtp._validate_extraction_state(state)
        drifted = copy.deepcopy(state)
        drifted["revision"] = "f" * 40
        with self.assertRaisesRegex(MoEError, "revision mismatch"):
            mlx_mtp._validate_extraction_state(drifted)
        drifted = copy.deepcopy(state)
        del drifted["completed_shards"][
            "model.safetensors-00014-of-00014.safetensors"
        ]["tensor_sha256"]["mtp.norm.weight"]
        with self.assertRaisesRegex(MoEError, "hash count mismatch"):
            mlx_mtp._validate_extraction_state(drifted)

    def test_installed_sidecar_metadata_is_accepted_without_payload_read(self) -> None:
        sidecar = mlx_mtp.DEFAULT_ROOT / "source-mtp" / "mtp.safetensors"
        if not sidecar.exists():
            self.skipTest("production MTP sidecar is not installed")
        self.assertEqual(
            mlx_mtp.require_verified_mtp_sidecar(
                mlx_mtp.DEFAULT_ROOT,
                verify_hash=False,
            ),
            sidecar,
        )

    def test_two_advancing_steps_match_independent_scalar_oracle(self) -> None:
        config, scalar_weights = make_fixture()
        gpu_weights = mlx_weights(scalar_weights)
        scalar_state = attention_reference.zeros_state(config.attention)
        gpu_state = mlx_attention.zeros_state(config.attention, dtype=mx.float32)

        for step in range(2):
            embedding = [
                math.sin((step * config.hidden_size + index + 1) * 0.127) * 0.3
                for index in range(config.hidden_size)
            ]
            target_hidden = [
                math.cos((step * config.hidden_size + index + 1) * 0.131) * 0.25
                for index in range(config.hidden_size)
            ]
            expected = reference.forward_step(
                embedding,
                target_hidden,
                scalar_state,
                scalar_weights,
                config,
            )
            actual = mlx_mtp.forward_step(
                mx.array(embedding, dtype=mx.float32),
                mx.array(target_hidden, dtype=mx.float32),
                gpu_state,
                gpu_weights,
                config,
            )
            mx.eval(
                actual.hidden,
                actual.state.keys,
                actual.state.values,
                actual.selected_experts,
                actual.routing_weights,
            )
            self.assertEqual(
                tuple(int(value) for value in actual.selected_experts.tolist()),
                expected.selected_experts,
            )
            for left, right in zip(
                actual.routing_weights.tolist(),
                expected.routing_weights,
            ):
                self.assertAlmostEqual(left, right, places=5)
            for left, right in zip(actual.hidden.tolist(), expected.hidden):
                self.assertAlmostEqual(left, right, places=4)
            for left, right in zip(
                flatten(actual.state.keys.tolist()),
                flatten(expected.state.keys),
            ):
                self.assertAlmostEqual(left, right, places=4)
            for left, right in zip(
                flatten(actual.state.values.tolist()),
                flatten(expected.state.values),
            ):
                self.assertAlmostEqual(left, right, places=4)
            scalar_state = expected.state
            gpu_state = actual.state


if __name__ == "__main__":
    unittest.main()

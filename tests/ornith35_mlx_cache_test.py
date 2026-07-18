#!/usr/bin/env python3
"""Atomic persistence and exact restore checks for Ornith-35 prefix state."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
TESTS = ROOT / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import ornith35_mlx_attention as attention
import ornith35_mlx_cache as cache
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
import ornith35_mlx_model_test as model_fixture
import ornith35_mlx_mtp_runtime as mtp_runtime
import ornith35_mlx_mtp_test as mtp_fixture
from ornith35_moe_reference import MoEError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def identity(name: str = "source", *, use_mtp: bool = False) -> cache.CacheIdentity:
    return cache.CacheIdentity(
        model_id="AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4",
        model_revision="85ffd2d0629ae5fa4f860dda356ec33161806c9b",
        source_sha256=digest("source"),
        runtime_revision="test-runtime",
        runtime_sha256=digest(name),
        tokenizer_sha256=digest("tokenizer"),
        chat_template_sha256=digest("template"),
        quantization_policy_sha256=digest("source-nvfp4"),
        rope_profile="native-262k",
        cache_dtype="BF16",
        mtp_profile=cache.MTP_PROFILE_FOLDED if use_mtp else cache.MTP_PROFILE_NONE,
        mtp_policy_sha256=digest("mtp-policy") if use_mtp else cache.MTP_NONE_POLICY_SHA256,
    )


def compare_state(
    checked: model.TextModelState,
    expected: model.TextModelState,
) -> None:
    if checked.position != expected.position:
        raise AssertionError("state position mismatch")
    for left, right in zip(checked.layers, expected.layers):
        if isinstance(right, gdn.MLXGDNState):
            if not bool(mx.array_equal(left.conv, right.conv).item()):
                raise AssertionError("convolution state mismatch")
            if not bool(mx.array_equal(left.recurrent, right.recurrent).item()):
                raise AssertionError("recurrent state mismatch")
        else:
            if isinstance(right, attention.MLXLinearAttentionState):
                right_keys = right.keys[:, : right.position]
                right_values = right.values[:, : right.position]
            else:
                right_keys = right.keys
                right_values = right.values
            if not bool(mx.array_equal(left.keys, right_keys).item()):
                raise AssertionError("key state mismatch")
            if not bool(mx.array_equal(left.values, right_values).item()):
                raise AssertionError("value state mismatch")


class MLXCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config, self.weights = model_fixture.make_bf16_fixture()
        initial = model.initial_state(self.weights, self.config)
        self.tokens = (7, 19, 11)
        self.transition = model.prefill_hidden_chunk(
            self.tokens,
            initial,
            self.weights,
            self.config,
            use_steel=False,
        )
        model.evaluate_chunk_transition(self.transition)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_round_trip_preserves_state_and_next_logits_exactly(self) -> None:
        path = cache.save_cache(
            self.root,
            self.tokens,
            self.transition.state,
            identity(),
            self.config,
        )
        restored = cache.load_cache(
            path,
            identity(),
            self.config,
            expected_tokens=self.tokens,
        )
        compare_state(restored.state, self.transition.state)
        self.assertEqual(restored.token_ids, self.tokens)
        self.assertEqual(restored.key, path.name)
        self.assertEqual(
            {entry.name for entry in path.iterdir()},
            {
                "manifest.json",
                "tokens.u32le",
                "layer-000.safetensors",
                "layer-001.safetensors",
            },
        )

        expected = model.forward_token(5, self.transition.state, self.weights, self.config)
        actual = model.forward_token(5, restored.state, self.weights, self.config)
        model.evaluate_result(expected)
        model.evaluate_result(actual)
        self.assertTrue(bool(mx.array_equal(actual.logits, expected.logits).item()))
        compare_state(actual.state, expected.state)

        duplicate = cache.save_cache(
            self.root,
            self.tokens,
            self.transition.state,
            identity(),
            self.config,
        )
        self.assertEqual(duplicate, path)

    def test_linear_state_is_compacted_to_immutable_active_prefix(self) -> None:
        initial = model.initial_state(self.weights, self.config)
        session = model.start_linear_decode_session(
            self.weights,
            initial,
            8,
            self.config,
        )
        model.prefill_linear_session_chunk(
            self.tokens,
            session,
            project_logits=False,
            use_steel=False,
        )
        path = cache.save_cache(
            self.root,
            self.tokens,
            session.state,
            identity(),
            self.config,
        )
        restored = cache.load_cache(path, identity(), self.config)
        compare_state(restored.state, session.state)
        attention_state = restored.state.layers[1]
        self.assertIsInstance(attention_state, attention.MLXAttentionState)
        self.assertEqual(attention_state.keys.shape[1], len(self.tokens))

    def test_mtp_prefix_round_trip_preserves_boundary_and_compacts_kv(self) -> None:
        mtp_config, scalar_weights = mtp_fixture.make_fixture()
        mtp_weights = mtp_fixture.mlx_weights(scalar_weights, mx.bfloat16)
        mtp_state = mtp_runtime.initial_context_state(
            mtp_weights,
            mtp_config,
            capacity=8,
        )
        context = mtp_runtime.append_authoritative_hidden(
            mtp_state,
            self.transition.hidden[:-1],
            self.tokens[1:],
            self.weights.embedding,
            mtp_weights,
            mtp_config,
            _validated=True,
        )
        prefix = mtp_runtime.MTPPrefixState(
            state=context.state,
            boundary_hidden=self.transition.hidden[-1],
        )
        path = cache.save_cache(
            self.root,
            self.tokens,
            self.transition.state,
            identity(use_mtp=True),
            self.config,
            mtp_prefix=prefix,
            mtp_config=mtp_config,
        )
        restored = cache.load_cache(
            path,
            identity(use_mtp=True),
            self.config,
            mtp_config=mtp_config,
        )
        self.assertIsNotNone(restored.mtp_prefix)
        restored_prefix = restored.mtp_prefix
        self.assertIsInstance(restored_prefix.state, attention.MLXAttentionState)
        self.assertEqual(restored_prefix.state.keys.shape[1], len(self.tokens) - 1)
        self.assertTrue(
            bool(
                mx.array_equal(
                    restored_prefix.state.keys,
                    context.state.keys[:, : len(self.tokens) - 1],
                ).item()
            )
        )
        self.assertTrue(
            bool(
                mx.array_equal(
                    restored_prefix.state.values,
                    context.state.values[:, : len(self.tokens) - 1],
                ).item()
            )
        )
        self.assertTrue(
            bool(
                mx.array_equal(
                    restored_prefix.boundary_hidden,
                    self.transition.hidden[-1],
                ).item()
            )
        )
        self.assertIn(cache.MTP_PREFIX_NAME, {entry.name for entry in path.iterdir()})

        mtp_path = path / cache.MTP_PREFIX_NAME
        with mtp_path.open("r+b") as handle:
            handle.seek(-1, 2)
            value = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(MoEError, "MTP prefix hash mismatch"):
            cache.load_cache(
                path,
                identity(use_mtp=True),
                self.config,
                mtp_config=mtp_config,
            )

    def test_one_token_mtp_prefix_persists_empty_kv(self) -> None:
        mtp_config, scalar_weights = mtp_fixture.make_fixture()
        mtp_weights = mtp_fixture.mlx_weights(scalar_weights, mx.bfloat16)
        initial = model.initial_state(self.weights, self.config)
        transition = model.forward_hidden_token(
            self.tokens[0],
            initial,
            self.weights,
            self.config,
        )
        model.evaluate_transition(transition)
        prefix = mtp_runtime.MTPPrefixState(
            state=mtp_runtime.initial_context_state(mtp_weights, mtp_config),
            boundary_hidden=transition.hidden,
        )
        path = cache.save_cache(
            self.root,
            self.tokens[:1],
            transition.state,
            identity(use_mtp=True),
            self.config,
            mtp_prefix=prefix,
            mtp_config=mtp_config,
        )
        restored = cache.load_cache(
            path,
            identity(use_mtp=True),
            self.config,
            mtp_config=mtp_config,
        )
        self.assertEqual(restored.mtp_prefix.state.keys.shape[1], 0)

    def test_mtp_identity_requires_exactly_one_mtp_payload(self) -> None:
        mtp_config, scalar_weights = mtp_fixture.make_fixture()
        mtp_weights = mtp_fixture.mlx_weights(scalar_weights, mx.bfloat16)
        prefix = mtp_runtime.MTPPrefixState(
            state=mtp_runtime.initial_context_state(mtp_weights, mtp_config),
            boundary_hidden=self.transition.hidden[-1],
        )
        with self.assertRaisesRegex(MoEError, "identity and MTP prefix presence disagree"):
            cache.save_cache(
                self.root,
                self.tokens,
                self.transition.state,
                identity(use_mtp=True),
                self.config,
                mtp_config=mtp_config,
            )
        with self.assertRaisesRegex(MoEError, "identity and MTP prefix presence disagree"):
            cache.save_cache(
                self.root,
                self.tokens,
                self.transition.state,
                identity(),
                self.config,
                mtp_prefix=prefix,
                mtp_config=mtp_config,
            )

    def test_rejects_identity_prefix_and_payload_corruption(self) -> None:
        path = cache.save_cache(
            self.root,
            self.tokens,
            self.transition.state,
            identity(),
            self.config,
        )
        with self.assertRaisesRegex(MoEError, "identity mismatch"):
            cache.load_cache(path, identity("candidate"), self.config)
        with self.assertRaisesRegex(MoEError, "token prefix mismatch"):
            cache.load_cache(
                path,
                identity(),
                self.config,
                expected_tokens=(7, 19, 12),
            )

        layer_path = path / "layer-001.safetensors"
        with layer_path.open("r+b") as handle:
            handle.seek(-1, 2)
            value = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(MoEError, "layer hash mismatch"):
            cache.load_cache(path, identity(), self.config)

    def test_failed_write_never_publishes_or_leaves_staging(self) -> None:
        with (
            mock.patch.object(cache.mx, "save_safetensors", side_effect=RuntimeError("write failed")),
            self.assertRaisesRegex(RuntimeError, "write failed"),
        ):
            cache.save_cache(
                self.root,
                self.tokens,
                self.transition.state,
                identity(),
                self.config,
            )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_lru_pruning_removes_oldest_unprotected_complete_entry(self) -> None:
        first = cache.save_cache(
            self.root,
            self.tokens,
            self.transition.state,
            identity(),
            self.config,
        )
        initial = model.initial_state(self.weights, self.config)
        second_tokens = (7, 19)
        second_transition = model.prefill_hidden_chunk(
            second_tokens,
            initial,
            self.weights,
            self.config,
            use_steel=False,
        )
        model.evaluate_chunk_transition(second_transition)
        second = cache.save_cache(
            self.root,
            second_tokens,
            second_transition.state,
            identity(),
            self.config,
        )
        os.utime(first, ns=(1_000_000_000, 1_000_000_000))
        os.utime(second, ns=(2_000_000_000, 2_000_000_000))

        result = cache.prune_cache(
            self.root,
            max_bytes=1 << 40,
            max_entries=1,
            protect=(second.name,),
        )
        self.assertEqual(result.removed_entries, 1)
        self.assertFalse(result.over_budget)
        self.assertFalse(first.exists())
        self.assertTrue(second.is_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)

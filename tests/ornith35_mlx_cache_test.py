#!/usr/bin/env python3
"""Atomic persistence and exact restore checks for Ornith-35 prefix state."""

from __future__ import annotations

import ast
import hashlib
import json
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
import ornith35_context as context
import ornith35_mlx_cache as cache
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
import ornith35_mlx_model_test as model_fixture
import ornith35_mlx_mtp_runtime as mtp_runtime
import ornith35_mlx_mtp_test as mtp_fixture
import ornith35_mlx_turboquant_cache as turboquant_cache
from ornith35_moe_reference import MoEError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def identity(
    name: str = "source",
    *,
    use_mtp: bool = False,
    rope_profile: str = context.NATIVE_PROFILE_ID,
    turboquant: bool = False,
) -> cache.CacheIdentity:
    return cache.CacheIdentity(
        model_id="AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4",
        model_revision="85ffd2d0629ae5fa4f860dda356ec33161806c9b",
        source_sha256=digest("source"),
        runtime_revision="test-runtime",
        runtime_sha256=digest(name),
        tokenizer_sha256=digest("tokenizer"),
        chat_template_sha256=digest("template"),
        quantization_policy_sha256=digest(
            "source-nvfp4-turboquant" if turboquant else "source-nvfp4"
        ),
        rope_profile=rope_profile,
        cache_dtype=(
            cache.CACHE_DTYPE_TURBOQUANT
            if turboquant
            else cache.CACHE_DTYPE_BF16
        ),
        mtp_profile=cache.MTP_PROFILE_FOLDED if use_mtp else cache.MTP_PROFILE_NONE,
        mtp_policy_sha256=digest("mtp-policy") if use_mtp else cache.MTP_NONE_POLICY_SHA256,
        state_schema=(
            cache.TURBOQUANT_STATE_SCHEMA
            if turboquant
            else cache.STATE_SCHEMA
        ),
    )


def compare_state(
    checked: model.TextModelState,
    expected: model.TextModelState,
) -> None:
    if checked.position != expected.position:
        raise AssertionError("state position mismatch")
    if checked.context_profile != expected.context_profile:
        raise AssertionError("state context profile mismatch")
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

    def test_runtime_identity_covers_local_import_closure(self) -> None:
        runtime_paths = set(cache.PRODUCTION_RUNTIME_FILES)
        pending = [
            ROOT / relative
            for relative in runtime_paths
            if relative.endswith(".py")
        ]
        checked = set()
        while pending:
            path = pending.pop()
            if path in checked:
                continue
            checked.add(path)
            tree = ast.parse(path.read_text(encoding="ascii"), filename=str(path))
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    modules.add(node.module.split(".")[0])
            for module_name in modules:
                if not module_name.startswith("ornith35_"):
                    continue
                relative = f"ornith35/tools/{module_name}.py"
                candidate = ROOT / relative
                if not candidate.is_file():
                    continue
                self.assertIn(relative, runtime_paths)
                pending.append(candidate)

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
        timing = restored.load_timing
        self.assertGreater(timing.total_s, 0.0)
        self.assertGreaterEqual(timing.manifest_s, 0.0)
        self.assertGreaterEqual(timing.tokens_s, 0.0)
        self.assertGreaterEqual(timing.payload_verify_s, 0.0)
        self.assertGreaterEqual(timing.payload_materialize_s, 0.0)
        self.assertGreaterEqual(timing.finalize_s, 0.0)
        self.assertGreaterEqual(
            timing.total_s,
            timing.manifest_s
            + timing.tokens_s
            + timing.payload_verify_s
            + timing.payload_materialize_s
            + timing.finalize_s,
        )
        self.assertEqual(
            timing.payload_bytes,
            sum(
                entry.stat().st_size
                for entry in path.iterdir()
                if entry.name != cache.MANIFEST_NAME
            ),
        )
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

    def test_yarn_cache_round_trip_and_native_mismatch_are_rejected(self) -> None:
        initial = model.initial_state(
            self.weights,
            self.config,
            context.YARN2_PROFILE_ID,
        )
        transition = model.prefill_hidden_chunk(
            self.tokens,
            initial,
            self.weights,
            self.config,
            use_steel=False,
        )
        model.evaluate_chunk_transition(transition)
        yarn_identity = identity(rope_profile=context.YARN2_PROFILE_ID)
        path = cache.save_cache(
            self.root,
            self.tokens,
            transition.state,
            yarn_identity,
            self.config,
        )
        restored = cache.load_cache(
            path,
            yarn_identity,
            self.config,
            expected_tokens=self.tokens,
        )
        compare_state(restored.state, transition.state)
        self.assertEqual(restored.state.context_profile, context.YARN2_PROFILE_ID)
        self.assertNotEqual(
            path.name,
            cache.cache_key(self.tokens, identity(), self.config),
        )
        with self.assertRaisesRegex(MoEError, "context profiles disagree"):
            cache.save_cache(
                self.root,
                self.tokens,
                transition.state,
                identity(),
                self.config,
            )
        with self.assertRaisesRegex(MoEError, "identity mismatch"):
            cache.load_cache(path, identity(), self.config)

    def test_mtp_identity_rejects_yarn_context(self) -> None:
        with self.assertRaisesRegex(MoEError, "MTP cache requires the native"):
            cache.validate_identity(
                identity(
                    use_mtp=True,
                    rope_profile=context.YARN2_PROFILE_ID,
                )
            )

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


class MLXTurboQuantCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = model.TextModelConfig(
            vocab_size=128,
            hidden_size=model.PRODUCTION_CONFIG.hidden_size,
            layer_types=(model.LAYER_ATTENTION,),
            gdn=model.PRODUCTION_CONFIG.gdn,
            attention=attention.PRODUCTION_CONFIG,
            moe=model.PRODUCTION_CONFIG.moe,
        )
        self.tokens = (7, 19, 11)
        source = mx.arange(2 * len(self.tokens) * 256).reshape(
            2,
            len(self.tokens),
            256,
        )
        self.keys = ((source % 257) - 128).astype(mx.bfloat16) / 256
        self.values = (((source * 17 + 3) % 263) - 131).astype(mx.bfloat16) / 192
        packed = turboquant_cache.compress_bf16_kv(
            self.keys,
            self.values,
            exact_tail=1,
        )
        mx.eval(
            packed.packed_keys,
            packed.key_norms,
            packed.packed_values,
            packed.value_norms,
            packed.exact_keys,
            packed.exact_values,
        )
        self.state = model.TextModelState(
            position=len(self.tokens),
            layers=(packed,),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_round_trip_restores_compact_packed_state_and_resumes_append(self) -> None:
        cache_identity = identity(turboquant=True)
        path = cache.save_cache(
            self.root,
            self.tokens,
            self.state,
            cache_identity,
            self.config,
        )
        restored = cache.load_cache(
            path,
            cache_identity,
            self.config,
            expected_tokens=self.tokens,
        )
        checked = restored.state.layers[0]
        expected = self.state.layers[0]
        self.assertIsInstance(checked, turboquant_cache.MLXPackedMSE4State)
        for name in (
            "packed_keys",
            "key_norms",
            "packed_values",
            "value_norms",
            "exact_keys",
            "exact_values",
        ):
            self.assertTrue(
                bool(mx.array_equal(getattr(checked, name), getattr(expected, name)).item()),
                name,
            )
        checked_kv = turboquant_cache.dequantize_state(checked)
        expected_kv = turboquant_cache.dequantize_state(expected)
        mx.eval(*checked_kv, *expected_kv)
        self.assertTrue(bool(mx.array_equal(checked_kv[0], expected_kv[0]).item()))
        self.assertTrue(bool(mx.array_equal(checked_kv[1], expected_kv[1]).item()))

        linear = turboquant_cache.linearize_state(checked, 8)
        update_source = mx.arange(2 * 256).reshape(2, 1, 256)
        key_update = ((update_source % 251) - 125).astype(mx.bfloat16) / 224
        value_update = (((update_source * 13 + 5) % 269) - 134).astype(mx.bfloat16) / 208
        advanced = turboquant_cache.advance_linear_state(
            linear,
            key_update,
            value_update,
        )
        direct = turboquant_cache.compress_bf16_kv(
            mx.concatenate((self.keys, key_update), axis=1),
            mx.concatenate((self.values, value_update), axis=1),
            exact_tail=1,
        )
        mx.eval(
            advanced.packed_keys,
            advanced.key_norms,
            advanced.packed_values,
            advanced.value_norms,
            advanced.exact_keys,
            advanced.exact_values,
            direct.packed_keys,
            direct.key_norms,
            direct.packed_values,
            direct.value_norms,
            direct.exact_keys,
            direct.exact_values,
        )
        history = turboquant_cache.packed_history(advanced)
        self.assertEqual(history, turboquant_cache.packed_history(direct))
        for name in ("packed_keys", "key_norms", "packed_values", "value_norms"):
            self.assertTrue(
                bool(
                    mx.array_equal(
                        getattr(advanced, name)[:, :history],
                        getattr(direct, name),
                    ).item()
                ),
                name,
            )
        self.assertTrue(bool(mx.array_equal(advanced.exact_keys, direct.exact_keys).item()))
        self.assertTrue(bool(mx.array_equal(advanced.exact_values, direct.exact_values).item()))

        manifest = json.loads((path / cache.MANIFEST_NAME).read_text(encoding="ascii"))
        self.assertEqual(manifest["schema"], cache.TURBOQUANT_STATE_SCHEMA)
        self.assertEqual(
            set(manifest["files"][0]["tensors"]),
            {
                "packed_keys",
                "key_norms",
                "packed_values",
                "value_norms",
                "exact_keys",
                "exact_values",
            },
        )

    def test_identity_isolation_rejects_mixed_state_formats_and_mtp(self) -> None:
        exact_identity = identity()
        packed_identity = identity(turboquant=True)
        self.assertNotEqual(
            cache.cache_key(self.tokens, exact_identity, self.config),
            cache.cache_key(self.tokens, packed_identity, self.config),
        )
        with self.assertRaisesRegex(MoEError, "attention state mismatch"):
            cache.save_cache(
                self.root,
                self.tokens,
                self.state,
                exact_identity,
                self.config,
            )

        exact_state = model.TextModelState(
            position=len(self.tokens),
            layers=(
                attention.MLXAttentionState(
                    keys=self.keys,
                    values=self.values,
                ),
            ),
        )
        with self.assertRaisesRegex(MoEError, "TurboQuant attention state mismatch"):
            cache.save_cache(
                self.root,
                self.tokens,
                exact_state,
                packed_identity,
                self.config,
            )
        with self.assertRaisesRegex(MoEError, "TurboQuant cache cannot contain MTP"):
            cache.validate_identity(identity(use_mtp=True, turboquant=True))
        with self.assertRaisesRegex(MoEError, "requires the native context profile"):
            cache.validate_identity(
                identity(
                    turboquant=True,
                    rope_profile=context.YARN2_PROFILE_ID,
                )
            )

    def test_packed_payload_corruption_is_rejected(self) -> None:
        cache_identity = identity(turboquant=True)
        path = cache.save_cache(
            self.root,
            self.tokens,
            self.state,
            cache_identity,
            self.config,
        )
        layer_path = path / "layer-000.safetensors"
        with layer_path.open("r+b") as handle:
            handle.seek(-1, 2)
            value = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(MoEError, "layer hash mismatch"):
            cache.load_cache(path, cache_identity, self.config)


if __name__ == "__main__":
    unittest.main(verbosity=2)

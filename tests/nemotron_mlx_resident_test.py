#!/usr/bin/env python3
"""Tests for Nemotron resident-runtime memory preflight."""

from __future__ import annotations

import json
import numpy as np
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "nemotron" / "tools"
sys.path.insert(0, str(TOOLS))
from nemotron_metadata import MetadataError  # noqa: E402
from nemotron_mlx_resident import (  # noqa: E402
    ResidentModel,
    preflight,
    require_extended_run,
    require_runtime_headroom,
    resident_requirement,
    restore_caches,
    save_logits,
    snapshot_caches,
)
from nemotron_prune_materialize import sha256_file  # noqa: E402


class MLXResidentTest(unittest.TestCase):
    def test_sets_experimental_expert_top_k_on_every_moe_layer(self) -> None:
        model = ResidentModel.__new__(ResidentModel)
        model.config = {"num_experts_per_tok": 22}
        model.pattern = "EM*E"
        blocks = [type("Block", (), {"top_k": 22})() for _ in model.pattern]
        model.blocks = blocks

        model.set_expert_top_k(18)

        self.assertEqual(blocks[0].top_k, 18)
        self.assertEqual(blocks[3].top_k, 18)
        self.assertEqual(blocks[1].top_k, 22)
        with self.assertRaisesRegex(MetadataError, "between 1 and 22"):
            model.set_expert_top_k(23)

    def test_saves_atomic_float32_logits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "logits.npy"
            save_logits(path, mx.array([[1.0, 2.0]], dtype=mx.bfloat16))
            values = np.load(path)
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(values.tolist(), [1.0, 2.0])
            self.assertFalse(path.with_name(path.name + ".part.npy").exists())

    def test_reset_recreates_only_stateful_layer_caches(self) -> None:
        model = ResidentModel.__new__(ResidentModel)
        model.pattern = "M*E"
        model.caches = {0: object(), 1: object()}
        model.reset()
        self.assertEqual(set(model.caches), {0, 1})
        self.assertIsInstance(model.caches[0], ArraysCache)
        self.assertIsInstance(model.caches[1], KVCache)

    def test_reset_reuses_allocated_state_storage(self) -> None:
        model = ResidentModel.__new__(ResidentModel)
        model.pattern = "M*"
        recurrent = ArraysCache(size=2)
        recurrent[0] = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
        recurrent[1] = mx.array([[[[3.0, 4.0]]]], dtype=mx.float32)
        attention = KVCache()
        keys = mx.array([[[[5.0, 6.0]]]], dtype=mx.float32)
        values = mx.array([[[[7.0, 8.0]]]], dtype=mx.float32)
        attention.update_and_fetch(keys, values)
        mx.eval(*recurrent.state, attention.keys, attention.values)
        key_storage = attention.keys
        value_storage = attention.values
        model.caches = {0: recurrent, 1: attention}

        model.reset()

        self.assertIs(model.caches[0], recurrent)
        self.assertIs(model.caches[1], attention)
        self.assertEqual(recurrent[0].tolist(), [[[0.0, 0.0]]])
        self.assertEqual(recurrent[1].tolist(), [[[[0.0, 0.0]]]])
        self.assertIs(attention.keys, key_storage)
        self.assertIs(attention.values, value_storage)
        self.assertEqual(attention.offset, 0)

    def test_prefill_advances_bounded_chunks_and_returns_final_position(self) -> None:
        model = ResidentModel.__new__(ResidentModel)
        chunks = []

        def forward_sequence(token_ids):
            chunks.append(token_ids)
            values = mx.array(token_ids, dtype=mx.float32).reshape(-1, 1)
            return values, values + 100.0

        model.forward_sequence = forward_sequence
        logits, hidden = model.prefill([1, 2, 3, 4, 5], 2)

        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])
        self.assertEqual(logits.item(), 5.0)
        self.assertEqual(hidden.item(), 105.0)

    def test_prefill_rejects_empty_input_and_invalid_chunk_size(self) -> None:
        model = ResidentModel.__new__(ResidentModel)
        with self.assertRaisesRegex(MetadataError, "at least one token"):
            model.prefill([], 1)
        with self.assertRaisesRegex(MetadataError, "must be positive"):
            model.prefill([1], 0)

    def test_requirement_includes_explicit_margin(self) -> None:
        self.assertEqual(resident_requirement(10 * 2**30, 1.5), int(11.5 * 2**30))

    def test_cache_snapshot_restores_recurrent_state_and_kv_offset(self) -> None:
        recurrent = ArraysCache(size=2)
        recurrent[0] = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
        recurrent[1] = mx.array([[[[3.0, 4.0]]]], dtype=mx.float32)
        attention = KVCache()
        initial_key = mx.array([[[[5.0, 6.0]]]], dtype=mx.float32)
        initial_value = mx.array([[[[7.0, 8.0]]]], dtype=mx.float32)
        attention.update_and_fetch(initial_key, initial_value)
        mx.eval(*recurrent.state, attention.keys, attention.values)

        caches = {0: recurrent, 1: attention}
        snapshot = snapshot_caches(caches)
        recurrent[0] = mx.array([[[9.0, 10.0]]], dtype=mx.float32)
        recurrent[1] = mx.array([[[[11.0, 12.0]]]], dtype=mx.float32)
        attention.update_and_fetch(initial_key + 10.0, initial_value + 10.0)
        mx.eval(*recurrent.state, attention.keys, attention.values)

        restore_caches(caches, snapshot)
        self.assertEqual(recurrent[0].tolist(), [[[1.0, 2.0]]])
        self.assertEqual(recurrent[1].tolist(), [[[[3.0, 4.0]]]])
        self.assertEqual(attention.offset, 1)
        self.assertEqual(
            attention.keys[..., : attention.offset, :].tolist(),
            initial_key.tolist(),
        )
        self.assertEqual(
            attention.values[..., : attention.offset, :].tolist(),
            initial_value.tolist(),
        )

    def test_preflight_accounts_for_quantized_mtp_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            sidecar = root / "sidecar"
            head = root / "head"
            for directory in (model, sidecar, head):
                directory.mkdir()
            (model / "nemotron_mlx_pack_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-runtime-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "payload_bytes": 10 * 2**30,
                    }
                )
            )
            (sidecar / "nemotron_mtp_pack_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-mtp-sidecar-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "payload_bytes": 2 * 2**30,
                    }
                )
            )
            (head / "nemotron_mtp_head_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-mtp-head-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "source_report_sha256": sha256_file(
                            model / "nemotron_mlx_pack_report.json"
                        ),
                        "payload_bytes": 1 * 2**30,
                    }
                )
            )
            device = {
                "max_recommended_working_set_size": 18 * 2**30,
                "memory_size": 32 * 2**30,
            }
            with (
                patch("nemotron_mlx_resident.mx.device_info", return_value=device),
                patch("nemotron_mlx_resident.iogpu_wired_limit_bytes", return_value=0),
            ):
                result = preflight(model, 0.5, sidecar, head)
            self.assertEqual(result["payload_bytes"], 13 * 2**30)
            self.assertEqual(result["mtp_head_payload_gib"], 1.0)
            self.assertEqual(result["required_gib"], 13.5)
            self.assertEqual(result["extended_working_set_gib"], 16.25)
            self.assertTrue(result["safe_to_attempt"])
            self.assertTrue(result["safe_for_extended_run"])
            learned = preflight(
                model,
                0.5,
                sidecar,
                head,
                additional_payload_bytes=256 * 2**20,
            )
            self.assertEqual(learned["payload_bytes"], 13 * 2**30 + 256 * 2**20)
            self.assertEqual(learned["additional_payload_gib"], 0.25)
            self.assertEqual(learned["required_gib"], 13.75)
            with patch(
                "nemotron_mlx_resident.embedding_layout",
                return_value=(model / "global.safetensors", 0, (8, 8), 1 * 2**30),
            ):
                paged = preflight(model, 0.5, sidecar, head, paged_embeddings=True)
            self.assertEqual(paged["payload_gib"], 12.0)
            self.assertEqual(paged["paged_embedding_gib"], 1.0)
            self.assertEqual(paged["required_gib"], 12.5)

    def test_extended_run_requires_allocator_gc_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "nemotron_mlx_pack_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-runtime-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "payload_bytes": 50 * 2**30,
                    }
                )
            )
            device = {
                "max_recommended_working_set_size": 52 * 2**30,
                "memory_size": 64 * 2**30,
            }
            with (
                patch("nemotron_mlx_resident.mx.device_info", return_value=device),
                patch(
                    "nemotron_mlx_resident.iogpu_wired_limit_bytes",
                    return_value=52 * 2**30,
                ),
            ):
                result = preflight(model, 0.5)
            self.assertTrue(result["safe_to_attempt"])
            self.assertLess(result["required_memory_fraction"], 0.8)
            self.assertFalse(result["safe_for_extended_run"])
            self.assertLess(result["allocator_gc_threshold_gib"], 50.5)
            with self.assertRaisesRegex(MetadataError, "extended-run memory guard failed"):
                require_extended_run(result)
            require_extended_run(result, allow_high_memory_risk=True)

    def test_measured_bounded_prefill_can_use_85_percent_physical_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "nemotron_mlx_pack_report.json").write_text(
                json.dumps(
                    {
                        "format": "nemotron-mlx-runtime-v1",
                        "status": "complete",
                        "source_revision": "revision",
                        "payload_bytes": 52 * 2**30,
                    }
                )
            )
            device = {
                "max_recommended_working_set_size": 57 * 2**30,
                "memory_size": 64 * 2**30,
            }
            with (
                patch("nemotron_mlx_resident.mx.device_info", return_value=device),
                patch(
                    "nemotron_mlx_resident.iogpu_wired_limit_bytes",
                    return_value=57 * 2**30,
                ),
            ):
                generic = preflight(model, 0.5)
                bounded = preflight(model, 0.5, transient_gib=1.625)
            self.assertFalse(generic["safe_for_extended_run"])
            self.assertTrue(bounded["safe_for_extended_run"])
            self.assertEqual(bounded["extended_transient_gib"], 1.625)

    def test_live_headroom_guard_accounts_for_active_cache_and_peak(self) -> None:
        memory = {
            "allocator_gc_threshold_bytes": 10 * 2**30,
            "allocator_gc_threshold_gib": 10.0,
        }
        with (
            patch("nemotron_mlx_resident.mx.get_active_memory", return_value=8 * 2**30),
            patch("nemotron_mlx_resident.mx.get_cache_memory", return_value=1 * 2**30),
            patch("nemotron_mlx_resident.mx.get_peak_memory", return_value=9 * 2**30),
        ):
            require_runtime_headroom(memory)
        with (
            patch("nemotron_mlx_resident.mx.get_active_memory", return_value=8 * 2**30),
            patch("nemotron_mlx_resident.mx.get_cache_memory", return_value=1 * 2**30),
            patch(
                "nemotron_mlx_resident.mx.get_peak_memory",
                return_value=int(9.9 * 2**30),
            ),
        ):
            with self.assertRaisesRegex(MetadataError, "live Metal headroom guard failed"):
                require_runtime_headroom(memory)


if __name__ == "__main__":
    unittest.main(verbosity=2)

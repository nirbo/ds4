#!/usr/bin/env python3
"""Cooperative background cache-warm state-machine checks."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_cache as cache
import ornith35_mlx_cache_warm as warm
import ornith35_runtime_coordination as coordination


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def identity() -> cache.CacheIdentity:
    return cache.CacheIdentity(
        model_id=cache.PRODUCTION_MODEL_ID,
        model_revision=cache.PRODUCTION_MODEL_REVISION,
        source_sha256=digest("source"),
        runtime_revision="test-runtime",
        runtime_sha256=digest("runtime"),
        tokenizer_sha256=digest("tokenizer"),
        chat_template_sha256=digest("template"),
        quantization_policy_sha256=digest("policy"),
        rope_profile="native-262k",
        cache_dtype=cache.CACHE_DTYPE_BF16,
    )


class CacheWarmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        system_file = root / "system.txt"
        system_file.write_text("Stable system prompt.\n", encoding="ascii")
        cache_root = root / "cache"
        key = "a" * 64
        self.spec = warm.WarmSpec(
            root=root,
            repo_root=ROOT,
            system_file=system_file,
            cache_root=cache_root,
            system_sha256=digest("system"),
            system_bytes=22,
            token_ids=tuple(range(100)),
            token_sha256=digest("tokens"),
            identity=identity(),
            final_key=key,
            context_profile="native-262k",
            mapped_embedding=True,
            quantized_lm_head=True,
            prefill_chunk=128,
            checkpoint_tokens=32,
            checkpoint_max_tokens=64,
            progress_tokens=16,
            cache_max_gib=1.0,
            cache_max_bytes=1 << 30,
            poll_seconds=0.05,
            max_wait_seconds=10.0,
            spec_sha256="b" * 64,
            job_dir=cache_root / warm.JOBS_DIRECTORY / key,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_spec_and_state_are_atomic_and_bound(self) -> None:
        warm._write_or_verify_spec(self.spec)
        self.assertEqual(
            warm._read_json(self.spec.job_dir / warm.SPEC_NAME),
            warm._spec_payload(self.spec),
        )
        writer = warm.StateWriter(self.spec)
        state = writer.update(
            "running",
            phase="prefill",
            progress_tokens=16,
            checkpoint_tokens=8,
            checkpoint_key="c" * 64,
        )
        self.assertEqual(state["job_key"], self.spec.final_key)
        self.assertEqual(state["spec_sha256"], self.spec.spec_sha256)
        self.assertEqual(
            warm._read_json(self.spec.job_dir / warm.STATE_NAME)["progress_tokens"],
            16,
        )
        waiting = writer.update("waiting", phase="foreground")
        self.assertEqual(waiting["progress_tokens"], 16)
        self.assertEqual(waiting["checkpoint_tokens"], 8)
        self.assertEqual(waiting["checkpoint_key"], "c" * 64)
        self.assertFalse(any(path.name.startswith(".state.json.part") for path in self.spec.job_dir.iterdir()))

    def test_state_reader_rejects_cross_job_or_invalid_progress(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)
        state = writer.update("running", phase="prefill", progress_tokens=16)
        state["spec_sha256"] = "0" * 64
        warm._atomic_write_json(self.spec.job_dir / warm.STATE_NAME, state)
        with self.assertRaisesRegex(warm.MoEError, "state spec mismatch"):
            warm.StateWriter(self.spec)

        state["spec_sha256"] = self.spec.spec_sha256
        state["progress_tokens"] = len(self.spec.token_ids) + 1
        warm._atomic_write_json(self.spec.job_dir / warm.STATE_NAME, state)
        with self.assertRaisesRegex(warm.MoEError, "state progress is invalid"):
            warm.StateWriter(self.spec)

    def test_invalid_state_cannot_leave_a_worker_claim_active(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)
        state = writer.update("queued", phase="launch")
        state["spec_sha256"] = "0" * 64
        warm._atomic_write_json(self.spec.job_dir / warm.STATE_NAME, state)
        token = "claim-one"
        warm._claim_job(self.spec.job_dir, token, 1, "worker")
        with self.assertRaisesRegex(warm.MoEError, "state spec mismatch"):
            warm._run_claimed(self.spec, token)
        self.assertFalse((self.spec.job_dir / warm.ACTIVE_DIRECTORY).exists())

    def test_build_spec_binds_the_exact_system_file(self) -> None:
        tokenizer = SimpleNamespace(
            encode=lambda _text: (10, 20, 30),
            tokenizer_sha256=digest("tokenizer"),
            template_sha256=digest("template"),
        )
        args = SimpleNamespace(
            root=self.spec.root,
            repo_root=ROOT,
            system_file=self.spec.system_file,
            cache_root=self.spec.cache_root,
            context_profile="native-262k",
            mapped_embedding=True,
            quantized_lm_head=True,
            prefill_chunk=128,
            checkpoint_tokens=32,
            checkpoint_max_tokens=64,
            progress_tokens=16,
            cache_max_gib=1.0,
            poll_seconds=0.05,
            max_wait_seconds=10.0,
        )
        with (
            mock.patch.object(warm, "load_text_tokenizer", return_value=tokenizer),
            mock.patch.object(warm.cache, "production_identity", return_value=identity()),
            mock.patch.object(warm.cache, "cache_key", return_value="d" * 64),
        ):
            built = warm.build_spec(args)
        payload = self.spec.system_file.read_bytes()
        self.assertEqual(built.system_bytes, len(payload))
        self.assertEqual(built.system_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(built.token_ids, (10, 20, 30))
        self.assertEqual(built.final_key, "d" * 64)

    def test_active_claim_is_exclusive_and_releasable(self) -> None:
        token = "claim-one"
        warm._claim_job(self.spec.job_dir, token, 1, "test")
        with self.assertRaisesRegex(warm.WarmError, "already active"):
            warm._claim_job(self.spec.job_dir, "claim-two", 1, "test")
        warm._release_claim(self.spec.job_dir, token)
        self.assertFalse((self.spec.job_dir / warm.ACTIVE_DIRECTORY).exists())

    def test_worker_bootstrap_failure_is_durable_and_releases_claim(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)
        writer.update("queued", phase="launch")
        token = "claim-one"
        warm._claim_job(self.spec.job_dir, token, 1, "launching")
        warm._fail_worker_bootstrap(self.spec.job_dir, token, ValueError("changed"))
        state = warm._read_json(self.spec.job_dir / warm.STATE_NAME)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["phase"], "bootstrap")
        self.assertIn("ValueError: changed", state["error"])
        self.assertFalse((self.spec.job_dir / warm.ACTIVE_DIRECTORY).exists())

    def test_detached_worker_command_binds_parent_job_and_spec(self) -> None:
        command = warm._worker_command(self.spec, "claim-one")
        self.assertEqual(
            command[command.index("--expected-job-key") + 1],
            self.spec.final_key,
        )
        self.assertEqual(
            command[command.index("--expected-spec-sha256") + 1],
            self.spec.spec_sha256,
        )

    def test_checkpoint_schedule_is_bounded(self) -> None:
        self.assertEqual(warm._next_checkpoint(self.spec, 0), 32)
        self.assertEqual(warm._next_checkpoint(self.spec, 32), 64)
        self.assertIsNone(warm._next_checkpoint(self.spec, 64))

    def test_complete_cache_is_strictly_verified_without_loading_model(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)
        path = self.spec.cache_root / self.spec.final_key
        lookup = cache.CacheLookupResult(
            path=path,
            token_count=len(self.spec.token_ids),
            scanned_entries=1,
            compatible_entries=1,
            matching_entries=1,
            elapsed_s=0.001,
        )
        restored = SimpleNamespace()
        with (
            mock.patch.object(warm.cache, "find_longest_prefix", return_value=lookup),
            mock.patch.object(warm.cache, "load_cache", return_value=restored) as load,
            mock.patch.object(warm.model, "load_text_model") as load_model,
            mock.patch.object(warm, "_release_mlx"),
        ):
            self.assertEqual(warm._warm_attempt(self.spec, writer), path)
        load.assert_called_once_with(
            path,
            self.spec.identity,
            warm.model.PRODUCTION_CONFIG,
            expected_tokens=self.spec.token_ids,
        )
        load_model.assert_not_called()
        state = warm._read_json(self.spec.job_dir / warm.STATE_NAME)
        self.assertEqual(state["phase"], "verify-existing")
        self.assertEqual(state["checkpoint_key"], self.spec.final_key)

    def test_preempted_worker_releases_and_resumes_to_completion(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)
        attempts = iter(
            (
                warm.WarmPreempted(24),
                self.spec.cache_root / self.spec.final_key,
            )
        )

        @contextmanager
        def available(_root):
            yield coordination.BackgroundLease(Path("model.lock"))

        def attempt(_spec, _writer):
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            return result

        with (
            mock.patch.object(warm.coordination, "background_lease", available),
            mock.patch.object(warm, "_warm_attempt", side_effect=attempt),
        ):
            self.assertEqual(warm.run_worker(self.spec, writer), 0)
        state = warm._read_json(self.spec.job_dir / warm.STATE_NAME)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["progress_tokens"], len(self.spec.token_ids))

    def test_waiting_worker_does_not_load_until_runtime_is_available(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)
        calls = 0

        @contextmanager
        def deferred_then_available(_root):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise coordination.BackgroundDeferred("foreground generation is pending")
            yield coordination.BackgroundLease(Path("model.lock"))

        with (
            mock.patch.object(
                warm.coordination,
                "background_lease",
                deferred_then_available,
            ),
            mock.patch.object(
                warm,
                "_warm_attempt",
                return_value=self.spec.cache_root / self.spec.final_key,
            ) as attempt,
            mock.patch.object(warm.time, "sleep") as sleep,
        ):
            self.assertEqual(warm.run_worker(self.spec, writer), 0)
        self.assertEqual(calls, 2)
        sleep.assert_called_once_with(self.spec.poll_seconds)
        attempt.assert_called_once()

    def test_deferred_timeout_counts_only_contiguous_waiting(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)

        @contextmanager
        def unavailable(_root):
            raise coordination.BackgroundDeferred("foreground generation is pending")
            yield

        with (
            mock.patch.object(warm.coordination, "background_lease", unavailable),
            mock.patch.object(warm, "_warm_attempt") as attempt,
            mock.patch.object(warm.time, "monotonic", side_effect=(10.0, 10.0, 21.0)),
            mock.patch.object(warm.time, "sleep") as sleep,
        ):
            self.assertEqual(warm.run_worker(self.spec, writer), 75)
        attempt.assert_not_called()
        sleep.assert_called_once_with(self.spec.poll_seconds)
        state = warm._read_json(self.spec.job_dir / warm.STATE_NAME)
        self.assertEqual(state["status"], "deferred-timeout")
        self.assertEqual(state["waited_s"], 11.0)

    def test_cancelled_waiting_job_never_acquires_model(self) -> None:
        warm._write_or_verify_spec(self.spec)
        (self.spec.job_dir / warm.CANCEL_NAME).write_text("cancel\n", encoding="ascii")
        writer = warm.StateWriter(self.spec)
        with mock.patch.object(warm, "_warm_attempt") as attempt:
            self.assertEqual(warm.run_worker(self.spec, writer), 0)
        attempt.assert_not_called()
        state = warm._read_json(self.spec.job_dir / warm.STATE_NAME)
        self.assertEqual(state["status"], "cancelled")

    def test_status_surfaces_a_crashed_worker_without_mutating_state(self) -> None:
        warm._write_or_verify_spec(self.spec)
        writer = warm.StateWriter(self.spec)
        writer.update("running", phase="prefill", progress_tokens=16)
        args = SimpleNamespace(
            root=self.spec.root,
            cache_root=self.spec.cache_root,
            job_key=self.spec.final_key,
        )
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            self.assertEqual(warm._status(args), 0)
        observed = output.getvalue()
        self.assertIn('"observed_status": "interrupted"', observed)
        self.assertIn('"worker": null', observed)
        self.assertEqual(
            warm._read_json(self.spec.job_dir / warm.STATE_NAME)["status"],
            "running",
        )

    def test_cancel_rejects_an_unsafe_existing_marker(self) -> None:
        warm._write_or_verify_spec(self.spec)
        marker = self.spec.job_dir / warm.CANCEL_NAME
        marker.symlink_to(self.spec.system_file)
        args = SimpleNamespace(
            root=self.spec.root,
            cache_root=self.spec.cache_root,
            job_key=self.spec.final_key,
        )
        with self.assertRaisesRegex(warm.MoEError, "cancellation marker is unsafe"):
            warm._cancel(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)

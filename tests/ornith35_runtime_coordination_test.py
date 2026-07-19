#!/usr/bin/env python3
"""Foreground-priority process coordination checks for Ornith-35."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_runtime_coordination as coordination


class RuntimeCoordinationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stale_foreground_marker_is_removed(self) -> None:
        self.assertEqual(coordination.active_foreground_requests(self.root), 0)
        marker_root = (
            self.root
            / coordination.RUNTIME_DIRECTORY
            / coordination.FOREGROUND_DIRECTORY
        )
        marker = marker_root / "123-dead.request"
        marker.write_text("{}\n", encoding="ascii")
        self.assertEqual(coordination.active_foreground_requests(self.root), 0)
        self.assertFalse(marker.exists())

    def test_stale_hidden_publication_is_removed(self) -> None:
        coordination.active_foreground_requests(self.root)
        marker_root = (
            self.root
            / coordination.RUNTIME_DIRECTORY
            / coordination.FOREGROUND_DIRECTORY
        )
        partial = marker_root / ".123-dead.part"
        partial.write_text("{}\n", encoding="ascii")
        self.assertEqual(coordination.active_foreground_requests(self.root), 0)
        self.assertFalse(partial.exists())

    def test_foreground_marker_is_cleaned_when_model_lock_setup_fails(self) -> None:
        coordination.active_foreground_requests(self.root)
        runtime_root = self.root / coordination.RUNTIME_DIRECTORY
        (runtime_root / coordination.MODEL_LOCK_NAME).mkdir()
        with self.assertRaises(coordination.RuntimeCoordinationError):
            with coordination.foreground_lease(self.root, timeout_s=1.0):
                self.fail("unsafe model lock unexpectedly opened")
        marker_root = runtime_root / coordination.FOREGROUND_DIRECTORY
        self.assertEqual(tuple(marker_root.iterdir()), ())

    def test_foreground_lease_blocks_background_ownership(self) -> None:
        with coordination.foreground_lease(self.root, timeout_s=1.0) as lease:
            self.assertGreaterEqual(lease.waited_s, 0.0)
            self.assertEqual(coordination.active_foreground_requests(self.root), 1)
            with self.assertRaises(coordination.BackgroundDeferred):
                with coordination.background_lease(self.root):
                    self.fail("background acquired a foreground-owned runtime")
        self.assertEqual(coordination.active_foreground_requests(self.root), 0)

    def test_pending_foreground_acquires_after_background_yields(self) -> None:
        code = """
from pathlib import Path
import sys
import ornith35_runtime_coordination as coordination
with coordination.foreground_lease(Path(sys.argv[1]), timeout_s=5.0, poll_s=0.01) as lease:
    print(f"acquired waited_s={lease.waited_s:.6f}", flush=True)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(TOOLS)
        process = None
        with coordination.background_lease(self.root):
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(self.root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            deadline = time.monotonic() + 3.0
            while (
                coordination.active_foreground_requests(self.root) == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertEqual(coordination.active_foreground_requests(self.root), 1)
            self.assertIsNone(process.poll())
        output, _ = process.communicate(timeout=3.0)
        self.assertEqual(process.returncode, 0, output)
        self.assertIn("acquired waited_s=", output)
        self.assertEqual(coordination.active_foreground_requests(self.root), 0)

    def test_unexpected_visible_marker_fails_closed(self) -> None:
        coordination.active_foreground_requests(self.root)
        marker_root = (
            self.root
            / coordination.RUNTIME_DIRECTORY
            / coordination.FOREGROUND_DIRECTORY
        )
        (marker_root / "unexpected").write_text("bad\n", encoding="ascii")
        with self.assertRaisesRegex(
            coordination.RuntimeCoordinationError,
            "unexpected foreground marker",
        ):
            coordination.active_foreground_requests(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)

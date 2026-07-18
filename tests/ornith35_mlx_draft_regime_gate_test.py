#!/usr/bin/env python3
"""Schema, accounting, and exact-state tests for the Ornith-35 draft gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_attention as attention
import ornith35_mlx_draft_regime_gate as gate
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
import ornith35_mlx_speculative as speculative


def record(
    prompt: str,
    *,
    draft: float = 1.0,
    target: float = 1.1,
    exact: bool = True,
) -> dict[str, object]:
    return {
        "accepted_future": 8,
        "detached_after_mtp_blocks": None,
        "draft_seconds": draft,
        "draft_steady_seconds": draft,
        "exact_replay": exact,
        "peak_gib": 23.0,
        "prompt_name": prompt,
        "proposed_future": 10,
        "steady_speedup": target / draft,
        "steady_transitions": 10,
        "target_seconds": target,
        "target_steady_seconds": target,
        "transitions": 10,
    }


class DraftRegimeGateTest(unittest.TestCase):
    def test_committed_corpus_has_strict_unique_schema(self) -> None:
        prompts = gate.load_prompts(gate.DEFAULT_PROMPTS)
        self.assertEqual(len(prompts), 12)
        self.assertEqual(len({prompt.name for prompt in prompts}), len(prompts))
        self.assertTrue(any(prompt.enable_thinking for prompt in prompts))
        self.assertTrue(any(not prompt.enable_thinking for prompt in prompts))
        self.assertEqual(
            gate.canonical_sha256([prompt.canonical() for prompt in prompts]),
            "7c6280e75c266c071c795711d5dd78e26067b1fd01366e508a0768aa995e30b5",
        )

    def test_prompt_loader_rejects_aliases_extras_and_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                '{"name":"one","prompt":"alias","enable_thinking":true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing required"):
                gate.load_prompts(path)
            path.write_text(
                '{"name":"one","user":"first","enable_thinking":true,"extra":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unknown fields"):
                gate.load_prompts(path)
            path.write_text(
                '{"name":"one","user":"first","enable_thinking":true}\n'
                '{"name":"one","user":"second","enable_thinking":false}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                gate.load_prompts(path)

    def test_capture_disjointness_uses_rendered_prompt_identity(self) -> None:
        prompts = [
            gate.EncodedPrompt(
                gate.GatePrompt("new-prompt", "new", None, True),
                hashlib.sha256(b"new rendered prompt").hexdigest(),
                (1, 2),
            )
        ]
        capture = {
            "format": "ornith35-mtp-teacher-capture-v2",
            "completed": {"0": {"prompt_sha256": hashlib.sha256(b"old").hexdigest()}},
        }
        gate.require_prompt_disjoint(prompts, capture)
        capture["completed"]["0"]["prompt_sha256"] = prompts[0].rendered_sha256
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            gate.require_prompt_disjoint(prompts, capture)

    def test_append_unique_tokens_preserves_one_pending_anchor(self) -> None:
        generated: list[int] = []
        gate.append_unique_tokens(generated, (7, 8, 9))
        gate.append_unique_tokens(generated, (9, 10))
        self.assertEqual(generated, [7, 8, 9, 10])
        with self.assertRaisesRegex(RuntimeError, "anchor"):
            gate.append_unique_tokens(generated, (11, 12))

    def test_exact_cursor_comparison_ignores_unused_linear_capacity(self) -> None:
        conv = mx.array([[1, 2]], dtype=mx.bfloat16)
        recurrent = mx.array([[3, 4]], dtype=mx.bfloat16)
        keys = mx.arange(8, dtype=mx.float32).reshape(1, 4, 2).astype(mx.bfloat16)
        values = (keys + 1).astype(mx.bfloat16)
        left = model.TextModelState(
            2,
            (
                gdn.MLXGDNState(conv, recurrent),
                attention.MLXLinearAttentionState(keys, values, 2, 4),
            ),
        )
        right = model.TextModelState(
            2,
            (
                gdn.MLXGDNState(conv, recurrent),
                attention.MLXAttentionState(keys[:, :2], values[:, :2]),
            ),
        )
        hidden = mx.array([1, 2], dtype=mx.bfloat16)
        logits = mx.array([3, 4, 5], dtype=mx.bfloat16)
        actual = speculative.GreedyTargetCursor(left, hidden, logits)
        expected = speculative.GreedyTargetCursor(right, hidden, logits)
        self.assertEqual(gate.exact_cursor_mismatches(actual, expected), [])

        broken_state = model.TextModelState(
            2,
            (
                gdn.MLXGDNState(conv, recurrent + 1),
                right.layers[1],
            ),
        )
        broken = speculative.GreedyTargetCursor(broken_state, hidden, logits)
        self.assertEqual(
            gate.exact_cursor_mismatches(broken, expected),
            ["layer-0-recurrent"],
        )

    def test_aggregate_applies_weighted_speed_and_prompt_floors(self) -> None:
        accepted = gate.aggregate_runs(
            [record("first"), record("first"), record("second")]
        )
        self.assertEqual(accepted["classification"], "production-beneficial")
        self.assertAlmostEqual(accepted["steady_speedup"], 1.1)
        self.assertEqual(accepted["faster_prompts"], 2)

        rejected = gate.aggregate_runs(
            [record("first"), record("second", draft=1.0, target=0.85)]
        )
        self.assertEqual(
            rejected["classification"],
            "exact-but-not-production-beneficial",
        )
        invalid = gate.aggregate_runs([record("first", exact=False)])
        self.assertEqual(invalid["classification"], "invalid-target-replay")

    def test_run_keys_and_atomic_state_are_deterministic(self) -> None:
        prompts = [
            gate.EncodedPrompt(gate.GatePrompt("first", "a", None, True), "a" * 64, (1,)),
            gate.EncodedPrompt(gate.GatePrompt("second", "b", None, False), "b" * 64, (2,)),
        ]
        self.assertEqual(
            gate.expected_run_keys("mtp-sampled", prompts, (11, 29)),
            [
                "first:seed-11",
                "first:seed-29",
                "second:seed-11",
                "second:seed-29",
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            gate.atomic_json(path, {"status": "running", "runs": {}})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "running")


if __name__ == "__main__":
    unittest.main()

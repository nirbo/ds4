#!/usr/bin/env python3
"""Text-only prompt rendering checks for the pinned Ornith-35 template."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_tokenizer as tokenizer


class TokenizerTest(unittest.TestCase):
    def test_metadata_file_requires_pinned_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            metadata.mkdir()
            path = metadata / "fixture.json"
            payload = b"fixture"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            original = tokenizer.EXPECTED_METADATA_FILES
            tokenizer.EXPECTED_METADATA_FILES = {
                **original,
                "fixture.json": (len(payload), digest),
            }
            try:
                state = {
                    "metadata_files": {
                        "fixture.json": {"bytes": len(payload), "sha256": digest}
                    }
                }
                self.assertEqual(
                    tokenizer._verify_metadata_file(root, state, "fixture.json"),
                    path,
                )
                state["metadata_files"]["fixture.json"]["sha256"] = "0" * 64
                with self.assertRaisesRegex(tokenizer.TokenizerError, "identity mismatch"):
                    tokenizer._verify_metadata_file(root, state, "fixture.json")
            finally:
                tokenizer.EXPECTED_METADATA_FILES = original

    def test_renders_no_thinking_user_prompt(self) -> None:
        self.assertEqual(
            tokenizer.render_text_prompt(
                "  Reply with exactly: OK  ",
                enable_thinking=False,
            ),
            "<|im_start|>user\nReply with exactly: OK<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_defaults_to_open_thinking_prompt(self) -> None:
        self.assertEqual(
            tokenizer.render_text_prompt(
                "Question",
                system=" Be concise. ",
            ),
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n",
        )

    def test_system_prefix_matches_the_complete_prompt_boundary(self) -> None:
        prefix = tokenizer.render_system_prefix(" Be concise. ")
        rendered = tokenizer.render_text_prompt("Question", system=" Be concise. ")
        self.assertEqual(prefix, "<|im_start|>system\nBe concise.<|im_end|>\n")
        self.assertTrue(rendered.startswith(prefix))

    def test_loads_a_stable_bounded_utf8_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "system.txt"
            payload = "Exact system prompt.\n".encode("utf-8")
            path.write_bytes(payload)
            loaded = tokenizer.load_prompt_text_file(path)
            self.assertEqual(loaded.text, payload.decode("utf-8"))
            self.assertEqual(loaded.byte_count, len(payload))
            self.assertEqual(loaded.sha256, hashlib.sha256(payload).hexdigest())

    def test_prompt_file_rejects_symlink_invalid_utf8_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.txt"
            target.write_text("prompt", encoding="ascii")
            symlink = root / "link.txt"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(tokenizer.TokenizerError, "cannot open"):
                tokenizer.load_prompt_text_file(symlink)

            invalid = root / "invalid.txt"
            invalid.write_bytes(b"\xff")
            with self.assertRaisesRegex(tokenizer.TokenizerError, "valid UTF-8"):
                tokenizer.load_prompt_text_file(invalid)

            with self.assertRaisesRegex(tokenizer.TokenizerError, "safe bound"):
                tokenizer.load_prompt_text_file(target, max_bytes=3)

    def test_rejects_empty_user_prompt(self) -> None:
        with self.assertRaisesRegex(tokenizer.TokenizerError, "must not be empty"):
            tokenizer.render_text_prompt("  ")


if __name__ == "__main__":
    unittest.main(verbosity=2)

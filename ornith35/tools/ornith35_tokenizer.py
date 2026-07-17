#!/usr/bin/env python3
"""Pinned tokenizer and exact text-only chat rendering for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


DEFAULT_ROOT = Path(
    "/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
)
SOURCE_STATE_FORMAT = "ornith35-source-metadata-v1"
EXPECTED_REPOSITORY = "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
EXPECTED_REVISION = "85ffd2d0629ae5fa4f860dda356ec33161806c9b"
EXPECTED_METADATA_FILES = {
    "tokenizer.json": (
        19_989_492,
        "6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706",
    ),
    "chat_template.jinja": (
        7_536,
        "182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e",
    ),
    "generation_config.json": (
        214,
        "56147698c439ec2da22f5820fcf76fa14788b6c0ebdd2b2e073c6cddfdd1ea30",
    ),
}
EXPECTED_SPECIAL_TOKENS = {
    "<|endoftext|>": 248_044,
    "<|im_start|>": 248_045,
    "<|im_end|>": 248_046,
    "<think>": 248_068,
    "</think>": 248_069,
}


class TokenizerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenizerError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizerError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise TokenizerError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _verify_metadata_file(root: Path, state: dict[str, Any], name: str) -> Path:
    entries = state.get("metadata_files")
    require(isinstance(entries, dict), "metadata state has no file inventory")
    entry = entries.get(name)
    require(isinstance(entry, dict), f"metadata state does not cover {name}")
    expected_bytes, expected_sha256 = EXPECTED_METADATA_FILES[name]
    require(entry.get("bytes") == expected_bytes, f"metadata identity mismatch: {name}")
    require(
        entry.get("sha256") == expected_sha256,
        f"metadata identity mismatch: {name}",
    )
    path = root / "metadata" / name
    require(path.is_file(), f"missing tokenizer metadata: {path}")
    require(path.stat().st_size == expected_bytes, f"metadata size mismatch: {name}")
    require(_sha256(path) == expected_sha256, f"metadata hash mismatch: {name}")
    return path


@dataclass(frozen=True)
class TextTokenizer:
    backend: Tokenizer
    eos_token_ids: frozenset[int]
    tokenizer_sha256: str
    template_sha256: str

    def encode(self, text: str) -> tuple[int, ...]:
        try:
            return tuple(self.backend.encode(text, add_special_tokens=False).ids)
        except Exception as exc:
            raise TokenizerError(f"tokenizer encode failed: {exc}") from exc

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        try:
            return self.backend.decode(list(token_ids), skip_special_tokens=False)
        except Exception as exc:
            raise TokenizerError(f"tokenizer decode failed: {exc}") from exc


def load_text_tokenizer(root: Path = DEFAULT_ROOT) -> TextTokenizer:
    state_path = root / "metadata" / "source-state.json"
    state = _load_json(state_path)
    require(state.get("format") == SOURCE_STATE_FORMAT, "unsupported metadata state")
    require(
        state.get("repository") == EXPECTED_REPOSITORY,
        "tokenizer repository mismatch",
    )
    require(state.get("revision") == EXPECTED_REVISION, "tokenizer revision mismatch")
    tokenizer_path = _verify_metadata_file(root, state, "tokenizer.json")
    template_path = _verify_metadata_file(root, state, "chat_template.jinja")
    generation_path = _verify_metadata_file(root, state, "generation_config.json")

    try:
        backend = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise TokenizerError(f"cannot load tokenizer {tokenizer_path}: {exc}") from exc
    for token, expected_id in EXPECTED_SPECIAL_TOKENS.items():
        require(backend.token_to_id(token) == expected_id, f"special-token mismatch: {token}")

    generation = _load_json(generation_path)
    eos = generation.get("eos_token_id")
    require(
        isinstance(eos, list) and eos == [248_046, 248_044],
        "generation EOS contract mismatch",
    )
    return TextTokenizer(
        backend=backend,
        eos_token_ids=frozenset(eos),
        tokenizer_sha256=_sha256(tokenizer_path),
        template_sha256=_sha256(template_path),
    )


def render_text_prompt(
    user: str,
    *,
    system: str | None = None,
    enable_thinking: bool = True,
) -> str:
    """Render the pinned template's system/user text-only generation subset."""
    require(isinstance(user, str) and user.strip(), "user prompt must not be empty")
    require(system is None or isinstance(system, str), "system prompt must be text")
    rendered = []
    if system is not None:
        rendered.append(f"<|im_start|>system\n{system.strip()}<|im_end|>\n")
    rendered.append(f"<|im_start|>user\n{user.strip()}<|im_end|>\n")
    rendered.append("<|im_start|>assistant\n")
    rendered.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(rendered)

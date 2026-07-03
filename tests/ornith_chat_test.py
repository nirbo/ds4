#!/usr/bin/env python3

import importlib.util
import argparse
import io
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("ornith_chat", TOOLS / "ornith_chat.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


def demo() -> None:
    assert mod.trim_completion("hello<|im_end|>ignored") == "hello"
    assert mod.trim_completion("hello<|endoftext|>ignored") == "hello"
    assert mod.visible_completion("<think>\nnotes\n</think>\n\nanswer<|im_end|>") == "answer"
    assert mod.visible_completion("</think>\n\nanswer<|im_end|>") == "answer"
    assert mod.visible_completion("answer\n</think><|im_end|>") == "answer"
    assert mod.assistant_history_completion("answer<|im_end|>", False) == "<think>\n\n</think>\n\nanswer"
    assert mod.assistant_history_completion("<think>", False) == "<think>\n\n</think>\n\n<think>"
    assert mod.assistant_history_completion("notes</think>\n\nanswer", True) == "<think>\nnotes</think>\n\nanswer"
    ids, scores = mod.parse_generator_output("header\n0\t19\t1.5\n1\t20\t-2\n")
    assert ids == [19, 20]
    assert scores == [1.5, -2.0]
    tokenizer = {
        "model": {"vocab": {"H": 0, "i": 1}, "merges": []},
        "added_tokens": [{"id": 2, "content": "<|im_end|>", "special": True}],
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tokenizer.json"
        path.write_text(json.dumps(tokenizer), encoding="utf-8")
        codec = mod.TokenCodec(path)
    assert codec.encode("Hi<|im_end|>") == [0, 1, 2]
    assert codec.decode([0, 1, 2]) == "Hi<|im_end|>"
    with tempfile.TemporaryDirectory() as td:
        messages = [{"role": "assistant", "content": "<think>\n\n</think>\n\nok"}]
        save_path = Path(td) / "chat.json"
        mod.save_messages(str(save_path), messages)
        assert json.loads(save_path.read_text(encoding="utf-8")) == messages
        prompt_path = Path(td) / "prompt.txt"
        prompt_path.write_text("file prompt\n", encoding="utf-8")
        args = argparse.Namespace(
            prompt="",
            prompt_file=str(prompt_path),
            messages=None,
            interactive=False,
        )
        mod.apply_prompt_file(args)
        assert args.prompt == "file prompt\n"
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO("stdin prompt")
        args = argparse.Namespace(prompt="", prompt_file="-", messages=None, interactive=False)
        mod.apply_prompt_file(args)
        assert args.prompt == "stdin prompt"
    finally:
        sys.stdin = old_stdin
    try:
        mod.apply_prompt_file(
            argparse.Namespace(prompt="inline", prompt_file="prompt.txt", messages=None, interactive=False)
        )
        raise AssertionError("positional prompt conflict was not rejected")
    except SystemExit as exc:
        assert "positional prompt" in str(exc)
    try:
        mod.apply_prompt_file(
            argparse.Namespace(prompt="", prompt_file="-", messages=None, interactive=True)
        )
        raise AssertionError("interactive stdin prompt was not rejected")
    except SystemExit as exc:
        assert "--interactive" in str(exc)


if __name__ == "__main__":
    demo()
    print("ornith_chat_test: ok")

#!/usr/bin/env python3

import importlib.util
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


if __name__ == "__main__":
    demo()
    print("ornith_chat_test: ok")

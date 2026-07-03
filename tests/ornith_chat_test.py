#!/usr/bin/env python3

import importlib.util
import sys
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
    ids, scores = mod.parse_generator_output("header\n0\t19\t1.5\n1\t20\t-2\n")
    assert ids == [19, 20]
    assert scores == [1.5, -2.0]


if __name__ == "__main__":
    demo()
    print("ornith_chat_test: ok")

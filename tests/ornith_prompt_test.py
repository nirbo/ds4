#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_prompt.py"
spec = importlib.util.spec_from_file_location("ornith_prompt", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    tokenizer = {
        "added_tokens": [
            {"id": 248045, "content": "<|im_start|>"},
            {"id": 248046, "content": "<|im_end|>"},
        ]
    }
    assert mod.special_token_ids(tokenizer)["<|im_start|>"] == 248045

    messages = [
        {"role": "system", "content": "  sys  "},
        {"role": "user", "content": " hello "},
    ]
    assert mod.render_text_chat(messages, enable_thinking=False) == (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    messages.append({"role": "assistant", "content": "<think>\nwhy\n</think>\n\nanswer"})
    messages.append({"role": "user", "content": "next"})
    assert "<think>\nwhy" not in mod.render_text_chat(messages, add_generation_prompt=False)


if __name__ == "__main__":
    demo()
    print("ornith_prompt_test: ok")

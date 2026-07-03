#!/usr/bin/env python3
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_decode_tokens.py"
spec = importlib.util.spec_from_file_location("ornith_decode_tokens", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


def test_decode() -> None:
    tokenizer = {
        "model": {"vocab": {"Ġ": 0, "H": 1, "i": 2, "ĠH": 3, "ĠHi": 4, "!": 5}, "merges": [["Ġ", "H"], ["ĠH", "i"]]},
        "added_tokens": [
            {"id": 6, "content": "<|im_end|>", "special": True},
            {"id": 7, "content": "<think>", "special": False},
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tokenizer.json"
        path.write_text(json.dumps(tokenizer), encoding="utf-8")
        vocab = mod.load_id_to_token(str(path))
        tok = mod.load_tokenizer(str(path))
    assert mod.decode([4, 5, 6], vocab) == " Hi!<|im_end|>"
    assert mod.encode(" Hi!", tok) == [4, 5]
    assert mod.encode(" Hi!<|im_end|>", tok) == [4, 5, 6]
    assert mod.encode("<think> Hi!", tok) == [7, 4, 5]


if __name__ == "__main__":
    test_decode()
    print("ornith_decode_tokens_test: ok")

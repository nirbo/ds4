#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_text_tensor_allowlist.py"
spec = importlib.util.spec_from_file_location("ornith_text_tensor_allowlist", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    index = {
        "weight_map": {
            "model.language_model.b": "a.safetensors",
            "model.visual.a": "a.safetensors",
            "lm_head.weight": "a.safetensors",
            "model.language_model.c": "b.safetensors",
        }
    }
    assert mod.text_tensors_for_shard(index, "a.safetensors") == [
        "lm_head.weight",
        "model.language_model.b",
    ]
    assert mod.text_tensors_for_shard(index, "missing.safetensors") == []


if __name__ == "__main__":
    demo()
    print("ornith_text_tensor_allowlist_test: ok")

#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_layout_check.py"
spec = importlib.util.spec_from_file_location("ornith_layout_check", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    cfg = {
        "text_config": {
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
        }
    }
    want = mod.expected_tensors(cfg)
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" in want
    assert "model.language_model.layers.1.self_attn.q_proj.weight" in want
    index = {"weight_map": {name: "model.safetensors" for name in want}}
    index["weight_map"]["model.visual.patch_embed.proj.weight"] = "model.safetensors"
    missing, unexpected = mod.check_layout(cfg, index)
    assert missing == []
    assert unexpected == []
    assert mod.visual_tensors(index) == ["model.visual.patch_embed.proj.weight"]

    del index["weight_map"]["lm_head.weight"]
    missing, unexpected = mod.check_layout(cfg, index)
    assert missing == ["lm_head.weight"]
    assert unexpected == []


if __name__ == "__main__":
    demo()
    print("ornith_layout_check_test: ok")

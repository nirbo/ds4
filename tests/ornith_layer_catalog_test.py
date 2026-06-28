#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_layer_catalog.py"
spec = importlib.util.spec_from_file_location("ornith_layer_catalog", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    cfg = {
        "text_config": {
            "layer_types": ["linear_attention", "full_attention"],
        }
    }
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    for layer, layer_type in enumerate(cfg["text_config"]["layer_types"]):
        prefix = f"model.language_model.layers.{layer}."
        attn = mod.LINEAR_ATTN if layer_type == "linear_attention" else mod.FULL_ATTN
        names.extend(prefix + name for name in attn)
        names.extend(prefix + name for name in mod.MOE)
    index = {"weight_map": {name: "model-00001.safetensors" for name in names}}

    catalog = mod.layer_catalog(cfg, index)
    assert catalog["num_layers"] == 2
    assert catalog["layer_type_counts"] == {"full_attention": 1, "linear_attention": 1}
    assert catalog["missing"] == []
    assert catalog["layers"][0]["type"] == "linear_attention"
    assert catalog["layers"][1]["type"] == "full_attention"
    assert catalog["layers"][0]["groups"]["moe"][0]["name"].endswith("input_layernorm.weight")
    assert catalog["language_shards"] == ["model-00001.safetensors"]

    del index["weight_map"]["lm_head.weight"]
    catalog = mod.layer_catalog(cfg, index)
    assert catalog["missing"] == ["lm_head.weight"]


if __name__ == "__main__":
    demo()
    print("ornith_layer_catalog_test: ok")

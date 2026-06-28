#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_shard_scope_report.py"
spec = importlib.util.spec_from_file_location("ornith_shard_scope_report", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    index = {
        "metadata": {"total_size": 1000},
        "weight_map": {
            "model.visual.patch_embed.proj.weight": "model-00001.safetensors",
            "model.language_model.embed_tokens.weight": "model-00001.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj": "model-00002.safetensors",
            "lm_head.weight": "model-00003.safetensors",
            "odd.tensor": "model-00003.safetensors",
        },
    }
    report = mod.shard_scope_report(index)
    assert report["total_weight_bytes"] == 1000
    assert report["shard_count"] == 3
    assert report["tensor_count"] == 5
    assert report["categories"] == {
        "language": 1,
        "language+other": 1,
        "language+visual": 1,
    }
    assert report["shards"][0]["scopes"] == {"language": 1, "visual": 1}
    assert report["shards"][1]["category"] == "language"
    assert mod.tensor_scope("lm_head.weight") == "language"
    assert mod.tensor_scope("model.visual.x") == "visual"
    assert mod.tensor_scope("x") == "other"


if __name__ == "__main__":
    demo()
    print("ornith_shard_scope_report_test: ok")

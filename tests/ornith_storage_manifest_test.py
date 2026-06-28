#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_storage_manifest.py"
spec = importlib.util.spec_from_file_location("ornith_storage_manifest", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    index = {
        "metadata": {"total_size": 300},
        "weight_map": {
            "b": "model-00002.safetensors",
            "a": "model-00001.safetensors",
            "c": "model-00002.safetensors",
        },
    }
    manifest = mod.shard_manifest(index, "org/model")
    assert manifest["total_weight_bytes"] == 300
    assert manifest["shard_count"] == 2
    assert manifest["tensor_count"] == 3
    assert manifest["shards"][0]["file"] == "model-00001.safetensors"
    assert manifest["shards"][1]["tensor_count"] == 2
    assert manifest["shards"][1]["url"].endswith("/model-00002.safetensors")


if __name__ == "__main__":
    demo()
    print("ornith_storage_manifest_test: ok")

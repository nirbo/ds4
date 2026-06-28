#!/usr/bin/env python3

import importlib.util
import json
import struct
import sys
import tempfile
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_storage_manifest.py"
spec = importlib.util.spec_from_file_location("ornith_storage_manifest", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    index = {
        "metadata": {"total_size": 300},
        "weight_map": {
            "model.language_model.b": "model-00002.safetensors",
            "model.visual.a": "model-00001.safetensors",
            "lm_head.weight": "model-00002.safetensors",
        },
    }
    manifest = mod.shard_manifest(index, "org/model")
    assert manifest["total_weight_bytes"] == 300
    assert manifest["source_total_weight_bytes"] == 300
    assert manifest["shard_count"] == 2
    assert manifest["tensor_count"] == 3
    assert manifest["shards"][0]["file"] == "model-00001.safetensors"
    assert manifest["shards"][1]["tensor_count"] == 2
    assert manifest["shards"][1]["url"].endswith("/model-00002.safetensors")

    text = mod.shard_manifest(index, "org/model", text_only=True)
    assert text["text_only"] is True
    assert text["shard_count"] == 1
    assert text["tensor_count"] == 2
    assert text["skipped_tensor_count"] == 1
    assert text["shards"][0]["file"] == "model-00002.safetensors"
    assert text["shards"][0]["skipped_tensor_count"] == 0

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        header = {
            "model.language_model.b": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "lm_head.weight": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]},
        }
        data = json.dumps(header).encode("utf-8")
        (root / "model-00002.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + b"bbbbcc")
        text = mod.shard_manifest(index, "org/model", text_only=True, safetensors_dir=root)
        assert text["selected_weight_bytes"] == 6
        assert text["total_weight_bytes"] == 6
        assert text["shards"][0]["selected_weight_bytes"] == 6
        out = StringIO()
        with redirect_stdout(out):
            mod.print_summary(text)
        assert "source total weight bytes: 300" in out.getvalue()
        assert "text-only weight bytes: 6" in out.getvalue()


if __name__ == "__main__":
    demo()
    print("ornith_storage_manifest_test: ok")

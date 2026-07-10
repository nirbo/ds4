#!/usr/bin/env python3

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ornith" / "tools" / "ornith_layer_calibration_bench.py"
SPEC = importlib.util.spec_from_file_location("ornith_layer_calibration_bench", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index = root / "model.safetensors.index.json"
        index.write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.language_model.layers.0.a": "model-00001.safetensors",
                        "model.language_model.layers.0.b": "model-00002.safetensors",
                        "model.language_model.layers.0.c": "model-00003.safetensors",
                        "model.language_model.layers.1.a": "model-00003.safetensors",
                        "model.language_model.layers.1.b": "model-00004.safetensors",
                        "model.visual.layers.0.a": "model-00099.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        assert MODULE.required_layer_shards(index, 0) == [
            "model-00001.safetensors",
            "model-00002.safetensors",
            "model-00003.safetensors",
        ]
        assert MODULE.required_layer_shards(index, 1) == [
            "model-00003.safetensors",
            "model-00004.safetensors",
        ]

        raw = root / "raw"
        preserved = root / "preserved"
        raw.mkdir()
        preserved.mkdir()
        (preserved / "model-00002.safetensors").write_bytes(b"test")
        assert MODULE.find_shard("model-00002.safetensors", raw, preserved).parent == preserved

        output = root / "result.json"
        MODULE.atomic_json(output, {"format": MODULE.FORMAT, "ok": True})
        assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True
        assert not output.with_name("result.json.part").exists()

    print("ornith_layer_calibration_bench_test: ok")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import importlib.util
import json
import struct
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


catalog_mod = load(TOOLS / "ornith_runtime_catalog.py", "ornith_runtime_catalog")
quant = load(TOOLS / "ornith_quantize_safetensors.py", "ornith_quantize_safetensors_for_catalog")


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def write_safetensors(path: Path) -> dict:
    chunks = {
        "model.language_model.embed_tokens.weight": b"".join(bf16(v) for v in [1, 2, 3, 4]),
        "model.language_model.norm.weight": b"".join(bf16(v) for v in [1, 1, 1, 1]),
        "lm_head.weight": b"".join(bf16(v) for v in [4, 3, 2, 1]),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": b"".join(bf16(v) for v in [1, -1, 1, -1]),
        "model.visual.blocks.0.attn.proj.weight": b"".join(bf16(v) for v in [9, 9, 9, 9]),
    }
    shapes = {
        "model.language_model.embed_tokens.weight": [2, 2],
        "model.language_model.norm.weight": [4],
        "lm_head.weight": [2, 2],
        "model.language_model.layers.0.mlp.experts.gate_up_proj": [1, 4],
        "model.visual.blocks.0.attn.proj.weight": [2, 2],
    }
    header = {}
    offset = 0
    for name, data in chunks.items():
        header[name] = {"dtype": "BF16", "shape": shapes[name], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks.values()))
    return {"weight_map": {name: path.name for name in chunks}}


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "model-00001-of-00122.safetensors"
        out = root / "model-00001-of-00122.ornq"
        policy_path = root / "iq1.policy.json"
        index = write_safetensors(src)
        policy_path.write_text(json.dumps({"rules": [{"contains": ".experts.gate_up_proj", "quant": "iq1"}]}), encoding="utf-8")
        with redirect_stdout(StringIO()):
            quant.quantize(src, out, block=4, threads=1, policy=quant.load_policy(policy_path))

        catalog = catalog_mod.build_catalog([out], index)
        assert catalog["format"] == catalog_mod.FORMAT
        assert catalog["shard_count"] == 1
        assert catalog["tensor_count"] == 4
        assert catalog["layer_count"] == 1
        assert catalog["missing_text_tensors"] == []
        assert catalog["unexpected_text_tensors"] == []
        assert not catalog_mod.validate_catalog(catalog)
        assert "model.visual.blocks.0.attn.proj.weight" not in catalog["tensors"]
        tensor = catalog["tensors"]["model.language_model.layers.0.mlp.experts.gate_up_proj"]
        assert tensor["shard"] == out.name
        assert tensor["quant"] == "iq1"
        assert tensor["layer"] == 0
        assert tensor["group"] == "routed_expert"
        native = root / "catalog.tsv"
        catalog_mod.write_native_tsv(catalog, native)
        text = native.read_text(encoding="utf-8")
        assert text.startswith("# ornith-runtime-catalog-tsv-v1\n")
        assert "\ttensor\t" not in text
        assert "\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\t" in text

        missing_index = {"weight_map": dict(index["weight_map"], **{"model.language_model.layers.0.mlp.gate.weight": src.name})}
        bad = catalog_mod.build_catalog([out], missing_index)
        assert "missing text tensor: model.language_model.layers.0.mlp.gate.weight" in catalog_mod.validate_catalog(bad)


if __name__ == "__main__":
    demo()
    print("ornith_runtime_catalog_test: ok")

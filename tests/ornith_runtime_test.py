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


quant = load(TOOLS / "ornith_quantize_safetensors.py", "ornith_quantize_safetensors")
runtime = load(TOOLS / "ornith_runtime.py", "ornith_runtime")


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def write_safetensors(path: Path):
    q4_values = [float((i % 17) - 8) for i in range(4098)]
    chunks = {
        "model.language_model.layers.0.input_layernorm.weight": b"".join(bf16(v) for v in [1.0, -2.0, 3.0, -4.0]),
        "model.language_model.layers.0.linear_attn.out_proj.weight": b"".join(bf16(v) for v in q4_values),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": b"".join(bf16(v) for v in [0.5, -0.5, 1.0, -1.0]),
        "model.visual.blocks.0.attn.proj.weight": b"".join(bf16(v) for v in [9.0, 10.0, 11.0, 12.0]),
    }
    shapes = {
        "model.language_model.layers.0.input_layernorm.weight": [4],
        "model.language_model.layers.0.linear_attn.out_proj.weight": [2, 2049],
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


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "src.safetensors"
        out = root / "out.ornq"
        write_safetensors(src)
        with redirect_stdout(StringIO()):
            quant.quantize(src, out, block=4, threads=2)

        with runtime.ORNQShard(out) as shard:
            assert set(shard.tensors) == {
                "model.language_model.layers.0.input_layernorm.weight",
                "model.language_model.layers.0.linear_attn.out_proj.weight",
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
            }
            norm = shard.tensors["model.language_model.layers.0.input_layernorm.weight"]
            q4 = shard.tensors["model.language_model.layers.0.linear_attn.out_proj.weight"]
            iq1 = shard.tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]
            assert norm.quant == "bf16"
            assert q4.quant == "q4"
            assert iq1.quant == "iq1"
            assert norm.dequantize() == [1.0, -2.0, 3.0, -4.0]
            assert len(q4.matvec([1.0] + [0.0] * 2048)) == 2
            assert iq1.role.group == "routed_expert"
            assert q4.role.group == "attention"
            report = runtime.memory_report([shard])
            catalog = runtime.layer_catalog([shard])
            layers = runtime.layer_views([shard])
            assert report["shards"] == 1
            assert report["tensors"] == 3
            assert report["layers_seen"] == [0]
            assert report["tensors_by_group"]["routed_expert"] == 1
            assert report["params_by_quant"]["bf16"] == 4
            assert report["params_by_quant"]["q4"] == 4098
            assert report["params_by_quant"]["iq1"] == 4
            assert sorted(catalog[0]) == ["attention", "norm", "routed_expert"]
            assert layers[0].summary()["groups"]["attention"] == 1
            assert len(layers[0].matvec("linear_attn.out_proj.weight", [1.0] + [0.0] * 2048)) == 2


if __name__ == "__main__":
    demo()
    print("ornith_runtime_test: ok")

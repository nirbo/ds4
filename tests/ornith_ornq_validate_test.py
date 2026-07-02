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
QUANT_PATH = TOOLS / "ornith_quantize_safetensors.py"
VALIDATE_PATH = TOOLS / "ornith_ornq_validate.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


quant = load(QUANT_PATH, "ornith_quantize_safetensors")
val = load(VALIDATE_PATH, "ornith_ornq_validate")


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "src.safetensors"
        out = root / "out.ornq"
        data = b"".join(bf16(v) for v in [1.0, -2.0, 3.0, -4.0])
        data_b = b"".join(bf16(v) for v in [0.25, -0.25, 0.75, -0.75])
        header = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [0, len(data)],
            },
            "model.language_model.layers.0.input_layernorm.weight": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [len(data), len(data) + len(data_b)],
            },
        }
        encoded = json.dumps(header).encode("utf-8")
        src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data + data_b)
        with redirect_stdout(StringIO()):
            quant.quantize(src, out, block=4, threads=2)
        h, data_start = val.read_ornq(out)
        assert val.check_offsets(out, h, data_start) == []
        reports = val.compare_source(out, src, samples=4)
        by_name = {report["name"]: report for report in reports}
        expert = by_name["model.language_model.layers.0.mlp.experts.gate_up_proj"]
        norm = by_name["model.language_model.layers.0.input_layernorm.weight"]
        assert expert["quant"] == "iq1"
        assert expert["samples"] == 4
        assert expert["mse"] >= 0.0
        assert norm["quant"] == "bf16"
        assert norm["mse"] == 0.0


if __name__ == "__main__":
    demo()
    print("ornith_ornq_validate_test: ok")

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
MOD_PATH = TOOLS / "ornith_quantize_safetensors.py"
spec = importlib.util.spec_from_file_location("ornith_quantize_safetensors", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def read_ornq(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fp:
        assert fp.read(8) == b"ORNQ1\0\0\0"
        n = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(n)), 16 + n


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "src.safetensors"
        dst = root / "out.ornq"
        data_a = b"".join(bf16(v) for v in [1.0, -2.0, 3.0, -4.0])
        data_b = b"".join(bf16(v) for v in [0.5, -0.5, 1.5, -1.5])
        header = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [0, len(data_a)],
            },
            "model.language_model.layers.0.input_layernorm.weight": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [len(data_a), len(data_a) + len(data_b)],
            },
        }
        encoded = json.dumps(header).encode("utf-8")
        src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data_a + data_b)
        with redirect_stdout(StringIO()):
            stats = mod.quantize(src, dst, block=4, threads=2)
        out, data_start = read_ornq(dst)
        tensors = out["tensors"]
        assert stats["tensors"] == 2
        assert tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]["quant"] == "iq1"
        assert tensors["model.language_model.layers.0.input_layernorm.weight"]["quant"] == "q4"
        assert dst.stat().st_size == data_start + 2 + 1 + 2 + 2


if __name__ == "__main__":
    demo()
    print("ornith_quantize_safetensors_test: ok")

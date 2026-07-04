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
err = load(TOOLS / "ornith_quant_error.py", "ornith_quant_error")


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def demo() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "src.safetensors"
        ornq = root / "out.ornq"
        raw = b"".join(bf16(v) for v in [1.0, -2.0, 3.0, -4.0])
        header = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [0, len(raw)],
            },
        }
        encoded = json.dumps(header).encode("utf-8")
        src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
        with redirect_stdout(StringIO()):
            quant.quantize(src, ornq, block=4, threads=2)
        report = err.run(src, ornq, root / "report.json", root / "report.md", threads=2, progress=0)
        row = report["tensors"][0]
        assert row["quant"] == "iq1"
        assert row["nparams"] == 4
        assert row["mean_abs_err"] > 0.0
        assert row["relative_l2"] > 0.0
        assert (root / "report.json").exists()
        assert "relative_l2" in (root / "report.md").read_text(encoding="utf-8")


if __name__ == "__main__":
    demo()
    print("ornith_quant_error_test: ok")

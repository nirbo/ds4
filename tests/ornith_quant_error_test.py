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
        policy = root / "iq1.policy.json"
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
        policy.write_text(json.dumps({"rules": [{"contains": ".experts.gate_up_proj", "quant": "iq1"}]}), encoding="utf-8")
        with redirect_stdout(StringIO()):
            quant.quantize(src, ornq, block=4, threads=2, policy=quant.load_policy(policy))
        report = err.run(src, ornq, root / "report.json", root / "report.md", threads=2, progress=0)
        row = report["tensors"][0]
        assert row["quant"] == "iq1"
        assert row["nparams"] == 4
        assert row["mean_abs_err"] > 0.0
        assert row["relative_l2"] > 0.0
        assert (root / "report.json").exists()
        assert "relative_l2" in (root / "report.md").read_text(encoding="utf-8")

        q4_src = root / "q4-src.safetensors"
        q4_ornq = root / "q4.ornq"
        q4_policy = root / "q4.policy.json"
        imatrix_raw = root / "imatrix.f32"
        imatrix_json = root / "imatrix.json"
        values = [((i % 31) - 15) / 64.0 for i in range(256)]
        q4_raw = b"".join(bf16(value) for value in values)
        q4_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        q4_header = {q4_name: {"dtype": "BF16", "shape": [1, 1, 256], "data_offsets": [0, len(q4_raw)]}}
        q4_encoded = json.dumps(q4_header).encode("utf-8")
        q4_src.write_bytes(struct.pack("<Q", len(q4_encoded)) + q4_encoded + q4_raw)
        q4_policy.write_text(json.dumps({"rules": [{"contains": ".experts.gate_up_proj", "quant": "q4_k"}]}), encoding="utf-8")
        imatrix_raw.write_bytes(struct.pack("<256f", *([1.0] * 256)))
        imatrix_json.write_text(
            json.dumps(
                {
                    "format": "ornith-imatrix-v1",
                    "source_model": "deepreinforce-ai/Ornith-1.0-397B",
                    "source_precision": "bf16",
                    "source_revision": "test-revision",
                    "statistic": "sum_squared_input_activation_per_expert",
                    "tensors": {
                        q4_name: {"file": imatrix_raw.name, "dtype": "float32-le", "shape": [1, 256]}
                    },
                }
            ),
            encoding="utf-8",
        )
        with redirect_stdout(StringIO()):
            quant.quantize(
                q4_src,
                q4_ornq,
                block=256,
                threads=2,
                policy=quant.load_policy(q4_policy),
                imatrix=quant.load_imatrix_manifest(imatrix_json),
            )
        weighted = err.run(
            q4_src,
            q4_ornq,
            root / "weighted.json",
            root / "weighted.md",
            threads=2,
            progress=0,
            imatrix=imatrix_json,
        )["tensors"][0]
        assert weighted["activation_weighted_relative_l2"] is not None
        assert weighted["activation_weighted_relative_l2"] > 0.0


if __name__ == "__main__":
    demo()
    print("ornith_quant_error_test: ok")

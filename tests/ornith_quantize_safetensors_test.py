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
VALIDATE_PATH = TOOLS / "ornith_ornq_validate.py"
val_spec = importlib.util.spec_from_file_location("ornith_ornq_validate_for_quant_test", VALIDATE_PATH)
val = importlib.util.module_from_spec(val_spec)
sys.modules[val_spec.name] = val
val_spec.loader.exec_module(val)


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
        policy_path = root / "policy.json"
        plan_path = root / "reap.json"
        data_a = b"".join(bf16(float(v)) for v in range(16))
        data_b = b"".join(bf16(v) for v in [0.5, -0.5, 1.5, -1.5])
        data_c = b"".join(bf16(v) for v in [2.5, -2.5, 3.5, -3.5])
        header = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, len(data_a)],
            },
            "model.language_model.layers.0.input_layernorm.weight": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [len(data_a), len(data_a) + len(data_b)],
            },
            "model.visual.blocks.0.attn.proj.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [len(data_a) + len(data_b), len(data_a) + len(data_b) + len(data_c)],
            },
        }
        encoded = json.dumps(header).encode("utf-8")
        src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data_a + data_b + data_c)
        policy_path.write_text(json.dumps({
            "name": "test-policy",
            "rules": [
                {"contains": ".experts.gate_up_proj", "quant": "q4"},
                {"contains": "input_layernorm", "quant": "bf16"},
            ],
            "default": "q4",
        }), encoding="utf-8")
        plan_path.write_text(json.dumps({
            "format": "ornith-reap-plan-v1",
            "layers": {"0": {"retained": [1, 3]}},
        }), encoding="utf-8")
        with redirect_stdout(StringIO()):
            stats = mod.quantize(src, dst, block=4, threads=2, policy=mod.load_policy(policy_path), reap_plan=mod.load_reap_plan(plan_path))
        out, data_start = read_ornq(dst)
        tensors = out["tensors"]
        assert stats["tensors"] == 2
        assert out["quant_policy"] == "test-policy"
        assert out["reap_plan"] == "ornith-reap-plan-v1"
        assert tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]["quant"] == "q4"
        assert tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]["shape"] == [2, 4]
        assert tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]["reap_retained_experts"] == [1, 3]
        assert tensors["model.language_model.layers.0.input_layernorm.weight"]["quant"] == "bf16"
        assert "model.visual.blocks.0.attn.proj.weight" not in tensors
        assert dst.stat().st_size == data_start + 8 + len(data_b)
        reports = {row["name"]: row for row in val.compare_source(dst, src, samples=8)}
        assert reports["model.language_model.layers.0.mlp.experts.gate_up_proj"]["samples"] == 8


if __name__ == "__main__":
    demo()
    print("ornith_quantize_safetensors_test: ok")

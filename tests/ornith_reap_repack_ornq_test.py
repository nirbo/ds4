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
val = load(TOOLS / "ornith_ornq_validate.py", "ornith_ornq_validate")
runtime = load(TOOLS / "ornith_runtime.py", "ornith_runtime")
catalog_mod = load(TOOLS / "ornith_runtime_catalog.py", "ornith_runtime_catalog")
repack = load(TOOLS / "ornith_reap_repack_ornq.py", "ornith_reap_repack_ornq")


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def add_tensor(header: dict, payload: list[bytes], name: str, shape: list[int], values: list[float]) -> None:
    start = sum(len(chunk) for chunk in payload)
    data = b"".join(bf16(v) for v in values)
    payload.append(data)
    header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, start + len(data)]}


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        raw = root / "model-00001-of-00122.safetensors"
        src_dir = root / "src"
        dst_dir = root / "dst"
        plan = root / "plan.json"
        out = src_dir / "model-00001-of-00122.ornq"
        header = {}
        payload = []
        add_tensor(header, payload, "model.language_model.embed_tokens.weight", [2, 2], [1, 2, 3, 4])
        add_tensor(header, payload, "model.language_model.layers.0.mlp.gate.weight", [4, 4], list(range(16)))
        add_tensor(header, payload, "model.language_model.layers.0.mlp.experts.gate_up_proj", [4, 4, 4], list(range(64)))
        add_tensor(header, payload, "model.language_model.layers.0.mlp.experts.down_proj", [4, 4, 4], list(range(64, 128)))
        add_tensor(header, payload, "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", [4, 4], list(range(128, 144)))
        encoded = json.dumps(header).encode("utf-8")
        raw.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))
        src_dir.mkdir()
        with redirect_stdout(StringIO()):
            quant.quantize(raw, out, block=4, threads=1)
        plan.write_text(json.dumps({
            "format": "ornith-reap-plan-v1",
            "layers": {"0": {"retained": [1, 3], "retained_count": 2}},
        }), encoding="utf-8")

        report = repack.run(src_dir, dst_dir, plan, max_shards=1)
        assert report["shards"][0]["pruned_tensors"] == 3
        assert report["skipped_shards"] == 0
        assert "saved_bytes" in report
        repacked = dst_dir / out.name
        h, data_start = val.read_ornq(repacked)
        assert val.check_offsets(dst_dir / out.name, h, data_start) == []
        with runtime.ORNQShard(repacked):
            pass
        tensors = h["tensors"]
        assert tensors["model.language_model.embed_tokens.weight"]["shape"] == [2, 2]
        assert tensors["model.language_model.layers.0.mlp.shared_expert.up_proj.weight"]["shape"] == [4, 4]
        for name in (
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.0.mlp.experts.down_proj",
        ):
            assert tensors[name]["shape"][0] == 2
            assert tensors[name]["reap_retained_experts"] == [1, 3]
        catalog = catalog_mod.build_catalog([repacked])
        assert catalog["tensors"]["model.language_model.layers.0.mlp.gate.weight"]["shape"] == [2, 4]
        assert catalog["tensors"]["model.language_model.layers.0.mlp.experts.gate_up_proj"]["shape"] == [2, 4, 4]
        assert catalog["tensors"]["model.language_model.layers.0.mlp.experts.down_proj"]["shape"] == [2, 4, 4]
        second = repack.run(src_dir, dst_dir, plan, max_shards=1)
        assert second["skipped_shards"] == 1
        assert second["shards"][0]["skipped"]


if __name__ == "__main__":
    demo()
    print("ornith_reap_repack_ornq_test: ok")

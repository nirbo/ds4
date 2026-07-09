#!/usr/bin/env python3

import importlib.util
import json
import math
import struct
import subprocess
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


quant = load(TOOLS / "ornith_quantize_safetensors.py", "ornith_quantize_safetensors_ds4_test")
runtime = load(TOOLS / "ornith_runtime.py", "ornith_runtime_ds4_test")
imatrix_mod = load(TOOLS / "ornith_imatrix_manifest.py", "ornith_imatrix_manifest_test")
catalog_mod = load(TOOLS / "ornith_runtime_catalog.py", "ornith_runtime_catalog_ds4_test")
error_mod = load(TOOLS / "ornith_quant_error.py", "ornith_quant_error_ds4_test")


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def demo() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "src.safetensors"
        out = root / "out.ornq"
        policy_path = root / "policy.json"
        imatrix_path = root / "imatrix.json"
        importance_path = root / "gate-up.f32"

        names = {
            "model.language_model.embed_tokens.weight": ("bf16", [1, 256]),
            "model.language_model.norm.weight": ("bf16", [256]),
            "lm_head.weight": ("q8_0", [1, 256]),
            "model.language_model.layers.0.self_attn.q_proj.weight": ("q8_0", [1, 256]),
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": ("q4_k", [1, 256]),
            "model.language_model.layers.0.mlp.experts.down_proj": ("q2_k", [2, 1, 256]),
            "model.language_model.layers.0.mlp.experts.gate_up_proj": ("iq2_xxs", [2, 1, 256]),
        }
        header = {}
        payload = bytearray()
        for offset, (name, (_mode, shape)) in enumerate(names.items()):
            values = [math.sin((i + 1) * (offset + 1) * 0.071) * 0.4 for i in range(math.prod(shape))]
            raw = b"".join(bf16(v) for v in values)
            start = len(payload)
            payload.extend(raw)
            header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, len(payload)]}
        encoded = json.dumps(header).encode("utf-8")
        src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

        policy_path.write_text(json.dumps({
            "name": "all-ds4-formats-test",
            "rules": [{"exact": name, "quant": mode} for name, (mode, _shape) in names.items()],
        }), encoding="utf-8")
        importance = [1.0 + (100.0 if i % 256 == 0 else 0.0) for i in range(512)]
        importance_path.write_bytes(struct.pack("<512f", *importance))
        gate_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        imatrix_path.write_text(json.dumps({
            "format": "ornith-imatrix-v1",
            "source_model": "deepreinforce-ai/Ornith-1.0-397B",
            "source_precision": "bf16",
            "source_revision": "unit-test-revision",
            "statistic": "sum_squared_input_activation_per_expert",
            "calibration_sha256": "unit-test",
            "tensors": {
                gate_name: {"file": importance_path.name, "dtype": "float32-le", "shape": [2, 256]},
            },
        }), encoding="utf-8")

        manifest = quant.load_imatrix_manifest(imatrix_path)
        with redirect_stdout(StringIO()):
            quant.quantize(src, out, block=256, threads=2, policy=quant.load_policy(policy_path), imatrix=manifest)
        report = imatrix_mod.validate(imatrix_path)
        assert report["tensors"] == 1
        with runtime.ORNQShard(out) as shard:
            assert {name: tensor.quant for name, tensor in shard.tensors.items()} == {
                name: mode for name, (mode, _shape) in names.items()
            }
            for tensor in shard.tensors.values():
                values = tensor.dequantize(limit=min(32, tensor.nparams))
                assert values and all(math.isfinite(v) for v in values)

        reap_out = root / "reap.ornq"
        reap_plan = {
            "format": "ornith-reap-plan-v1",
            "quality_profile": "final",
            "source_model": "deepreinforce-ai/Ornith-1.0-397B",
            "source_precision": "bf16",
            "source_revision": "unit-test-revision",
            "layers": {"0": {"retained": [1]}},
        }
        with redirect_stdout(StringIO()):
            quant.quantize(src, reap_out, block=256, threads=2, policy=quant.load_policy(policy_path), reap_plan=reap_plan, imatrix=manifest)
        with runtime.ORNQShard(reap_out) as shard:
            assert shard.tensors[gate_name].shape == [1, 1, 256]
            assert shard.tensors[gate_name].quant == "iq2_xxs"
            assert shard.tensors["model.language_model.layers.0.mlp.experts.down_proj"].shape == [1, 1, 256]
        with redirect_stdout(StringIO()):
            reap_error = error_mod.run(src, reap_out, root / "reap-error.json", root / "reap-error.md", threads=2, progress=0)
        assert len(reap_error["tensors"]) == len(names)

        catalog = catalog_mod.build_catalog([out])
        catalog_tsv = root / "catalog.tsv"
        catalog_mod.write_native_tsv(catalog, catalog_tsv)
        probe = root / "native-probe"
        subprocess.run([
            "cc", "-O2", "-std=c11", "-I.", "ornith/ornith.c",
            "tests/ornith_native_ds4_formats_probe.c", "-lm", "-o", str(probe),
        ], cwd=ROOT, check=True)
        subprocess.run([str(probe), str(catalog_tsv), str(root)], check=True, stdout=subprocess.DEVNULL)
        with redirect_stdout(StringIO()):
            error_report = error_mod.run(src, out, root / "error.json", root / "error.md", threads=2, progress=0)
        assert {row["quant"] for row in error_report["tensors"]} >= {"q8_0", "q2_k", "q4_k", "iq2_xxs"}

        missing = root / "missing.ornq"
        try:
            with redirect_stdout(StringIO()):
                quant.quantize(src, missing, block=256, threads=1, policy=quant.load_policy(policy_path))
        except ValueError as exc:
            assert "requires a per-expert activation imatrix" in str(exc)
        else:
            raise AssertionError("iq2_xxs accepted without an imatrix")


if __name__ == "__main__":
    demo()
    print("ornith_ds4_formats_test: ok")

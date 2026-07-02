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
MOD_PATH = TOOLS / "ornith_stream_run.py"
spec = importlib.util.spec_from_file_location("ornith_stream_run", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from ornith_safetensors_filter import read_header


def write_safetensors(path: Path):
    chunks = {
        "model.language_model.a": b"aaaa",
        "model.visual.b": b"bbbb",
    }
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, data in chunks.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks.values()))


def bf16(v: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return struct.pack("<H", bits >> 16)


def write_bf16_safetensors(path: Path):
    chunks = {
        "model.language_model.layers.0.input_layernorm.weight": b"".join(bf16(v) for v in [1.0, -2.0, 3.0, -4.0]),
        "model.visual.blocks.0.attn.proj.weight": b"".join(bf16(v) for v in [5.0, -6.0, 7.0, -8.0]),
    }
    header = {}
    offset = 0
    for name, data in chunks.items():
        header[name] = {"dtype": "BF16", "shape": [len(data) // 2], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks.values()))


def read_ornq(path: Path) -> dict:
    with path.open("rb") as fp:
        assert fp.read(8) == b"ORNQ1\0\0\0"
        n = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(n))


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source"
        raw = root / "raw"
        out = root / "out"
        allow = root / "allow"
        source.mkdir()
        allow.mkdir()
        write_safetensors(source / "a.safetensors")
        (source / "b.safetensors").write_bytes(b"copy")
        (source / "c.safetensors").write_bytes(b"extra")
        (allow / "a.text.allowlist").write_text("model.language_model.a\n", encoding="utf-8")

        plan = {
            "shards": [
                {"file": "a.safetensors", "action": "filter", "text_tensor_count": 1, "skipped_tensor_count": 1},
                {"file": "b.safetensors", "action": "copy", "text_tensor_count": 1, "skipped_tensor_count": 0},
                {"file": "c.safetensors", "action": "copy", "text_tensor_count": 1, "skipped_tensor_count": 0},
            ]
        }
        manifest = {
            "shards": [
                {"file": "a.safetensors", "url": str(source / "a.safetensors")},
                {"file": "b.safetensors", "url": str(source / "b.safetensors")},
                {"file": "c.safetensors", "url": str(source / "c.safetensors")},
            ]
        }
        plan_path = root / "plan.json"
        manifest_path = root / "manifest.json"
        state_path = root / "state.json"
        log_path = root / "run.log"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        args = mod.parse_args([
            "--plan", str(plan_path),
            "--manifest", str(manifest_path),
            "--state", str(state_path),
            "--raw-dir", str(raw),
            "--out-dir", str(out),
            "--allowlist-dir", str(allow),
            "--log", str(log_path),
            "--progress-interval", "0",
            "--max-shards", "2",
        ])
        with redirect_stdout(StringIO()):
            assert mod.run(args) == 0

        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert [shard["status"] for shard in state["shards"]] == ["done", "done", "pending"]
        assert not (raw / "a.safetensors").exists()
        assert not (raw / "b.safetensors").exists()
        header, _ = read_header(out / "a.safetensors")
        assert sorted(name for name in header if name != "__metadata__") == ["model.language_model.a"]
        assert (out / "b.safetensors").read_bytes() == b"copy"
        assert not (raw / "c.safetensors").exists()
        assert not (out / "c.safetensors").exists()
        log = log_path.read_text(encoding="utf-8")
        assert "download-ready shard=a.safetensors" in log
        assert "process-verified shard=b.safetensors" in log

        qroot = Path(tempfile.mkdtemp(dir=td))
        qsource = qroot / "source"
        qraw = qroot / "raw"
        qout = qroot / "out"
        qallow = qroot / "allow"
        qsource.mkdir()
        qallow.mkdir()
        write_bf16_safetensors(qsource / "q.safetensors")
        qplan = {
            "shards": [
                {"file": "q.safetensors", "action": "filter", "text_tensor_count": 1, "skipped_tensor_count": 1},
            ]
        }
        qmanifest = {"shards": [{"file": "q.safetensors", "url": str(qsource / "q.safetensors")}]}
        qplan_path = qroot / "plan.json"
        qmanifest_path = qroot / "manifest.json"
        qstate_path = qroot / "state.json"
        qlog_path = qroot / "run.log"
        qplan_path.write_text(json.dumps(qplan), encoding="utf-8")
        qmanifest_path.write_text(json.dumps(qmanifest), encoding="utf-8")
        qargs = mod.parse_args([
            "--plan", str(qplan_path),
            "--manifest", str(qmanifest_path),
            "--state", str(qstate_path),
            "--raw-dir", str(qraw),
            "--out-dir", str(qout),
            "--allowlist-dir", str(qallow),
            "--log", str(qlog_path),
            "--processor", "quantize",
            "--progress-interval", "0",
        ])
        with redirect_stdout(StringIO()):
            assert mod.run(qargs) == 0
        qstate = json.loads(qstate_path.read_text(encoding="utf-8"))
        assert qstate["shards"][0]["status"] == "done"
        assert not (qraw / "q.safetensors").exists()
        qheader = read_ornq(qout / "q.ornq")
        assert list(qheader["tensors"]) == ["model.language_model.layers.0.input_layernorm.weight"]
        assert "quant-validate tensor=model.language_model.layers.0.input_layernorm.weight" in qlog_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    demo()
    print("ornith_stream_run_test: ok")

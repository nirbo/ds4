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
        (allow / "a.text.allowlist").write_text("model.language_model.a\n", encoding="utf-8")

        plan = {
            "shards": [
                {"file": "a.safetensors", "action": "filter", "text_tensor_count": 1, "skipped_tensor_count": 1},
                {"file": "b.safetensors", "action": "copy", "text_tensor_count": 1, "skipped_tensor_count": 0},
            ]
        }
        manifest = {
            "shards": [
                {"file": "a.safetensors", "url": str(source / "a.safetensors")},
                {"file": "b.safetensors", "url": str(source / "b.safetensors")},
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
        ])
        with redirect_stdout(StringIO()):
            assert mod.run(args) == 0

        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert [shard["status"] for shard in state["shards"]] == ["done", "done"]
        assert not (raw / "a.safetensors").exists()
        assert not (raw / "b.safetensors").exists()
        header, _ = read_header(out / "a.safetensors")
        assert sorted(name for name in header if name != "__metadata__") == ["model.language_model.a"]
        assert (out / "b.safetensors").read_bytes() == b"copy"
        log = log_path.read_text(encoding="utf-8")
        assert "download-ready shard=a.safetensors" in log
        assert "process-verified shard=b.safetensors" in log


if __name__ == "__main__":
    demo()
    print("ornith_stream_run_test: ok")

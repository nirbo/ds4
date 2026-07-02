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


def load(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


download_mod = load("ornith_download_shard")
process_mod = load("ornith_process_shard")
filter_mod = load("ornith_safetensors_filter")


def write_demo(path: Path):
    chunks = {
        "model.language_model.a": b"aaaa",
        "model.visual.b": b"bbbb",
    }
    header = {}
    offset = 0
    for name, data in chunks.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks.values()))


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = root / "run.log"
        src = root / "src.bin"
        dst = root / "dst.bin"
        src.write_bytes(b"x" * 4096)
        with redirect_stdout(StringIO()):
            stats = download_mod.download(str(src), dst, expected_size=4096, log_path=log, interval=0)
        assert stats == {"bytes": 4096}
        assert dst.read_bytes() == src.read_bytes()
        assert "download-done" in log.read_text(encoding="utf-8")
        assert "rate=" in log.read_text(encoding="utf-8")

        raw = root / "raw.safetensors"
        out = root / "out.safetensors"
        allow = root / "allow.txt"
        write_demo(raw)
        allow.write_text("model.language_model.a\n", encoding="utf-8")
        with redirect_stdout(StringIO()):
            process_mod.process("filter", raw, out, allowlist=allow, log_path=log, interval=0)
        header, _ = filter_mod.read_header(out)
        assert sorted(header) == ["model.language_model.a"]
        assert "filter tensor=1/1" in log.read_text(encoding="utf-8")
        assert "process-done" in log.read_text(encoding="utf-8")
        bench = root / "bench.safetensors"
        with redirect_stdout(StringIO()):
            process_mod.benchmark("filter", raw, bench, allowlist=allow, log_path=log, interval=0)
        assert not bench.exists()
        assert "benchmark-cleanup" in log.read_text(encoding="utf-8")


if __name__ == "__main__":
    demo()
    print("ornith_stream_wrappers_test: ok")

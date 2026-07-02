#!/usr/bin/env python3

import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_safetensors_filter.py"
spec = importlib.util.spec_from_file_location("ornith_safetensors_filter", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def write_demo(path: Path):
    chunks = {
        "model.language_model.a": b"aaaa",
        "model.visual.b": b"bbbbbb",
        "lm_head.weight": b"cc",
    }
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, data in chunks.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks.values()))


def tensor_bytes(path: Path, name: str) -> bytes:
    header, data_start = mod.read_header(path)
    start, end = header[name]["data_offsets"]
    with path.open("rb") as fp:
        fp.seek(data_start + start)
        return fp.read(end - start)


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "src.safetensors"
        dst = root / "dst.safetensors"
        write_demo(src)
        stats = mod.filter_safetensors(src, dst, text_only=True)
        header, _ = mod.read_header(dst)
        assert stats == {"selected": 2, "bytes": 6}
        assert header["__metadata__"] == {"format": "pt"}
        assert sorted(name for name in header if name != "__metadata__") == ["lm_head.weight", "model.language_model.a"]
        assert tensor_bytes(dst, "model.language_model.a") == b"aaaa"
        assert tensor_bytes(dst, "lm_head.weight") == b"cc"


if __name__ == "__main__":
    demo()
    print("ornith_safetensors_filter_test: ok")

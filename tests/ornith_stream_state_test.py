#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_stream_state.py"
spec = importlib.util.spec_from_file_location("ornith_stream_state", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    plan = {
        "shards": [
            {"file": "a", "action": "filter", "text_tensor_count": 2, "skipped_tensor_count": 1},
            {"file": "b", "action": "copy", "text_tensor_count": 1, "skipped_tensor_count": 0},
            {"file": "c", "action": "skip", "text_tensor_count": 0, "skipped_tensor_count": 1},
        ]
    }
    state = mod.new_state(plan)
    assert mod.counts(state) == {
        "pending": 2,
        "downloading": 0,
        "downloaded": 0,
        "processing": 0,
        "done": 0,
        "failed": 0,
    }

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        raw_a = root / "a.raw"
        raw_b = root / "b.raw"
        raw_a.write_bytes(b"raw-a")
        raw_b.write_bytes(b"raw-b")
        assert mod.start_download(state)["file"] == "a"
        assert mod.start_download(state) is None
        mod.mark_downloaded(state, "a", raw_a)
        assert mod.start_process(state)["file"] == "a"
        assert mod.start_process(state) is None
        assert mod.start_download(state)["file"] == "b"
        mod.mark_downloaded(state, "b", raw_b)

        out = Path(td) / "a.out"
        out.write_bytes(b"ok")
        shard = mod.mark_done(state, "a", out, delete_raw=True)
        assert shard["output_size"] == 2
        assert not raw_a.exists()
        assert raw_b.exists()
        assert mod.verify_done(state) == []
        out.write_bytes(b"changed")
        assert mod.verify_done(state) == ["a: size changed", "a: sha256 changed"]


if __name__ == "__main__":
    demo()
    print("ornith_stream_state_test: ok")

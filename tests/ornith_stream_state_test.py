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
    assert mod.counts(state) == {"pending": 2, "running": 0, "done": 0, "failed": 0}
    assert mod.start_next(state)["file"] == "a"
    assert mod.counts(state)["running"] == 1
    mod.mark_failed(state, "a", "network")
    assert mod.start_next(state)["file"] == "a"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "a.out"
        out.write_bytes(b"ok")
        shard = mod.mark_done(state, "a", out)
        assert shard["output_size"] == 2
        assert mod.verify_done(state) == []
        out.write_bytes(b"changed")
        assert mod.verify_done(state) == ["a: size changed", "a: sha256 changed"]


if __name__ == "__main__":
    demo()
    print("ornith_stream_state_test: ok")

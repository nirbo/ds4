#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
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
    configured = mod.new_state(plan, {"policy": "a"})
    mod.require_run_config(configured, {"policy": "a"})
    try:
        mod.require_run_config(configured, {"policy": "b"})
    except ValueError as exc:
        assert "refusing to mix" in str(exc)
    else:
        raise AssertionError("changed stream configuration was accepted")

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

        log = root / "state.log"
        with redirect_stdout(StringIO()):
            mod.log_event(log, "failed shard=a error=test")
        assert "failed shard=a error=test" in log.read_text(encoding="utf-8")

        interrupted = mod.new_state(plan)
        raw_c = root / "c.raw"
        raw_c.write_bytes(b"raw-c")
        interrupted["shards"][0]["status"] = "processing"
        interrupted["shards"][0]["raw"] = str(raw_c)
        interrupted["shards"][1]["status"] = "downloading"
        messages = mod.recover_interrupted(interrupted, root)
        assert interrupted["shards"][0]["status"] == "downloaded"
        assert interrupted["shards"][1]["status"] == "failed"
        assert "resume-process-retry shard=a" in messages[0]
        assert "resume-download-retry shard=b" in messages[1]

        retry_first = mod.new_state(plan)
        retry_first["shards"][0]["status"] = "failed"
        retry_first["shards"][1]["status"] = "downloaded"
        assert mod.start_download(retry_first)["file"] == "a"


if __name__ == "__main__":
    demo()
    print("ornith_stream_state_test: ok")

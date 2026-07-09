#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_build_calibration_dataset.py"
spec = importlib.util.spec_from_file_location("ornith_build_calibration_dataset", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo() -> None:
    a = mod.build(ROOT, 32, 397)
    b = mod.build(ROOT, 32, 397)
    assert a == b
    assert len(a) == 32
    assert all(item["messages"][-1]["role"] == "user" for item in a)
    assert any("tools" in item for item in a)
    json.dumps(a)


if __name__ == "__main__":
    demo()
    print("ornith_calibration_dataset_test: ok")

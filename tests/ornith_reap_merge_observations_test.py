#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_reap_merge_observations.py"
spec = importlib.util.spec_from_file_location("ornith_reap_merge_observations", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "a.json"
        b = root / "b.json"
        a.write_text(json.dumps({"layers": {"0": {
            "total_tokens": 2,
            "expert_frequency": [2, 0],
            "weighted_expert_frequency_sum": [0.5, 0.0],
            "ean_mean": [4.0, 0.0],
            "reap": [1.0, 0.0],
            "max_activations": [3.0, 0.0],
        }}}), encoding="utf-8")
        b.write_text(json.dumps({"layers": {"0": {
            "total_tokens": 3,
            "expert_frequency": [1, 2],
            "weighted_expert_frequency_sum": [0.25, 0.75],
            "ean_mean": [10.0, 6.0],
            "reap": [2.0, 3.0],
            "max_activations": [1.0, 9.0],
        }}}), encoding="utf-8")
        out = mod.run([a, b])
        layer = out["layers"]["0"]
        assert layer["total_tokens"] == 5
        assert layer["expert_frequency"] == [3, 2]
        assert layer["weighted_expert_frequency_sum"] == [0.75, 0.75]
        assert layer["ean_mean"] == [6.0, 6.0]
        assert layer["reap"] == [4.0 / 3.0, 3.0]
        assert layer["max_activations"] == [3.0, 9.0]


if __name__ == "__main__":
    demo()
    print("ornith_reap_merge_observations_test: ok")

#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_reap_plan.py"
spec = importlib.util.spec_from_file_location("ornith_reap_plan", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    data = {
        "layers": {
            "0": {
                "reap": [0.01, 0.02, 10.0, 0.03],
                "expert_frequency": [1, 99, 1, 1],
                "max_activations": [0.1, 0.1, 100.0, 0.1],
            },
            "1": {
                "reap": [0.04, 0.03, 0.02, 0.01],
                "expert_frequency": [1, 1, 1, 1],
                "max_activations": [0.1, 0.1, 0.1, 0.1],
            },
        }
    }
    plan = mod.build_plan(data, compression_ratio=0.5, metric="reap", min_retained=2, preserve_top_fraction=0.25, preserve_outliers=False)
    assert 1 not in plan["layers"]["0"]["pruned"]
    assert 2 not in plan["layers"]["0"]["pruned"]
    assert plan["layers"]["0"]["pruned_count"] == 2
    assert plan["layers"]["1"]["pruned_count"] == 2
    assert len(plan["layers"]["1"]["retained"]) == 2


if __name__ == "__main__":
    demo()
    print("ornith_reap_plan_test: ok")

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
    assert plan["layers"]["0"]["observed_count"] == 4
    assert plan["layers"]["0"]["unobserved_count"] == 0
    assert plan["layers"]["0"]["candidate_count"] == 2
    assert plan["layers"]["1"]["pruned_count"] == 2
    assert len(plan["layers"]["1"]["retained"]) == 2
    unobserved = {"layers": {"0": {
        "reap": [0.0, 0.01, 0.02, 0.03],
        "expert_frequency": [0, 1, 1, 1],
        "max_activations": [0.0, 0.0, 0.0, 0.0],
    }}}
    guarded = mod.build_plan(unobserved, compression_ratio=0.75, metric="reap", min_retained=1, preserve_top_fraction=0.0, preserve_outliers=False)
    assert 0 not in guarded["layers"]["0"]["pruned"]
    assert guarded["layers"]["0"]["observed_count"] == 3
    assert guarded["layers"]["0"]["unobserved_count"] == 1
    assert guarded["layers"]["0"]["unobserved_preserved_count"] == 1
    assert guarded["layers"]["0"]["observed_fraction"] == 0.75
    allowed = mod.build_plan(unobserved, compression_ratio=0.75, metric="reap", min_retained=1, preserve_top_fraction=0.0, preserve_outliers=False, preserve_unobserved=False)
    assert 0 in allowed["layers"]["0"]["pruned"]
    hybrid = mod.build_plan(data, compression_ratio=0.5, metric="reap", min_retained=2, preserve_top_fraction=0.0, preserve_outliers=False, strategy="hybrid")
    assert 2 not in hybrid["layers"]["0"]["pruned"]
    many = {"layers": {}}
    for i in range(8):
        many["layers"][str(i)] = {
            "reap": [float(j) for j in range(10)],
            "ean_mean": [float(j) for j in range(10)],
            "expert_frequency": [j + 1 for j in range(10)],
            "max_activations": [float(j) for j in range(10)],
        }
    uniform = mod.build_plan(many, compression_ratio=0.3, metric="reap", min_retained=1, preserve_top_fraction=0.0, preserve_outliers=False)
    late = mod.build_plan(many, compression_ratio=0.3, metric="reap", min_retained=1, preserve_top_fraction=0.0, preserve_outliers=False, strategy="hybrid", layer_profile="late-protect")
    assert sum(v["pruned_count"] for v in uniform["layers"].values()) == sum(v["pruned_count"] for v in late["layers"].values())
    assert late["layers"]["7"]["pruned_count"] < late["layers"]["0"]["pruned_count"]


if __name__ == "__main__":
    demo()
    print("ornith_reap_plan_test: ok")

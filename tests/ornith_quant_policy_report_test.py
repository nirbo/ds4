#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_quant_policy_report.py"
spec = importlib.util.spec_from_file_location("ornith_quant_policy_report", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        catalog = root / "catalog.json"
        policy = root / "policy.json"
        catalog.write_text(json.dumps({
            "tensors": {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": {
                    "shape": [4],
                    "nparams": 4,
                    "nbytes": 3,
                    "group": "routed_expert",
                    "shard": "a.ornq",
                },
                "model.language_model.layers.0.input_layernorm.weight": {
                    "shape": [4],
                    "nparams": 4,
                    "nbytes": 8,
                    "group": "norm",
                    "shard": "a.ornq",
                },
            }
        }), encoding="utf-8")
        policy.write_text(json.dumps({
            "name": "tiny",
            "rules": [
                {"contains": ".experts.", "quant": "q4"},
                {"contains": "layernorm", "quant": "bf16"},
            ],
            "default": "q4",
        }), encoding="utf-8")
        report = mod.run(catalog, policy, 4)
        assert report["projected_bytes"] == 12
        assert report["by_quant"]["q4"]["projected"] == 4
        assert report["by_quant"]["bf16"]["projected"] == 8


if __name__ == "__main__":
    demo()
    print("ornith_quant_policy_report_test: ok")

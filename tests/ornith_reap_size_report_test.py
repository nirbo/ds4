#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_reap_size_report.py"
spec = importlib.util.spec_from_file_location("ornith_reap_size_report", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        catalog = root / "catalog.json"
        plan = root / "plan.json"
        policy = root / "policy.json"
        catalog.write_text(json.dumps({"tensors": {
            "model.language_model.layers.0.mlp.experts.down_proj": {
                "shape": [4, 8, 2], "nbytes": 36, "group": "routed_expert", "layer": 0, "kind": "mlp.experts.down_proj"
            },
            "model.language_model.layers.0.mlp.gate.weight": {
                "shape": [4, 2], "nbytes": 16, "group": "router", "layer": 0, "kind": "mlp.gate.weight"
            },
            "model.language_model.layers.0.input_layernorm.weight": {
                "shape": [2], "nbytes": 4, "group": "norm", "layer": 0, "kind": "input_layernorm.weight"
            },
        }}), encoding="utf-8")
        plan.write_text(json.dumps({"layers": {"0": {"retained_count": 2}}}), encoding="utf-8")
        policy.write_text(json.dumps({
            "name": "tiny",
            "rules": [{"contains": ".experts.", "quant": "q4"}, {"contains": "layernorm", "quant": "bf16"}],
            "default": "q4",
        }), encoding="utf-8")
        report = mod.run(catalog, plan, policy, block=8)
        assert report["by_group"]["routed_expert"]["projected"] == 24
        assert report["by_group"]["router"]["projected"] == 4
        assert report["by_group"]["norm"]["projected"] == 4
        assert report["projected_bytes"] == 32


if __name__ == "__main__":
    demo()
    print("ornith_reap_size_report_test: ok")

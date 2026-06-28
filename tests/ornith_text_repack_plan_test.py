#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_text_repack_plan.py"
spec = importlib.util.spec_from_file_location("ornith_text_repack_plan", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    index = {
        "weight_map": {
            "model.language_model.a": "a.safetensors",
            "model.visual.b": "a.safetensors",
            "lm_head.weight": "b.safetensors",
            "model.visual.c": "c.safetensors",
        }
    }
    plan = mod.shard_plan(index)
    assert plan["shard_count"] == 3
    assert plan["text_tensor_count"] == 2
    assert plan["skipped_tensor_count"] == 2
    assert plan["actions"] == {"copy": 1, "filter": 1, "skip": 1}
    assert plan["shards"][0] == {
        "file": "a.safetensors",
        "action": "filter",
        "text_tensor_count": 1,
        "skipped_tensor_count": 1,
    }
    actions = mod.dry_run_actions(plan, Path("/src"), Path("/dst"), Path("/allow"))
    assert actions == [
        "filter /src/a.safetensors /dst/a.safetensors --allowlist /allow/a.text.allowlist # text=1 skipped=1",
        "copy /src/b.safetensors /dst/b.safetensors # text=1",
    ]


if __name__ == "__main__":
    demo()
    print("ornith_text_repack_plan_test: ok")

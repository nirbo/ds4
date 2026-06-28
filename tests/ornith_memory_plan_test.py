#!/usr/bin/env python3

import importlib.util
import sys
import json
import struct
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_memory_plan.py"
spec = importlib.util.spec_from_file_location("ornith_memory_plan", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    cfg = {
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 4,
            "vocab_size": 100,
            "num_experts": 3,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 5,
            "shared_expert_intermediate_size": 7,
            "layer_types": ["linear_attention", "full_attention"],
        }
    }
    shape = mod.shape_from_config(cfg)
    assert shape.layers == 2
    assert shape.full_attention_layers == 1
    assert shape.linear_attention_layers == 1

    buckets = mod.estimate_buckets(shape, 1000)
    assert buckets.routed_expert_params == 2 * 3 * (3 * 4 * 5)
    assert buckets.active_routed_expert_params == 2 * 1 * (3 * 4 * 5)
    assert buckets.shared_expert_params == 2 * 3 * 4 * 7
    assert buckets.router_params == 2 * 4 * 3
    assert buckets.non_routed_upper_params == 1000 - buckets.routed_expert_params

    index = {"metadata": {"total_size": 2000}}
    assert mod.total_params_from_index(index) == 1000

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        header = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "BF16",
                "shape": [4, 5],
                "data_offsets": [0, 40],
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [40, 72],
            },
            "visual.patch_embed.weight": {
                "dtype": "BF16",
                "shape": [2, 3],
                "data_offsets": [72, 84],
            },
        }
        data = json.dumps(header).encode("utf-8")
        (root / "model-00001-of-00001.safetensors").write_bytes(
            struct.pack("<Q", len(data)) + data
        )
        exact = mod.exact_buckets_from_headers(root, None)
        assert exact.param("routed_experts") == 20
        assert exact.byte("routed_experts") == 40
        assert exact.param("attention") == 16
        assert exact.param("vision") == 6


if __name__ == "__main__":
    demo()
    print("ornith_memory_plan_test: ok")

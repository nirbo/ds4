#!/usr/bin/env python3
"""Validate Ornith tensor names from a local safetensors index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LINEAR_ATTN = [
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.A_log",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
]

FULL_ATTN = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]

MOE = [
    "input_layernorm.weight",
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "post_attention_layernorm.weight",
]

GLOBAL_TEXT = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "lm_head.weight",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def text_config(config: dict) -> dict:
    return config.get("text_config") or config


def expected_tensors(config: dict) -> list[str]:
    cfg = text_config(config)
    layer_types = cfg["layer_types"]
    out = list(GLOBAL_TEXT)
    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}."
        if layer_type == "linear_attention":
            attn = LINEAR_ATTN
        elif layer_type == "full_attention":
            attn = FULL_ATTN
        else:
            raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer}")
        out.extend(prefix + name for name in attn)
        out.extend(prefix + name for name in MOE)
    return out


def visual_tensors(index: dict) -> list[str]:
    return sorted(name for name in index["weight_map"] if name.startswith("model.visual."))


def check_layout(config: dict, index: dict) -> tuple[list[str], list[str]]:
    have = set(index["weight_map"])
    want = set(expected_tensors(config))
    missing = sorted(want - have)
    unexpected_text = sorted(
        name for name in have - want
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    )
    return missing, unexpected_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--show-visual", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    index = load_json(args.index)
    cfg = text_config(config)
    missing, unexpected = check_layout(config, index)
    print(f"text layers: {cfg['num_hidden_layers']}")
    print(f"expected text tensors: {len(expected_tensors(config))}")
    print(f"index tensors: {len(index['weight_map'])}")
    print(f"visual tensors: {len(visual_tensors(index))}")
    if missing:
        print("missing:")
        for name in missing:
            print(f"  {name}")
    if unexpected:
        print("unexpected text tensors:")
        for name in unexpected:
            print(f"  {name}")
    if args.show_visual:
        print("visual:")
        for name in visual_tensors(index):
            print(f"  {name}")
    if missing or unexpected:
        return 1
    print("ornith layout: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

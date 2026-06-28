#!/usr/bin/env python3
"""Estimate Ornith compression targets from local metadata.

This intentionally does not download from Hugging Face. Give it local copies of
config.json and, optionally, model.safetensors.index.json.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


GIB = 1024**3


@dataclass(frozen=True)
class OrnithShape:
    layers: int
    hidden: int
    vocab: int
    max_context: int
    experts: int
    experts_per_token: int
    moe_intermediate: int
    shared_intermediate: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    full_attention_layers: int
    linear_attention_layers: int


@dataclass(frozen=True)
class ParamBuckets:
    total_params_hint: int
    routed_expert_params: int
    routed_gate_up_params: int
    routed_down_params: int
    active_routed_expert_params: int
    shared_expert_params: int
    router_params: int
    non_routed_upper_params: int


@dataclass(frozen=True)
class ExactBuckets:
    params: dict[str, int]
    bytes: dict[str, int]

    def param(self, name: str) -> int:
        return self.params.get(name, 0)

    def byte(self, name: str) -> int:
        return self.bytes.get(name, 0)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def text_config(config: dict) -> dict:
    return config.get("text_config") or config


def shape_from_config(config: dict) -> OrnithShape:
    cfg = text_config(config)
    layer_types = cfg.get("layer_types") or []
    return OrnithShape(
        layers=int(cfg["num_hidden_layers"]),
        hidden=int(cfg["hidden_size"]),
        vocab=int(cfg["vocab_size"]),
        max_context=int(cfg["max_position_embeddings"]),
        experts=int(cfg["num_experts"]),
        experts_per_token=int(cfg["num_experts_per_tok"]),
        moe_intermediate=int(cfg["moe_intermediate_size"]),
        shared_intermediate=int(cfg.get("shared_expert_intermediate_size", 0)),
        attention_heads=int(cfg["num_attention_heads"]),
        kv_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(cfg["head_dim"]),
        full_attention_layers=sum(1 for t in layer_types if t == "full_attention"),
        linear_attention_layers=sum(1 for t in layer_types if t == "linear_attention"),
    )


def total_params_from_index(index: dict | None) -> int:
    if not index:
        return 0
    total_size = int((index.get("metadata") or {}).get("total_size") or 0)
    return total_size // 2


def tensor_scope(name: str) -> str:
    if name.startswith(("visual.", "vision.", "vision_model.", "model.visual.")):
        return "vision"
    if name.startswith("model.language_model.") or name == "lm_head.weight":
        return "language"
    return "other"


def index_scope_counts(index: dict | None) -> dict[str, int]:
    if not index:
        return {}
    counts = Counter(tensor_scope(name) for name in index.get("weight_map", {}))
    return dict(sorted(counts.items()))


def dtype_size(dtype: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    return sizes.get(dtype.upper(), 0)


def product(values: list[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def tensor_bucket(name: str) -> str:
    if name.startswith(("visual.", "vision.", "vision_model.", "model.visual.")):
        return "vision"
    if ".experts.gate_up_proj" in name:
        return "routed_gate_up"
    if ".experts.down_proj" in name:
        return "routed_down"
    if ".experts." in name:
        return "routed_experts"
    if "shared_expert" in name:
        return "shared_experts"
    if ".self_attn." in name or ".linear_attn." in name or ".attention." in name:
        return "attention"
    if "embed_tokens" in name or name.endswith("lm_head.weight"):
        return "embedding_output"
    if "router" in name or name.endswith(".mlp.gate.weight") or ".block_sparse_moe.gate." in name:
        return "routers"
    if ".mlp." in name:
        return "other_mlp"
    return "other"


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = fp.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(header.decode("utf-8"))


def exact_buckets_from_headers(safetensors_dir: Path, index: dict | None) -> ExactBuckets:
    if index and index.get("weight_map"):
        files = sorted(set(index["weight_map"].values()))
    else:
        files = sorted(p.name for p in safetensors_dir.glob("*.safetensors"))
    params: dict[str, int] = {}
    bytes_: dict[str, int] = {}

    for filename in files:
        header = read_safetensors_header(safetensors_dir / filename)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = [int(v) for v in meta.get("shape") or []]
            count = product(shape)
            offsets = meta.get("data_offsets")
            if offsets and len(offsets) == 2:
                nbytes = int(offsets[1]) - int(offsets[0])
            else:
                nbytes = count * dtype_size(str(meta.get("dtype", "")))
            bucket = tensor_bucket(name)
            params[bucket] = params.get(bucket, 0) + count
            bytes_[bucket] = bytes_.get(bucket, 0) + nbytes
    return ExactBuckets(params=params, bytes=bytes_)


def estimate_buckets(shape: OrnithShape, total_params_hint: int = 0) -> ParamBuckets:
    gate_up_matrix = 2 * shape.hidden * shape.moe_intermediate
    down_matrix = shape.hidden * shape.moe_intermediate
    expert_matrix = gate_up_matrix + down_matrix
    routed_gate_up = shape.layers * shape.experts * gate_up_matrix
    routed_down = shape.layers * shape.experts * down_matrix
    routed = routed_gate_up + routed_down
    active_routed = shape.layers * shape.experts_per_token * expert_matrix
    shared = shape.layers * 3 * shape.hidden * shape.shared_intermediate
    router = shape.layers * shape.hidden * shape.experts
    non_routed = max(total_params_hint - routed, 0)
    return ParamBuckets(
        total_params_hint=total_params_hint,
        routed_expert_params=routed,
        routed_gate_up_params=routed_gate_up,
        routed_down_params=routed_down,
        active_routed_expert_params=active_routed,
        shared_expert_params=shared,
        router_params=router,
        non_routed_upper_params=non_routed,
    )


def bytes_for_bits(params: int, bits: float, overhead: float = 1.0) -> float:
    return params * bits / 8.0 * overhead


def full_attention_kv_bytes(shape: OrnithShape, ctx: int, bits: float) -> float:
    values_per_token_layer = 2 * shape.kv_heads * shape.head_dim
    return shape.full_attention_layers * ctx * values_per_token_layer * bits / 8.0


def gib(n: float) -> float:
    return n / GIB


def fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return str(n)


def print_exact_buckets(exact: ExactBuckets | None) -> None:
    if not exact:
        return
    print("Exact local tensor buckets")
    for name in sorted(exact.params):
        print(f"  {name}: {fmt_params(exact.params[name])} params, {gib(exact.bytes[name]):.2f} GiB BF16/source bytes")
    print()


def print_index_scopes(index: dict | None) -> None:
    counts = index_scope_counts(index)
    if not counts:
        return
    print("Index tensor scopes")
    for name, count in counts.items():
        print(f"  {name}: {count} tensors")
    print("  byte split unavailable without safetensors headers")
    print()


def recipe_params(buckets: ParamBuckets, exact: ExactBuckets | None) -> tuple[int, int]:
    if not exact:
        return buckets.routed_expert_params, buckets.non_routed_upper_params
    routed = exact.param("routed_experts") + exact.param("routed_gate_up") + exact.param("routed_down")
    non_routed = sum(v for k, v in exact.params.items() if k not in ("routed_experts", "routed_gate_up", "routed_down", "vision"))
    return routed, non_routed


def expert_recipe_bytes(buckets: ParamBuckets, gate_up_bits: float, down_bits: float, overhead: float) -> float:
    return (
        bytes_for_bits(buckets.routed_gate_up_params, gate_up_bits, overhead) +
        bytes_for_bits(buckets.routed_down_params, down_bits, overhead)
    )


def exact_expert_recipe_bytes(exact: ExactBuckets, gate_up_bits: float, down_bits: float, overhead: float) -> float:
    gate_up = exact.param("routed_gate_up")
    down = exact.param("routed_down")
    other = exact.param("routed_experts")
    if gate_up or down:
        return (
            bytes_for_bits(gate_up, gate_up_bits, overhead) +
            bytes_for_bits(down, down_bits, overhead) +
            bytes_for_bits(other, max(gate_up_bits, down_bits), overhead)
        )
    return bytes_for_bits(other, gate_up_bits, overhead)


def print_plan(
    shape: OrnithShape,
    buckets: ParamBuckets,
    exact: ExactBuckets | None,
    index: dict | None,
    args: argparse.Namespace,
) -> None:
    print("Ornith shape")
    print(f"  layers: {shape.layers}")
    print(f"  hidden: {shape.hidden}")
    print(f"  vocab: {shape.vocab}")
    print(f"  max context: {shape.max_context}")
    print(f"  experts: {shape.experts}")
    print(f"  experts/token: {shape.experts_per_token}")
    print(f"  MoE intermediate: {shape.moe_intermediate}")
    print(f"  attention heads: {shape.attention_heads} q, {shape.kv_heads} kv, head_dim {shape.head_dim}")
    print(f"  attention layers: {shape.linear_attention_layers} linear, {shape.full_attention_layers} full")
    print()

    print("Parameter buckets")
    if buckets.total_params_hint:
        print(f"  total checkpoint hint: {fmt_params(buckets.total_params_hint)} params")
    else:
        print("  total checkpoint hint: unavailable")
    print(f"  routed experts: {fmt_params(buckets.routed_expert_params)} params")
    print(f"    gate/up: {fmt_params(buckets.routed_gate_up_params)} params")
    print(f"    down: {fmt_params(buckets.routed_down_params)} params")
    print(f"  active routed experts/token: {fmt_params(buckets.active_routed_expert_params)} params")
    print(f"  shared experts: {fmt_params(buckets.shared_expert_params)} params")
    print(f"  routers: {fmt_params(buckets.router_params)} params")
    if buckets.total_params_hint:
        print(f"  non-routed upper bound: {fmt_params(buckets.non_routed_upper_params)} params")
    print()
    print_index_scopes(index)
    print_exact_buckets(exact)

    print("Memory recipes")
    print("  expert_recipe  nonrouted_bits  estimated_weights")
    routed_params, non_routed_params = recipe_params(buckets, exact)
    recipes = args.expert_recipes
    recipe_weights: list[tuple[str, float]] = []
    for gate_up_bits, down_bits in recipes:
        if exact:
            routed_bytes = exact_expert_recipe_bytes(exact, gate_up_bits, down_bits, args.routed_overhead)
        else:
            routed_bytes = expert_recipe_bytes(buckets, gate_up_bits, down_bits, args.routed_overhead)
        recipe_name = f"gu{gate_up_bits:g}/d{down_bits:g}"
        if non_routed_params:
            non_routed_bytes = bytes_for_bits(
                non_routed_params,
                args.nonrouted_bits,
                args.nonrouted_overhead,
            )
            total = routed_bytes + non_routed_bytes
            recipe_weights.append((recipe_name, total))
            print(f"  {recipe_name:>13}  {args.nonrouted_bits:>14g}  {gib(total):8.2f} GiB")
        else:
            recipe_weights.append((recipe_name, routed_bytes))
            print(f"  {recipe_name:>13}  {'?':>14}  routed-only {gib(routed_bytes):.2f} GiB")
    print()

    print("Full-attention KV cache")
    print(f"  kv bits: {args.kv_bits:g}")
    for ctx in args.contexts:
        if ctx > shape.max_context:
            continue
        kv = full_attention_kv_bytes(shape, ctx, args.kv_bits)
        print(f"  ctx {ctx:>6}: {gib(kv):6.2f} GiB")
    print("  linear-attention recurrent state is not included; it should be constant-size, not O(context).")
    print()

    print("Weights + full-attention KV + scratch")
    print(f"  scratch reserve: {args.scratch_gib:g} GiB")
    for recipe_name, weight_bytes in recipe_weights:
        vals = []
        for ctx in args.contexts:
            if ctx > shape.max_context:
                continue
            total = weight_bytes + full_attention_kv_bytes(shape, ctx, args.kv_bits) + args.scratch_gib * GIB
            vals.append(f"ctx{ctx}={gib(total):.1f}GiB")
        print(f"  {recipe_name}: " + ", ".join(vals))
    print()

    print("Fit targets")
    for target in args.targets:
        print(f"  {target:g} GiB: {'target, not proof'}")


def parse_bits_csv(s: str) -> list[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("empty bit list")
    return out


def parse_int_csv(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("empty integer list")
    return out


def parse_expert_recipes(s: str) -> list[tuple[float, float]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "/" in part:
            gu, down = part.split("/", 1)
            out.append((float(gu), float(down)))
        else:
            bits = float(part)
            out.append((bits, bits))
    if not out:
        raise argparse.ArgumentTypeError("empty expert recipe list")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--index", type=Path)
    p.add_argument("--safetensors-dir", type=Path)
    p.add_argument("--routed-bits", type=parse_bits_csv, default=parse_bits_csv("1,1.5,2"))
    p.add_argument("--expert-recipes", type=parse_expert_recipes, default=parse_expert_recipes("1/1,1/2,1.5/1.5,2/2"))
    p.add_argument("--nonrouted-bits", type=float, default=4.0)
    p.add_argument("--kv-bits", type=float, default=16.0)
    p.add_argument("--routed-overhead", type=float, default=1.08)
    p.add_argument("--nonrouted-overhead", type=float, default=1.03)
    p.add_argument("--scratch-gib", type=float, default=4.0)
    p.add_argument("--contexts", type=parse_int_csv, default=parse_int_csv("8192,32768,65536,262144"))
    p.add_argument("--targets", type=parse_bits_csv, default=parse_bits_csv("32,64,96,128"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    index = load_json(args.index) if args.index else None
    shape = shape_from_config(config)
    buckets = estimate_buckets(shape, total_params_from_index(index))
    exact = exact_buckets_from_headers(args.safetensors_dir, index) if args.safetensors_dir else None
    print_plan(shape, buckets, exact, index, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure routed gate/up quantization error on captured real Ornith activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from ornith_ornq_validate import read_ornq
from ornith_quant_formats import IQ2_XXS_GRID, quant_bytes
from ornith_safetensors_filter import read_header


TENSOR = "model.language_model.layers.0.mlp.experts.gate_up_proj"


def decode_weight(data: bytes, mode: str, rows: int, ncols: int):
    import numpy as np

    if ncols % 256:
        raise ValueError(f"{mode} columns must be 256-aligned")
    blocks = rows * (ncols // 256)
    block_bytes = {"iq2_xxs": 66, "q2_k": 84, "q4_k": 144}.get(mode)
    if block_bytes is None:
        raise ValueError(f"unsupported activation-error quant: {mode}")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(blocks, block_bytes)
    output = np.empty((blocks, 256), dtype=np.float32)
    if mode == "q2_k":
        d = packed[:, 80:82].copy().view("<f2").reshape(blocks).astype(np.float32)
        dmin = packed[:, 82:84].copy().view("<f2").reshape(blocks).astype(np.float32)
        for index in range(256):
            group = index // 16
            rem = index & 127
            q = (packed[:, 16 + (index // 128) * 32 + (rem & 31)] >> ((rem // 32) * 2)) & 3
            output[:, index] = d * (packed[:, group] & 15) * q - dmin * (packed[:, group] >> 4)
        return output.reshape(rows, ncols)
    if mode == "q4_k":
        d = packed[:, :2].copy().view("<f2").reshape(blocks).astype(np.float32)
        dmin = packed[:, 2:4].copy().view("<f2").reshape(blocks).astype(np.float32)
        scales = packed[:, 4:16]
        for index in range(256):
            group = index // 32
            if group < 4:
                scale = scales[:, group] & 63
                minimum = scales[:, group + 4] & 63
            else:
                scale = (scales[:, group + 4] & 15) | ((scales[:, group - 4] >> 6) << 4)
                minimum = (scales[:, group + 4] >> 4) | ((scales[:, group] >> 6) << 4)
            rem = index & 63
            q = (packed[:, 16 + (index // 64) * 32 + (rem & 31)] >> (4 if rem >= 32 else 0)) & 15
            output[:, index] = d * scale * q - dmin * minimum
        return output.reshape(rows, ncols)
    scales = packed[:, :2].copy().view("<f2").reshape(blocks).astype(np.float32)
    grid_table = np.asarray(IQ2_XXS_GRID, dtype=np.uint16)
    for group in range(8):
        offset = 2 + group * 8
        grids = packed[:, offset : offset + 4].copy().view("<u4").reshape(blocks)
        aux = packed[:, offset + 4 : offset + 8].copy().view("<u4").reshape(blocks)
        group_scale = scales * (0.5 + (aux >> 28).astype(np.float32))
        for subgroup in range(4):
            grid = grid_table[(grids >> (8 * subgroup)) & 255]
            for item in range(8):
                value = group_scale * (2 * ((grid >> (2 * item)) & 3).astype(np.float32) + 1)
                negative = (aux & (1 << (7 * subgroup + item))) != 0
                output[:, group * 32 + subgroup * 8 + item] = np.where(negative, -value, value)
    return output.reshape(rows, ncols)


def relative_l2(torch, reference, candidate) -> float:
    delta = reference.float() - candidate.float()
    denominator = torch.linalg.vector_norm(reference.float())
    return float((torch.linalg.vector_norm(delta) / denominator).item()) if denominator else 0.0


def run(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    raw_header, _raw_data = read_header(args.source)
    ornq_header, ornq_data = read_ornq(args.ornq)
    raw_meta = raw_header[args.tensor]
    quant_meta = ornq_header["tensors"][args.tensor]
    mode = quant_meta["quant"]
    if mode not in ("iq2_xxs", "q2_k", "q4_k"):
        raise ValueError(f"unsupported routed expert quant: {mode}")
    retained = [int(value) for value in quant_meta.get("reap_retained_experts", [])]
    if not retained:
        retained = list(range(int(raw_meta["shape"][0])))
    retained_position = {expert: position for position, expert in enumerate(retained)}
    rows = int(quant_meta["shape"][1])
    ncols = int(quant_meta["shape"][2])
    expert_bytes = quant_bytes(rows * ncols, mode, int(ornq_header["block_size"]))
    quant_base = ornq_data + int(quant_meta["data_offsets"][0])

    trace = torch.load(args.trace, map_location="cpu", weights_only=False)
    inputs = torch.cat(trace["moe_inputs"], dim=0)
    indices = torch.cat(trace["router_indices"], dim=0)
    frequencies = []
    for expert in retained:
        count = int((indices == expert).sum().item())
        if count:
            frequencies.append((count, expert))
    selected = [expert for _count, expert in sorted(frequencies, reverse=True)[: args.max_experts]]
    if not selected:
        raise ValueError("none of the retained experts were selected in the trace")

    reports = []
    projection_ref_sq = projection_err_sq = 0.0
    intermediate_ref_sq = intermediate_err_sq = 0.0
    with safe_open(args.source, framework="pt", device="cpu") as source, args.ornq.open("rb") as quant_file:
        raw_slice = source.get_slice(args.tensor)
        with torch.inference_mode():
            for expert in selected:
                token_rows = torch.where((indices == expert).any(dim=1))[0]
                current = inputs[token_rows].to(device=args.device, dtype=torch.bfloat16)
                raw_weight = raw_slice[expert : expert + 1][0].to(device=args.device)
                quant_file.seek(quant_base + retained_position[expert] * expert_bytes)
                packed = quant_file.read(expert_bytes)
                if len(packed) != expert_bytes:
                    raise EOFError(f"short IQ2 expert read: {expert}")
                quant_weight = torch.from_numpy(decode_weight(packed, mode, rows, ncols)).to(
                    device=args.device, dtype=torch.bfloat16
                )
                raw_projection = F.linear(current, raw_weight)
                quant_projection = F.linear(current, quant_weight)
                raw_gate, raw_up = raw_projection.chunk(2, dim=-1)
                quant_gate, quant_up = quant_projection.chunk(2, dim=-1)
                raw_intermediate = F.silu(raw_gate) * raw_up
                quant_intermediate = F.silu(quant_gate) * quant_up

                projection_delta = raw_projection.float() - quant_projection.float()
                intermediate_delta = raw_intermediate.float() - quant_intermediate.float()
                projection_ref_sq += float(raw_projection.float().square().sum().item())
                projection_err_sq += float(projection_delta.square().sum().item())
                intermediate_ref_sq += float(raw_intermediate.float().square().sum().item())
                intermediate_err_sq += float(intermediate_delta.square().sum().item())
                reports.append(
                    {
                        "expert": expert,
                        "tokens": int(token_rows.numel()),
                        "projection_relative_l2": relative_l2(torch, raw_projection, quant_projection),
                        "intermediate_relative_l2": relative_l2(torch, raw_intermediate, quant_intermediate),
                        "projection_max_abs": float(projection_delta.abs().max().item()),
                        "intermediate_max_abs": float(intermediate_delta.abs().max().item()),
                    }
                )
                del current, raw_weight, quant_weight, raw_projection, quant_projection

    report = {
        "format": "ornith-expert-activation-error-v1",
        "source": str(args.source),
        "ornq": str(args.ornq),
        "trace": str(args.trace),
        "tensor": args.tensor,
        "quant": mode,
        "device": args.device,
        "sampled_experts": len(reports),
        "sampled_token_expert_pairs": sum(row["tokens"] for row in reports),
        "projection_relative_l2": (projection_err_sq / projection_ref_sq) ** 0.5,
        "intermediate_relative_l2": (intermediate_err_sq / intermediate_ref_sq) ** 0.5,
        "experts": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--ornq", required=True, type=Path)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tensor", default=TENSOR)
    p.add_argument("--max-experts", type=int, default=16)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = p.parse_args()
    report = run(args)
    print(
        f"experts={report['sampled_experts']} pairs={report['sampled_token_expert_pairs']} "
        f"projection_relative_l2={report['projection_relative_l2']:.6g} "
        f"intermediate_relative_l2={report['intermediate_relative_l2']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

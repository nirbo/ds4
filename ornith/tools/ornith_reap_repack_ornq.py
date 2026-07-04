#!/usr/bin/env python3
"""Materialize a REAP expert-retention plan by copying/pruning existing .ornq shards."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from ornith_ornq_validate import read_ornq
from ornith_quantize_safetensors import product, quant_bytes
from ornith_runtime import classify_tensor


MAGIC = b"ORNQ1\0\0\0"


def is_prunable(meta: dict) -> bool:
    return meta.get("group") == "routed_expert" or (meta.get("group") == "router" and meta.get("kind") == "mlp.gate.weight")


def retained_for(meta: dict, plan: dict) -> list[int] | None:
    layer = meta.get("layer")
    if layer is None or str(layer) not in plan["layers"] or not is_prunable(meta):
        return None
    return [int(v) for v in plan["layers"][str(layer)]["retained"]]


def classify_meta(name: str, meta: dict) -> dict:
    out = dict(meta)
    role = classify_tensor(name)
    out.setdefault("layer", role.layer)
    out.setdefault("group", role.group)
    out.setdefault("kind", role.kind)
    return out


def slice_bytes(meta: dict, block: int) -> int:
    shape = [int(v) for v in meta["shape"]]
    if len(shape) < 2:
        raise ValueError(f"{meta}: cannot slice rank < 2")
    n = product(shape[1:])
    if meta["quant"] != "bf16" and n % block:
        raise ValueError(f"slice params {n} not divisible by block {block}")
    return quant_bytes(n, meta["quant"], block)


def copy_range(src, dst, src_off: int, n: int) -> None:
    src.seek(src_off)
    remaining = n
    while remaining:
        chunk = src.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise EOFError("short read")
        dst.write(chunk)
        remaining -= len(chunk)


def repack_one(src: Path, dst: Path, plan: dict) -> dict:
    header, data_start = read_ornq(src)
    block = int(header["block_size"])
    out_header = dict(header)
    out_header["source"] = src.name
    out_header["reap_plan"] = plan.get("format", "ornith-reap-plan")
    out_header["tensors"] = {}
    jobs = []
    pruned = 0
    offset = 0
    for name, meta in sorted(header["tensors"].items()):
        meta = classify_meta(name, meta)
        keep = retained_for(meta, plan)
        start, end = [int(v) for v in meta["data_offsets"]]
        if keep is None:
            nbytes = end - start
            jobs.append((start, nbytes, None))
        else:
            sb = slice_bytes(meta, block)
            old_shape = [int(v) for v in meta["shape"]]
            if not keep:
                raise ValueError(f"{name}: retained expert list is empty")
            if len(set(keep)) != len(keep):
                raise ValueError(f"{name}: duplicate retained expert id")
            if min(keep) < 0 or max(keep) >= old_shape[0]:
                raise ValueError(f"{name}: retained expert id outside shape {old_shape[0]}")
            meta["shape"] = [len(keep)] + old_shape[1:]
            meta["reap_retained_experts"] = keep
            nbytes = sb * len(keep)
            jobs.append((start, sb, keep))
            pruned += 1
        meta["data_offsets"] = [offset, offset + nbytes]
        out_header["tensors"][name] = meta
        offset += nbytes
    encoded = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    tmp = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as sfp, tmp.open("wb") as dfp:
        dfp.write(MAGIC)
        dfp.write(struct.pack("<Q", len(encoded)))
        dfp.write(encoded)
        for start, nbytes, keep in jobs:
            if keep is None:
                copy_range(sfp, dfp, data_start + start, nbytes)
            else:
                for expert in keep:
                    copy_range(sfp, dfp, data_start + start + expert * nbytes, nbytes)
    tmp.replace(dst)
    return {"file": dst.name, "bytes": dst.stat().st_size, "tensors": len(out_header["tensors"]), "pruned_tensors": pruned}


def run(src_dir: Path, dst_dir: Path, plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shards = []
    for src in sorted(src_dir.glob("*.ornq")):
        shards.append(repack_one(src, dst_dir / src.name, plan))
    if not shards:
        raise ValueError(f"{src_dir}: no .ornq files")
    return {"format": "ornith-reap-repack-report-v1", "source": str(src_dir), "dest": str(dst_dir), "plan": str(plan_path), "shards": shards}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", required=True, type=Path)
    p.add_argument("--dst-dir", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    report = run(args.src_dir, args.dst_dir, args.plan)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"repacked shards={len(report['shards'])} dest={args.dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

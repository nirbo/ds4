#!/usr/bin/env python3
"""Measure DS4-style candidate quantization error on raw Ornith BF16 tensors."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from ornith_safetensors_filter import read_header
from ornith_quantize_safetensors import product


ROOT = Path(__file__).resolve().parents[2]
RAW_C = ROOT / "ornith" / "tools" / "ornith_ds4_candidate_error.c"
QUANTS_C = ROOT / "ornith" / "tools" / "ornith_ds4_quants.c"


def compile_raw_tool(out: Path) -> Path:
    newest = max(RAW_C.stat().st_mtime, QUANTS_C.stat().st_mtime)
    if out.exists() and out.stat().st_mtime >= newest:
        return out
    subprocess.run(
        ["cc", "-O3", "-std=c11", "-pthread", str(RAW_C), str(QUANTS_C), "-lm", "-o", str(out)],
        check=True,
    )
    return out


def parse_stats(line: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for item in line.strip().split():
        key, value = item.split("=", 1)
        if key in {"count", "row_size"}:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def candidate_size(nrows: int, row_size: int) -> int:
    return nrows * row_size


def run_candidate(raw_tool: Path, source: Path, source_data_start: int, meta: dict, qtype: str, threads: int, progress_rows: int, experts: list[int] | None = None) -> dict:
    shape = [int(v) for v in meta["shape"]]
    if len(shape) < 2:
        raise ValueError(f"candidate tensor must have at least 2 dims, got {shape}")
    ncols = shape[-1]
    full_nrows = product(shape[:-1])
    selected = experts
    rows_per_expert = product(shape[1:-1]) if len(shape) == 3 else full_nrows
    if selected is not None and len(shape) != 3:
        raise ValueError("--expert requires a rank-3 fused expert tensor")
    if selected is not None and (not selected or min(selected) < 0 or max(selected) >= shape[0]):
        raise ValueError(f"invalid expert selection for shape {shape}: {selected}")
    ranges = [(0, full_nrows)] if selected is None else [(expert * rows_per_expert, rows_per_expert) for expert in selected]
    totals: dict[str, float | int] = {"count": 0, "sum_abs_src": 0.0, "sum_abs_err": 0.0, "sum_sq_src": 0.0, "sum_sq_err": 0.0, "max_abs": 0.0, "row_size": 0}
    for row_offset, nrows in ranges:
        cmd = [
            str(raw_tool), str(source), qtype,
            str(source_data_start + int(meta["data_offsets"][0]) + row_offset * ncols * 2),
            str(nrows), str(ncols), str(threads), str(progress_rows),
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.stderr:
            print(proc.stderr, end="")
        if proc.returncode:
            raise RuntimeError(f"{qtype} candidate failed rc={proc.returncode}: {proc.stderr.strip()}")
        current = parse_stats(proc.stdout)
        for key in ("count", "sum_abs_src", "sum_abs_err", "sum_sq_src", "sum_sq_err"):
            totals[key] += current[key]
        totals["max_abs"] = max(float(totals["max_abs"]), float(current["max_abs"]))
        totals["row_size"] = current["row_size"]
    stats = totals
    nrows = sum(count for _, count in ranges)
    count = int(stats["count"])
    sum_abs_src = float(stats["sum_abs_src"])
    sum_abs_err = float(stats["sum_abs_err"])
    sum_sq_src = float(stats["sum_sq_src"])
    sum_sq_err = float(stats["sum_sq_err"])
    row_size = int(stats["row_size"])
    return {
        "type": qtype,
        "shape": shape if selected is None else [len(selected), *shape[1:]],
        "nrows": nrows,
        "ncols": ncols,
        "nparams": nrows * ncols,
        "row_size": row_size,
        "bytes": candidate_size(nrows, row_size),
        "bits_per_param": candidate_size(nrows, row_size) * 8 / (nrows * ncols),
        "count": count,
        "mean_abs_src": sum_abs_src / count,
        "mean_abs_err": sum_abs_err / count,
        "rmse": math.sqrt(sum_sq_err / count),
        "relative_l2": math.sqrt(sum_sq_err / sum_sq_src) if sum_sq_src else 0.0,
        "max_abs": float(stats["max_abs"]),
        "experts": selected,
        "synthetic_imatrix_scope": "per_expert" if selected is not None and qtype == "iq2_xxs" else ("tensor" if qtype == "iq2_xxs" else None),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Ornith DS4-Style Quant Candidate Error",
        "",
        f"- source: `{report['source']}`",
        f"- tensor: `{report['tensor']}`",
        f"- shape: `{report['shape']}`",
        "",
        "| type | bits/param | bytes | mean_abs_err | rmse | relative_l2 | max_abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["candidates"]:
        lines.append(
            f"| {row['type']} | {row['bits_per_param']:.6g} | {row['bytes']} | "
            f"{row['mean_abs_err']:.6g} | {row['rmse']:.6g} | "
            f"{row['relative_l2']:.6g} | {row['max_abs']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source: Path, tensor: str, types: list[str], out_json: Path, out_md: Path, threads: int, progress_rows: int, experts: list[int] | None = None) -> dict:
    header, source_data_start = read_header(source)
    if tensor not in header:
        raise ValueError(f"{tensor}: missing from {source}")
    meta = header[tensor]
    if meta.get("dtype") != "BF16":
        raise ValueError(f"{tensor}: source dtype {meta.get('dtype')} != BF16")
    raw_tool = compile_raw_tool(out_json.parent / "ornith_ds4_candidate_error")
    rows = []
    for qtype in types:
        print(f"candidate tensor={tensor} type={qtype}", flush=True)
        rows.append(run_candidate(raw_tool, source, source_data_start, meta, qtype, threads, progress_rows, experts))
    report = {
        "source": str(source),
        "tensor": tensor,
        "shape": [int(v) for v in meta["shape"]],
        "candidates": rows,
        "experts": experts,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out_md, report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--tensor", required=True)
    p.add_argument("--type", action="append", dest="types", choices=["iq2_xxs", "q2_k", "q4_k"])
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--progress-rows", type=int, default=32768)
    p.add_argument("--expert", action="append", type=int, dest="experts", help="rank-3 expert id; repeat to test a per-expert synthetic imatrix sample")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    types = args.types or ["iq2_xxs", "q2_k", "q4_k"]
    run(args.source, args.tensor, types, args.out_json, args.out_md, args.threads, args.progress_rows, args.experts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

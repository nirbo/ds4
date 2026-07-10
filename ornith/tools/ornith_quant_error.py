#!/usr/bin/env python3
"""Compare raw BF16 safetensors against dequantized Ornith .ornq tensors."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from ornith_ornq_validate import read_ornq
from ornith_safetensors_filter import read_header
from ornith_quantize_safetensors import product


ROOT = Path(__file__).resolve().parents[2]
RAW_C = ROOT / "ornith" / "tools" / "ornith_quant_error_raw.c"


def compile_raw_tool(out: Path) -> Path:
    if out.exists() and out.stat().st_mtime >= RAW_C.stat().st_mtime:
        return out
    subprocess.run(["cc", "-O3", "-std=c11", "-pthread", str(RAW_C), "-lm", "-o", str(out)], check=True)
    return out


def classify(name: str) -> str:
    if ".experts." in name:
        return "routed_expert"
    if ".shared_expert" in name:
        return "shared_expert"
    if ".mlp.gate" in name:
        return "router"
    if "attn" in name:
        return "attention"
    if "norm" in name:
        return "norm"
    return "global"


def parse_stats(line: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for item in line.strip().split():
        key, value = item.split("=", 1)
        out[key] = int(value) if key == "count" else float(value)
    return out


def compare_tensor(raw_tool: Path, source: Path, ornq: Path, source_data_start: int, ornq_data_start: int, source_meta: dict, ornq_meta: dict, block: int, threads: int, progress: int, imatrix_entry: dict | None = None) -> dict:
    shape = [int(v) for v in ornq_meta["shape"]]
    nparams = product(shape)
    retained = [int(v) for v in ornq_meta.get("reap_retained_experts", [])]
    source_slice_params = product([int(v) for v in source_meta["shape"][1:]]) if retained or imatrix_entry else 0
    cmd = [
        str(raw_tool),
        str(source),
        str(ornq),
        ornq_meta["quant"],
        str(source_data_start + int(source_meta["data_offsets"][0])),
        str(ornq_data_start + int(ornq_meta["data_offsets"][0])),
        str(nparams),
        str(block),
        str(threads),
        str(progress),
        str(source_slice_params),
        ",".join(str(v) for v in retained) if retained else "-",
    ]
    if imatrix_entry is not None:
        ncols = int(source_meta["shape"][-1])
        cmd += [str(imatrix_entry["path"]), str(ncols)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stderr:
        print(proc.stderr, end="")
    if proc.returncode:
        raise RuntimeError(f"{ornq_meta['quant']} compare failed rc={proc.returncode}: {proc.stderr.strip()}")
    stats = parse_stats(proc.stdout)
    count = int(stats["count"])
    sum_abs_src = float(stats["sum_abs_src"])
    sum_abs_err = float(stats["sum_abs_err"])
    sum_sq_src = float(stats["sum_sq_src"])
    sum_sq_err = float(stats["sum_sq_err"])
    row = {
        "quant": ornq_meta["quant"],
        "shape": shape,
        "nparams": nparams,
        "count": count,
        "mean_abs_src": sum_abs_src / count,
        "mean_abs_err": sum_abs_err / count,
        "rmse": math.sqrt(sum_sq_err / count),
        "relative_l2": math.sqrt(sum_sq_err / sum_sq_src) if sum_sq_src else 0.0,
        "max_abs": float(stats["max_abs"]),
    }
    weighted_src = float(stats.get("sum_weighted_sq_src", 0.0))
    weighted_err = float(stats.get("sum_weighted_sq_err", 0.0))
    row["activation_weighted_relative_l2"] = math.sqrt(weighted_err / weighted_src) if weighted_src else None
    return row


def aggregate(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in rows:
        name = row[key]
        group = groups.setdefault(name, {
            key: name,
            "tensors": 0,
            "nparams": 0,
            "sum_mean_abs_err_weighted": 0.0,
            "sum_sq_err": 0.0,
            "sum_sq_src": 0.0,
            "max_abs": 0.0,
        })
        n = int(row["nparams"])
        group["tensors"] += 1
        group["nparams"] += n
        group["sum_mean_abs_err_weighted"] += float(row["mean_abs_err"]) * n
        group["sum_sq_err"] += float(row["rmse"]) ** 2 * n
        src_rms = float(row["mean_abs_src"])
        rel = float(row["relative_l2"])
        if rel:
            group["sum_sq_src"] += (float(row["rmse"]) / rel) ** 2 * n
        elif float(row["rmse"]) == 0.0:
            group["sum_sq_src"] += src_rms * src_rms * n
        group["max_abs"] = max(group["max_abs"], float(row["max_abs"]))
    out = []
    for group in groups.values():
        n = int(group["nparams"])
        out.append({
            key: group[key],
            "tensors": group["tensors"],
            "nparams": n,
            "mean_abs_err": group["sum_mean_abs_err_weighted"] / n if n else 0.0,
            "rmse": math.sqrt(group["sum_sq_err"] / n) if n else 0.0,
            "relative_l2": math.sqrt(group["sum_sq_err"] / group["sum_sq_src"]) if group["sum_sq_src"] else 0.0,
            "max_abs": group["max_abs"],
        })
    return sorted(out, key=lambda item: item[key])


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Ornith Quantization Error Report",
        "",
        f"- source: `{report['source']}`",
        f"- ornq: `{report['ornq']}`",
        f"- tensors: {len(report['tensors'])}",
        "",
        "## By Quant",
        "",
        "| quant | tensors | params | mean_abs_err | rmse | relative_l2 | max_abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_quant"]:
        lines.append(f"| {row['quant']} | {row['tensors']} | {row['nparams']} | {row['mean_abs_err']:.6g} | {row['rmse']:.6g} | {row['relative_l2']:.6g} | {row['max_abs']:.6g} |")
    lines += [
        "",
        "## Tensors",
        "",
        "| quant | group | params | mean_abs_src | mean_abs_err | rmse | relative_l2 | activation_weighted_relative_l2 | max_abs | name |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["tensors"]:
        weighted = row.get("activation_weighted_relative_l2")
        weighted_text = f"{weighted:.6g}" if weighted is not None else "-"
        lines.append(f"| {row['quant']} | {row['group']} | {row['nparams']} | {row['mean_abs_src']:.6g} | {row['mean_abs_err']:.6g} | {row['rmse']:.6g} | {row['relative_l2']:.6g} | {weighted_text} | {row['max_abs']:.6g} | `{row['name']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source: Path, ornq: Path, out_json: Path, out_md: Path, threads: int, progress: int, imatrix: Path | None = None) -> dict:
    source_header, source_data_start = read_header(source)
    ornq_header, ornq_data_start = read_ornq(ornq)
    block = int(ornq_header["block_size"])
    raw_tool = compile_raw_tool(out_json.parent / "ornith_quant_error_raw")
    imatrix_entries = {}
    if imatrix is not None:
        manifest = json.loads(imatrix.read_text(encoding="utf-8"))
        if manifest.get("format") != "ornith-imatrix-v1":
            raise ValueError(f"unsupported imatrix format: {manifest.get('format')!r}")
        for name, entry in manifest.get("tensors", {}).items():
            path = Path(entry["file"])
            if not path.is_absolute():
                path = imatrix.parent / path
            imatrix_entries[name] = {**entry, "path": path}
    rows = []
    for name, ornq_meta in sorted(ornq_header["tensors"].items()):
        if name not in source_header:
            raise ValueError(f"{name}: missing from source")
        source_meta = source_header[name]
        if source_meta.get("dtype") != "BF16":
            raise ValueError(f"{name}: source dtype {source_meta.get('dtype')} != BF16")
        retained = ornq_meta.get("reap_retained_experts")
        if retained is None and list(source_meta["shape"]) != list(ornq_meta["shape"]):
            raise ValueError(f"{name}: shape mismatch")
        if retained is not None:
            if list(source_meta["shape"][1:]) != list(ornq_meta["shape"][1:]) or len(retained) != int(ornq_meta["shape"][0]):
                raise ValueError(f"{name}: invalid REAP source mapping")
        print(f"compare name={name} quant={ornq_meta['quant']} params={product([int(v) for v in ornq_meta['shape']])}", flush=True)
        row = compare_tensor(raw_tool, source, ornq, source_data_start, ornq_data_start, source_meta, ornq_meta, block, threads, progress, imatrix_entries.get(name))
        row["name"] = name
        row["group"] = classify(name)
        rows.append(row)
    report = {
        "source": str(source),
        "ornq": str(ornq),
        "block_size": block,
        "tensors": rows,
        "by_quant": aggregate(rows, "quant"),
        "by_group": aggregate(rows, "group"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out_md, report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--ornq", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--progress", type=int, default=128 * 1024 * 1024)
    p.add_argument("--imatrix", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run(args.source, args.ornq, args.out_json, args.out_md, args.threads, args.progress, args.imatrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

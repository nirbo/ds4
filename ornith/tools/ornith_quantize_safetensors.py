#!/usr/bin/env python3
"""Quantize BF16 safetensors into the experimental Ornith compact format."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import time
import os
from pathlib import Path

from ornith_safetensors_filter import read_header


ROOT = Path(__file__).resolve().parents[2]
RAW_C = ROOT / "ornith" / "tools" / "ornith_quantize_bf16_raw.c"


def log(path: Path | None, message: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + message
    print(line, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def quant_mode(name: str) -> str:
    if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
        return "iq1"
    return "q4"


def quant_bytes(nparams: int, mode: str, block: int) -> int:
    blocks = math.ceil(nparams / block)
    if mode == "iq1":
        return sum(2 + math.ceil(min(block, nparams - i * block) / 8) for i in range(blocks))
    if mode == "q4":
        return sum(2 + math.ceil(min(block, nparams - i * block) / 2) for i in range(blocks))
    raise ValueError(f"unsupported mode: {mode}")


def compile_raw_tool(out: Path) -> Path:
    if out.exists() and out.stat().st_mtime >= RAW_C.stat().st_mtime:
        return out
    subprocess.run(["cc", "-O3", "-std=c11", "-pthread", str(RAW_C), "-lm", "-o", str(out)], check=True)
    return out


def build_header(src: Path, block: int) -> tuple[dict, list[dict], int]:
    header, data_start = read_header(src)
    out = {
        "format": "ornith-quant-smoke-v1",
        "source": src.name,
        "block_size": block,
        "tensors": {},
    }
    if "__metadata__" in header:
        out["source_metadata"] = header["__metadata__"]
    jobs = []
    offset = 0
    for name in sorted(k for k in header if k != "__metadata__"):
        meta = header[name]
        if meta.get("dtype") != "BF16":
            raise ValueError(f"{name}: only BF16 is supported")
        shape = [int(v) for v in meta["shape"]]
        nparams = product(shape)
        start, end = [int(v) for v in meta["data_offsets"]]
        if end - start != nparams * 2:
            raise ValueError(f"{name}: BF16 byte size mismatch")
        mode = quant_mode(name)
        nbytes = quant_bytes(nparams, mode, block)
        out["tensors"][name] = {
            "source_dtype": "BF16",
            "quant": mode,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        jobs.append({"name": name, "mode": mode, "byte_offset": data_start + start, "nparams": nparams})
        offset += nbytes
    return out, jobs, offset


def quantize(src: Path, dst: Path, block: int = 256, threads: int | None = None, log_path: Path | None = None) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        tmp.unlink()
    raw_tool = compile_raw_tool(dst.parent / "ornith_quantize_bf16_raw")
    header, jobs, expected_bytes = build_header(src, block)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data_start = 16 + len(encoded)
    tmp.write_bytes(b"ORNQ1\0\0\0" + struct.pack("<Q", len(encoded)) + encoded)
    with tmp.open("ab") as fp:
        fp.truncate(data_start + expected_bytes)
    started = time.time()
    threads = threads or max((os.cpu_count() or 4) - 2, 1)
    log(log_path, f"quant-start src={src} dst={dst} tensors={len(jobs)} expected_payload={expected_bytes}")
    for job in jobs:
        out_offset = data_start + header["tensors"][job["name"]]["data_offsets"][0]
        log(log_path, f"quant-tensor name={job['name']} mode={job['mode']} params={job['nparams']} threads={threads}")
        proc = subprocess.Popen(
            [
                str(raw_tool),
                str(src),
                str(tmp),
                job["mode"],
                str(job["byte_offset"]),
                str(out_offset),
                str(job["nparams"]),
                str(block),
                str(threads),
                str(128 * 1024 * 1024),
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.strip()
            if line:
                log(log_path, line)
        rc = proc.wait()
        if rc:
            raise RuntimeError(f"raw quant failed rc={rc} tensor={job['name']}")
    tmp.replace(dst)
    elapsed = max(time.time() - started, 0.001)
    size = dst.stat().st_size
    log(log_path, f"quant-done dst={dst} bytes={size} elapsed={elapsed:.2f}s rate={size / elapsed / 1024**2:.1f}MiB/s")
    return {"tensors": len(jobs), "bytes": size}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--threads", type=int)
    p.add_argument("--log", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    quantize(args.src, args.dst, args.block, args.threads, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import re
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


VALID_QUANTS = {"bf16", "iq1", "q4"}


def _layer_id(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _rule_matches(rule: dict, name: str) -> bool:
    if "exact" in rule and name != rule["exact"]:
        return False
    contains = rule.get("contains")
    if isinstance(contains, str) and contains not in name:
        return False
    if isinstance(contains, list) and not all(str(item) in name for item in contains):
        return False
    if "regex" in rule and not re.search(str(rule["regex"]), name):
        return False
    layer = _layer_id(name)
    if "layer_min" in rule and (layer is None or layer < int(rule["layer_min"])):
        return False
    if "layer_max" in rule and (layer is None or layer > int(rule["layer_max"])):
        return False
    return True


def load_policy(path: Path | None) -> dict | None:
    if path is None:
        return None
    policy = json.loads(path.read_text(encoding="utf-8"))
    for rule in policy.get("rules", []):
        q = rule.get("quant")
        if q not in VALID_QUANTS:
            raise ValueError(f"{path}: unsupported quant {q!r}")
    default = policy.get("default")
    if default is not None and default not in VALID_QUANTS:
        raise ValueError(f"{path}: unsupported default quant {default!r}")
    return policy


def quant_mode(name: str, shape: list[int], nparams: int, policy: dict | None = None) -> str:
    if policy:
        for rule in policy.get("rules", []):
            if _rule_matches(rule, name):
                return str(rule["quant"])
        if "default" in policy:
            return str(policy["default"])
    if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
        return "iq1"
    if len(shape) < 2 or nparams <= 4096:
        return "bf16"
    return "q4"


def quant_bytes(nparams: int, mode: str, block: int) -> int:
    if mode == "bf16":
        return nparams * 2
    full_blocks, partial = divmod(nparams, block)
    if mode == "iq1":
        return full_blocks * (2 + math.ceil(block / 8)) + (2 + math.ceil(partial / 8) if partial else 0)
    if mode == "q4":
        return full_blocks * (2 + math.ceil(block / 2)) + (2 + math.ceil(partial / 2) if partial else 0)
    raise ValueError(f"unsupported mode: {mode}")


def copy_range(src: Path, dst: Path, src_offset: int, dst_offset: int, nbytes: int) -> None:
    remaining = nbytes
    with src.open("rb") as sfp, dst.open("r+b") as dfp:
        sfp.seek(src_offset)
        dfp.seek(dst_offset)
        while remaining:
            chunk = sfp.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"{src}: short read at {src_offset}")
            dfp.write(chunk)
            remaining -= len(chunk)


def compile_raw_tool(out: Path) -> Path:
    if out.exists() and out.stat().st_mtime >= RAW_C.stat().st_mtime:
        return out
    subprocess.run(["cc", "-O3", "-std=c11", "-pthread", str(RAW_C), "-lm", "-o", str(out)], check=True)
    return out


def build_header(src: Path, block: int, policy: dict | None = None) -> tuple[dict, list[dict], int]:
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
        if name.startswith("model.visual."):
            continue
        meta = header[name]
        if meta.get("dtype") != "BF16":
            raise ValueError(f"{name}: only BF16 is supported")
        shape = [int(v) for v in meta["shape"]]
        nparams = product(shape)
        start, end = [int(v) for v in meta["data_offsets"]]
        if end - start != nparams * 2:
            raise ValueError(f"{name}: BF16 byte size mismatch")
        mode = quant_mode(name, shape, nparams, policy)
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


def quantize(src: Path, dst: Path, block: int = 256, threads: int | None = None, log_path: Path | None = None, policy: dict | None = None) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        tmp.unlink()
    raw_tool = compile_raw_tool(dst.parent / "ornith_quantize_bf16_raw")
    header, jobs, expected_bytes = build_header(src, block, policy)
    if policy:
        header["quant_policy"] = policy.get("name", "inline")
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data_start = 16 + len(encoded)
    tmp.write_bytes(b"ORNQ1\0\0\0" + struct.pack("<Q", len(encoded)) + encoded)
    with tmp.open("ab") as fp:
        fp.truncate(data_start + expected_bytes)
    started = time.time()
    threads = threads or max(min((os.cpu_count() or 4) - 2, 6), 1)
    log(log_path, f"quant-start src={src} dst={dst} tensors={len(jobs)} expected_payload={expected_bytes}")
    for job in jobs:
        out_offset = data_start + header["tensors"][job["name"]]["data_offsets"][0]
        log(log_path, f"quant-tensor name={job['name']} mode={job['mode']} params={job['nparams']} threads={threads}")
        if job["mode"] == "bf16":
            copy_range(src, tmp, job["byte_offset"], out_offset, job["nparams"] * 2)
            log(log_path, f"copy-bf16 name={job['name']} bytes={job['nparams'] * 2}")
            continue
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
    p.add_argument("--policy", type=Path, help="JSON policy with first-match tensor quant rules")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    quantize(args.src, args.dst, args.block, args.threads, args.log, load_policy(args.policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

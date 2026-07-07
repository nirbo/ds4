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
DS4_QUANTS_C = ROOT / "ornith" / "tools" / "ornith_ds4_quants.c"


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


VALID_QUANTS = {"bf16", "iq1", "q4", "q2_k"}


def _layer_id(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _layer_kind(name: str) -> str | None:
    m = re.search(r"\.layers\.\d+\.(.+)$", name)
    return m.group(1) if m else None


def retained_experts(name: str, plan: dict | None) -> list[int] | None:
    if not plan:
        return None
    layer = _layer_id(name)
    kind = _layer_kind(name)
    if layer is None or str(layer) not in plan.get("layers", {}):
        return None
    if ".experts." not in name and kind != "mlp.gate.weight":
        return None
    return [int(v) for v in plan["layers"][str(layer)]["retained"]]


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


def load_reap_plan(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


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
    if mode == "q2_k":
        if block != 256 or nparams % 256:
            raise ValueError("q2_k requires block=256 and a 256-aligned tensor")
        return (nparams // 256) * 84
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
    latest = max(RAW_C.stat().st_mtime, DS4_QUANTS_C.stat().st_mtime)
    if out.exists() and out.stat().st_mtime >= latest:
        return out
    subprocess.run(["cc", "-O3", "-std=c11", "-pthread", str(RAW_C), str(DS4_QUANTS_C), "-lm", "-o", str(out)], check=True)
    return out


def build_header(src: Path, block: int, policy: dict | None = None, reap_plan: dict | None = None) -> tuple[dict, list[dict], int]:
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
        start, end = [int(v) for v in meta["data_offsets"]]
        nparams = product(shape)
        if end - start != nparams * 2:
            raise ValueError(f"{name}: BF16 byte size mismatch")
        keep = retained_experts(name, reap_plan)
        if keep is not None:
            if len(shape) < 2:
                raise ValueError(f"{name}: cannot REAP-slice rank < 2")
            if not keep or len(set(keep)) != len(keep) or min(keep) < 0 or max(keep) >= shape[0]:
                raise ValueError(f"{name}: invalid retained expert list")
            old_shape = shape
            shape = [len(keep)] + old_shape[1:]
            nparams = product(shape)
        mode = quant_mode(name, shape, nparams, policy)
        if mode == "q2_k" and (block != 256 or len(shape) < 2 or shape[-1] % 256 or nparams % 256):
            raise ValueError(f"{name}: q2_k requires block=256 and last dim divisible by 256")
        nbytes = quant_bytes(nparams, mode, block)
        tmeta = {
            "source_dtype": "BF16",
            "quant": mode,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        if keep is not None:
            tmeta["reap_retained_experts"] = keep
        out["tensors"][name] = tmeta
        jobs.append({
            "name": name,
            "mode": mode,
            "byte_offset": data_start + start,
            "nparams": nparams,
            "retained": keep,
            "source_slice_params": product(meta["shape"][1:]) if keep is not None else 0,
        })
        offset += nbytes
    return out, jobs, offset


def stage_reap_bf16(src: Path, stage: Path, byte_offset: int, slice_params: int, keep: list[int]) -> None:
    with src.open("rb") as sfp, stage.open("wb") as dfp:
        for expert in keep:
            sfp.seek(byte_offset + expert * slice_params * 2)
            remaining = slice_params * 2
            while remaining:
                chunk = sfp.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(f"{src}: short read while staging REAP slice")
                dfp.write(chunk)
                remaining -= len(chunk)


def quantize(src: Path, dst: Path, block: int = 256, threads: int | None = None, log_path: Path | None = None, policy: dict | None = None, reap_plan: dict | None = None) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        tmp.unlink()
    raw_tool = compile_raw_tool(dst.parent / "ornith_quantize_bf16_raw")
    header, jobs, expected_bytes = build_header(src, block, policy, reap_plan)
    if policy:
        header["quant_policy"] = policy.get("name", "inline")
    if reap_plan:
        header["reap_plan"] = reap_plan.get("format", "ornith-reap-plan")
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
            if job["retained"] is None:
                copy_range(src, tmp, job["byte_offset"], out_offset, job["nparams"] * 2)
            else:
                with tmp.open("r+b") as dfp, src.open("rb") as sfp:
                    dfp.seek(out_offset)
                    for expert in job["retained"]:
                        sfp.seek(job["byte_offset"] + expert * job["source_slice_params"] * 2)
                        remaining = job["source_slice_params"] * 2
                        while remaining:
                            chunk = sfp.read(min(8 * 1024 * 1024, remaining))
                            if not chunk:
                                raise EOFError(f"{src}: short read while copying REAP bf16")
                            dfp.write(chunk)
                            remaining -= len(chunk)
            log(log_path, f"copy-bf16 name={job['name']} bytes={job['nparams'] * 2}")
            continue
        in_path = src
        in_offset = job["byte_offset"]
        stage = None
        if job["retained"] is not None:
            stage = dst.with_name(f"{dst.name}.{len(job['retained'])}.reap.bf16")
            if stage.exists():
                stage.unlink()
            stage_reap_bf16(src, stage, job["byte_offset"], job["source_slice_params"], job["retained"])
            in_path = stage
            in_offset = 0
        proc = subprocess.Popen(
            [
                str(raw_tool),
                str(in_path),
                str(tmp),
                job["mode"],
                str(in_offset),
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
        if stage is not None:
            stage.unlink()
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
    p.add_argument("--reap-plan", type=Path, help="JSON REAP plan applied before quantization")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    quantize(args.src, args.dst, args.block, args.threads, args.log, load_policy(args.policy), load_reap_plan(args.reap_plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

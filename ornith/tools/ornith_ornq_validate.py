#!/usr/bin/env python3
"""Validate experimental Ornith .ornq files."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from pathlib import Path

from ornith_safetensors_filter import read_header
from ornith_quant_formats import TYPE_LAYOUT, full_block_bytes, value as ds4_quant_value


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def bf16_to_float(raw: bytes) -> float:
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<H", raw)[0] << 16))[0]


def read_ornq(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fp:
        magic = fp.read(8)
        if magic != b"ORNQ1\0\0\0":
            raise ValueError(f"{path}: bad magic {magic!r}")
        n = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(n))
    return header, 16 + n


def check_offsets(path: Path, header: dict, data_start: int) -> list[str]:
    errors = []
    file_size = path.stat().st_size
    spans = []
    for name, meta in header["tensors"].items():
        start, end = meta["data_offsets"]
        if start < 0 or end < start:
            errors.append(f"{name}: invalid offsets {start}:{end}")
        if data_start + end > file_size:
            errors.append(f"{name}: offset exceeds file size")
        spans.append((start, end, name))
    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if start != cursor:
            errors.append(f"{name}: gap/overlap before offset {start}, expected {cursor}")
        cursor = end
    if data_start + cursor != file_size:
        errors.append(f"payload size mismatch: header={cursor} file={file_size - data_start}")
    return errors


def sample_indices(n: int, samples: int) -> list[int]:
    if n <= samples:
        return list(range(n))
    rng = random.Random(1)
    picks = {0, n // 2, n - 1}
    while len(picks) < samples:
        picks.add(rng.randrange(n))
    return sorted(picks)


def read_source_value(fp, source_data_start: int, tensor_start: int, i: int) -> float:
    fp.seek(source_data_start + tensor_start + i * 2)
    return bf16_to_float(fp.read(2))


def read_ornq_value(fp, data_start: int, meta: dict, i: int, block: int) -> float:
    start, _ = meta["data_offsets"]
    mode = meta["quant"]
    if mode == "bf16":
        fp.seek(data_start + start + i * 2)
        return bf16_to_float(fp.read(2))
    if mode in TYPE_LAYOUT:
        qk, size = TYPE_LAYOUT[mode]
        block_idx, in_block = divmod(i, qk)
        base = data_start + start + block_idx * size
        fp.seek(base)
        return ds4_quant_value(mode, fp.read(size), in_block)
    block_idx = i // block
    in_block = i % block
    base = data_start + start + block_idx * full_block_bytes(mode, block)
    fp.seek(base)
    scale = bf16_to_float(fp.read(2))
    if mode == "iq1":
        fp.seek(base + 2 + in_block // 8)
        sign = 1.0 if fp.read(1)[0] & (1 << (in_block % 8)) else -1.0
        return scale * sign
    fp.seek(base + 2 + in_block // 2)
    byte = fp.read(1)[0]
    q = (byte >> 4) if (in_block & 1) else (byte & 15)
    if q >= 8:
        q -= 16
    return scale * q


def compare_source(ornq: Path, source: Path, samples: int) -> list[dict]:
    ornq_header, ornq_data_start = read_ornq(ornq)
    source_header, source_data_start = read_header(source)
    block = int(ornq_header["block_size"])
    reports = []
    with ornq.open("rb") as qfp, source.open("rb") as sfp:
        for name, meta in ornq_header["tensors"].items():
            if name not in source_header:
                raise ValueError(f"{name}: missing from source")
            source_meta = source_header[name]
            keep = meta.get("reap_retained_experts")
            if keep is None and list(meta["shape"]) != list(source_meta["shape"]):
                raise ValueError(f"{name}: shape mismatch")
            if keep is not None and list(meta["shape"][1:]) != list(source_meta["shape"][1:]):
                raise ValueError(f"{name}: REAP slice shape mismatch")
            n = product(meta["shape"])
            slice_params = product(meta["shape"][1:]) if keep is not None else 0
            source_start = int(source_meta["data_offsets"][0])
            err2 = 0.0
            max_abs = 0.0
            count = 0
            for i in sample_indices(n, samples):
                source_i = i
                if keep is not None:
                    source_i = int(keep[i // slice_params]) * slice_params + (i % slice_params)
                a = read_source_value(sfp, source_data_start, source_start, source_i)
                b = read_ornq_value(qfp, ornq_data_start, meta, i, block)
                if not math.isfinite(b):
                    raise ValueError(f"{name}: non-finite dequant at {i}")
                e = a - b
                err2 += e * e
                max_abs = max(max_abs, abs(e))
                count += 1
            reports.append({
                "name": name,
                "quant": meta["quant"],
                "samples": count,
                "mse": err2 / count if count else 0.0,
                "max_abs": max_abs,
            })
    return reports


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ornq", required=True, type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--samples", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    header, data_start = read_ornq(args.ornq)
    errors = check_offsets(args.ornq, header, data_start)
    print(f"ornq: {args.ornq}")
    print(f"tensors: {len(header['tensors'])}")
    print(f"block size: {header['block_size']}")
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print("offsets: ok")
    if args.source:
        for report in compare_source(args.ornq, args.source, args.samples):
            print(
                f"{report['quant']} samples={report['samples']} "
                f"mse={report['mse']:.6g} max_abs={report['max_abs']:.6g} {report['name']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

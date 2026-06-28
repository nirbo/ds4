#!/usr/bin/env python3
"""Copy selected tensors from one safetensors file into another."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def is_text_tensor(name: str) -> bool:
    return name.startswith("model.language_model.") or name == "lm_head.weight"


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fp:
        raw = fp.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        header_len = struct.unpack("<Q", raw)[0]
        data = fp.read(header_len)
        if len(data) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(data.decode("utf-8")), 8 + header_len


def load_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def selected_names(header: dict, text_only: bool, allowlist: set[str] | None) -> list[str]:
    names = [name for name in header if name != "__metadata__"]
    if text_only:
        names = [name for name in names if is_text_tensor(name)]
    if allowlist is not None:
        names = [name for name in names if name in allowlist]
    return sorted(names)


def filter_safetensors(src: Path, dst: Path, text_only: bool = False, allowlist: set[str] | None = None) -> dict:
    header, data_start = read_header(src)
    names = selected_names(header, text_only, allowlist)
    out_header = {}
    chunks = []
    offset = 0
    with src.open("rb") as fp:
        for name in names:
            meta = dict(header[name])
            start, end = [int(v) for v in meta["data_offsets"]]
            fp.seek(data_start + start)
            chunk = fp.read(end - start)
            if len(chunk) != end - start:
                raise ValueError(f"{src}: truncated tensor {name}")
            meta["data_offsets"] = [offset, offset + len(chunk)]
            out_header[name] = meta
            chunks.append(chunk)
            offset += len(chunk)

    encoded = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    with dst.open("wb") as fp:
        fp.write(struct.pack("<Q", len(encoded)))
        fp.write(encoded)
        for chunk in chunks:
            fp.write(chunk)
    return {"selected": len(names), "bytes": offset}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--allowlist", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stats = filter_safetensors(args.src, args.dst, args.text_only, load_allowlist(args.allowlist))
    print(f"selected tensors: {stats['selected']}")
    print(f"copied bytes: {stats['bytes']}")
    print(f"wrote: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

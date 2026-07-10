#!/usr/bin/env python3
"""Dependency-free ModelOpt NVFP4 reference decode for Nemotron weights."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

from nemotron_metadata import MetadataError, load_json, require


E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1(nibble: int) -> float:
    require(isinstance(nibble, int) and 0 <= nibble <= 15, "E2M1 nibble is out of range")
    magnitude = E2M1_VALUES[nibble & 0x7]
    return -magnitude if nibble & 0x8 else magnitude


def decode_e4m3fn(byte: int) -> float:
    require(isinstance(byte, int) and 0 <= byte <= 255, "E4M3 byte is out of range")
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0:
        value = math.ldexp(float(mantissa), -9)
    elif exponent == 0xF and mantissa == 0x7:
        return math.nan
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return sign * value


class SafetensorsFile:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        require(len(self._map) >= 8, f"truncated safetensors file: {path}")
        self.header_size = struct.unpack_from("<Q", self._map, 0)[0]
        require(0 < self.header_size <= len(self._map) - 8, f"invalid safetensors header: {path}")
        try:
            header = json.loads(self._map[8 : 8 + self.header_size])
        except json.JSONDecodeError as exc:
            raise MetadataError(f"invalid safetensors JSON in {path}: {exc}") from exc
        require(isinstance(header, dict), f"invalid safetensors header object: {path}")
        self.tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
        self.payload_offset = 8 + self.header_size

    def close(self) -> None:
        self._map.close()
        self._file.close()

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def entry(self, name: str) -> dict[str, Any]:
        entry = self.tensors.get(name)
        require(isinstance(entry, dict), f"tensor not found in {self.path.name}: {name}")
        return entry

    def byte(self, name: str, offset: int) -> int:
        entry = self.entry(name)
        start, end = entry["data_offsets"]
        require(0 <= offset < end - start, f"tensor byte offset out of range: {name}")
        return self._map[self.payload_offset + start + offset]

    def f32_scalar(self, name: str) -> float:
        entry = self.entry(name)
        require(entry.get("dtype") == "F32" and entry.get("shape") == [], f"expected F32 scalar: {name}")
        start, end = entry["data_offsets"]
        require(end - start == 4, f"invalid F32 scalar size: {name}")
        return struct.unpack_from("<f", self._map, self.payload_offset + start)[0]


class NVFP4Weight:
    def __init__(self, shard: SafetensorsFile, prefix: str):
        self.shard = shard
        self.prefix = prefix
        self.weight_name = prefix + ".weight"
        self.scale_name = prefix + ".weight_scale"
        self.global_scale_name = prefix + ".weight_scale_2"
        weight = shard.entry(self.weight_name)
        scale = shard.entry(self.scale_name)
        global_scale = shard.entry(self.global_scale_name)

        require(weight.get("dtype") == "U8", f"expected packed U8 NVFP4 weight: {self.weight_name}")
        require(scale.get("dtype") == "F8_E4M3", f"expected E4M3 block scales: {self.scale_name}")
        require(global_scale.get("dtype") == "F32" and global_scale.get("shape") == [], f"expected global F32 scale: {self.global_scale_name}")
        weight_shape = weight.get("shape")
        scale_shape = scale.get("shape")
        require(
            isinstance(weight_shape, list)
            and len(weight_shape) == 2
            and all(isinstance(value, int) and value > 0 for value in weight_shape),
            f"invalid packed weight shape: {self.weight_name}",
        )
        require(
            isinstance(scale_shape, list)
            and len(scale_shape) == 2
            and all(isinstance(value, int) and value > 0 for value in scale_shape),
            f"invalid block scale shape: {self.scale_name}",
        )
        self.rows = weight_shape[0]
        self.columns = weight_shape[1] * 2
        self.blocks_per_row = scale_shape[1]
        require(scale_shape[0] == self.rows, f"weight/scale row mismatch: {prefix}")
        require(self.blocks_per_row * 16 == self.columns, f"weight/scale block mismatch: {prefix}")
        self.packed_columns = weight_shape[1]
        self.global_scale = shard.f32_scalar(self.global_scale_name)
        require(math.isfinite(self.global_scale) and self.global_scale > 0, f"invalid global scale: {prefix}")

    def value(self, row: int, column: int) -> float:
        require(0 <= row < self.rows, "NVFP4 row is out of range")
        require(0 <= column < self.columns, "NVFP4 column is out of range")
        packed = self.shard.byte(self.weight_name, row * self.packed_columns + column // 2)
        nibble = (packed >> 4) & 0xF if column & 1 else packed & 0xF
        scale_byte = self.shard.byte(
            self.scale_name,
            row * self.blocks_per_row + column // 16,
        )
        block_scale = decode_e4m3fn(scale_byte)
        require(math.isfinite(block_scale) and block_scale >= 0, f"invalid NVFP4 block scale in {self.prefix}")
        return decode_e2m1(nibble) * block_scale * self.global_scale

    def matvec_row(self, row: int, vector: Iterable[float]) -> float:
        values = list(vector)
        require(len(values) == self.columns, "NVFP4 matvec input width mismatch")
        return math.fsum(self.value(row, column) * values[column] for column in range(self.columns))


def locate_prefix(source_dir: Path, index: dict[str, Any], prefix: str) -> tuple[Path, list[str]]:
    names = [prefix + suffix for suffix in (".weight", ".weight_scale", ".weight_scale_2")]
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict), "index has no weight map")
    shards = {weight_map.get(name) for name in names}
    require(None not in shards, f"NVFP4 tensor triplet is incomplete: {prefix}")
    require(len(shards) == 1, f"NVFP4 tensor triplet spans shards: {prefix}")
    shard = next(iter(shards))
    return source_dir / shard, names


def parse_columns(text: str, width: int) -> list[int]:
    if text == "sample":
        candidates = [0, 1, 2, 15, 16, width // 2, width - 2, width - 1]
        return sorted(set(column for column in candidates if 0 <= column < width))
    columns = [int(value) for value in text.split(",") if value]
    require(columns and all(0 <= column < width for column in columns), "invalid column selection")
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--tensor-prefix", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--columns", default="sample")
    parser.add_argument("--matvec-ones", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index = load_json(args.source_dir / "model.safetensors.index.json")
        shard_path, _ = locate_prefix(args.source_dir, index, args.tensor_prefix)
        with SafetensorsFile(shard_path) as shard:
            weight = NVFP4Weight(shard, args.tensor_prefix)
            require(0 <= args.row < weight.rows, "row is out of range")
            columns = parse_columns(args.columns, weight.columns)
            print(
                f"nvfp4: prefix={args.tensor_prefix} shard={shard_path.name} "
                f"shape=[{weight.rows},{weight.columns}] global_scale={weight.global_scale:.9g}"
            )
            for column in columns:
                print(f"value row={args.row} column={column} value={weight.value(args.row, column):.9g}")
            if args.matvec_ones:
                print(f"matvec_ones row={args.row} value={weight.matvec_row(args.row, [1.0] * weight.columns):.9g}")
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        print(f"nemotron NVFP4 error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

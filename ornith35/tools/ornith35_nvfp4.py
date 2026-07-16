#!/usr/bin/env python3
"""Dependency-free reference decode for Ornith-35 packed ModelOpt NVFP4."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(
    "/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
)
VERIFIED_STATE_FORMAT = "ornith35-source-verified-v1"
EXPECTED_REPOSITORY = "AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4"
EXPECTED_REVISION = "85ffd2d0629ae5fa4f860dda356ec33161806c9b"
EXPECTED_WEIGHT_BYTES = 23_741_821_016
EXPECTED_WEIGHT_SHA256 = "68a4b2b8605076825302be20132cf69342b44a0385c19e6de741af5ec3114ca0"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class NVFP4Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NVFP4Error(message)


def decode_e2m1(nibble: int) -> float:
    require(isinstance(nibble, int) and 0 <= nibble <= 15, "E2M1 nibble is out of range")
    magnitude = E2M1_VALUES[nibble & 0x7]
    return -magnitude if nibble & 0x8 else magnitude


def decode_e4m3fn(byte: int) -> float:
    require(isinstance(byte, int) and 0 <= byte <= 255, "E4M3 byte is out of range")
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0:
        value = math.ldexp(float(mantissa), -9)
    elif exponent == 0xF and mantissa == 0x7:
        return math.nan
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return -value if byte & 0x80 else value


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NVFP4Error(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def require_verified_source(root: Path) -> Path:
    state = load_json(root / "source-nvfp4-state.json")
    require(state.get("format") == VERIFIED_STATE_FORMAT, "source is not verification-bound")
    require(state.get("repository") == EXPECTED_REPOSITORY, "verified repository mismatch")
    require(state.get("revision") == EXPECTED_REVISION, "verified revision mismatch")
    weight = state.get("weight")
    require(isinstance(weight, dict), "verified state has no weight")
    require(weight.get("name") == "model.safetensors", "verified weight name mismatch")
    require(weight.get("bytes") == EXPECTED_WEIGHT_BYTES, "verified weight size mismatch")
    require(weight.get("sha256") == EXPECTED_WEIGHT_SHA256, "verified weight hash mismatch")
    source = root / "source-nvfp4" / "model.safetensors"
    require(source.is_file() and source.stat().st_size == EXPECTED_WEIGHT_BYTES, "verified source is absent")
    return source


class SafetensorsFile:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            require(len(self._map) >= 8, f"truncated safetensors file: {path}")
            self.header_bytes = struct.unpack_from("<Q", self._map, 0)[0]
            require(
                0 < self.header_bytes <= len(self._map) - 8,
                f"invalid safetensors header: {path}",
            )
            try:
                header = json.loads(self._map[8 : 8 + self.header_bytes])
            except json.JSONDecodeError as exc:
                raise NVFP4Error(f"invalid safetensors JSON in {path}: {exc}") from exc
            require(isinstance(header, dict), f"invalid safetensors header object: {path}")
            self.tensors = {
                name: entry for name, entry in header.items() if name != "__metadata__"
            }
            self.payload_offset = 8 + self.header_bytes
        except Exception:
            if hasattr(self, "_map"):
                self._map.close()
            self._file.close()
            raise

    def close(self) -> None:
        self._map.close()
        self._file.close()

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def entry(self, name: str) -> dict[str, Any]:
        entry = self.tensors.get(name)
        require(isinstance(entry, dict), f"tensor not found: {name}")
        return entry

    def tensor_nbytes(self, name: str) -> int:
        entry = self.entry(name)
        offsets = entry.get("data_offsets")
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(value, int) for value in offsets),
            f"invalid tensor offsets: {name}",
        )
        start, end = offsets
        payload_bytes = len(self._map) - self.payload_offset
        require(0 <= start <= end <= payload_bytes, f"tensor range is outside payload: {name}")
        return end - start

    def byte(self, name: str, offset: int) -> int:
        entry = self.entry(name)
        self.tensor_nbytes(name)
        start, end = entry["data_offsets"]
        require(0 <= offset < end - start, f"tensor byte offset out of range: {name}")
        return self._map[self.payload_offset + start + offset]

    def f32_vector1(self, name: str) -> float:
        entry = self.entry(name)
        require(
            entry.get("dtype") == "F32" and entry.get("shape") == [1],
            f"expected F32[1]: {name}",
        )
        require(self.tensor_nbytes(name) == 4, f"invalid F32[1] size: {name}")
        start, end = entry["data_offsets"]
        return struct.unpack_from("<f", self._map, self.payload_offset + start)[0]

    def tensor_bytes(self, name: str) -> bytes:
        entry = self.entry(name)
        self.tensor_nbytes(name)
        start, end = entry["data_offsets"]
        return bytes(self._map[self.payload_offset + start : self.payload_offset + end])


class NVFP4Weight:
    def __init__(self, source: SafetensorsFile, prefix: str):
        self.source = source
        self.prefix = prefix
        self.weight_name = prefix + ".weight_packed"
        self.scale_name = prefix + ".weight_scale"
        self.global_scale_name = prefix + ".weight_global_scale"
        weight = source.entry(self.weight_name)
        scale = source.entry(self.scale_name)
        global_scale = source.entry(self.global_scale_name)

        require(weight.get("dtype") == "U8", f"expected packed U8 weight: {self.weight_name}")
        require(scale.get("dtype") == "F8_E4M3", f"expected FP8 scales: {self.scale_name}")
        require(
            global_scale.get("dtype") == "F32" and global_scale.get("shape") == [1],
            f"expected F32[1] global scale: {self.global_scale_name}",
        )
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
            f"invalid scale shape: {self.scale_name}",
        )
        self.rows = weight_shape[0]
        self.packed_columns = weight_shape[1]
        self.columns = self.packed_columns * 2
        self.blocks_per_row = scale_shape[1]
        require(
            source.tensor_nbytes(self.weight_name) == self.rows * self.packed_columns,
            f"packed weight payload mismatch: {prefix}",
        )
        require(
            source.tensor_nbytes(self.scale_name) == scale_shape[0] * scale_shape[1],
            f"scale payload mismatch: {prefix}",
        )
        require(scale_shape[0] == self.rows, f"weight/scale row mismatch: {prefix}")
        require(self.columns % 16 == 0, f"NVFP4 input width is not block aligned: {prefix}")
        require(self.blocks_per_row * 16 == self.columns, f"weight/scale block mismatch: {prefix}")
        self.global_scale = source.f32_vector1(self.global_scale_name)
        require(
            math.isfinite(self.global_scale) and self.global_scale > 0,
            f"invalid global scale: {prefix}",
        )

    def value(self, row: int, column: int) -> float:
        require(0 <= row < self.rows, "NVFP4 row is out of range")
        require(0 <= column < self.columns, "NVFP4 column is out of range")
        packed = self.source.byte(
            self.weight_name,
            row * self.packed_columns + column // 2,
        )
        nibble = packed >> 4 if column & 1 else packed & 0xF
        scale_byte = self.source.byte(
            self.scale_name,
            row * self.blocks_per_row + column // 16,
        )
        block_scale = decode_e4m3fn(scale_byte)
        require(
            math.isfinite(block_scale) and block_scale >= 0,
            f"invalid block scale in {self.prefix}",
        )
        return decode_e2m1(nibble) * block_scale * self.global_scale

    def matvec_row(self, row: int, vector: Iterable[float]) -> float:
        values = list(vector)
        require(len(values) == self.columns, "NVFP4 matvec input width mismatch")
        return math.fsum(
            self.value(row, column) * values[column]
            for column in range(self.columns)
        )


def parse_columns(text: str, width: int) -> list[int]:
    if text == "sample":
        candidates = (0, 1, 2, 15, 16, width // 2, width - 2, width - 1)
        return sorted({column for column in candidates if 0 <= column < width})
    try:
        columns = [int(value) for value in text.split(",") if value]
    except ValueError as exc:
        raise NVFP4Error(f"invalid column selection: {text}") from exc
    require(columns and all(0 <= column < width for column in columns), "invalid column selection")
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tensor-prefix", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--columns", default="sample")
    parser.add_argument("--matvec-ones", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_path = require_verified_source(args.root)
        with SafetensorsFile(source_path) as source:
            weight = NVFP4Weight(source, args.tensor_prefix)
            require(0 <= args.row < weight.rows, "row is out of range")
            print(
                f"nvfp4 prefix={args.tensor_prefix} "
                f"shape=[{weight.rows},{weight.columns}] "
                f"global_scale={weight.global_scale:.9g}"
            )
            for column in parse_columns(args.columns, weight.columns):
                print(
                    f"value row={args.row} column={column} "
                    f"value={weight.value(args.row, column):.9g}"
                )
            if args.matvec_ones:
                value = weight.matvec_row(args.row, [1.0] * weight.columns)
                print(f"matvec_ones row={args.row} value={value:.9g}")
        return 0
    except (NVFP4Error, OSError, ValueError) as exc:
        print(f"ornith35 NVFP4 error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

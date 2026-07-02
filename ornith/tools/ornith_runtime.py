#!/usr/bin/env python3
"""Reference loader and CPU kernels for experimental Ornith .ornq shards."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"ORNQ1\0\0\0"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def bf16_to_float(raw: bytes | int) -> float:
    value = raw if isinstance(raw, int) else struct.unpack("<H", raw)[0]
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def full_block_bytes(mode: str, block: int) -> int:
    if mode == "iq1":
        return 2 + math.ceil(block / 8)
    if mode == "q4":
        return 2 + math.ceil(block / 2)
    if mode == "bf16":
        return 2
    raise ValueError(f"unknown quant mode: {mode}")


@dataclass(frozen=True)
class TensorRole:
    name: str
    layer: int | None
    group: str
    kind: str


@dataclass(frozen=True)
class ORNQTensor:
    shard: "ORNQShard"
    name: str
    quant: str
    shape: list[int]
    data_offsets: tuple[int, int]

    @property
    def nparams(self) -> int:
        return product(self.shape)

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def payload_offset(self) -> int:
        return self.shard.data_start + self.data_offsets[0]

    @property
    def role(self) -> TensorRole:
        return classify_tensor(self.name)

    def value(self, i: int) -> float:
        if i < 0 or i >= self.nparams:
            raise IndexError(i)
        block = self.shard.block_size
        base = self.payload_offset
        if self.quant == "bf16":
            return bf16_to_float(self.shard.mm[base + i * 2:base + i * 2 + 2])
        block_idx = i // block
        in_block = i % block
        bbase = base + block_idx * full_block_bytes(self.quant, block)
        scale = bf16_to_float(self.shard.mm[bbase:bbase + 2])
        if self.quant == "iq1":
            sign_byte = self.shard.mm[bbase + 2 + in_block // 8]
            return scale if sign_byte & (1 << (in_block % 8)) else -scale
        packed = self.shard.mm[bbase + 2 + in_block // 2]
        q = (packed >> 4) if (in_block & 1) else (packed & 15)
        if q >= 8:
            q -= 16
        return scale * q

    def dequantize(self, limit: int | None = None) -> list[float]:
        n = self.nparams if limit is None else min(limit, self.nparams)
        return [self.value(i) for i in range(n)]

    def matvec(self, x: list[float]) -> list[float]:
        if len(self.shape) != 2:
            raise ValueError(f"{self.name}: matvec requires a 2D tensor")
        rows, cols = self.shape
        if len(x) != cols:
            raise ValueError(f"{self.name}: input length {len(x)} != {cols}")
        out = []
        for r in range(rows):
            acc = 0.0
            row = r * cols
            for c, xv in enumerate(x):
                acc += self.value(row + c) * xv
            out.append(acc)
        return out


class ORNQShard:
    def __init__(self, path: Path):
        self.path = path
        self.fp = path.open("rb")
        self.mm = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mm[:8] != MAGIC:
            self.close()
            raise ValueError(f"{path}: bad magic")
        header_len = struct.unpack("<Q", self.mm[8:16])[0]
        self.header = json.loads(self.mm[16:16 + header_len])
        self.data_start = 16 + header_len
        self.block_size = int(self.header["block_size"])
        self.tensors = {
            name: ORNQTensor(
                self,
                name,
                meta["quant"],
                [int(v) for v in meta["shape"]],
                (int(meta["data_offsets"][0]), int(meta["data_offsets"][1])),
            )
            for name, meta in self.header["tensors"].items()
        }
        self.validate()

    def close(self) -> None:
        self.mm.close()
        self.fp.close()

    def __enter__(self) -> "ORNQShard":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def validate(self) -> None:
        spans = sorted((t.data_offsets[0], t.data_offsets[1], name) for name, t in self.tensors.items())
        cursor = 0
        for start, end, name in spans:
            if start != cursor or end < start:
                raise ValueError(f"{self.path}: invalid span for {name}: {start}:{end}, expected {cursor}")
            cursor = end
        if self.data_start + cursor != self.path.stat().st_size:
            raise ValueError(f"{self.path}: payload size mismatch")
        for name, tensor in self.tensors.items():
            expected = quant_bytes(tensor.nparams, tensor.quant, self.block_size)
            if expected != tensor.nbytes:
                raise ValueError(f"{name}: {tensor.quant} bytes {tensor.nbytes} != expected {expected}")
            if name.startswith("model.visual."):
                raise ValueError(f"{name}: vision tensor present in text runtime shard")


def quant_bytes(nparams: int, mode: str, block: int) -> int:
    if mode == "bf16":
        return nparams * 2
    full, partial = divmod(nparams, block)
    each = full_block_bytes(mode, block)
    tail = full_block_bytes(mode, partial) if partial else 0
    return full * each + tail


def classify_tensor(name: str) -> TensorRole:
    match = LAYER_RE.match(name)
    if not match:
        return TensorRole(name, None, "global", name)
    layer = int(match.group(1))
    rest = match.group(2)
    if ".experts." in rest:
        group = "routed_expert"
    elif rest.startswith("mlp.shared_expert"):
        group = "shared_expert"
    elif rest.startswith("mlp."):
        group = "router"
    elif "attn" in rest:
        group = "attention"
    elif rest.endswith("layernorm.weight") or ".norm." in rest:
        group = "norm"
    else:
        group = "layer"
    return TensorRole(name, layer, group, rest)


def load_dir(root: Path) -> list[ORNQShard]:
    return [ORNQShard(path) for path in sorted(root.glob("*.ornq"))]


def memory_report(shards: list[ORNQShard]) -> dict:
    quant_bytes_by_mode: Counter[str] = Counter()
    params_by_mode: Counter[str] = Counter()
    tensors_by_group: Counter[str] = Counter()
    layers = set()
    for shard in shards:
        for tensor in shard.tensors.values():
            quant_bytes_by_mode[tensor.quant] += tensor.nbytes
            params_by_mode[tensor.quant] += tensor.nparams
            role = tensor.role
            tensors_by_group[role.group] += 1
            if role.layer is not None:
                layers.add(role.layer)
    return {
        "shards": len(shards),
        "tensors": sum(len(shard.tensors) for shard in shards),
        "layers_seen": sorted(layers),
        "bytes_by_quant": dict(sorted(quant_bytes_by_mode.items())),
        "params_by_quant": dict(sorted(params_by_mode.items())),
        "tensors_by_group": dict(sorted(tensors_by_group.items())),
    }


def layer_catalog(shards: list[ORNQShard]) -> dict[int, dict[str, list[str]]]:
    layers: dict[int, dict[str, list[str]]] = {}
    for shard in shards:
        for tensor in shard.tensors.values():
            role = tensor.role
            if role.layer is None:
                continue
            groups = layers.setdefault(role.layer, {})
            groups.setdefault(role.group, []).append(tensor.name)
    for groups in layers.values():
        for names in groups.values():
            names.sort()
    return dict(sorted(layers.items()))


def print_report(report: dict) -> None:
    print(f"shards: {report['shards']}")
    print(f"tensors: {report['tensors']}")
    print(f"layers_seen: {len(report['layers_seen'])}")
    print("bytes_by_quant:")
    for name, value in report["bytes_by_quant"].items():
        print(f"  {name}: {value}")
    print("tensors_by_group:")
    for name, value in report["tensors_by_group"].items():
        print(f"  {name}: {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.path.glob("*.ornq")) if args.path.is_dir() else [args.path]
    shards = [ORNQShard(path) for path in paths]
    try:
        print_report(memory_report(shards))
    finally:
        for shard in shards:
            shard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

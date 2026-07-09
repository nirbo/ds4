#!/usr/bin/env python3
"""Validate Ornith per-expert activation-imatrix manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from ornith_quantize_safetensors import load_imatrix_manifest


def validate(path: Path, require_complete: bool = False) -> dict:
    manifest = load_imatrix_manifest(path)
    assert manifest is not None
    errors = []
    expected = set()
    if require_complete:
        for layer in range(60):
            prefix = f"model.language_model.layers.{layer}.mlp.experts"
            expected.add(f"{prefix}.gate_up_proj")
            expected.add(f"{prefix}.down_proj")
    missing = sorted(expected - set(manifest["tensors"]))
    if missing:
        errors.append(f"missing {len(missing)} routed tensor imatrices")
    total_values = 0
    digest = hashlib.sha256()
    for name, entry in sorted(manifest["tensors"].items()):
        shape = [int(v) for v in entry["shape"]]
        file = Path(manifest["_base_dir"]) / entry["file"]
        total_values += math.prod(shape)
        with file.open("rb") as fp:
            while raw := fp.read(1024 * 1024):
                digest.update(raw)
                if len(raw) % 4:
                    errors.append(f"{name}: partial float32 payload")
                    break
                values = struct.unpack(f"<{len(raw) // 4}f", raw)
                if any(not math.isfinite(v) or v < 0 for v in values):
                    errors.append(f"{name}: invalid importance value")
                    break
    if errors:
        raise ValueError("; ".join(errors))
    payload_sha256 = digest.hexdigest()
    declared = manifest.get("payload_sha256")
    if declared and declared != payload_sha256:
        raise ValueError(f"payload sha256 mismatch: {payload_sha256} != {declared}")
    return {
        "tensors": len(manifest["tensors"]),
        "values": total_values,
        "bytes": total_values * 4,
        "payload_sha256": payload_sha256,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    report = validate(args.manifest, args.require_complete)
    print(f"imatrix tensors={report['tensors']} values={report['values']} bytes={report['bytes']} sha256={report['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

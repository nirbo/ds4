#!/usr/bin/env python3

import math
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_C = ROOT / "ornith" / "tools" / "ornith_ds4_candidate_error.c"
QUANTS_C = ROOT / "ornith" / "tools" / "ornith_ds4_quants.c"


def bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", bits >> 16)


def parse_stats(line: str) -> dict[str, float]:
    out = {}
    for item in line.strip().split():
        key, value = item.split("=", 1)
        out[key] = float(value)
    return out


def test_ds4_candidate_error_tool_runs_all_types(tmp_path: Path) -> None:
    raw = tmp_path / "tiny.bf16"
    rows, cols = 3, 256
    raw.write_bytes(b"".join(bf16(math.sin(i * 0.07) * 0.25 + math.cos(i * 0.013) * 0.05) for i in range(rows * cols)))
    tool = tmp_path / "ornith_ds4_candidate_error"
    subprocess.run(
        ["cc", "-O2", "-std=c11", "-pthread", str(RAW_C), str(QUANTS_C), "-lm", "-o", str(tool)],
        check=True,
    )
    for qtype in ("q2_k", "q4_k", "iq2_xxs"):
        proc = subprocess.run(
            [str(tool), str(raw), qtype, "0", str(rows), str(cols), "2", "0"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stats = parse_stats(proc.stdout)
        assert int(stats["count"]) == rows * cols
        assert stats["sum_sq_src"] > 0.0
        assert stats["sum_sq_err"] >= 0.0
        assert stats["row_size"] > 0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_ds4_candidate_error_tool_runs_all_types(Path(td))

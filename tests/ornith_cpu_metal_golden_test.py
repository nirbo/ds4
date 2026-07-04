#!/usr/bin/env python3
import subprocess
import sys


def parse(raw: str) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    scores: list[float] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            ids.append(int(parts[1]))
            scores.append(float(parts[2]))
    return ids, scores


def run(bin_path: str, catalog: str, shards: str, backend: str | None) -> tuple[list[int], list[float]]:
    cmd = [bin_path, catalog, shards, "0,1", "1", "4", "1", "32"]
    if backend:
        cmd.append(backend)
    return parse(subprocess.check_output(cmd, text=True))


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: ornith_cpu_metal_golden_test.py CPU_BIN METAL_BIN CATALOG SHARDS")
    cpu_ids, cpu_scores = run(sys.argv[1], sys.argv[3], sys.argv[4], None)
    metal_ids, metal_scores = run(sys.argv[2], sys.argv[3], sys.argv[4], "metal")
    assert cpu_ids == metal_ids, (cpu_ids, metal_ids)
    assert len(cpu_scores) == len(metal_scores)
    for a, b in zip(cpu_scores, metal_scores):
        assert abs(a - b) <= 1e-3, (a, b)
    print("ornith_cpu_metal_golden_test: ok")


if __name__ == "__main__":
    main()

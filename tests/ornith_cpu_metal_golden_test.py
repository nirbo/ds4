#!/usr/bin/env python3
import os
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


def run(bin_path: str, catalog: str, shards: str, case: tuple[str, int, int, int, int], backend: str | None) -> tuple[list[int], list[float]]:
    prompt, max_new, layers, top_k, vocab_limit = case
    cmd = [bin_path, catalog, shards, prompt, str(max_new), str(layers), str(top_k), str(vocab_limit)]
    if backend:
        cmd.append(backend)
    return parse(subprocess.check_output(cmd, text=True))


def check_case(name: str, case: tuple[str, int, int, int, int], cpu_bin: str, metal_bin: str, catalog: str, shards: str, tol: float) -> None:
    cpu_ids, cpu_scores = run(cpu_bin, catalog, shards, case, None)
    metal_ids, metal_scores = run(metal_bin, catalog, shards, case, "metal")
    assert cpu_ids == metal_ids, (name, cpu_ids, metal_ids)
    assert len(cpu_scores) == len(metal_scores), name
    for a, b in zip(cpu_scores, metal_scores):
        assert abs(a - b) <= tol, (name, a, b)


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: ornith_cpu_metal_golden_test.py CPU_BIN METAL_BIN CATALOG SHARDS")
    check_case("quick", ("0,1", 1, 4, 1, 32), sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 1e-3)
    if os.getenv("ORNITH_OPERATING_GOLDEN"):
        fizzbuzz = "248045,846,198,7734,83979,83152,303,351,13,248046,198,248045,74455,198,248068,271,248069,271"
        check_case("fizzbuzz-first-token", (fizzbuzz, 1, 60, 10, 0), sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 1e-2)
    print("ornith_cpu_metal_golden_test: ok")


if __name__ == "__main__":
    main()

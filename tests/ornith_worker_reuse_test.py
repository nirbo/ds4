#!/usr/bin/env python3

import subprocess
import sys


def read_response(proc: subprocess.Popen[str]) -> list[str]:
    lines: list[str] = []
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line == "":
            raise AssertionError("worker exited")
        if line == "\n":
            return lines
        lines.append(line.rstrip("\n"))


def main() -> None:
    if len(sys.argv) < 6:
        raise SystemExit("usage: ornith_worker_reuse_test.py BIN CATALOG SHARDS LAYERS TOP_K VOCAB_LIMIT [metal]")
    cmd = [sys.argv[1], "--worker", *sys.argv[2:]]
    proc = subprocess.Popen(cmd, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        prompt = "0,1"
        proc.stdin.write(f"1\t{prompt}\n")
        proc.stdin.flush()
        first = read_response(proc)
        assert first and "session=reset" in first[0], first
        token = first[1].split("\t")[1]
        proc.stdin.write(f"1\t{prompt},{token}\n")
        proc.stdin.flush()
        second = read_response(proc)
        assert second and "session=reuse" in second[0], second
    finally:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()

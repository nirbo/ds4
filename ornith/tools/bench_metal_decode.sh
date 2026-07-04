#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CATALOG="${CATALOG:-/Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv}"
SHARDS="${SHARDS:-/Users/nir/dev/models/Ornith-1.0-397B/quant-full/out}"
BIN="${BIN:-/tmp/ornith_generate_metal}"
PROMPT="${PROMPT:-0,1}"
LAYERS="${LAYERS:-60}"
TOP_K="${TOP_K:-10}"
VOCAB_LIMIT="${VOCAB_LIMIT:-0}"
MAX_NEW_LIST="${MAX_NEW_LIST:-16 64}"
CACHE_MB_LIST="${CACHE_MB_LIST:-0 512 1024 2048}"
OUT="${OUT:-$ROOT/ornith-metal-bench.log}"
LEDGER="${LEDGER:-$ROOT/ornith-perf-ledger.jsonl}"

build_bin() {
  clang -DORNITH_WITH_METAL -O3 -std=c11 -I"$ROOT/ornith" \
    "$ROOT/ornith/ornith.c" "$ROOT/ornith/ornith_generate.c" "$ROOT/ornith/ornith_metal.m" \
    -framework Foundation -framework Metal -lm -o "$BIN"
}

run_case() {
  local name="$1"
  local max_new="$2"
  shift 2
  local started
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    printf '\n[%s] %s max_new=%s prompt=%s layers=%s top_k=%s vocab=%s\n' "$started" "$name" "$max_new" "$PROMPT" "$LAYERS" "$TOP_K" "$VOCAB_LIMIT"
    printf 'env'
    for kv in "$@"; do printf ' %s' "$kv"; done
    printf '\n'
    printf 'cmd %s %s %s %s %s %s %s %s metal\n' "$BIN" "$CATALOG" "$SHARDS" "$PROMPT" "$max_new" "$LAYERS" "$TOP_K" "$VOCAB_LIMIT"
  } | tee -a "$OUT"
  local result
  result="$(env "$@" "$BIN" "$CATALOG" "$SHARDS" "$PROMPT" "$max_new" "$LAYERS" "$TOP_K" "$VOCAB_LIMIT" metal)"
  printf '%s\n' "$result" | tee -a "$OUT"
  python3 - "$LEDGER" "$name" "$started" "$max_new" "$PROMPT" "$LAYERS" "$TOP_K" "$VOCAB_LIMIT" "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || printf unknown)" "$(hostname -s 2>/dev/null || hostname)" "$*" "$result" <<'PY'
import json
import re
import sys

ledger, name, started, max_new, prompt, layers, top_k, vocab_limit, git_sha, host, env_s, output = sys.argv[1:]
header = re.search(r"backend=(\S+) generated=(\d+) layers=(\d+) expert_top_k=(\d+) vocab_limit=(\d+) seconds=([0-9.]+)", output)
if not header:
    raise SystemExit("bench output missing backend header")
seconds = float(header.group(6))
tokens = []
scores = []
for line in output.splitlines():
    parts = line.split()
    if len(parts) == 3 and parts[0].isdigit():
        tokens.append(int(parts[1]))
        scores.append(float(parts[2]))
record = {
    "ts": started,
    "git": git_sha,
    "host": host,
    "case": name,
    "prompt": prompt,
    "max_new": int(max_new),
    "layers": int(layers),
    "top_k": int(top_k),
    "vocab_limit": int(vocab_limit),
    "env": env_s.split(),
    "backend": header.group(1),
    "generated": int(header.group(2)),
    "seconds": seconds,
    "tok_s": (int(header.group(2)) / seconds) if seconds else None,
    "tokens": tokens,
    "scores": scores,
}
with open(ledger, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
print(f"ledger {ledger} tok_s={record['tok_s']:.6f}")
PY
}

if [[ ! -x "$BIN" || "$ROOT/ornith/ornith_metal.m" -nt "$BIN" || "$ROOT/ornith/ornith.c" -nt "$BIN" || "$ROOT/ornith/ornith_generate.c" -nt "$BIN" ]]; then
  build_bin
fi

printf 'ornith metal decode bench\n' | tee -a "$OUT"
printf 'catalog=%s\nshards=%s\nbin=%s\nledger=%s\n' "$CATALOG" "$SHARDS" "$BIN" "$LEDGER" | tee -a "$OUT"

for n in $MAX_NEW_LIST; do
  for mb in $CACHE_MB_LIST; do
    run_case "cache_${mb}" "$n" "ORNITH_METAL_SELECTED_EXPERT_CACHE_MB=$mb"
  done
done

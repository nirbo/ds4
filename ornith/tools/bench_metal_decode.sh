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
OUT="${OUT:-$ROOT/ornith-metal-bench.log}"

build_bin() {
  clang -DORNITH_WITH_METAL -O3 -std=c11 -I"$ROOT/ornith" \
    "$ROOT/ornith/ornith.c" "$ROOT/ornith/ornith_generate.c" "$ROOT/ornith/ornith_metal.m" \
    -framework Foundation -framework Metal -lm -o "$BIN"
}

run_case() {
  local name="$1"
  local max_new="$2"
  shift 2
  {
    printf '\n[%s] %s max_new=%s prompt=%s layers=%s top_k=%s vocab=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$max_new" "$PROMPT" "$LAYERS" "$TOP_K" "$VOCAB_LIMIT"
    printf 'env'
    for kv in "$@"; do printf ' %s' "$kv"; done
    printf '\n'
    printf 'cmd %s %s %s %s %s %s %s %s metal\n' "$BIN" "$CATALOG" "$SHARDS" "$PROMPT" "$max_new" "$LAYERS" "$TOP_K" "$VOCAB_LIMIT"
  } | tee -a "$OUT"
  env "$@" "$BIN" "$CATALOG" "$SHARDS" "$PROMPT" "$max_new" "$LAYERS" "$TOP_K" "$VOCAB_LIMIT" metal | tee -a "$OUT"
}

if [[ ! -x "$BIN" || "$ROOT/ornith/ornith_metal.m" -nt "$BIN" || "$ROOT/ornith/ornith.c" -nt "$BIN" || "$ROOT/ornith/ornith_generate.c" -nt "$BIN" ]]; then
  build_bin
fi

: > "$OUT"
printf 'ornith metal decode bench\n' | tee -a "$OUT"
printf 'catalog=%s\nshards=%s\nbin=%s\n' "$CATALOG" "$SHARDS" "$BIN" | tee -a "$OUT"

for n in $MAX_NEW_LIST; do
  run_case cache_off "$n" ORNITH_METAL_SELECTED_EXPERT_CACHE_MB=0
  run_case cache_512 "$n" ORNITH_METAL_SELECTED_EXPERT_CACHE_MB=512
  run_case cache_2048 "$n" ORNITH_METAL_SELECTED_EXPERT_CACHE_MB=2048
done

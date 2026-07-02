#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/Users/nir/dev/models/Ornith-1.0-397B}"
JOB_DIR="${JOB_DIR:-$MODEL_DIR/quant-full}"
PLAN="${PLAN:-$MODEL_DIR/ornith-text-repack-plan.json}"
MANIFEST="${MANIFEST:-$MODEL_DIR/ornith-text-storage-manifest.json}"
ALLOWLIST_DIR="${ALLOWLIST_DIR:-$MODEL_DIR}"
RAW_DIR="${RAW_DIR:-$JOB_DIR/raw}"
OUT_DIR="${OUT_DIR:-$JOB_DIR/out}"
STATE="${STATE:-$JOB_DIR/state.json}"
LOG="${LOG:-$JOB_DIR/run.log}"
TEE_LOG="${TEE_LOG:-$JOB_DIR/stdout.log}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
DOWNLOAD_METHOD="${DOWNLOAD_METHOD:-hf}"

mkdir -p "$JOB_DIR" "$RAW_DIR" "$OUT_DIR"

echo "ornith quant stream"
echo "  plan:      $PLAN"
echo "  manifest:  $MANIFEST"
echo "  state:     $STATE"
echo "  raw_dir:   $RAW_DIR"
echo "  out_dir:   $OUT_DIR"
echo "  log:       $LOG"
echo "  stdout:    $TEE_LOG"
echo

python3 -u "$ROOT/ornith/tools/ornith_stream_run.py" \
  --plan "$PLAN" \
  --manifest "$MANIFEST" \
  --state "$STATE" \
  --raw-dir "$RAW_DIR" \
  --out-dir "$OUT_DIR" \
  --allowlist-dir "$ALLOWLIST_DIR" \
  --log "$LOG" \
  --progress-interval "$PROGRESS_INTERVAL" \
  --download-method "$DOWNLOAD_METHOD" \
  --processor quantize \
  "$@" 2>&1 | tee -a "$TEE_LOG"

exit "${PIPESTATUS[0]}"

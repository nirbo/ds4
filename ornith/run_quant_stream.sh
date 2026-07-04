#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/Users/nir/dev/models/Ornith-1.0-397B}"
JOB_DIR="${JOB_DIR:-$MODEL_DIR/quant-full}"
PLAN="${PLAN:-$MODEL_DIR/ornith-text-repack-plan.json}"
MANIFEST="${MANIFEST:-$MODEL_DIR/ornith-text-storage-manifest.json}"
ALLOWLIST_DIR="${ALLOWLIST_DIR:-$MODEL_DIR}"
RAW_DIR="${RAW_DIR:-$JOB_DIR/raw}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-${OUT_DIR:-$JOB_DIR/out}}"
STATE="${STATE:-$JOB_DIR/state.json}"
LOG="${LOG:-$JOB_DIR/run.log}"
TEE_LOG="${TEE_LOG:-$JOB_DIR/stdout.log}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
DOWNLOAD_METHOD="${DOWNLOAD_METHOD:-hf}"
QUANT_POLICY="${QUANT_POLICY:-}"
REAP_PLAN="${REAP_PLAN:-}"

for arg in "$@"; do
  if [ "$arg" = "--keep-raw" ]; then
    echo "error: ornith/run_quant_stream.sh always deletes raw source shards after verified quantization" >&2
    exit 2
  fi
done

mkdir -p "$JOB_DIR" "$RAW_DIR" "$LOCAL_OUT_DIR"

echo "ornith quant stream"
echo "  plan:      $PLAN"
echo "  manifest:  $MANIFEST"
echo "  state:     $STATE"
echo "  raw_dir:   $RAW_DIR"
echo "  out_dir:   $LOCAL_OUT_DIR"
echo "  log:       $LOG"
echo "  stdout:    $TEE_LOG"
if [ -n "$REAP_PLAN" ]; then
  echo "  reap_plan: $REAP_PLAN"
fi
if [ -n "$QUANT_POLICY" ]; then
  echo "  policy:    $QUANT_POLICY"
fi
echo "  raw policy: delete after .ornq validation and state verification"
echo

POLICY_ARGS=()
if [ -n "$QUANT_POLICY" ]; then
  POLICY_ARGS=(--policy "$QUANT_POLICY")
fi
REAP_ARGS=()
if [ -n "$REAP_PLAN" ]; then
  REAP_ARGS=(--reap-plan "$REAP_PLAN")
fi

python3 -u "$ROOT/ornith/tools/ornith_stream_run.py" \
  --plan "$PLAN" \
  --manifest "$MANIFEST" \
  --state "$STATE" \
  --raw-dir "$RAW_DIR" \
  --out-dir "$LOCAL_OUT_DIR" \
  --allowlist-dir "$ALLOWLIST_DIR" \
  --log "$LOG" \
  --progress-interval "$PROGRESS_INTERVAL" \
  --download-method "$DOWNLOAD_METHOD" \
  --processor quantize \
  "${POLICY_ARGS[@]}" \
  "${REAP_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$TEE_LOG"

exit "${PIPESTATUS[0]}"

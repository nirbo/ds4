#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="deepreinforce-ai/Ornith-1.0-397B"
MODEL_DIR="${MODEL_DIR:-/Users/nir/dev/models/Ornith-1.0-397B}"
JOB_DIR="${JOB_DIR:-$MODEL_DIR/layer-calibration-bench}"
RAW_DIR="${RAW_DIR:-$JOB_DIR/raw}"
PRESERVED_DIR="${PRESERVED_DIR:-$MODEL_DIR/raw-cache}"
ENV_DIR="${CALIBRATION_ENV:-$MODEL_DIR/calibration-env}"
PYTHON="${CALIBRATION_PYTHON:-$ENV_DIR/bin/python}"
HF_REVISION="${HF_REVISION:-5e3e761811e804c295c1d3c0ce68b21da6154209}"
PROMPTS="${PROMPTS:-$JOB_DIR/calibration-prompts.jsonl}"
RESULT="${RESULT:-$JOB_DIR/result.json}"
OUTPUT_TENSOR="${OUTPUT_TENSOR:-$JOB_DIR/output-mps.pt}"
STATS_OUT="${STATS_OUT:-$JOB_DIR/stats}"
LOG="${LOG:-$JOB_DIR/run.log}"
TEE_LOG="${TEE_LOG:-$JOB_DIR/stdout.log}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
KEEP_DOWNLOADED="${KEEP_DOWNLOADED:-0}"
HF_HOME="${HF_HOME:-$JOB_DIR/hf-cache}"

mkdir -p "$JOB_DIR" "$RAW_DIR"

if [ ! -x "$PYTHON" ]; then
  echo "error: calibration Python is missing: $PYTHON" >&2
  echo "create it with:" >&2
  echo "  uv venv --python 3.12 '$ENV_DIR'" >&2
  echo "  uv pip install --python '$PYTHON' 'torch>=2.8' 'transformers>=5.8.1' safetensors psutil" >&2
  exit 2
fi

if [ ! -f "$PRESERVED_DIR/model-00002-of-00122.safetensors" ]; then
  echo "error: preserved shard 2 is missing: $PRESERVED_DIR/model-00002-of-00122.safetensors" >&2
  exit 2
fi

if [ ! -f "$PROMPTS" ]; then
  python3 "$ROOT/ornith/tools/ornith_build_calibration_dataset.py" \
    --root "$ROOT" \
    --out "$PROMPTS" \
    --limit 8
fi

echo "ornith layer calibration benchmark"
echo "  model:       $MODEL_DIR"
echo "  revision:    $HF_REVISION"
echo "  layer:       0"
echo "  raw:         $RAW_DIR"
echo "  preserved:   $PRESERVED_DIR/model-00002-of-00122.safetensors"
echo "  prompts:     $PROMPTS"
echo "  max prompts: $MAX_PROMPTS"
echo "  max length:  $MAX_LENGTH"
echo "  result:      $RESULT"
echo "  log:         $LOG"
echo

downloaded=()
for shard in model-00001-of-00122.safetensors model-00003-of-00122.safetensors; do
  dst="$RAW_DIR/$shard"
  if [ ! -f "$dst" ]; then
    HF_HOME="$HF_HOME" python3 -u "$ROOT/ornith/tools/ornith_download_shard.py" \
      --url "https://huggingface.co/$MODEL_ID/resolve/$HF_REVISION/$shard" \
      --dst "$dst" \
      --method hf \
      --repo "$MODEL_ID" \
      --filename "$shard" \
      --revision "$HF_REVISION" \
      --hf-max-workers 1 \
      --log "$LOG"
    downloaded+=("$dst")
  fi
done

set +e
PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON" -u "$ROOT/ornith/tools/ornith_layer_calibration_bench.py" \
  --model-dir "$MODEL_DIR" \
  --index "$MODEL_DIR/model.safetensors.index.json" \
  --raw-dir "$RAW_DIR" \
  --preserved-dir "$PRESERVED_DIR" \
  --prompts "$PROMPTS" \
  --out "$RESULT" \
  --dump-output "$OUTPUT_TENSOR" \
  --stats-out "$STATS_OUT" \
  --revision "$HF_REVISION" \
  --layer 0 \
  --max-prompts "$MAX_PROMPTS" \
  --max-length "$MAX_LENGTH" \
  --device mps 2>&1 | tee -a "$TEE_LOG"
rc="${PIPESTATUS[0]}"
set -e

if [ "$rc" -eq 0 ] && [ "$KEEP_DOWNLOADED" = 0 ]; then
  for path in \
    "$RAW_DIR/model-00001-of-00122.safetensors" \
    "$RAW_DIR/model-00003-of-00122.safetensors"; do
    rm -f "$path"
    printf '%s cleanup-raw file=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$path" | tee -a "$LOG"
  done
  rm -rf "$HF_HOME"
elif [ "$rc" -ne 0 ]; then
  echo "benchmark failed; downloaded shards retained for diagnosis" | tee -a "$LOG"
fi

exit "$rc"

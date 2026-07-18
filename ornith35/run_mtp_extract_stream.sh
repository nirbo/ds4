#!/bin/zsh
set -euo pipefail

repo_root=${0:A:h:h}
default_model_root=/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4
model_root=${ORNITH35_MODEL_DIR:-$default_model_root}
raw_dir=${MTP_RAW_DIR:-$model_root/source-mtp-raw}
output_dir=${MTP_OUTPUT_DIR:-$model_root/source-mtp}
state_path=${MTP_STATE:-$model_root/source-mtp-state.json}
hf_home=${MTP_HF_HOME:-$model_root/mtp-hf-cache}
if [[ -n ${HF_BIN:-} ]]; then
  hf_bin=$HF_BIN
elif [[ -x "$model_root/hf-env-1.21/bin/hf" ]]; then
  hf_bin=$model_root/hf-env-1.21/bin/hf
else
  hf_bin=hf
fi
python_bin=${PYTHON_BIN:-python3}
log_path=${MTP_LOG:-$model_root/logs/mtp-extract-stream.log}
token_path=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
revision=59d61f3ce65a6d9863b86d2e96597125219dc754
repository=Qwen/Qwen3.5-35B-A3B
minimum_free_bytes=$((8 * 1024 * 1024 * 1024))
maximum_cache_bytes=$((64 * 1024 * 1024))
shards=(
  model.safetensors-00013-of-00014.safetensors
  model.safetensors-00014-of-00014.safetensors
)

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

free_bytes() {
  local blocks
  blocks=$(df -Pk "$model_root" | awk 'NR == 2 { print $4 }')
  case "$blocks" in
    ''|*[!0-9]*)
      print -u2 "cannot determine free disk blocks for $model_root"
      return 2
      ;;
  esac
  print $((blocks * 1024))
}

allocated_bytes() {
  local target_path=$1
  local blocks
  if [[ ! -e "$target_path" ]]; then
    print 0
    return
  fi
  blocks=$(du -sk "$target_path" | awk '{ print $1 }')
  case "$blocks" in
    ''|*[!0-9]*)
      print -u2 "cannot determine allocated disk blocks for $target_path"
      return 2
      ;;
  esac
  print $((blocks * 1024))
}

state_status() {
  "$python_bin" - "$state_path" "${1:-}" <<'PY'
import json
from pathlib import Path
import sys

state = Path(sys.argv[1])
if not state.is_file():
    print("absent")
    raise SystemExit(0)
try:
    value = json.loads(state.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"invalid MTP state: {exc}", file=sys.stderr)
    raise SystemExit(2)
if not isinstance(value, dict) or value.get("status") not in ("partial", "complete"):
    print("invalid MTP state schema", file=sys.stderr)
    raise SystemExit(2)
shard = sys.argv[2]
if shard:
    completed = value.get("completed_shards")
    if not isinstance(completed, dict):
        print("invalid MTP completed-shards state", file=sys.stderr)
        raise SystemExit(2)
    print("complete" if shard in completed else "pending")
else:
    print(value["status"])
PY
}

run_extractor() {
  PYTHONPATH="$repo_root/ornith35/tools" \
    caffeinate -dimsu "$python_bin" \
      "$repo_root/ornith35/tools/ornith35_mtp_extract.py" \
      --root "$model_root" \
      --raw-dir "$raw_dir" \
      --output-dir "$output_dir" \
      --state "$state_path" \
      "$@"
}

for candidate_path in "$model_root" "$raw_dir" "$output_dir" "$hf_home"; do
  [[ -n "$candidate_path" && "$candidate_path" != "/" ]] || {
    print -u2 "unsafe MTP path: $candidate_path"
    exit 2
  }
done
[[ "$raw_dir" != "$output_dir" ]] || {
  print -u2 "MTP raw and output directories must differ"
  exit 2
}
command -v "$hf_bin" >/dev/null || {
  print -u2 "Hugging Face CLI is unavailable: $hf_bin"
  exit 2
}
command -v "$python_bin" >/dev/null || {
  print -u2 "Python is unavailable: $python_bin"
  exit 2
}

case ${1:-} in
  --self-test)
    (( $# == 1 )) || {
      print -u2 "usage: ${0:t} [--plan|--self-test]"
      exit 2
    }
    available=$(free_bytes)
    allocated=$(allocated_bytes "$model_root")
    print "mtp-stream-self-test free_bytes=$available allocated_bytes=$allocated"
    exit 0
    ;;
  --plan)
    (( $# == 1 )) || {
      print -u2 "usage: ${0:t} [--plan|--self-test]"
      exit 2
    }
    print "$(timestamp) mtp-stream-plan"
    print "  raw_dir:    $raw_dir"
    print "  output_dir: $output_dir"
    print "  state:      $state_path"
    print "  hf_home:    $hf_home"
    print "  hf_cache:   $raw_dir/.cache/huggingface (bounded and local)"
    print "  log:        $log_path"
    print "  hf_bin:     $hf_bin"
    print "  python_bin: $python_bin"
    "$python_bin" "$repo_root/ornith35/tools/ornith35_mtp_extract.py" \
      --root "$model_root" --plan
    exit 0
    ;;
  "") ;;
  *)
    print -u2 "usage: ${0:t} [--plan|--self-test]"
    exit 2
    ;;
esac

cleanup_default_hf_home=0
if [[ -z ${MTP_HF_HOME+x} ]]; then
  cleanup_default_hf_home=1
fi

mkdir -p "$raw_dir" "$output_dir" "${log_path:h}" "$hf_home"
exec > >(tee -a "$log_path") 2>&1

print "$(timestamp) mtp-stream-start"
print "  repository: $repository"
print "  revision:   $revision"
print "  raw_dir:    $raw_dir"
print "  output_dir: $output_dir"
print "  state:      $state_path"
print "  hf_home:    $hf_home"
print "  hf_cache:   $raw_dir/.cache/huggingface (bounded and local)"
print "  log:        $log_path"
print "  raw policy: delete only after full source hash, copied-range readback, and state fsync"

if [[ $(state_status) == complete ]]; then
  print "$(timestamp) mtp-state-already-complete"
  run_extractor --delete-source
  rm -rf -- "$raw_dir/.cache"
  if (( cleanup_default_hf_home )); then
    rm -rf -- "$hf_home"
  fi
  print "$(timestamp) mtp-stream-complete state=$state_path output=$output_dir/mtp.safetensors"
  exit 0
fi

for shard in "${shards[@]}"; do
  shard_state=$(state_status "$shard")
  if [[ "$shard_state" == complete ]]; then
    print "$(timestamp) mtp-shard-skip-complete shard=$shard"
    run_extractor --shard "$shard" --max-shards 1 --delete-source
    rm -rf -- "$raw_dir/.cache"
    continue
  fi
  [[ "$shard_state" == pending || "$shard_state" == absent ]] || {
    print -u2 "$(timestamp) mtp-state-error shard=$shard status=$shard_state"
    exit 2
  }

  available=$(free_bytes)
  if (( available < minimum_free_bytes )); then
    print -u2 "$(timestamp) mtp-space-error free_bytes=$available required=$minimum_free_bytes"
    exit 2
  fi
  print "$(timestamp) mtp-download-start shard=$shard free_bytes=$available"
  HF_HOME="$hf_home" \
  HF_TOKEN_PATH="$token_path" \
  HF_HUB_CACHE="$raw_dir/.cache/huggingface/hub" \
  HF_ASSETS_CACHE="$raw_dir/.cache/huggingface/assets" \
  HF_XET_CACHE="$raw_dir/.cache/huggingface/xet" \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_DOWNLOAD_TIMEOUT=120 \
    caffeinate -dimsu "$hf_bin" download \
      "$repository" "$shard" \
      --revision "$revision" \
      --local-dir "$raw_dir" \
      --max-workers 1
  cache_bytes=$(allocated_bytes "$raw_dir/.cache")
  print "$(timestamp) mtp-download-done shard=$shard cache_bytes=$cache_bytes"
  if (( cache_bytes > maximum_cache_bytes )); then
    print -u2 "$(timestamp) mtp-cache-error bytes=$cache_bytes limit=$maximum_cache_bytes"
    rm -rf -- "$raw_dir/.cache"
    print -u2 "$(timestamp) mtp-cache-cleaned retry_is_safe=true"
    exit 2
  fi

  run_extractor --shard "$shard" --max-shards 1 --delete-source
  rm -rf -- "$raw_dir/.cache"
  print "$(timestamp) mtp-shard-done shard=$shard free_bytes=$(free_bytes)"
done

if [[ $(state_status) != complete ]]; then
  print "$(timestamp) mtp-finalize-resume state=$state_path"
  run_extractor --delete-source
fi
if [[ $(state_status) != complete ]]; then
  print -u2 "$(timestamp) mtp-incomplete-error state=$state_path"
  exit 2
fi
if (( cleanup_default_hf_home )); then
  rm -rf -- "$hf_home"
fi
print "$(timestamp) mtp-stream-complete state=$state_path output=$output_dir/mtp.safetensors"

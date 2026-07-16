#!/bin/bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
model_root=${NEMOTRON_MODEL_DIR:-/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}
work_dir=${WORK_DIR:-$model_root/backbone-lowbit-work/layer1-bf16-pilot-v1}
raw_dir=${RAW_DIR:-$work_dir/raw-bf16}
fit_dir=${FIT_DIR:-$work_dir/fit-representative}
contract=${BF16_CONTRACT:-$model_root/metadata-bf16/layer-001-contract.json}
contexts=${CONTEXT_DIR:-$model_root/backbone-lowbit-work/context-layer1-balanced200-v1}
proxy_source=${PROXY_SOURCE_DIR:-$model_root/source-nvfp4}
proxy_state=${PROXY_SOURCE_STATE:-$model_root/source-nvfp4-state.json}
mlx_python=${MLX_PYTHON:-$model_root/mlx-env/bin/python}
experts=${EXPERTS:-69,83,100,120,234,258,447,506}
fit_strategy=${FIT_STRATEGY:-bf16-endpoint}
refine_steps=${REFINE_STEPS:-24}
refine_batch_size=${REFINE_BATCH_SIZE:-16}
refine_warmup_steps=${REFINE_CODE_WARMUP_STEPS:-12}
refine_endpoint_lr=${REFINE_ENDPOINT_LEARNING_RATE:-0.001}
refine_code_lr=${REFINE_CODE_LEARNING_RATE:-0.002}
margin_gib=${DISK_MARGIN_GIB:-5.0}
xet_fixed_download_concurrency=${XET_FIXED_DOWNLOAD_CONCURRENCY:-4}
xet_min_fetch_mib=${XET_MIN_FETCH_MIB:-64}
xet_max_fetch_mib=${XET_MAX_FETCH_MIB:-256}
xet_range_mib=${XET_RANGE_MIB:-128}
xet_stall_timeout_seconds=${XET_STALL_TIMEOUT_SECONDS:-300}
stdout_log=$work_dir/stdout.log

mkdir -p "$work_dir"
exec > >(tee -a "$stdout_log") 2>&1

printf '%s\n' "nemotron backbone low-bit BF16 pilot"
printf '  contract:     %s\n' "$contract"
printf '  contexts:     %s\n' "$contexts"
printf '  proxy source: %s\n' "$proxy_source"
printf '  raw dir:      %s\n' "$raw_dir"
printf '  fit dir:      %s\n' "$fit_dir"
printf '  experts:      %s\n' "$experts"
printf '  fit strategy: %s\n' "$fit_strategy"
printf '  refinement:   steps=%s batch=%s warmup=%s endpoint_lr=%s code_lr=%s\n' \
    "$refine_steps" "$refine_batch_size" "$refine_warmup_steps" \
    "$refine_endpoint_lr" "$refine_code_lr"
printf '  Xet profile:  fixed_concurrency=%s fetch=%s..%sMiB range=%sMiB stall=%ss adaptive=false HP=false\n' \
    "$xet_fixed_download_concurrency" "$xet_min_fetch_mib" "$xet_max_fetch_mib" \
    "$xet_range_mib" "$xet_stall_timeout_seconds"
printf '  stdout log:   %s\n' "$stdout_log"
printf '  raw policy:   retained until explicit deletion approval\n'
df -h "$model_root"

download_args=(
    "$mlx_python" "$repo_root/nemotron/tools/nemotron_bf16_download.py"
    --contract "$contract"
    --job-dir "$work_dir"
    --raw-dir "$raw_dir"
    --margin-gib "$margin_gib"
    --xet-fixed-concurrency "$xet_fixed_download_concurrency"
    --xet-min-fetch-mib "$xet_min_fetch_mib"
    --xet-max-fetch-mib "$xet_max_fetch_mib"
    --xet-range-mib "$xet_range_mib"
    --xet-stall-timeout-seconds "$xet_stall_timeout_seconds"
)
if [ -n "${DOWNLOAD_MAX_SHARDS:-}" ]; then
    download_args+=(--max-shards "$DOWNLOAD_MAX_SHARDS")
fi
"${download_args[@]}"

if [ "${DOWNLOAD_ONLY:-0}" = "1" ] || [ -n "${DOWNLOAD_MAX_SHARDS:-}" ]; then
    printf '%s\n' "bounded download stop requested; rerun without DOWNLOAD_MAX_SHARDS to fit"
    exit 0
fi

"$mlx_python" "$repo_root/nemotron/tools/nemotron_bf16_download.py" \
    --contract "$contract" \
    --job-dir "$work_dir" \
    --raw-dir "$raw_dir" \
    --margin-gib "$margin_gib" \
    --xet-fixed-concurrency "$xet_fixed_download_concurrency" \
    --xet-min-fetch-mib "$xet_min_fetch_mib" \
    --xet-max-fetch-mib "$xet_max_fetch_mib" \
    --xet-range-mib "$xet_range_mib" \
    --xet-stall-timeout-seconds "$xet_stall_timeout_seconds" \
    --validate-only

"$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_backbone_fit.py" \
    --contract "$contract" \
    --raw-dir "$raw_dir" \
    --proxy-source-dir "$proxy_source" \
    --proxy-source-state "$proxy_state" \
    --contexts "$contexts" \
    --output-dir "$fit_dir" \
    --experts "$experts" \
    --fit-strategy "$fit_strategy" \
    --refine-steps "$refine_steps" \
    --refine-batch-size "$refine_batch_size" \
    --refine-code-warmup-steps "$refine_warmup_steps" \
    --refine-endpoint-learning-rate "$refine_endpoint_lr" \
    --refine-code-learning-rate "$refine_code_lr"

"$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_backbone_fit.py" \
    --contract "$contract" \
    --raw-dir "$raw_dir" \
    --proxy-source-dir "$proxy_source" \
    --proxy-source-state "$proxy_state" \
    --contexts "$contexts" \
    --output-dir "$fit_dir" \
    --experts "$experts" \
    --fit-strategy "$fit_strategy" \
    --refine-steps "$refine_steps" \
    --refine-batch-size "$refine_batch_size" \
    --refine-code-warmup-steps "$refine_warmup_steps" \
    --refine-endpoint-learning-rate "$refine_endpoint_lr" \
    --refine-code-learning-rate "$refine_code_lr" \
    --validate-only

"$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_backbone_pilot.py" \
    --fit-dir "$fit_dir" \
    --output "$work_dir/pilot-report.json"

df -h "$model_root"
printf 'pilot complete; raw BF16 shards remain at %s\n' "$raw_dir"

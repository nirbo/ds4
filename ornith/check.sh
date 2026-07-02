#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

python3 tests/ornith_memory_plan_test.py
python3 tests/ornith_prompt_test.py
python3 tests/ornith_layout_check_test.py
python3 tests/ornith_storage_manifest_test.py
python3 tests/ornith_shard_scope_report_test.py
python3 tests/ornith_layer_catalog_test.py
python3 tests/ornith_iq1_test.py
python3 tests/ornith_safetensors_filter_test.py
python3 tests/ornith_text_repack_plan_test.py
python3 tests/ornith_text_tensor_allowlist_test.py
python3 tests/ornith_stream_state_test.py
python3 tests/ornith_stream_wrappers_test.py
python3 tests/ornith_stream_run_test.py
python3 tests/ornith_quantize_safetensors_test.py
python3 tests/ornith_ornq_validate_test.py
python3 tests/ornith_runtime_test.py
python3 -m py_compile \
  ornith/tools/fetch_ornith_metadata.py \
  ornith/tools/ornith_memory_plan.py \
  ornith/tools/ornith_prompt.py \
  ornith/tools/ornith_layout_check.py \
  ornith/tools/ornith_storage_manifest.py \
  ornith/tools/ornith_shard_scope_report.py \
  ornith/tools/ornith_layer_catalog.py \
  ornith/tools/ornith_iq1.py \
  ornith/tools/ornith_safetensors_filter.py \
  ornith/tools/ornith_text_repack_plan.py \
  ornith/tools/ornith_text_tensor_allowlist.py \
  ornith/tools/ornith_stream_state.py \
  ornith/tools/ornith_download_shard.py \
  ornith/tools/ornith_process_shard.py \
  ornith/tools/ornith_stream_run.py \
  ornith/tools/ornith_quantize_safetensors.py \
  ornith/tools/ornith_ornq_validate.py \
  ornith/tools/ornith_runtime.py \
  tests/ornith_memory_plan_test.py \
  tests/ornith_prompt_test.py \
  tests/ornith_layout_check_test.py \
  tests/ornith_storage_manifest_test.py \
  tests/ornith_shard_scope_report_test.py \
  tests/ornith_layer_catalog_test.py \
  tests/ornith_iq1_test.py \
  tests/ornith_safetensors_filter_test.py \
  tests/ornith_text_repack_plan_test.py \
  tests/ornith_text_tensor_allowlist_test.py \
  tests/ornith_stream_state_test.py \
  tests/ornith_stream_wrappers_test.py \
  tests/ornith_stream_run_test.py \
  tests/ornith_quantize_safetensors_test.py \
  tests/ornith_ornq_validate_test.py \
  tests/ornith_runtime_test.py
bash -n ornith/run_quant_stream.sh
cc -O3 -std=c11 -pthread ornith/tools/ornith_quantize_bf16_raw.c -lm -o /tmp/ornith_quantize_bf16_raw_check

if [ -f /Users/nir/dev/models/Ornith-1.0-397B/config.json ] &&
   [ -f /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json ]; then
  python3 ornith/tools/ornith_layout_check.py \
    --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
    --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json >/dev/null
  python3 ornith/tools/ornith_layer_catalog.py \
    --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
    --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json >/dev/null
fi

echo "ornith checks: ok"

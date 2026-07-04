#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

python3 tests/ornith_memory_plan_test.py
python3 tests/ornith_prompt_test.py
python3 tests/ornith_decode_tokens_test.py
python3 tests/ornith_chat_test.py
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
python3 tests/ornith_quant_policy_report_test.py
python3 tests/ornith_reap_plan_test.py
python3 tests/ornith_ornq_validate_test.py
python3 tests/ornith_quant_error_test.py
python3 tests/ornith_ds4_quant_candidate_error_test.py
python3 tests/ornith_runtime_test.py
python3 tests/ornith_runtime_catalog_test.py
python3 tests/ornith_bench_metal_decode_test.py
python3 -m py_compile \
  ornith/tools/fetch_ornith_metadata.py \
  ornith/tools/ornith_decode_tokens.py \
  ornith/tools/ornith_memory_plan.py \
  ornith/tools/ornith_prompt.py \
  ornith/tools/ornith_chat.py \
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
  ornith/tools/ornith_quant_policy_report.py \
  ornith/tools/ornith_reap_plan.py \
  ornith/tools/ornith_ornq_validate.py \
  ornith/tools/ornith_quant_error.py \
  ornith/tools/ornith_ds4_quant_candidate_error.py \
  ornith/tools/ornith_runtime.py \
  ornith/tools/ornith_runtime_catalog.py \
  tests/ornith_memory_plan_test.py \
  tests/ornith_prompt_test.py \
  tests/ornith_decode_tokens_test.py \
  tests/ornith_chat_test.py \
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
  tests/ornith_quant_policy_report_test.py \
  tests/ornith_reap_plan_test.py \
  tests/ornith_ornq_validate_test.py \
  tests/ornith_quant_error_test.py \
  tests/ornith_ds4_quant_candidate_error_test.py \
  tests/ornith_runtime_test.py \
  tests/ornith_runtime_catalog_test.py \
  tests/ornith_bench_metal_decode_test.py \
  tests/ornith_cpu_metal_golden_test.py \
  tests/ornith_worker_reuse_test.py
bash -n ornith/run_quant_stream.sh
cc -O3 -std=c11 -pthread ornith/tools/ornith_quantize_bf16_raw.c -lm -o /tmp/ornith_quantize_bf16_raw_check
cc -DORNITH_TESTING -O2 -std=c11 -I. ornith/ornith.c tests/ornith_native_catalog_loader_test.c -lm -o /tmp/ornith_native_catalog_loader_test
/tmp/ornith_native_catalog_loader_test
cc -O2 -std=c11 -Iornith ornith/ornith.c ornith/ornith_step_smoke.c -lm -o /tmp/ornith_step_smoke
cc -O3 -std=c11 -Iornith ornith/ornith.c ornith/ornith_generate.c -lm -o /tmp/ornith_generate
if [ "$(uname -s)" = "Darwin" ]; then
  clang -DORNITH_TESTING -O2 -std=c11 -I. -Iornith \
    ornith/ornith.c ornith/ornith_metal.m tests/ornith_metal_matvec_test.m \
    -framework Foundation -framework Metal -lm -o /tmp/ornith_metal_matvec_test
  /tmp/ornith_metal_matvec_test
  clang -O2 -std=c11 -I. -Iornith \
    ornith/ornith.c ornith/ornith_metal.m tests/ornith_metal_gdn_test.m \
    -framework Foundation -framework Metal -lm -o /tmp/ornith_metal_gdn_test
  /tmp/ornith_metal_gdn_test
  clang -O2 -std=c11 -Iornith \
    ornith/ornith.c ornith/ornith_metal.m ornith/ornith_metal_step_smoke.m \
    -framework Foundation -framework Metal -lm -o /tmp/ornith_metal_step_smoke
  clang -DORNITH_WITH_METAL -O3 -std=c11 -Iornith \
    ornith/ornith.c ornith/ornith_metal.m ornith/ornith_generate.c \
    -framework Foundation -framework Metal -lm -o /tmp/ornith_generate_metal
fi

if [ -f /Users/nir/dev/models/Ornith-1.0-397B/config.json ] &&
   [ -f /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json ]; then
  python3 ornith/tools/ornith_layout_check.py \
    --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
    --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json >/dev/null
  python3 ornith/tools/ornith_layer_catalog.py \
    --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
    --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json >/dev/null
fi

if [ -d /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out ] &&
   [ -f /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json ]; then
  python3 ornith/tools/ornith_runtime_catalog.py \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
    --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json >/dev/null
fi

if [ -d /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out ] &&
   [ -f /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv ]; then
  /tmp/ornith_native_catalog_loader_test \
    /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out >/dev/null
  /tmp/ornith_step_smoke \
    /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
    0 1 1 5 32 >/dev/null
  /tmp/ornith_step_smoke \
    /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
    0 4 1 5 32 1 decode >/dev/null
  /tmp/ornith_step_smoke \
    /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
    0,1 4 1 5 32 1 decode >/dev/null
  /tmp/ornith_generate \
    /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
    0,1 1 4 1 32 >/dev/null
  printf '1\t0,1\nquit\n' | /tmp/ornith_generate --worker \
    /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
    4 1 32 >/dev/null
  python3 tests/ornith_worker_reuse_test.py \
    /tmp/ornith_generate \
    /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
    /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
    4 1 32
  if [ "$(uname -s)" = "Darwin" ]; then
    /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      0,1 1 4 1 32 metal >/dev/null
    ORNITH_METAL_SHARED_RESIDENT_MB=64 /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      0,1 1 4 1 32 metal >/dev/null
    ORNITH_METAL_BUFFER_MOE=1 /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      0,1 1 4 1 32 metal >/dev/null
    ORNITH_METAL_ROUTER_TOPK=1 /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      0,1 1 4 1 32 metal >/dev/null
    ORNITH_METAL_RESIDENT_LAYER=1 /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      0,1 1 4 1 32 metal >/dev/null
    ORNITH_METAL_TOKEN_LOOP=1 /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      0,1 1 4 1 32 metal >/dev/null
    python3 tests/ornith_cpu_metal_golden_test.py \
      /tmp/ornith_generate \
      /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out
    printf '1\t0,1\nquit\n' | /tmp/ornith_generate_metal --worker \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      4 1 32 metal >/dev/null
    python3 tests/ornith_worker_reuse_test.py \
      /tmp/ornith_generate_metal \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      4 1 32 metal
    /tmp/ornith_metal_step_smoke \
      /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
      /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
      0 1 1 5 32 >/dev/null
  fi
fi

echo "ornith checks: ok"

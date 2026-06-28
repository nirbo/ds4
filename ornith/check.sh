#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

python3 tests/ornith_memory_plan_test.py
python3 tests/ornith_prompt_test.py
python3 tests/ornith_layout_check_test.py
python3 tests/ornith_storage_manifest_test.py
python3 -m py_compile \
  ornith/tools/fetch_ornith_metadata.py \
  ornith/tools/ornith_memory_plan.py \
  ornith/tools/ornith_prompt.py \
  ornith/tools/ornith_layout_check.py \
  ornith/tools/ornith_storage_manifest.py \
  tests/ornith_memory_plan_test.py \
  tests/ornith_prompt_test.py \
  tests/ornith_layout_check_test.py \
  tests/ornith_storage_manifest_test.py

if [ -f /Users/nir/dev/models/Ornith-1.0-397B/config.json ] &&
   [ -f /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json ]; then
  python3 ornith/tools/ornith_layout_check.py \
    --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
    --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json >/dev/null
fi

echo "ornith checks: ok"

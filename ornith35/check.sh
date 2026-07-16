#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
python3 "$repo_root/tests/ornith35_fetch_metadata_test.py"
python3 "$repo_root/tests/ornith35_metadata_test.py"
python3 "$repo_root/tests/ornith35_nvfp4_test.py"
python3 "$repo_root/tests/ornith35_source_verify_test.py"

model_dir=${ORNITH35_MODEL_DIR:-/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4}
metadata_dir="$model_dir/metadata"
config="$metadata_dir/config.json"
header="$metadata_dir/model.safetensors.header.json"
state="$metadata_dir/source-state.json"

if [ -f "$config" ] && [ -f "$header" ] && [ -f "$state" ]; then
    python3 "$repo_root/ornith35/tools/ornith35_metadata.py" \
        --config "$config" \
        --header "$header" \
        --source-state "$state" \
        --out "$metadata_dir/catalog.json"
else
    printf '%s\n' "ornith35 metadata smoke skipped: run ornith35_fetch_metadata.py target"
fi

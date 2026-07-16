#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
model_dir=${ORNITH35_MODEL_DIR:-/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4}
python3 "$repo_root/tests/ornith35_fetch_metadata_test.py"
python3 "$repo_root/tests/ornith35_metadata_test.py"
python3 "$repo_root/tests/ornith35_nvfp4_test.py"
python3 "$repo_root/tests/ornith35_source_verify_test.py"

mlx_python="$model_dir/mlx-env/bin/python"
if [ -x "$mlx_python" ]; then
    mlx_versions=$($mlx_python -c 'import importlib.metadata as m; print(m.version("mlx"), m.version("mlx-metal"))')
    if [ "$mlx_versions" != "0.32.0 0.32.0" ]; then
        printf '%s\n' "ornith35 MLX version mismatch: expected 0.32.0 0.32.0, found $mlx_versions" >&2
        exit 1
    fi
    "$mlx_python" "$repo_root/tests/ornith35_mlx_nvfp4_test.py"
else
    printf '%s\n' "ornith35 MLX smoke skipped: expected $mlx_python"
fi

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

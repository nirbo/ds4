#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
python3 "$repo_root/tests/nemotron_metadata_test.py"
python3 "$repo_root/tests/nemotron_safetensors_inventory_test.py"
python3 "$repo_root/tests/nemotron_prune_materialize_test.py"
python3 "$repo_root/tests/nemotron_nvfp4_test.py"

model_dir=${NEMOTRON_MODEL_DIR:-/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}
metadata_dir="$model_dir/metadata"
config="$metadata_dir/config.json"
index="$metadata_dir/model.safetensors.index.json"
manifest="$metadata_dir/source-manifest.json"

if [ -f "$config" ] && [ -f "$index" ] && [ -f "$manifest" ]; then
    python3 "$repo_root/nemotron/tools/nemotron_metadata.py" \
        --config "$config" \
        --index "$index" \
        --source-manifest "$manifest" \
        --out "$metadata_dir/nemotron-metadata-catalog.json"
else
    printf '%s\n' "nemotron metadata smoke skipped: set NEMOTRON_MODEL_DIR to a pinned metadata directory"
fi

source_dir="$model_dir/source-nvfp4"
source_state="$model_dir/source-nvfp4-state.json"
if [ -d "$source_dir" ] && [ -f "$source_state" ]; then
    python3 "$repo_root/nemotron/tools/nemotron_safetensors_inventory.py" \
        --source-dir "$source_dir" \
        --source-state "$source_state" \
        --out "$metadata_dir/nemotron-safetensors-inventory.json"
else
    printf '%s\n' "nemotron safetensors inventory skipped: verified source snapshot is unavailable"
fi

if [ "$(uname -s)" = "Darwin" ]; then
    metal_test="${TMPDIR:-/tmp}/nemotron_metal_nvfp4_test"
    cc -O3 -fobjc-arc -Wall -Wextra \
        -I"$repo_root/nemotron" \
        "$repo_root/nemotron/nemotron_metal.m" \
        "$repo_root/tests/nemotron_metal_nvfp4_test.m" \
        -framework Foundation -framework Metal -lm -o "$metal_test"
    if [ -d "$source_dir" ]; then
        "$metal_test" \
            "$source_dir/model-00001-of-00017.safetensors" \
            backbone.layers.1.mixer.experts.0.up_proj
    else
        "$metal_test"
    fi
    rm -f "$metal_test"
fi

mlx_python="$model_dir/mlx-env/bin/python"
if [ -x "$mlx_python" ]; then
    "$mlx_python" "$repo_root/tests/nemotron_mlx_nvfp4_test.py"
    if [ -d "$source_dir" ]; then
        "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_nvfp4.py" \
            --source-dir "$source_dir" \
            --tensor-prefix backbone.layers.1.mixer.experts.0.up_proj \
            --repeats 20
    fi
else
    printf '%s\n' "nemotron MLX NVFP4 smoke skipped: isolated MLX environment is unavailable"
fi

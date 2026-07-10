#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
python3 "$repo_root/tests/nemotron_metadata_test.py"
python3 "$repo_root/tests/nemotron_safetensors_inventory_test.py"
python3 "$repo_root/tests/nemotron_prune_materialize_test.py"
python3 "$repo_root/tests/nemotron_nvfp4_test.py"
python3 "$repo_root/tests/nemotron_mlx_pack_test.py"
python3 "$repo_root/tests/nemotron_mlx_prune_plan_test.py"
python3 "$repo_root/tests/nemotron_ngram_lookup_test.py"

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
    "$mlx_python" - <<'PY'
from importlib.metadata import version

required = "0.32.0"
actual = version("mlx")
if actual != required:
    raise SystemExit(f"validated Nemotron runtime requires mlx=={required}; found {actual}")
print(f"nemotron MLX runtime: version={actual}")
PY
    "$mlx_python" "$repo_root/tests/nemotron_mlx_nvfp4_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_moe_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_linear_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_calibrate_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_compare_logits_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_resident_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_verify_bench_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_mtp_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_speculative_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_paged_embeddings_test.py"
    "$mlx_python" "$repo_root/tests/nemotron_mlx_head_certificate_test.py"
    if [ -d "$source_dir" ]; then
        "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_nvfp4.py" \
            --source-dir "$source_dir" \
            --tensor-prefix backbone.layers.1.mixer.experts.0.up_proj \
            --repeats 20
        "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_linear.py" \
            --source-dir "$source_dir" \
            --tensor-prefix backbone.layers.0.mixer.in_proj \
            --repeats 20
        "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_mamba.py" \
            --source-dir "$source_dir" \
            --layer 0 \
            --repeats 10
        "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_attention.py" \
            --source-dir "$source_dir" \
            --layer 7 \
            --context 8 \
            --repeats 10
        "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_stream_forward.py" \
            --source-dir "$source_dir" \
            --token-ids 0 \
            --max-layers 2
        if [ "${NEMOTRON_MLX_REAL_MOE:-0}" = "1" ]; then
            "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_moe.py" \
                --source-dir "$source_dir" \
                --layer 1 \
                --repeats 20
            "$mlx_python" "$repo_root/nemotron/tools/nemotron_mlx_moe_layer.py" \
                --source-dir "$source_dir" \
                --layer 1 \
                --repeats 20
        else
            printf '%s\n' "nemotron real MLX MoE benchmark skipped; set NEMOTRON_MLX_REAL_MOE=1 to run"
        fi
    fi
else
    printf '%s\n' "nemotron MLX NVFP4 smoke skipped: isolated MLX environment is unavailable"
fi

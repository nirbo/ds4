#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
model_dir=${ORNITH35_MODEL_DIR:-/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4}
python3 "$repo_root/tests/ornith35_fetch_metadata_test.py"
python3 "$repo_root/tests/ornith35_metadata_test.py"
python3 "$repo_root/tests/ornith35_context_test.py"
python3 "$repo_root/tests/ornith35_turboquant_reference_test.py"
python3 "$repo_root/tests/ornith35_dspark_test.py"
python3 "$repo_root/tests/ornith35_nvfp4_test.py"
python3 "$repo_root/tests/ornith35_source_verify_test.py"
python3 "$repo_root/tests/ornith35_mtp_extract_test.py"

launcher_tmp=$(mktemp -d)
python_path=$(command -v python3)
trap 'rmdir "$launcher_tmp"' 0 1 2 3 15
ORNITH35_MODEL_DIR="$launcher_tmp" \
  HF_BIN=/usr/bin/true \
  PYTHON_BIN="$python_path" \
  "$repo_root/ornith35/run_mtp_extract_stream.sh" --self-test
rmdir "$launcher_tmp"
trap - 0 1 2 3 15

mlx_python="$model_dir/mlx-env/bin/python"
if [ -x "$mlx_python" ]; then
    runtime_versions=$($mlx_python -c 'import importlib.metadata as m; print(m.version("mlx"), m.version("mlx-metal"), m.version("tokenizers"))')
    if [ "$runtime_versions" != "0.32.0 0.32.0 0.23.1" ]; then
        printf '%s\n' "ornith35 runtime version mismatch: expected 0.32.0 0.32.0 0.23.1, found $runtime_versions" >&2
        exit 1
    fi
    "$repo_root/ornith35/build_extensions.sh"
    "$mlx_python" "$repo_root/tests/ornith35_tokenizer_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_generate_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_generate_mtp_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_sampling_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_nvfp4_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_dense_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_gdn_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_attention_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_turboquant_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_turboquant_characterize_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_turboquant_cache_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_linear_cache_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_moe_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_layer_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_model_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_speculative_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_mtp_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_mtp_runtime_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_mtp_teacher_capture_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_mtp_distill_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_draft_regime_gate_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_dspark_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_dspark_runtime_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_cache_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_vocab_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_profile_test.py"
    "$mlx_python" "$repo_root/tests/ornith35_mlx_prefill_profile_test.py"
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

dspark_metadata_dir="$model_dir/metadata-dspark"
if [ -f "$dspark_metadata_dir/config.json" ] \
    && [ -f "$dspark_metadata_dir/model.safetensors.header.json" ] \
    && [ -f "$dspark_metadata_dir/source-state.json" ]; then
    python3 "$repo_root/ornith35/tools/ornith35_dspark.py" \
        --metadata-dir "$dspark_metadata_dir" \
        --out "$dspark_metadata_dir/catalog.json"
else
    printf '%s\n' "ornith35 DSpark metadata smoke skipped: run ornith35_fetch_metadata.py dspark"
fi

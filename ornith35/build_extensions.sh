#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
model_dir=${ORNITH35_MODEL_DIR:-/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4}
mlx_python="$model_dir/mlx-env/bin/python"
jobs=${ORNITH35_BUILD_JOBS:-6}

if [ ! -x "$mlx_python" ]; then
    printf '%s\n' "missing Ornith-35 MLX Python: $mlx_python" >&2
    exit 1
fi

cd "$repo_root/ornith35/extensions/kv_cache"
"$mlx_python" setup.py build_ext -j"$jobs" --inplace

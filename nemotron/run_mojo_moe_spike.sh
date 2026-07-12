#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
model_dir=${NEMOTRON_MODEL_DIR:-/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}
mojo_env=${NEMOTRON_MOJO_ENV:-$model_dir/mojo-env-26.4}
mojo=$mojo_env/bin/mojo
log=${NEMOTRON_MOJO_LOG:-$model_dir/metadata/mojo-moe-spike.log}

if [ ! -x "$mojo" ]; then
    printf '%s\n' "nemotron Mojo error: missing toolchain at $mojo" >&2
    exit 1
fi

version=$($mojo --version 2>&1)
case "$version" in
    *"Mojo 1.0.0b2"*) ;;
    *)
        printf '%s\n' "nemotron Mojo error: expected Mojo 1.0.0b2; found $version" >&2
        exit 1
        ;;
esac

mkdir -p "$(dirname "$log")"
{
    printf '%s\n' "nemotron Mojo selected-expert spike"
    printf '  toolchain: %s\n' "$version"
    printf '  source:    %s\n' "$repo_root/nemotron/mojo/nemotron_mojo_moe.mojo"
    printf '  log:       %s\n' "$log"
    "$mojo" run "$repo_root/nemotron/mojo/nemotron_mojo_device.mojo"
    "$mojo" run "$repo_root/nemotron/mojo/nemotron_mojo_moe.mojo"
} 2>&1 | tee "$log"

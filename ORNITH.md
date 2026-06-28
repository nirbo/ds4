# Ornith Runtime Notes

This branch is the integration branch for the Ornith local-runtime work.
Feature work should branch from `ornith-main`, then merge back here after it is
tested for correctness, numerical integrity, and performance.

DS4 remains an upstream-syncable reference. Do not make Ornith depend on DS4
source files for model-specific runtime logic. If DS4 code is useful, copy the
needed piece into `ornith_*` files and adapt it there.

Performance changes are welcome only with checks that show output integrity and
numerical accuracy are preserved.

## Current Tools

Run all current Ornith metadata checks:

```sh
./ornith/check.sh
```

`ornith/tools/ornith_memory_plan.py` estimates compression targets from local
metadata:

```sh
python3 ornith/tools/ornith_memory_plan.py \
  --config /path/to/config.json \
  --index /path/to/model.safetensors.index.json
```

If the safetensors shards are available locally, add `--safetensors-dir DIR`.
The tool reads only safetensors headers to bucket tensor bytes; it does not load
full model tensors into memory.

`ornith/tools/ornith_prompt.py` is the executable reference for text-only chat
rendering and tokenizer special-token inspection:

```sh
python3 ornith/tools/ornith_prompt.py \
  --tokenizer /Users/nir/dev/models/Ornith-1.0-397B/tokenizer.json \
  --print-specials
```

`ornith/tools/ornith_layout_check.py` validates the text tensor names expected
by the Ornith runtime against a local safetensors index:

```sh
python3 ornith/tools/ornith_layout_check.py \
  --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

`ornith/tools/ornith_storage_manifest.py` creates a small shard manifest for
future network-storage downloads without downloading any weights:

```sh
python3 ornith/tools/ornith_storage_manifest.py \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

Do not download Hugging Face files on this machine without explicit approval.

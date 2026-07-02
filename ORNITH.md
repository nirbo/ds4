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

Add `--text-only` to exclude vision tensors from the manifest.
Add `--safetensors-dir DIR` to read local shard headers and report selected
bytes without loading tensor data.

`ornith/tools/ornith_shard_scope_report.py` classifies shards as language,
visual, other, or mixed from the local index:

```sh
python3 ornith/tools/ornith_shard_scope_report.py \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

`ornith/tools/ornith_layer_catalog.py` builds a checked per-layer map of text
tensors to shards:

```sh
python3 ornith/tools/ornith_layer_catalog.py \
  --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

`ornith/tools/ornith_iq1.py` is a synthetic reference harness for 1-bit and
ternary expert quantization, weighted block scaling, and packed dot-product
checks. It does not read model weights.

```sh
python3 ornith/tools/ornith_iq1.py
```

`ornith/tools/ornith_safetensors_filter.py` copies selected tensors from a
safetensors shard into a new shard. Use `--text-only` to exclude vision tensors.

`ornith/tools/ornith_text_repack_plan.py` reports which shards can be copied,
filtered, or skipped for a text-only repack. It reads only the local index.
Add `--dry-run --src-dir SRC --dst-dir DST --allowlist-dir DIR` to print the
future copy/filter actions without touching weights.
Add `--execute` to run those actions against already-local shard files.

`ornith/tools/ornith_text_tensor_allowlist.py` writes the text tensor names for
one shard, suitable for `ornith_safetensors_filter.py --allowlist`.

`ornith/tools/ornith_stream_state.py` tracks resumable shard streaming:
pending, downloading, downloaded, processing, done, failed, output size, raw
deletion, and sha256 verification. It supports one standby downloaded shard
while another shard processes.

`ornith/tools/ornith_download_shard.py` downloads one shard to `*.part` with
human-readable byte progress logs. `ornith/tools/ornith_process_shard.py`
copies or filters one local shard with progress logs. Add `--benchmark-only`
to process once, report throughput, and delete the benchmark output.

`ornith/tools/ornith_stream_run.py` runs the one-shard processing loop with one
background prefetch shard, state updates, output verification, and raw deletion.
Use `--max-shards N` for bounded smoke tests. The current processor filters or
copies safetensors shards; final quantized output is not implemented yet.
Use `--download-method hf` for Hugging Face CLI/Xet downloads instead of the
stdlib fallback downloader.

`ornith/tools/ornith_quantize_safetensors.py` writes the experimental `.ornq`
smoke quantization format. Routed expert tensors use IQ1 blocks; other BF16
tensors use symmetric Q4 blocks. The C helper uses pthread workers and fixed
output offsets.

Do not download Hugging Face files on this machine without explicit approval.

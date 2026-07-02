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
copies, filters, or quantizes one local shard with progress logs. Add
`--benchmark-only` to process once, report throughput, and delete the benchmark
output.

`ornith/tools/ornith_stream_run.py` runs the one-shard processing loop with one
background prefetch shard, state updates, output verification, and raw deletion.
Use `--max-shards N` for bounded smoke tests. Add `--processor quantize` to
write compact `.ornq` outputs, validate them against the still-local raw shard,
then delete raw after state verification unless `--keep-raw` is set.
Use `--download-method hf` for Hugging Face CLI/Xet downloads instead of the
stdlib fallback downloader.

`ornith/run_quant_stream.sh` starts the standard quantized streaming job and
tees live stdout to `quant-full/stdout.log` while `ornith_stream_run.py` keeps
writing its structured log to `quant-full/run.log`:

```sh
ornith/run_quant_stream.sh --max-shards 2
```

Omit `--max-shards` only after approving the full weight-download run.
Quantized outputs go to local disk at `quant-full/out` by default, or
`LOCAL_OUT_DIR=/local/path` if overridden. Downloaded source shards are
temporary and the launcher rejects `--keep-raw`; raw files are deleted after
`.ornq` validation and state verification.

Resume behavior is state-file driven. Re-run the same command after an
interrupted job:

```sh
ornith/run_quant_stream.sh
```

Completed shards stay `done` and are not repeated. A caught download/process
failure is marked `failed` and retried. A hard stop during `processing` retries
from the already-downloaded raw shard if it still exists. A hard stop during
`downloading`, or processing without a raw shard, retries the download for that
shard. Raw shards are deleted only after `.ornq` validation and state
verification. If interruption leaves a raw shard behind, rerun uses or replaces
it according to the state file.

`ornith/tools/ornith_quantize_safetensors.py` writes the experimental `.ornq`
smoke quantization format. Vision tensors are skipped. Routed expert tensors
use IQ1 blocks, small/sensitive tensors are copied as BF16, and remaining BF16
matrix tensors use symmetric Q4 blocks. The C helper uses pthread workers,
chunked I/O, and fixed output offsets.

`ornith/tools/ornith_ornq_validate.py` validates `.ornq` headers and can sample
dequantized values against a source safetensors shard.

`ornith/tools/ornith_runtime.py` is the current reference runtime foundation:
it mmap-loads `.ornq` shards, validates tensor spans/sizes, classifies tensor
roles, reports quantized memory by mode/group, and provides CPU reference
dequant/matvec helpers. It is not the final inference runtime.

`ornith/tools/ornith_runtime_catalog.py` builds the compact tensor catalog used
as the bridge from quantized shards to native runtime work:

```sh
python3 ornith/tools/ornith_runtime_catalog.py \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json \
  --out /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.json \
  --native-out /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv
```

The current generated catalog is text-only, validates exact non-vision tensor
coverage against the safetensors index, and contains 122 shards, 1038 tensors,
and 60 layers. The TSV sidecar is consumed by `ornith.c` so the native runtime
does not need a JSON parser.

`ornith.h` and `ornith.c` are the first native runtime boundary. They load the
TSV catalog, validate `.ornq` shard magic/sizes, check tensor payload ranges,
mmap shards, and provide tensor lookup, layer-aware tensor lookup,
BF16/Q4/IQ1 scalar decode, reference matvec, 3D expert-slice matvec, RMSNorm,
top-k helpers, and a narrow MoE layer smoke path. The layer smoke path performs
input RMSNorm, router matvec/top-k, routed expert gate/up/down, and shared
expert contribution for correctness composition; it is not the final optimized
token loop. The native boundary also includes reference embedding lookup and
lm-head top-k scoring. Native checks validate MoE tensor shape compatibility
across all layers when the full local catalog is present. Current real-output
probe:

```sh
cc -O2 -std=c11 -I. ornith/ornith.c tests/ornith_native_catalog_loader_test.c \
  -o /tmp/ornith_native_catalog_loader_test
/tmp/ornith_native_catalog_loader_test \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out
```

`ornith/ornith_step_smoke.c` is a bounded native token-step probe:

```sh
cc -O2 -std=c11 -Iornith ornith/ornith.c ornith/ornith_step_smoke.c \
  -lm -o /tmp/ornith_step_smoke
/tmp/ornith_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0 1 1 5 32
```

Arguments are `TOKEN_ID LAYERS EXPERT_TOP_K OUT_TOP_K VOCAB_LIMIT`. The current
real one-layer capped smoke returns top rows from the first 32 lm-head rows:

```text
0  12  2.24126315
1   9  1.6809659
2  10  1.45022964
3   5  1.41391909
4   1  1.39811707
```

Add `REPEATS` as the final argument to measure repeated smoke steps. Current
fast native kernels decode BF16/Q4/IQ1 directly from mapped payloads while
keeping scalar decode as the test reference:

```sh
/tmp/ornith_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0 1 1 5 32 20
```

Current CPU timing sample: 20 one-layer capped steps in 1.137730 seconds.

`ornith/ornith_metal.m` adds narrow Metal BF16/Q4/IQ1 matvec kernels over
mapped `.ornq` shard spans plus a Metal-backed token-step smoke CLI:

```sh
clang -O3 -std=c11 -Iornith \
  ornith/ornith.c ornith/ornith_metal.m ornith/ornith_metal_step_smoke.m \
  -framework Foundation -framework Metal -lm -o /tmp/ornith_metal_step_smoke
/tmp/ornith_metal_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0 1 1 5 32 20
```

Current side-by-side one-layer capped timing sample:

```text
CPU:   20 repeats in 1.137730 seconds
Metal: 20 repeats in 0.763141 seconds
```

The Metal and CPU top-k order matches; scores differ only by small float-order
rounding.

Current smoke artifacts live in:

```sh
/Users/nir/dev/models/Ornith-1.0-397B/quant-smoke
```

The two checked outputs are text-only:

- `model-00001-of-00122.ornq`: 16 tensors, Q4 plus BF16 passthrough, no
  `model.visual.*` tensors
- `model-00002-of-00122.ornq`: routed expert `gate_up_proj` in IQ1

The latest smoke log is `quant-text-current.log`; sampled validation logs are
`validate-00001.log` and `validate-00002.log`.

Do not download Hugging Face files on this machine without explicit approval.

# Agent Notes

This branch is for the Ornith local-runtime project.

## Goal

Build a narrow, model-specific runtime and quantization path for
`deepreinforce-ai/Ornith-1.0-397B`, aiming to run an excellent coding MoE on
consumer hardware with less memory than DS4 requires.

Initial target:

- text-only Ornith only for now
- vision tensors (`model.visual.*`) are excluded from runtime, packing, loading,
  memory targets, and tests except for metadata filters that prove they are
  skipped
- aggressive routed-expert compression, likely IQ1/IQ1.5-class before IQ2
- correctness and numerical integrity before speed
- consumer targets: 64 GB unified-memory Mac first, then 32 GB NVIDIA plus host
  memory/offload experiments if useful

## Branch Workflow

`ornith-main` is the integration branch. Create feature branches from it, test
there, then merge back after verification.

## DS4 Relationship

DS4 code is reference material only. Keep DS4 source upstream-syncable.

Do not make Ornith runtime code depend on DS4 model-specific files. If a DS4
utility or pattern is useful, copy the needed code into `ornith_*` files and
adapt it there.

Small approved Ornith metadata files live outside the repo at
`/Users/nir/dev/models/Ornith-1.0-397B`. Do not download model weights or other
large Hugging Face files without explicit user approval.

Derived text-only metadata in that directory:

- `ornith-text-storage-manifest.json`
- `ornith-text-repack-plan.json`
- `model-00001-of-00122.text.allowlist`
- `ornith-runtime-catalog.json`
- `ornith-runtime-catalog.tsv`

Quant smoke artifacts in that directory:

- `quant-smoke/model-00001-of-00122.ornq`
- `quant-smoke/model-00002-of-00122.ornq`
- `quant-smoke/quant-text-current.log`
- `quant-smoke/validate-00001.log`
- `quant-smoke/validate-00002.log`

Streaming smoke tests use `ornith/tools/ornith_stream_run.py` with
`--max-shards N`. Keep `N` small unless explicitly approved. Use
`--processor quantize` for compact `.ornq` outputs; the runner validates each
`.ornq` against the raw shard before deleting the raw file. Prefer
`--download-method hf` for real Hugging Face pulls.

Use `ornith/run_quant_stream.sh --max-shards N` to launch the standard
quantized stream with live stdout teeing. Omit `--max-shards` only after
explicit approval for the full weight-download job.
Quantized outputs go to local disk at `quant-full/out` by default, or
`LOCAL_OUT_DIR=/local/path` if overridden. The launcher rejects `--keep-raw`;
raw source shards are temporary and deleted after verified quantization.

Resumption is state-file driven. Re-running the same command skips `done`
shards, retries caught `failed` shards, recovers stale `processing` from the
raw shard when present, and retries the download for stale `downloading` or
missing-raw processing shards.

## Quality Rules

- Keep the runtime model-specific, not generic.
- Validate tensor metadata strictly before running kernels.
- Keep public APIs narrow: frontends should not know tensor internals.
- Add the smallest runnable check for non-trivial logic.
- Performance changes are welcome only when tested for correctness and numeric
  drift.

## Likely Layout

- `ornith.h`: public engine/session boundary.
- `ornith.c`: model metadata, tokenizer/prompt rendering, reference path, and
  session logic. Current implementation loads the native TSV runtime catalog,
  validates shard magic/sizes, mmaps shards, and exposes tensor lookup plus
  layer-aware tensor lookup, BF16/Q4/IQ1 scalar decode, reference matvec,
  3D expert-slice matvec, RMSNorm, top-k helpers, and a narrow MoE layer smoke
  path. It also has reference embedding lookup and lm-head top-k scoring.
  Hot matvec paths decode BF16/Q4/IQ1 directly from mapped payloads; scalar
  `ornith_tensor_value` remains the correctness reference in tests.
  `ornith/ornith_metal.m` provides narrow Metal matvecs and a Metal-backed
  token-step smoke path for macOS; CPU remains the reference path.
  `ornith/ornith_step_smoke.c` runs a bounded native token-step smoke from a
  TSV catalog, with optional vocab cap for fast real-model probes.
  Native checks validate MoE tensor shape compatibility across all layers when
  the local full quantized catalog is present.
  These execution paths are for correctness composition, not final performance.
- `ornith_quantize.c`: safetensors/GGUF conversion and quantization experiments.
- `ornith_metal.m`, `ornith_cuda.cu`: backend code if the experiment reaches
  GPU graph work.
- `ornith/tools/ornith_stream_run.py`: resumable one-download-ahead shard
  streaming smoke/run controller.
- `ornith/tools/ornith_quantize_safetensors.py`: experimental text-only
  `.ornq` smoke quantizer; vision tensors are skipped, routed experts use IQ1
  blocks, small/sensitive tensors stay BF16, and other BF16 matrix tensors use
  Q4.
- `ornith/tools/ornith_runtime.py`: reference `.ornq` loader/catalog,
  memory report, and CPU dequant/matvec helpers. It is not the final inference
  runtime.
- `ornith/tools/ornith_runtime_catalog.py`: builds the compact runtime tensor
  catalog from `.ornq` shards and validates exact text tensor coverage when the
  safetensors index is available. Use `--native-out` to write the TSV consumed
  by `ornith.c`.
- `tests/ornith_*`: focused tests and metadata/quantization checks.

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
  session logic.
- `ornith_quantize.c`: safetensors/GGUF conversion and quantization experiments.
- `ornith_metal.m`, `ornith_cuda.cu`: backend code if the experiment reaches
  GPU graph work.
- `tests/ornith_*`: focused tests and metadata/quantization checks.

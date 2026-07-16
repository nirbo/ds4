# Agent Notes

This branch is for the Ornith 1.0 35B local-runtime project.

## Goal

Build a narrow, model-specific compression and inference path for
`AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`. The first production
target is a 64 GB M4 Max, with an RTX 5090 32 GB as a second target.

Initial requirements:

- text-only runtime; vision is intentionally omitted until the text path is
  correct, stable, and fast
- preserve the checkpoint's MLP-only, weight-only NVFP4 representation
- preserve BF16 full attention, GatedDeltaNet, routers, embeddings, LM head,
  norms, and recurrent parameters
- native 262,144-token context must remain the correctness baseline
- add a separately selected YaRN factor-2 profile for 524,288 tokens
- make long coding sessions practical through exact prefix reuse, persistent
  cache checkpoints, incremental prefill, and background cache warming
- accelerate decode with exact target-verified MTP and DSpark only after the
  target path is authoritative
- correctness and numerical integrity before performance, then optimize the
  complete prefill and decode paths rather than isolated kernels

The selected checkpoint is abliterated. Treat all generated code and tool
requests as untrusted and retain the normal execution sandbox.

## Branch Workflow

`ornith35-main` is the integration branch. Create `feature/ornith35-*` branches
from it, implement and test one coherent feature there, then merge the verified
feature back into `ornith35-main`. Push both branches when the feature is
complete.

Do not merge `ornith-main`, `nemotron-main`, or this branch into the DS4 `main`
branch. Do not rewrite existing DS4 files into shared target abstractions.

`ORNITH35_EXPERIMENTS.md` is the durable work ledger. Check an experiment only
after implementation, correctness testing, quality measurement, and relevant
performance/memory measurement. Record `SUCCESS`, `PARTIAL`, or `REJECTED`
with evidence.

## Independence

DS4, Ornith 397B, Nemotron, MLX-LM, llama.cpp, DeepSpec, and other projects are
reference material only. If logic is useful, copy and adapt it into
`ornith35_*` files. The Ornith-35 implementation must not depend on `ds4_*`,
`ornith_*`, or `nemotron_*` model-specific files.

Never assume that the 397B Ornith tensor layout, quantization format, pruning
policy, or Metal kernel matches this 35B checkpoint. Validate every copied
shape and equation against the pinned metadata and checkpoint.

## Model Storage

Model metadata and future artifacts live outside the repository at:

`/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`

Pinned sources:

- target: `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4` at
  `85ffd2d0629ae5fa4f860dda356ec33161806c9b`
- DSpark bootstrap draft:
  `pablogrant/ORNITH-1.0_35B_AEON_PABLOG-OPTIMIZED_UNCENSORED_DSPARK-DRAFT_NVFP4`
  at `9383b3c33ddf982114a4f72e07c890bfd6c35df2`
- MTP bootstrap source: `Qwen/Qwen3.5-35B-A3B` at
  `59d61f3ce65a6d9863b86d2e96597125219dc754`

Small metadata belongs under `metadata/`, `metadata-dspark/`, and
`metadata-mtp-source/`. Immutable weights, converted runtime files, caches,
experiments, and logs must use separate sibling directories.

Pinned Qwen3.5-MoE architecture references live under
`source-notes/transformers-5.10.1/`. `source-state.json` binds the exact files
to Transformers tag `v5.10.1` and commit
`90c3ae54d448d4906b6167317ea5a7f5d48a232d`. They are reference code only;
copy and adapt required equations into `ornith35_*` files.

The Apple runtime environment lives at `$ORNITH35_MODEL_DIR/mlx-env` and is
pinned to MLX `0.32.0`. `ornith35/check.sh` runs Metal-backed tests when it is
present and rejects any other installed MLX version.

Do not download weights or other large files without explicit user approval.
Before an approved download, report:

- exact file and indexed payload sizes and shard counts
- current free disk space
- cache and temporary-space behavior
- expected peak disk use
- verification, cleanup, and resumption behavior

Avoid hidden duplicate Hugging Face caches. Bind every durable state file to
the repository revision and hashes of source files, policies, tools, and
outputs. Never delete the sole retained source checkpoint without explicit
approval.

## Architecture Facts

Validate these against the pinned config and tensor header:

- `Qwen3_5MoeForConditionalGeneration`
- 40 text layers: 30 GatedDeltaNet and 10 full-attention layers, repeating
  three linear-attention layers then one full-attention layer
- hidden size 2,048
- 256 routed experts per layer, top-8 plus one shared expert
- routed and shared expert intermediate size 512
- 16 query heads, 2 KV heads, head dimension 256
- GatedDeltaNet uses 16 key heads and 32 value heads at dimension 128
- native context 262,144, partial RoPE factor 0.25, theta 10,000,000
- config advertises one MTP layer, but the released Ornith target contains no
  `mtp.*` tensors; MTP must be introduced and validated as a separate sidecar

## Context And Prefill Contracts

The default context profile is native 262,144. The 524,288 profile uses static
YaRN factor 2 with `original_max_position_embeddings=262144`; changing only a
length limit is invalid. Native and YaRN caches are never interchangeable.

An exact persistent cache contains all full-attention K/V tensors, every
GatedDeltaNet recurrent and convolution state, the exact token prefix, and the
next position. Cache identity includes model and runtime hashes, tokenizer and
chat-template hashes, quantization policy, RoPE profile, cache dtype, token
prefix hash, and state schema version.

Start with BF16 K/V as the correctness reference. Quantized K/V is a separate
quality-gated experiment. Rotating windows, eviction, sparse attention,
CacheBlend-style non-prefix reuse, and prompt compression change semantics and
must never silently replace the exact path.

For coding sessions, keep stable system/tool/repository content first and
volatile diffs/conversation last. Use content-addressed prefix checkpoints,
incremental append, background prefill, and bounded LRU disk retention. A cold
524K prefill remains quadratic in the ten full-attention layers; capacity is
not evidence of acceptable cold-start latency.

## Compression And Quality

The source checkpoint is already a mixed-precision quality baseline. Do not
requantize it broadly and do not prune experts initially. Strip vision only
after proving the text-only tensor view is complete.

Keep target, MTP, and DSpark artifacts separate. Target verification remains
authoritative, so speculative paths must reproduce the target distribution or
exact greedy output under the selected sampling contract.

Use official or independently generated baseline logits and substantial coding
evaluations. Arithmetic prompts are smoke tests only. Add the smallest runnable
test for every non-trivial rule, and accept performance work only with numeric
drift, memory, and end-to-end timing evidence.

## Likely Layout

- `ORNITH35.md`: current architecture, evidence, commands, and decisions
- `ORNITH35_EXPERIMENTS.md`: forward-work ledger and measured outcomes
- `ornith35/`: target-specific launchers and runtime code
- `ornith35/tools/ornith35_fetch_metadata.py`: bounded metadata and header
  fetcher that cannot download weight payloads
- `ornith35/tools/ornith35_metadata.py`: strict config/header catalog and
  context-memory model
- `ornith35/tools/ornith35_source_verify.py`: full source size, header, payload,
  and SHA-256 acceptance with atomic revision-bound state
- `ornith35/tools/ornith35_nvfp4.py`: dependency-free packed E2M1/FP8-scale
  reference decoder and CPU numerical oracle
- `ornith35/tools/ornith35_mlx_nvfp4.py`: MLX composition boundary and custom
  Metal matvec for the exact Ornith packed NVFP4 triplet
- `ornith35/tools/ornith35_gdn_reference.py`: dependency-free scalar oracle for
  the exact one-token GatedDeltaNet recurrence
- `ornith35/tools/ornith35_mlx_gdn.py`: immutable-state MLX GatedDeltaNet
  one-token composition and strict BF16 layer loader
- `ornith35/tools/ornith35_attention_reference.py`: dependency-free scalar
  oracle for Qwen3.5 gated GQA decode and text RoPE
- `ornith35/tools/ornith35_mlx_attention.py`: immutable BF16 K/V state and
  one-token MLX full-attention composition
- `ornith35/tools/ornith35_moe_reference.py`: dependency-free scalar top-k,
  packed-NVFP4 expert, and shared-expert oracle
- `ornith35/tools/ornith35_mlx_moe.py`: GPU-owned router and selected-expert
  Metal path with no expert-ID readback to Python
- `ornith35/tools/ornith35_mlx_layer.py`: exact centered RMSNorm, residual,
  token-mixer state, and MoE composition for both decoder-layer types
- `ornith35/tools/ornith35_mlx_model.py`: strict text-only 40-layer loader,
  full-vocabulary one-token logits, and position-bound aggregate state
- `ornith35/tools/ornith35_*`: future conversion, MLX, Metal, cache, MTP,
  DSpark, and quality tools
- `tests/ornith35_*`: focused tests

These are ownership boundaries, not mandatory abstractions. Add only what a
verified feature needs.

# Agent Notes

This branch is for the Nemotron 3 Super local-runtime and compression project.

## Goal

Build a narrow, model-specific compression and inference path for
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`, aiming to preserve its coding
quality while making it practical on consumer hardware.

Initial targets:

- 64 GB unified-memory Apple Silicon, with enough headroom for runtime state
- RTX 5090 32 GB plus 96 GB host-memory offload when useful
- retain NVIDIA's native QAT NVFP4 mixed-precision representation first
- prune routed experts structurally before considering another quantization pass
- preserve surviving quantized tensors and their scales byte-for-byte
- correctness and numerical integrity before performance, then optimize the
  complete decode path rather than isolated kernels

The official NVFP4 checkpoint is approximately 74.78 GiB of indexed tensor
payload. It is already small enough for a full local download when disk headroom
permits, but model weights or other large files still require explicit user
approval before download.

## Branch Workflow

`nemotron-main` is the integration branch. Create `feature/nemotron-*` branches
from it, implement and test one coherent feature there, then merge the verified
feature back into `nemotron-main` and push both the feature and integration
branches as appropriate.

Do not merge `ornith-main` into `nemotron-main`. Do not merge target-specific
work directly into the DS4 `main` branch.

## DS4 And Ornith Relationship

DS4 and Ornith are reference material only. Keep the original DS4 source
upstream-syncable and keep the target implementations independent.

If a DS4 or Ornith utility or pattern is useful, copy only the relevant logic
into `nemotron_*` files and adapt it to NemotronH. Nemotron code must not depend
on `ds4_*` or `ornith_*` model-specific files. In particular, do not carry over
Ornith tensor layouts, Qwen3.5 attention assumptions, `.ornq` policies, or Metal
kernels without independently proving they match NemotronH.

## Model Storage

Small approved model metadata and future artifacts live outside the repository
at:

`/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`

The complete verified NVFP4 source is pinned at
`source-nvfp4/` under that directory. Its immutable revision is
`4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6`; verification details are in
`source-nvfp4-state.json`. Never modify or delete these source shards. Derived
artifacts belong in sibling directories.

Small upstream source references live under `source-notes/` in the same model
directory. `source-notes/revisions.json` pins the ModelOpt, vLLM, MLX-LM, and
oMLX commits used to establish NVFP4 decode, model, calibration, MTP, and
runtime semantics. These repositories are reference code only; copy and adapt
required model-specific logic into `nemotron_*` files.

Keep immutable upstream metadata, transient downloads, calibration output,
compressed candidates, and logs in distinct subdirectories. Record the exact
Hugging Face revision and hashes in every durable state file.

Do not download weights or other large Hugging Face files without explicit user
approval. Before an approved large download, report:

- indexed payload size and shard count
- current free disk space
- cache and temporary-space behavior
- expected peak disk use
- cleanup and resumption behavior

Avoid hidden duplicate Hugging Face caches. For streamed conversion, keep at
most one shard processing and one shard prefetched, validate output atomically,
record completion, then delete the raw shard. Never delete a sole retained raw
source unless the user explicitly approves it.

## Compression Baseline

Treat the official ModelOpt NVFP4 checkpoint as the first production baseline.
Its mixed precision is training-aware and must not be flattened into one broad
format.

The first compression candidate is prune-only:

1. Observe the unpruned official model.
2. Rank routed experts using activation-aware evidence suitable for LatentMoE.
3. Slice expert tensors, router rows, and router correction bias consistently.
4. Update expert-count metadata and remapping tables.
5. Preserve every retained NVFP4/FP8/BF16 payload and scale exactly.
6. Validate the materialized checkpoint before deleting any source data.

Do not collect final REAP observations from an already pruned, damaged, or
requantized candidate. Keep quant-only, prune-only, and combined compression
results separate so quality regressions can be localized.

Do not quantize NVFP4 or FP8 weights again merely to reduce storage. Any second
quantization pass is a separate experiment requiring full-tensor error,
activation error, logits, and downstream quality evidence.

## Quality Rules

- Keep the runtime model-specific, not generic.
- Validate tensor names, shapes, dtypes, quantization metadata, and shard sizes
  strictly before processing.
- Require exact identity for retained payloads in prune-only candidates.
- Bind resumable jobs to immutable source revision and hashes of every plan,
  policy, calibration artifact, and tool version.
- Use official or independently generated baseline logits and substantial coding
  evaluations; arithmetic prompts are smoke tests, not quality acceptance.
- Measure quant-only, prune-only, and combined candidates independently.
- Add the smallest runnable test for every non-trivial rule.
- Accept performance changes only with correctness and numerical-drift checks.
- Keep public APIs narrow: frontends must not know tensor internals.
- Keep human-readable logs for every download, transition, validation, cleanup,
  failure, and resumed operation.

## Initial Architecture Facts

The target is NemotronH, not the Qwen-derived Ornith architecture. Current
official metadata describes 88 decoder layers composed of 40 Mamba2 layers,
40 LatentMoE layers, and 8 full-attention layers. The MoE uses 512 routed
experts with 22 selected experts per token, plus latent projections and shared
expert paths. Every implementation must validate these facts against the pinned
checkpoint rather than assuming they remain unchanged.

## Likely Layout

- `NEMOTRON.md`: current design, evidence, commands, and decisions.
- `nemotron/`: model-specific C, Objective-C, CUDA, policies, and launchers.
- `nemotron/tools/nemotron_metadata.py`: immutable metadata and layout catalog.
- `nemotron/tools/nemotron_stream_run.py`: resumable bounded-download runner.
- `nemotron/tools/nemotron_nvfp4.py`: ModelOpt NVFP4 metadata and payload tools.
- `nemotron/tools/nemotron_mlx_nvfp4.py`: MLX composition boundary for the
  packed ModelOpt NVFP4 Metal kernel. The isolated environment lives at
  `$NEMOTRON_MODEL_DIR/mlx-env`; the check script skips it when unavailable.
- `nemotron/tools/nemotron_mlx_moe.py`: GPU-owned top-k LatentMoE path using
  MLX `gather_qmm` over ModelOpt NVFP4 experts. The routine check runs its
  scalar synthetic test; set `NEMOTRON_MLX_REAL_MOE=1` for the 3 GiB-peak real
  layer benchmark.
- `nemotron/tools/nemotron_mlx_pack.py`: direct, resumable prune-to-runtime
  materializer. It slices routers, renumbers retained experts, omits MTP when
  requested, and writes each layer's expert NVFP4 tensors pre-stacked without
  changing retained payload bytes. Use `--max-groups N` for bounded smokes.
- `nemotron/tools/nemotron_mlx_linear.py`: ModelOpt FP8 and BF16 MLX linear
  primitives. FP8 defaults to native MXFP8 qmm with shared unity scales and the
  checkpoint scalar folded into activations; a custom Metal decoder remains
  the independent numerical reference.
- `nemotron/tools/nemotron_mlx_mamba.py`: one-token Nemotron Mamba2 composition
  over official BF16 convolution/state tensors and exact ModelOpt FP8
  projections, with persistent MLX `ArraysCache` recurrence.
- `nemotron/tools/nemotron_safetensors_inventory.py`: exact header and size
  validation without loading tensor payloads.
- `nemotron/tools/nemotron_prune_materialize.py`: revision-bound,
  exact-preserving expert remap and resumable artifact materialization.
- `nemotron/nemotron_metal.m` and `metal/nemotron_nvfp4.metal`: independent
  Apple Metal runtime boundary and packed ModelOpt NVFP4 kernels.
- `nemotron/tools/nemotron_reap_*`: calibration, planning, reporting, and
  exact-preserving structural materialization.
- `tests/nemotron_*`: focused metadata, compression, numerical, and runtime
  checks.

These names describe ownership boundaries, not mandatory abstractions. Add only
the files needed by verified work.

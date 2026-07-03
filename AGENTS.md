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

Small upstream implementation notes live in
`/Users/nir/dev/models/Ornith-1.0-397B/source-notes`. These are source files
only, not model weights. They currently include the vLLM Qwen3.5 wrapper,
Qwen3-Next attention source, Qwen Gated DeltaNet layer, recurrent/conv helper
kernels, and gated RMSNorm reference path used to derive the Ornith attention
equations, plus the upstream Hugging Face Qwen3.5 MoE model source used to
verify raw checkpoint layouts. Important verified layouts:

- `linear_attn.in_proj_qkv` is raw HF contiguous `[query, key, value]`.
- full-attention `q_proj` is per-head `[query, gate]` and must be unpacked
  per head before q-norm/RoPE and output gating.

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
  token-step smoke path for macOS; CPU remains the reference path. The Metal
  routed path batches selected IQ1 expert slices for real `top_k=10` probes,
  and Metal now scores capped/full lm-head rows before CPU top-k selection.
  For very large row counts such as the 248k-row lm-head, Metal uses one
  thread per row instead of one threadgroup per row to avoid dispatch-grid
  overhead. Current Metal performance work includes block-256-specialized Q4
  and routed IQ1 kernels, fused selected-expert gate/up + SiLU + down + mix for
  Ornith routed IQ1 tensors, selected expert-slice staging into compact Metal
  buffers to avoid sparse mmap GPU page faults, staged Q4 shared-expert Metal
  matvecs, a specialized block-256 Q4 router by default, and optional `trace`
  timing output from `ornith_metal_step_smoke`. Router modes:
  `ORNITH_METAL_ROUTER=0` restores CPU router scoring for A/B,
  `ORNITH_METAL_ROUTER=serial` uses the old serial Metal accumulation path,
  `ORNITH_METAL_ROUTER=parallel` uses the generic parallel Metal matvec, and
  the unset default uses the specialized Q4 router.
  `ornith/ornith_step_smoke.c` runs a bounded native token-step smoke from a
  TSV catalog, with optional vocab cap for fast real-model probes. Its
  optional `decode` mode validates attention tensor layout and uses
  `post_attention_layernorm` before MoE. Full-attention layers implement the
  first-token causal shortcut. Linear-attention layers implement an exact
  zero-prior-state first-token Gated DeltaNet path for CPU decode smoke:
  q/k/v conv, q/k L2 norm, headwise beta gate, per-value-head gated RMSNorm,
  and output projection. `ornith_decode_sequence_smoke_limited` adds a narrow
  CPU sequence path with persistent per-linear-layer conv/SSM state and
  per-full-attention-layer KV state. Ornith/Qwen3.5 layer and q/k norms are
  Gemma-style RMSNorm (`x * (1 + weight)`). Full-attention sequence smoke
  applies q/k RMSNorm, text-only partial RoPE, causal attention, per-head
  q/gate unpacking, optional q-gate, and output projection.
  `ornith/ornith_generate.c` runs the current greedy CPU reference generator.
  Built with `ORNITH_WITH_METAL`, the same CLI supports a Metal hybrid backend
  for attention projection/output matvecs, fused linear-attention GDN+out-proj,
  post-attention MoE, and lm-head scoring. Recurrent conv/KV/SSM orchestration,
  RoPE/softmax, residuals, and CPU fallback logic still live on the CPU side.
  It also supports `--worker`, a persistent stdin/stdout request loop used by
  `ornith/tools/ornith_chat.py --interactive` so interactive sessions reuse the
  mapped catalog/shards instead of spawning a new native generator per turn.
  This is process/mmap reuse only; the frontend still sends a full rendered
  prompt each turn until native KV/session reuse is implemented.
  `ORNITH_METAL_ATTN_MATVEC=0`, `ORNITH_METAL_BATCH_MATVEC=0`,
  `ORNITH_METAL_GDN=0`, and `ORNITH_METAL_ROUTER=0` disable those decode hooks
  for A/B checks.
  Verified real smokes on the full quantized `.ornq` set:
  raw `2+2=` generates token 19 (`4`), and the chat-shaped prompt starts with
  token 248068 (`<think>`). The earlier Metal MoE/lm-head hybrid dropped raw
  `2+2=` full-vocab generation from about 69.5s to about 35s. Metal attention
  matvec + GDN hooks ran raw `2+2=`, max_new=1, 60 layers, top_k=10,
  full vocab in about 9s. Batched Metal projection matvecs plus fast BF16
  RMSNorm plus fast attention scalar decode ran that probe in about 3.8s.
  Serial Metal router scoring now runs full-vocab max_new=1 in about 3.18s,
  full-vocab max_new=8 in about 4.91s, full-vocab max_new=16 in about 6.87s,
  and capped-vocab max_new=16 in about 6.31s. Parallel Metal router hit about
  9.58s for capped-vocab max_new=32 with unchanged token IDs but larger score
  drift than serial router. Fused GDN+out-proj keeps token IDs and scores
  unchanged and reduced full-vocab max_new=16 samples by about 0.36-0.53s.
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

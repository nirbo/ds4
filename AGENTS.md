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
- aggressive routed-expert compression, but current measured IQ1 is too lossy;
  DS4-style Q2 candidates are now the active quantization ladder before REAP
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

Raw shard 2 is explicitly approved and preserved for repeated quantization
experiments at
`/Users/nir/dev/models/Ornith-1.0-397B/raw-cache/model-00002-of-00122.safetensors`.
Do not delete it unless the user says the quantization experiments are finished.
The transient quant-error raw scratch at
`/Users/nir/dev/models/Ornith-1.0-397B/quant-error/raw` is disposable.

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
equations, the upstream Hugging Face Qwen3.5 MoE model source used to verify
raw checkpoint layouts, and a shallow clone of `CerebrasResearch/reap` for
REAP pruning reference code. Important verified layouts:

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

Use `ornith/tools/bench_metal_decode.sh` for repeatable local Metal decode
performance checks. It writes `ornith-metal-bench.log` by default and compares
selected expert cache budgets on fixed raw-token prompts.

## Quality Rules

- Keep the runtime model-specific, not generic.
- Validate tensor metadata strictly before running kernels.
- Keep public APIs narrow: frontends should not know tensor internals.
- Add the smallest runnable check for non-trivial logic.
- Performance changes are welcome only when tested for correctness and numeric
  drift.
- Current Apple Metal API findings and leverage order are documented in
  `ORNITH.md` under "Metal API Findings". The shortest useful path is command
  scheduling first, then GPU-owned router/expert residency; avoid adding heaps,
  indirect command buffers, or Metal 4 sparse machinery before that dependency
  is real.

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
  matvecs with fused gate/up + SiLU product, persistent host scratch reuse for
  Metal generation MoE hooks, a specialized block-256 Q4 router by default,
  row-8 Q4 block-256 projection/output/down matvecs by default, opt-in full
  routed-expert layer residency via
  `ORNITH_METAL_RESIDENT_LAYER_MB`, shared-expert Q4 residency by default via
  `ORNITH_METAL_SHARED_RESIDENT_MB`, default overlap of shared-expert GPU work
  with CPU selected-routed-slice staging via `ORNITH_METAL_OVERLAP_SHARED`,
  selected routed-expert IQ1 slice caching by default via
  `ORNITH_METAL_SELECTED_EXPERT_CACHE_MB=512` with `0` disabling it and
  `2048` covering all 60 layers for longer generations,
  fused Metal linear attention by default via `ORNITH_METAL_LINEAR_ATTN`,
  opt-in linear-attention Q4 weight residency via
  `ORNITH_METAL_LINEAR_RESIDENT_MB`, fused resident Metal self attention for
  `token_cap <= 256` by default via `ORNITH_METAL_SELF_ATTN`, optional `trace`
  timing output from `ornith_metal_step_smoke`, token-loop input RMSNorm fused
  into the linear/self-attention Metal command buffer when profiling is off,
  token-loop final RMSNorm + lm-head + raw top-k encoded into one command
  buffer for `k <= 64`, token-loop final hidden copyback skipped by default
  with `ORNITH_METAL_TOKEN_X_COPYBACK=1` for A/B,
  opt-in router top-k/softmax on Metal via `ORNITH_METAL_ROUTER_TOPK=1`,
  opt-in default-path lm-head GPU top-k via `ORNITH_METAL_LMHEAD_GPU_TOPK=1`
  with a parallel greedy `k=1` reduction,
  experimental opt-in GPU-selected resident routed MoE via
  `ORNITH_METAL_GPU_SELECTED_ROUTE=1` fused into the router/top-k command
  buffer when resident expert tensors are available, and
  real-generation timing with `ORNITH_METAL_PROFILE=1`. Router modes:
  `ORNITH_METAL_ROUTER=0` restores CPU router scoring for A/B,
  `ORNITH_METAL_ROUTER=serial` uses the old serial Metal accumulation path,
  `ORNITH_METAL_ROUTER=parallel` uses the generic parallel Metal matvec, and
  unset, `ORNITH_METAL_ROUTER=specialized`, or `q4` uses the specialized Q4
  router.
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
  mapped catalog/shards and native `ornith_session` state instead of spawning a
  new native generator per turn. Worker reuse is exact-prefix only: if the next
  rendered prompt is not an extension of the stepped token history, or if the
  session capacity is too small, the worker resets safely. Python interactive
  history stores the assistant prefill scaffold plus decoded completion so
  ordinary chat turns can hit native KV/SSM reuse.
  `ORNITH_METAL_ATTN_MATVEC=0`, `ORNITH_METAL_BATCH_MATVEC=0`,
  `ORNITH_METAL_GDN=0`, `ORNITH_METAL_LINEAR_ATTN=0`,
  `ORNITH_METAL_SELF_ATTN=0`, and `ORNITH_METAL_ROUTER=0` disable those
  decode hooks for A/B checks.
  `ORNITH_METAL_Q4_ROW8=0` restores the older row-4 Q4
  block-256 projection matvec path.
  Quality localization notes: the local Python env lacks Hugging Face
  `tokenizers`, so `ornith_chat.py` uses its fallback encoder; for the
  fizzbuzz prompt the fallback still encodes Ornith special tokens as single
  IDs. The failing `--nothink` fizzbuzz prompt matched CPU and Metal on the
  first token (`248068` / `<think>`, scores within ~4e-5), but CPU full-vocab
  60-layer first-token took about 249s, so full operating-point CPU checks are
  opt-in via `ORNITH_OPERATING_GOLDEN=1 tests/ornith_cpu_metal_golden_test.py`.
  `--no-think-scaffold --nothink` avoids the empty think-block prefill and
  changes the failure from "meaningless task" to coherent repetition, but
  still does not produce code. One-shot `ornith_chat.py` supports sampled
  decoding with `--temperature`, `--sample-top-k`, `--top-p`, and `--seed`
  for quality probes; interactive worker sampling is intentionally not wired.
  Sampling and longer thinking-enabled fizzbuzz probes still failed, so
  quantization degradation or a shared runtime math/layout bug remain live.
  `ornith/tools/ornith_quant_error.py` performs raw-BF16 versus dequantized
  `.ornq` error reports with a C scanner for full-tensor comparisons. Shard 2
  IQ1 gate/up showed relative L2 `0.603748`; shard 3 BF16 was exact, Q4 was
  `0.13132`, and IQ1 down was `0.950618`. Shard 3 re-quantized
  byte-identically, so the current evidence points to the IQ1 recipe being too
  lossy rather than a corrupt quantization run. Reports are stored outside the
  repo at `/Users/nir/dev/models/Ornith-1.0-397B/quant-error/reports/`.
  `ornith/tools/ornith_ds4_quant_candidate_error.py` measures copied DS4
  quantizers (`iq2_xxs`, `q2_k`, `q4_k`) against raw BF16 tensors without
  writing large candidate shards. On layer-0 `gate_up_proj`, synthetic-imatrix
  `IQ2_XXS` relative L2 was `0.657291`, `Q2_K` was `0.297341`, and `Q4_K` was
  `0.0716374`. On layer-0 `down_proj`, `IQ2_XXS` was `0.743402`, `Q2_K` was
  `0.441085`, and `Q4_K` was `0.0546557`. DS4's published recipe uses
  `IQ2_XXS` for routed gate/up and `Q2_K` for routed down with a real imatrix;
  without a real Ornith imatrix, `Q2_K` is the smallest promising measured
  candidate so far and `Q4_K` is the current quality ceiling.
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
  The Q4 block-256 Metal matvec now has a DS4-inspired row-4 path. Synthetic
  tests cover a non-multiple-of-four row count, and paired real smokes kept
  token IDs/scores unchanged while improving max_new=8 60-layer samples by
  roughly 4-8%.
  The IQ1 block-256 routed-expert slice path also has a row-8 Metal kernel.
  The Metal matvec test calls it through an `ORNITH_TESTING` wrapper with a
  non-multiple-of-eight row count. Paired full-vocab max_new=16 samples kept
  token IDs/scores unchanged, improved from about 7.31s to about 5.67s with
  row-4, then reached about 5.51-5.72s with row-8.
  The specialized Q4 router also uses a row-8 Metal kernel, covered through an
  `ORNITH_TESTING` wrapper in the Metal matvec test. Paired max_new=32 samples
  kept token IDs/scores unchanged and trimmed roughly 2% from the current
  generation path.
  General Q4 block-256 projection matvecs now also default to the row-8 kernel.
  Paired full-vocab raw-token `0,1` samples kept token IDs unchanged, showed
  expected small score drift from reduction order, and improved max_new=32
  from about 7.04s to 6.92s and max_new=64 from about 11.85s to 11.38s.
  Metal generation MoE hooks reuse their host scratch/indices across layers
  and tokens; paired 60-layer capped-vocab max_new=16/32 samples kept token
  IDs/scores unchanged and were neutral to modestly faster while removing
  per-layer allocator churn.
  `ORNITH_METAL_SHARED_RESIDENT_MB` defaults to 512 MiB and keeps shared-expert
  Q4 matrices in Metal shared buffers instead of copying them every layer/token.
  Set it to `0` to disable. It preserved
  token IDs/scores and improved full-vocab raw-token `0,1` max_new=64 from
  about 11.30-11.38s to 11.12-11.17s and max_new=128 from about 20.94s to
  19.99s. Set `0` when memory pressure matters more than a small speedup.
  `ORNITH_METAL_RESIDENT_LAYER_MB=1024` keeps the first fitting full routed
  expert layer resident in Metal shared buffers. It preserves token IDs/scores
  and helped longer top_k=10 capped-vocab samples, but stays opt-in because it
  spends about 1 GiB.
  The default Metal linear-attention hook fuses QKV/Z/A/B projections,
  depthwise conv+SiLU, GDN recurrence, and out-proj in one command buffer, and
  keeps per-layer conv/SSM state plus linear-attention constants resident for
  one-shot generation. Public session calls copy recurrence state back so
  follow-up session calls remain correct. Set `ORNITH_METAL_LINEAR_ATTN=0` to
  restore the older batch-projection + GDN hook path. A warm 60-layer
  capped-vocab raw-token `0,1` sample (`max_new=10`, `top_k=4`,
  `vocab_limit=128`) kept token IDs unchanged and improved from 6.584314 s to
  3.033750 s, with small expected score drift.
  `ORNITH_METAL_LINEAR_RESIDENT_MB=3072` additionally keeps linear-attention
  Q4 projection weights resident. It kept token IDs unchanged and helped a
  capped top_k=4 sample (`max_new=64`, `vocab_limit=32`) from 12.879188 s to
  8.551661 s, but was slower on the warm realistic top_k=10 full-vocab sample
  (10.493942 s default mapped weights versus 10.777743 s resident), so the
  default is off.
  The default Metal self-attention hook covers Ornith's periodic full-attention
  layers for decode sessions with `token_cap <= 256`: q/k/v projections, q/k
  norm, RoPE, resident KV append, causal softmax/value mix, gate, and out-proj
  run in Metal. Longer contexts fall back before the hook owns KV state. Set
  `ORNITH_METAL_SELF_ATTN=0` to restore the older self-attention matvec path.
  A warm 60-layer capped-vocab raw-token `0,1` sample (`max_new=10`, `top_k=4`,
  `vocab_limit=128`) kept token IDs unchanged and improved from 3.021742 s to
  2.983191 s; a longer supported sample (`max_new=64`, `vocab_limit=32`) was
  neutral within noise, 7.969136 s off versus 7.975130 s on.
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
  blocks by default, small/sensitive tensors stay BF16, and other BF16 matrix
  tensors use Q4. It also accepts JSON policies via `--policy`; policy rules
  support exact name, substring, regex, and layer ranges.
- `ornith/tools/ornith_quant_policy_report.py`: applies a quant policy to the
  local runtime catalog without reading raw weights. Current checked policies:
  `ornith-routed-q4.policy.json` projects to `187.45 GiB`, and
  `ornith-routed-last6-q4.policy.json` projects to `65.95 GiB` (`+13.50 GiB`
  over current `.ornq`).
- `ornith/tools/ornith_reap_plan.py`: guarded REAP-style expert pruning plan
  builder. It consumes future observer JSON, prunes lowest per-layer saliency,
  preserves activation outliers plus top frequency/REAP experts, and enforces
  `--min-retained`. It preserves zero-frequency/unobserved experts by default;
  use `--allow-prune-unobserved` only for explicit experiments with adequate
  calibration coverage. It writes manifests only; it does not edit weights.
- `ornith/ornith_reap_observe.c`: native REAP observer CLI. It runs the normal
  decode path with a MoE hook, emits planner-compatible JSON, and currently
  records selected experts only. `ornith/check.sh` compiles it always and runs
  a tiny observe-to-plan smoke when the full local `.ornq` catalog is present.
- `ornith/tools/ornith_reap_calibrate.py`: calibration wrapper around
  `ornith_reap_observe`. It accepts comma-separated token-id
  prompt lines, optionally text prompts with `--text-prompts --tokenizer`, and
  uses native `--prompts` mode so the model is loaded/mapped once for the whole
  prompt file. The observer records expert counts per layer, so it remains
  compatible with future REAP-pruned layers that have different retained counts.
- `ornith/tools/ornith_reap_size_report.py`: estimates post-REAP `.ornq` size
  from a keep/drop plan plus quant policy. It only shrinks layers present in
  the plan; unobserved layers are unchanged.
  A two-prompt all-layer native smoke (`max_new=1`, `top_k=10`) took about 86s
  with one model mapping. With default unobserved preservation and
  `min_retained=384`, it pruned 1,630/30,720 experts and projected `62.80 GiB`
  under `ornith-routed-last6-q4`; allowing unobserved pruning projected
  `50.60 GiB` but is not quality-safe from such a tiny calibration set.
- `ornith/tools/ornith_reap_repack_ornq.py`: materializes a REAP plan against
  existing `.ornq` shards by copying unplanned tensors and slicing planned
  routed experts plus matching router rows along the leading expert dimension.
  It records `reap_retained_experts` in output headers. This is for local
  reduced-shard validation without raw weights; the preferred final quality
  path is still REAP on raw weights before quantization.
- `ornith/tools/ornith_runtime.py`: reference `.ornq` loader/catalog,
  memory report, and CPU dequant/matvec helpers. It is not the final inference
  runtime.
- `ornith/tools/ornith_runtime_catalog.py`: builds the compact runtime tensor
  catalog from `.ornq` shards and validates exact text tensor coverage when the
  safetensors index is available. Use `--native-out` to write the TSV consumed
  by `ornith.c`.
- `tests/ornith_*`: focused tests and metadata/quantization checks.

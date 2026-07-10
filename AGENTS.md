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
- aggressive routed-expert compression, but measured IQ1 and activation-aware
  IQ2_XXS are both too lossy for routed gate/up; the active quality floor is
  per-expert-imatrix `Q2_K` for routed gate/up and down, `Q8_0` dense matrices,
  and BF16 routing/norm/state tensors
- the existing `quant-full`-derived REAP observations and plans are diagnostic
  only; final REAP data must be collected from original BF16/FP16 inference
- final compression must test quant-only, REAP-only, and combined candidates
  separately; do not infer quality from arithmetic prompts
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

Corrected compression tools and rules:

- `ornith_build_calibration_dataset.py` builds the deterministic coding-heavy
  calibration JSONL.
- `ornith_collect_bf16_calibration.py` is the full-residency external collector.
- `ornith_layer_calibration_bench.py` and
  `run_layer_calibration_bench.sh` prove the local layer-streamed BF16 path on
  MPS. The local environment is at
  `/Users/nir/dev/models/Ornith-1.0-397B/calibration-env`.
- `ornith_imatrix_manifest.py` validates all binary imatrix payloads.
- `ornith_reap_plan.py` defaults to final quality gates; use
  `--quality-profile experiment` only for explicitly diagnostic plans.
- `ornith-ds4-q2-q2-imatrix.policy.json` is the active low-bit quality floor.
  `ornith-ds4-iq2-q2-imatrix.policy.json` is retained only for rejected-format
  regression evidence.
- `run_quant_stream.sh` binds state to hashes of policy, REAP plan, and imatrix;
  changing any of them requires a new job directory. Final jobs also pin
  `HF_REVISION` to the immutable revision recorded by calibration.

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

The old `quant-smoke/` directory was deleted on 2026-07-07 to recover disk
space; recreate with `ornith/tools/ornith_stream_run.py --max-shards N` if
another small streaming smoke is needed.

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
  candidate so far and `Q4_K` is the current quality ceiling. `.ornq` now
  supports `q2_k` write/read/validate/reference decode through the copied DS4
  quantizer. A preserved raw shard-2 full-tensor q2_k smoke measured relative
  L2 `0.297341`, `rmse` `0.000313371`, and `max_abs` `0.0197601` against BF16
  source; the native C q2_k tensor-value and slice-matvec path was also probed
  on a nonzero decoded value. The temporary 1.3G q2-smoke `.ornq` was deleted
  after validation; reports remain under
  `/Users/nir/dev/models/Ornith-1.0-397B/quant-error/reports/`.
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
  support exact name, substring, regex, and layer ranges. It accepts
  `--reap-plan PLAN.json` to slice raw BF16 routed experts and router rows
  before quantization. `ornith/run_quant_stream.sh` forwards this with
  `REAP_PLAN=/path/to/plan.json`; quant policies pass with
  `QUANT_POLICY=/path/to/policy.json`. Raw shard 2 smoke with
  `reap-calibration-77pct/plan-r0.25.json` validated a 512->384 expert
  `gate_up_proj` slice against the original raw safetensors source.
  Current overnight candidates:
  `quant-reap35-last19-q4` used `plan-r0.35.json` and
  `ornith-reap35-routed-last19-q4.policy.json`, projected `63.52 GiB`.
  The completed full run was deleted on 2026-07-07 to recover disk:
  `/Users/nir/dev/models/Ornith-1.0-397B/quant-reap35-last19-q4`: 122 `.ornq`
  shards, 122 state entries `done`, no raw `.safetensors` left in `raw/`,
  catalog files `catalog.json` and `catalog.tsv`, and actual catalog payload
  about `63.52 GiB` (`22.82 GiB` IQ1 payload, `45.38 GiB` Q4 payload, plus
  BF16 norms). Native loader passes with 122 shards, 1038 tensors, 60 layers;
  4-layer decode and CPU generation smokes agree. Mixed IQ1/Q4 routed layers
  required a Metal Q4 3D slice fallback, now covered by
  `ornith_metal_matvec_test`. Real Metal smokes pass through 59 layers with
  `expert_top_k=1`; the retained Q4 routed tail originally needed a
  fused/staged Metal path. That path now accepts Q4 expert tensors as well as
  IQ1 and brings the full 60-layer `top_k=1`, vocab-32 probe to about 10s.
  Quality result after that fix: non-REAP full quant returns token `19` (`4`)
  for raw `2+2=`, `quant-reap35-last19-q4` returns token `241784` (`Золо`),
  and the lighter local diagnostic
  `/Users/nir/dev/models/Ornith-1.0-397B/reap-r10-min448` (10% prune, 461
  experts/layer, 48G) returns `4` and continues `2+2=4`. Treat 35% pruning as
  too aggressive for the current calibration/selection recipe; 10% preserves
  the arithmetic smoke but still fails the short fizzbuzz coding probe.
  `reap-r10-min448` was deleted on 2026-07-04 to recover disk; recreate from
  `quant-full/out` plus `reap-calibration-77pct/plan-r0.10-min448.json` if
  needed.
  Earlier writable candidate was
  `ornith/policies/ornith-reap10-routed-last6-q4.policy.json` with
  `reap-calibration-77pct/plan-r0.10-min448.json`: 10% REAP, 461 retained
  experts/layer, last 6 routed layers Q4, projected about `59.83 GiB`.
  Preserved raw shard 2 smoke passed after slicing layer-0 `gate_up_proj` to
  461 experts and quantizing as IQ1 (`mse=6.3994e-07`,
  `max_abs=0.00500488`). DS4-style candidate error on that raw tensor measured
  `q2_k` much better than `iq2_xxs` (`relative_l2` `0.297` vs `0.657`).
  Full `quant-reap10-last6-q4` run completed cleanly at 60G but failed raw
  `2+2=`, returning token `85557` (`aab`). Likely policy bug: default Q4
  changed 150 small/sensitive tensors from BF16 to Q4 (`linear_attn.A_log`,
  `linear_attn.dt_bias`, `mlp.shared_expert_gate.weight`). Corrected rerun
  policy is
  `ornith/policies/ornith-reap10-last6-q4-sensitive-bf16.policy.json`; it keeps
  those tensors BF16, still projects about `59.83 GiB`, and passed preserved
  raw shard 2 smoke. Full `quant-reap10-sensitive-last6-q4` run completed with
  122 state entries `done`, 122 `.ornq` shards, no raw `.safetensors` left,
  and catalog payload `59.83 GiB` (`38.74 GiB` IQ1, `21.09 GiB` Q4, plus
  BF16 sensitive tensors). CPU/Metal golden smoke passed. Raw `2+2=` returned
  token `19` (`4`) and continued coherently as `4，4+4=8，`; the short
  fizzbuzz coding probe still repeated the request instead of producing code.
  Preserved raw shard 2 validation passed (`iq1`, 4096 samples,
  `mse=4.00805e-07`, `max_abs=0.00958252`). Treat this artifact as valid and
  arithmetic-safe, but not coding-quality-safe. The failed
  `quant-reap10-last6-q4` artifact was deleted on 2026-07-05 to recover about
  60G. The `quant-reap10-sensitive-last6-q4` artifact was deleted on
  2026-07-08 to recover disk; reports and notes remain.
  Current q2_k policy size results from the local 77% REAP calibration:
  full routed q2_k plus last-6 Q4 is too large (`116.81 GiB`), routed-down
  q2_k plus last-6 Q4 is also too large (`78.83 GiB`), routed-down q2_k with
  no Q4 tail projects to `68.78 GiB` at 10% REAP, `65.30 GiB` at 15% REAP,
  `61.68 GiB` at 20% REAP, `58.06 GiB` at 25% REAP, and `50.96 GiB` at 35%
  REAP. A narrower policy,
  `ornith/policies/ornith-reap-routed-down-q2k-last1-q4-sensitive-bf16.policy.json`,
  keeps only routed layer 59 at Q4; with the new `plan-r0.20.json` it projects
  to `63.17 GiB`, making it the current best 64GB-target candidate. Full
  `quant-reap20-down-q2k-last1-q4` completed on 2026-07-09 with 122 state
  entries `done`, 122 `.ornq` shards, no raw `.safetensors` or `.part` files
  left, catalogs present, and HF cache deleted. Actual catalog payload is
  `63.17 GiB`: `26.95 GiB` IQ1, `33.29 GiB` q2_k, `7.58 GiB` Q4, plus BF16
  sensitive tensors. Native loader and 1/4-layer decode smokes passed. Correct
  raw `2+2=` token prompt `17,10,17,28`, full-vocab Metal, 60 layers,
  `expert_top_k=10`, returned token `19` (`4`) for `max_new=1`; the 8-token
  continuation was weak but structured: `4\nA.\nB.\n`. Keep the artifact for
  comparison, but it is not coding-quality-positive yet. The artifact was
  deleted on 2026-07-09 to recover disk. Full
  `quant-reap25-down-q2k` completed on
  2026-07-07 with 122 state entries `done`, 122 `.ornq` shards, no raw
  `.safetensors` or `.part` files left, catalog files present, and actual
  payload `58.058 GiB` (`25.67 GiB` IQ1, `31.71 GiB` q2_k, `4.96 GiB` Q4,
  plus BF16 sensitive tensors). Native loader passed, 1-layer and 4-layer CPU
  decode smokes passed, and a full 60-layer raw `2+2=` CPU probe with
  `expert_top_k=1`, `vocab_limit=32` returned token `17` (`2`) in 38.47s.
  A Metal q2_k block-256 slice kernel now covers routed down-proj and the
  fused routed MoE path accepts IQ1 gate/up plus q2_k down. The full 60-layer
  capped raw-token `0,1` probe returns the same token as CPU and improved from
  about 24.7s to 18.1s after cleanup; routed fused time is no longer the
  bottleneck. Full-vocab raw `2+2=`, `expert_top_k=10`, returns token `19`
  (`4`), but the 8-token continuation is weak (`4\nA. 2+2`), so this
  artifact is still not coding-quality-positive. The artifact was deleted on
  2026-07-08 to recover disk; reports and notes remain.
  `quant-reap40-last22-q4` uses `plan-r0.40.json` and
  `ornith-reap40-routed-last22-q4.policy.json`, projected `63.15 GiB`.
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
  calibration coverage. `--strategy hybrid` combines normalized REAP,
  frequency, EAN, and max-activation scores; `--layer-profile late-protect`
  keeps the total prune target but shifts more pruning into earlier layers and
  less into the final quarter. Plan manifests include per-layer
  observed/unobserved coverage fields. It writes manifests only; it does not
  edit weights. Generated candidates:
  `plan-r0.20-hybrid-lateprotect.json` projects to `63.32 GiB` with
  `ornith-reap-routed-down-q2k-last1-q4-sensitive-bf16.policy.json`, and
  `plan-r0.25-hybrid-lateprotect.json` projects to `59.65 GiB`.
  A disk-local repack from `quant-full/out` using the 20% hybrid late-protect
  plan lives at
  `/Users/nir/dev/models/Ornith-1.0-397B/hybrid-r20-lateprotect-repack`: 122
  shards, cataloged `42.93 GiB` payload, native loader and 4-layer decode
  smoke passed, and full-vocab Metal `2+2=` (`17,10,17,28`) generated
  `4\n2+2=4\n` for 8 tokens. This is the current best pruning-plan evidence,
  but it is repacked from already-quantized `quant-full`; final quality still
  needs raw-weight quantization with the same plan. The repack artifact was
  deleted on 2026-07-09 to recover disk.
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
  Use `--max-prompt-tokens N` to bound prefill cost during coverage probes.
- `ornith/tools/ornith_reap_merge_observations.py`: merges multiple REAP
  observer JSON files by summing counts/weights, recomputing frequency-weighted
  means, and keeping max activations. Use it to accumulate disk-light
  calibration batches without rerunning previous prompts.
  Current merged smoke result: 11,129/30,720 observed expert slots (`36.2%`),
  7,632 pruned experts, 384-394 retained per layer, projected `50.68 GiB`.
  Adding one bounded 24-prompt/four-token coding batch raised coverage to
  15,449/30,720 (`50.3%`) and reached the full safe 25% prune: 7,680 experts
  pruned, 384 retained per layer, projected `50.60 GiB`.
  Current stronger calibration lives at
  `/Users/nir/dev/models/Ornith-1.0-397B/reap-calibration-77pct`: 23,665/30,720
  observed slots (`77.0%`). It contains `observations.json`, `summary.json`,
  and 25/30/35/40% prune plans. Current-quant projected sizes are
  40.48/38.14/35.71/33.37 GiB.
- `ornith/tools/ornith_reap_size_report.py`: estimates post-REAP `.ornq` size
  from a keep/drop plan plus quant policy. It only shrinks layers present in
  the plan; unobserved layers are unchanged.
  A two-prompt all-layer native smoke (`max_new=1`, `top_k=10`) took about 86s
  with one model mapping. With default unobserved preservation and
  `min_retained=384`, it pruned 1,630/30,720 experts and projected `62.80 GiB`
  under `ornith-routed-last6-q4`; allowing unobserved pruning projected
  `50.60 GiB` but is not quality-safe from such a tiny calibration set.
  A bounded four-prompt coding-token smoke took about 3.5 minutes, reached
  5,780/30,720 observed expert slots (`18.8%` coverage), pruned 4,602 experts,
  retained 403-474 per layer, and projected `56.83 GiB`; still not enough
  coverage for a final quality-sensitive cut.
  A bounded twelve-prompt/four-token coding smoke reached 10,359/30,720
  observed slots (`33.7%`), pruned 7,490 experts, retained 384-405 per layer,
  and projected `50.92 GiB`; this is the current best disk-light probe, still
  pending broader calibration and downstream quality checks.
- `ornith/tools/ornith_reap_repack_ornq.py`: materializes a REAP plan against
  existing `.ornq` shards by copying unplanned tensors and slicing planned
  routed experts plus matching router rows along the leading expert dimension.
  It records `reap_retained_experts` in output headers. This is for local
  reduced-shard validation without raw weights; the preferred final quality
  path is still REAP on raw weights before quantization. Re-runs skip valid
  destination shards, and `--max-shards N` bounds smoke runs.
  Earlier reduced artifact lived at
  `/Users/nir/dev/models/Ornith-1.0-397B/reap-keep384-50pct`: 122 shards,
  1038 tensors, 60 layers, `40.48 GiB` output from `52.45 GiB` source,
  `11.97 GiB` saved. Catalogs are `catalog.json` and `catalog.tsv`.
  Validation passed native loader, 4-layer decode smoke, full 60-layer CPU
  capped-vocab generation, full 60-layer Metal capped-vocab generation, and
  full 60-layer Metal full-vocab generation. It was deleted on 2026-07-04 to
  recover disk. Keep `quant-full/out` as fallback and source for alternate
  REAP plans until explicitly removed.
- `ornith/tools/ornith_runtime.py`: reference `.ornq` loader/catalog,
  memory report, and CPU dequant/matvec helpers. It is not the final inference
  runtime.
- `ornith/tools/ornith_runtime_catalog.py`: builds the compact runtime tensor
  catalog from `.ornq` shards and validates exact text tensor coverage when the
  safetensors index is available. Use `--native-out` to write the TSV consumed
  by `ornith.c`.
- `tests/ornith_*`: focused tests and metadata/quantization checks.

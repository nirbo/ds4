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

`NEMOTRON_EXPERIMENTS.md` is the durable forward-work ledger. Each experiment
uses its own feature branch. Mark its checkbox only after implementation,
correctness testing, quality measurement, and performance/memory measurement,
then record `SUCCESS`, `PARTIAL`, or `REJECTED` with evidence before merging.

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
directory. `source-notes/revisions.json` pins the ModelOpt, vLLM, MLX-LM, oMLX,
and Gefen commits used to establish NVFP4 decode, model, calibration, MTP,
runtime, and compact optimizer semantics. It also pins LiveCodeBench, NeMo
Skills, and NeMo Evaluator revisions used for the dated coding protocol. The
small LiveCodeBench checkout lives at `source-notes/livecodebench`. These
repositories are reference code only; copy and adapt required model-specific
logic into `nemotron_*` files.

The rejected Apple Mojo kernel experiment uses
`$NEMOTRON_MODEL_DIR/mojo-env-26.4` (Modular 26.4, Mojo 1.0.0b2). Its pinned
Modular and MLX source references live at `source-notes/modular/` and
`source-notes/mlx/`. `nemotron/run_mojo_moe_spike.sh` is the logged opt-in
runner. Do not add Mojo to the resident Apple runtime unless a future kernel
beats the matched MLX control and a zero-copy ownership boundary is proven.

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
  `$NEMOTRON_MODEL_DIR/mlx-env` and is pinned to validated MLX `0.32.0`; the
  check script skips it when unavailable and rejects a different installed
  MLX version.
- `nemotron/tools/nemotron_mlx_moe.py`: GPU-owned top-k LatentMoE path using
  MLX `gather_qmm` over ModelOpt NVFP4 experts. The routine check runs its
  scalar synthetic test; set `NEMOTRON_MLX_REAL_MOE=1` for the 3 GiB-peak real
  layer benchmark.
- `nemotron/tools/nemotron_mlx_pack.py`: direct, resumable prune-to-runtime
  materializer. It slices routers, renumbers retained experts, omits MTP when
  requested, and writes each layer's expert NVFP4 tensors pre-stacked without
  changing retained payload bytes. It supports strict uniform v1 plans and
  Nemotron-only nonuniform plans with explicit per-layer counts. Use
  `--max-groups N` for bounded smokes.
- `nemotron/tools/nemotron_mlx_linear.py`: ModelOpt FP8 and BF16 MLX linear
  primitives. FP8 defaults to native MXFP8 qmm with shared unity scales and the
  checkpoint scalar folded into activations; a custom Metal decoder remains
  the independent numerical reference. Its real-tensor benchmark accepts one
  to eight tokens; native MLX remains preferred after custom FP8 batching lost
  by about 1.5x at two tokens.
- `nemotron/tools/nemotron_mlx_mamba.py`: Nemotron Mamba2 composition over
  official BF16 convolution/state tensors and exact ModelOpt FP8 projections,
  with persistent MLX `ArraysCache` recurrence. Its model-specific Metal kernel
  executes 2-8 unpadded recurrent steps in one launch while preserving
  token-order state updates and optional rollback capture; longer, padded, and
  multi-capture sequences retain the independently gated fallback. Set
  `NEMOTRON_DISABLE_SHORT_SSM=1` only for matched reference diagnostics.
- `nemotron/tools/nemotron_mlx_moe_layer.py`: complete one-token LatentMoE
  layer with RMSNorm, BF16 routing, latent projections, top-22 packed experts,
  shared expert, and residual kept in one lazy MLX graph. The dominant
  BF16/BF16/FP8/FP8 layout uses one module-level compiled tail with every
  layer weight passed dynamically. Five static-signature compiled tails cover
  the other FP8/BF16/NVFP4 assignments, so all 40 MoE layers avoid per-layer
  weight-capturing graphs while preserving the checkpoint's exact mixed
  precision. Its benchmark accepts one to eight tokens and reports routing
  separately for verifier hotpath work.
- `nemotron/tools/nemotron_mlx_attention.py`: periodic full-attention layer
  with specialized BF16 decode projections and GPU-owned MLX KV cache. The
  checkpoint k/v scales are quantized-cache calibration metadata, not factors
  in ordinary BF16 attention.
- `nemotron/tools/nemotron_mlx_stream_forward.py`: low-memory official-source
  baseline runner. It retains KV/SSM state but loads/releases one layer at a
  time, supports layer-major prompt prefill, full-vocabulary logits, and
  per-layer router capture. Its revision-bound virtual-pruning mode must match
  a physical candidate exactly and enables byte-matched plan comparisons
  without another full artifact. It is a quality/calibration path, not the
  final resident-weight runtime.
- `nemotron/tools/nemotron_mlx_trajectory_attribution.py`: resumable source
  teacher replay over stored LiveCodeBench reasoning or MBPP response
  trajectories. It captures sparse hidden states after the prompt, measures
  exact virtual-pruning error, and ranks removed experts by route-weighted
  output contribution. Captures and reports are bound to source, dataset,
  report, token, plan, and tool hashes.
- `nemotron/tools/nemotron_mlx_trajectory_plan.py`: conservative add-only
  aggregation of trajectory evidence. It never removes a template expert,
  enforces per-layer addback floors and caps, and records exact added bytes.
  Failed-generation add480/add640 experiments reduced local source-output error
  but both scored 0/4 on disjoint hard generation gates and were deleted. The
  fixed-budget successor swaps 400 layer/expert identities within r25 without
  changing its 54.4974 GiB payload. Its materialized runtime is
  `candidate-mbpp-success-swap400-r25size-mlx`; it scores 75/100 deterministic
  MBPP and 155/164 HumanEval with bit-exact virtual/physical logits. Its matched
  hidden LiveCodeBench gate scores 36/60 samples and 22/30 tasks versus r25's
  37/60 and 22/30, a mixed paired trade without category collapse. Ordinary
  decode measures 24.462 tok/s; its candidate-bound 32K MTP map reaches 36.538
  tok/s with exact output at 54.333 GiB peak. It is the preferred balanced
  runtime; `candidate-mbpp-success-guard400-mlx` is the quality-headroom
  rollback and r25 remains the smaller control baseline.
- `nemotron/tools/nemotron_mlx_calibrate.py`: resumable diverse-corpus router
  observer. It aggregates counts, score mass, selected latent output norms,
  route-weighted output contribution, and maxima per expert.
- `nemotron/tools/nemotron_livecodebench_calibration.py`: provenance-bound
  coding-corpus builder with date-window and evaluated-task exclusion.
- `nemotron/tools/nemotron_livecodebench_compare.py`: strict paired evaluator
  comparison by task, repeat, and seed, including sample/task flips and failure
  classes.
- `nemotron/tools/nemotron_mlx_livecodebench.py`: resident, resumable coding
  gate. `--repeat-offset` selects a disjoint deterministic repeat range and is
  bound into report identity; nonzero offsets are intentionally nonstandard
  NVIDIA-protocol slices.
- `nemotron/tools/nemotron_mlx_protected_plan.py`: fixed-size specialist expert
  protection. It preserves broad core and unknown experts, admits only positive
  joint-score swaps, and never changes a layer's retained expert count.
- `nemotron/tools/nemotron_mlx_repack.py`: resumable incremental plan
  materializer. It hard-links mapping-identical runtime groups from a validated
  base candidate and rewrites only changed MoE layers from the pinned source,
  including add-only plans with different per-layer expert counts.
- `nemotron/tools/nemotron_mlx_prune_plan.py`: guarded plan builder. It ranks
  per layer from normalized activation evidence, protects every unobserved
  expert, and enforces prune-ratio-specific coverage thresholds.
- `nemotron/tools/nemotron_mlx_layer_sensitivity.py`,
  `nemotron_mlx_layer_allocate.py`, and `nemotron_mlx_plan_compare.py`: held-out
  layer curves, exact-budget dynamic programming, and resumable independent
  full-logit comparison for nonuniform plans. The current r25 candidate spans
  308-512 experts per layer, occupies 54.4974 GiB, and is quality-PARTIAL
  pending broader evaluation. Its first deterministic 100-task MBPP gate scored
  74/100, exactly tied with r20 with four paired wins each. It is the preferred
  broadly tested 64 GB control; the 55.6540 GiB success-attributed guard400
  candidate is preferred after scoring 75/100. A separate 20-task HumanEval
  gate also tied r25 and r20 at
  17/20 with identical pass/fail outcomes. The complete corrected HumanEval
  run scored 154/164 (93.90%). It runs at `23.610 tok/s` ordinary
  and `34.195 tok/s` with the candidate-bound MTP path, peaking at 54.333 GiB.
- `nemotron/tools/nemotron_mlx_nested_thin.py`: exact-total nested thinning and
  bounded coding-trajectory repair. The superseded repair150 artifact was
  removed after its 74/100 MBPP and 153/164 HumanEval evidence was retained.
  The current safe-memory frontier removes 1,000 experts strictly within the
  preferred r25 survivors. Its materialized runtime is
  rejected `candidate-r25-nested-remove1000-mlx`: `51.6059 GiB` logical and
  `50.6059 GiB` resident with paged embeddings. Its eight-category virtual gate
  retained 7/8 source top tokens with mean KL `0.07787`; physical/virtual
  logits are bit-exact and short decode reaches `24.423 tok/s`. Its complete
  MBPP gate scored 70/100 versus preferred swap400-r25size's 74/100, with two
  candidate-only and six control-only passes. Do not promote it or spend
  HumanEval/LiveCodeBench compute on the unchanged plan. Its fixed-size
  trajectory repair20 and repair40 successors reduced local source-output
  error but worsened independent eight-category mean KL, so neither was
  materialized. The physical remove1000 artifact was deleted after those gates;
  its plans and reports make it reproducible. Removing 200 more experts is also
  rejected because tool-calling KL rose to `2.28239` and changed top-1.
  The shallower remove400 runtime is the preferred memory-first fallback:
  `53.3408 GiB` logical, 74/100 MBPP, 155/164 HumanEval, and 36/60 hidden
  LiveCodeBench samples across 21/30 tasks. It saves `1.1566 GiB` versus the
  balanced runtime and is stable at a 57 GiB wired cap with bounded prefill.
  Under the earlier cacheless MTP control, its candidate-bound 32K map plus the
  candidate-specific `mtp-sidecar-e256-remove400-nvfp4` reached `45.751 tok/s`
  across repeated 512-token coding controls with exact target output and a
  `53.712 GiB` peak. The candidate-specific e128 sidecar is the 0.370 GiB
  smaller fallback, while shared `mtp-sidecar-e128-nvfp4` remains the
  broad-workload fallback. The cacheless depth-two preference and depth-three
  rejection are historical; the prompt-prefilled cache result documented under
  `nemotron_mlx_speculative.py` supersedes that runtime policy. A 512 MiB MLX
  allocator cache remains a measured regression.
- `nemotron/tools/nemotron_mlx_targeted_repair.py`: fixed-size same-layer
  source-teacher repair experiment over paired recovery and inverse-guard
  trajectories. Its 10-, 20-, and 40-swap repair150 plans all caused a severe
  tool-calling logit regression; retain the tool for reproducibility, but do
  not materialize those plans or treat local attribution improvement as a
  quality acceptance signal.
- `nemotron/tools/nemotron_mlx_layer_distill.py`: bounded teacher/candidate
  layer-output fitter. Per-channel affine correction overfit and scalar affine
  correction improved held-out r35 local output error by only 1.87%; neither is
  a runtime feature. Rank-4 residual regression also failed held-out validation.
  Future work must train replacement expert or router behavior rather than
  silently attaching any of these diagnostic corrections.
  A ReLU-squared latent adapter before `fc2_latent` also failed with 189
  training tokens; do not revisit small correction sidecars without materially
  new evidence or a larger expert-parameter training design.
- `nemotron/tools/nemotron_mlx_router_distill.py`: bounded retained-router
  trainer and strict virtual override loader. The r30 local-MSE experiment
  improved held-out layer error slightly but failed its two-case full-logit
  gate, including one source-top-token flip. Its artifact is diagnostic only;
  do not pack it or confuse local output fitting with end-to-end Router KD.
- `nemotron/tools/nemotron_mlx_kd_gradient_audit.py`: representative Mamba,
  attention, and MoE input-VJP audit for future streamed Router KD. Frozen
  native-BF16 fallbacks replace inference-only custom kernels during backward;
  NVFP4/FP8 weights remain quantized and frozen. The real audit peaks at
  5.266 GiB and passes sub-1e-6 forward-parity checks.
- `nemotron/tools/nemotron_mlx_streamed_router_kd.py`: resumable manual
  layer-streamed Router KD. It saves bounded forward activations, forms true
  full-vocabulary teacher KL, reloads one frozen layer at a time in reverse,
  and writes only retained BF16 router rows after a measured improving line
  search. Frozen BF16 and FP8 inference projections use gradient-capable
  exact-value fallbacks; expert NVFP4 payloads remain unchanged. The one-step
  r30 proof lives at
  `$NEMOTRON_MODEL_DIR/layer-distill/streamed-router-kd-r30-def-final` and is a
  mechanism certificate, not a deployable quality artifact.
- `nemotron/tools/nemotron_mlx_router_kd_train.py`,
  `nemotron_mlx_router_kd_ablate.py`, and `nemotron_mlx_router_kd_export.py`:
  multi-prefix Router KD, layer ablation, and exact source/trained composition.
  The accepted bounded-logit composition trains 30 MoE routers and restores
  the last 10, but its physical 51.6059 GiB candidate scored 73/100 MBPP and
  149/164 HumanEval versus unmodified r30's 74/100 and 150/164. It has no
  candidate-only task wins and is rejected for promotion. Do not treat its
  improved bounded KL as downstream quality acceptance.
- `nemotron/tools/nemotron_mlx_plan_compare.py` supports provenance-bound
  router reversion and BF16 delta damping for rejection analysis. Layer-30
  reversion and 0.75 damping recovered the MBPP regression but failed the
  independent logit gates; these are diagnostics, not runtime policies.
- `nemotron/tools/nemotron_mlx_shared_subspace.py`: bounded post-training
  shared-expert representation screen. It tests paired prototype/residual and
  shared input/output union bases against real routed latent inputs. All three
  formats failed even on the globally strongest supported expert pair; do not
  build kernels or materialize them without a genuinely trained representation.
- `nemotron/tools/nemotron_mlx_dense_proxy.py`: one-layer dense functional
  proxy gate. Evaluating every retained substitute on removed-expert contexts
  still lost to hard pruning; nearest-prototype replacement remains rejected.
- `nemotron/tools/nemotron_mlx_width_prune.py`: aligned 16-neuron NVFP4 width
  pruning gate. It preserves all 512 router choices and copies retained up rows,
  down columns, scales, and global scales exactly. At an equal 25% routed-byte
  cut, an independent screen confirmed width pruning beats whole-expert removal
  on 10/13 candidate layers. This is promising hybrid evidence, not yet a
  materialized or end-to-end accepted candidate.
- `nemotron/tools/nemotron_mlx_hybrid_plan.py` and
  `nemotron/tools/nemotron_mlx_hybrid_ablate.py`: strict mixed expert/width
  virtual plans and full-logit layer ablation. The broad and four-layer plans
  are rejected: local-error wins did not reliably preserve tool-calling top-1.
  Layer 54 is the only promoted width substitution. Its complete eight-category
  virtual run preserved 8/8 top-1 and improved mean KL and aggregate drift over
  nonuniform r25.
- `nemotron/tools/nemotron_mlx_hybrid_materialize.py`: incremental physical
  materializer. It hard-links every unchanged nonuniform-r25 file, rebuilds
  only the accepted width layer, writes compatible runtime and paged-embedding
provenance, and rereads every replacement tensor for exact equality. The
  layer-54 candidate was materialized at `candidate-hybrid-width54-r25-mlx`.
  It occupied `54.7182 GiB` logically but added only about 1.2 GiB of disk
  blocks. Its
  100-task MBPP gate scored 73/100 versus 74/100 for nonuniform r25, with one
  control-only pass and no hybrid-only wins. Keep it as evidence, not the
  preferred runtime candidate. Its derived artifact and the superseded r20
  candidate were removed after evaluation; compact reports remain under
  `quality/`, and both can be reproduced from the immutable source.
- `nemotron/tools/nemotron_mlx_compare_logits.py`: full-vocabulary baseline to
  candidate metrics, including centered drift, cosine, KL, top-k overlap, and
  baseline-top-token rank.
- `nemotron/tools/nemotron_mlx_mbpp.py`: deterministic, resumable MBPP pass@1
  evaluator for resident candidates. It resets only sequence state between
  tasks, records generated code and failures atomically, binds reports to all
  runtime inputs and tool versions, and executes assertions under a macOS
  sandbox with CPU, file-size, wall-time, filesystem, and network limits.
- `nemotron/tools/nemotron_mlx_humaneval.py`: HumanEval adapter over the same
  resident and sandbox boundaries. It handles full-function and body-only
  completions, preserves imports and helper definitions before the target,
  executes the official `check(candidate)` harness, and binds reports to both
  evaluator and shared helper source. `nemotron_mlx_humaneval_rescore.py`
  provenance-binds corrected offline scoring when stored deterministic
  responses outlive a harness fix.
- `nemotron/tools/nemotron_mlx_livecodebench.py`: resumable stdin/stdout
  and functional public-test evaluator for LiveCodeBench. It supports direct
  code, full thinking, low-effort thinking, seeded temperature/top-p sampling,
  and repeated samples under the same filesystem, process, network, CPU, and
  wall-time sandbox boundaries. It separates reasoning from final code and
  reports sample pass@1 independently from task pass-any. Reports explicitly
  list mismatches against NVIDIA's reference protocol. The earlier
  reasoning-disabled 30-task gate scored 16/30 but is not comparable to
  NVIDIA's score. A low-effort, temperature-1.0/top-p-0.95 smoke solved a
  previously failed hard task in 2/4 samples without truncation, proving the
  protocol materially affects the result. Official comparison still requires
  full thinking, eight repeats, the dated v5/v6 split, the official hidden
  tests, and a sufficiently high token cap. Its current NVIDIA AAI prompt,
  six-second timeout, all-public-case default, dated-state binding, and
  standard/low-budget protocol audits are aligned. Use `--dry-run` to validate
  future samples without loading weights.
- `nemotron/tools/nemotron_livecodebench_dataset.py`: pinned HTTP-range Parquet
  cataloger for the official LiveCodeBench release v5/v6 data. It verifies the
  complete local prompt/public-test copy, attaches official dates and
  functional metadata, and reports exact private-column transfer requirements
  without fetching hidden tests. The materialized public catalogs live in
  `quality/livecodebench-official-public-v5-2407-2412.jsonl` (315 tasks) and
  `quality/livecodebench-official-public-v6-2408-2505.jsonl` (454 tasks).
  Corresponding state files bind hashes and provenance. The v6 hidden tests
  have now been materialized and indexed; v5 hidden tests remain absent. The
  v5 compressed transfer would be 2,331,147,468 bytes.
- `nemotron/tools/nemotron_livecodebench_private.py`: resumable bounded-range
  v6 private-test materializer. It validates Xet object identity, processes one
  Parquet dictionary column at a time, writes one atomic JSONL output per
  source shard, and hash-checks every completed shard on resume. The completed
  corpus lives at `quality/livecodebench-private-v6-2408-2505/`: 454 tasks,
  15,684 tests, 2,478,970,742 transferred bytes, and 4,033,506,110 output bytes.
- `nemotron/tools/nemotron_livecodebench_private_index.py`: provenance-bound
  byte-offset index over the private JSONL outputs. Evaluators validate all
  source hashes once, then seek directly to selected task rows without loading
  the 3.8 GiB corpus.
- `nemotron/tools/nemotron_mlx_livecodebench_rescore.py`: provenance-bound
  offline rescoring for stored generations after dated-state or sandbox-harness
  changes. It supports the same indexed private tests and never regenerates
  model output. The first hidden rescore kept task `3525` passing across all 42
  public/private cases.
  A subsequent two-sample hard-task run solved `abc391_f` once and passed all
  43 public/private cases; the other sample reached the 8,192-token local cap.
  The completed 30-task balanced hidden gate scored 37/60 samples (61.67%):
  easy 18/20, medium 15/20, and hard 4/20, with six hard truncations and three
  execution timeouts. Task pass-any was 22/30. This is a low-budget staged
  result, not a comparison to NVIDIA's full-thinking 78.57% v6 score. Do not
  change pruning based on it without a matched unpruned or less-pruned control.
- `nemotron/tools/nemotron_mlx_resident.py`: packed-candidate resident generator
  with a hard Metal-cap preflight. Never bypass the preflight; a kernel wired
  limit below the reported requirement can fail allocation or destabilize the
  machine. The 20% candidate has now run successfully at `57.736 GiB` peak and
  about `23.6 tok/s` steady-state decode on the 64 GB M4 Max with the temporary
  `iogpu.wired_limit_mb=60672` setting. Use `--token-timings` for per-transition
  measurements; the CLI excludes the first prefill-produced token and avoids
  an unused final forward when calculating decode throughput. Its sequence
  path batches projection work but preserves one-token Mamba recurrence order;
  cache snapshots must restore both recurrent arrays and KV buffers/offsets.
  The generation CLI compiles each initialized, unpadded Mamba signature as a
  pure graph with explicit recurrent inputs/outputs; `--no-compile-mamba`
  restores the eager control. Full logits and every Mamba state are bit-exact.
  Launchers preserve the live preflight kernel cap after a run; they must never
  restore MLX's lower default over a user-approved `iogpu.wired_limit_mb`.
  Extended evaluations additionally require the payload plus margin to remain
  below both 85% of physical memory and MLX's 95%-of-working-set allocator-GC
  boundary. A July 13, 2026 remove400 LiveCodeBench run crossed the latter and
  triggered an `IOGPUGroupMemory::remove_memory_object()` kernel panic. Cache
  reset now synchronizes Metal and reuses KV/Mamba storage in place. Extended
  generic preflight reserves 3.25 GiB of transient workspace and quality
  runners enforce a live 128 MiB stop reserve. Resident prompt prefill now
  advances the same
  Mamba/KV state in report-bound 128-token chunks. On remove400's 793-token
  MBPP task 380, this reduced peak from 54.520 to 53.461 GiB, preserved top-1
  with KL `1.89e-7`, and reproduced the prior completion byte-for-byte. The
  bounded quality path uses its measured `1.625 GiB` transient allowance:
  remove400 passes preflight exactly at `iogpu.wired_limit_mb=58368`; larger
  preferred candidates do not. Do not bypass the extended-run guard unattended.
- `nemotron/tools/nemotron_mlx_verify_bench.py`: full-candidate 2/4/8-token
  target verification benchmark with full-logit sequential parity and exact
  rollback checks. `--compile-mamba` additionally compares compiled execution
  directly against eager full logits and recurrent state. Current measured
  target-pass speedups are 1.65x, 2.22x, and 2.58x respectively; block 16
  reaches 3.84x. These are not end-to-end speculative-generation claims.
- `nemotron/tools/nemotron_mlx_runtime_profile.py`: bounded resident target
  profiler. It restores identical state between samples and reports
  uninstrumented block time alongside synchronized per-layer Mamba, MoE,
  attention, final-norm, and vocabulary-head costs. Synchronization inflation
  is explicit. `--trace-repeats` provides sleep-delimited unsynchronized
  windows for Xcode Metal System Trace. Use it to choose hot paths, not as an
  end-to-end throughput claim.
- `nemotron/tools/nemotron_mlx_topk_sweep.py`: low-memory, provenance-bound
  routed-expert count screen. It changes only the runtime selection count and
  never rewrites candidate weights. Uniform top-20 preserved 16/16 broad final
  top tokens but improved block-two verification by only 3.58%; top-18 already
  flipped one broad winner. Keep production at native top-22. The retained
  `--expert-top-k` profiler override is diagnostic, not a generation policy.
- `nemotron/tools/nemotron_mlx_mtp.py`: official one-depth Nemotron MTP
  composition, packed-sidecar runtime, and caller-owned attention cache.
  Megatron's cacheless `forward_single_position` remains a control, but
  NVIDIA's deployed vLLM autoregressive path advances private MTP KV and
  teacher-prefills accepted target transitions. Production prompt mode must
  prefill prompt transitions, checkpoint before recursion, discard rejected
  speculative K/V, and replay only accepted authoritative target transitions.
  Cacheless, generated-only, and prompt-prefilled results must remain distinct.
- `nemotron/tools/nemotron_mlx_mtp_bench.py`: offline target-trace capture,
  source BF16 acceptance measurement, score-mass expert planning, and compact
  sidecar evaluation. Target and full BF16 MTP run in separate processes.
- `nemotron/tools/nemotron_mlx_mtp_teacher_capture.py`: resumable resident
  teacher capture for learned drafts. MTP supplies proposals but only exact
  target-verified trajectory prefixes are written. It commits one BF16 shard
  per prompt, binds every runtime input in `state.json`, and supports a
  no-model `--validate-only` audit. Rejected verifier suffixes are
  counterfactual and must never be mixed into ordinary supervised rows.
- `nemotron/tools/nemotron_mtp_teacher_prompts.py`: deterministic balanced
  prompt selection for learned-draft captures. The accepted 80-prompt corpus
  spans ten oMLX source categories and is hash-bound into capture state.
- `nemotron/tools/nemotron_mlx_mtp_recursive_features.py`: resumable official
  MTP hidden/proposal materializer over exact teacher rows. It enables
  official-first continuation training without loading the resident target.
- `nemotron/tools/nemotron_mlx_mtp_distill_features.py`: exact recursive MTP
  teacher materializer. It stores float32 hidden states plus reduced-head top-k
  logits for every depth in atomic, resumable, provenance-bound shards. The
  complete 51,200-row artifact is
  `mtp-distill-features-e128-balanced50k-d3-top32`.
- `nemotron/tools/nemotron_mlx_mtp_distill.py`: Metal trainer for residual-hidden
  and fused-token official-first students. Hard labels must be authoritative
  target tokens; rejected official MTP proposals are soft evidence only. The
  best fused student reached 40.612 tok/s with exact verification after its
  weights were stored for specialized contiguous BF16 matvec, but the matched
  official path reached 44.236 tok/s, so it remains diagnostic.
- `nemotron/tools/nemotron_mlx_mtp_norm_calibrate.py`: zero-runtime-cost
  diagnostic fitter for the official MTP final BF16 RMSNorm. It improved broad
  held-out recursive acceptance, but both full and damped forms failed to beat
  the matched 44.236 tok/s resident control. Overrides remain explicit,
  hash-bound diagnostics and must not become the production default.
- `nemotron/tools/nemotron_mlx_mtp_chain_bench.py`: recursive physical MTP
  acceptance gate. It accepts either legacy target traces or the exact v2
  teacher capture, supports prompt-disjoint slices, and strictly binds optional
  final-norm calibration artifacts. Cache modes `none`, `generated`, and
  `prompt` are distinct report identities; unscored prompt transitions are
  warmed only in `prompt` mode.
- `nemotron/tools/nemotron_mlx_mtp_predictor.py`: compact Metal-trained direct
  or official-first learned predictor. Its optional runtime loader verifies the
  exact target, vocabulary map, and artifact. The fused-token layout uses the
  runtime's specialized BF16 Metal matvec and reduced median draft latency from
  7.707 ms to 2.137 ms. Learned variants remain rejected for promotion because
  the best end-to-end result still trails official recursive MTP.
- `nemotron/tools/nemotron_mlx_gefen.py`: diagnostic MLX port of Gefen optimizer
  state at revision `704034f0d62871cc651a5ebae7b5547c55e0fc37`. It delivered
  7.96x less optimizer state but was about 2x slower unfused. The Apple port's
  weighted-Lloyd codebook is an explicit approximation to upstream exact DP;
  use AdamW for the current predictor and do not claim optimizer equivalence.
- `nemotron/tools/nemotron_mlx_mtp_pack.py` and
  `nemotron_mlx_mtp_quantize.py`: exact BF16 expert-subset materialization and
  explicitly separate MTP-only Q4 experiments. The quantizer and loader support
  exact per-tensor BF16 retention with strict config/payload agreement. The
  target checkpoint remains byte-identical; draft quantization is accepted only
  through measured acceptance and exact target verification. An e256 screen of
  every fixed projection found ranking-trace gains that failed an independent
  trace, so the production sidecar remains uniformly NVFP4.
- `nemotron/tools/nemotron_mlx_mtp_head_quantize.py`: revision-bound optional
  draft-only vocabulary-head quantization. The artifact is never a silent
  replacement for the authoritative BF16 target head. Runtime loading verifies
  its hash, payload, format, and shape before use.
- `nemotron/tools/nemotron_mlx_mtp_vocab_head.py`: deterministic, corpus-ranked
  reduced MTP vocabulary builder. The accepted 32K artifact stores only a
  128 KiB target-token map; `bf16_gather_matvec` projects those exact rows from
  the already-resident target head without copying weights. A nested 64K map
  improved offline draft recall but direct and low-margin adaptive resident
  paths both regressed throughput without increasing accepted drafts; retain
  it as diagnostic evidence and keep 32K as the production projection.
- `nemotron/tools/nemotron_mlx_mtp_chain_bench.py`: provenance-bound recursive
  MTP acceptance benchmark over contiguous authoritative target traces. It can
  run the full BF16 source or a fixed expert subset and records per-depth route
  mass for `nemotron_mlx_mtp_depth_plan.py`. Equal-depth and recursive-weighted
  cacheless e256 plans improved BF16 chain matches slightly, but the best
  conservative plan lost one match after NVFP4 quantization. A later disjoint
  cache-aware plan reduced depth-one and independent coding acceptance despite
  a small aggregate held-out gain. No replacement sidecar remains.
- `nemotron/tools/nemotron_mlx_mtp_hidden_adapter.py` and
  `nemotron_mlx_mtp_hidden_adapter_eval.py`: bounded rank-16 recursive-hidden
  correction trainer and independent trace gate. The correction improved local
  depth-two acceptance but was neutral in a matched resident run, so its runtime
  hook was removed. Keep these as diagnostics and do not load the artifact in
  production without materially broader evidence and an end-to-end win.
- `nemotron/tools/nemotron_mlx_mtp_blend_plan.py`: fixed-budget MTP expert-plan
  adaptation from normalized source and candidate score mass. The remove400
  8/16-swap blends at adaptation weights 0.5, 0.75, and 0.9 failed to dominate
  both source and candidate traces, so no blended sidecar was materialized.
- `nemotron/tools/nemotron_ngram_lookup.py`: bounded prompt/generated-token
  lookup drafts. The promoted opt-in policy uses 3-8-token keys, four-token
  proposals, two matching prior continuations, and first-token MTP agreement.
- `nemotron/tools/nemotron_paged_embeddings.py`: exact mmap-backed BF16 input
  rows. The performance default verifies the revision-bound 1 GiB payload hash,
  uses a 256-row MLX cache, and recovers exactly 1 GiB of active Metal memory.
- `nemotron/tools/nemotron_mlx_head_certificate.py`: provenance-bound
  NVFP4-head candidate recall and conservative groupwise exact-winner bounds.
  The route is rejected: useful candidate sizes cannot certify most tokens.
- `nemotron/tools/nemotron_mlx_speculative.py`: exact adaptive one- to
  three-draft resident generator. Guard400's performance default combines
  `mtp-sidecar-e128-nvfp4` with the candidate-bound
  `mtp-vocab-map-bf16-e32768-mbpp-success-guard400`; a 128-token coding control
  measured 34.932 tok/s, 79.03% draft acceptance, 1.408x speedup, 55.490 GiB
  peak, and exact integrity. The legacy r20 map is
  `mtp-vocab-map-bf16-e32768`. Omit `--mtp-lm-head` to retain the full-head
  acceptance fallback. Older candidates retain their measured policies. The
  remove400 performance default uses prompt-prefilled MTP KV and confidence-
  gated depth three with its candidate-specific
  256-expert NVFP4 sidecar, fused short-sequence SSM recurrence, fused draft
  reductions, batched verifier winners, all-layout compiled MoE tails, and
  candidate-bound 32K map. A 512-token gate reached `50.221 tok/s` (`1.943x`)
  with exact output at `53.705 GiB` peak. Its matched 256-token cacheless and
  prompt-cached depth-three controls measured `45.980/49.379 tok/s`. Pure
  Mamba graph reuse is the CLI default; an earlier paired
  256-token run reached `46.123 tok/s` versus `45.167 tok/s` eager with exact
  output and `53.689 GiB` peak. Use `--no-compile-mamba` only for controls. The
  128-expert candidate-specific sidecar remains the
  0.370 GiB smaller fallback and reaches `43.888 tok/s`; the generic sidecar
  control reaches `40.535 tok/s`. Use
  `iogpu.wired_limit_mb=60672` for guard400 or `58368` for remove400,
  `--margin-gib 0.5`, `--cache-limit-mib 256`, `--capture-rollback`,
  `--paged-embeddings`, `--embedding-cache-rows 256`, `--mtp-cache-mode prompt`,
  and three drafts with third attempt/output thresholds `2.0/1.0`. The e256
  path is for bounded generation and does not pass the conservative unattended
  extended-run preflight. Confidence-gated rollback
  capture was also rejected: conservative margins recovered at most 0.23% on
  matched controls, below run variance, while narrower margins caused expensive
  replay independently. Keep full accepted-state capture. The optional
  lower-memory fallback combines
  `mtp-sidecar-e64-nvfp4` with `mtp-lm-head-nvfp4`; it remains exact because the
  BF16 target verifies every draft, but is slower than the default. Unquantized
  32/48/64/96-expert sidecars either page badly or fail combined verification
  memory and are not production choices.
  Consensus-gated lookup drafting is a separate opt-in for repetitive code; it
  measured a 16.2% paired gain with exact output but remains disabled by
  default on unstructured workloads. The old 45.300/45.317 tok/s depth-three
  rejection applied to cacheless MTP and is superseded by the persistent-cache
  result. `--cycle-trace` atomically records policy and target outcomes after
  token-identity validation.
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

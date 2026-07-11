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
directory. `source-notes/revisions.json` pins the ModelOpt, vLLM, MLX-LM, and
oMLX commits used to establish NVFP4 decode, model, calibration, MTP, and
runtime semantics. It also pins LiveCodeBench, NeMo Skills, and NeMo Evaluator
revisions used for the dated coding protocol. The small LiveCodeBench checkout
lives at `source-notes/livecodebench`. These repositories are reference code
only; copy and adapt required model-specific logic into `nemotron_*` files.

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
  the independent numerical reference.
- `nemotron/tools/nemotron_mlx_mamba.py`: one-token Nemotron Mamba2 composition
  over official BF16 convolution/state tensors and exact ModelOpt FP8
  projections, with persistent MLX `ArraysCache` recurrence.
- `nemotron/tools/nemotron_mlx_moe_layer.py`: complete one-token LatentMoE
  layer with RMSNorm, BF16 routing, latent projections, top-22 packed experts,
  shared expert, and residual kept in one lazy MLX graph.
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
- `nemotron/tools/nemotron_mlx_calibrate.py`: resumable diverse-corpus router
  observer. It aggregates counts, score mass, selected latent output norms,
  route-weighted output contribution, and maxima per expert.
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
  64 GB candidate. A separate 20-task HumanEval gate also tied r25 and r20 at
  17/20 with identical pass/fail outcomes. The complete corrected HumanEval
  run scored 154/164 (93.90%). It runs at `23.610 tok/s` ordinary
  and `34.195 tok/s` with the candidate-bound MTP path, peaking at 54.333 GiB.
- `nemotron/tools/nemotron_mlx_layer_distill.py`: bounded teacher/candidate
  layer-output fitter. Per-channel affine correction overfit and scalar affine
  correction improved held-out r35 local output error by only 1.87%; neither is
  a runtime feature. Rank-4 residual regression also failed held-out validation.
  Future work must train replacement expert or router behavior rather than
  silently attaching any of these diagnostic corrections.
  A ReLU-squared latent adapter before `fc2_latent` also failed with 189
  training tokens; do not revisit small correction sidecars without materially
  new evidence or a larger expert-parameter training design.
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
  Launchers preserve the live preflight kernel cap after a run; they must never
  restore MLX's lower default over a user-approved `iogpu.wired_limit_mb`.
- `nemotron/tools/nemotron_mlx_verify_bench.py`: full-candidate 2/4/8-token
  target verification benchmark with full-logit sequential parity and exact
  rollback checks. Current measured target-pass speedups are 1.65x, 2.22x, and
  2.58x respectively; block 16 reaches 3.84x. These are not end-to-end
  speculative-generation claims.
- `nemotron/tools/nemotron_mlx_mtp.py`: official one-depth Nemotron MTP
  composition and packed-sidecar runtime. Megatron's speculative path is
  stateless: `forward_single_position` receives the target's final normalized
  hidden state and accepted-token embedding and does not maintain an MTP KV
  cache or prefill the MTP head. Do not add a persistent MTP cache.
- `nemotron/tools/nemotron_mlx_mtp_bench.py`: offline target-trace capture,
  source BF16 acceptance measurement, score-mass expert planning, and compact
  sidecar evaluation. Target and full BF16 MTP run in separate processes.
- `nemotron/tools/nemotron_mlx_mtp_pack.py` and
  `nemotron_mlx_mtp_quantize.py`: exact BF16 expert-subset materialization and
  explicitly separate MTP-only Q4 experiments. The target checkpoint remains
  byte-identical; draft quantization is accepted only through measured
  acceptance and exact target verification.
- `nemotron/tools/nemotron_mlx_mtp_head_quantize.py`: revision-bound optional
  draft-only vocabulary-head quantization. The artifact is never a silent
  replacement for the authoritative BF16 target head. Runtime loading verifies
  its hash, payload, format, and shape before use.
- `nemotron/tools/nemotron_mlx_mtp_vocab_head.py`: deterministic, corpus-ranked
  reduced MTP vocabulary builder. The accepted 32K artifact stores only a
  128 KiB target-token map; `bf16_gather_matvec` projects those exact rows from
  the already-resident target head without copying weights.
- `nemotron/tools/nemotron_mlx_mtp_chain_bench.py`: provenance-bound recursive
  MTP acceptance benchmark over contiguous authoritative target traces.
- `nemotron/tools/nemotron_ngram_lookup.py`: bounded prompt/generated-token
  lookup drafts. The promoted opt-in policy uses 3-8-token keys, four-token
  proposals, two matching prior continuations, and first-token MTP agreement.
- `nemotron/tools/nemotron_paged_embeddings.py`: exact mmap-backed BF16 input
  rows. The performance default verifies the revision-bound 1 GiB payload hash,
  uses a 256-row MLX cache, and recovers exactly 1 GiB of active Metal memory.
- `nemotron/tools/nemotron_mlx_head_certificate.py`: provenance-bound
  NVFP4-head candidate recall and conservative groupwise exact-winner bounds.
  The route is rejected: useful candidate sizes cannot certify most tokens.
- `nemotron/tools/nemotron_mlx_speculative.py`: exact adaptive one- or
  two-draft resident generator. The performance default combines
  `mtp-sidecar-e128-nvfp4` with
  `mtp-vocab-map-bf16-e32768`. Repeated same-prompt controls averaged
  `34.75 tok/s` with paged embeddings at approximately `57.34 GiB` peak and
  exact output. Omit `--mtp-lm-head` to retain the full-head acceptance
  fallback. Adaptive depth two is exact and opt-in, but its repeated 2-3% gain
  missed the 5% promotion gate, so depth one remains the default. Use
  `iogpu.wired_limit_mb=60672`, `--margin-gib 0.5`, `--capture-rollback`,
  `--paged-embeddings`, and `--embedding-cache-rows 256`. The optional
  lower-memory fallback combines
  `mtp-sidecar-e64-nvfp4` with `mtp-lm-head-nvfp4`; it remains exact because the
  BF16 target verifies every draft, but is slower than the default. Unquantized
  32/48/64/96-expert sidecars either page badly or fail combined verification
  memory and are not production choices.
  Consensus-gated lookup drafting is a separate opt-in for repetitive code; it
  measured a 16.2% paired gain with exact output but remains disabled by
  default on unstructured workloads.
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

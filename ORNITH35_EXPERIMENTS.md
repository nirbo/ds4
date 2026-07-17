# Ornith 35B Experiment Ledger

An item is checked only after implementation, correctness validation, quality
measurement, and relevant memory/performance measurement. Every completed item
must record `SUCCESS`, `PARTIAL`, or `REJECTED` with evidence.

## Bootstrap

- [x] Pin and validate target, DSpark, and MTP metadata without weight payloads.
  `SUCCESS` (2026-07-16): pinned revisions and LFS identities; bounded range
  reads cataloged the 93,346-tensor target, 44-tensor DSpark draft, and 785 MTP
  tensors without downloading weight payloads. Ten corruption/shape/index
  tests pass, as does strict validation of all 92,520 target NVFP4 companions.
- [x] Download and hash the immutable AEON NVFP4 source after explicit approval.
  `SUCCESS` (2026-07-16): the single 23,741,821,016-byte file resumed without
  duplication and passed exact header, payload, and SHA-256 verification. The
  accepted digest is `68a4b2b8605076825302be20132cf69342b44a0385c19e6de741af5ec3114ca0`;
  revision-bound state is external `source-nvfp4-state.json`.
- [ ] Prove complete text-only tensor coverage and exact vision exclusion.
- [ ] Establish authoritative source logits and coding-quality controls.

## Target Runtime

- [x] Establish the dependency-free NVFP4 CPU oracle against synthetic vectors
  and real retained expert samples.
  `SUCCESS` (2026-07-16): corrected compressed-tensors semantics divide the
  FP8 block scale by stored `weight_global_scale`; a pinned upstream converter,
  dequantizer, and unit test independently establish the convention. Synthetic
  nibble/FP8/packing tests and four real projection samples pass.
- [x] Decode ModelOpt NVFP4 experts accurately in MLX on Apple Silicon.
  `SUCCESS` (2026-07-16): real routed/shared gate, up, and down projections from
  layers 0, 19, and 39 match the CPU oracle at `1.09e-7` to `1.77e-7` relative
  L2 and no more than `2.39e-7` maximum absolute error. Isolated Metal matvecs
  measured 29.6-37.0 GB/s; selected-expert synthetic parity remains passing.
- [ ] Compose one complete GatedDeltaNet layer against an independent reference.
  Synthetic mechanism complete: the MLX one-token path matches an independent
  scalar oracle across three sequential state transitions and preserves exact
  rollback snapshots. Real BF16 checkpoint comparison remains required before
  this item can be accepted.
- [ ] Compose one complete full-attention layer against an independent reference.
  Synthetic mechanism complete: three sequential positions match an
  independent scalar oracle for output and K/V state, and the native 262K
  boundary proves RoPE trigonometry stays FP32 before BF16 casting. Real BF16
  checkpoint comparison remains required before this item can be accepted.
- [ ] Compose one complete MoE layer with exact top-8 routing and shared expert.
  Synthetic mechanism complete: packed scalar parity covers sorted routing,
  renormalized top-k weights, selected gate/up/down projections, and the gated
  shared expert. Router IDs remain MLX arrays into Metal. Real-layer and
  full-logit comparison remain required before this item can be accepted.
- [ ] Run the complete 40-layer text target with full-vocabulary logits.
  Real mechanism smoke complete: all 40 layers load from the verified source at
  21.267 GiB active/21.638 GiB peak, and two sequential tokens produce finite
  full logits, normalized routes, and valid aggregate state. After two warmup
  transitions, a ten-token run measured eight full-logit tokens at 23.053 ms
  mean (43.378 tok/s). Chat-formatted generation then returned exact `OK` and,
  with thinking plus recommended seeded sampling, emitted a correct prime
  function at 41.995 tok/s after 902 tokens. These remain bounded coherence
  smokes; independent source logits and substantial coding evaluation are
  required before acceptance.
- [ ] Materialize or directly load the text-only resident runtime.

## Decode Hotpath

- [x] Establish a durable full-model target profiler before changing kernels.
  `SUCCESS` (2026-07-16): the profiler separates graph construction from Metal
  execution, reports forced-boundary component medians and synchronization
  inflation, verifies optimized logits against the retained fallback, and can
  collect an explicitly requested Metal trace. The initial target measured
  22.768 ms mean (43.921 tok/s), with 3.184 ms graph construction and 19.541 ms
  execution. Native `.gputrace` capture duplicated about 23 GiB of resident
  resources; the trace was inspected and removed immediately.
- [x] Compile complete GatedDeltaNet/MoE graphs to remove dispatch overhead.
  `REJECTED` (2026-07-16): a compiled real layer improved its isolated time
  from 0.747 to 0.613 ms and the complete graph reached 52.157 tok/s, but target
  logits drifted by 0.1875 maximum absolute and 1.134% relative L2. Compiling
  only the GatedDeltaNet mixer reached 46.522 tok/s but increased drift to
  0.5625 maximum absolute and 3.001% relative L2. Both paths were removed.
- [x] Concatenate GatedDeltaNet input projections into one exact matvec.
  `REJECTED` (2026-07-16): the isolated operation improved from 0.281 to
  0.255 ms with exact output, but a 60-sample full-model comparison improved
  only 0.31% (44.078 to 44.213 tok/s). The extra resident representation and
  complexity did not justify the end-to-end return, so it was removed.
- [x] Pair routed and shared NVFP4 gate/up projections in custom Metal kernels.
  `SUCCESS` (2026-07-16): each pair reuses the input vector in one dispatch
  while preserving each projection's FP32 accumulation order. Unit, MoE, and
  real-model comparisons are bit-exact. In the final alternating 80-sample
  comparison this raised the original path from 44.026 to 44.662 tok/s (1.45%).
- [x] Fuse selected NVFP4 down projections with the ordered routing reduction.
  `SUCCESS` (2026-07-16): one threadgroup evaluates all eight selected experts
  for each output row, performs the original per-expert BF16 rounding, applies
  BF16 routing weights, and sums slots in MLX's original order. This removes
  the `[8, 2048]` intermediate and adds 3.03% over paired gate/up, reaching
  46.014 tok/s and 4.52% over the retained fallback. A 147-transition real
  trajectory preserved every full-vocabulary logit, route, recurrent state,
  and K/V value bit-for-bit at 21.638 GiB peak. The seeded 903-token thinking
  coding smoke remained token-identical and improved from 41.995 to 43.560
  tok/s (3.73%).
- [x] Replace the MLX router sort and normalization with a custom Metal top-8.
  `REJECTED` (2026-07-16): a stable 256-way bitonic kernel preserved selected
  experts and BF16 routing exactly and reduced the isolated selection block
  from 163.6 to 108.4 us. Full-model alternating timing improved only 0.37%
  (45.585 to 45.755 tok/s), which did not justify retaining a custom sorter.
- [x] Select experts in the logit domain and softmax only retained scores.
  `REJECTED` (2026-07-16): the checkpoint's exact RMSNorm/router norm bound
  limits every possible router-logit spread to 89.775, so FP32 probabilities
  cannot underflow and ordering remains monotonic. Six balanced 50-round blocks
  nevertheless showed only a 0.42% trimmed gain (45.588 to 45.780 tok/s).
  Because finite-precision normalization is not universally bit-equivalent,
  the small return did not justify changing semantics; the path was removed.
- [x] Fuse the production GatedDeltaNet recurrent update and core reductions.
  `SUCCESS` (2026-07-16): the Metal kernel preserves MLX's FP32 multiply/add
  boundaries and exact reduction order while eliminating decayed-state,
  memory, delta, and core-reduction intermediates. It reduced the isolated
  recurrence from 229.0 to 151.6 us and a real GDN layer from 486.3 to 468.7 us.
  A 200-sample full-model A/B improved 45.773 to 47.220 tok/s (3.16%), and a
  275-transition trajectory preserved every logit, route, recurrent state, and
  K/V value bit-for-bit at 21.638 GiB peak. The token-identical 903-token
  thinking smoke improved from 43.560 to 44.097 tok/s, 5.01% over the original
  41.995 tok/s target path. The full Ornith-35 suite passes.
- [x] Fuse the production GatedDeltaNet convolution shift and depthwise dot.
  `SUCCESS` (2026-07-16): one Metal dispatch writes the next four-slot BF16
  convolution state and evaluates the exact ordered FP32 dot. All six balanced
  50-round blocks improved; the 300-sample 5%-trimmed A/B moved 47.051 to
  47.289 tok/s (0.51%). A 279-transition trajectory preserved every logit,
  route, convolution/recurrent state, and K/V value bit-for-bit. The unchanged
  seeded 903-token thinking completion reached 44.987 tok/s in a fresh run,
  while peak memory remained 21.638 GiB.
- [x] Fuse production residuals with the following RMSNorm mean-square.
  `SUCCESS` (2026-07-16): the Metal kernel reproduces pinned MLX 0.32.0's
  512-thread, four-values-per-thread FP32 reduction order, emits the exact
  BF16-rounded residual, and carries its exact mean-square into centered
  RMSNorm. The isolated boundary improved 176.5 to 144.5 us. All six balanced
  50-round full-model blocks improved; the 300-sample 5%-trimmed result moved
  47.704 to 48.959 tok/s (2.63%). A 279-transition trajectory preserved every
  logit, route, convolution/recurrent state, and K/V value bit-for-bit. The
  unchanged 903-token thinking completion reached 46.298 tok/s at the same
  21.638 GiB peak.

## Context And Cache

- [ ] Validate native 262,144-token RoPE and cache semantics.
- [ ] Implement YaRN factor-2 loading for 524,288 tokens.
- [ ] Prove native and YaRN cache profiles cannot be mixed.
- [ ] Implement exact in-memory prefix reuse for K/V and GatedDeltaNet state.
- [ ] Implement atomic persistent prompt-cache checkpoints.
- [ ] Prove save/restore and incremental append logit parity.
- [ ] Add content-addressed workspace cache lookup and bounded disk LRU.
- [ ] Add background cache warming without blocking foreground decode.
- [ ] Evaluate eight-bit K/V against BF16 long-context quality and speed.

## Prefill Performance

- [ ] Establish 2K, 32K, 128K, 262K, and bounded 524K TTFT baselines.
- [ ] Use native Steel flash attention with Ornith-specific GQA tuning.
- [ ] Fuse RMSNorm, QKV, RoPE, and cache writes where numerically safe.
- [ ] Implement chunk-parallel GatedDeltaNet prefill on Metal.
- [ ] Group routed tokens into batched NVFP4 expert GEMMs.
- [ ] Tune chunk scheduling for throughput, scratch memory, and watchdog safety.
- [ ] Measure cold prefill, restored-prefix, and incremental-suffix paths separately.

## Speculative Decode

- [ ] Measure the public matched DSpark draft against the authoritative target.
- [ ] Extract and validate the official Qwen3.5 MTP bootstrap tensors.
- [ ] Distill an Ornith-targeted MTP sidecar if bootstrap acceptance is inadequate.
- [ ] Train or extend an Ornith-targeted DSpark draft if needed.
- [ ] Implement exact block verification with GDN/KV snapshot and rollback.
- [ ] Tune adaptive MTP-versus-DSpark scheduling by context and acceptance.
- [ ] Measure exact generation speed at 2K, 128K, 262K, and 524K context.

## Optional Semantic Changes

- [ ] Evaluate repository-aware AST/symbol retrieval before model changes.
- [ ] Evaluate CacheBlend-style non-prefix reuse under matched coding gates.
- [ ] Evaluate a target-trained sparse-attention indexer for cold 524K prefill.
- [ ] Consider further quantization only after the source runtime is accepted.
- [ ] Consider expert pruning only after quant-only quality is established.

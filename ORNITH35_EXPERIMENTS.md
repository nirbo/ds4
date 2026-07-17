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
- [x] Emit the following centered RMSNorm from the fused residual dispatch.
  `SUCCESS` (2026-07-16): the kernel preserves MLX's precise reciprocal square
  root and every FP32/BF16 operation boundary, then hands the normalized result
  directly to the next layer or final LM head. This removes 80 normalization
  graphs while retaining the mean-square-only and fully materialized fallbacks.
  All six balanced 50-round blocks improved; the 300-sample 5%-trimmed A/B
  moved 48.618 to 50.782 tok/s (4.45%), 15.35% above the original 44.026 tok/s
  graph. A 279-transition trajectory preserved every logit, route,
  convolution/recurrent state, and K/V value bit-for-bit. The unchanged
  903-token thinking completion reached 47.847 tok/s, 13.94% above its original
  41.995 tok/s run, with peak memory unchanged at 21.638 GiB.
- [x] Keep the GatedDeltaNet recurrence core resident through norm and z gate.
  `SUCCESS` (2026-07-16): the exact standalone core-gate kernel reduced its
  isolated stage from 207.6 to 141.7 us but added only 0.58% end-to-end, so the
  standalone kernel was not retained. Folding it into the exact recurrence
  dispatch keeps 128 core values per head in threadgroup memory, avoids the
  global FP32 core tensor, and improved the already optimized graph from 50.325
  to 52.134 tok/s (3.60%) across six balanced 50-round blocks. This is 18.42%
  above the original 44.026 tok/s graph. A 279-transition trajectory preserved
  every logit, route, convolution/recurrent state, and K/V value bit-for-bit.
  The unchanged 903-token thinking completion reached 49.430 tok/s, 17.70%
  above its original 41.995 tok/s run, at the same 21.638 GiB peak.
- [x] Fuse the one-token routed/shared down projections and gated MoE merge.
  `SUCCESS` (2026-07-17): a singleton use of every batched MoE kernel was
  rejected because it slowed a real layer by 2.8%. A targeted row-layout sweep
  instead found four routed-down output rows per SIMD group. The accepted
  nine-SIMD kernel evaluates the top-8 routed experts and shared expert for
  four rows, retains each projection's lane reduction and BF16 boundaries,
  performs the ordered routed sum, and applies the BF16 shared gate and final
  add before writing output. The isolated down/merge stage improved from
  159.36 to 142.19 us (1.12x). A balanced 250-sample full-model A/B improved
  54.582 to 56.826 tok/s (4.11%) at the unchanged 21.638 GiB peak. All 162
  tensors matched bit-for-bit in the one-token comparison, and a separate
  64-transition greedy trajectory matched every logit, hidden value, route,
  convolution/recurrent value, K/V value, and chosen token.
- [x] Validate immutable decode inputs once per advancing session.
  `SUCCESS` (2026-07-17): `TextDecodeSession` deeply validates every mixer,
  MoE, norm, cache shape, dtype, layer type, and attention position before
  issuing a sealed session. Nested layer, GatedDeltaNet, attention, and MoE
  calls then skip only those already-proven invariant checks; token bounds and
  live operation contracts remain checked. Sessions are immutable and each
  transition returns a new state-bound session, preserving rollback. A
  balanced 250-sample full-model A/B improved 56.271 to 57.516 tok/s (2.21%)
  and reduced the durable profiler's host graph construction to 2.241 ms. All
  162 one-token tensors and every tensor and chosen token in a separate
  64-transition advancing comparison remained bit-identical. Deliberately
  malformed cache position and deep GatedDeltaNet weight shape are rejected at
  session creation. The full suite passes, and the generation CLI now uses the
  session path after checked prefill.
- [x] Fuse selected and shared gate/up projection with exact BF16 SiLU.
  `SUCCESS` (2026-07-17): one eight-SIMD Metal kernel evaluates all top-8
  routed and shared gate/up rows, reuses the hidden vector, and writes only the
  activated BF16 intermediates consumed by the fused down path. An exhaustive
  check found that fast Metal `exp` differs from MLX 0.32's precise BF16
  sigmoid at exactly one finite BF16 input, `-6.84375`; a real layer reached
  that value, so the accepted kernel uses `metal::precise::exp` and retains a
  targeted regression. The isolated gate/up/SiLU stage improved from 167.61 to
  151.41 us and real layer 19 from 0.3001 to 0.2824 ms. All six balanced
  50-round blocks favored fusion; the 5%-trimmed full-model result improved
  57.176 to 60.424 tok/s (5.68%). A 128-transition greedy trajectory matched
  every full-vocabulary logit, hidden value, route, convolution/recurrent
  value, K/V value, and chosen token bit-for-bit. Peak memory remains
  21.638 GiB and the full suite passes.
- [x] Carry the decode QKV projection through convolution and SiLU.
  `SUCCESS` (2026-07-17): the custom BF16 GEMV retains MLX 0.32's four
  contiguous columns per lane and ordered shuffle reduction, rounds each QKV
  row at the original boundary, then shifts convolution state, evaluates the
  exact four-term FP32 depthwise dot, and applies precise FP32 SiLU in the same
  dispatch. A geometry sweep kept one row per SIMD group and selected eight
  groups by complete-model timing. The real layer-0 stage improved from
  354.13 to 311.92 us (1.135x). In the final balanced 240-sample comparison,
  every block improved and the 5%-trimmed full-model result moved from 61.101
  to 61.986 tok/s (1.45%). A separate 128-transition greedy trajectory matched
  all 162 full-vocabulary logit, hidden, route, convolution/recurrent, and K/V
  tensors plus every chosen token bit-for-bit. Peak memory remains 21.638 GiB
  and the full suite passes. Prefill retains its faster batched QKV path.
- [x] Combine the BF16 router and shared-expert gate projection.
  `SUCCESS` (2026-07-17): the loader joins the 256 router rows and one shared
  gate row into one authoritative allocation; the public tensors are views, so
  steady-state weights gain only one row and the split fallback uses the same
  bytes. Native MLX GEMV returned bit-identical router and gate values for 100
  random real-layer inputs. All six balanced 40-round decode blocks improved;
  the 5%-trimmed result moved from 61.887 to 62.061 tok/s (0.28%). A 128-step
  advancing greedy trajectory matched all 162 logits, hidden, route,
  convolution/recurrent, and K/V tensors plus every chosen token bit-for-bit.
  Exact 128-token prefill also matched all 162 tensors and remained effectively
  neutral at 386.109 versus 386.287 tok/s. Active memory was 21.511 GiB and the
  measured peak was 21.640 GiB.
- [x] Fuse attention Q/K RMSNorm, gate split, and partial RoPE for decode.
  `SUCCESS` (2026-07-17): one 32-thread Metal group per Q/K head reproduces
  MLX 0.32's exact 256-wide row reduction, precise reciprocal square root,
  centered norm weight, BF16 rounding, query-gate split, and partial text RoPE.
  A first 64-thread prototype passed a one-step check but drifted by one BF16
  query value at layer 15 on token 73; that topology was rejected and its
  deterministic failing seed is retained as a regression. The corrected path
  passed 300 varied random trials and reduced a real attention layer from a
  443.65 us median to 345.09 us 5%-trimmed. All six balanced 40-round
  full-model blocks improved; 5%-trimmed decode moved from 61.703 to 64.048
  tok/s (3.80%). A separate 128-step greedy trajectory matched all 162 tensors
  per transition, 20,736 comparisons total, plus every selected token
  bit-for-bit. Peak memory remained 21.640 GiB. Reusing one exact RoPE table
  across all ten attention layers also preserved all 162 tensors in a
  128-token prefill; its 412.888 to 413.278 tok/s change (0.09%) is effectively
  neutral but removes duplicate trigonometry graphs.
- [x] Pair the GatedDeltaNet `b` and `a` input projections.
  `REJECTED` (2026-07-17): one authoritative 64-row allocation and native MLX
  GEMV preserved both projections for 100 random real-layer inputs and all 162
  full-model tensors. The isolated projection group improved from 232.63 to
  143.55 us, but the complete real GDN transition had the same 423 us median.
  Across six balanced 40-round full-model blocks, the 5%-trimmed result moved
  backward from 64.161 to 64.119 tok/s (-0.07%). MLX already hides these tiny
  independent GEMVs inside the complete lazy graph, so the shared allocation,
  runtime flag, and tests were removed. Combining `z/b/a` was exact but slower
  in isolation, while folding them into the custom QKV kernel changed their
  BF16 reductions and was rejected before model integration.
- [x] Keep exact GQA decode and continuation prefill in the two-head cache.
  `SUCCESS` (2026-07-17): grouped batch dimensions map sixteen query heads to
  two K/V heads with eight query groups, eliminating `mx.repeat` without
  changing either BF16 matmul, scaling, FP32 softmax, BF16 probability, or
  value-reduction arithmetic. All 162 full-model tensors matched at prefixes
  0, 128, 256, 512, 1,024, 4,096, and 16,384. Balanced decode remained neutral
  at an empty cache, improved 52.020 to 60.017 tok/s at 4K (15.37%), and
  improved 33.072 to 50.155 tok/s at 16K (51.65%); every block improved. A
  controlled 16K run reduced transient peak by about 254 MiB. A separate
  128-step greedy trajectory matched 20,736 tensors and every selected token
  bit-for-bit. Exact non-Steel prefill retains repeated GQA below the measured
  1,280-token crossover, then groups the existing prefix. At a 4K prefix, an
  exact 128-token continuation improved 315.963 to 348.531 tok/s (10.31%) and
  preserved all 162 tensors; the empty-prefix path therefore keeps its faster
  original layout.

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

- [x] Project the full vocabulary only for the final prompt token.
  `SUCCESS` (2026-07-17): the hidden-transition API materializes the complete
  rollback state without executing the unused 248,320-way LM head or exporting
  diagnostic routes for intermediate prompt tokens. Eight real transitions
  preserved hidden values, routes, all GatedDeltaNet state, and all K/V values
  bit-for-bit. Six alternating 23-token warm A/B pairs improved token-serial
  prefill from 52.427 to 58.658 tok/s (11.88%); final logits and state remained
  bit-identical at the unchanged 21.638 GiB peak.
- [ ] Establish 2K, 32K, 128K, 262K, and bounded 524K TTFT baselines.
- [x] Use native Steel flash attention with Ornith-specific GQA tuning.
  `PARTIAL` (2026-07-17): MLX 0.32 Steel consumes Ornith's native 16-query/2-KV
  GQA tensors directly with lower-right causal masking for retained-prefix plus
  chunk semantics. `vmap` projections and vectorized FP32 RoPE preserve every
  BF16 K/V value bit-for-bit. Representative real layers 3, 19, and 39 show
  BF16-scale output drift; a 1,024-prefix/128-token continuation measured
  0.001953 maximum absolute and 2.83e-4 relative L2. Layer 39 at chunk 256
  improved 56.238 to 5.371 ms (10.47x, 47,665 token-layers/s). Full-model
  gating rejected broad activation: one-shot chunks 32-128 amplified the local
  difference to 4.48-5.48% final-logit relative L2 and changed routes. The
  implementation remains isolated behind `use_steel`; the exact frontend
  explicitly disables it pending selective-layer or precision recovery.
- [ ] Fuse RMSNorm, QKV, RoPE, and cache writes where numerically safe.
- [x] Implement chunk-parallel GatedDeltaNet prefill on Metal.
  `SUCCESS` (2026-07-17): token-batched MLX `vmap` projections retain the
  one-token BF16 accumulation contract, one Metal convolution dispatch walks
  each channel's exact four-slot history, and one head-parallel Metal kernel
  advances the FP32 recurrence through the chunk in token order. Isolated
  chunk kernels and real layers 0, 18, and 38 preserve every output,
  convolution value, and recurrent value bit-for-bit. On real layer 0, chunk
  128 improved 5,484 to 19,459 token-layers/s (3.55x); layer 38 at chunk 256
  improved 47.154 to 12.977 ms (3.63x, 19,728 token-layers/s).
- [x] Group routed tokens into batched NVFP4 expert GEMMs.
  `SUCCESS` (2026-07-17): four GPU-owned Metal primitives batch shared
  projections, selected gate/up projections, and ordered weighted-down
  reductions without expert-ID readback. Synthetic tests are bit-exact against
  every one-token kernel. Real layers 0, 19, and 39 preserve outputs, top-8
  expert IDs, and BF16 routing weights bit-for-bit across eight-token batches.
  On real layer 19, chunk 128 improved 6,642 to 16,626 token-layers/s (2.50x)
  and chunk 256 improved 6,757 to 16,966 token-layers/s (2.51x). Complete
  prefill integration remains gated on the sequence token mixers.
- [x] Compose bounded exact chunks through all 40 decoder layers.
  `SUCCESS` (2026-07-17): per-token `vmap` preserves BF16 dense, router, and
  shared-gate reductions; token-indexed Metal
  residual/RMSNorm threadgroups preserve every one-token operation boundary.
  A power-of-two scheduler uses 128/64/32/16/8 chunks and a serial tail while
  projecting logits only at the end. Real 25-, 128-, and 259-token runs kept
  final logits and every GDN/KV state bit-for-bit. Warm 128-token prefill
  improved 57.580 to 152.594 tok/s (2.65x); `(128,128,1,1,1)` improved a
  259-token prompt from 56.868 to 147.331 tok/s (2.59x). Peak stayed bounded at
  21.808 GiB in the controlled multi-chunk run.
- [x] Batch exact router softmax, top-8, and retained-route normalization.
  `SUCCESS` (2026-07-17): token-wise BF16 router GEMVs remain authoritative,
  while MLX evaluates all independent 256-way FP32 row softmaxes, sorts, and
  eight-way renormalizations in batched GPU dispatches. Synthetic 128-row
  routing, random real-weight inputs across all 40 MoE layers, and a complete
  128-token model transition were bit-identical to independent token routing.
  The full transition compared 162 tensors: final logits and hidden values,
  every route, and all recurrent/K/V state. Real layer 19 improved from 9.602
  to 7.745 ms (1.24x), and warm exact 128-token prefill improved from 152.327
  to 166.480 tok/s (9.29%) at a 21.732 GiB peak. This supersedes a custom
  sorter: MLX's native row dispatch is both exact and faster end to end.
- [x] Batch the exact full-attention prelude, cache append, and output projection.
  `SUCCESS` (2026-07-17): Q/K/V and output projections remain independent
  token-wise GEMVs under `vmap`; Q/K normalization and RoPE are batched; K/V is
  appended and head-repeated once. Only each causal token's score, FP32
  softmax, and value reduction remains token-authoritative. Real layers 3, 19,
  and 39 were bit-identical to serial decode at zero- and 1,024-token prefixes.
  Layer 39 chunk 128 improved from 27.080 to 4.905 ms (5.52x) at an empty
  prefix and from 38.724 to 5.131 ms (7.55x) after 1,024 tokens. A complete
  128-token model transition preserved all 162 compared logit, hidden, route,
  recurrent, and K/V tensors while improving from 166.339 to 240.973 tok/s
  (44.87%). The exact `(128,128,1,1,1)` path preserved all 82 final/state
  tensors and improved from 160.434 to 231.064 tok/s (44.02%) at a 21.749 GiB
  peak. Steel remains disabled; no numerical relaxation is involved.
- [x] Pack multiple exact NVFP4 output rows into each prefill threadgroup.
  `SUCCESS` (2026-07-17): selected/shared gate-up SIMD groups evaluate two
  rows while reusing token loads; shared down evaluates four; routed down uses
  four row groups per 1,024-thread group and evaluates four rows per SIMD
  group. Every row retains its original block/pair accumulation sequence,
  `simd_sum`, BF16 rounding point, and ordered top-8 reduction. Synthetic
  minimal-layout parity and a full 128-token model comparison were bit-exact
  across all 162 logit, hidden, route, recurrent, and K/V tensors. Real layer
  19 MoE improved from 7.777 to 4.479 ms (1.74x). Warm 128-token prefill
  improved from 240.298 to 314.459 tok/s (30.86%) at a 21.751 GiB peak; the
  exact 259-token schedule improved from 231.653 to 299.260 tok/s (29.18%) at
  21.748 GiB. Smaller valid shapes select an adaptive divisor.
- [x] Fuse batched selected/shared gate-up projection with BF16 SiLU.
  `REJECTED` (2026-07-17): the combined branch-uniform kernel preserved all
  162 full-model tensors but slowed every balanced block; the 5%-trimmed
  128-token result fell from 411.478 to 409.366 tok/s (0.51%). Two separate
  branch-free prototypes were also exact. Their best complete-model result was
  only 412.736 versus 412.015 tok/s (0.18%), while all nine real-layer sweeps
  over 8/16/32 SIMD groups and 1/2/4 rows were slower than the retained stage
  (2.793 ms retained versus 2.810 ms best prototype). Moving the precise
  sigmoid into a projection lane loses more parallelism than the removed
  intermediates and dispatches recover. All prototype code was removed.
- [x] Keep exact GatedDeltaNet recurrent columns GPU-local through each chunk.
  `SUCCESS` (2026-07-17): every SIMD lane caches its four decayed FP32 values,
  eliminating the duplicate state read and decay. Thirty-two SIMD groups
  preserve each value column's original lane reduction while exposing all 128
  columns concurrently. The accepted column-major chunk then carries each
  independent state column through all tokens in registers, writes final state
  once, and performs the unchanged per-token RMSNorm/gate in a second kernel.
  Synthetic minimal-layout checks and real layers 0, 18, and 38 match serial
  output, convolution, and recurrent state bit-for-bit. The isolated recurrence
  fell from 3.413 to 1.154 ms (2.96x), and real layer 38 fell from 6.548 to
  4.056 ms (1.61x). A full 128-token transition preserved all 162 compared
  tensors and improved from 314.459 to 384.555 tok/s (22.29%) at a 21.757 GiB
  peak. The 259-token schedule reached 362.168 tok/s. Applying the same cached
  32-group kernel to decode preserved all 162 tensors and improved 52.840 to
  53.877 tok/s (1.96%).
- [x] Tune chunk scheduling for throughput, scratch memory, and watchdog safety.
  `SUCCESS` (2026-07-17): a direct exact 256-token chunk and two consecutive
  exact 128-token chunks matched all 162 compared final, route, recurrent, and
  K/V tensors bit-for-bit. Two 128-token chunks reached 385.335 tok/s, while
  one 256-token chunk reached 375.361 tok/s, making the larger graph 2.59%
  slower; it also raised peak memory to 21.903 GiB. The production cap remains
  128, with power-of-two decomposition and a serial tail. Because 256 lost on
  both throughput and memory, 512 was not run or exposed. Existing CLI tests
  reject chunk sizes above 128, bounding scratch use and graph/watchdog risk.
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

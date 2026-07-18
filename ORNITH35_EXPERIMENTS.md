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

## Selective Compression

- [x] Keep exact BF16 embeddings outside wired MLX residency.
  `SUCCESS` (2026-07-17): the generator maps the sole verified safetensors
  source and copies only requested 4 KiB rows or bounded prompt batches. Three
  balanced 128-step full-model A/Bs preserved every logit, hidden value, route,
  and state bit-for-bit. MLX model activity fell from 21.268 to 20.320 GiB and
  profiler peak from 21.640 to 20.692 GiB. Decode cost 0.17%-0.74%; exact
  128-token prefill cost 0.18% at zero prefix and 0.08% at 4K. The source file
  and reclaimable OS pages remain; `--no-mapped-embedding` is the resident
  fallback. Combined with hybrid Q8/32 head projection, the measured path used
  19.967 GiB active/20.278 GiB peak and reached 68.027 tok/s.
- [x] Quality-gate affine Q8/32 for the untied LM head.
  `SUCCESS` (2026-07-17): raw Q8/32 reduces the head from 0.9473 to 0.5328 GiB
  but changed 9 of 4,096 greedy choices over sixteen coding domains. The
  accepted hybrid ranks the full vocabulary with Q8, maps the top 64 BF16 source
  rows, and re-scores them with a Metal reduction matching the complete head.
  All 4,096 reranked choices matched and every Q8 candidate pool retained all
  source top-20 tokens; mean full-distribution KL was 1.2783e-4. Separate
  256-token greedy and seeded recommended-sampling generations were byte
  identical. A balanced 272-step greedy A/B retained every choice and all 80
  persistent tensors while improving 62.666 to 64.729 tok/s (3.29%); a balanced
  136-step sampled A/B improved 61.923 to 64.825 tok/s (4.69%). Production peak
  fell from 20.692 to 20.278 GiB. The hybrid is now the generator default;
  `--no-quantized-lm-head` retains the complete BF16 authority.
- [x] Evaluate affine Q8/32 for the input embedding.
  `REJECTED` (2026-07-17): although it saved 0.4144 GiB, local row error
  amplified through all 40 layers. Three 128-step teacher-forced trajectories
  had 4.27%-4.68% mean logit relative L2, changed 5,525, 6,171, and 6,243 of
  40,960 routed expert IDs, and produced greedy mismatches beginning at steps
  35, 55, and 30. The generation CLI does not expose this path.

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
- [x] Replace decode-only FP4/FP8 arithmetic decoders with exact bit conversion.
  `SUCCESS` (2026-07-17): the two fused one-token MoE kernels convert E2M1 and
  E4M3FN encodings through exact half-bit layouts rather than a lookup and
  dynamic `exp2`. Exhaustive checks cover all 16 FP4 and 256 FP8 encodings,
  including the two reserved NaNs. Batched prefill retains its faster prior
  decoder. Real layer-19 one-token MoE improved 2.00%, and a balanced
  160-round complete-model comparison preserved all 162 logit, hidden, route,
  recurrent, convolution, and K/V tensors while improving 63.949 to 64.504
  tok/s (0.87%). Active/peak memory remained 20.503/20.515 GiB.
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
- [x] Feed convolved Q/K/V directly into the decode recurrence.
  `SUCCESS` (2026-07-17): one 32-SIMD threadgroup per value head reproduces the
  two 128-wide FP32 L2-normalization trees and addresses repeated query/key
  heads directly inside the existing recurrence/core/gate dispatch. This
  removes materialized FP32 Q/K/V and two repeat graphs while leaving beta and
  decay arithmetic unchanged. Synthetic production-shape parity and a real
  layer-0 check preserved output, convolution state, and every FP32 recurrent
  value bit-for-bit; the real mixer improved from 422.31 to 377.08 us (1.12x).
  A balanced 160-round full-model A/B retained all 162 tensors and improved
  68.147 to 69.860 tok/s (2.51%) without memory growth. A separate 128-step
  greedy trajectory matched all 20,736 tensor comparisons and selected tokens.
  The durable profiler independently measured 68.351 to 69.790 tok/s, reduced
  synchronized GDN mixer cost from 12.149 to 11.194 ms, and reported zero logit
  drift. Prefill remains on its separately optimized chunk recurrence.
- [x] Fuse GatedDeltaNet beta and decay scalar graphs for decode.
  `SUCCESS` (2026-07-17): one 32-thread Metal dispatch reproduces MLX 0.32's
  stable sigmoid, compensated `log1p` softplus, precise exponential/logarithm
  operations, and FP32 output boundaries. A broad 500-batch randomized oracle
  matched all 16,000 beta/decay scalar pairs bit-for-bit. Real layer-0 mixer
  latency improved from 387.90 to 358.54 us (1.082x). A balanced 180-round
  full-model A/B retained all 162 tensors and improved 70.031 to 72.259 tok/s
  (3.18%) without memory growth. A separate 128-step greedy trajectory matched
  all 20,736 tensor comparisons and selected tokens. The durable 20-sample
  profiler independently measured 69.812 to 72.229 tok/s, reduced synchronized
  GDN mixer cost from 11.127 to 10.235 ms, and reported zero logit drift.
- [x] Fuse all one-token GatedDeltaNet input and transition preparation.
  `SUCCESS` (2026-07-17): one dispatch preserves MLX 0.32's distinct exact
  GEMV trees: `BM=8, BN=1` for QKV/z and small-output `BM=1, BN=8, TM=4` for
  b/a, including ordered cross-SIMD reduction and BF16 rounding before beta and
  decay. This fixes the prior naive concatenation's reduction mismatch without
  a joined weight allocation. A 300-input real-weight probe and production
  transition test were bit-exact. Real layer-0 improved from 350.35 to 339.01
  us (1.034x). A 180-round full-model A/B preserved all 162 tensors and moved
  71.538 to 72.411 tok/s (1.22%). All six balanced 40-round blocks improved;
  aggregate decode moved 71.389 to 72.056 tok/s (0.93%). A separate 128-step
  trajectory matched all 20,736 tensor comparisons and selected tokens. The
  independent sequential profiler was noise-limited at 71.945 versus 72.008
  tok/s, but reduced synchronized GDN mixer cost from 10.308 to 9.931 ms
  (3.66%) with zero logit drift and unchanged 20.033/20.278 GiB active/peak
  memory.
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
- [x] Fuse post-mixer residual/RMSNorm with the combined MoE route projection.
  `SUCCESS` (2026-07-17): seventeen threadgroups each reproduce MLX 0.32's
  exact 512-thread centered-norm reduction, retain its rounded BF16 output in
  threadgroup memory, and evaluate 16 router rows with the original one-SIMD
  GEMV tree. Only task zero writes the shared residual and normalized outputs;
  no grid barrier, atomic reduction, or joined weight allocation is required.
  A 300-input real-weight probe and production-shape test were bit-exact. The
  complete real layer-0 path improved from 488.32 to 483.96 us. All six
  balanced full-model blocks improved; aggregate decode moved from 72.492 to
  73.097 tok/s (0.84%). A separate 128-step trajectory matched all 20,736
  tensors and selected tokens. Independent 20-sample profiler processes moved
  72.119 to 73.411 tok/s (1.79%), reduced execution from 12.304 to 12.137 ms,
  reported zero logit drift, and retained 20.033/20.278 GiB active/peak memory.
- [x] Compile each fixed-shape production GatedDeltaNet layer after exact
  kernel stabilization.
  `SUCCESS` (2026-07-17): the 30 GatedDeltaNet layers are independently bound,
  compiled, and fully warmed when a production BF16 session starts; the ten
  position-dependent attention layers remain uncompiled. Exact custom kernels
  now preserve the reduction and rounding boundaries that the rejected early
  whole-model compile changed. All six balanced 40-round blocks improved; the
  5%-trimmed full-model result moved from 73.136 to 81.396 tok/s (+11.29%) with
  all 162 tensors unchanged. A separate 128-step immutable greedy trajectory
  matched 20,736 tensor comparisons and every token. The production linear K/V
  session also matched 20,736 tensors and every token while moving 72.793 to
  80.470 tok/s (+10.55%). Independent 20-sample processes measured 73.476 to
  81.343 tok/s, reduced median host construction from 1.566 to 0.753 ms and
  execution from 12.091 to 11.531 ms, and retained 20.033/20.278 GiB
  active/peak memory. The feature defaults on with
  `--no-compiled-gdn-layers` as the exact fallback.
- [x] Compile the fixed-shape tail after each full-attention mixer.
  `SUCCESS` (2026-07-17): only residuals, centered norms, routing, MoE, and the
  following norm are bound; attention, RoPE, K/V append, and variable-length
  GQA remain dynamic. All six balanced 40-round blocks improved, with the
  5%-trimmed full-model result moving from 81.454 to 82.993 tok/s (+1.89%) and
  all 162 tensors unchanged. A separate 128-step immutable trajectory matched
  20,736 tensor comparisons and every token. The production linear-cache path
  also matched 20,736 tensors and tokens while improving from 80.044 to 82.092
  tok/s (+2.56%). Independent 20-sample processes measured 81.300 to 82.982
  tok/s, reduced median host construction from 0.744 to 0.562 ms, left
  execution effectively flat at 11.532 versus 11.506 ms, and retained
  20.033/20.278 GiB active/peak memory. The feature defaults on with
  `--no-compiled-attention-tails` as the exact fallback.
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
- [x] Reuse exact long-prefix reduction kernels for one-query decode.
  `REJECTED` (2026-07-17): a decode-specific score/value geometry removed the
  prefill kernel's three unused query slots and improved an isolated real
  attention layer by 18.1% at 106K, 12.0% at 131K, and 14.1% at 262K. The
  complete lazy graph moved the other way: exact advancing-session A/Bs fell
  0.8% at 4K, 6.8% at 16K, 12.5% at 64K, 15.8% at 106K, and 17.2% at 131K.
  Ten forced three-dispatch pipelines inhibit MLX scheduling enough to erase
  the isolated gain. Below MLX's 4,096-element looped-softmax threshold, the
  long reduction topology also diverged after 40 real transitions. All custom
  decode kernels and selectors were removed; grouped native MLX remains the
  production path.

## Context And Cache

- [ ] Validate native 262,144-token RoPE and cache semantics.
- [ ] Implement YaRN factor-2 loading for 524,288 tokens.
- [ ] Prove native and YaRN cache profiles cannot be mixed.
- [x] Implement exact in-memory prefix reuse for K/V and GatedDeltaNet state.
  `SUCCESS` (2026-07-17): an MLX 0.32 C++ primitive aliases fixed-capacity BF16
  buffers and a paired Metal kernel appends K and V together without allocating
  or copying the retained prefix. The mutable `TextLinearDecodeSession` has one
  owner, eager commit, fixed capacity, and no rollback contract; immutable
  sessions remain untouched. Prefix conversion copies all ten K/V pairs once,
  directly reuses all GatedDeltaNet state, and generation releases the source
  prefix afterward. Unit trajectories and paired real-checkpoint sessions
  matched all 162 tensors bit-for-bit. Empty-cache decode was neutral (64.201
  versus 64.230 tok/s); linear decode improved 60.160 to 61.022 tok/s at 4K,
  50.556 to 53.341 at 16K, 29.498 to 35.791 at 64K, and 11.419 to 15.531 at
  262K, gains of 1.43%, 5.51%, 21.33%, and 36.01%. The extension append added
  under 2 MiB while mutating paired 64 MiB test caches, proving no hidden
  full-cache allocation. Matched greedy production paths emitted identical text
  and EOS.
- [x] Implement atomic persistent prompt-cache checkpoints.
  `SUCCESS` (2026-07-17): exact prefixes are stored as canonical token bytes,
  one safetensors file per decoder layer, and a strict provenance manifest.
  Linear K/V is compacted one attention layer at a time, limiting save scratch
  rather than materializing a second 5-10 GiB cache. Every payload is hashed,
  verified, fsynced, and then published by atomic directory rename. Restore
  rejects identity, prefix, schema, dtype, shape, metadata, size, hash, symlink,
  and unexpected-file drift. Injected write failure published no entry and left
  no staging directory. A real 128-token checkpoint occupied 67,525,182 bytes,
  saved in 0.330 s, restored in 0.098 s, and peaked only 0.016 GiB above the
  20.571 GiB active model.
- [x] Prove save/restore and incremental append logit parity.
  `SUCCESS` (2026-07-17): synthetic immutable and fixed-capacity states restore
  exactly and continue with bit-identical full-vocabulary logits. The real
  checkpoint benchmark compared all 80 restored K/V, convolution, and recurrent
  tensors plus the next token's logits and 80 successor tensors bit-for-bit.
- [ ] Add content-addressed workspace cache lookup and bounded disk LRU.
  `PARTIAL` (2026-07-17): cache keys bind the exact token prefix and all runtime
  provenance. `--cache-system-prefix` automatically restores or atomically
  warms the rendered system segment, and write-enabled generation enforces a
  protected 24 GiB/64-entry LRU by default. Generic longest-prefix discovery
  across repository snapshots remains open.
- [ ] Add background cache warming without blocking foreground decode.
- [ ] Evaluate eight-bit K/V against BF16 long-context quality and speed.
- [ ] Characterize real Ornith K/V and build a TurboQuant numerical oracle.
  `QUEUED` (2026-07-18): retain BF16 as the authority; measure every full-
  attention layer/head/channel and compare paper-faithful 3.5-bit quantization
  with conservative asymmetric K/V precision. GatedDeltaNet state and model
  weights are outside this experiment.
- [ ] Implement direct packed TurboQuant attention and persistence on Metal.
  `QUEUED` (2026-07-18): encode online without CPU readback, score packed keys
  and aggregate packed values without full BF16 reconstruction, retain a BF16
  recent tail if quality requires it, and give the cache an incompatible,
  provenance-bound schema. Select the BF16/compressed crossover by measurement.
- [ ] Gate TurboQuant on long-context quality, memory, and end-to-end speed.
  `QUEUED` (2026-07-18): test native and YaRN profiles with logits, exact-greedy
  agreement, sampled quality, coding, RULER/needle retrieval, persistent
  restore, active/peak memory, cache I/O, TTFT, and decode throughput. The H100
  attention-logit result is reference evidence, not an Apple performance claim.

## Prefill Performance

- [x] Establish a durable exact state-prefill profiler.
  `SUCCESS` (2026-07-17): two independent fixed-capacity K/V sessions prove
  the forced-boundary composition against the unfenced production graph across
  all 80 persistent tensors. A real 128-token run measured 398.714 tok/s at a
  21.020 GiB peak. Synchronized attribution assigned 170.068 ms to MoE,
  120.763 ms to GatedDeltaNet mixers, 38.223 ms to full-attention mixers, and
  under 14.1 ms to every norm, embedding, RoPE, and final K/V stage combined.
  The profiler does not create the roughly 23 GiB duplicate Metal trace.
- [x] Schedule batched selected gate/up jobs in expert-major order.
  `REJECTED` (2026-07-17): the GPU-only permutation preserved every MoE output,
  route, and selected ID. Random layer-19 inputs improved only 0.60%. Real
  layer-19 activations were strongly clustered (97 active experts; 86 of 1,024
  jobs on the busiest expert), but ordering regressed 4.417 to 4.432 ms
  (0.34%). Apple GPU caching/scheduling already captures the available
  locality, so the prototype was removed completely.
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
- [x] Join all four GatedDeltaNet input projections for chunk prefill.
  `REJECTED` (2026-07-17): a paired real-model 128-token state-prefill A/B
  showed that the larger joined matrix selects a different MLX batched-GEMV
  reduction. It changed 76 of 80 persistent state tensors and was also 0.42%
  slower, moving from 399.513 to 397.831 tok/s across six alternating rounds.
  Active memory was 20.541 GiB and peak memory was 20.774 GiB. The prototype
  was removed completely; decode retains its separately optimized exact path.
- [x] Implement chunk-parallel GatedDeltaNet prefill on Metal.
  `SUCCESS` (2026-07-17): token-batched MLX `vmap` projections retain the
  one-token BF16 accumulation contract, one Metal convolution dispatch walks
  each channel's exact four-slot history, and one head-parallel Metal kernel
  advances the FP32 recurrence through the chunk in token order. Isolated
  chunk kernels and real layers 0, 18, and 38 preserve every output,
  convolution value, and recurrent value bit-for-bit. On real layer 0, chunk
  128 improved 5,484 to 19,459 token-layers/s (3.55x); layer 38 at chunk 256
  improved 47.154 to 12.977 ms (3.63x, 19,728 token-layers/s).
- [x] Reuse BF16 dense weights across exact prompt-token reductions.
  `SUCCESS` (2026-07-17): one SIMD group keeps eight independent token
  accumulators while loading each four-column BF16 weight group once. Every
  token retains the authoritative four-column loop, ordered shuffle reduction,
  and BF16 output boundary. Tile 16 lost occupancy and tile 32 regressed, so
  production retains tile 8. Real layer 18 improved from 4.075 to 2.805 ms
  (1.45x). Across all production chunk sizes, 8/16/32/64/128-token state
  prefill improved by 7.71%/10.08%/10.99%/12.30%/13.11%. A balanced 16-round
  128-token full-model A/B preserved all 80 persistent tensors and improved
  398.991 to 452.429 tok/s (13.39%) at a 20.717 GiB peak. The observable final
  path preserved all 162 logits, routes, and state checks and improved 395.661
  to 446.704 tok/s. Decode remains on its separately tuned one-token kernels.
- [x] Apply exact token tiling to full-attention prefill projections.
  `SUCCESS` (2026-07-17): BF16 Q/K/V/output rows retain the same per-token
  accumulation and rounding contract; singleton final-query and decode work
  remains on MLX. Real layer 19 improved from 5.325 to 4.066 ms at an empty
  prefix (1.31x) and from 5.092 to 4.067 ms after 1,024 tokens (1.25x). On top
  of retained GDN tiling, a balanced 16-round 128-token model A/B preserved all
  80 states and improved 453.495 to 468.467 tok/s (3.30%). Gains across
  8/16/32/64/128-token chunks were 1.27%/2.69%/2.77%/3.34%/3.36%. The final
  observable path preserved all 162 checks and improved 448.660 to 463.828
  tok/s. Peak remained 20.717 GiB and no resident allocation was added.
- [x] Reuse router and shared-expert weights across prompt tokens.
  `SUCCESS` (2026-07-17): the BF16 router holds eight independent exact token
  reductions per SIMD group. Packed shared gate/up/down kernels hold four token
  accumulators while decoding each E2M1 weight and FP8 block scale once. Every
  token retains the authoritative block/pair accumulation, SIMD reduction, and
  BF16 rounding boundaries; routed experts and one-token decode are unchanged.
  Synthetic production-shape composition and real layer 19 preserved output,
  top-8 IDs, and routing weights bit-for-bit. The real layer improved from
  4.402 to 4.196 ms. A balanced 24-round 128-token full-model A/B preserved all
  80 persistent tensors and improved 462.890 to 477.203 tok/s (3.09%) at the
  unchanged 20.717 GiB peak.
- [x] Feed BF16 activations directly to packed NVFP4 prefill kernels.
  `SUCCESS` (2026-07-17): Metal converts each BF16 load to the identical FP32
  operand at use, removing materialized FP32 copies for routed/shared gate-up
  and down without changing accumulation or rounding boundaries. Synthetic
  production-shape MoE composition preserved output, routes, and selected IDs
  bit-for-bit. Real layer 19 improved from 4.258 to 4.150 ms (1.026x). A
  balanced 24-round 128-token full-model A/B preserved all 80 persistent state
  tensors and improved 470.442 to 478.455 tok/s (1.70%) at the unchanged
  20.717 GiB peak. The observable path preserved all 162 logits, hidden, route,
  and state checks while improving 480.330 to 488.445 tok/s (1.69%). The same
  exact one-token experiment was neutral at 68.903 versus 68.944 tok/s, so
  decode retains its prior FP32 materialization and its wider input contract
  was removed.
- [x] Fuse full-attention chunk Q/K normalization, partial RoPE, and gate split.
  `SUCCESS` (2026-07-17): production-shape Metal kernels preserve every BF16
  query, gate, and key element bit-for-bit, including high native-context RoPE
  positions and the final layer's key-only path. The real layer-19 attention
  chunk improved from 4.305 to 4.102 ms (4.95%). A longer alternating
  complete-model comparison retained all 80 persistent state tensors and
  improved 459.320 to 461.487 tok/s (0.47%); the separately optimized final
  path retained all 162 hidden, logit, route, and state checks and improved
  454.507 to 457.097 tok/s (0.57%). The paired peak remained bounded at
  20.894 GiB and production adds no resident allocation.
- [x] Reuse routed gate/up weights across expert-grouped prompt tokens.
  `REJECTED` (2026-07-17): two GPU-only prototypes sorted all 1,024 selected
  jobs by expert and scattered results back to original token/slot order. The
  first used fixed 2/4/8-job tiles with exact mixed-boundary fallbacks; the
  second built expert-aligned tiles from GPU counts and prefix sums so every
  data tile was homogeneous. Both preserved every FP32 gate/up result
  bit-for-bit. Random inputs activated 227 experts and the best aligned tile
  slowed 2.494 to 3.388 ms. A real coding-prompt layer-19 trajectory activated
  only 97 experts, with 86 jobs on the busiest expert, but the best tile still
  slowed 2.454 to 3.258 ms (25%). Existing cross-SIMD cache reuse plus higher
  parallelism beats serial multi-token accumulators on this M4 Max. All
  prototype code was removed.
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
- [x] Evaluate MLX 0.32 native NVFP4 gather-QMM for routed prefill.
  `REJECTED` (2026-07-17): native E2M1/E4M3 group-16 gather-QMM is materially
  faster, reducing real layer-19 selected gate/up from 2.531 to 2.031 ms when
  each selected input is pre-scaled by the checkpoint's per-expert global
  factor. Its reduction topology nevertheless changed 244 of 1,048,576 BF16
  outputs (maximum absolute 0.00390625); post-scaling native BF16 outputs
  changed 273,292 values. This violates the exact production contract, so no
  native gather-QMM code or duplicate weight representation is retained.
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
- [x] Prefill directly into the fixed-capacity production K/V cache.
  `SUCCESS` (2026-07-17): the generator creates its single-owner linear session
  before token zero, and a paired Metal primitive transposes native contiguous
  K/V projections directly into final cache storage. This removes the
  immutable-to-linear handoff and its transient second 5 GiB native-context
  allocation. Real-checkpoint 128-token continuation tests preserved all 161
  hidden, route, recurrent, convolution, and active K/V tensors bit-for-bit.
  Throughput was neutral at zero/4K prefixes, +0.18% at 16K, +0.67% at 64K,
  and +0.58% at 262K. Production native-capacity memory is the 21.27 GiB model,
  one 5 GiB K/V cache, and bounded scratch; the 42.19 GiB native benchmark peak
  deliberately retained source, immutable, and linear caches together.
- [x] Batch exact attention reductions beyond the measured long-prefix crossover.
  `SUCCESS` (2026-07-17): model-specific Metal kernels reproduce MLX 0.32.0's
  BF16 score GEMV shuffle tree, 1,024-thread looped FP32 softmax, BF16
  probability boundary, and value GEMVT reduction while processing a complete
  causal chunk in three dispatches. The conservative production crossover is
  106,496 cached tokens. A real layer with deterministic nonzero 131K K/V was
  bit-exact and improved 1.31x; isolated zero-cache comparisons were exact and
  improved 1.19x at the threshold, 1.28x at 131K, and 2.00x at native 262K.
  Complete 128-token, 40-layer continuations retained all 161 hidden, route,
  recurrent, convolution, and appended K/V tensors bit-for-bit. End-to-end
  throughput moved from 36.743 to 48.172 tok/s at 131K (1.31x) and from 8.800
  to 22.591 tok/s at native context (2.57x). The 26.72 and 32.71 GiB peaks
  deliberately retained separate source and candidate linear caches; the
  generator uses one cache and enables the exact selector by default.
- [x] Fold BF16 score scaling into exact long-prefix softmax.
  `REJECTED` (2026-07-17): the fused kernel matched every probability, final
  attention value, and K/V element bit-for-bit at 106K, 131K, and native 262K
  prefixes. Layer-local chunk-128 latency changed by only +0.40%, +0.28%, and
  +0.04%, respectively. Separate native-context processes measured the same
  3.058 GiB peak with and without the materialized multiply, proving MLX already
  reuses the score buffer across that boundary. The prototype was removed.
- [x] Fuse exact long-prefix softmax with value reduction.
  `SUCCESS` (2026-07-17): a 15,360-element BF16 probability tile eliminates the
  materialized probability tensor while reproducing the established FP32
  softmax tree, BF16 boundary, value reduction, output, and K/V bit-for-bit.
  Deterministic nonzero-K/V chunk-128 layer tests improved from 189.190 to
  183.548 ms at 106,496 (3.07%) and 240.706 to 233.759 ms at 131,072 (2.97%).
  A 40-layer A/B retained all 80 persistent tensors and improved 47.937 to
  48.455 tok/s (1.08%). Separate processes reduced layer-local peak scratch
  from 1.559 to 1.059 GiB at 131K and, when forced, from 3.058 to 2.058 GiB at
  native context. Chunk-64 remained favorable, while chunk-8/16 throughput fell
  18.63%/12.15% at 131K and chunk-32 was neutral. Realistic K/V regressed 5.74%
  at 139K and the one-query final layer regressed 2.57%, so production enables
  fusion only for chunks of at least 64 tokens from 106,496 through 131,072 and
  retains the split exact path everywhere else.
- [x] Elide unobservable final-layer work from non-final prompt chunks.
  `SUCCESS` (2026-07-17): layers 0-38 execute unchanged while layer 39 projects
  and appends only the K/V that future tokens can observe. Paired real-model
  linear sessions preserved all 80 persistent recurrent, convolution, and K/V
  tensors bit-for-bit. Exact chunk-128 throughput improved from 389.721 to
  399.652 tok/s cold, 94.760 to 103.466 at 64K, 47.934 to 52.589 at 131K, and
  22.482 to 24.391 at native 262K. The native paired peak was 38.19 GiB because
  it retained a source state and two 5 GiB candidate caches; production owns
  one cache.
- [x] Evaluate only the last observable token in the final prompt layer.
  `SUCCESS` (2026-07-17): layer 39 appends K/V for the complete final chunk but
  projects only its last query and runs residual, MoE, final norm, and LM head
  only for that token. Real-checkpoint comparisons retained final hidden and
  full-vocabulary logits, every final-token route, and all persistent state
  bit-for-bit across 162 checks. Chunk-128 throughput improved from 386.472 to
  396.696 tok/s cold, 95.282 to 103.380 at 64K, and 47.619 to 52.135 at 131K.
  A memory-pressure-heavy two-cache native stress remained exact and improved
  18.423 to 19.189 tok/s at a 38.20 GiB peak. Full-hidden APIs remain the
  unchanged authority and fallback.
- [ ] Measure cold prefill, restored-prefix, and incremental-suffix paths separately.
  The persistent-cache benchmark now reports save and verified restore latency;
  substantial-prefix TTFT and suffix-length sweeps remain open.

## Speculative Decode

- [x] Pin the released DSpark schema and target auxiliary-state interface.
  `SUCCESS` (2026-07-17): the dependency-free validator binds the exact public
  revision, legacy anchor-plus-seven proposal semantics, all 44 tensor names,
  dtypes, shapes, contiguous offsets, and the 1,657,163,778-byte payload. An
  audit of training-era Speculators and vLLM `v0.24.0` established that IDs
  `9,19,29` select the residual outputs after target decoder layers `8,18,28`.
  Opt-in immutable and single-owner Metal decode/prefill APIs capture those
  pre-final-norm outputs without retaining unrequested layers. Tiny-model tests
  match explicit layer composition and preserve target hidden, logits, and
  state exactly. A clean-parent/current production A/B measured 28.721 versus
  28.586 ms for exact block-8 verification at the same 21.098/21.105 GiB
  active/peak footprint, showing no capture-disabled regression. This validates
  the interface only. The 1.543358 GiB public draft was subsequently accepted
  and measured under the separate public-draft experiment below.
- [x] Implement the released DSpark equations and exact target-state integration.
  `SUCCESS` (2026-07-17): a dependency-free scalar oracle and independent MLX
  path implement standard Qwen3 RMSNorm, the training-authoritative full
  256-dimensional RoPE, projected target-prefix K/V, three noncausal draft
  layers, the 32K head, sequential rank-256 Markov correction, confidence, and
  target-vocabulary mapping. Deterministic tests compare all context K/V,
  hidden rows, base/corrected logits, confidence values, and IDs. A strict
  synthetic safetensors test accepts all 44 exact tensors and rejects schema
  drift. The target verifier now returns only committed auxiliary rows: full
  acceptance and every forced mismatch position preserve exact target state,
  while repeated draft/target sessions keep both cursors position-aligned.
  Target and draft K/V now use separate fixed-capacity Metal buffers with
  checked ownership; stale sessions fail before modifying shared storage. At
  native context these caches total exactly 6.5 GiB (5.0 target plus 1.5 draft).
  The full 171-test suite passes. A real four-block linear-cache trajectory
  reproduced 33 serial greedy tokens and all 82 target observables exactly;
  block-8 target verification measured 29.309 ms first and 29.042 ms steady at
  21.223/21.293 GiB active/peak with auxiliary capture and the exact BF16 block
  head. A separate source-acceptance profile now binds the companion repository,
  revision, metadata format, directory, exact size, and SHA-256 before atomically
  publishing `source-dspark-state.json`. The real payload exposed a critical
  training-format detail: `d2t` stores offsets, so target token ID is draft
  index plus `d2t[index]`, and the reconstructed IDs must exactly equal the
  selected `t2d` population. Scalar and MLX paths now enforce that inverse and
  reject direct-ID interpretation.
- [x] Measure the public matched DSpark draft against the authoritative target.
  `REJECTED` (2026-07-18): the 1,657,168,394-byte source passed full SHA-256
  verification and strict-loaded all 44 tensors. A 64-step binary-search coding
  prompt matched an independently advanced serial greedy target exactly for 89
  emitted tokens. The draft accepted 24/448 future proposals, averaging 0.375
  future tokens per block with no all-accepted block. Standalone proposal cost
  was 4.176 ms; the initial full-block path measured 35.211 tok/s versus a fair
  78.166 tok/s serial target baseline. The retained staged verifier below raised
  this to 59.382 tok/s at 21.657/21.723 GiB active/peak, but that remains a
  rejected-for-production 0.755x of its paired 78.686 tok/s target. Mean
  confidence remains close to the published validation value, supporting
  equation alignment. The later revision-bound 12-prompt gate replayed every
  emitted trajectory through an independent target session and matched hidden,
  logits, all 30 recurrent states, and all active K/V rows exactly. It accepted
  only 35/672 future proposals (5.208%), reached 58.745 versus 77.855 tok/s
  (0.7555x steady), lost on all 12 prompts, and peaked at 21.776 GiB. The public
  draft is therefore measured and retained only as a mechanism reference.
- [x] Remove redundant exact-verifier boundaries and stage low-acceptance work.
  `SUCCESS` (2026-07-18): exact block decisions now use one row-wise argmax and
  materialize with target state in one synchronization. Real all-accepted
  block-8 verification improved from 29.083 to 27.409 ms (5.76%). A configurable
  causal stage stops at the first failed proposal; width one reuses normal
  optimized target decode, needs no rollback replay or exact BF16 block head,
  and saves 0.947 GiB. Synthetic full acceptance and every mismatch position
  matched unstaged emitted tokens, target hidden/logits/state, and draft K/V
  exactly. The 64-step real run reproduced the same 89 target tokens, improved
  35.211 to 59.382 tok/s (68.6%), and reduced 22.604/22.674 to 21.657/21.723 GiB
  active/peak. Its acceptance histogram was `[46,14,2,2,0,0,0,0]`; accepted and
  rejected first-slot confidence means of 0.414686 and 0.391888 reject confidence
  scheduling. All 171 tests, including the Metal extension, pass. This is a
  retained verifier win, not approval to enable the slower public draft.
- [x] Build a strict one-shard-at-a-time Qwen3.5 MTP extractor.
  `SUCCESS` (2026-07-18): the dependency-free extractor pins both exact source
  files, their full and header SHA-256 identities, all 785 BF16 `mtp.*` ranges,
  and the deterministic 1,689,376,064-byte sidecar layout. Atomic state binds
  source and runtime revisions plus metadata, extractor, per-tensor, and final
  output hashes. Source identity is fixed across full hashing, extraction, and
  deletion; every copied range is read back before state commit. Resume verifies
  prior output, handles stale raw files, and recovers interrupted initialization
  or final rename. Eleven corruption, immutability, resumption, cleanup, and
  no-download planning tests pass, as does the complete 182-test suite including
  the rebuilt Metal
  extension. The production metadata-only plan reports a 7.071164 GiB transfer
  and 6.572544 GiB conservative peak without creating an artifact or cache.
- [x] Extract and validate the official Qwen3.5 MTP bootstrap tensors.
  `SUCCESS` (2026-07-18): the approved streaming run downloaded only pinned
  shards 13 and 14, verified their full SHA-256 identities, read back every
  copied tensor range, and accepted the deterministic 1,689,376,064-byte
  sidecar with 1,689,281,536 payload bytes and 785 BF16 tensors. Its final
  SHA-256 is
  `11c9043bf0c92c1eea7b4c6ffbadeb890a080a84d301872a29839be209099c1f`.
  `source-mtp-state.json` records both completed shards and every tensor hash;
  the raw shards and dedicated transfer cache were removed only after final
  acceptance.
- [x] Implement the pinned Qwen3.5 MTP equation and exact target reconciliation.
  `SUCCESS` (2026-07-18): an independent scalar oracle and strict MLX BF16
  loader implement normalized embedding/target-hidden fusion, `mtp.fc`, the
  gated full-attention decoder layer, dense top-8-plus-shared MoE, final norm,
  and target-head projection. The vLLM-compatible shifted alignment pairs each
  target hidden row with its following authoritative token. Block verification
  now returns every committed final target hidden row, and MTP K/V is rebuilt
  only from those rows and committed tokens after full or partial acceptance.
  Synthetic scalar/MLX, serial/batch, complete-acceptance, rollback, stale
  anchor, schema, hash, and real-sidecar tests pass. A real position-60
  block-3 probe matched all 41 captured layer boundaries, routes, final hidden,
  and next token bit-for-bit between compiled and serial target paths. The
  complete 191-test suite, including the rebuilt Metal extension, passes.
- [x] Establish production-beneficial bootstrap MTP acceptance on coding work.
  `REJECTED` (2026-07-18): exact target verification preserves serial greedy
  output, but the unmodified Qwen bootstrap does not generalize uniformly to
  the post-trained Ornith target. A 32-block Rust LRU trajectory generated 77
  exact target tokens, accepted 44/64 future proposals (68.75%), and measured
  79.910 versus 79.433 tok/s, or 1.006x, at 21.669/23.142 GiB active/peak.
  A 32-block C++ queue trajectory generated 68 exact tokens, accepted 35/64
  (54.69%), and measured 68.070 versus 78.812 tok/s, or 0.864x. Block two on
  that same workload accepted 22/32 but remained slower at 70.103 versus
  79.524 tok/s. A coherent source-BF16-head mode is retained for validation;
  it used 22.081/23.556 GiB and was slower in absolute throughput than the
  hybrid Q8/32 target head. Fixed block scheduling is therefore not approved
  for production.
- [x] Distill an Ornith-targeted MTP sidecar if bootstrap acceptance is inadequate.
  `SUCCESS` (2026-07-18): a resumable, provenance-bound capture generated 5,044
  target rows and 4,096 scored positions from 32 coding prompts. Rank-32 LoRA
  training changes only `mtp.fc`, then folds into one BF16 2,048-by-4,096
  replacement with zero runtime adapter operations. The accepted seed-29
  artifact is 16,777,533 bytes with SHA-256
  `a42cf411862e04485124bcefc96a5092abd7f2e25b4eb663ac4b703b8e4bbc11`.
  Prompt-disjoint trace acceptance improved from 734/896 (81.92%) to 761/896
  (84.93%), while hidden relative L2 improved from 0.88169 to 0.78947. A
  rank-64/harder-loss run, the seed-17 candidate, and BF16 interpolation did
  not improve the selected aggregate. Exact resident gates then exposed and
  repaired three independent target-boundary defects: batched route
  renormalization, chunk GDN rollback, and compiled generic RMSNorm. The full
  203-test Ornith-35 suite, including rebuilt Metal extensions, passes.
- [ ] Train or extend an Ornith-targeted DSpark draft if needed.
- [x] Implement exact block verification with GDN/KV snapshot and rollback.
  `SUCCESS` (2026-07-17): the greedy verifier evaluates up to eight target
  tokens causally, preserves the generator's single-token Q8 LM-head reduction
  order, and makes the consumed-state versus pending emitted-token contract
  explicit. Forced mismatches at every position matched a separate
  accepted-prefix evaluation across all 80 persistent tensors plus hidden and
  logits. Immutable attention K/V truncates directly; the fixed-capacity path
  rolls back its logical cursor. A compact normalized-input journal reconstructs
  only the 30 GatedDeltaNet states and avoids full attention/MoE replay. Eight
  real block-8 iterations matched 65 compiled-target greedy
  tokens exactly. Eight serial transitions took 97.016 ms versus 41.433 ms for
  all-accepted verification; exact forced-mismatch rollback took 43.614 to
  47.938 ms at 20.154/20.278 GiB active/peak. This proves the target mechanism,
  not end-to-end DSpark acceleration; draft cost and acceptance remain open.
- [x] Compile numerically safe fixed-block verifier tails and project the exact
  BF16 vocabulary block once.
  `SUCCESS` (2026-07-17): forty fixed-token compiled tails bind only the
  post-mixer residual, MoE, second residual, and next RMSNorm graph. They
  improved an exact block-8 A/B from 41.581 to 39.241 ms (5.63%) without state
  or hidden drift. A model-specific BF16 LM-head kernel reads each retained
  source weight once for up to eight token rows while reproducing eight
  independent GEMV reduction trees bit-for-bit. With the compiled tails it
  reduced verification to 31.172 ms. Intermediate and bonus choices use those
  exact logits; the returned cursor still projects one Q8/32 vector so it
  remains identical to normal generation. The retained BF16 head raises
  active memory by 0.944 GiB, to about 21.10 GiB.
  A later real block-3 MTP trajectory exposed an uncovered BF16 boundary:
  `mx.compile` fused the batched MoE SiLU product, first drifting after layer 5
  and eventually changing the bonus token. Those pre-fix MTP timings are
  invalid. The retained model-specific BF16 SiLU Metal kernel now fixes both
  multiplication rounds explicitly; a whole-MoE compiled regression and the
  real 41-boundary position-60 probe are bit-exact. The repaired 32-block MTP
  runs above independently compare every emitted block with serial target
  output and supersede the unsafe measurements.
- [x] Specialize the exact block-8 GatedDeltaNet and attention mixer hotpath.
  `SUCCESS` (2026-07-17): bounded Metal kernels fuse GDN QKV/Z projection,
  convolution, SiLU, B/A transition, beta/decay, direct-convolved recurrent
  column update, core normalization, and z gate while preserving every FP32
  reduction and BF16 boundary. A separate branch-uniform kernel joins the
  attention Q/K/V dispatches without joining their storage. Balanced full
  verifier A/Bs were independently exact and measured 2.81%, 0.84%, 1.53%,
  0.61%, 0.72%, and 0.62% gains for the retained stages. The final production
  run reproduced a 65-token target trajectory and all forced rollback tensors;
  eight serial transitions took 96.737 ms versus 29.331 ms all-accepted
  verification (3.298x less target work, 306.839 emitted-token/s target
  ceiling). Mismatch positions one through seven took 32.180 to 34.402 ms at
  21.098/21.105 GiB active/peak. The full `ornith35/check.sh` suite passes.
- [x] Compile the complete small-block GatedDeltaNet mixer graph.
  `REJECTED` (2026-07-17): whole-mixer and complete-layer compilation changed
  recurrent state values despite matching shapes and produced downstream
  drift. Only the independently bit-exact post-mixer tails are retained.
- [x] Reuse routed-expert weights across matching proposal positions.
  `REJECTED` (2026-07-17): the real block routes only about 28 unique experts
  per layer for 64 selected slots, but exact all-token and tile-2 Metal kernels
  slowed the 31.17 ms verifier to 44.96 and 43.27 ms. Register pressure,
  divergence, and the extra down-reduction stage outweighed reduced weight
  reads. All prototype code was removed.
- [x] Add exact measured-yield MTP-to-target scheduling.
  `SUCCESS` (2026-07-18): block three is retained after exact block two through
  five sweeps; two cannot amortize draft cost and four/five lose acceptance.
  The scheduler waits for eight MTP blocks and detaches permanently to the
  already-owned target cache when the latest four-block future acceptance is
  below 70%. Detached decode evaluates no MTP projection, layer, or context
  reconciliation. The final prompt-disjoint seven-run gate at revision
  `2f541a4199af770f5eb1f853570412f7dd509266` was serial-token-exact and measured
  414 target transitions at 82.873 versus 77.511 tok/s (1.069x), with every
  prompt between 1.027x and 1.149x and 23.377 GiB peak memory. The accepted
  state binds the gate log SHA-256
  `f266cdaf59f1ac92c72002830af3dce558af9f81ff6006a38b90c6bda9362e31`.
- [x] Integrate exact adaptive MTP into normal greedy and sampled generation.
  `SUCCESS` (2026-07-18): prompt hidden rows stream into fixed-capacity BF16 MTP
  K/V with no prompt-sized hidden copy. Sampled proposals now use the complete
  MTP top-k/top-p distribution and standard `min(1,p/q)` acceptance with the
  normalized positive `p-q` residual, exactly reconstructing the target
  distribution. At 32 prompt tokens greedy output was byte-identical, accepted
  22/22 future tokens, and improved 76.796 to 85.530 tok/s (1.114x) at a 23.063
  GiB peak. At 35 prompt tokens recommended sampling accepted 40/46 and improved
  77.639 to 86.458 tok/s (1.114x) at 23.112 GiB peak. Unlimited MTP regressed at
  269/1,057/5,204 prompt tokens to 71.713/56.335/46.814 tok/s versus target
  77.630/76.571/70.887, and 5,204-token prefill fell from 455.565 to 326.865
  tok/s. Production therefore skips MTP above 256 prompt tokens before sidecar
  load or prefill; the protected 269-token command retained target output and
  reached 78.452 tok/s at 20.278 GiB. The complete 212-test suite and real
  short/long generation gates pass. A later broad prompt-disjoint gate supersedes
  the one-prompt sampled speed claim for scheduling: all 36 sampled trajectories
  were target-state exact, but reached only 0.9340x steady throughput overall
  and 0.9726x on the thinking subset. Sampled MTP is therefore not a measured
  production regime.
- [x] Persist exact suffix-independent MTP prefix state with target checkpoints.
  `SUCCESS` (2026-07-18): cache schema v2 stores target state at position `N`,
  compact MTP K/V through `N-1`, and the final authoritative target hidden row.
  It never persists a sampled pending token; restore attaches the boundary row
  to the first uncached suffix token. Target-only and MTP entries have distinct
  content identities, with MTP entries bound to the sidecar, selected folded
  adapter, source/adaptation state, exact runtime, and config hashes. Synthetic
  split-prefill save/restore reproduced target hidden, logits, all recurrent/K/V
  state, MTP K/V, and final MTP hidden bit-for-bit. Corrupt draft payloads,
  missing/extra draft state, wrong identity, and the valid zero-length K/V case
  are covered. A real 12-token system-prefix cache occupied 0.061 GiB, restored
  in 0.062 seconds, and both warm and restored runs returned exact `READY` with
  2/2 future-token acceptance at a 22.355 GiB peak. All 216 tests pass.
- [x] Compare accepted MTP against DSpark and select by prompt/context regime.
  `SUCCESS` (2026-07-18): `ornith35_mlx_draft_regime_gate.py` commits atomic,
  resumable, revision-bound results after every run and teacher-forces each
  emitted path through a fresh target cache. It compares hidden, logits, every
  recurrent state, and active K/V bit-for-bit, then uses that same path as the
  paired timing baseline. The committed 12-prompt corpus has 51-68 rendered
  tokens, mixes thinking policies and languages, and has no rendered-prompt
  overlap with the 32-prompt distillation capture. At revision
  `193cfb086fa3fb7de32ecfebadb2cbf7a62c6348`, sampled MTP was exact on 36/36
  seeded runs but measured 74.150 versus 78.127 tok/s and 0.9340x steady, with
  only 2/12 prompt groups faster. Greedy MTP was exact on 12/12 and measured
  77.140 versus 77.968 tok/s and 0.9778x steady overall. Its measured
  thinking-mode subset was the only useful regime: 81.439 versus 77.976 tok/s,
  1.0323x steady, 85.05% future acceptance, and a 0.9661x per-prompt floor;
  non-thinking greedy fell to 0.8385x steady. DSpark was exact on 12/12 but
  accepted 5.208% and measured 0.7555x steady. A 4/4/0.75 early-detach MTP sweep
  reduced thinking-mode steady speedup to 1.0114x and is rejected; 8/4/0.70 is
  retained. Production remains target-only by default. Explicit MTP is selected
  only for greedy thinking generation at no more than 256 prompt tokens; the
  existing zero prompt ceiling is the deliberate unmeasured-regime override.
  Result SHA-256 values are `4365c9e9fd3da4ca72fc596a490ea0aad6fa8ccf19fb62d31f164b782759598c`
  (sampled MTP), `e26b9d30600f2d47533c07ad6a3962b5747aea4f7795809296d642ce3d64862a`
  (greedy MTP), and `988d8a2650cba62db34ed16dadfbaa12bf441624a04a0c34b33e5d72cbb71617`
  (DSpark).
- [ ] Measure exact generation speed at 2K, 128K, 262K, and 524K context.

## Optional Semantic Changes

- [ ] Evaluate repository-aware AST/symbol retrieval before model changes.
- [ ] Evaluate CacheBlend-style non-prefix reuse under matched coding gates.
- [ ] Evaluate a target-trained sparse-attention indexer for cold 524K prefill.
- [ ] Consider further quantization only after the source runtime is accepted.
- [ ] Consider expert pruning only after quant-only quality is established.

# Ornith 1.0 35B Local Runtime

This branch builds a model-specific Apple Silicon runtime around
`AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`. It is independent of
the original DS4 implementation, the previous Ornith 397B experiment, and the
Nemotron implementation.

## Selected Checkpoint

Target revision:

`85ffd2d0629ae5fa4f860dda356ec33161806c9b`

The repository contains one `model.safetensors` file of 23,741,821,016 bytes
(22.111294 GiB). Its indexed tensor payload is 23,728,491,712 bytes
(22.098880 GiB). Its mixed precision is intentional:

- all routed experts and the shared-expert MLP use weight-only NVFP4 W4A16
- full attention remains BF16
- GatedDeltaNet and recurrent parameters remain BF16/FP32
- routers, embeddings, LM head, and norms remain BF16
- the vision tower remains BF16

The initial runtime is text-only. Omitting `model.visual.*` is expected to save
893,142,496 bytes (0.831804 GiB), leaving exactly 22,835,349,216 bytes
(21.267076 GiB) of text tensor payload. These values are regenerated from the
pinned safetensors header by `ornith35_metadata.py`.

This is a quality-preserving starting point, not an invitation to flatten the
checkpoint into a uniform four-bit format. Further quantization or expert
pruning begins only after the unmodified text model passes runtime and coding
quality gates.

### Selective Vocabulary Quantization

The BF16 embedding and untied BF16 LM head remain the numerical authority. The
default generator leaves the embedding bytes untouched but memory-maps the
verified source and copies only each requested 4 KiB BF16 row, or one bounded
prompt batch, into MLX. This removes the complete 0.9473 GiB embedding from
wired GPU allocations. Production model activity fell from 21.268 to 20.320
GiB, and the full profiler peak from 21.640 to 20.692 GiB. Three balanced
128-step full-model comparisons preserved every logit, hidden value, route,
and state bit-for-bit. Decode was 0.17%-0.74% slower; exact 128-token prefill
was 0.18% slower at an empty prefix and 0.08% slower at 4K. The source mapping
and reclaimable OS file pages still exist, so this is a wired-allocation gain,
not a smaller source artifact. `--no-mapped-embedding` restores full residency.

The default generator converts the 248,320 by 2,048 LM head to MLX affine Q8
with 32-value BF16 scale/bias groups during loading. It reduces that resident
matrix from 0.9473 to 0.5328 GiB, saving 0.4144 GiB without a derived disk
artifact. The verified BF16 head remains mapped as the numerical authority.

The candidate knee was measured on real target hidden states. Q6/64 increased
full-logit relative L2 to 1.48%, Q5/64 to 2.93%, and Q4/64 to 6.01%; those
formats were rejected. Raw Q8/32 averaged about 0.53% relative L2 and changed
9 of 4,096 source greedy choices across sixteen coding domains. Production
therefore uses Q8 only to find at least 64 candidates. It reads those exact
BF16 rows from the source mapping and applies a custom one-row-per-SIMD Metal
reduction matching the full-head arithmetic before greedy or top-20 sampling.
The 4,096-position gate recovered every greedy choice and every candidate pool
contained all source top-20 tokens; mean full-distribution KL was 1.2783e-4.

Separate 256-token greedy and seeded recommended-sampling runs produced
byte-identical output. A balanced 272-step greedy production A/B retained all
choices and all 80 persistent tensors while improving 62.666 to 64.729 tok/s
(3.29%). A balanced 136-step recommended-sampling A/B improved 61.923 to
64.825 tok/s (4.69%). Production active/peak memory is about 19.97/20.28 GiB.
`--no-quantized-lm-head` restores the fully resident BF16 head.

Replacing vision and the BF16 head with the accepted text-only hybrid payload
would produce a 22,390,359,776-byte (20.852648 GiB) runtime artifact before its
new safetensors header. That artifact is not materialized yet; the current
loader derives Q8 at startup and retains the sole verified 22.111 GiB source.

Quantizing the input embedding with the same Q8/32 format was rejected. Across
three 128-step teacher-forced trajectories it changed 5,525-6,243 routed expert
IDs out of 40,960, amplified mean logit drift to 4.27%-4.68%, and produced
greedy mismatches beginning at steps 30-55 for only the same 0.4144 GiB saving.

The earlier raw-Q8 kernel-only profile reached 68.027 tok/s; it excludes exact
candidate reranking and is retained only as an upper bound, not a production
claim.

The dependency-free CPU oracle in `ornith35_nvfp4.py` implements the exact
packed E2M1 values, E4M3FN block scales, FP32 global scale, low-nibble-first
packing, and 16-value block composition used by this checkpoint. The first
Apple boundary in `ornith35_mlx_nvfp4.py` performs the same matvec in a custom
Metal kernel inside a lazy MLX graph. In compressed-tensors storage,
`weight_global_scale` is the inverse second-level scale: dequantization is
`E2M1 * E4M3FN / weight_global_scale`. This differs from Transformer Engine's
multiplicative `s_global` notation. The convention is pinned to
compressed-tensors commit `c98cc8dd5edf60de1a3832ba378c8a2a008fb413` under
external `source-notes/compressed-tensors/`.

The complete source passed exact size, header, payload, and SHA-256 acceptance
on 2026-07-16. Four retained gate/up/down/shared projections across layers 0,
19, and 39 matched the CPU oracle at `1.09e-7` to `1.77e-7` relative L2 and at
most `2.39e-7` absolute error. The custom Metal matvec measured 29.6 to 37.0
GB/s on those isolated shapes. The source checksum is
`68a4b2b8605076825302be20132cf69342b44a0385c19e6de741af5ec3114ca0`.

The GatedDeltaNet equations are pinned to Transformers `v5.10.1` commit
`90c3ae54d448d4906b6167317ea5a7f5d48a232d`; hashes for the copied upstream
references live in external `source-notes/transformers-5.10.1/source-state.json`.
`ornith35_gdn_reference.py` is a dependency-free scalar oracle, while
`ornith35_mlx_gdn.py` implements immutable one-token MLX state transitions.
Three-token synthetic output/state parity is a mechanism check. The optimized
recurrence matches the real BF16 fallback exactly; promotion of a complete
layer still requires an independent checkpoint-derived numerical comparison.

The production GatedDeltaNet recurrence now uses a model-shape Metal kernel.
One threadgroup per value head performs the two exact 128-wide reductions and
writes only the authoritative FP32 next state and core output, avoiding the
materialized decayed-state, memory, delta, and reduction intermediates. Explicit
FP32 multiply/add boundaries prevent Metal contraction from changing state.
The generic MLX composition remains the fallback and is still used by synthetic
non-production shapes.

The full-attention counterpart follows Qwen3.5's per-head interleaved
query/gate projection layout, `(1 + weight)` Q/K RMSNorm, 64 rotary dimensions,
two KV heads repeated across sixteen query heads, FP32 softmax, and post-
attention sigmoid gating. Text tokens use the same position in all three mRoPE
axes, so the interleaving reduces exactly to the standard partial-RoPE
calculation implemented by the scalar and MLX paths. Their three-token
synthetic output and K/V-state parity is not yet a real-weight acceptance.

The MoE decode boundary keeps router softmax, sorted top-8 IDs, retained score
renormalization, selected packed expert projection, shared-expert gating, and
the final reduction in one lazy MLX graph. Exact custom Metal kernels evaluate
gate/up pairs in one dispatch. The production down kernel assigns one SIMD
group to each selected expert plus one to the shared expert, advances four
output rows per group, performs the ordered routed sum, and applies the BF16
shared gate and final add before writing output. It preserves the original
FP32 accumulation and BF16 rounding points while avoiding expert-ID readback,
duplicate input loads, the `[8, 2048]` routed-down intermediate, and a separate
shared-down dispatch. The scalar oracle decodes the actual
E2M1/E4M3FN/global-scale representation, and real full-model comparisons bind
the optimized path to that retained composition.

`ornith35_mlx_layer.py` composes these boundaries in checkpoint order: centered
input RMSNorm, GDN or gated GQA, first residual, centered post-attention
RMSNorm, routed plus shared MoE, and second residual. It propagates immutable
GDN or K/V state and router observations without host synchronization.

`ornith35_mlx_model.py` is the first complete text graph boundary. It loads
only the explicit embedding, 40 decoder layers, final norm, and untied LM head;
there is no vision field or wildcard tensor load. Its aggregate state binds the
next position to every attention cache, and each token produces the complete
248,320-entry target logit vector. A real two-token smoke loaded all 40 layers
in 14.99 seconds, held 21.267 GiB active with a 21.638 GiB peak, and produced
finite full-vocabulary logits with normalized routing and valid recurrent/K/V
state. A ten-token follow-up separated compilation from execution: after two
warmup transitions, eight full-logit tokens averaged 23.053 ms, or 43.378
tok/s. This proves the resident mechanism and a short-context performance
baseline, not target quality; tokenized coding generation and independent
logits remain required.

The durable one-token profiler separates Python graph construction, Metal
execution, and forced-synchronization component attribution. In an alternating
80-sample comparison, the original graph measured 44.026 tok/s, paired gate/up
measured 44.662 tok/s, and paired gate/up plus fused routed-down reduction
measured 46.014 tok/s: a 4.52% end-to-end target-only gain. Full logits remained
bit-identical. A separate 147-transition trajectory kept every logit, expert
route, GatedDeltaNet state, and attention cache bit-identical at the unchanged
21.638 GiB peak.

With the exact GatedDeltaNet recurrence enabled, a 200-sample alternating run
improved the already optimized fallback from 45.773 to 47.220 tok/s (3.16%).
The cumulative fixed-state gain over the original 44.026 tok/s graph is about
7.25%. A 275-transition trajectory preserved every full-vocabulary logit,
expert route, recurrent state, and K/V value bit-for-bit.

The production GatedDeltaNet convolution now shifts its four-slot BF16 state
and evaluates the depthwise FP32 dot in one Metal dispatch. Six balanced
50-round blocks all favored the fused path; the 5%-trimmed result improved
47.051 to 47.289 tok/s (0.51%). A separate 279-transition trajectory preserved
every full-vocabulary logit, route, convolution/recurrent state, and K/V value
bit-for-bit, with the peak still 21.638 GiB.

The production residual/RMSNorm boundary follows the pinned MLX 0.32.0
[`all_reduce`](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/reduction/reduce_all.h)
layout exactly: 512 threads consume four contiguous FP32 values each and use
two ordered SIMD reductions. One dispatch now writes the BF16-rounded residual
and its exact mean-square for the following centered RMSNorm. A balanced
300-sample A/B improved the fully optimized materialized path from 47.704 to
48.959 tok/s (2.63%). A 279-transition trajectory kept every full-vocabulary
logit, route, recurrent/convolution state, and K/V value bit-for-bit.

The same exact reduction dispatch can also apply MLX's precise reciprocal
square root, centered norm weight, and BF16 output boundaries before it exits.
Each decoder layer now hands that normalized output directly to the following
consumer; the last layer receives the final norm weight. This removes 80
separate normalization graphs. Against the retained mean-square-only path, all
six balanced 50-round blocks improved and the 5%-trimmed result moved 48.618 to
50.782 tok/s (4.45%). That is 15.35% above the original 44.026 tok/s graph. A
279-transition trajectory again preserved every full-vocabulary logit, route,
recurrent/convolution state, and K/V value bit-for-bit.

GatedDeltaNet now keeps each head's 128-value recurrence core in threadgroup
memory through its RMSNorm, BF16 norm weight, and stable `z` SiLU gate. A
standalone exact prototype improved the complete graph by only 0.58% and was
not retained. Integrating the work into the recurrence dispatch avoids the
global FP32 core tensor and a second launch. The original MLX composition
remains the fallback. The balanced 300-sample result improved 50.325 to 52.134
tok/s (3.60%), 18.42% above the original 44.026 tok/s graph. A 279-transition
trajectory kept all logits, routes, recurrent/convolution state, and K/V
values bit-for-bit.

One-token MoE down projection now uses a separate decode layout from batched
prefill. Applying all batched kernels to a singleton was bit-exact but 2.8%
slower, so only the useful row packing was retained. A nine-SIMD Metal group
computes top-8 routed and shared down projections for four output rows and
finishes the BF16 gated merge in the same dispatch. The isolated stage improved
from 159.36 to 142.19 us. A balanced 250-sample full-model comparison improved
54.582 to 56.826 tok/s (4.11%) with the peak unchanged at 21.638 GiB. All 162
one-token tensors and all tensors and chosen tokens across a separate 64-step
greedy trajectory remained bit-identical.

Decode now crosses the fully checked model boundary once, after prompt
prefill, and binds the accepted weights, config, and immutable rollback state
into a sealed `TextDecodeSession`. Session creation deeply verifies every
mixer, MoE, norm, and cache contract. Advancing a session returns a new session
bound to the generated state, so retained prior sessions remain valid rollback
points; only repeated invariant checks are removed from nested hotpath calls.
A balanced 250-sample comparison improved 56.271 to 57.516 tok/s (2.21%), with
host graph construction at 2.241 ms in the durable profiler and peak memory
unchanged at 21.638 GiB. Full one-token and 64-transition comparisons were
bit-identical. The checked public token API remains the fallback and session
creation rejects malformed cache position and deep weight-shape drift.

The selected and shared gate/up projections now share one eight-SIMD Metal
dispatch with their BF16 SiLU activation, eliminating separate selected,
shared, sigmoid, and multiply graphs. The fused sigmoid follows MLX 0.32's
[`Sigmoid`](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/unary_ops.h)
implementation, including a precise exponential. Exhaustive BF16 testing found
that fast Metal math changes exactly the finite input `-6.84375`, which occurred
in real layer 36; that edge now has a dedicated regression. A balanced
300-sample comparison improved the validated-session baseline from 57.176 to
60.424 tok/s (5.68%). All logits, routes, recurrent/convolution state, K/V
state, and selected tokens remained bit-identical through a separate 128-step
greedy trajectory, at the unchanged 21.638 GiB peak.

Those two fused one-token MoE kernels now decode FP4 and FP8 values through
their exact half-bit layouts instead of a lookup and dynamic `exp2`. All 16
E2M1 and 256 E4M3FN encodings, including reserved NaNs, match the scalar
authority. Batched prefill keeps its faster prior decoder. Real layer-19 MoE
improved 2.00%; a balanced full-model A/B retained all 162 tensors and improved
63.949 to 64.504 tok/s (0.87%) without memory growth. MLX's native NVFP4
gather-QMM was faster for selected prefill but remains rejected because even
its closest global-scale formulation changed 244 of 1,048,576 BF16 outputs.

GatedDeltaNet decode now carries its 8,192-row QKV projection directly through
the four-slot convolution update and FP32 SiLU. The custom projection follows
MLX 0.32's [`GEMVKernel`](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/gemv.h)
column loop and shuffle reduction, preserving the BF16 projection boundary
before the already-authoritative convolution arithmetic. A complete-model
geometry sweep selected eight SIMD groups with one output row each. The real
layer-0 stage improved from 354.13 to 311.92 us, and a balanced 240-sample
comparison improved 61.101 to 61.986 tok/s (1.45%). A separate 128-step greedy
trajectory preserved every compared logit, hidden value, route, recurrent and
convolution value, K/V value, and selected token bit-for-bit at the unchanged
21.638 GiB peak. The batched prefill projection remains separate and unchanged.

The decode recurrence now consumes the fused convolution's BF16 Q/K/V output
directly. One 32-SIMD threadgroup per value head reproduces the two 128-wide
FP32 L2-normalization trees, repeats each key/query head twice by addressing,
and retains the normalized values inside the existing recurrence/core/gate
dispatch. This removes materialized FP32 Q/K/V, two repeat graphs, and their
dispatches without changing beta or decay arithmetic. Real layer-0 mixer
latency improved from 422.31 to 377.08 us (1.12x). A balanced 160-round
full-model A/B retained all 162 tensors and improved 68.147 to 69.860 tok/s
(2.51%); a separate 128-step greedy trajectory matched all 20,736 tensor
comparisons and selected tokens. The durable profiler measured 68.351 to
69.790 tok/s, with GatedDeltaNet mixer attribution falling from 12.149 to
11.194 ms and no memory increase.

GatedDeltaNet decode also evaluates all 32 beta and decay scalars in one Metal
dispatch. The kernel reproduces MLX 0.32's stable sigmoid, compensated
`log1p(exp(-abs(x)))` softplus, precise exponential/logarithm operations, and
FP32 output boundaries exactly. A 500-batch randomized oracle matched all
16,000 beta/decay scalar pairs bit-for-bit. Real layer-0 mixer latency improved
from 387.90 to 358.54 us (1.082x). A balanced 180-round full-model A/B retained
all 162 tensors and improved 70.031 to 72.259 tok/s (3.18%); a separate
128-step greedy trajectory matched all 20,736 tensor comparisons and selected
tokens. The durable 20-sample profiler independently measured 69.812 to 72.229
tok/s, reduced synchronized GatedDeltaNet mixer cost from 11.127 to 10.235 ms,
and reported zero logit drift or memory growth.

The production GatedDeltaNet input stage now emits QKV/convolution/SiLU, z,
beta, and decay from one exact Metal dispatch. QKV and z retain MLX 0.32's
`BM=8, BN=1` one-SIMD-per-row tree. The 32-row b/a projections instead retain
MLX's small-output `BM=1, BN=8, TM=4` tree, including ordered cross-SIMD
reduction and the original BF16 rounding before transition arithmetic. This
corrects the reduction mismatch that rejected the earlier naive concatenation
attempt without adding a joined weight allocation. A 300-input real-weight
projection probe and the complete transition tests were bit-exact. Real
layer-0 improved from 350.35 to 339.01 us (1.034x). Six balanced full-model
blocks all improved, with aggregate decode moving from 71.389 to 72.056 tok/s
(0.93%); a separate 128-step trajectory matched all 20,736 tensor comparisons
and selected tokens. The independent sequential profiler was noise-limited at
71.945 versus 72.008 tok/s, but reduced synchronized GatedDeltaNet mixer cost
from 10.308 to 9.931 ms (3.66%) with zero drift and no memory growth.

Each MoE layer stores its 256-row router and one-row shared-expert gate as one
257-row BF16 allocation. One native MLX GEMV now emits both results, while the
split fallback reads exact views of the same bytes. A real-weight 100-input
check, a complete 128-token prefill, and a separate 128-step greedy trajectory
were bit-identical. Balanced decode improved from 61.887 to 62.061 tok/s; exact
128-token prefill was effectively neutral at 386.109 versus 386.287 tok/s.
Resident memory remained 21.511 GiB with a 21.640 GiB measured peak.

Decode now folds the post-mixer residual, centered RMSNorm, and that 257-row
router/shared-gate projection into one exact Metal dispatch. Seventeen
threadgroups independently reproduce MLX 0.32's 512-thread norm reduction,
keep the rounded normalized vector in threadgroup memory, and assign one SIMD
group to each router row. Only the first threadgroup writes the common residual
and normalized outputs. This preserves both original trees without a grid-wide
barrier, joined weights, or atomic reduction. A 300-input real-weight probe was
bit-exact. The complete real layer-0 path improved from 488.32 to 483.96 us.
All six balanced full-model blocks improved, moving aggregate decode from
72.492 to 73.097 tok/s (0.84%); a separate 128-step trajectory matched all
20,736 tensors and selected tokens. Independent 20-sample profiler processes
measured 72.119 to 73.411 tok/s, reduced execution from 12.304 to 12.137 ms,
and reported zero drift with unchanged 20.033/20.278 GiB active/peak memory.

Production decode now compiles each of the 30 fixed-shape GatedDeltaNet layers
as a separate weight-bound MLX graph. The ten position-dependent
full-attention layers and their K/V state remain on the established uncompiled
path. This is materially different from the rejected early whole-model compile:
the accepted exact Metal kernels now pin the reduction and rounding boundaries
inside each GatedDeltaNet layer, so MLX compilation removes host graph-building
work without changing those calculations. Session creation compiles and fully
warms all 30 graphs; `--no-compiled-gdn-layers` is the explicit fallback.

All six balanced 40-round full-model blocks improved. The 5%-trimmed aggregate
moved from 73.136 to 81.396 tok/s (+11.29%) while preserving all 162 result and
persistent-state tensors. A separate 128-step immutable greedy trajectory
matched 20,736 tensors and every selected token bit-for-bit. The production
single-owner linear K/V path passed the same 128-step/20,736-tensor gate and
improved from 72.793 to 80.470 tok/s (+10.55%). Independent 20-sample profiler
processes measured 73.476 to 81.343 tok/s, cut median host construction from
1.566 to 0.753 ms and execution from 12.091 to 11.531 ms, and retained the
20.033/20.278 GiB active/peak memory measurements.

The same session compiler now binds only the fixed-shape residual, router, MoE,
and following norm after each of the ten full-attention mixers. Attention,
RoPE, K/V append, and cache-length-dependent GQA remain on their authoritative
dynamic path. All six balanced 40-round blocks improved, with the 5%-trimmed
result moving from 81.454 to 82.993 tok/s (+1.89%) while all 162 tensors
remained identical. A separate 128-step immutable trajectory matched 20,736
tensors and every greedy token. The production linear-cache trajectory also
matched 20,736 tensors and tokens while improving from 80.044 to 82.092 tok/s
(+2.56%). Independent 20-sample processes measured 81.300 to 82.982 tok/s,
reduced median host construction from 0.744 to 0.562 ms, left execution
effectively flat at 11.532 versus 11.506 ms, and retained the 20.033/20.278 GiB
active/peak measurements. `--no-compiled-attention-tails` retains the exact
uncompiled tail.

Full-attention decode now applies centered Q/K RMSNorm, the query-gate split,
and partial text RoPE in one Metal dispatch. Each 256-wide head uses the same
32-lane, two-four-value-block reduction topology as MLX 0.32's
[`row_reduce_looped`](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/reduction/reduce_row.h),
including precise reciprocal square root and the original BF16 boundaries. An
initial two-SIMD implementation passed ordinary random tests but produced one
BF16 query difference at trajectory token 73; it was rejected, its failing
seed is now a regression, and the accepted topology passed 300 varied random
trials. The real layer-3 path improved from a 443.65 us median to 345.09 us
5%-trimmed. All six balanced 40-round full-model blocks improved, with the
trimmed result moving from 61.703 to 64.048 tok/s (3.80%). A separate 128-step
greedy trajectory compared 20,736 logit, hidden, route, recurrent,
convolution, and K/V tensors bit-for-bit. Peak memory remained 21.640 GiB.
Prefill reuses one exact RoPE table across all ten attention layers; all 162
128-token tensors matched and throughput was effectively neutral at 412.888
versus 413.278 tok/s.

Exact GQA no longer expands each two-head K/V cache to sixteen physical heads.
Queries are reshaped into two K/V groups of eight and both BF16 matmuls retain
the same scale, FP32 softmax, BF16 probability, and value-reduction sequence as
the repeated-cache fallback. Empty-cache decode remains neutral at about 64.05
tok/s, while balanced full-model decode improved from 52.020 to 60.017 tok/s
at a 4,096-token prefix (15.37%) and from 33.072 to 50.155 tok/s at 16,384
tokens (51.65%). The 16K grouped path reduced measured transient peak by about
254 MiB. All 162 tensors matched at every measured prefix from zero through
16K, and a separate 128-step greedy trajectory preserved 20,736 tensors and
every selected token bit-for-bit. Exact non-Steel continuation prefill uses
the same grouping after a measured 1,280-token prefix; shorter prefixes retain
the faster repeated path. At a 4K prefix, a 128-token continuation improved
from 315.963 to 348.531 tok/s (10.31%) with all 162 tensors unchanged.

Advancing decode can now opt into one fixed-capacity BF16 K/V allocation per
attention layer. A model-specific MLX 0.32 C++ primitive aliases those buffers,
and one Metal dispatch writes both K and V into the next unused range. The
explicit mutable `TextLinearDecodeSession` owns this path, commits eagerly, and
cannot branch or roll back; the existing immutable session remains unchanged
and is selected with `--no-linear-kv-cache`. Prefix conversion copies K/V once,
reuses every GatedDeltaNet state directly, and releases the immutable prefix
before generation continues. Capacity costs exactly 20 KiB per token across
all ten attention layers: 5 GiB at native 262K or 10 GiB at 524K.

Paired real-checkpoint measurements alternated identical advancing sessions and
compared logits, hidden values, routes, convolution/recurrent state, and active
K/V bit-for-bit, 162 tensors total. Empty-cache decode remained neutral at
64.20 versus 64.23 tok/s. At 4K it moved from 60.16 to 61.02 tok/s (+1.43%);
at 16K, from 50.56 to 53.34 (+5.51%); at 64K, from 29.50 to 35.79 (+21.33%);
and at 262K, from 11.42 to 15.53 (+36.01%). The 262K stress harness peaked at
37.70 GiB because it deliberately retained the source, immutable successor,
and linear cache together; a production native-capacity linear session adds
5 GiB to the 21.27 GiB resident model. Matched greedy production runs emitted
the same `READY` token and EOS under both cache switches.

Production prefill now creates the same linear session before token zero, so
each exact prompt chunk advances directly into its final fixed-capacity K/V
buffers. The paired Metal primitive accepts native contiguous
`[tokens, heads, width]` projections and transposes them while writing into
`[heads, capacity, width]`; it does not materialize two intermediate
transposes. This removes immutable-cache construction followed by a
linear-cache copy and therefore avoids a transient second 5 GiB K/V allocation
at native context. The immutable prefill path remains available with
`--no-linear-kv-cache`.

Paired real-checkpoint 128-token continuation tests compared final hidden
values, routes, all GatedDeltaNet state, and active K/V bit-for-bit, 161 tensors
total. Linear prefill was effectively neutral at zero and 4K prefixes, improved
0.18% at 16K, 0.67% at 64K, and 0.58% at 262K. The 262K stress harness peaked
at 42.19 GiB because it intentionally retained source, immutable, and linear
caches concurrently; production holds only the resident 21.27 GiB model plus
one 5 GiB native-capacity cache and bounded graph scratch.

Long-prefix attention now switches at a measured 106,496-token crossover to a
model-specific three-dispatch Metal path. Its score kernel preserves MLX
0.32.0's BF16 GEMV lane assignment and explicit shuffle tree, its softmax
preserves the 1,024-thread looped FP32 reduction and BF16 probability boundary,
and its value kernel reproduces the corresponding BF16 GEMVT accumulation.
This is arithmetic batching, not approximate attention: real nonzero 131K K/V
and every measured prefix through native 262K produced bit-identical output and
K/V. The implementation was derived against official MLX tag `v0.32.0`, commit
`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`.

Inside the measured 106,496 through 131,072-token band, the exact path further
fuses softmax and value reduction. A 15,360-element BF16 probability tile stays
in 30 KiB of threadgroup memory while preserving the original FP32 softmax
tree, BF16 probability boundary, and value accumulation order. Deterministic
nonzero-K/V layer tests improved 3.07% at the lower bound and 2.97% at 131K; a
40-layer A/B retained all 80 persistent tensors and improved 47.937 to 48.455
tok/s. Separate processes reduced layer-local peak scratch from 1.559 to 1.059
GiB at 131K. Chunk-64 remained favorable, while chunk-8/16 throughput fell by
18.63%/12.15% at 131K and chunk 32 was neutral. Production therefore uses
fusion only for chunks of at least 64 tokens. The selector deliberately returns
to split kernels above 131,072: realistic K/V regressed by 5.74% at 139K, and
forcing fusion at native context would save 1 GiB of scratch but lose
throughput. The final-layer one-query path also remains split because fusion
was 2.57% slower there.

Complete 128-token continuation A/Bs retained all 161 hidden, route,
GatedDeltaNet, convolution, and appended K/V tensors bit-for-bit. At 131K,
throughput improved from 36.743 to 48.172 tok/s (31.10%); at native context it
improved from 8.800 to 22.591 tok/s (156.72%). The measured 26.72 and 32.71 GiB
peaks intentionally held separate source and candidate linear caches. The
production generator holds one fixed-capacity cache, enables this selector by
default, and exposes `--no-exact-long-attention` as the authoritative fallback.

Prompt scheduling also treats the final decoder layer according to what future
tokens can actually observe. For every non-final multi-token chunk, layers
0-38 execute unchanged and layer 39 projects and appends only K/V. Its query,
attention output, residual, MoE, and final norm cannot affect persistent state
or any later layer. For the final multi-token chunk, all layer-39 K/V is still
appended, but only the last query and its residual, MoE, final norm, and logits
are evaluated. The pre-existing full-hidden chunk APIs remain unchanged as the
fallback and bitwise oracle.

Paired real-checkpoint linear sessions compared all 80 persistent tensors
bit-for-bit for the state-only path. Chunk-128 throughput improved from 389.72
to 399.65 tok/s at an empty prefix, 94.76 to 103.47 at 64K, 47.93 to 52.59 at
131K, and 22.48 to 24.39 at native 262K. The final-token path retained final
hidden, full-vocabulary logits, every route, and all state across 162 checks.
It improved from 386.47 to 396.70 tok/s cold, 95.28 to 103.38 at 64K, and
47.62 to 52.14 at 131K. A two-cache native stress remained exact and improved
18.42 to 19.19 tok/s at a 38.20 GiB peak; production owns only one cache.

`ornith35_tokenizer.py` hash-checks the pinned tokenizer, template, and
generation config before loading the standalone Rust tokenizer. The first
end-to-end prompt rendered the official no-thinking text subset, returned
exactly `OK`, and stopped on EOS at 43.70 tok/s. Thinking is the production
default. Under the model card's `temperature=0.6`, `top_p=0.95`, `top_k=20`
contract with seed 0, a bounded prime-function task reached EOS, separated its
reasoning from the final answer, and emitted correct code at 41.995 tok/s. With
the exact MoE fusions enabled, the same seeded 903-token completion remained
token-identical and reached 43.560 tok/s. The exact GatedDeltaNet recurrence
then raised it to 44.097 tok/s, 5.01% above the original 41.995 tok/s baseline.
With the exact convolution fusion also enabled, a fresh run of that same
completion reached 44.987 tok/s. Because this last figure is not a simultaneous
A/B, the controlled 0.51% measurement is the convolution speedup claim. It
used 21.638 GiB peak. Carrying the exact residual mean-square into the next norm
then reached 46.298 tok/s, and the full residual-plus-normalized-output kernel
reached 47.847 tok/s on the unchanged 903-token completion. The final result is
13.94% above the original 41.995 tok/s run, again at 21.638 GiB peak. Keeping
the GatedDeltaNet core inside recurrence then reached 49.430 tok/s, 17.70%
above the original run, without changing the generated tokens or memory peak.
The controlled speedup claims remain the separate balanced A/B measurements.
These are coherent mechanism smokes, not a coding benchmark or an independent
source-logit certificate.

## Architecture

The text model is Qwen3.5 MoE:

- 40 layers, repeating three GatedDeltaNet layers and one full-attention layer
- 30 GatedDeltaNet layers and 10 full-attention layers total
- hidden width 2,048
- 256 experts per layer, top-8 routed plus one shared expert
- expert intermediate width 512
- 16 attention query heads, 2 KV heads, head dimension 256
- 16 linear-attention key heads and 32 value heads, dimension 128
- native context 262,144 tokens

The released Ornith checkpoint declares `mtp_num_hidden_layers=1` but contains
no MTP tensors. MTP therefore remains a sidecar experiment rather than part of
the authoritative target.

## Context Profiles

The runtime exposes two explicit profiles:

| Profile | Tokens | RoPE | Status |
| --- | ---: | --- | --- |
| `native-262k` | 262,144 | checkpoint default | correctness baseline |
| `yarn2-524k` | 524,288 | static YaRN factor 2 | primary long-context goal |

The factor-2 profile retains the checkpoint's interleaved mRoPE sections,
partial rotary factor 0.25, and theta 10,000,000, while setting
`original_max_position_embeddings` to 262,144. Its cache is incompatible with
the native profile. `ornith35_context.py` reproduces the pinned Transformers
5.10.1 static-YaRN equation: rotary-dimension correction bounds `(14, 22)`,
factor-2 interpolation, and attention scaling `1 + 0.1 * ln(2)`. Text positions
share one coordinate across the three mRoPE axes, so their interleaving is
unchanged.

The selected profile is immutable across aggregate model state, all ten K/V
states, and the single-owner linear cache. It also participates in persistent
cache identity and runtime hashing. Save, restore, rollback, or continuation
across profiles fails before tensor use. MTP is deliberately native-only because
the released sidecar was neither trained nor validated with static YaRN.

Select the extended profile explicitly with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_generate.py \
  --context-profile yarn2-524k \
  --prompt 'Reply only with READY.' --no-thinking --temperature 0 --max-tokens 8
```

On the real checkpoint, matched 17-token bounded runs returned exact `READY.`
under both profiles. Native measured 231.581 prefill tok/s and 76.306 decode
tok/s; YaRN measured 241.580 and 75.061 tok/s. Both used 19.967 GiB active and
20.278 GiB peak. These runs prove complete runtime composition and absence of a
short-context penalty; they do not certify retrieval, coding quality, or cold
prefill latency near 524,288 tokens.

Only ten layers carry full K/V history. At BF16, their cache costs exactly
20,480 bytes per token before allocator overhead:

- 262,144 tokens: 5.00 GiB
- 524,288 tokens: 10.00 GiB
- 1,010,000 tokens: about 19.26 GiB

The thirty FP32 GatedDeltaNet matrix states total about 60 MiB. Convolution
state is about 1.4 MiB. An eight-bit K/V experiment would halve the dominant
cache size, but BF16 remains the reference until long-context quality proves
otherwise.

TurboQuant is a separate K/V-cache experiment. Its published LLM result applies
to online vector quantization of K/V, not to this checkpoint's already packed
NVFP4 weights. The roughly 20 GiB resident model payload and fixed-size
GatedDeltaNet state are unchanged.

The first real-model characterization now has an independent scalar oracle,
exact d=128/d=256 spherical Lloyd-Max codebooks, version-stable Gaussian
matrices, deterministic MLX QR rotations, QJL direct scoring, calibrated
channel splitting, and physical packed-byte accounting. Diagnostic inputs were
captured from the real 40-layer prefill path; all ten layers' projected BF16
keys and values matched the authoritative cache bit-for-bit before scoring.

Two calibration and two disjoint holdout prompts used 128 tokens each. Every
profile covered 2,560 causal query-head cases and 160 gated/O-projected mixer
outputs. A separate full-model state injection compressed each prefix once,
kept one recent K/V token exact, and teacher-forced four baseline-greedy steps
per holdout. The BF16 control was exact. Uniform K4-MSE/V4-MSE retained all 8/8
greedy choices, 0.938 mean top-8 vocabulary recall, 0.00208 mean logit KL, and
0.01494 maximum KL. Its aligned physical cache is 1.270 GiB at 262,144 tokens
and 2.539 GiB at 524,288 tokens, a 3.94x reduction from BF16.

QJL did not transfer well to this 256-dimensional head: at four key bits its
mean relative score error was 0.0739 versus 0.0326 for MSE, and its downstream
mixer error was also worse. The calibrated 3.5-bit MSE split costs 1.133 GiB at
native context but raised mean trajectory KL to 0.00849 for only 0.137 GiB of
additional savings. K4-MSE/V4-MSE is therefore the conservative direct-Metal
candidate; the split remains an experimental comparison.

The bounded evidence is not a production quality pass. The paper does not
specify its outlier selector, and its stated 32-at-3-bit plus 96-at-2-bit
example computes to 2.25 rather than 2.5 bits. The exact source pins are in
`ornith35/turboquant_sources.json`; the external report is
`experiments/turboquant-characterize-v1/report.json` under the model root. Its
20,699,960-byte payload has SHA-256
`991e6446e0c437eff65205402147b39e6d46af4efa8f0cffd609a4295b6226a5`.

The bounded characterization can be reproduced without persisting activation
traces:

```bash
MODEL_ROOT=/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4
PYTHONPATH=ornith35/tools "$MODEL_ROOT/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_turboquant_characterize.py \
  --tokens-per-prompt 128 --trajectory-steps 4
```

The first direct Metal implementation now stores each historical key or value
as 128 packed K4 bytes plus one BF16 norm and retains exactly one recent BF16
token. Encoding performs norm reduction, deterministic Haar rotation,
Lloyd-Max indexing, and nibble packing in one GPU dispatch. A target-specific
MLX extension appends paired packed K/V and norms into four aliased,
fixed-capacity buffers, so decode does not copy or reallocate the prefix.
Packed key scoring shares each decode across all eight GQA queries, and packed
value aggregation reduces token work across a 256-thread group before one
inverse rotation of the final 16 vectors. Neither path materializes historical
BF16 K/V.

The Metal encoder is bit-identical to the retained MLX graph authority on
contiguous and noncontiguous random BF16 vectors and on zero vectors. Direct
scores, probabilities, and attended values match materialized quantized K/V in
the focused tests. Fixed-capacity and immutable states also produce identical
results, including lazy five-step advancement. The complete repository gate,
including the rebuilt extension, passes.

On a paired synthetic one-layer M4 Max benchmark, physical K/V storage was
3.94x smaller. Including online cache maintenance, the packed path crossed BF16
near 20K tokens and measured 1.161x at 32K, 1.405x at 131K, and 1.437x at 262K.
At native context, packed attention took 3.415 ms versus 5.086 ms for BF16 and
used 130.001 MiB versus 512.000 MiB for one full-attention layer. The packed
encoder/append measured 0.251 ms versus 0.181 ms for BF16 append in that run.
These remain component throughput results; full long-context generation is not
yet measured.

The packed state is now integrated through the production 40-layer model and
generator. `--turboquant-kv` retains exact BF16 chunked prefill, converts its
immutable or fixed-capacity active prefix once on GPU, and then owns all ten
attention histories in packed fixed-capacity buffers during decode. Restored
packed prefixes resume serially without reconstructing historical BF16 K/V.
The feature is explicitly opt-in, native-profile-only, and incompatible with
MTP and system-prefix warming while their combined semantics remain untested.

Packed persistent entries use the separate
`ornith35-prefix-state-turboquant-k4-v1` schema. They compact active U8 payloads
and BF16 norms, retain one exact BF16 tail, preserve all GatedDeltaNet state,
and verify provenance, metadata, shape, byte count, and SHA-256 before restore.
A real 128-token gate persisted 65,590,947 bytes, restored every active packed,
tail, convolution, and recurrent tensor exactly, and resumed packed decode.
The direct production trajectory retained 16/16 greedy choices with 0.984375
mean top-8 recall, 0.00557997 mean KL, and 0.042777 maximum KL. At this short
prefix packed decode was 13.922 ms versus 13.123 ms BF16, consistent with the
measured roughly 20K crossover. A real CLI smoke generated `READY`, saved its
packed cache in 0.321 seconds, and peaked at 20.278 GiB. Native long-context
coding/retrieval quality and full-model speed remain the acceptance gate.

The first two disjoint 20,480-token full-model gates confirm the component
crossover. Security/coding context improved from 17.400 to 16.443 ms per decode
step (1.0582x), while story/retrieval context improved from 17.364 to 16.438 ms
(1.0563x). The packed histories occupied 101.661 MiB rather than 400 MiB BF16.
Both retained 15/16 teacher-forced greedy choices: one changed a decision whose
source margin was only 0.125 by the same 0.125, and one resolved an exact source
tie differently. Mean KL was 0.00827 and 0.00634. These bounded results justify
the opt-in path but do not replace broader coding, sampled, needle-retrieval,
and longer-prefix gates.

Run the real runtime, persistence, and bounded quality check with:

```bash
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_turboquant_runtime_gate.py \
  --prompt-tokens 128 --steps 16 --chunk 128
```

Reproduce the bounded crossover benchmark with:

```bash
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_turboquant_cache_bench.py \
  --tokens 8192,16384,20480,32768,65536,131072,262144 \
  --warmup 8 --rounds 40
```

The experiment retains BF16 as the authority and compares a calibrated 3.5-bit
candidate with conservative uniform and asymmetric K/V profiles. Compressed cache
identity binds the quantizer algorithm and version, precision allocation,
rotation seed and codebooks, packed layout, RoPE profile, and tail policy. The
Metal path must score packed keys and aggregate packed values directly; full
BF16 materialization is not an acceptable performance implementation. Short
contexts remain on BF16 until a measured crossover, and acceptance requires
long-context logits, generation, coding, retrieval, persistent restore,
memory, prefill, and decode evidence.

Primary references:

- <https://arxiv.org/abs/2504.19874>
- <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>

## Prefill Design

A prompt transition now stops at the final normalized hidden state unless its
logits are actually needed. Generation projects the 248,320-way LM head only
for the last prompt token, while preserving the complete recurrent and K/V
state for every token. Eight real-model transitions matched the full-logit
path bit-for-bit. Across six alternating warm 23-token A/B pairs, this exact
change raised token-serial prefill from 52.427 to 58.658 tok/s (11.88%) with
bit-identical final logits and the same 21.638 GiB peak. This removes known
waste; it is not a substitute for sequence-parallel prefill.

The NVFP4 MoE path now also has token-batched Metal kernels for shared
gate/up/down projections and each token's selected top-8 gate/up/down work.
Expert IDs remain GPU-owned, and the routed-down kernel retains the decode
path's per-expert BF16 rounding and ordered reduction. Synthetic comparisons
are bit-exact. Real layers 0, 19, and 39 preserve outputs, selected experts,
and routing weights bit-for-bit across eight-token batches. On layer 19, a
128-token chunk improved isolated MoE throughput from 6,642 to 16,626
token-layers/s (2.50x), while 256 tokens improved 6,757 to 16,966 (2.51x).
This primitive is ready for chunk composition; it does not by itself change
the current token-serial frontend.

GatedDeltaNet prefill has an exact chunk path. Its dense BF16 projections now
use a token-tiled Metal reduction: one SIMD group retains eight independent
token accumulators and loads each four-column weight group once, while every
token keeps MLX's authoritative column and shuffle order. The tiny `b`/`a`
projections remain on MLX because custom dispatch was neutral. A Metal
convolution kernel walks each channel's four-slot history, and a head-parallel
kernel advances FP32 recurrence in token order. Real layer 18 improved from
4.075 to 2.805 ms. A complete exact 128-token state-prefill A/B preserved all
80 tensors and improved 398.991 to 452.429 tok/s; the observable final path
preserved all 162 checks and improved 395.661 to 446.704 tok/s. Production
chunk gains rise from 7.71% at eight tokens to 13.11% at 128, with no resident
weight or cache allocation added. Decode keeps its separate one-token kernels.

The same exact token tiling covers chunked full-attention Q/K/V and output
projections. It activates only for BF16 chunks of at least eight tokens;
one-token final-query and decode projections retain their tuned MLX paths.
Real layer 19 improved 1.31x at an empty prefix and 1.25x after 1,024 tokens.
Incrementally over GatedDeltaNet tiling, complete 128-token state prefill kept
all 80 tensors unchanged and improved 453.495 to 468.467 tok/s. The observable
final path preserved all 162 checks and reached 463.828 tok/s. Smaller scheduler
chunks also improve, and the paired peak remains 20.717 GiB.

MoE prefill now reuses its BF16 router and packed shared-expert weights across
prompt tokens as well. Eight router tokens share each exact BF16 reduction;
four shared gate/up/down tokens share each packed E2M1 decode and FP8 block
scale while retaining every token's original block, pair, SIMD reduction, and
BF16 boundary. Routed experts remain on their existing GPU-owned top-8 path.
Real layer 19 improved from 4.402 to 4.196 ms. A balanced 24-round complete
128-token comparison preserved all 80 persistent tensors and improved 462.890
to 477.203 tok/s (3.09%) at the unchanged 20.717 GiB peak. Decode does not use
the batched kernels and is unchanged.

Batched NVFP4 prefill also consumes the model's BF16 activations directly.
Metal converts each loaded value to the same FP32 operand used by the retained
kernel, eliminating four materialized FP32 activation copies per MoE layer and
halving their read width without changing accumulation. Real layer 19 improved
from 4.258 to 4.150 ms (1.026x). Incrementally over token tiling, a balanced
24-round complete-model A/B preserved all 80 state tensors and improved
470.442 to 478.455 tok/s (1.70%) at the unchanged 20.717 GiB peak. The same
observable final-token path preserved all 162 logits, hidden, route, and state
checks while improving 480.330 to 488.445 tok/s (1.69%). The same
representation change was exact but neutral for one-token decode (68.903 versus
68.944 tok/s), so decode retains its prior FP32 input materialization.

Full-attention chunks fuse exact centered Q/K RMSNorm, partial RoPE, and query
gate splitting in one production-shape Metal dispatch per token/head grid. A
key-only variant covers the unobservable final-layer cache path. Both retain
the prior BF16 reduction and arithmetic boundaries bit-for-bit. Real layer 19
improved from 4.305 to 4.102 ms (4.95%); complete state-only and final-token
prefill improved by 0.47% and 0.57% across 80 and 162 exact checks,
respectively, without a resident allocation.

Full-attention prefill now uses MLX 0.32's native Steel scaled-dot-product
attention without expanding Ornith's two K/V heads to its sixteen query heads.
The lower-right causal mask gives each chunk query the retained prefix and only
its causal chunk positions. `vmap` keeps Q/K/V projection reductions aligned
with decode, and vectorized FP32 RoPE keeps the resulting BF16 K/V cache
bit-exact. Steel's fused attention output is numerically, not universally
bitwise, equivalent: a real 1,024-prefix/128-token continuation measured
0.001953 maximum absolute and 2.83e-4 relative L2 while retaining exact K/V.
Real layer 39 at chunk 256 improved from 56.238 to 5.371 ms (10.47x). Complete
model gating then rejected broad Steel activation: chunks 32-128 amplified the
local difference to 4.48-5.48% final-logit relative L2 and changed downstream
routes. Steel remains an isolated experiment for selective-layer or higher-
precision recovery; the production prefill path explicitly disables it.

The accepted model-level path batches exact GatedDeltaNet and MoE work while
retaining token-authoritative attention score, softmax, and value reductions.
Q/K/V and output projections remain independent token GEMVs under `vmap`, Q/K
normalization and RoPE are batched, and each layer appends and head-repeats K/V
once per chunk. Real attention layers 3, 19, and 39 matched serial decode
bit-for-bit at zero- and 1,024-token prefixes. Layer 39 chunk 128 improved from
27.080 to 4.905 ms (5.52x) at an empty prefix and from 38.724 to 5.131 ms
(7.55x) after a 1,024-token prefix. This is the production exact path; Steel
remains disabled.

Token-indexed Metal residual/RMSNorm threadgroups preserve decode's FP32
reduction and BF16 rounding. Router logits preserve those token-wise GEMV
results, with weight-reusing exact token tiling for chunks of at least eight;
their independent
256-way FP32 softmax, sort, and retained top-8 normalization now use native
batched row dispatches. Random real-weight inputs across every layer and a
complete 128-token model transition matched the former token-wise route
bit-for-bit across 162 final, route, recurrent, and K/V tensors. This reduced
real layer 19 from 9.602 to 7.745 ms and raised warm exact 128-token prefill
from 152.327 to 166.480 tok/s (9.29%) at a 21.732 GiB peak.

The scheduler uses only compiled
power-of-two chunks up to 128 and a serial tail, and projects the vocabulary
only after the final segment. Real 25-, 128-, and 259-token runs preserved
every final logit and all GDN/KV state bit-for-bit. At 128 tokens, warm prefill
first improved from 57.580 to 152.594 tok/s (2.65x). A multi-chunk 259-token run used
`(128,128,1,1,1)` and improved 56.868 to 147.331 tok/s (2.59x) at a 21.808 GiB
peak. The first cold 25-token CLI run, including new chunk-kernel compilation,
reached 79.293 tok/s and then decoded at 50.828 tok/s.

With exact batched attention composition, a full 128-token transition retained
all 162 compared tensors bit-for-bit and reached 240.973 tok/s, 44.87% above
the already batched-router path. The 259-token multi-chunk schedule retained
all 82 final/state tensors and reached 231.064 tok/s at a 21.749 GiB peak.

The batched NVFP4 kernels subsequently pack more independent output rows into
each threadgroup and reuse token loads across two or four rows per SIMD group.
The routed-down dispatch covers sixteen output rows per 1,024-thread group
while retaining one SIMD group per selected-expert reduction. Each row keeps
the same lane assignment, accumulation sequence, BF16 rounding, and ordered
top-8 sum. Real layer 19 MoE improved from 7.777 to 4.479 ms (1.74x). A full
128-token comparison retained all 162 tensors bit-for-bit and improved from
240.298 to 314.459 tok/s (30.86%) at a 21.751 GiB peak. The exact 259-token
schedule reached 299.260 tok/s at 21.748 GiB.

GatedDeltaNet recurrence now caches each lane's four FP32 state elements and
uses 32 SIMD groups to cover all 128 value columns. For chunk prefill, each
independent recurrent column remains in registers through the token sequence;
the kernel writes final state once and materializes exact FP32 cores for a
second, numerically identical RMSNorm/gate kernel. Real layers 0, 18, and 38
match serial output and state bit-for-bit. The isolated recurrence improved
from 3.413 to 1.154 ms (2.96x), and layer 38 improved from 6.548 to 4.056 ms
(1.61x). Full 128-token prefill retained all 162 compared tensors and reached
384.555 tok/s at a 21.757 GiB peak; the 259-token schedule reached 362.168
tok/s. Decode uses the same cached 32-group arithmetic and improved from
52.840 to 53.877 tok/s with complete bitwise parity.

The final scheduler comparison retains 128 as a measured cap, not an assumed
one. A direct 256-token chunk and two 128-token chunks matched all 162 compared
tensors bit-for-bit, but the two bounded chunks reached 385.335 tok/s versus
375.361 tok/s for 256. The larger graph also raised peak memory to 21.903 GiB.
The scheduler therefore keeps power-of-two chunks through 128 plus a serial
tail; 512 was not tested after 256 lost on both speed and memory. The CLI and
unit contract reject larger requested chunks.

A cold 524K prefill is not expected to be interactive. The ten causal
full-attention layers alone require approximately 22.5 PFLOPs for QK and AV.
The practical coding design avoids paying that cost repeatedly:

1. Keep system instructions, tool schemas, and a stable repository snapshot at
   the front of the token stream.
2. Save exact content-addressed cache checkpoints at bounded token intervals.
3. Append diffs, conversation, and the current request after the stable prefix.
4. Resume from the longest exact prefix and process only the new suffix.
5. Build or refresh workspace caches in the background and retain them with a
   disk-aware LRU policy.

An exact checkpoint includes all ten layers of target K/V, all GatedDeltaNet
matrix and convolution states, exact token IDs, next position, and complete
model, runtime, tokenizer, template, RoPE, and dtype provenance. Cache schema
v2 can also include suffix-independent MTP state: target state is stored at
position `N`, MTP K/V through `N-1`, and the final authoritative target hidden
row. It never stores the sampled pending token. Restore appends that boundary
row with the first uncached suffix token, exactly reconstructing the shifted
MTP stream for any continuation.

Checkpoints use one safetensors file per target layer, optional compact
`mtp-prefix.safetensors`, canonical token bytes, and a strict manifest. MTP and
target-only identities cannot collide; the MTP policy hash binds the pinned
sidecar, selected folded adapter, their state files, exact MTP config, and
runtime bytes. Linear K/V is compacted one layer at a time, so native-context
persistence does not allocate a second complete cache. Every file is hashed,
shape/metadata checked, fsynced, and atomically published; restore verifies
every durable byte before exposing immutable state.

The generator can automatically content-address an exact rendered system
segment with `--cache-system-prefix`, restore it before model-state allocation,
or warm it with the state-only prefill path on a miss. `--save-cache` persists
the complete prompt for an explicitly resumed longer prompt. Cache-writing
runs apply a protected 24 GiB, 64-entry LRU by default; set `--cache-max-gib`
or `--cache-root` explicitly when disk requirements differ. Native and YaRN
entries cannot collide because the authoritative RoPE profile is part of their
identity and is checked against restored model state.

A real 128-token linear-cache round trip occupied 67,525,182 bytes, saved in
0.330 seconds, and restored in 0.098 seconds. All 80 persistent tensors and the
next token's logits plus successor state matched bit-for-bit, with a 20.587 GiB
peak versus 20.571 GiB active memory. Substantial-prefix TTFT sweeps and generic
longest repository-prefix discovery remain forward work.

A real MTP-enabled 12-token system-prefix checkpoint occupied 0.061 GiB; its
draft payload was 26,970 bytes and represented 11 K/V positions plus one 2,048
element BF16 boundary row. Verified restore took 0.062 seconds. Warm and
restored greedy runs both emitted exact `READY`, accepted 2/2 future tokens,
and peaked at 22.355 GiB. Synthetic split-prefill tests additionally preserve
target logits/state and complete MTP context bit-for-bit after durable restore.

The remaining prefill work targets profiled full-model bottlenecks, exact
prefix persistence/restoration, incremental suffix timing, and any fusion that
can preserve the established BF16 boundaries. The production scheduler is
already bounded at 128 tokens, and direct linear K/V writes are active.

Chunking improves memory and scheduling but does not change full attention's
quadratic arithmetic.

## Decode Acceleration

Two draft sources are pinned:

- DSpark bootstrap revision
  `9383b3c33ddf982114a4f72e07c890bfd6c35df2`, one 1.5434 GiB BF16 draft
- Qwen3.5 MTP bootstrap revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754`

The DSpark draft is one 1,657,168,394-byte (1.543358 GiB) BF16 file. The Qwen
source contains 785 required `mtp.*` tensors with 1,689,281,536 bytes
(1.573266 GiB) of useful payload. Those tensors span source shards 13 and 14,
whose complete files total 7,592,604,208 bytes (7.071164 GiB). Extraction must
therefore be resumable and atomic: download one pinned shard, verify its full
SHA-256, copy only indexed MTP ranges into a sidecar, verify the sidecar, and
remove the raw shard only after durable state records success.

`ornith35_mtp_extract.py` now implements that contract. It validates the exact
metadata and raw header inventory, creates a deterministic 1,689,376,064-byte
(1.573354 GiB) safetensors sidecar, and binds atomic progress to the source and
repository revisions plus metadata, tool, tensor-range, and output hashes.
Source identity must remain unchanged from full-file hashing through copy and
deletion. A resumed run re-reads every completed output range before accepting
another shard and safely recovers interruption before or after final rename.

The approved extraction is complete. `source-mtp/mtp.safetensors` is
1,689,376,064 bytes with a 94,520-byte header, 1,689,281,536 payload bytes,
785 BF16 tensors, and SHA-256
`11c9043bf0c92c1eea7b4c6ffbadeb890a080a84d301872a29839be209099c1f`.
`source-mtp-state.json` binds that output to both verified source shards and
every copied tensor hash. The transient raw shards and dedicated Hugging Face
cache were removed after acceptance.

The pinned vLLM `v0.24.0` implementation also establishes the bootstrap MTP
equation: use the target embedding table, independently RMS-normalize the
target hidden state and current-token embedding, concatenate them to width
4,096, apply the BF16 `mtp.fc` projection to width 2,048, execute one Qwen3.5
full-attention decoder layer, apply the MTP final norm, and project with the
normal target LM head. The bootstrap source has no dedicated embedding tensor.
This is an initialization contract, not evidence that its draft distribution
matches the post-trained and abliterated Ornith target.

`ornith35_mtp_reference.py` and `ornith35_mlx_mtp.py` now implement that
contract independently. The runtime uses the target embedding and head without
copying them into the sidecar, strict-loads all 785 BF16 tensors, supports
one-token and causal batched MTP state advancement, and keeps routing on Metal.
The target alignment follows pinned vLLM behavior: target hidden rows are not
shifted, following token IDs are shifted left, and the last target hidden row
pairs with the pending authoritative token. After verification, only committed
target hidden rows and tokens rebuild MTP K/V; speculative hidden state is
never allowed to become authoritative.

The exact verifier uncovered several real numerical defects rather than an MTP
alignment defect. MLX compilation first fused the batched BF16 MoE SiLU product
and removed an eager rounding boundary. Later resident trajectories exposed a
different batched top-8 routing reduction, non-bit-exact small-chunk GDN rollback,
and a compiled generic RMSNorm reduction that crossed BF16 boundaries. Dedicated
BF16 Metal SiLU and centered width-2,048 RMSNorm kernels, independent small-route
normalization, and serial accepted-prefix GDN replay now define one target
contract. At target position 60, compiled and uncompiled block-3 paths match all
41 captured boundaries, all expert routes, final hidden state, and next token
bit-for-bit. Every MTP benchmark block is also checked against an independently
advanced serial target trajectory; all pre-fix timing is discarded.

The repaired public bootstrap is mechanically correct but not a general accelerator.
On a 32-block Rust LRU prompt, it generated 77 serial-identical target tokens,
accepted 44/64 future tokens, and measured 79.910 versus 79.433 tok/s at
21.669/23.142 GiB active/peak. On a 32-block C++ concurrent-queue prompt, it
generated 68 serial-identical tokens but accepted only 35/64 and measured
68.070 versus 78.812 tok/s. Block two also lost on that prompt. A coherent
source-BF16-head mode is available for validation, but its 78.801 tok/s absolute
MTP rate and 22.081 GiB active footprint do not beat the hybrid Q8/32 mode.
The public Qwen MTP bootstrap therefore remains disabled by default.

Target-derived distillation is now accepted. A provenance-bound capture contains
32 coding prompts, 5,044 target rows, and 4,096 scored future positions. Training
updates only a rank-32 LoRA over `mtp.fc`; the selected seed-29 update is folded
offline into one BF16 2,048-by-4,096 replacement, so inference adds no adapter
operations. Its 16,777,533-byte artifact is
`experiments/mtp-distill-coding-v1/adapter-r32-e8-s29-v2/mtp-fc.safetensors`
with SHA-256
`a42cf411862e04485124bcefc96a5092abd7f2e25b4eb663ac4b703b8e4bbc11`.
On prompt-disjoint teacher traces it improves candidate acceptance from
734/896 (81.92%) to 761/896 (84.93%) and hidden relative L2 from 0.88169 to
0.78947. Rank 64, a second loss weighting, and seed interpolation did not beat
this folded candidate. Block three is retained: block two cannot amortize its
draft cost, while blocks four and five lose throughput as acceptance decays.

Fixed MTP remains unsafe because yield changes within one response. The exact
adaptive controller observes accepted future positions for at least eight
blocks, then falls back permanently to the already-owned target cache when the
most recent four-block acceptance falls below 70%. The target-only path performs
no MTP forward or reconciliation. At repository revision
`2f541a4199af770f5eb1f853570412f7dd509266`, seven prompt-disjoint coding runs
were token-identical to independent serial generation. They produced 414 target
transitions at 82.873 tok/s versus 77.511 tok/s target-only, or 1.069x, with
23.377 GiB peak memory. Per-prompt speedups were 1.076, 1.027, 1.033, 1.149,
1.065, 1.041, and 1.055. The accepted adapter state binds the exact gate log at
SHA-256 `f266cdaf59f1ac92c72002830af3dce558af9f81ff6006a38b90c6bda9362e31`.

That seven-prompt result did not generalize to every generation mode. The
revision-bound draft-regime gate uses a separate 12-prompt, 51-68-token coding
corpus with no rendered-prompt overlap against the 32-prompt teacher capture.
It replays every emitted trajectory through a fresh target session and requires
hidden, logits, all recurrent state, and active K/V to match bit-for-bit. At
revision `193cfb086fa3fb7de32ecfebadb2cbf7a62c6348`, all 36 sampled MTP runs were
exact but reached 74.150 versus 78.127 tok/s, or 0.9340x steady. All 12 greedy
runs were exact and reached 77.140 versus 77.968 tok/s, or 0.9778x steady.
The useful conditional result was greedy thinking mode: eight runs reached
81.439 versus 77.976 tok/s, 1.0323x steady, with 85.05% future acceptance and
a 0.9661x worst-prompt floor. Greedy non-thinking mode was only 0.8385x steady,
and sampled thinking mode remained below parity at 0.9726x. A four-block early
detachment sweep suppressed later-recovering prompts and is rejected; the
8-block minimum, 4-block window, and 70% threshold remain authoritative.

Run the exact acceptance benchmark with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_mtp_bench.py \
  --root "$ORNITH35_MODEL_DIR" --steps 32 --block-tokens 3 \
  --adaptation-dir \
    "$ORNITH35_MODEL_DIR/experiments/mtp-distill-coding-v1/adapter-r32-e8-s29-v2" \
  --adaptive-fallback
```

Run the resumable, path-matched regime gate with:

```sh
RESULTS="$ORNITH35_MODEL_DIR/experiments/draft-regime-v1"
mkdir -p "$RESULTS"
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_draft_regime_gate.py \
  --root "$ORNITH35_MODEL_DIR" --engine mtp-greedy \
  --output "$RESULTS/mtp-greedy.json"
```

Use `mtp-sampled` or `dspark-greedy` for the other independently loaded engine;
each invocation commits one small atomic JSON record after every trajectory and
resumes only when all source, runtime, corpus, policy, and tool identities match.

The public DSpark draft is directly matched to a byte-identical rehost of the
selected AEON target, but its published acceptance is only preliminary. It is
a mechanism bootstrap, not a production speed claim. Qwen MTP tensors may be
used to initialize an Ornith sidecar, but the accepted target-specific fold
above is required because Ornith post-training and abliteration changed the
target distribution.

The released DSpark metadata must be interpreted with its training-era
semantics. Its 2026-07-01 config omits `sample_from_anchor`, explicitly requests
seven speculative tokens, and uses block size eight. Upstream added
`sample_from_anchor` on 2026-07-13 in commit `a0be7bb`; applying that newer
default retroactively is wrong. The released draft therefore uses the legacy
layout: slot zero is the known anchor and is excluded from loss, while slots
one through seven predict seven speculative tokens.

The implementation contract was checked against Speculators commit
`4b25261ab792a6cbe1a15d17a5c110e9fb27f71b`, immediately before the public
checkpoint, and vLLM `v0.24.0` commit
`ee0da84ab9e04ac7610e28580af62c365e898389`. vLLM auxiliary-state index zero
is the embedding output; index `N` is the unnormalized residual output after
decoder layer `N-1`. The released IDs `9,19,29` therefore mean target decoder
layers `8,18,28`, not `9,19,29`. Their three BF16 2,048-wide outputs are
concatenated in that order, projected by `fc.weight` from 6,144 to 2,048, then
passed through the draft's standard Qwen3 RMSNorm.

The same pinned Speculators vocabulary-mapping implementation establishes that
`d2t` is an offset table, not a direct target-token table. Draft index `i` maps
to target token `i + d2t[i]`. In the released payload, those 32,000
reconstructed IDs are strictly increasing and exactly equal the true entries
of the 248,320-wide `t2d` mask. Interpreting the raw offsets as token IDs
produces 15,961 adjacent duplicates and is invalid. Both scalar and MLX paths
enforce the reconstructed inverse relation before proposal.

The nested draft config inherited `partial_rotary_factor=0.25` from the Qwen3.5
target, but that field did not control the model used to train this checkpoint.
Training instantiated Transformers' Qwen3 rotary class, which rotates the full
256-dimensional draft head. Applying vLLM's later generic partial-RoPE path
would rotate only 64 dimensions and silently change the released model. The
Ornith runtime therefore pins full-head NeoX RoPE as the authoritative public
checkpoint equation; a 64-dimensional compatibility mode is not enabled.

`ornith35_dspark.py` now enforces this legacy contract and the complete public
weight schema before a payload can be accepted. The draft has 44 tensors:
42 BF16 tensors containing 828,329,729 parameters, one 32,000-entry I64
draft-to-target map, and one 248,320-entry boolean target-vocabulary mask. It
contains three dense Qwen3 full-attention layers, a 32K draft head, a rank-256
vanilla Markov bias, and a confidence head over the 2,048 draft features plus
the 256 Markov features. The indexed payload is exactly 1,657,163,778 bytes.

The target model exposes opt-in decode and prefill capture for those vLLM
indices, including the optimized single-owner linear session. Capture retains
only requested layer outputs and leaves the normal result, persistent state,
and compiled target path unchanged. Synthetic Metal tests compare every
captured row with explicit decoder-layer composition; this establishes the
target side of the interface independently of the accepted public draft.

The independent draft implementation now consists of a dependency-free scalar
oracle, a strict MLX BF16 loader, and a Metal-backed composition path. It caches
each draft layer's K/V projection of the accepted target prefix, runs the three
noncausal Qwen3 draft layers over one anchor plus seven mask slots, applies the
32K head, then resolves each slot sequentially through the rank-256 Markov
correction before mapping back to the 248,320-token target vocabulary. The
confidence head uses the matching draft hidden row and previous-target-token
Markov embedding. No proposal-loop token ID is read back to Python.

Both production caches now use fixed-capacity, aliased BF16 Metal storage.
Target verification owns one mutable target session and can roll back only its
logical position plus the journaled GatedDeltaNet states. The draft context has
its own lock-protected generation and position owner; attempting to reuse an
older context fails before proposal or append work. At the native 262,144-token
limit, the ten target full-attention layers require exactly 5.0 GiB of K/V and
the three draft layers require exactly 1.5 GiB, for a 6.5 GiB combined cache
budget. The released draft is natively limited to 262,144 positions; DSpark
remains disabled for the separate 524,288 YaRN profile until its RoPE behavior
is independently quality-gated.

`ornith35_mlx_dspark_runtime.py` binds that draft state to the exact target
verifier cursor. Target verification captures the three requested residual
streams in the same pass as its compact GatedDeltaNet rollback journal. A full
acceptance appends every committed row; a mismatch appends only the accepted
prefix; an anchor mismatch appends nothing. Repeated synthetic target/draft
steps preserve `draft_context.position == target_cursor.state.position`, and
the scalar and MLX implementations agree on projected context, every K/V row,
hidden states, base/corrected logits, confidence, and selected IDs. Full and
partial target commits are bit-identical between immutable and fixed-capacity
K/V, and stale target or draft sessions are rejected.

The real target path remains exact with capture enabled. A four-block advancing
fixed-cache run reproduced 33 serial greedy tokens and the matching immutable
block schedule across all 82 target observables. With the exact BF16 block head,
block-8 verification measured 29.309 ms for the first block and 29.042 ms steady,
with 21.223 GiB active and 21.293 GiB peak memory. The BF16 block head accounts
for about 0.947 GiB but avoids repeated hybrid-Q8 candidate projection and is
the recommended speculative configuration. These measurements cover the
production target interface only and remain the target-verifier baseline.

The public draft payload is now downloaded separately, accepted by exact file
size and full SHA-256, and strict-loaded as 44 tensors in 0.238 seconds. A
64-step coding-prompt run reconstructed 89 emitted target tokens exactly and
matched a separate serial greedy target trajectory. It accepted 24 of 448
future proposals, or 0.375 future tokens per block; no block accepted all seven.
Standalone draft proposal measured 4.176 ms, complete steady steps measured
about 37.7 ms, and the run used 22.604 GiB active and 22.674 GiB peak memory.
Its 35.211 tok/s end-to-end rate was only 0.450x the fairly measured 78.166
tok/s serial target rate. Mean confidence was 0.270160, close to the published
validation mean of 0.288678, which supports the equation and auxiliary-state
alignment but does not establish broad language quality.

The retained low-acceptance verifier removes two independent costs. Exact
block-head decisions are reduced once for the complete block and materialized
with target state in one synchronization; this reduced isolated all-accepted
block-8 verification from 29.083 to 27.409 ms (5.76%). A configurable causal
stage then stops verification as soon as a proposal fails. Stage width one
uses the normal optimized target decode path, so it requires no rollback replay,
compiled prefill tail, or 0.947 GiB exact block head. Every forced mismatch and
full acceptance matched the unstaged verifier's emitted tokens, target state,
logits, and draft context exactly in tests. On the same real 64-step trajectory,
stage width one retained the exact 89-token serial output and raised DSpark to
59.382 tok/s while reducing active/peak memory to 21.657/21.723 GiB. That is a
68.6% gain over the initial public-draft path, but still only 0.755x its paired
78.686 tok/s target baseline.

The acceptance distribution explains the remaining loss: 46/64 blocks accepted
no future token, 14 accepted one, two accepted two, and two accepted three; no
block reached position four. First-slot confidence was not a useful scheduler:
accepted blocks averaged 0.414686 versus 0.391888 for rejected blocks, with
high-confidence failures and low-confidence successes. The released preview is
therefore an exact mechanism bootstrap, not a production accelerator on this
M4 Max coding trajectory. A stronger target-specific draft is required;
target-specific draft training remains open.

The broader gate closes that measurement question for the released draft. On
12 disjoint coding prompts, staged DSpark reproduced 12/12 target trajectories
and all target state exactly, but accepted only 35/672 future proposals (5.208%).
It reached 58.745 versus 77.855 tok/s, or 0.7555x steady, lost on every prompt,
and peaked at 21.776 GiB. DSpark remains a verified reference implementation
and is not part of normal generation.

The measured production-shape target command is:

```sh
$ORNITH35_MODEL_DIR/mlx-env/bin/python \
  ornith35/tools/ornith35_mlx_speculative_bench.py \
  --root "$ORNITH35_MODEL_DIR" \
  --proposal-tokens 8 --trajectory-blocks 4 \
  --linear-target-cache --capture-dspark-aux --exact-bf16-block-head
```

Run the accepted public draft against the same target and an independently
advanced serial greedy baseline with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_dspark_bench.py \
  --root "$ORNITH35_MODEL_DIR" --steps 64 --draft-rounds 5 --log-every 8 \
  --target-stage-tokens 1
```

MTP and DSpark are initially competing drafters. Both must use block target
verification with exact recurrent-state snapshot and rollback. At long
context, a block verifier should reuse K/V tiles across proposal positions;
otherwise verification remains dominated by long-cache traffic.

The target-side greedy verifier is now implemented independently in
`ornith35_mlx_speculative.py`. Its cursor explicitly records the consumed target
state and the still-unconsumed token predicted by that state, preventing anchor
and bonus-token alignment errors. A rejected first token performs no target
forward. Later rejection evaluates one causal target block, truncates immutable
attention K/V directly or rolls back the fixed cache's logical cursor, and
reconstructs only the 30 GatedDeltaNet states from captured normalized inputs;
it never replays attention or MoE layers.

Production block-8 testing retained the generator's single-token Q8 LM-head
reduction order. All 80 persistent tensors plus hidden and logits matched a
separate accepted-prefix target evaluation bit-for-bit at every forced mismatch
position. Eight consecutive blocks reproduced 65 compiled-target greedy tokens,
including every bonus token. At a 20.154 GiB active and 20.278 GiB peak
footprint, eight serial transitions took 97.016 ms versus 41.433 ms for one
all-accepted exact verification (2.342x less target work). Forced mismatch
positions one through seven took 43.614 through 47.938 ms with exact compact
rollback. These are target-only ceilings; DSpark proposal cost and its measured
1.695 mean accepted length still need an end-to-end gate.

The fixed block-8 hotpath now preserves that authority while reducing target
work further. Forty compiled tails bind only the numerically safe graph after
each token mixer: post-mixer residual/RMSNorm, MoE, second residual, and the
next layer's RMSNorm. Compiling a complete GatedDeltaNet mixer was rejected
because recurrent state ceased to be bit-exact. The retained BF16 block-head
kernel evaluates up to eight independent vocabulary reductions while reading
each head weight once; the cursor still uses the normal one-token Q8/32
projection. This adds 0.944 GiB but reduced the verifier from 39.241 to about
31.2 ms after compiled tails had already improved 41.581 to 39.241 ms.

Bounded Metal kernels then remove the remaining small-block intermediates.
GatedDeltaNet directly joins QKV/Z projection, convolution, SiLU, B/A
transition, beta/decay, normalized convolved recurrence, column-resident state,
core normalization, and the z gate without changing the established FP32
reduction or BF16 rounding boundaries. Full attention joins its block-8 Q/K/V
projection dispatches while retaining separate source tensors and outputs.
The final real-checkpoint run reproduced 65 target tokens exactly and matched
all 82 cursor/state tensors at every forced mismatch. Eight serial transitions
took 96.737 ms versus 29.331 ms for all-accepted verification, a 3.298x target
speedup and 306.839 emitted-token/s target-only ceiling. Forced mismatches took
32.180 through 34.402 ms at 21.098/21.105 GiB active/peak memory. Reusing
experts across proposal positions was rejected: exact tile-2 and full-block
kernels slowed verification to 43.27 and 44.96 ms.

Reproduce the verifier, prefix-length sweep, forced rollback checks, and exact
greedy trajectory with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_speculative_bench.py \
  --proposal-tokens 8 --rounds 3 --sweep-prefixes --trajectory-blocks 8 \
  --exact-bf16-block-head --capture-dspark-aux
```

## Storage

External root:

`/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`

Planned children:

- `metadata/`: target metadata and header-only inventory
- `metadata-dspark/`: DSpark metadata and header-only inventory
- `metadata-mtp-source/`: Qwen MTP metadata and shard map
- `source-nvfp4/`: immutable target source after approval
- `source-dspark/`: immutable, hash-verified DSpark source after approval
- `source-mtp-raw/`: transient current Qwen source shard during extraction
- `source-mtp/`: separately verified MTP sidecar
- `runtime-text/`: future text-only runtime artifact
- `cache/`: provenance-bound workspace prompt caches
- `quality/`: logits and coding reports
- `logs/`: human-readable operation logs

The target source is present and immutable under `source-nvfp4/`. The separately
accepted public draft is present under `source-dspark/`, bound by
`source-dspark-state.json` to its repository, revision, size, and full SHA-256.
Derived runtime artifacts must remain in sibling directories and must not
modify or replace either accepted source.

The minimal Apple environment is reproducible with:

```sh
python3 -m venv "$ORNITH35_MODEL_DIR/mlx-env"
"$ORNITH35_MODEL_DIR/mlx-env/bin/pip" install \
  -r ornith35/requirements-mlx.txt
"$ORNITH35_MODEL_DIR/mlx-env/bin/pip" install --no-deps \
  -r ornith35/requirements-tokenizer.txt
```

The separate `--no-deps` tokenizer install avoids pulling a networking stack
into a runtime that only reads the already verified local `tokenizer.json`.

Build the pinned MLX/Metal extension after Xcode's Metal toolchain is present:

```sh
ornith35/build_extensions.sh
```

After the target download completes, accept it with:

```sh
python3 ornith35/tools/ornith35_source_verify.py
```

The verifier compares the complete file size, raw safetensors header, payload
decomposition, and full SHA-256 against the pinned metadata. It reports hash
throughput at 1 GiB intervals and writes `source-nvfp4-state.json` atomically
only after every check passes.

Accept or reverify the separately pinned DSpark source with:

```sh
python3 ornith35/tools/ornith35_source_verify.py --profile dspark
```

The DSpark profile reads only `metadata-dspark/` and
`source-dspark/model.safetensors`, pins the public companion repository,
revision, 1,657,168,394-byte file size, and full SHA-256, then atomically writes
`source-dspark-state.json`. It cannot silently accept the target checkpoint or
metadata from another companion.

Inspect the validated MTP transfer and disk plan without downloading anything:

```sh
ornith35/run_mtp_extract_stream.sh --plan
```

After explicit approval for the two weight shards, run the resumable stream:

```sh
ornith35/run_mtp_extract_stream.sh
```

The launcher uses `hf download` with one worker and Xet disabled. It keeps no
prefetched second shard: each source is downloaded into `source-mtp-raw/`,
fully hashed, copied and read back, committed to `source-mtp-state.json`, then
deleted before the next download. Hugging Face local-directory metadata and the
default dedicated cache are removed only after verified progress; interrupted
`.incomplete` data remains in the raw directory so `hf download` can resume it.
All hub, assets, and Xet cache paths are forced below that same raw directory,
and the launcher stops if post-download cache allocation exceeds 64 MiB rather
than silently retaining a second payload.
The current two-shard contract has a conservative 7,057,215,608-byte
(6.572544 GiB) peak including a fully allocated sidecar.

Run the bounded resident source smoke with explicit token IDs using:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_model.py \
  --tokens 248044,9707,9707,9707,9707,9707,9707,9707,9707,9707
```

The first two transitions are compilation/warmup and are reported separately
from the post-warmup aggregate. This command is a full-graph mechanism smoke,
not a language-quality evaluation.

Run a chat-formatted target generation with thinking and the recommended
sampling defaults using:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_generate.py \
  --prompt 'Write a Python function is_prime(n: int) -> bool.' \
  --max-tokens 1024 --seed 0
```

Thinking is enabled unless `--no-thinking` is passed. Use `--temperature 0`
for exact greedy diagnostics. Generated text and code remain untrusted and are
never executed by this command.

The generator uses exact mapped BF16 embeddings and the single-owner linear K/V
cache by default. Pass `--no-mapped-embedding` for full embedding residency or
`--no-linear-kv-cache` for the immutable rollback-capable comparison path.
The hybrid Q8/32 plus exact BF16 candidate rerank is enabled by default. Pass
`--no-quantized-lm-head` for the fully resident BF16 authority. The 30
fixed-shape GatedDeltaNet layers are compiled and warmed by default; pass
`--no-compiled-gdn-layers` for the exact uncompiled session path. The fixed
tails after all ten attention mixers are also compiled by default; pass
`--no-compiled-attention-tails` to retain their exact uncompiled path.

Enable the accepted target-derived MTP sidecar with `--mtp --temperature 0`.
Normal generation selects it only when thinking is enabled and the rendered
prompt has at most 256 tokens. This is the sole prompt-disjoint measured regime;
target-only remains the default. The generator
selects the folded rank-32 seed-29 adapter under the model root, streams prompt
hidden rows directly into a fixed-capacity BF16 MTP cache, and never retains a
prompt-sized hidden-state copy. Greedy proposals remain target-token exact. For
positive temperature, proposals are sampled from the MTP distribution `q`,
accepted with `min(1, p(x)/q(x))`, and replaced after rejection from normalized
`max(p-q, 0)`. This reproduces the target top-k/top-p distribution exactly; the
sidecar never makes an authoritative token or state decision. The sampled path
is retained for correctness experiments but is skipped by the normal measured
selector because its broad gate was slower than target-only.

The short-prefix evidence began with narrow wins. At 32 prompt
tokens, greedy MTP accepted 22/22 future tokens and improved 76.796 to 85.530
tok/s with byte-identical output and a 23.063 GiB peak. At 35 prompt tokens,
recommended sampling accepted 40/46 futures and improved 77.639 to 86.458
tok/s at a 23.112 GiB peak. MTP attention and context construction lose on
longer measured prefixes: unlimited runs at 269, 1,057, and 5,204 prompt tokens
reached 71.713, 56.335, and 46.814 tok/s versus target-only 77.630, 76.571, and
70.887 tok/s. Therefore `--mtp` automatically remains target-only above 256
prompt tokens, before loading or prefilling the sidecar. The protected
269-token run reached 78.452 tok/s and 20.278 GiB peak. The broader gate above
also rejects sampled and non-thinking MTP. Use `--mtp-max-prompt-tokens 0` only
to override all three guards for experimental length, sampling, or thinking
runs.

Persistent schema-v2 caches support effective short-prefix MTP for
`--load-cache`, `--save-cache`, and `--cache-system-prefix`. They retain MTP K/V
through the penultimate token plus the final target hidden row, so no random or
greedy pending-token choice enters the reusable prefix. The cache identity
binds the exact sidecar and folded adapter; target-only and MTP caches are never
interchanged. A prompt outside the measured MTP regime follows the normal
target-only cache path. MTP remains opt-in; the broad gate does not justify
making it the default.

Warm and automatically reuse an exact system prefix with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_generate.py \
  --system "$SYSTEM_PROMPT" --prompt "$USER_PROMPT" \
  --cache-system-prefix --cache-max-gib 24
```

The first invocation atomically warms the exact system segment; later prompts
with the same system, model/runtime, tokenizer/template, policy, and RoPE
identity restore it automatically. Use `--save-cache` to retain the whole
rendered prompt and `--load-cache PATH` to resume it explicitly. Verify the
real-checkpoint persistence path with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_cache_bench.py --tokens 128
```

Run the paired long-prefix benchmark with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_linear_cache_bench.py \
  --prefixes 0,4096,16384,65536,262000 --warmup 8 --rounds 40
```

Run the paired exact continuation-prefill benchmark with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_linear_prefill_bench.py \
  --prefixes 0,4096,16384,65536 --chunk 128 --warmup 2 --rounds 6
```

Run the exact long-attention arithmetic and crossover regression with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_long_attention_bench.py \
  --prefixes 106496,131072,262016 --chunk 128 --warmup 1 --rounds 4
```

Add `--nonzero-cache --prefixes 131072` for the deterministic nonzero-K/V
quality case.

Run the fused softmax/value selector regression with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_long_attention_bench.py \
  --feature fused-softmax-value --nonzero-cache \
  --prefixes 106496,131072,139264 --chunk 128 --warmup 2 --rounds 6
```

Run the paired state-only and final-token prompt benchmarks with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_state_prefill_bench.py \
  --prefixes 0,4096,65536,131072,262016 \
  --chunk 128 --warmup 1 --rounds 4

PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_final_prefill_bench.py \
  --prefixes 0,4096,65536,131072 \
  --chunk 128 --warmup 1 --rounds 4
```

Run the substantial Q8 candidate-recall/greedy gate and balanced production
rerank timing with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_vocab_quality.py --steps 256

PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_vocab_rerank_bench.py \
  --warmup 16 --rounds 256
```

Profile exact state-only prompt chunks without creating a Metal trace with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_prefill_profile.py \
  --root "$ORNITH35_MODEL_DIR" --chunk 128 --repeats 5
```

`prefill-profile-target` is the unfenced production graph. Component timings
deliberately synchronize after each layer stage and are only for hotspot
ranking. The profiler uses independent linear K/V buffers and rejects any
state mismatch across all 80 persistent tensors. The initial exact run reached
398.714 tok/s; synchronized cost was dominated by MoE (170.068 ms),
GatedDeltaNet mixers (120.763 ms), and full-attention mixers (38.223 ms).

Reproduce the paired real-layer token-tiled BF16 projection gate with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_dense_bench.py \
  --root "$ORNITH35_MODEL_DIR" --layer 18 --tokens 128 \
  --warmup 3 --rounds 40
```

Reproduce the corresponding full-attention projection gate with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_attention_dense_bench.py \
  --root "$ORNITH35_MODEL_DIR" --layer 19 --prefix 0 --tokens 128 \
  --warmup 3 --rounds 40

PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_attention_dense_bench.py \
  --root "$ORNITH35_MODEL_DIR" --feature fused-qk-rope \
  --layer 19 --prefix 0 --tokens 128 --warmup 4 --rounds 48
```

Reproduce the paired real-layer token-tiled MoE gate with:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_moe_dense_bench.py \
  --root "$ORNITH35_MODEL_DIR" --layer 19 --tokens 128 \
  --warmup 4 --rounds 64

PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_moe_dense_bench.py \
  --root "$ORNITH35_MODEL_DIR" --feature direct-bf16 \
  --layer 19 --tokens 128 --warmup 4 --rounds 64
```

Profile the complete target graph with both exact MoE optimizations using:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_profile.py \
  --root "$ORNITH35_MODEL_DIR" --repeats 10
```

Pass `--no-fused-residual-mean-square`, `--no-fused-residual-rmsnorm`,
`--no-fused-postnorm-router`,
`--no-fused-gdn-convolution`, `--no-fused-gdn-recurrence`,
`--no-fused-gdn-core-gate`, `--no-fused-gdn-recurrence-inputs`,
`--no-fused-gdn-beta-decay`,
`--no-fused-gdn-input-transition`,
`--no-paired-moe-gate-up`, and
`--no-fused-moe-routed-down` together for the retained numerical/performance
fallback. Add `--no-compiled-gdn-layers` to measure that fallback without the
fixed-shape GatedDeltaNet session compiler, and add
`--no-compiled-attention-tails` to disable the corresponding attention tails.
Passing only
`--no-fused-residual-rmsnorm` selects the exact mean-square-only path. Component
timings deliberately force synchronization and are for hotspot ranking; only
`profile-target` is the production validated-session end-to-end timing. A
`.gputrace` capture can
duplicate roughly the full resident weight allocation, so use `--capture` only
with more than 23 GiB of disposable disk headroom and remove the trace after
analysis.

## Bootstrap Evidence

The metadata-only bootstrap fetched pinned configs, tokenizer assets, API
manifests, and bounded safetensors headers. It did not transfer tensor payloads.
Strict validation proves:

- 93,346 target tensors occupy one contiguous 23,728,491,712-byte payload
- all 40 layers contain routed experts 0 through 255
- all 92,520 routed/shared MLP NVFP4 packed weights, FP8 scales, and FP32 global
  scales have the expected names, dtypes, and shapes
- every non-MLP tensor is BF16 and the released target contains no `mtp.*`
  tensors
- the DSpark draft has 42 BF16 tensors plus one integer and one boolean map,
  44 tensors total
- all 785 Qwen MTP tensors resolve to, and are present in, the exact shards
  named by the pinned weight index

The generated catalogs and source state live under the external metadata
directories. `ornith35/check.sh` reruns synthetic corruption tests and the
strict real-header catalog whenever those metadata files are present.

## Retired Nemotron Storage

The 188 GiB external Nemotron tree was removed on 2026-07-16 after confirming
that `nemotron-main` was clean and pushed. A 33 MiB provenance archive remains
at `/Users/nir/dev/models/nemotron-provenance-20260716.tgz`, SHA-256
`a52565250327e59c57e2c641274bd21dd1de746541934ab535dd46ce4afc800c`.
The cleanup raised free disk space from 64 GiB to 252 GiB.

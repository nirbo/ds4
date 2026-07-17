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
gate/up pairs in one dispatch and fuse every selected down projection with its
ordered routing reduction. They preserve the original FP32 accumulation and
BF16 rounding points while avoiding expert-ID readback, duplicate input loads,
and the `[8, 2048]` routed-down intermediate. The scalar oracle decodes the
actual E2M1/E4M3FN/global-scale representation; synthetic parity still requires
a real layer and full-logit comparison before promotion.

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

The runtime will expose two explicit profiles:

| Profile | Tokens | RoPE | Status |
| --- | ---: | --- | --- |
| `native-262k` | 262,144 | checkpoint default | correctness baseline |
| `yarn2-524k` | 524,288 | static YaRN factor 2 | primary long-context goal |

The factor-2 profile retains the checkpoint's interleaved mRoPE sections,
partial rotary factor 0.25, and theta 10,000,000, while setting
`original_max_position_embeddings` to 262,144. Its cache is incompatible with
the native profile.

Only ten layers carry full K/V history. At BF16, their cache costs exactly
20,480 bytes per token before allocator overhead:

- 262,144 tokens: 5.00 GiB
- 524,288 tokens: 10.00 GiB
- 1,010,000 tokens: about 19.26 GiB

The thirty FP32 GatedDeltaNet matrix states total about 60 MiB. Convolution
state is about 1.4 MiB. An eight-bit K/V experiment would halve the dominant
cache size, but BF16 remains the reference until long-context quality proves
otherwise.

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

GatedDeltaNet prefill now has an exact chunk path as well. MLX `vmap` batches
the dense projections without changing the one-token BF16 reduction, a Metal
convolution kernel walks each channel's four-slot history, and a single
head-parallel Metal dispatch advances the FP32 recurrent state through the
chunk in token order. The recurrence remains sequential where the mathematics
requires it, but projection and head work is parallel and repeated Python and
dispatch boundaries are removed. Isolated kernels and real layers 0, 18, and
38 preserve output, convolution state, and recurrent state bit-for-bit. Chunk
128 improved real layer 0 by 3.55x; chunk 256 improved layer 38 by 3.63x to
19,728 token-layers/s. Full-model use still awaits the chunked attention and
layer composition boundaries.

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

Token-indexed Metal
residual/RMSNorm threadgroups preserve decode's FP32 reduction and BF16
rounding, and per-token `vmap` preserves dense, router, and shared-gate
reductions. Router logits retain those token-wise GEMVs, but their independent
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

An exact checkpoint includes all ten layers of K/V, all GatedDeltaNet matrix
and convolution states, exact token IDs, next position, and complete model,
runtime, tokenizer, template, RoPE, and dtype provenance. Checkpoints are
written atomically and validated before replacing an older state.

The remaining prefill work will target:

- fused RMSNorm, QKV, RoPE, and K/V writes
- bounded chunk scheduling that avoids giant lazy graphs and GPU watchdog risk

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

The public DSpark draft is directly matched to a byte-identical rehost of the
selected AEON target, but its published acceptance is only preliminary. It is
a mechanism bootstrap, not a production speed claim. Qwen MTP tensors may be
used to initialize an Ornith sidecar, but target-specific distillation is
expected because Ornith post-training and abliteration changed the target
distribution.

MTP and DSpark are initially competing drafters. Both must use block target
verification with exact recurrent-state snapshot and rollback. At long
context, a block verifier should reuse K/V tiles across proposal positions;
otherwise verification remains dominated by long-cache traffic.

## Storage

External root:

`/Users/nir/dev/models/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`

Planned children:

- `metadata/`: target metadata and header-only inventory
- `metadata-dspark/`: DSpark metadata and header-only inventory
- `metadata-mtp-source/`: Qwen MTP metadata and shard map
- `source-nvfp4/`: immutable target source after approval
- `runtime-text/`: future text-only runtime artifact
- `cache/`: provenance-bound workspace prompt caches
- `quality/`: logits and coding reports
- `logs/`: human-readable operation logs

The target source is now present and immutable under `source-nvfp4/`; no draft
weights have been downloaded. Derived runtime artifacts must remain in sibling
directories and must not modify or replace this sole accepted source.

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

After the target download completes, accept it with:

```sh
python3 ornith35/tools/ornith35_source_verify.py
```

The verifier compares the complete file size, raw safetensors header, payload
decomposition, and full SHA-256 against the pinned metadata. It reports hash
throughput at 1 GiB intervals and writes `source-nvfp4-state.json` atomically
only after every check passes.

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

Profile the complete target graph with both exact MoE optimizations using:

```sh
PYTHONPATH=ornith35/tools \
  "$ORNITH35_MODEL_DIR/mlx-env/bin/python" \
  ornith35/tools/ornith35_mlx_profile.py \
  --root "$ORNITH35_MODEL_DIR" --repeats 10
```

Pass `--no-fused-residual-mean-square`, `--no-fused-residual-rmsnorm`,
`--no-fused-gdn-convolution`, `--no-fused-gdn-recurrence`,
`--no-fused-gdn-core-gate`, `--no-paired-moe-gate-up`, and
`--no-fused-moe-routed-down` together for the retained numerical/performance
fallback. Passing only
`--no-fused-residual-rmsnorm` selects the exact mean-square-only path. Component
timings deliberately force synchronization and are for hotspot ranking; only
`profile-target` is the production end-to-end timing. A `.gputrace` capture can
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

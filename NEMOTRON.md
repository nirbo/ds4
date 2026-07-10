# Nemotron 3 Super Project Notes

This branch is the integration branch for a narrow local compression and
inference path targeting
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`.

Feature work branches from `nemotron-main` under `feature/nemotron-*`, is tested
there, and merges back only after correctness, numerical integrity, storage,
and relevant performance checks pass.

DS4 and Ornith remain independent references. Useful infrastructure may be
copied into Nemotron-specific files and adapted, but this target must not depend
on either model's implementation.

## Why This Target

Nemotron 3 Super is substantially more tractable than Ornith as a first
consumer-hardware compression target:

- 120B total parameters and about 12B active parameters per token
- hybrid NemotronH architecture with Mamba2, LatentMoE, and full attention
- official QAT mixed-precision NVFP4 checkpoint
- official NVFP4 quality reported close to the BF16 model on several coding and
  reasoning evaluations
- routed experts account for most checkpoint storage, giving structural pruning
  meaningful leverage

The first objective is a high-quality artifact in the 54-60 GiB range that can
run with useful context and runtime headroom on a 64 GB Mac. A 32 GB RTX 5090
target will initially require host-memory offload; fitting the complete model in
32 GB is not a quality-safe first milestone.

## Official Checkpoint Baseline

Current indexed sizes from the official repositories:

| Source | Indexed bytes | GiB | Shards | Intended use |
| --- | ---: | ---: | ---: | --- |
| NVFP4 | 80,297,329,824 | 74.78 | 17 | Primary prune-only baseline |
| FP8 | 128,334,890,312 | 119.52 | 26 | Higher-precision diagnostics |
| BF16 | 247,222,108,160 | 230.24 | 50 | Reference and new quantization research |

These are tensor payload sizes, not peak download-space requirements. Temporary
files, filesystem allocation, Hugging Face metadata, and accidental duplicate
caches must be considered before fetching weights.

The index also contains 1,040 `mtp.*` tensors. The config declares one
next-token prediction layer with MTP pattern `*E`. The metadata catalog keeps
these separate from backbone experts. We may eventually omit MTP from a
non-speculative runtime artifact, but only after verifying the official forward
path and measuring its exact byte contribution; pruning code must not mistake
MTP experts for backbone experts.

The primary model directory is:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

The initial metadata snapshot is pinned to Hugging Face revision
`4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6` and stored under `metadata/` in
that directory. `source-manifest.json` records the source identity and SHA-256
hashes of the downloaded metadata files.

No weight download starts without explicit approval. Small metadata may be
stored there to pin architecture, shard maps, revisions, and hashes.

### Verified Local Source

The complete official NVFP4 snapshot was downloaded on 2026-07-10 to:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/source-nvfp4
```

It is pinned to revision `4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6`.
Hugging Face CLI `1.21.0` verified all 36 repository files with both missing
and extra files treated as failures. The source contains 17 safetensors shards,
80,317,948,856 bytes of safetensors files including headers, and
80,365,683,780 bytes across all repository files. Its config and index match
the separately hashed metadata snapshot byte-for-byte.

The 152 KiB `.cache/huggingface` local-dir metadata was removed after the
strict verification because the verifier otherwise reports its own metadata as
extra files. The source tree is now immutable input. Future jobs must write to
separate directories and must not modify or delete source shards.

The durable local verification record is:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/source-nvfp4-state.json
```

## Compression Strategy

NVIDIA's NVFP4 checkpoint is already a mixed-precision, QAT-produced artifact.
The initial path therefore avoids dequantizing and requantizing it.

### Phase 1: Exact Structural Pruning

1. Pin and catalog the official NVFP4 checkpoint revision.
2. Decode enough ModelOpt metadata to validate every quantized tensor and scale.
3. Collect activation-aware routing and expert-output statistics from the
   unmodified checkpoint.
4. Generate conservative 5%, 10%, 20%, 25%, 30%, and 35% expert-pruning plans.
5. Materialize candidates by slicing routed-expert tensors, router rows, and
   router correction bias together.
6. Prove that every retained payload and scale is byte-identical to the source.
7. Compare logits and downstream coding quality against the official NVFP4
   baseline.

For LatentMoE, routing frequency alone is insufficient. The observer should at
minimum retain route counts, routing weights, activation maxima, and a REAP-like
route-weighted expert-output norm measured in the latent expert space before the
output projection. The exact equation must be verified against NVIDIA's model
implementation before it becomes a quality gate.

### Size Leverage

The validated shard headers attribute 59.0628 GiB to backbone routed experts
and 5.4805 GiB to optional MTP tensors. Every backbone expert occupies exactly
3,096,592 bytes, and its BF16 router-weight row plus F32 correction-bias value
occupies 8,196 bytes. Exact uniform payload projections are:

| Removed | Retained/layer | With MTP | Base runtime without MTP |
| ---: | ---: | ---: | ---: |
| 0% | 512 | 74.78 GiB | 69.30 GiB |
| 10% | 461 | 68.88 GiB | 63.40 GiB |
| 15% | 435 | 65.88 GiB | 60.40 GiB |
| 20% | 410 | 62.99 GiB | 57.50 GiB |
| 25% | 384 | 59.98 GiB | 54.50 GiB |
| 30% | 358 | 56.97 GiB | 51.49 GiB |
| 35% | 333 | 54.08 GiB | 48.60 GiB |

NVIDIA's ordinary causal-model class ignores `mtp.*` checkpoint keys; they are
consumed by runtimes implementing MTP speculative decoding. A base-runtime
artifact can therefore omit them without changing ordinary autoregressive
logits, at the cost of losing MTP acceleration. The initial Mac candidates are
15% and 20% pruning without MTP: 15% is the quality-first boundary, while 20%
leaves materially safer runtime headroom. These projections do not establish
acceptable pruning quality; activation-aware evaluation still decides the cut.

### Phase 2: Optional Quantization Research

Only after a prune-only quality point is established should another format be
considered. NVFP4-to-low-bit conversion is double quantization and is not the
preferred source for format research. A serious new quantization recipe should
start from BF16, retain sensitive Mamba state, routing, normalization, latent
projection, embedding, and output tensors at measured precision, and use
activation-aware per-tensor or per-expert evidence.

## Infrastructure Plan

The reusable ideas from Ornith are operational rather than architectural:

- one shard processing while one successor downloads
- bounded disk use with explicit cache placement
- atomic output validation before raw deletion
- durable state transitions and exact resumption
- immutable source and policy identity
- human-readable stdout plus durable logs
- metadata-only size planning
- calibration coverage and quality reports

They will be copied and rewritten under `nemotron_*` names. Ornith `.ornq`,
Qwen3.5 layouts, quantization policies, REAP observations, and runtime kernels
are not valid inputs for Nemotron.

## Initial Milestones

1. Metadata-only model catalog and architecture assertions.
2. ModelOpt NVFP4 tensor/scale parser with small synthetic fixtures.
3. Resumable download and exact-copy validation smoke on one approved shard.
4. Exact structural-pruning materializer and metadata remapping tests.
5. Official NVFP4 baseline runner and LatentMoE observer.
6. Conservative prune-only candidate ladder and quality evaluation.
7. Model-specific Metal runtime, followed by CUDA/offload work where useful.

Runtime optimization starts only after a candidate can be loaded and compared
against a trusted implementation. The final hot path should minimize CPU/GPU
boundaries, retain recurrent state on the device, and measure full-token decode
rather than isolated matrix kernels.

## Current Tools

Run the current repository and pinned-metadata checks:

```sh
./nemotron/check.sh
```

`nemotron/tools/nemotron_metadata.py` verifies the immutable metadata hashes,
NemotronH architecture, hybrid layer pattern, shard sequence, ModelOpt mixed
precision, complete eight-object NVFP4 expert groups, and matching router
tensors. It writes the derived catalog atomically to the external metadata
directory. Set `NEMOTRON_MODEL_DIR` when using a different local storage root.

`nemotron/tools/nemotron_safetensors_inventory.py` reads only the 17 shard
headers. It validates all tensor offsets, shapes, dtypes, payload sizes, index
ownership, routed-expert groups, and router leading dimensions. Its exact byte
inventory separates optional MTP storage and emits uniform pruning projections
without loading weight payloads.

`nemotron/tools/nemotron_prune_materialize.py` applies a revision-bound expert
plan without dequantizing any tensor. It renames retained expert tensors,
slices BF16 router rows and F32 correction bias in matching order, updates both
ModelOpt metadata maps, optionally omits MTP, validates every output shard with
an ordered payload SHA-256, and records resumable state atomically. Use
`--dry-run` before allocating output and `--max-shards N` for bounded smokes.

A real-shard smoke used the public count-based keep-90 remap only as a tooling
fixture, not as a production quality plan. The exact dry run projected 17
source shards to 15 output shards, 148,500 tensors, and 63.40 GiB without MTP.
Source shard 1 materialized to a 4,658,552,008-byte output in about four seconds,
including an output reread and ordered payload SHA-256. Re-running skipped the
verified shard and advanced to the next source shard, proving state-driven
resumption. The large smoke output was deleted; its log and state remain at:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/metadata/materializer-smoke-20260710
```

The real dry run also established that omitting MTP removes two entire source
shards. The materializer renumbers the remaining files to a contiguous output
sequence and validates the regenerated index. Full finalization tests additionally
require identical ModelOpt target sets across config groups, `quantized_layers`,
and `hf_quant_config.json`; the public keep-90 artifact leaves one of those maps
stale, which this path does not permit.

`nemotron/tools/nemotron_nvfp4.py` is the dependency-free scalar oracle for the
native ModelOpt representation. It mmaps safetensors, validates the packed U8,
E4M3 block-scale, and F32 global-scale relationship, decodes even columns from
the low nibble and odd columns from the high nibble, and provides a reference
matvec row. Production kernels must agree with this oracle before performance
results are accepted.

### NVFP4 Decode Baseline

The reference equation, confirmed against NVIDIA ModelOpt and vLLM source, is:

```text
weight[row, column] = E2M1(nibble) * E4M3(weight_scale[row, column/16])
                      * weight_scale_2
```

Two E2M1 values share each U8 along the logical input dimension: even columns
use the low nibble and odd columns use the high nibble. E2M1 magnitudes are
`0, 0.5, 1, 1.5, 2, 3, 4, 6`; bit 3 is the sign. `weight_scale` is nonnegative
E4M3FN and `weight_scale_2` is the FP32 tensor-wide global scale. The checkpoint
also stores activation `input_scale`, but vLLM's W4A16 ModelOpt path explicitly
discards it and computes with higher-precision activations, which is the Apple
runtime baseline.

Pinned source-only references:

- NVIDIA ModelOpt commit `d69d5aab8bcc7f905d39f96953621286bc2533be`
- vLLM commit `95ed0feaa5cd7fb16d72c53ce04950aaf07c4698`
- MLX-LM commit `a790972f0f844d81067ed45c28b524220a10c019`
- oMLX commit `6342b4d9c0dce296366f061cee066aeea16305dc`

They live outside the repository under the model directory's `source-notes/`.

### Metal NVFP4 Baseline

`nemotron/nemotron_metal.m` and `metal/nemotron_nvfp4.metal` implement the first
native Apple path directly over packed U8 weights and raw E4M3 scales. No
expanded weight tensor or CPU dequantization is created. The kernel assigns one
SIMD group per output row, decodes each 16-value block scale once, and uses
`simd_sum` for accumulation.

The synthetic test covers non-multiple-of-eight row counts, every E2M1 nibble,
subnormal/normal E4M3 scales, and mixed signs. The real layer-1 expert
`up_proj` (`2688x1024`) agrees with the scalar oracle at relative L2
`1.28e-7`, maximum absolute error `1.49e-7`. Three 100-dispatch samples on the
M4 Max measured `0.0355-0.0360 ms` per projection, or `43.4-44.0 GB/s` over
weights, scales, input, and output. This is the single-expert baseline, not the
final MoE design; selected experts must be batched and gate/up/activation/down
must be fused to remove thousands of token-level dispatches.

### MLX Composition Baseline

`nemotron/tools/nemotron_mlx_nvfp4.py` exposes the same packed kernel through
`mx.fast.metal_kernel`, so subsequent model composition can remain lazy and
GPU-owned. It consumes packed U8 weights, raw E4M3 block scales, the FP32
global scale, and FP32 activations directly. The synthetic check agrees with
the scalar oracle, and the real layer-1 expert `up_proj` measured relative L2
`1.31e-7` with maximum absolute error `1.44e-7`.

The isolated environment currently pins MLX `0.31.2`, MLX-LM `0.31.3` from
commit `a790972f0f844d81067ed45c28b524220a10c019`, and Transformers `5.13.0`.
The published MLX-LM `0.31.3` wheel contains a broken tokenizer registration
that passes `"NewlineTokenizer"` as a string; the pinned upstream source passes
the class object and imports correctly. Do not restore the broken wheel over
the pinned source build.

MLX also provides a native `nvfp4` `quantized_matmul`/`gather_qmm`. ModelOpt's
extra tensor-wide `weight_scale_2` can be folded into each expert activation
without changing the represented weight. A real 22-expert layer-1 probe using
batched native up projection, ReLU-squared, down projection, and score mixing
measured about `0.16 ms` per queued MoE call. This is an architectural result,
not a final token benchmark: the production path should use selected-expert
batched operations and compare them with a fused Nemotron-specific kernel.

### oMLX Findings

The pinned oMLX source is useful reference material in four distinct areas:

1. oQe measures normalized output sensitivity and collects activation-imatrix
   evidence. It tracks MoE expert coverage explicitly and uses a broad coding,
   reasoning, tool-use, and multilingual calibration corpus. Our pruning
   observer should likewise reject an under-covered plan rather than infer that
   unseen experts are unimportant.
2. Its mixed-precision planner gives mandatory protection to routers, state
   parameters, and output-sensitive tensors. Routed experts stay at the base
   bit width because boosting all experts is byte-expensive. For this official
   QAT checkpoint, exact NVFP4 retention remains safer than oQ-style
   requantization, but the sensitivity and imatrix collection strategy applies.
3. Its MTP implementation verifies multiple draft positions in one backbone
   call, keeps recurrent/KV rollback exact, and uses verify-shape Metal kernels.
   Nemotron's 5.48 GiB MTP head should remain an optional sidecar until ordinary
   decode is correct; later, acceptance rate and net throughput decide whether
   a compressed MTP sidecar earns its memory.
4. Continuous batching and paged SSD KV caching improve concurrent service and
   repeated long-prefix time-to-first-token. They do not reduce base model
   weight residency or single-stream weight bandwidth, so they are later
   serving features rather than current fit/decode blockers.

The immediate leverage order is therefore: complete exact model execution in
MLX, collect full-model routing/output sensitivity with expert-coverage gates,
materialize a conservative prune-only candidate, establish quality, then add
MTP and long-context cache tiering.

### GPU-Owned LatentMoE Baseline

`nemotron/tools/nemotron_mlx_moe.py` implements the complete routed-expert
subpath for Nemotron's latent MoE. Router indices remain on GPU; each selected
expert's ModelOpt global scale is folded into its activation; MLX
`gather_qmm` reads only selected packed expert matrices; ReLU-squared, down
projection, score weighting, and expert reduction remain in the lazy MLX graph.

The synthetic test dequantizes both up and down matrices independently and
checks the complete selected-expert equation. A real layer-1 check over all 512
resident experts and top-k 22 produced exact agreement between gathered and
explicitly selected qmm paths. On the M4 Max it measured approximately
`0.19-0.23 s` to stack/evaluate the layer and `0.21 ms` per warm MoE call.
Materializing the stacks from the official per-expert layout used `1.48 GiB`
active memory and `2.96 GiB` peak memory. The production runtime artifact must
therefore store each layer's expert weights/scales pre-stacked on disk; stacking
all 40 layers at model load would violate the 64 GB memory target.

### Direct MLX Runtime Packing

`nemotron/tools/nemotron_mlx_pack.py` converts the immutable NVIDIA layout
directly into a model-specific MLX runtime layout. It can apply a revision-bound
expert plan during the same pass, slices router rows and correction bias in the
same retained order, optionally omits MTP, and groups the result into one global
shard plus one shard per backbone layer. Expert tensors become stacked U8
packed weights, raw E4M3 block scales, and per-expert FP32 global scales. Every
payload segment is copied exactly and each completed group is reread and hashed.
Resumption verifies completed payload hashes before advancing.

This direct path is required for disk and memory safety. It avoids both a
57-60 GiB intermediate Hugging Face-style pruned artifact and runtime
`mx.stack` allocations. The unpruned no-MTP projection is 89 groups, 1,300
tensors, and `69.30 GiB`; the count-based 10% tooling fixture projects to
`63.40 GiB`. A bounded real smoke packed global state, layer 0, and layer 1 in
about three seconds. The pruned layer-1 shard retained 461 experts in
`1,478,307,524` bytes and produced exactly the same output as the corresponding
old expert IDs in the source layout. A packed unpruned layer peaked at about
`1.485 GiB`, versus `2.96 GiB` when stacking the official per-expert layout.

The public count-based keep-90 map remains a tooling fixture only, not a quality
plan. The large smoke outputs were deleted after validation; state and logs are
retained at:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/metadata/mlx-pack-smoke-20260710
```

### Fixed Mixed-Precision Linears

`nemotron/tools/nemotron_mlx_linear.py` covers the checkpoint's scalar-scaled
FP8 and BF16 matrices. Safetensors exposes `F8_E4M3` payloads to MLX as raw U8.
The default path reinterprets those exact bytes for native MXFP8 qmm, supplies a
reused E8M0 unity-scale matrix, and folds ModelOpt's tensor-wide FP32 scale into
the activation. No FP8 weight conversion occurs. A separate custom Metal
E4M3FN decoder provides an independent result for numerical tests.

The real layer-0 Mamba `in_proj` (`18560x4096`, about 76 MB) agrees with scalar
row decoding at relative L2 below `1e-7`. Warm 200-call samples measured
approximately `0.172-0.207 ms`, or `367-442 GB/s`; the earlier custom kernel
was about `238 GB/s`. Reused unity scales add one byte per 32 weights for each
distinct matrix shape, not for every tensor instance.

### Mamba2 Decode Baseline

`nemotron/tools/nemotron_mlx_mamba.py` composes an official Mamba2 layer using
the upstream MLX-LM NemotronH equations, original BF16 RMSNorm/convolution/SSM
parameters, exact ModelOpt FP8 projections, and persistent GPU-owned
`ArraysCache` state. The loader validates every required tensor and only accepts
Mamba positions from the checkpoint's hybrid pattern.

Four recurrent steps agree between native MXFP8 qmm and the independent custom
FP8 decoder at relative L2 around `1e-7`; output maximum error is around
`1e-7` and recurrent-state maximum error remains below `2e-5`. Repeated real
layer-0 warm decode samples measured about `0.34-0.40 ms` per token. The
dual-path validation peak was about `355 MiB`; a production runtime keeps only
the native path.

### LatentMoE Layer Baseline

`nemotron/tools/nemotron_mlx_moe_layer.py` composes the full expert block:
RMSNorm, BF16 sigmoid router with correction bias and top-22 normalization,
mixed-precision latent input projection, selected packed NVFP4 experts,
mixed-precision latent output projection, ReLU-squared shared expert, weighted
reduction, and residual. Router indices, expert activations, and reductions
stay in the MLX graph.

Real optimized/reference checks pass on layers 1, 3, and 87, covering different
official FP8/NVFP4/BF16 assignments. Relative L2 remained below `1.4e-7`; the
largest absolute difference was about `3.1e-5` on the high-magnitude final
layer. Warm source-layout measurements were approximately `0.43-0.89 ms` per
layer, with layer 1 around `0.64 ms`. Source-layout validation peaks near 3 GiB
because it builds the expert stack; the verified packed runtime layout removes
that duplicate allocation.

At current isolated-layer rates, the 40 Mamba+LatentMoE pairs account for
roughly `39-42 ms` per generated token before periodic attention and final-head
work. This is a projection, not a full-model throughput claim, but it supports
the feasibility of useful M4 Max decode once the remaining graph is composed.

### Full-Attention Decode Baseline

`nemotron/tools/nemotron_mlx_attention.py` loads the eight periodic attention
layers with original BF16 q/k/v/o weights and GPU-owned MLX KV cache. NVIDIA's
official `modeling_nemotron_h.py` confirms that ordinary attention applies no
RoPE and does not multiply projected keys/values by the checkpoint `k_scale`
or `v_scale`; those scalars are exported calibration metadata for optional
quantized KV caches.

The one-token BF16 projections use a Nemotron row-parallel Metal kernel, while
multi-token prefill falls back to MLX matrix multiplication. A four-token
full-sequence computation and incremental cached decode agree at relative L2
about `3.8e-7` and maximum absolute error below `5e-7`, which independently
checks the custom decode projections against generic prefill. The specialized
path reduced layer-7 context-32 decode from about `1.21 ms` to `0.28 ms`.
Context-128/256 samples measured about `0.20-0.23 ms`; all eight attention
layers should therefore contribute only a few milliseconds per token at short
and moderate contexts.

### Full Official-Checkpoint Forward

`nemotron/tools/nemotron_mlx_stream_forward.py` now composes all 88 layers,
embedding, final RMSNorm, and full-vocabulary BF16 head directly from the
official NVIDIA checkpoint. It keeps SSM/KV state resident but loads and
releases one layer's weights at a time, so the 74.78 GiB source can run on this
64 GB Mac without first creating a compressed candidate. Layer-major prompt
prefill loads each layer once while evaluating recurrent/cache state at every
position; a two-token partial-path comparison matches token-major execution.

The first complete token-0 forward took `23.97 s`, peaked at `4.19 GiB`, and
recorded all 40 routed layers. A tokenizer-derived `Hello` forward selected
token 1044 (`,`) as its top continuation. More importantly, raw `2+2=` encoded
as `1050,1043,1050,1061`; its four-position full forward took `22.97 s`, peaked
at `3.45 GiB`, and selected token 1052 (`4`) with score `77.57`, ahead of token
1049 (`1`) at `75.80`. This is the first factual end-to-end quality result on
the official local NVFP4 weights and is the baseline anchor for pruning-plan
comparisons.

Example:

```sh
NEMOTRON_MODEL_DIR=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_stream_forward.py \
  --source-dir "$NEMOTRON_MODEL_DIR/source-nvfp4" \
  --prompt '2+2=' \
  --top-k 8
```

### Activation-Aware Calibration

`nemotron/tools/nemotron_mlx_calibrate.py` runs layer-major official-source
forwards over a pinned diverse corpus and records, for every routed expert:
selection count, router-score mass, latent output-norm mass, route-weighted
latent output contribution, maximum score, and maximum output norm. Runs are
revision/corpus/config bound, update state atomically after each batch, and
resume from completed batch indices.

The current calibration uses oMLX's pinned oQe corpus at commit
`6342b4d9c0dce296366f061cee066aeea16305dc`. Sixteen interleaved 32-token
batches cover tool calling, chat, mixed text, reasoning, code, English, Korean,
Chinese, Japanese, and additional community samples. The 512-token result is:

- 98.550% of all 20,480 layer/expert slots observed
- at least 485 and at most 512 experts observed per layer
- about 27 seconds per batch on the layer-streamed source path

The durable observation is:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/metadata/calibration-oqe32.json
```

`nemotron/tools/nemotron_mlx_prune_plan.py` builds uniform-count per-layer plans
from normalized route-weighted output contribution (55%), router score mass
(20%), frequency (15%), and maximum output norm (10%). Unobserved experts are
always protected. Coverage guards rise with prune ratio: 85% for 10%, 90% for
15%, 93% for 20%, and 96% beyond 20%.

Current plans under `plans/oqe32-512tok/` retain 461, 436, 410, and 384 experts
for approximately 10%, 15%, 20%, and 25% cuts. The 20% no-MTP direct runtime
pack projects to `57.50 GiB` and is the first serious 64 GB candidate. These
plans are activation-informed but not yet quality-approved; candidate logits,
coding evaluations, and baseline comparisons remain mandatory.

### First 20% Candidate

The first full activation-informed candidate lives at:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/candidate-oqe512-r20-mlx
```

It completed with 89/89 groups verified, no partial files, 1,300 runtime
tensors, 410 experts per MoE layer, MTP omitted, and `61,745,143,264` payload
bytes (`57.5046 GiB`). Retained source payloads are byte-identical. The full
pack took about 106 seconds and left 111 GiB free on the development disk.

The candidate preserves the `2+2=` top token (`4`) and reduces layer-streamed
four-token prefill from about 23 seconds to 11.6 seconds. Full-vocabulary logit
comparisons currently cover arithmetic, factual completion, and a coding
prefix:

| Prompt | Top-1 | Centered rel-L2 | Cosine | KL | Top-64 overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| `2+2=` | same (`4`) | 0.0408 | 0.99918 | 0.0178 | 56/64 |
| `The capital of France is` | same (` Paris`) | 0.0329 | 0.99950 | 0.0126 | 60/64 |
| `def fibonacci(n):` | baseline top ranks 2nd | 0.0319 | 0.99954 | 0.0184 | 59/64 |

The coding prefix swaps two nearly tied whitespace/control tokens (` ` and
`\\`), so this evidence is promising but not yet a coding-quality acceptance.
Across the three probes, mean centered relative L2 is 0.0352 and mean KL is
0.0163. Reports and compact `.npy` logits live under the model directory's
`quality/` tree.

`nemotron/tools/nemotron_mlx_compare_logits.py` computes these metrics from
full-vocabulary arrays. `nemotron_mlx_stream_forward.py --logits-out PATH.npy`
writes the arrays atomically.

The current kernel limit is a separate resident-runtime constraint:
`iogpu.wired_limit_mb` is 49,152 MB on this Mac, below the 57.5 GiB artifact.
Do not attempt to wire the whole model until that limit is raised deliberately;
the layer-streamed quality path remains safe without doing so.

### Resident Runtime Preflight

`nemotron/tools/nemotron_mlx_resident.py` preloads all 88 packed blocks, keeps
Mamba/KV state on GPU, and evaluates each token as one lazy MLX graph. Before
loading anything large it validates the completed pack report, reads both
Apple's recommended working-set size and the live `iogpu.wired_limit_mb`, adds
an explicit runtime margin, and refuses to continue when the effective cap is
too small.

For the current 57.5046 GiB candidate with a 1.5 GiB runtime margin, preflight
requires 59.0046 GiB and rounds the requested kernel setting to 60,672 MiB.
The current kernel cap is 49,152 MiB, so resident execution is intentionally
blocked. The temporary setting required before the first resident benchmark is:

```sh
sudo sysctl -w iogpu.wired_limit_mb=60672
```

This leaves limited non-wired memory on a 64 GB machine. Close memory-heavy
applications first. The runtime still calls `mx.set_wired_limit` only for the
validated requirement and caps MLX's allocator cache; the sysctl merely raises
the kernel ceiling and resets on reboot.

After raising it, the guarded first run is:

```sh
NEMOTRON_MODEL_DIR=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_resident.py \
  --model-dir "$NEMOTRON_MODEL_DIR/candidate-oqe512-r20-mlx" \
  --prompt '2+2=' \
  --max-new-tokens 8
```

## Acceptance Gates

A candidate is not promoted based on size or a few prompts. It must pass:

- strict metadata, tensor shape, dtype, and quantization validation
- exact retained-payload validation for prune-only artifacts
- deterministic reference logits at multiple prompt and context lengths
- coding, reasoning, instruction-following, and long-context evaluations
- explicit quant-only, prune-only, and combined comparisons
- memory accounting that includes weights, recurrent/KV state, scratch, runtime,
  and operating-system headroom
- repeatable prefill and decode performance measurements on target hardware

Every failed candidate remains clearly labeled diagnostic and must not become a
source for later calibration or production compression.

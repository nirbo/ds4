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

Forward-looking size and performance work is tracked in
[`NEMOTRON_EXPERIMENTS.md`](NEMOTRON_EXPERIMENTS.md). Complete one independently
measured experiment at a time and record successful and rejected outcomes.

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
these separate from backbone experts. The base runtime omits them and an
independently planned and validated sidecar supplies speculative decoding;
backbone pruning code must never mistake MTP experts for backbone experts.

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
- MLX `0.32.0` commit `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`
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

The isolated environment currently pins MLX `0.32.0`, MLX-LM `0.31.3` from
commit `a790972f0f844d81067ed45c28b524220a10c019`, and Transformers `5.13.0`.
The published MLX-LM `0.31.3` wheel contains a broken tokenizer registration
that passes `"NewlineTokenizer"` as a string; the pinned upstream source passes
the class object and imports correctly. Do not restore the broken wheel over
the pinned source build.

MLX `0.32.0` was promoted only after an isolated `0.31.2`/`0.32.0` A/B on the
complete resident candidate. It caches each single-row RMSNorm input in
registers and adds a small-batch quantized matvec path. One-token Mamba state
remains within `1.53e-5` of the independent FP8 reference. The small-batch
kernel changes batched-versus-incremental Mamba state by at most `3.05e-5` and
outputs by at most `1.20e-7`; these bounds are now explicit acceptance limits.
The full block-2 verifier remained top-1 identical, with full-logit relative L2
`8.32e-8`, maximum absolute error `1.53e-5`, and accepted-cache drift
`1.53e-5`.

The production 128-token speculative run generated identical token IDs and
measured `33.158 tok/s`, up from `32.027 tok/s` on `0.31.2`. Median target
verification fell from `52.487 ms` to `50.559 ms`; MTP fell from `4.199 ms` to
`4.093 ms`; peak MLX memory fell from `58.339 GiB` to `58.333 GiB`. The
ordinary 64-token arithmetic run measured `41.792 ms` median decode versus
`42.937 ms`, with the same token IDs and `57.736 GiB` peak.

MLX safetensors loading itself is not a steady-state decode hot path. In this
release, `mx.load` parses the header and creates one lazy `Load` primitive per
tensor. First evaluation allocates the Metal-visible destination and reads it
with `pread`; `ParallelFileReader` uses four workers and 32 MiB chunks. It does
not mmap tensor payloads. Repeated `mx.load` calls in the resident constructor
select disjoint fixed/expert/global tensors, so they repeat small header and
file-descriptor work but do not evaluate duplicate 57 GiB payloads. The
observed roughly 10-second first page-in is a one-time persistent-process cost,
not part of the 41-42 ms decode transition.

MLX command-buffer thresholds were also swept. Raising the 50 MiB default to
2 GiB improved an isolated queued MoE test but slowed full-model decode to
`42.338 ms` and raised peak memory. Lower thresholds from 8-32 MiB moved median
decode by no more than about 0.3%, with less consistent tail latency. The
default remains selected; there is no production environment override.

Three subsequent Metal experiments were rejected rather than retained:

- A fused selected-expert NVFP4 down projection and router reduction reached
  `0.119 ms` at its best 2-SIMD layout, versus `0.070 ms` for MLX's native
  gather plus reduction. Its reduced intermediate traffic did not compensate
  for replacing MLX's vectorized FP4 dot implementation.
- A selected-expert up kernel cached the common 1024-value latent input in
  threadgroup memory and reused register tiles across rows. Its best 32-row
  layout took `0.082 ms`, versus `0.064 ms` for native `gather_qmm`; Apple's
  cache hierarchy already keeps the small shared input hot.
- A cached one-token SSM kernel eliminated repeated A/dA exponentiation and
  B/C loads but remained state-bandwidth-bound at `0.0759 ms`, versus
  `0.0752 ms` upstream. FP16 recurrent state would halve that traffic and save
  about 80 MiB, but one layer accumulated `1.47e-3` relative output drift and
  `4.60` maximum state error over 256 tokens, so reduced-precision state is not
  accepted.
- Adjacent verifier tokens shared 9.15 of 22 routed experts per layer on
  average, leaving 34.85 unique matrices among 44 routes. A source-built MLX
  `0.32.0` test removed the `B / E >= 4` guard on its existing sorted-RHS
  `gather_qmm` kernel and supplied GPU-sorted routes. At realistic nine-expert
  overlap, grouped up projection took `0.366 ms` versus `0.133 ms`, and the
  full expert pair took `0.966 ms` versus `0.249 ms`. The grouped matrix kernel
  is tuned for substantially longer same-expert runs; route sorting and short
  slices make it unsuitable for block-2 verification.

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

The pinned oMLX checkout was refreshed through commit
`6342b4d9c0dce296366f061cee066aeea16305dc` (2026-07-10). Its own published
engine comparisons show ordinary single-stream decode remaining close to
`mlx-lm`; oMLX's large user-visible gains come from exact prefix reuse,
continuous batching, and speculative decode. Its current native MTP patch does
not implement Nemotron-H, but its two-token verify, cache rollback, greedy
identity, stochastic residual sampling, and measured-cost adaptive-depth logic
are directly relevant references for a future Nemotron MTP path.

The immediate leverage order is therefore: complete exact model execution in
MLX, collect full-model routing/output sensitivity with expert-coverage gates,
materialize a conservative prune-only candidate, establish quality, then add
MTP and long-context cache tiering.

### DFlash And DSpark Findings

DFlash (`https://github.com/z-lab/dflash`) uses a trained lightweight block
diffusion model conditioned on selected target hidden states to draft a block
in one forward pass. Its public large-target drafters are commonly about 0.8B
BF16 parameters and its reference implementation includes an MLX backend plus
recurrent rollback for Qwen Gated DeltaNet. No compatible Nemotron-H drafter is
published; draft weights are target-specific and cannot be borrowed from Qwen,
Gemma, GPT-OSS, or DeepSeek.

DeepSeek's DeepSpec (`https://github.com/deepseek-ai/DeepSpec`) implements
DSpark, DFlash, and EAGLE-3 training/evaluation. DSpark adds a low-rank Markov
head for cheap intra-block dependence and a confidence head for adaptive
verification length. The public code currently supports Qwen3 and Gemma4.
Its default workflow assumes eight GPUs and warns that the Qwen3-4B target
hidden-state cache is roughly 38 TB, so training a quality Nemotron drafter is
a separate external-compute project rather than an immediate local step.

The reusable architectural decision is to keep target block verification
drafter-agnostic. Native Nemotron MTP is the first backend because its weights
are already trained. A future DFlash/DSpark backend can reuse the same exact
Mamba/KV verification and rollback path if target-pass batching proves useful.

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

### Nonuniform 25% Candidate

`nemotron_mlx_layer_sensitivity.py` captures source hidden states once and
measures exact hard-pruning behavior at every MoE layer for 10-45% cuts.
`nemotron_mlx_layer_allocate.py` applies a monotonic robust-output cost and
exact dynamic programming to spend a fixed expert total by layer.
`nemotron_mlx_plan_compare.py` compares uniform and nonuniform plans on a
separate corpus through revision-bound virtual pruning. Virtual output has been
proven exactly equal to the physically packed runtime.

The r25 allocation protects six layers completely, keeps others at measured
10/15/30% cuts, and uses 40% cuts in 19 tolerant layers. Counts span 308-512
and average exactly 384, so payload is byte-matched with uniform r25. The
materialized candidate is:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/candidate-nonuniform-r25-mlx
```

It contains `54.4974 GiB` of validated payload and preserves all retained
NVFP4, FP8, BF16, scales, and router rows byte-for-byte. On an untouched
eight-category full-logit comparison, it retained source top-1 on 7/8 cases
versus 5/8 for uniform r25; mean/worst KL were `0.08358/0.21477` versus
`0.45654/3.23938`. Mean centered relative-L2 was narrowly worse (`0.07554`
versus `0.07385`), and individual coding/general cases were mixed. It is
therefore promising, not yet a replacement for r20.

With paged embeddings, resident ordinary decode peaked at `53.729 GiB` and
measured `23.610 tok/s` over 63 transitions. This saves about 3 GiB versus r20
without changing ordinary throughput. A reduced 32K MTP map is bound to this
candidate under `mtp-vocab-map-bf16-e32768-nonuniform-r25/`; speculative
decode measured `34.195 tok/s`, 76.19% acceptance, and `1.419x` speedup over
the paired `24.099 tok/s` ordinary control with exact token integrity. Peak
memory was `54.333 GiB`. This run also verified that the launcher preserves the
approved kernel cap after completion.

#### Initial layerwise distillation result

`nemotron_mlx_layer_distill.py` streams the immutable source teacher, executes
a virtual pruned student on identical hidden states, and fits tiny corrections
without changing retained quantized payloads. Per-channel affine fitting
overfit and failed held-out validation. A scalar scale/bias per layer was
stable on eight training and eight validation categories at r35, but reduced
mean local output relative-L2 by only 1.87% (`0.07499` to `0.07358`) and did
not improve the worst case. The 5.8 KiB sidecar is not integrated into the
runtime. A subsequent rank-4
hidden-to-routed-residual experiment was also rejected: its full held-out run
worsened mean output error from `0.07499` to `0.07563` and routed worst-case
error from `0.879` to `1.406`. Post-layer linear correction is therefore
closed; meaningful recovery must alter replacement expert or routing behavior
and pass an independent full-logit gate.

A subsequent ReLU-squared latent adapter targeted the missing 1024-dimensional
expert aggregate before `fc2_latent`. With 189 diverse training tokens it still
worsened held-out mean output error (`0.07499` to `0.07526`) and routed worst
error (`0.879` to `1.356`). Only 8/40 layers passed both local mean and maximum
error gates, and the best improvement was 1.09%. Small correction sidecars are
therefore exhausted; further recovery needs actual replacement expert
parameter training or mathematically constructed expert merging.

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

The kernel limit is a separate resident-runtime constraint. This Mac normally
exposes a 49,152 MiB `iogpu.wired_limit_mb`, below the 57.5 GiB artifact. The
limit was deliberately raised to 60,672 MiB for the first guarded resident
tests; this temporary setting resets on reboot. The layer-streamed quality path
remains the low-memory fallback.

### Resident Runtime Preflight

`nemotron/tools/nemotron_mlx_resident.py` preloads all 88 packed blocks, keeps
Mamba/KV state on GPU, and evaluates each token as one lazy MLX graph. Before
loading anything large it validates the completed pack report, reads both
Apple's recommended working-set size and the live `iogpu.wired_limit_mb`, adds
an explicit runtime margin, and refuses to continue when the effective cap is
too small.

For the current 57.5046 GiB candidate with a 1.5 GiB runtime margin, preflight
requires 59.0046 GiB and rounds the requested kernel setting to 60,672 MiB.
The temporary setting used for resident benchmarks is:

```sh
sudo sysctl -w iogpu.wired_limit_mb=60672
```

This leaves limited non-wired memory on a 64 GB machine. Close memory-heavy
applications first. The runtime sets MLX to the validated effective kernel cap
and preserves that cap after completion; it does not restore MLX's lower
default over the user-approved setting. The sysctl raises the kernel ceiling
and resets on reboot.

After raising it, the guarded first run is:

```sh
NEMOTRON_MODEL_DIR=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_resident.py \
  --model-dir "$NEMOTRON_MODEL_DIR/candidate-oqe512-r20-mlx" \
  --prompt '2+2=' \
  --max-new-tokens 8
```

The first full resident run succeeded. It peaked at `57.736 GiB`, preserved
the arithmetic continuation (`2+2=4, 2+3=5`), and established that the packed
candidate can execute entirely resident on the 64 GB M4 Max. Corrected
steady-state timing over 31 actual token transitions measured `42.21 ms`
median, `43.52 ms` p95, and `23.60 tok/s`. A 64-token raw coding continuation
produced a correct recursive Fibonacci implementation and example at
`22.43 tok/s`, peaking at the same `57.736 GiB`.

The runner reports time-to-first-token separately from decode transitions and
does not execute an unused final model forward. `--token-timings` prints every
measured transition for performance investigations. Initial model mapping,
page-in, and graph compilation take about 10 seconds in the current one-shot
CLI; a persistent serving process should pay that cost once.

The official checkpoint's MTP tensors add `5.4805 GiB`, so they are not attached
unchanged to the 20% candidate. The validated compact sidecar path is described
below.

### Multi-Token Verification

`nemotron/tools/nemotron_mlx_verify_bench.py` exercises the shared target-side
foundation required by native MTP, DFlash, and DSpark. It snapshots all Mamba
and attention caches, derives deterministic draft tokens from the target,
compares full-vocabulary block logits against sequential decode, restores the
pre-block state, and reports paired median timings.

Two implementation details were required for correctness and bounded memory:

- MLX's generic multi-token SSM scan produced small output drift but left the
  layer-0 recurrent state about `6.4e-4` relative from one-token recurrence.
  `mamba_sequence_exact` now batches the expensive FP8 input/output projections
  while applying each small SSM update in the same order as decode. Under the
  validated MLX `0.32.0` small-batch kernel, output drift is at most `1.20e-7`
  and recurrent/captured-state drift is at most `3.05e-5`; full verifier logits,
  rollback, accepted cache state, and token identity remain separately gated.
- Generic multi-row BF16 matrix multiplication caused a Metal out-of-memory
  failure under the resident cap. The verification kernel now reads each BF16
  weight row once and accumulates 2-32 token vectors together. On the real
  1 GiB vocabulary head, block 2/4/8 took about `2.71/2.53/3.38 ms` and matched
  individual matvec output exactly.

Full 88-layer verification on the 20% candidate produced:

| Block | Sequential | Batched | Target-pass speedup | Verified tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 84.96 ms | 51.61 ms | 1.65x | 38.75 |
| 4 | 170.06 ms | 76.79 ms | 2.22x | 52.09 |
| 7 | 318.21 ms | 129.39 ms | 2.46x | 54.10 |
| 8 | 337.23 ms | 130.73 ms | 2.58x | 61.20 |
| 16 | 676.09 ms | 175.99 ms | 3.84x | 90.91 |

All block sizes preserved top-1 tokens. Full-logit relative L2 ranged from
`5.9e-8` to `4.7e-7`; maximum absolute error remained below `1e-4`, and the
post-restore next-token logits matched exactly. Peak MLX memory was
`58.104 GiB`, below the active 59.25 GiB kernel cap.

The accepted-token cache capture now also records the exact convolution and SSM
state after any requested verifier row. On the full target, restoring the
captured block-2 state and continuing differs from true incremental logits by
only `1.14e-5` maximum absolute error. Captured block-2 verification measures
`54.3 ms`, versus `52.2 ms` without capture, so rejection rollback does not
require target recomputation.

### Native MTP Sidecar

NVIDIA Megatron Core commit
`1aa880d0ea1dfddef567785ebc9a384f2a600d18` establishes the inference contract.
`compute_mtp_single_step` calls `forward_single_position` after target
verification with the target's final normalized hidden state and the freshly
sampled accepted token. The MTP head computes
`eh_proj([enorm(embedding), hnorm(hidden)])`, runs its BF16 attention and latent
MoE layers, applies its own final norm, and reuses the target `lm_head`.
Critically, it passes no inference cache: MTP is stateless, has no prompt
prefill, and must not retain an MTP KV cache.

`nemotron_mlx_mtp_bench.py` captured 375 target rows over eight coding prompts,
including 256 scored decode transitions, without co-residing the target and
full MTP. The full official BF16 head achieved 80.08% top-1 and 98.05% top-5
acceptance. Decode-only router score mass produces independently bound expert
plans. Exact BF16 subset results were:

| MTP experts | Payload | Top-1 acceptance | Median MTP |
| ---: | ---: | ---: | ---: |
| 32 | 0.555 GiB | 62.50% | 4.60 ms |
| 48 | 0.719 GiB | 69.53% | 4.59 ms |
| 64 | 0.883 GiB | 71.88% | 4.60 ms |
| 96 | 1.212 GiB | 75.78% | 4.09 ms specialized Metal |
| 128 | 1.540 GiB | 77.34% | 4.57 ms reference |

`nemotron_mlx_mtp_pack.py` slices router rows in the same ranked order and
stacks retained BF16 expert payloads without changing their bytes. A custom
Metal switch kernel reads only the 22 selected experts directly from
`[expert,row,column]`; it is bit-exact to individual BF16 matvecs and avoids
MLX `gather_mm`'s 39 ms transposed-materialization path.

BF16 sidecars still leave insufficient driver headroom or cause severe paging
when combined with the resident target. MTP-only Q4 is therefore an explicit
second-stage experiment; it never changes target weights or generated output,
because every draft is verified. `nemotron_mlx_mtp_quantize.py` records
full-tensor error for every changed matrix, while the acceptance suite measures
activation/logit impact. At 96 experts:

| MTP format | Payload | Top-1 acceptance | Median MTP |
| --- | ---: | ---: | ---: |
| affine Q4 | 0.341 GiB | 71.48% | 3.33 ms |
| MXFP4 | 0.322 GiB | 74.61% | 3.11 ms |
| NVFP4 | 0.341 GiB | 76.56% | 3.11 ms |

NVFP4-128 is the quality-oriented default. It occupies `0.433877 GiB`, scores
77.73% on the 256-transition calibration, and uses the same selected top-22
compute as NVFP4-96. The durable artifact is:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/mtp-sidecar-e128-nvfp4
```

The 96-expert sidecar remains an intermediate diagnostic. The measured
lower-memory fallback is described below. Plans, target traces, acceptance
reports, hashes, and failed-candidate evidence live under `mtp-reference/`
beside the source and candidates.

#### Draft-only vocabulary head

The target's BF16 `lm_head` is about 1 GiB and is reused by the default MTP
sidecar. A second, MTP-only NVFP4 head was tested to reduce draft latency. The
revision-bound `nemotron_mlx_mtp_head_quantize.py` artifact is `0.281250 GiB`;
its loader checks the source revision, artifact hash, payload size, metadata,
quantization settings, and `[131072, 4096]` source shape. Resident preflight
accounts for it separately.

On the existing 256-transition coding trace, the current MLX environment gave:

| Sidecar | Draft head | Payloads | Top-1 | Median MTP |
| --- | --- | ---: | ---: | ---: |
| NVFP4-128 | BF16 shared | 0.434 GiB | 77.73% | 3.110 ms |
| NVFP4-128 | NVFP4 separate | 0.434 + 0.281 GiB | 77.34% | 1.377 ms |
| NVFP4-96 | NVFP4 separate | 0.341 + 0.281 GiB | 76.17% | 1.381 ms |
| NVFP4-64 | NVFP4 separate | 0.249 + 0.281 GiB | 71.88% | 1.378 ms |

The faster head cannot be added to the 128-expert default profitably. At a
256 MiB MLX cache it exceeded available transient Metal memory. A 128 MiB cache
ran exactly but reached only `19.695 tok/s`. The 96-expert combination reached
only `22.029 tok/s` because memory-pressure outliers remained.

The 64-expert combination is a valid lower-memory fallback. A 128-token run
with exact accepted-state rollback measured `30.885 tok/s` versus
`23.728 tok/s` ordinary decode (`1.302x`), 77.42% prompt acceptance, `2.221 ms`
median MTP, `51.668 ms` median verification, and `58.436 GiB` peak. Output token
IDs exactly matched ordinary BF16-target greedy decode. It is smaller but does
not replace the faster 128-expert shared-BF16-head default.

Quantizing the authoritative target head was also rejected. On captured final
target hidden states, NVFP4 changed 13 of 256 scored greedy decisions (94.92%
agreement), MXFP8 changed 19 (92.58%), and the best tested affine format still
changed 9 (96.48%). Centered-logit metrics alone were misleadingly strong, so
the BF16 target head remains authoritative.

Exact BF16 candidate re-ranking does not rescue the NVFP4 target head at useful
cost. `nemotron_mlx_head_certificate.py` computes per-row, per-16-value-group
BF16/NVFP4 error norms and a conservative Cauchy upper bound for every excluded
token. Top-4 candidates recalled the BF16 winner on all 256 coding transitions
but certified only one. Top-512 certified 47/256; even 32,768 exact BF16
candidates (256 MiB of row reads per token) certified only 232/256. The
remaining full-head fallback rate makes the route slower and less predictable
than retaining the BF16 head, so no inference runtime uses it.

The durable fallback artifacts are:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/mtp-sidecar-e64-nvfp4
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/mtp-lm-head-nvfp4
```

Run the fallback with the measured 256 MiB cache budget:

```sh
NEMOTRON_MODEL_DIR=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_speculative.py \
  --model-dir "$NEMOTRON_MODEL_DIR/candidate-oqe512-r20-mlx" \
  --mtp-sidecar "$NEMOTRON_MODEL_DIR/mtp-sidecar-e64-nvfp4" \
  --mtp-lm-head "$NEMOTRON_MODEL_DIR/mtp-lm-head-nvfp4" \
  --prompt $'Complete this Python function:\n\ndef binary_search(values, target):\n' \
  --max-new-tokens 128 \
  --warmup-cycles 10 \
  --margin-gib 0.5 \
  --cache-limit-mib 256 \
  --capture-rollback
```

The original full-head 128-token resident benchmark used the 20% target,
NVFP4-128 sidecar, block-2 verification, and exact accepted-state rollback. It
produced exactly the same token IDs as ordinary greedy decode and measured:

- `32.027 tok/s` speculative versus `23.659 tok/s` ordinary (`1.354x`)
- 86.44% acceptance on the measured coding continuation
- `4.199 ms` median MTP and `52.487 ms` median target verification
- `58.339 GiB` peak MLX memory

#### Reduced-vocabulary shared target head

`nemotron_mlx_mtp_vocab_head.py` builds deterministic reduced draft
vocabularies from even-indexed training rows in each category of the retained
oMLX calibration corpus, holding odd-indexed rows out. All 1,000 added/control
tokens are mandatory. Category-normalized token frequency ranks the remaining
rows, and low token IDs fill any unused budget deterministically.

Copied BF16 8K/16K/32K heads occupied 64/128/256 MiB and caused increasing
resident pressure. The accepted representation stores only sorted target token
IDs and uses `bf16_gather_matvec` to read exact selected rows from the target's
already-resident BF16 head. The 32K map is 131,072 bytes and covers 96.46% of
611,402 held-out corpus tokens; the worst category covers 92.56%.

On the 256-transition coding trace:

| MTP projection | Top-1 acceptance | Median MTP |
| --- | ---: | ---: |
| Full 131K BF16 | 77.73% | 3.114 ms |
| Shared 32K BF16 map | 75.00% | 1.417 ms |

Per-prompt acceptance was unchanged on three of eight coding prompts and lost
one to three accepted drafts out of 32 on the other five. Every generated token
still comes from the unchanged BF16 target verifier.

Alternating fresh-process 128-token controls measured `33.046/32.916 tok/s`
for the full head and `34.716/33.272 tok/s` for the shared 32K map: means of
`32.981` and `33.994 tok/s`, a repeatable 3.1% gain. All runs preserved exact
token IDs and stayed at `58.335 GiB` peak or below. Three additional 96-token
coding prompts were exact; the map materially improved the Rust case, was
approximately 1.3% slower on SQL, and improved the graph case by about 2.4%.
The full head therefore remains an explicit workload fallback.

The promoted map is:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/mtp-vocab-map-bf16-e32768
```

Use the temporary 60,672 MiB kernel limit, close other memory-heavy programs,
and run:

```sh
NEMOTRON_MODEL_DIR=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_speculative.py \
  --model-dir "$NEMOTRON_MODEL_DIR/candidate-oqe512-r20-mlx" \
  --mtp-sidecar "$NEMOTRON_MODEL_DIR/mtp-sidecar-e128-nvfp4" \
  --mtp-lm-head "$NEMOTRON_MODEL_DIR/mtp-vocab-map-bf16-e32768" \
  --prompt $'Complete this Python function:\n\ndef binary_search(values, target):\n' \
  --max-new-tokens 128 \
  --warmup-cycles 10 \
  --margin-gib 0.5 \
  --cache-limit-mib 256 \
  --capture-rollback
```

#### Recursive two-token MTP

The MTP sidecar can recursively consume its own final normalized hidden state
to propose a second token. On the retained 256-transition coding trace, the
shared 32K head's conditional acceptance was 75.00% at depth one and 64.21% at
depth two; acceptance fell below 48% at depths three and four. The full head
showed the same shape, so longer recursive chains are not justified.

The exact verifier supports cache snapshots at arbitrary block prefixes. A
real block-3 validation matched incremental top-1 output and bounded prefix
continuation drift to `1.14440918e-05`. The runtime uses the cheaper hot path:
it captures only the state after draft one and replays from the pre-block state
if the earlier draft is rejected.

Adaptive depth two is opt-in. The best measured thresholds produced
`34.872/34.982 tok/s` on repeated 128-token runs, versus `34.137 tok/s` for the
identical one-draft control, with exact output and approximately `58.50 GiB`
peak. This is a valid 2-3% gain but misses the experiment's 5% promotion gate,
so one-draft generation remains the default. Ungated recursion was materially
slower.

```sh
NEMOTRON_MODEL_DIR=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_speculative.py \
  --model-dir "$NEMOTRON_MODEL_DIR/candidate-oqe512-r20-mlx" \
  --mtp-sidecar "$NEMOTRON_MODEL_DIR/mtp-sidecar-e128-nvfp4" \
  --mtp-lm-head "$NEMOTRON_MODEL_DIR/mtp-vocab-map-bf16-e32768" \
  --max-new-tokens 128 \
  --warmup-cycles 10 \
  --margin-gib 0.5 \
  --cache-limit-mib 128 \
  --capture-rollback \
  --max-draft-tokens 2 \
  --draft-margin-threshold 1.5 \
  --second-draft-margin-threshold 1.0
```

The speculative launcher retains the full pre-approved 59.25 GiB process cap
for transient Metal buffers but still refuses any target/sidecar pair whose
explicit payload-plus-margin requirement exceeds it. The 0.5 GiB margin is
specific to this measured combined runtime; the ordinary resident CLI retains
its 1.5 GiB default.

#### Prompt and generated-token lookup drafting

`nemotron_ngram_lookup.py` maintains an LRU-bounded index over prompt tokens and
committed generated tokens. It searches the longest 3-8-token suffix, requires
two prior occurrences to agree on the complete continuation, and requires the
first proposed token to match MTP before target verification. Misses and
disagreements reuse the already-computed MTP draft, so the median CPU lookup
cost is approximately 0.02 ms.

Four-token lookup blocks are the accepted horizon. On a repetitive Python
coding control, two alternating final pairs averaged `26.711 tok/s` with lookup
versus `22.983 tok/s` for MTP alone, a 16.2% gain. Lookup-token acceptance was
95.0%, output token IDs exactly matched ordinary greedy decode, and peak MLX
memory was approximately `58.54 GiB`. An independent templated-test prompt and
a non-repetitive reasoning prompt emitted no consensus-qualified lookup blocks.

The feature remains opt-in because its benefit depends on repeated token
structure:

```sh
NEMOTRON_MODEL_DIR=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_speculative.py \
  --model-dir "$NEMOTRON_MODEL_DIR/candidate-oqe512-r20-mlx" \
  --mtp-sidecar "$NEMOTRON_MODEL_DIR/mtp-sidecar-e128-nvfp4" \
  --mtp-lm-head "$NEMOTRON_MODEL_DIR/mtp-vocab-map-bf16-e32768" \
  --max-new-tokens 128 \
  --warmup-cycles 10 \
  --margin-gib 0.5 \
  --cache-limit-mib 128 \
  --capture-rollback \
  --lookup-max-draft-tokens 4 \
  --lookup-min-key-tokens 3 \
  --lookup-max-key-tokens 8
```

#### Exact paged input embeddings

`nemotron_paged_embeddings.py` removes the 1 GiB BF16 input embedding table
from active Metal residency. It validates a revision-bound catalog, verifies
the exact payload SHA-256, mmaps `global.safetensors`, and constructs only the
requested 8 KiB BF16 rows. A bounded 256-row MLX cache covers repeated target
and MTP accesses.

Full-vocabulary logits were exactly equal to the resident-table path: zero max
absolute and relative-L2 drift. Active/peak ordinary memory fell from
`57.677/57.736 GiB` to `56.677/56.736 GiB`. Paired speculative controls
averaged `34.748 tok/s` paged versus `32.816 tok/s` resident, while paired
ordinary decode differed by only 0.14%. The combined runtime peaks near
`57.34 GiB`, approximately 1 GiB below the previous default.

Generate or revalidate the catalog after materializing a different candidate:

```sh
PYTHONPATH=nemotron/tools "$NEMOTRON_MODEL_DIR/mlx-env/bin/python" \
  nemotron/tools/nemotron_paged_embeddings.py \
  "$NEMOTRON_MODEL_DIR/candidate-oqe512-r20-mlx"
```

Use the new performance default by adding these options to the speculative
command above:

```sh
--paged-embeddings --embedding-cache-rows 256 --cache-limit-mib 256
```

Paged embeddings also stabilize adaptive depth two at `35.977 tok/s` on the
default coding control and improve consensus lookup to `30.459 tok/s` on its
repetitive control. Those drafting modes retain their existing opt-in policy.

### Rejected Full-Router Proxy Experts

The full-router proxy experiment tested whether the original 512-way router
could be retained while removed experts were redirected to functionally
similar retained experts. `nemotron_mlx_proxy_calibrate.py` captures
co-selection, same-input expert-output cosine, route-score products, and
category coverage from the immutable source. `nemotron_mlx_proxy_plan.py`
constructs a provenance-bound original-to-prototype map, and
`nemotron_mlx_proxy_compare.py` compares it directly with exact hard pruning on
identical source hidden states.

The 512-token calibration found poor substitutes: median best supported output
cosine was `0.1144`, with only 3.19% above `0.3`. At the 20% physical-expert
reduction, six held-out early/middle/late layer cases produced routed-output
relative-L2 geometric means of `0.11326` for proxies and `0.07783` for hard
pruning. Complete-output error was likewise worse (`0.01652` versus `0.01136`).
Mapping also left `21.359/22` unique prototypes per token on average, so score
aggregation would remove only 2.91% of expert dispatches.

The proxy path is therefore rejected before resident-runtime integration.
Artifacts and hashes are recorded in `NEMOTRON_EXPERIMENTS.md`; its observer
remains useful evidence for later layerwise merging or distillation work.

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

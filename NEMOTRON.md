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

The established balanced runtime is now 54.50 GiB and runs on the 64 GB Mac.
The next research objective is a trained mixed low-bit backbone in roughly the
32-40 GiB weight range without giving up the accepted quality gates. A roughly
21-23 GiB artifact is a later stretch point requiring low-bit experts plus
structural pruning; it is not inferred safe from either result independently.

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

### Mojo NVFP4 Spike

A standalone Mojo experiment under `nemotron/mojo/` implements the full
22-expert `1024 -> 2688 -> 1024` NVFP4 routed MLP on Apple's Metal backend.
It retains packed E2M1 weights and E4M3FN scales, keeps all intermediate state
on GPU, and tests expert-owned fusion, row-parallel execution, fused router
reduction, and four-row SIMD tiles. The isolated Modular 26.4 / Mojo 1.0.0b2
environment lives at `$NEMOTRON_MODEL_DIR/mojo-env-26.4`; run the logged probe
with `nemotron/run_mojo_moe_spike.sh`.

The latest compiler reduced the best exact Mojo path to about `0.29-0.30 ms`,
from about `0.77 ms` on stable 26.2. MLX's native complete selected-expert path
still measured `0.175186 ms` on real layer 1. Mojo is therefore rejected as an
Apple runtime dependency: it remains roughly 67% slower and there is no proven
zero-copy ownership boundary between its `DeviceContext` and MLX arrays. This
does not classify Mojo's NVIDIA NVFP4 path; a 5090 test is separate work.

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

The resident path compiles the post-routing tail for the dominant
BF16/BF16/FP8/FP8 projection assignment used by 34 of the 40 MoE layers. One
module-level graph accepts every weight and scale as a dynamic input, avoiding
the startup and residency failure of per-layer weight-capturing graphs. The
synthetic cache regression and real one-, two-, three-, and eight-token gates
are bit-exact against the eager equation. Five static-signature compiled tails
apply the same dynamic-weight boundary to the remaining six layers, covering
every checkpoint FP8/BF16/NVFP4 assignment without flattening its mixed
precision. Those real layers also remain bit-exact at one, two, three, and
eight tokens.

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

#### Fixed-budget coding expert protection

The original nonuniform allocation used only eight 32-token samples for layer
sensitivity, and its broad expert ranking used 512 total calibration tokens.
That evidence is sufficient for an initial candidate but weak for specialists
used during longer coding work. `nemotron_livecodebench_calibration.py` now
builds a provenance-bound coding corpus outside the target evaluation window.
The first corpus uses all 43 pinned July 2024 LiveCodeBench v5 tasks (13 easy,
16 medium, 14 hard); the target v6 gate starts in August, so no v6 evaluation
task was used for calibration. Corpus SHA-256 is
`f3a98e1f97bdbb76391ebe16bfc57064e69440046b5a6bc8746e77ab8edabde0`.

The official unpruned source observed the first 128 tokens of each task. The
complete 5,504-token calibration covered 99.419% of all layer/expert slots,
with at least 496 of 512 experts observed in every MoE layer. Its SHA-256 is
`f4ffcb49d35b854cd0fbe218f78dae522fd65e6a8ceea33e1958698833e4ee27`.

`nemotron_mlx_protected_plan.py` reallocates expert identities inside an
existing nonuniform plan without changing any layer's expert count. Every
expert unobserved by the broad calibration remains protected, the top 50% of
each retained budget by broad importance is immutable, specialist evidence
requires at least two route events, and only positive joint broad-plus-coding
score swaps are accepted. The validated conservative settings use a 25%
specialist score weight and cap changes at 2% of each layer's retained experts.
They swap 187 of 15,360 retained slots across 34 pruned layers (1.22%); all six
fully retained layers remain unchanged. The model payload therefore remains
`54.4974 GiB`. Plan SHA-256 is
`7b35d5d764042638a9e1f69fcdee2d751b0048cf52fd57de0c720a6fe6627bda`.

On the untouched eight-category full-logit gate, the protected plan preserved
source top-1 on 7/8 cases, increased mean top-64 overlap from 55.125 to 55.5,
and reduced mean KL from `0.08358` to `0.07095` (15.1%). Both coding controls
improved (`0.07279` to `0.04883` and `0.05021` to `0.03693`). Mean centered
relative-L2 also improved slightly (`0.07554` to `0.07434`), while raw
relative-L2 rose from `0.01731` to `0.01858`. A more aggressive 5% reservation was rejected after
mean KL regressed to `0.11021` and top-1 fell to 6/8. Accepted comparison
SHA-256 is
`04cb1153a58eaf1291324b6d44a3e87db1068ad9cb0e4d207f3fe215c86d8caf`.

The logit-gated plan was materialized at:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/candidate-protected-r25-mlx
```

`nemotron_mlx_repack.py` validated the completed base pack and both immutable
plans, hard-linked 55 mapping-identical groups, and atomically rebuilt the 34
changed MoE groups directly from the pinned source. The logical payload remains
`54.4974 GiB`, while only `37.4778 GiB` of new payload blocks were allocated.
Every completed group has a durable payload digest and exact target tensor
schema in both resumable repack state and standard complete pack state. Pack
report SHA-256 is
`a49789c05ec1421d993f5c6616147c94ba0958aa33fa62f16f372fefec0591d8`.

A 22-token coding prompt produced bit-exact full-vocabulary logits between
virtual source pruning and the materialized resident runtime: zero relative-L2,
zero maximum error, identical top-64, and identical top token. Parity report
SHA-256 is
`ebbefb19e2b55ef23f7b288bc99b5c76da610cb7c520eda2dd4d835e596d37ec`.
Paged-embedding resident loading peaked at `53.729 GiB`. A 16-token ordinary
decode smoke measured `24.261 tok/s` over 15 transitions, with `41.206 ms`
median and `42.262 ms` p95 latency. The prior candidate remains retained until
the protected artifact passes substantive generation-quality acceptance.

It did not pass that acceptance gate. The exact prior 30-task dated-v6 hidden
protocol was replayed with identical tasks, repeats, seeds, low-effort sampling,
8,192-token ceiling, and all official cases. Protected r25 scored 36/60 versus
37/60 for baseline r25, and task pass-any fell from 22/30 to 20/30:

| Difficulty | Baseline r25 | Protected r25 | Paired baseline-only | Paired protected-only |
| --- | ---: | ---: | ---: | ---: |
| Easy | 18/20 | 20/20 | 0 | 2 |
| Medium | 15/20 | 15/20 | 2 | 2 |
| Hard | 4/20 | 1/20 | 4 | 1 |
| Overall | 37/60 | 36/60 | 6 | 5 |

At task level, protected r25 gained `3616` but lost `abc391_f`, `3696`, and
`3692`. The 15.1% small-corpus KL improvement therefore did not predict hard
generation quality. The likely methodological fault is that specialist
calibration observed only the first 128 prompt tokens, not expert routing over
long algorithmic reasoning trajectories. This plan is rejected and must not
replace the original nonuniform r25 candidate. Candidate report SHA-256 is
`a342d68e4e28e7c433513dc8566ba6a4992e3aec7044c02366fdb1112ca765b3`;
strict paired comparison SHA-256 is
`717e1126b1103affb6dbdaf7be671527df42bd3bdf935380575897ac7ef6771b`.
The reproducible materialized artifact was removed after rejection to recover
its 37.48 GiB of unique blocks; plans, calibration, reports, and the incremental
materializer remain durable.

The first deterministic 100-task MBPP gate scored 74/100 for both nonuniform
r25 and r20. Their paired differences were balanced: r20 alone passed tasks
286, 146, 288, and 277, while r25 alone passed 216, 125, 501, and 398. The
pass/fail outcome matched on the other 92 tasks. This does not prove general
quality equivalence, but it clears the first substantive coding gate while
preserving r25's approximately 3 GiB payload advantage. Nonuniform r25 is now
the preferred 64 GB candidate pending broader coding and instruction tests.

#### Reasoning-trajectory expert addback

The rejected fixed-size protection experiment established that short prompt
observations do not predict long coding generation quality. The replacement
diagnostic uses `nemotron_mlx_trajectory_attribution.py` to teacher-force a
stored reasoning trajectory through the immutable source in layer-major order.
It captures only sparse post-prompt hidden states, so all 88 source layers run
with a measured 3.47-4.37 GiB peak instead of requiring a resident 74.78 GiB
source. Each retained-plan comparison is virtual and exact; no source payload
is rewritten.

On a 64-generated-token all-layer smoke, r25 mean local output relative-L2 was
`0.05495`. Per-trajectory restoration of 4, 8, and 16 removed experts reduced
the corresponding local errors sharply. Fully retained layers remained
bit-exact, directly isolating the observed drift to expert removal rather than
the official NVFP4 representation or source runner.

`nemotron_mlx_trajectory_plan.py` aggregates normalized route-weighted expert
output contribution and weights it by each layer's measured source-output
error. It is deliberately add-only: no expert retained by the nonuniform r25
template can be evicted. The first bounded plan uses six reasoning trajectories
with 256 generated tokens and 32 sparse states each, requires at least four and
at most sixteen additions per pruned layer, and restores 320 layer/expert slots.
Every slot costs exactly 3,104,788 bytes including router state, so the plan
adds `993,532,160` bytes and projects from `54.4974` to `55.4227 GiB`. It still
saves about 13.88 GiB against the unpruned no-MTP runtime and about 2.08 GiB
against r20.

Exact replay against the aggregated plan, rather than each trajectory's
optimistic private ranking, produced:

| Split | Tasks | Base r25 mean local output rel-L2 | Add320 | Improvement | Improved / equal / regressed layer-task pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | 6 | 0.064978 | 0.042465 | 34.65% | 204 / 36 / 0 |
| Disjoint holdout | 3 | 0.062326 | 0.048782 | 21.73% | 100 / 18 / 2 |

The two holdout regressions were negligible (`1.14e-4` and `7e-6` absolute
relative-L2 increases). The plan is stored at
`plans/trajectory-addback-g256-6/plan-r25-add320.json` under the model root;
SHA-256 is
`d1f798a6952262f8de2cbc6d7760da4c82a390c32396915487a280d8a5af5ad6`.
Calibration and holdout artifacts live under `trajectory-attribution/` and
occupy only about 188 MiB.

The plan was subsequently materialized at:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/candidate-trajectory-add320-mlx
```

The generalized incremental repacker hard-linked 55 mapping-identical groups
from r25 and rebuilt 34 changed MoE groups from the pinned source. All 89 groups
validated, no partial files remained, retained payloads are byte-identical, and
the exact artifact contains `59,509,695,904` payload bytes (`55.4227 GiB`). It
allocated `38.4031 GiB` of new disk blocks. Pack-report SHA-256 is
`d22167ccbc6115efcf8b444ac27a5cbfe37719288b40fb4be01c0e01d8873187`.

On `2+2=`, virtual source pruning and the physical resident runtime produced
identical 131,072-entry F32 logits: zero max error, zero relative-L2, identical
top-64, and identical top token. Both logit arrays have SHA-256
`5296da26ffec2d09058919e6b91ca8e4d01eee34c6d368e412da1641cea5f863`.
Paged-embedding residency peaked at `54.655 GiB`. A 32-token coding-prefix run
measured `24.315 tok/s`, `41.081 ms` median, and `41.968 ms` p95 decode latency,
so the addback did not reduce ordinary decode throughput relative to r25.

Two initial untouched generation smokes passed. Easy task `3651` passed all 33
official cases in 214 tokens. More importantly, hard task `abc375_e` repeat 0
passed all 52 official cases in 982 tokens; the same seed/repeat failed on r25
and passed on the prior r20 control. Report SHA-256 values are respectively
`2c6efe81dede2407613d9e002aff485f5a25c1065b914709ceaecf0da0611e9a` and
`fc8ba1610094692a2f6bf84e4a6dc8962ded8eba29b6d25c3acb85c8287aad0c`.

This is still not a promoted quality result. The six calibration tasks came
from the existing v6 gate and must not be counted as unbiased benchmark
evidence. The three holdout tasks were disjoint but remain a small local-error
gate. Promotion requires a substantive generation evaluation on untouched
tasks. Until that gate passes, the original nonuniform r25 artifact remains
preferred.

##### Complete-failure trajectory refinement

The next untouched hard gate replayed repeat 0 for `arc186_d`, `arc182_a`,
`arc193_b`, and `arc190_d`. Baseline r25 had failed all four. Add320 also failed
all four, but every generation completed without reaching the 8,192-token cap,
producing 13,841 reasoning/output tokens in total. The failures were one invalid
program and three wrong algorithms, so token budget alone was not the cause.
Gate-report SHA-256 is
`fad7a3ef2252367f65fe43ce09a29b09bcfa846bbaaa0de30eb8eb3c361498ba`.

All four complete candidate trajectories were teacher-forced through the
immutable source. The 7,998-token longest replay peaked at only `4.479 GiB`.
Sixty-four sparse states across each full trajectory then ranked experts still
absent from add320. A conservative second stage restores 160 additional
layer/expert slots, with a floor of two and cap of eight per pruned layer. It
adds `0.46265 GiB` and projects to `55.8854 GiB`, still about 1.62 GiB below
r20. The exact expert identities are recorded at:

```text
plans/trajectory-addback-hardfail-full4/plan-add160.json
SHA-256: 742fb56e52732e9142190d97e07066ecdb66ad5f9ccabf174d7f9f3338df05d3
```

Layers 14, 19, 23, 30, 32, 37, 81, 85, and 87 reached the eight-expert cap.
The largest mean local-error reductions came from layers 30, 87, 81, 23, 74,
83, 85, 17, and 37. Exact replay on the four training failures reduced mean
local output relative-L2 from `0.053922` to `0.043634` (19.08%); all 136 pruned
layer/trajectory pairs improved and 24 fully retained controls remained exact.

More importantly, three independent existing holdout trajectories improved
from `0.048782` to `0.045550` mean local output relative-L2 (6.62%). Ninety
pairs improved, 26 were exact, and four regressed negligibly; the worst absolute
increase was `2.67e-4`. This cleared the virtual evidence gate but not generation
acceptance.

The reproducible add320 runtime was removed before materialization, recovering
39 GiB. Add480 was then repacked directly from r25: 55 unchanged groups are
hard-linked and 34 groups were rewritten and individually validated. The final
payload is `55.8854 GiB`; paged resident loading peaked at `55.118 GiB`.
Full-vocabulary physical logits are bit-exact with the virtual source path
(`max_abs=0`, KL=0, top-64 overlap 64/64). Comparison-report SHA-256 is
`2f14d09e98a097af240071b4d141db08addcc84dedb394266fc8ca62f3a93e98`.

A disjoint deterministic repeat-1 generation gate then tested the same four
hard tasks. Add480 scored 0/4: `arc186_d` still truncated at 8,192 tokens,
`arc182_a` refused after 1,933 tokens, `arc193_b` changed from the earlier r25
truncation to a complete 1,634-token program but remained wrong, and `arc190_d`
produced a runtime error. Report SHA-256 is
`5b853b088febfbe9d2ef1b65557a0ea4feb2490e8f4bb0682f8737df1afd1e90`.
The candidate is therefore not promoted. These complete repeat-1 trajectories
were subsequently attributed, but the resulting cumulative add640 plan still
scored 0/4 on a second disjoint repeat. Failure-chasing attribution can reduce
local teacher error without proving that the source or a less-pruned candidate
would solve the task, so failed trajectories are no longer used as the primary
expert-selection signal. Both physical candidates were deleted; their plans
and reports remain reproducible.

##### Successful-trajectory attribution

The causal replacement starts from demonstrated pass/fail flips in the
deterministic 100-task MBPP gate. MBPP support in
`nemotron_mlx_trajectory_attribution.py` reconstructs the exact checkpoint chat
prompt and stored response, teacher-forces it through the immutable source, and
retains the same revision, dataset, report, token, plan, and tool-hash binding
as LiveCodeBench captures.

Four tasks passed under r20 but failed under r25: `286`, `146`, `288`, and
`277`. Attributing their known-successful r20 trajectories and adding 320
layer/expert slots directly to r25 reduced mean local output relative-L2 by
54.42% on those trajectories, with 136 improved, 24 exact, and zero regressed
layer/task pairs. Four inverse r25-success trajectories improved by 33.49%; one
of 160 pairs regressed by only `1.65e-4`. The plan SHA-256 is
`0b18fd3ece82c00d61f6b07b0e3ea4cffaf2c3c446e26892744b2c1bdcb69426`.
Its physical `55.4227 GiB` candidate was bit-exact with virtual pruning and
scored 74/100, gaining task 286 and losing task 351 relative to r25.

Task 351's known-correct r25 trajectory then supplied an explicit regression
guard. Restoring 80 more layer/expert slots reduced its local error by 40.28%
without a layer regression. Across the earlier success controls, mean local
error also improved by 3.97% and 9.19% for the r20-success and r25-success sets;
the worst absolute regressions were `9.61e-4` and `1.44e-4`. The final plan is:

```text
plans/mbpp-r20-success-add320-guard351-add80/plan-add80.json
SHA-256: 32062677e7aca9c9ea179bb43a714c1d0cee6d0ab99b48aa8fa7547bc90b93ef
```

The materialized runtime is:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/candidate-mbpp-success-guard400-mlx
```

It retains 314-512 experts per MoE layer, averages 394, contains
`59,758,078,944` bytes (`55.6540 GiB`) of validated payload, and peaked at
`54.887 GiB` with paged embeddings. Retained payloads and scales remain
byte-identical. Virtual and physical full-vocabulary logits were exactly equal.
After a later cleanup/rebuild, the pack report reproduced its original SHA-256
`b96c4350f481e534180659c6abf1a9077403370b3b8c1669e0cb41003022dc58`,
providing an independent deterministic-materialization check.

The full deterministic MBPP result is 75/100 (report SHA-256
`0c2e0f2df9009e98fcf90ac583e91d72715c8078591d7d7ed08fdc386d4e887e`).
Relative to r25 it gains tasks 286 and 277 and loses task 376; relative to r20
it gains 216, 125, 501, and 398 while losing 146, 376, and 288. This is a net
one-task improvement over both controls.

A final 40-slot task-376 guard reduced that trajectory's local error by 33.49%
but failed to recover task 376 and reintroduced the task-351 failure. Its
aggregate remained 75/100 only because task 50 flipped to pass. The report
SHA-256 is
`eddd752b0c6d2d4cfd266936987163b2172b95e861f5c333e8507b931bd9d04f`.
This guard440 plan is rejected and its physical artifact was deleted.

Two independent benchmark families then tested whether the MBPP gain merely
overfit its attribution source. On the complete corrected HumanEval gate,
guard400 tied r25 at 154/164. It gained `HumanEval/54`, lost `HumanEval/127`,
and matched pass/fail on the other 162 tasks. Report SHA-256 is
`002af6ad61f24a37546e7a908424f06026980024e5b29f2c255309d86aa26e58`.

The stronger matched hidden LiveCodeBench gate used the same 30 balanced v6
tasks, two low-effort samples, seeds, 8,192-token cap, and every official test
as the r25 control. Guard400 scored 36/60 samples versus 37/60, but covered
23/30 tasks versus 22/30 and had no task-level loss: 22 tasks passed under both,
one passed only under guard400, and seven failed under both. Its easy/medium/
hard sample scores were 19/12/5 versus r25's 18/15/4. The strict paired matrix
was 33 both-pass, four r25-only, three guard400-only, and 20 both-fail. This is a
mixed one-sample trade, not evidence of a broad regression; importantly, the
guard improves hard-task and task-coverage results instead of only its MBPP
source domain. The generation and paired-report SHA-256 values are:

```text
guard400 generation: 5b851e0e9268f44ca007a9ecf9d6b904b57259d568ba49af4e385a6fc9e142c9
strict paired report: 5a0c721d3442aba3874f18a15d4e77790b0af76d39bd916dd4be8c9be5a19e10
```

Performance remains intact. A 64-token coding control measured `24.573 tok/s`,
`40.623 ms` median, `41.519 ms` p95, and `54.887 GiB` peak. The exact shared
32K MTP vocabulary map was rebound to guard400 at
`mtp-vocab-map-bf16-e32768-mbpp-success-guard400`; its 128 KiB artifact is
byte-identical to the prior map but bound to guard400's pack-report hash. With
the shared 128-expert NVFP4 MTP sidecar, a 128-token control measured
`34.932 tok/s`, 79.03% acceptance, `1.408x` speedup, `55.490 GiB` peak, and
exact token integrity.

Preferred local launch:

```sh
MODEL_ROOT=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$MODEL_ROOT/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_speculative.py \
  --model-dir "$MODEL_ROOT/candidate-mbpp-success-guard400-mlx" \
  --mtp-sidecar "$MODEL_ROOT/mtp-sidecar-e128-nvfp4" \
  --mtp-lm-head \
    "$MODEL_ROOT/mtp-vocab-map-bf16-e32768-mbpp-success-guard400" \
  --max-new-tokens 512 \
  --warmup-cycles 10 \
  --margin-gib 0.5 \
  --cache-limit-mib 256 \
  --capture-rollback \
  --paged-embeddings \
  --embedding-cache-rows 256
```

Guard400 is therefore promoted as the preferred quality-oriented 64 GB
runtime. Nonuniform r25 remains the smaller control and rollback baseline,
saving 1.1566 GiB when memory headroom matters. Further expert additions are
not justified by local layer error alone and require a new causal quality gain.

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

Dense functional assignment then evaluated every retained expert on the exact
activation contexts of each removed expert. It remained 2.08% worse than hard
pruning on the sensitive layer-8 heldout mean, closing the remaining nearest-
prototype mapping gap.

`nemotron_mlx_router_distill.py` next tested a bounded router-only recovery
inspired by Router KD. It freezes all experts and projections, differentiates
only through the selected experts' gate scores, exports BF16 retained-router
rows, and keeps the source row whenever a disjoint per-layer validation split
does not improve. On the r30 success200 plan, 23/40 layers changed. Mean local
output relative-L2 improved only 0.75% (`0.054575` to `0.054165`), while mean
routed error improved 0.92%. Report SHA-256 is
`d293317760aa42e9ee5d6ab5e2cd8fee5601d1ffdbe51d7c8b0e7b5cca0c650a`.

The independent full-logit gate rejected the result after two coding cases.
Completion KL worsened from `0.046032` to `0.051560`; debug KL improved from
`0.049644` to `0.038965`, but its source top token changed. The trained path
therefore retained only 1/2 source top tokens versus 2/2 for unmodified r30.
Partial report SHA-256 is
`70c46a90d01eb2957cc8f4a4ae6d735c2ece72c5b99ecce82d6532127c70cfaf`.
Do not materialize these routers. This rejects teacher-forced local output MSE,
not end-to-end Router KD: the published method backpropagates next-token KL
through the complete compressed model, which this low-memory streamed
experiment does not approximate safely.

The follow-up `nemotron_mlx_kd_gradient_audit.py` establishes that a bounded
manual backward pass is technically possible with the current checkpoint.
Two-token input VJPs are finite and nonzero through representative Mamba,
attention, and LatentMoE blocks at a `5.266 GiB` peak. Mamba uses its production
path with bit-exact forward output. The inference-only BF16 Metal projections
in attention and `fc2_latent` have no VJP, but frozen native-BF16 matmul
fallbacks preserve forward output to `8.58e-7` and `3.34e-7` relative-L2 for
attention and MoE respectively. NVFP4 and FP8 paths remain quantized; MoE
expert indices are explicitly stop-gradient while selected gate scores retain
their sparse gradient. Audit-report SHA-256 is
`123dfc079c7f802ab0ea01f1f76256538f67c0f77c410aabd116e974f1141006`.

`nemotron_mlx_streamed_router_kd.py` now implements that manual reverse pass.
It atomically checkpoints every small layer-boundary activation and cotangent,
reloads exactly one frozen block at a time, stops gradients through discrete
expert IDs, and differentiates retained router scores against the unpruned
teacher's full 131,072-entry next-token distribution. Resumption is bound to
the immutable source-state, plan, tool, prompt, and optimization hashes.
Inference-only BF16 and ModelOpt FP8 projections use exact-value native MLX
fallbacks during training; retained NVFP4 expert payloads are not decoded,
modified, or rewritten.

The final two-token r30 mechanism proof passed all structural gates. Its
gradient-safe student matched the production virtual-pruning path at
`5.77e-9` KL, `1.746e-4` centered relative-L2, identical top-64 membership,
and identical top-1. All 40 MoE layers produced finite nonzero router
gradients. The full process peaked at `4.195 GiB`. A `1e-4` BF16 Adam-style
step was correctly rejected for overshooting; the guarded `5e-5` step reduced
teacher KL from `0.00532214` to `0.00319917` (39.89%), improved centered
relative-L2 from `0.11228` to `0.07227`, increased top-64 overlap from 59 to
62, and preserved the teacher top token. Every zero-gradient retained row was
verified unchanged after BF16 export.

```text
artifact: layer-distill/streamed-router-kd-r30-def-final/router.safetensors
artifact SHA-256: f1be61bb2383cd3eebc43da8dfb3811fbb3a76b280eeeec8ed1750005695fb72
report SHA-256: 9898268f825d06e016aaebb466aa4ff7688ea8f405879ad9403ce6573850e170
```

This closed the implementation gate, not the quality gate. The two-token
sidecar remains overfit and must not be packed or promoted.

The required multi-sample follow-up is now complete. Twenty-four training
objectives cover eight categories at 8-token, 16-token, and full-prefix
positions; validation uses eight disjoint full prompts. The accepted `5e-5`
step improved validation mean KL from `0.070435` to `0.064885` and maximum KL
from `0.214089` to `0.182079`, while preserving 7/8 source top tokens. All 40
router gradients were finite, 13,457 retained rows changed, and every
zero-gradient row remained exact. Artifact SHA-256 is
`db335b5b8c436deb85838152683919fb62f97511bdab9e2d86769e92946f9eb7`;
report SHA-256 is
`61a3f84cac040926eaac906e278f891537a132ae7e3fa5948e4c104c66f65053`.

Independent July 2024 LiveCodeBench logits rejected the complete update on
mean KL. Reverting the last 10 MoE routers produced the only composition that
passed both bounded gates: validation mean KL was `0.069264`, and the 12-case
coding mean KL was `0.146217`, both slightly better than unmodified r30. The
exact composed artifact SHA-256 is
`542584257abbac40745f850a5bd7f9da9ab1e958040c8b087c98f359344a1497`.

That composition was physically packed at `51.6059 GiB`. All 89 groups passed
write-time and cold-resume payload hashing, all 40 physical routers were
byte-identical to the sidecar, and virtual versus physical 131,072-way coding
logits were bit-exact. Paged resident inference peaked at `50.838 GiB` and
measured `24.139 tok/s`, with `41.391 ms` median decode latency.

Substantive generation nevertheless rejects it. Deterministic MBPP scored
73/100 versus 74/100 for unmodified r30, losing only task 125; complete
HumanEval scored 149/164 versus 150/164, losing only HumanEval/147. There were
no candidate-only wins. The paired report SHA-256 is
`c20cf1909fca87f2ebcf211376971b6f062ca61a7b45e9f26c98872e4760c0b1`.
Layer-30 source reversion and BF16 delta damping recovered MBPP 125, but full
reversion regressed the broad and disjoint coding means, while 0.75 damping
raised disjoint coding mean KL to `0.148401` and worst KL to `0.619237`.

Decision: the bounded streamed training mechanism is valid, but this
multi-sample Router-KD artifact is not a deployable quality recovery. Do not
promote or retain the physical candidate as the preferred runtime. Further
recovery should change expert capacity or the pruning allocation using
downstream-success evidence, not continue post-hoc router fitting on this
corpus. The rejected physical pack was deleted after its hashes, parity,
performance, and downstream reports were retained; it remains reproducible
from the pinned source, r30 plan, and composed router sidecar.

#### Aligned expert-width alternative

Nemotron's routed MLP width is 2,688, exactly 168 NVFP4 groups of 16 neurons.
`nemotron_mlx_width_prune.py` keeps all 512 experts and the original router but
selects 126 groups per expert for a 25% routed-payload cut. Matching up rows and
down columns can be sliced with their NVFP4 block scales and per-expert global
scales unchanged. The existing gather-QMM path accepts the 2,016-neuron shape,
so this representation also reduces selected-expert arithmetic by 25%.

An all-layer concatenated screen against uniform r25 whole-expert pruning found
13/40 width wins. Rechecking those layers on independent 128-token calibration
sequences confirmed 10/13, with mean local output-error ratios of `0.508` on
layer 1, `0.763` on layer 3, `0.850` on layer 8, `0.910` on layer 59, and
`0.943` on layer 70. The confirmed subset's mean ratio was `0.923`. This does
not justify width pruning globally; it supports a same-budget per-layer hybrid
materializer followed by full-logit validation.

That validation rejected naive promotion. A uniform-r25 hybrid produced a math
KL regression, and an eight-layer nonuniform hybrid lost coding top-1. Full-
logit ablation reduced the set to layers 1, 8, 19, and 54. Their combined
eight-category run improved mean KL from `0.07049` to `0.05254`, but lost the
tool-calling top token. Tool-specific ablation showed that layers 1, 8, and 19
each caused that flip; layer 54 alone preserved it. Physical packing therefore
remains blocked on broader validation of the layer-54-only plan rather than the
more attractive local-error aggregate.

The complete layer-54-only run subsequently passed: 8/8 top-1 was preserved,
mean KL improved from `0.07049` to `0.06130`, mean centered drift improved from
`0.09597` to `0.09381`, and worst KL remained slightly better. Its
expert-equivalent average is `385.9`, projecting approximately 54.8 GiB without
MTP, about 0.29 GiB above nonuniform r25. This is the sole width candidate to
materialize when disk headroom permits.

The incremental materializer avoided another full checkpoint by hard-linking
105 unchanged files from nonuniform r25 and writing only layer 54. The physical
candidate contains `54.7182 GiB` of indexed payload while consuming about
1.2 GiB of additional disk blocks. Its complete eight-token physical forward
is bit-exact with the accepted virtual plan (`max_abs=0`). Paged ordinary decode
measured `23.997 tok/s`, 41.71 ms median, and `53.953 GiB` peak. Candidate-bound
MTP measured `35.068 tok/s`, 76.92% acceptance, `1.433x` speedup, and
`54.553 GiB` peak with exact output integrity.

A deterministic 20-task MBPP gate then compared the physical hybrid directly
with nonuniform r25 under greedy decoding and the checkpoint chat template.
Both passed 15/20 tasks, matched pass/fail on all 20, and produced byte-identical
responses on 15. The remaining five generations differed without changing any
test outcome. This supports quality parity for the promoted layer-54 change,
but 20 tasks are not a final coding-quality acceptance set.

Expanding the same deterministic sample to 100 tasks exposed one regression:
the hybrid scored 73/100 versus 74/100 for nonuniform r25. Task 376 was the only
pass/fail disagreement; nonuniform r25 correctly replaced only repeated tuple
occurrences with `MSP`, while the hybrid replaced every member whose total
frequency exceeded one. There were no hybrid-only wins. Since the hybrid is
also 0.2207 GiB larger, layer 54 width pruning is rejected as the preferred
artifact despite its favorable logit and MTP throughput measurements.
The derived hybrid directory was removed on July 11, 2026 after its reports
were finalized; recreating it requires only the retained nonuniform-r25
candidate and recorded plan.

`nemotron_mlx_mbpp.py` keeps candidate weights resident, resets Mamba and KV
state between tasks, and writes an atomic report after every task. Reports bind
the model pack report, runtime metadata, tokenizer and chat template, dataset,
evaluator source, task selection, generation limit, and sandbox interpreter.
Generated code runs under `sandbox-exec` with writes confined to a temporary
task directory, network denied, and CPU, file-size, and wall-time limits.
Reports live in `quality/mbpp-*.json`; the four 5+15-task report hashes are:

```text
hybrid:    8b30755ca96a736d24b91e7adf402bad2ae638a8d9726ee29c97859b2a5d84f4
hybrid+5:  6e8f89dd4208874c8cd1a143fc3525e84934efebbc7dad888dc84bb4c462a086
control:   edafb43030fd976f85ecd9fbced55703bb60cf588efea11ec76f0bae4e669420
control+5: 0ccb79fc77d3c81480b5cd770bb517ae98c31ee89f3d5729dc085606402f5a5c
hybrid+20: 7e0e0ee8a0035b3c364d99931f442706867bc4cb7fa02663e432fa71a9258e8f
control+20:518d73cee21025f4b7c7cb05310b9082c5fc64e124f1cbf9126bf0e522070f5c
r20-100:   f4b865bd1696480dc6ba8dde698ed17da99aec7fd43a99a4a61a80b7c1548e00
```

HumanEval provides an independent function-completion gate with different
prompt and test structure. `nemotron_mlx_humaneval.py` handles either full
functions or body-only completions, restores imports from the supplied prompt,
and executes the official `check(candidate)` function under the same sandbox.
On the first deterministic 20-task sample, nonuniform r25 and r20 both scored
17/20 with identical pass/fail outcomes on all tasks, despite only five
byte-identical responses. Neither run approached the 768-token generation cap.
The provenance-bound report hashes are:

```text
r25: 9914e2f13b6636b26803052d8e8f34cc2990d345a4a4ce1fcee5d1684f244858
r20: 56715e5352053f4e7ad5a8fffa16b631281ffb118fc590051f9a9f998c5c9460
```

The complete 164-task r25 run initially reported 152/164. Failure review found
that the adapter restored imports but omitted prompt-defined helpers when a
model response supplied a complete target function. Preserving the full prompt
preamble and provenance-bound offline rescoring corrected tasks 50 and 38;
task 32 remained a genuine algorithm failure after its `poly` helper was
restored. The accepted result is 154/164 (93.90%), with no timeouts, syntax
errors, or 768-token truncations. The longest response used 463 tokens.

```text
generation report: 69c2575284f9511b54c81d3900753ef8acd6f1b6c2e76e5ea2a98c2124ec8140
corrected rescore:  a3abbab2ce83fc211ce8884caea4286a30eafdf697a85eeab0a8e6724bbf0961
```

LiveCodeBench adds longer stdin/stdout competitive-programming problems. A
plain user prompt caused the model to spend its 2,048-token budget reasoning in
prose. Continuing an assistant message prefilled with a Python code fence
forced direct code generation: on the first hard passing task this reduced
output from 1,469 to 266 tokens and latency from 76.8 to 25.6 seconds without
changing the result.

The first deterministic 10-task public-test gate scored 5/10: easy 2/2, medium
2/3, and hard 1/5. One hard answer exhausted 2,048 tokens and is explicitly
reported as truncated; the other failures completed and produced wrong output.
This is an initial public-test gate, not a comparable official LiveCodeBench
score, but it exposes a real capability gap hidden by HumanEval.

```text
tasks 0-4: a21bdec5a83c6023a15318e45874d552fe7e3fadf7241565e9171c6959295eb9
tasks 5-9: 4730cfdcb7c0d7eb2dbbdc0b772e31e77bdb45b6150f5572783e1d338e0bddae
```

The follow-up gate is a deterministic 30-task balanced sample: 10 each of easy,
medium, and hard, interleaved by rank. Validate every input and the resident
memory requirement without loading weights:

```bash
MODEL_ROOT=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
"$MODEL_ROOT/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_livecodebench.py \
  --model-dir "$MODEL_ROOT/candidate-nonuniform-r25-mlx" \
  --dataset "$MODEL_ROOT/source-notes/omlx/omlx/eval/data/livecodebench.jsonl" \
  --output "$MODEL_ROOT/quality/livecodebench-r25-stratified-10x3.json" \
  --samples-per-difficulty 10 \
  --max-new-tokens 2048 \
  --python /opt/homebrew/bin/python3 \
  --dry-run
```

Remove only `--dry-run` to execute. The job is atomically resumable at task
boundaries and writes generated source, test failures, timings, and explicit
truncation state after every problem.

The run completed at 16/30 (53.33%): easy 9/10, medium 5/10, and hard 2/10.
Only `arc186_d` reached the 2,048-token cap; the other 13 failures completed and
produced incorrect public-test output. Generation used 8,765 tokens and 584.4
seconds. The 70,890-byte report is bound to the exact evaluator and generation
helper sources:

```text
34d183647e3cd59d5b8902181eca4e10b0203c52375dea3c3d29f666607fe1bb
```

This confirms a specific quality profile: basic contest tasks are reliable,
medium tasks are mixed, and hard algorithm design remains weak. It does not
indicate a runtime or compression-integrity failure.

That 53.33% result is not comparable to NVIDIA's published LiveCodeBench
score. It used reasoning disabled, greedy decoding, one sample per task, a
2,048-token cap, public tests only, and an oMLX dataset copy without dated-split
metadata. NVIDIA's published protocol enables thinking, samples at temperature
1.0 and top-p 0.95, permits up to 131,072 generated tokens, and uses eight
repeats on the dated v5 or v6 split. NVIDIA separately reports that its official
NVFP4 checkpoint is close to BF16 on LiveCodeBench v6 (78.57 versus 78.25), so
NVFP4 itself does not explain the local smoke score.

The evaluator now supports thinking, low-effort thinking, stochastic sampling,
deterministic repeat seeds, and repeat-aware resumption. It separates reasoning
from the final answer and reports sample pass@1 separately from task pass-any.
Every report also lists mismatches against NVIDIA's reference protocol so a
bounded local run cannot be mistaken for an official score.

A protocol-direction smoke used low-effort thinking, temperature 1.0, top-p
0.95, and four seeds on the previously failed hard problem `abc391_f`. Two of
four samples passed all three public-case bundles, one reached the third bundle
before returning a wrong answer, and one emitted an invalid refusal. None
truncated. Generation used 8,738 tokens over 417.0 seconds. The earlier greedy,
reasoning-disabled attempt failed its first public bundle. This proves that the
evaluation protocol materially suppressed the prior result; it does not yet
establish candidate parity because low-effort mode, four repeats, the 4,096
token cap, the local dataset, and public-only scoring remain mismatched.

```text
one easy low-effort smoke: 4e690b496e4ea1c338006e75693c9abe06fde9f8826b5097e6cb9f8226ac7c7a
one hard low-effort smoke: d1def36dd8bc39ec3cd9f6ef91e3ed467b62627fb4e271de5b91b3d50d8a017b
hard four-repeat smoke:    0fd408862b6d368df41c948bfaf35622b5c2b602ff3179ca92d5bd7b954d6cbb
```

A full-thinking sample of the same hard task then reached an 8,192-token local
cap after 371.1 seconds without emitting `</think>` or final code. The 27,388
character trace was coherent, had 141 unique paragraphs with no repeated
paragraphs, and was still validating its proposed heap algorithm when cut off.
This is a cap/verbosity failure rather than evidence of a malformed decode, and
it explains why NVIDIA allows 131,072 output tokens. Its report hash is
`c97dc56fd14a07e0a8fef51eaa5bc939ff3adf1439a8c817427f7fc331b0edb1`.
Low-effort thinking is therefore the practical local screening mode; final
quality acceptance still needs bounded paired full-thinking runs with a much
higher cap.

Long-trace stop detection now matches the checkpoint's exact single-token
`</think>` and code-fence delimiters instead of decoding the complete growing
response after every token. Replaying the 8,191-token trace preserved the stop
decision while reducing cumulative stop-scan time from 3.810 seconds to 0.00078
seconds. This removes quadratic tokenizer overhead from high-cap evaluations;
it does not alter model sampling or generated token IDs.

References: [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16),
[NVIDIA quantization results](https://docs.nvidia.com/nemotron/nightly/nemotron/super3/quantization.html),
and [NVIDIA reproducibility configuration](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/nemotron-3-super/reproducibility.md).

### Dated LiveCodeBench Protocol

`nemotron_livecodebench_dataset.py` range-reads the pinned official Parquet
release at revision `c52cd175916e995019dcd848d1054b419d2e70b5`. It fetched
only selected non-private columns and strictly matched every official prompt,
starter, difficulty, and public case against the existing 1,055-row oMLX copy.
The complete catalogs required less than 1.9 MiB of network transfer each:

| Split | Tasks | Easy | Medium | Hard | Metadata transfer | Hidden-test transfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 2024-07..2024-12 | 315 | 78 | 102 | 135 | 1,782,594 B | 2,331,147,468 B |
| v6 2024-08..2025-05 | 454 | 110 | 141 | 203 | 1,847,732 B | 2,478,535,067 B |

The private strings are dictionary-encoded as one large page per Parquet row
group and have no offset index. Individual hidden rows therefore cannot be
retrieved with small range requests.

After explicit approval, the v6 private columns were streamed directly from
the five required immutable Xet objects. No Hugging Face cache or raw Parquet
file was created. Each output shard was validated and committed to resumable
state before continuing. The completed artifact contains 454 tasks and 15,684
hidden tests:

```text
transfer: 2,478,970,742 bytes
output:   4,033,506,110 bytes
state:    quality/livecodebench-private-v6-2408-2505.state.json
data:     quality/livecodebench-private-v6-2408-2505/
index:    quality/livecodebench-private-v6-2408-2505.index.json
```

The random-access index records each task's file, byte offset, line length, and
test count. It validates every output file hash and lets bounded evaluations
read only selected task rows. With this private index attached, a v6 low-budget
dry run using NVIDIA's complete settings reports zero protocol mismatches.

The evaluator now mirrors NVIDIA's AAI LiveCodeBench prompt construction,
supports both stdin and functional tasks, defaults to the official six-second
execution timeout and all public cases, and binds reports to the dated state
file. `standard` and `low-budget` protocol profiles distinguish full thinking
from NVIDIA's low-effort v6 configuration. With official generation values and
the dated state, dry runs report only `public_tests_only` as a mismatch.

A bounded low-budget v6 smoke selected deterministic tasks `3525` (medium,
functional) and `abc391_f` (hard, stdin). Task `3525` passed both public cases
after 1,900 generated tokens. `abc391_f` reached the deliberately reduced
4,096-token cap before final code. The report scored 1/2 and is not an accuracy
estimate; it validates both task paths and confirms that small token ceilings
remain unsuitable for hard reasoning. Report SHA-256:
`6265a9dde1d822df861461b0dd78116863d0a02c1aa1abf3d10035cd8da9188f`.
After Xet-hash provenance and final six-second/all-case harness hardening, the
stored code was rescored without regeneration. Both outcomes were unchanged;
the final provenance-bound rescore SHA-256 is
`e914c3fcdbd43be4522ca25d9650f1e67462a3b0a18bb26d296ff6db0194d62e`.
The same stored generations were then rescored against the official private
corpus. Task `3525` passed its two public and 40 hidden functional cases;
`abc391_f` remained a generation-cap failure and did not reach hidden cases.
No outcome changed. The final full-data rescore SHA-256 is
`4490f5e063a1b040c7d3ce58028ce642a868b4de3fbb93e775c20be4d13d71d5`.

The first newly generated hard-task hidden evaluation used two low-budget
samples of `abc391_f` with an 8,192-token local ceiling. One sample exhausted
the ceiling without a valid final answer. The second completed in 7,023 tokens
and passed all 43 official cases: three public and 40 hidden. Sample pass@1 was
1/2 and task pass-any was 100%; this is a single-task protocol validation, not
an aggregate LiveCodeBench score. Report SHA-256:
`363c49b1969ea8b6b6e44d7f53f12b137582065416dfce521c3e2d047273f625`.

The next staged gate ran 30 deterministic dated-v6 tasks, balanced as 10 easy,
10 medium, and 10 hard, with two low-budget samples each and every official
public/private case. It completed 60/60 generations with no infrastructure
failure:

| Slice | Passed samples | Pass@1 | Tasks passing at least once |
| --- | ---: | ---: | ---: |
| Easy | 18/20 | 90% | 9/10 |
| Medium | 15/20 | 75% | 9/10 |
| Hard | 4/20 | 20% | 4/10 |
| Overall | 37/60 | 61.67% | 22/30 (73.33%) |

The run generated 124,723 tokens over 5,688.3 generation seconds (94.8
minutes), averaging 21.93 generated tokens/s. Failure classification was 14
wrong answers, six 8,192-token truncations, and three official six-second test
timeouts. All truncations were hard tasks. Excluding truncations gives 37/54
(68.52%); excluding truncations and timeouts gives 37/51 (72.55%). The Wilson
95% interval for the raw 37/60 sample proportion is 49.0-72.9%.

This result is materially stronger than the old reasoning-disabled public-test
smoke, but it is not comparable to NVIDIA's published 78.57% NVFP4 v6 score.
It uses balanced difficulty sampling, low-effort thinking, two repeats, and an
8,192-token cap; NVIDIA uses the natural 454-task distribution, full thinking,
eight repeats, and a 131,072-token cap. Even crediting every truncation as a
pass would produce only 43/60 (71.67%), so the cap is not the sole observed
gap. A matched unpruned control is required before attributing the remaining
hard-task deficit to the 25% nonuniform pruning plan.

```text
report: e8c3aceb1913000c155abfb8c055bb19f25dfc6438dc1bfa9b6c61f89462c4b3
```

```bash
MODEL_ROOT=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
python3 nemotron/tools/nemotron_livecodebench_dataset.py \
  --manifest nemotron/data/livecodebench_release_v6_manifest.json \
  --local-dataset "$MODEL_ROOT/source-notes/omlx/omlx/eval/data/livecodebench.jsonl" \
  --output "$MODEL_ROOT/quality/livecodebench-official-public-v6-2408-2505.jsonl" \
  --state "$MODEL_ROOT/quality/livecodebench-official-public-v6-2408-2505.state.json" \
  --start-date 2024-08-01 --end-date 2025-05-31
```

### First 20% Candidate

The first full activation-informed candidate lives at:

```text
/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/candidate-oqe512-r20-mlx
```

This derived candidate was removed on July 11, 2026 after the paired MBPP and
HumanEval controls completed. Its reports remain in `quality/`, and the
immutable source and recorded plan are sufficient to reproduce it.

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
  while applying each small SSM update in the same order as decode. A
  Nemotron-specific Metal kernel now keeps 2-8 unpadded recurrent steps inside
  one launch, including optional accepted-prefix state capture. Longer, padded,
  and multi-capture sequences retain the reference loop. Under the validated
  MLX `0.32.0` path, layer output drift is at most `1.20e-7` and recurrent/
  captured-state drift is at most `3.05e-5`; full verifier logits, rollback,
  accepted cache state, and token identity remain separately gated. Set
  `NEMOTRON_DISABLE_SHORT_SSM=1` to force the reference loop for diagnostics.
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

On remove400, the matched short-block verifier comparison measured:

| Block | Token-loop batch | Fused SSM batch | Improvement |
| ---: | ---: | ---: | ---: |
| 2 | 47.479 ms | 45.651 ms | 3.85% |
| 3 | 60.250 ms | 57.612 ms | 4.38% |

Full-logit and cache checks passed at block sizes 2, 3, 4, and 8. Relative L2
remained below `1.45e-7`, maximum absolute drift below `3.06e-5`, and every
target top-1 remained unchanged.

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
`1aa880d0ea1dfddef567785ebc9a384f2a600d18` supplied the initial cacheless
serial control. `compute_mtp_single_step` calls `forward_single_position` after
target verification with the target's final normalized hidden state and the
freshly sampled accepted token. The MTP head computes
`eh_proj([enorm(embedding), hnorm(hidden)])`, runs its BF16 attention and latent
MoE layers, applies its own final norm, and reuses the target `lm_head`.

That helper is not NVIDIA's complete deployed speculative contract. The
[current NVIDIA deployment guide](https://docs.nvidia.com/nemotron/latest/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html)
uses vLLM autoregressive MTP. Pinned vLLM commit
`2bd8957627bfb5668c46f2bc359bef47d371270c` explicitly advances draft
positions because each step produces new KV, and its next draft prefill
replays accepted target states while excluding rejected positions. The exact
source files and hashes are recorded under `source-notes/vllm-mtp/`.

The production Apple path therefore owns a separate `NemotronMTPCache` per
sequence. Prompt mode preloads prompt transitions, recursive drafts append
speculative K/V, and verification restores the cycle checkpoint before
replaying only accepted transitions from authoritative target hidden states.
Cacheless mode remains an explicit numerical and performance control. An empty
cache is byte-identical to the old one-token path: logits, hidden state, route
scores, and selected experts all had zero maximum absolute difference.

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

### Fixed-Budget Success-Aware Expert Allocation

`nemotron_mlx_trajectory_swap_plan.py` reallocates expert identities without
changing any layer's retained count. It starts from nonuniform r25, treats the
400-slot guard400 difference as the eligible addback pool, protects experts
unobserved by calibration, and evicts only retained experts outside configurable
broad, specialist, recovery, and regression-guard cores. Retained ModelOpt
payloads and scales are never requantized.

The accepted guard50 plan combines the original broad calibration, a disjoint
224-token specialist calibration, four r20-only successful MBPP trajectories,
four inverse r25-success controls, and task 351 as a stricter regression guard:

```text
plans/mbpp-success-swap400-r25size/plan-swap400-sensitivity-guard50.json
SHA-256: 8a67086c830ee87d267c8c4296c890b6621c904b9fb5d1c297eb3560a67894cf
```

It changes 400 layer/expert identities across 34 MoE layers while retaining
exactly the r25 expert budget and `54.4974 GiB` payload. On the untouched
eight-category logit gate it preserved 7/8 source top tokens, improved centered
relative-L2 from `0.07554` to `0.07183`, and produced mean KL `0.08318` versus
r25's `0.08358`. Materialized and virtual 131,072-way logits for `2+2=` are
bit-exact (`max_abs=0` and identical array SHA-256), and resident peak is
`53.729 GiB`.

The complete deterministic MBPP result is 75/100, exactly matching every
guard400 pass/fail outcome while improving over r25's 74/100. The complete
HumanEval result is 155/164 versus 154/164 for both controls. Relative to r25,
it gains `HumanEval/108` and `/54` and loses `/130`; the lost task generated an
incorrect 768-token response, so the aggregate gain is not treated as clean
dominance. Report SHA-256 values are:

```text
MBPP:      63c6aa95140a6d35684304e479a514cb7b1fe2d0c6d15275cff9c719e0a4b7a6
HumanEval: ebecdd794146ca7e7a67934f934a3708b916d018c775af024292f9428def5a63
pack:      c3418ade6d5d7bafe5b5841a6b27aecba67fed4e0dcaac8a5b98685505fef934
```

A 64-token coding control measured `24.462 tok/s`, `40.856 ms` median, and
`53.730 GiB` peak. The exact shared 32K MTP vocabulary map is bound to this
pack at `mtp-vocab-map-bf16-e32768-mbpp-success-swap400-r25size`. With the
existing 128-expert NVFP4 MTP sidecar, a 128-token control measured
`36.538 tok/s`, 85.0% draft acceptance, `1.491x` speedup, and `54.333 GiB`
peak with exact token identity.

```sh
MODEL_ROOT=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$MODEL_ROOT/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_speculative.py \
  --model-dir "$MODEL_ROOT/candidate-mbpp-success-swap400-r25size-mlx" \
  --mtp-sidecar "$MODEL_ROOT/mtp-sidecar-e128-nvfp4" \
  --mtp-lm-head \
    "$MODEL_ROOT/mtp-vocab-map-bf16-e32768-mbpp-success-swap400-r25size" \
  --max-new-tokens 512 --warmup-cycles 10 --margin-gib 0.5 \
  --cache-limit-mib 256 --capture-rollback --paged-embeddings \
  --embedding-cache-rows 256
```

The matched hidden LiveCodeBench replay scored 36/60 samples and covered 22/30
tasks. R25 scored 37/60 and 22/30; guard400 scored 36/60 and 23/30. Against r25,
the strict sample matrix is 30 both-pass, seven r25-only, six candidate-only,
and 17 both-fail. Easy/medium/hard sample scores changed from 18/15/4 to
19/14/3. This is a real mixed trade rather than strict dominance, but there is
no broad category collapse. Paired report SHA-256 values are:

```text
versus r25:      96bf57ce1d12c6ff01bcba7c83a1e94b4092e9a8f8c06036a6b78135c80199ef
versus guard400: 56a33bc838e088d3eb19a55b8110fe3d73a1da7fb439c61a3fb0ac822299f90d
```

The fixed-budget candidate is promoted as the preferred balanced 64 GB runtime:
it retains r25's size, improves MBPP and HumanEval by one task each, preserves
r25's hidden LiveCodeBench task coverage, and matches guard400's sample score
while saving 1.1566 GiB. Guard400 remains the quality-headroom rollback for
users willing to spend that memory for one additional covered hidden task.

### R30 Success-Aware Virtual Candidate

The next size tier starts from the held-out nonuniform r30 allocation: 14,360
routed-expert slots, 359 per layer on average, and approximately `51.49 GiB`
of projected no-MTP payload. `nemotron_mlx_plan_union.py` creates a
provenance-bound candidate pool without treating a union as materializable.
Its trajectory-only mode imports only experts recorded in an accepted plan's
`trajectory_swap.by_layer.added`, and can rank and bound that pool by its
recorded joint evidence. This avoids accidentally importing unrelated r25
survivors when r25 and r30 use different layer budgets.

Fresh r30 attribution reused the immutable source captures for four r20-only
MBPP successes, four inverse controls, and task 351. Applying all 400 prior
success candidates reduced local routed-output error strongly, but failed the
broad gate: mean KL was 0.9% worse and worst KL was 12.6% worse than base r30.
That plan remains diagnostic and must not be materialized.

The bounded plan retains the strongest 200 candidates by r30 combined evidence
and swaps them into 31 layers without changing any layer's expert count. All
200 swaps have positive joint score; broad, specialist, trajectory, guard, and
calibration-unobserved cores remain protected:

```text
plans/r30-success-aware/plan-r30-success-swap200.json
SHA-256: 535d5c7b30cc8cd8044a8e65b566477e4f104f7b64928465afd477012625a68f
```

On stored trajectories, summed per-layer routed-output relative-L2 versus base
r30 fell by 27.6% on the four recovery cases, 22.8% on the four controls, and
31.3% on task 351. On the untouched eight-category, 16-token full-logit gate,
the candidate retained the same 6/8 source top tokens while improving mean KL
from `0.11707` to `0.10903`, mean centered relative-L2 from `0.07797` to
`0.07097`, mean top-64 overlap from `54.125` to `55.125`, and worst KL from
`0.32592` to `0.32524`.

```text
full-logit report SHA-256:
8b7626e5e6e669edd89bcde5403d943276b6a77d3c04cdcd818c9c7c34566d2b
```

The plan was subsequently materialized incrementally from the preferred r25
runtime. It hard-linked 53 unchanged groups, rewrote and validated 36 groups,
and contains `51.6059 GiB` of logical payload while allocating `37.6605 GiB`
of new disk blocks. Pack-report SHA-256 is
`1982b4c7c33aa442df62cab1861d86abe5305c2488a4ae65d2146a2b3b64178a`.
Virtual and physical `2+2=` logits are bit-exact across all 131,072 entries
(`max_abs=0`, KL=0, top-64 overlap 64/64); comparison-report SHA-256 is
`2f14d09e98a097af240071b4d141db08addcc84dedb394266fc8ca62f3a93e98`.

Resident paged decode peaked at `50.838 GiB`, saving exactly `2.892 GiB`
relative to the preferred r25 runtime. A 64-token coding control measured
`24.404 tok/s`, `40.913 ms` median, and `41.814 ms` p95, preserving r25's
performance. Downstream quality did not pass promotion, however. Deterministic
MBPP scored 74/100 versus r25's 75/100, with two r30-only and three r25-only
passes. HumanEval scored 150/164 versus 155/164, with one r30-only and six
r25-only passes. Report SHA-256 values are:

```text
MBPP:     84f190976ff9a91ade35629971c332857996a5d30be3c3122e6ecb157e1df9c9
HumanEval: f26a50c06c59a5351d81b9e71b320b03ceb5a205cef41e80dc6c5043085fd7c3
```

Two fixed-size HumanEval-guard refinements then used source-teacher attribution
from the six proven r25-success/r30-failure trajectories and HumanEval/130 as
an inverse guard. The 100-swap plan improved every aggregate local trajectory
group but worsened broad mean KL by 12.5% and worst KL by 29.6%. A minimal
36-swap plan, one replacement per pruned MoE layer, still worsened broad mean
KL by 5.3% and worst KL by 7.9%. Their broad-report SHA-256 values are
`9caa97275701bd186f14c29301ce7ca5ca3d1e11fea4b9280be554c658e06066`
and `8cea52e8298f362d8f060f4587ab4ad0ee6afcba34c2b8044bd5555e03625307`.
Both are rejected before materialization.

Decision: r30 proves that a `~50.84 GiB` resident profile can run at full
speed, but its measured coding-quality loss is too large for promotion as the
default. It remains a reproducible memory-first diagnostic. Further work
should test an intermediate r25-r30 expert budget or training-aware repair;
do not continue failure-specific post-hoc swaps merely because they improve
teacher-forced local error.

### Intermediate R27.5 Nested Candidate

`nemotron_mlx_layer_allocate.py` now accepts named exact global expert totals,
not only totals represented by a uniform source plan. Two direct dynamic-
programming allocations at 14,860 experts were rejected virtually. The first
used a stale sensitivity range ending at r35 and collapsed tool calling. The
corrected r10-r45 allocation still regressed the first five independent broad
categories. Exact budgeting is valid, but changing layer budgets also changed
too many accepted r25 expert identities.

`nemotron_mlx_nested_thin.py` instead removes 500 experts strictly from the
preferred r25 survivor set. It leaves all r0 layers untouched and preserves
the plan's broad, specialist, trajectory, and unobserved protection catalogs.
The calibration-only plan passed the eight-category broad gate, but increased
aggregate routed-output loss by 21-65% on each of ten known r25-success coding
trajectories. Global trajectory-weighted reranking reduced that local loss but
caused tool-calling KL above 2.0, so those plans were rejected.

The accepted virtual refinement uses bounded same-layer swaps over the broad-
passing nested plan. Screening found a sharp boundary: 150 swaps retained the
tool-calling top token and improved its KL from `0.19711` to `0.17233`; 200
swaps changed the top token and raised KL to `2.24992`. Repair150 changes 33
layers while preserving exactly 14,860 routed-expert slots, or 371.5 per MoE
layer. Its final plan is:

```text
plans/r275-nested-repair/plan-r275-repair150.json
SHA-256: 5cc931829f4ddc97c254e336904e3c28571f6b2250980958e8ad030eab398cf0
```

On the independent eight-category, 16-token full-logit gate, repair150 retains
the same 7/8 source top tokens as preferred r25. Mean KL improves from
`0.08318` to `0.07954`, worst KL from `0.19711` to `0.18961`, and both coding
cases improve (`0.03924 -> 0.02134` and `0.03622 -> 0.03010`). Mean centered
relative-L2 regresses from `0.07183` to `0.07593`, and mean top-64 overlap
falls from 55.375 to 55.0, so this remains a candidate pending downstream
generation rather than a promoted default. Broad-report SHA-256 is
`b80327ae1430ee7334e97dfb5c04b73ad18dae51da6605cb411e70178d5b282e`.

The final incremental materialization hard-links 55 mapping-identical groups,
rewrites and validates 34 groups, and contains `53.0516 GiB` logical payload
while allocating `36.0321 GiB` of new blocks. This saves `1.4458 GiB` versus
preferred r25. Pack-report SHA-256 is
`b6695961e722f90a5f5cd2f739be1f7b963660ed53ff17dd075ee17497f20d25`.
Virtual source pruning and the physical pack produce bit-identical `2+2=`
logits across all 131,072 entries; comparison-report SHA-256 is
`1a7825da5c3d0bc107f725e8e56b31256fb8543d3bd918aa2d03780d8875a8d3`.

With `iogpu.wired_limit_mb=56320`, the guarded resident run peaked at
`52.283 GiB` (`52.225 GiB` active) and decoded 63 measured transitions at
`23.289 tok/s`, with `42.925 ms` median and `43.354 ms` p95 latency. The
preflight requires 54,016 MiB with paged embeddings and a 0.5 GiB margin;
do not bypass that check. Resident-log SHA-256 is
`a8b2ed469b1d8d4716843fc02401bfbed31369314e71b01b1d2b6319ae5c6402`.

Paired generation gates do not justify promoting repair150 over preferred r25.
It scores 74/100 MBPP versus 75/100: repair150 alone passes task 376, while r25
alone passes tasks 342 and 39. It scores 153/164 HumanEval versus 155/164:
repair150 alone passes `HumanEval/130`, while r25 alone passes `/127`, `/129`,
and `/134`. The two MBPP report SHA-256 values are
`164fe7579ae30c51d9eccafa31125a8b0db8fae9784fa4339f9d3f984940a73e` and
`bcf4b713fc97fe9f7da1a46e848bead35c28bcb17e9a61d61f4f2ca7010a0cf7`;
the HumanEval report SHA-256 is
`4aae2070ca8503373432c214b1b9ccd8a6620ce1be3c181489e3610cbd7f3807`.
Repair150 therefore remains a reproducible memory-first runtime that saves
about 1.45 GiB, not the balanced default.

`nemotron_mlx_targeted_repair.py` tested whether source-teacher attribution on
the five lost tasks could recover quality through fixed-size same-layer swaps,
while the two repair150-only successes acted as inverse guards. This path was
rejected before materialization. Even the 10- and 20-swap plans flipped the
tool-calling top token and raised its KL from `0.172329` to `2.149887` and
`2.150421`; report SHA-256 values are
`49edf463f8638eb6d3ee2191364f1c46ebd78d20fee4e1ee4adbf5b80150f6f0` and
`50c6e2945d0f8e4ec2cf7fe096a081ff438457f1341a9c4f9f847fa32be9299b`.
The 40-swap eight-category gate confirmed the failure: mean KL rose to
`0.335244`, only 4/8 source top tokens survived, and tool-calling KL reached
`2.166359`. Its report SHA-256 is
`01374d07bdad1e837e5a25fecf55f3afe7562fd76a0de62209a8c00e05e4be23`.
Do not materialize these targeted plans or infer end-to-end repair from lower
teacher-forced route-output error.

### Nested Safe-Memory Frontier And IOGPU Incident

A strict r25-nested sweep next tested whether the r27.5 repair machinery had
stopped too early. Removing 400 experts from the preferred survivor set
produced a `53.3408 GiB` logical runtime and saved `1.1566 GiB`. It scored
74/100 MBPP versus preferred r25's 75/100 and tied HumanEval at 155/164. Its
64-token resident run reached `23.434 tok/s` and peaked at `52.572 GiB`.
Plan SHA-256 is
`3d1f1a2639e6361b320b18d65995f4d48e23601f2646b9320c59c725b3cad8eb`.

The matched hidden LiveCodeBench run was interrupted after four passing
samples by a full macOS kernel panic. The panic was not an ordinary OOM:
memory pressure and compressor state were healthy, while the panicked Python
task owned 3,467,243 pages. The exact signature was
`IOGPUGroupMemory::remove_memory_object() memory object not found` in
`IOGPUFamily(130.13)`. Panic-report SHA-256 is
`26292da55fefa128be302ccc9b62bd7351da7c1d85726dbb85a458ce2222de04`.

MLX 0.32.0 allocator source explains the relevant boundary. Cached Metal
buffers are forcibly released once active plus cached memory reaches 95% of
`recommendedMaxWorkingSetSize`. With the 55 GiB wired setting, that threshold
was 52.25 GiB, below remove400's measured peak. Near-cap decoding therefore
repeatedly exercised residency-set removal, matching the panic path. The
resident reset now synchronizes Metal, zeros Mamba recurrence in place, and
retains allocated KV capacity. Extended evaluations also require both:

- payload plus explicit margin at or below 85% of physical memory;
- payload plus margin below MLX's allocator-GC threshold.

The MBPP, HumanEval, and LiveCodeBench runners use a bounded 512 MiB reuse
cache and log active, cached, and peak Metal memory after each sample.
`--allow-high-memory-risk` is an explicit attended override, not a production
or unattended default.

The rare 56 GiB stop was traced to MBPP task 380's 793-token prompt. A whole-
prompt remove400 control peaked at `54.520 GiB`, with active plus cache ending
at `54.121 GiB` against the 57 GiB cap's `54.15 GiB` allocator-GC boundary.
Stateful 128-token prefill advances the same Mamba recurrence and attention KV
state while bounding projection workspace. It reduced peak to `53.461 GiB`
and active plus cache to `53.072 GiB`; prefill time rose from 14.36 to 15.63
seconds. Full-vocabulary final logits retained the same argmax and 10/10 top
tokens, with cosine `0.999999978`, KL `1.89e-7`, and maximum absolute drift
`0.06938`. The complete task then passed with the exact prior 26-token response
byte-for-byte and peaked at `53.457 GiB`. MBPP, HumanEval, and LiveCodeBench now
bind `prefill_chunk_size` into report identity and default to 128. This controls
prompt workspace; the conservative extended-run guard remains necessary for
long generated KV state.

The complete deterministic 100-task rerun then scored the same 74/100 and
generated the same 5,622 tokens. Every response and extracted program matched
the prior whole-prompt report byte-for-byte. The new report and explicit
cross-runtime comparison SHA-256 values are
`c9ad1e2cde24a81ec8dbedfb820240a6987ff424587a53e4648f532992b8cfbb`
and `03214dbe9e0cfc3c51425ba8226814e6b46d62a2cbad5603405c3406247b7de1`.

The hidden 30-task, two-repeat LiveCodeBench gate also completed instead of
recreating the panic. Across 104,225 generated tokens, including two complete
8,192-token truncations, peak Metal memory was `53.591 GiB`. It scored 36/60
samples and 21/30 tasks: easy 19/20, medium 13/20, and hard 4/20. Report
SHA-256 is
`7fdff41f1a5d1880f9463fc2efda4321441f8aacc4de563491568f289b1908d2`.
The balanced control's prior matched-task report scored 36/60 and 22/30, with
easy 19/20, medium 14/20, and hard 3/20. There were five sample wins each way.
Because model and prefill runtime both differ and sampling is stochastic, this
is a mixed descriptive comparison rather than an isolated pruning estimate;
the provenance-bound comparison SHA-256 is
`0a31faae0a6ddc9d6745cc397a06352614794f77192a6fc20aa35e97bde82aa2`.

The measured bounded-prefill quality path reserves `1.625 GiB` above resident
payload and permits at most 85% physical-memory occupancy while retaining the
allocator and live-reserve gates. Remove400 now passes unattended preflight at
exactly `iogpu.wired_limit_mb=58368`: projected working set `53.9658 GiB`
versus a `54.15 GiB` allocator boundary. Generic and unbounded runtime paths
retain the conservative `3.25 GiB` transient allowance. Larger candidates
remain rejected at this cap.

Remove400 now has its own exact shared-target 32K MTP vocabulary map at
`mtp-vocab-map-bf16-e32768-r25-nested-remove400`. The 128 KiB token map is
bound to the physical candidate's pack report; it does not duplicate or alter
the authoritative BF16 target head. The speculative runtime evaluates each
MTP token and confidence reduction together and batches all target-verifier
row winners into one Metal synchronization. The shared 128-expert sidecar
averaged `40.535 tok/s` versus `24.191 tok/s` ordinary decode (`1.676x`) over
two 512-token coding controls, with exact token identity and a `53.358 GiB`
peak.

Teacher replay over remove400's own coding trace produced a fixed-size
candidate-specific plan sharing 103/128 experts with the generic plan. Its
NVFP4 artifact, `mtp-sidecar-e128-remove400-nvfp4`, adds no resident memory
over the generic sidecar. Two exact 512-token controls averaged `41.683 tok/s`
versus `24.212 tok/s` ordinary (`1.722x`) with 93.65% draft acceptance and the
same `53.358 GiB` peak. Exact reasoning and independent coding controls reached
`38.208` and `39.887 tok/s`. A technical-instruction trace accepted one fewer
draft than the generic sidecar, so the candidate-specific artifact is the
coding-optimized remove400 default and the generic artifact remains the broad
fallback. Because the authoritative target verifies every draft, neither
sidecar changes greedy output quality.

The fused short-sequence SSM kernel then raised two exact 512-token controls to
`42.737` and `41.869 tok/s`, averaging `42.303 tok/s` versus `24.131 tok/s`
ordinary (`1.753x`) at a `53.342 GiB` peak. This is a 1.49% end-to-end gain over
the pre-kernel candidate-sidecar mean without changing weights, draft policy,
or accepted token IDs.

Compiling the weight-parameterized tail of the dominant MoE layout reduced
full target verification from `45.651` to `44.840 ms` for two tokens and from
`57.612` to `56.032 ms` for three tokens. Top-1 and rollback/capture checks
passed with maximum full-logit drift `1.53e-5`. Two exact 512-token controls
then measured `43.786` and `43.747 tok/s`, averaging `43.767 tok/s` versus
`25.223 tok/s` ordinary (`1.735x`) at a `53.344 GiB` peak. This is 3.46% above
the short-SSM mean with unchanged 93.65% draft acceptance and output tokens.
Log SHA-256 values are
`709efd53ea2770ca4877ee347d784777396c95ce53ed52f65bcbaa3c93014e3e` and
`9df8ba12dbbb2b5677262dca8ea2a5319ea6f74e092415c4ed293adde2582fcb`.

Compiling the remaining six mixed-precision tails reduced two-token target
verification again from `44.840` to `44.295 ms` and three-token verification
from `56.032` to `55.716 ms`, with the same exactness envelope. Three exact
512-token controls measured `44.240`, `43.543`, and `43.880 tok/s`, averaging
`43.888 tok/s` versus `25.394 tok/s` ordinary (`1.728x`) with no more than
`53.342 GiB` peak. The incremental speculative gain is only 0.28% over the
dominant-only mean, while ordinary decode improves 0.68%; the extension is
retained because it is exact, memory-neutral, and removes eager graph overhead
from every MoE precision layout. Log SHA-256 values are
`e67a212004237bb7cfb409b67d2d44fdeef5331912ca29e44273a623632d5531`,
`9272abd38b28562640e3dd0e2f706ad44e8d997fcfa9c638119d21f4e61dd8b6`, and
`c37c900edfaa1930720a7d078cd0b714b7980222cc620f22dd450dd8fc267699`.

A larger candidate-specific MTP sidecar then traded 0.370118 GiB for higher
acceptance without increasing selected-expert compute. The 256-expert NVFP4
artifact is `0.803994 GiB`; it improved recursive matches on the ranking trace
from e128's 177/93/32 to 186/110/48 at depths one/two/three. On an independent
eight-prompt trace, matches improved from 146/59/19 to 161/83/30. Two exact
cacheless 512-token resident controls measured `45.515` and `45.986 tok/s`, averaging
`45.751 tok/s` versus `25.372 tok/s` ordinary (`1.803x`) with 96.83% draft
acceptance and `53.712 GiB` peak. This is a 4.24% gain over the e128 production
mean. Artifact SHA-256 is
`fc1075d04fa822bb6027f9bb49d357952283b527685035a1d7cb13a5abe63bba`;
resident log SHA-256 values are
`b62ee2eade746d5fb824fea3f472017beacd34cf3f9ce593a1cdf01a5503c015` and
`d3d02a505f9284344a63a676dc8261678655998eaf15285964de98cdcdd48338`.
This established the remove400 coding-performance option before persistent MTP
cache support; e128 remains the smaller fallback. The e256 path passes bounded
generation preflight at a 57 GiB wired cap but not the conservative unattended
extended-run gate.

Under cacheless MTP semantics, the stronger draft did not make depth three
economical. An early selective exact run accepted 41/58 third drafts but fell
to `43.366 tok/s` and raised
peak memory to `54.025 GiB`, within roughly 128 MiB of the allocator boundary.
A later opt-in implementation separated the pre-compute and emission gates and
recorded complete atomic cycle traces. A no-emission observation found that
third margins at least 2.0 were correct 12/15 despite only 23/51 accuracy
overall. Actual 512-token policies then reached `45.300 tok/s` at attempt/output
thresholds 2.0/2.0 and `45.317 tok/s` at 2.0/1.5, with exact token identity and
`53.714 GiB` peak. Both remained below the `45.751 tok/s` depth-two mean. That
policy is superseded by the prompt-prefilled cache result later in this section.

Retuning e256's adaptive margins also failed the end-to-end gate. Thresholds
1.0/0.5 increased second-draft rate but reached only `45.693 tok/s`; keeping the
first threshold at 1.5 and lowering only the second to 0.5 reached
`45.332 tok/s`. Both were exact and memory-neutral, but neither beat the
`45.751 tok/s` production mean. Keep first/second thresholds at 1.5/1.0.

Expanding the shared target projection from 32K to a nested 64K vocabulary
improved e256's offline first-depth matches from 186/256 to 193/256 on the
ranking trace and from 161/256 to 167/256 independently. It still accepted the
same 305/315 resident drafts and slowed exact decode to `44.778 tok/s`. A
low-margin adaptive fallback was also negative: it changed only one of 45
fallback attempts on the control and none of 65 attempts on an independent C++
prompt, with no accepted-draft gain. The temporary runtime path was removed.
The 256 KiB 64K map remains diagnostic; the 32K map remains production.

Confidence-gating accepted-state capture was also rejected. Boundaries 1.0/4.0
halved capture writes on the production trace but averaged `45.628 tok/s`,
below the `45.751 tok/s` full-capture mean, and caused expensive replay on an
independent prompt. Conservative 2.0/4.0 boundaries covered every observed
miss but reached only `45.809 tok/s` on the production trace and `38.902 tok/s`
versus `38.812 tok/s` independently. Those differences are below variance, so
the temporary policy was removed and full capture remains the exact default.

A follow-up wrapper that also compiled RMSNorm and routing was removed. It was
bit-exact and looked substantially faster under isolated per-layer
synchronization, but full verifier timing was unchanged and two long controls
showed only a 0.20% speculative change with a slight ordinary-decode
regression. The tail-only compiler boundary remains the production design.

A matching attempt to compile each Mamba layer's post-recurrence gated norm and
FP8 output projection was also removed. It preserved every output and recurrent
state exactly across all 40 layers and improved the isolated tail by about 33%,
but alternating full-model ordinary runs moved only 0.23% and two speculative
controls averaged `43.993 tok/s`, statistically indistinguishable from the
`43.888 tok/s` production floor. The existing native Mamba composition remains
the production path.

`nemotron_mlx_runtime_profile.py` now provides a bounded resident profile with
identical cache restoration between samples. On the preferred swap400 target,
uninstrumented one/two/three-token blocks measured `38.881/44.552/55.584 ms`.
For the two-token block, forced layer synchronization attributed `24.927 ms`
to 40 Mamba layers, `29.367 ms` to 40 MoE layers, `3.627 ms` to eight attention
layers, and only `2.575 ms` to the full BF16 vocabulary head. The synchronized
total was 1.379x the real lazy graph, so the component values rank hot paths
but are not additive production latency. The profile rules out target-head
micro-optimization as the next large gain and points to recurrent/projection
fusion or better verifier amortization. The profile log SHA-256 is
`f3876bb289e44269d8932ac4523fd00ac75a9f4208c8d367ed26c57795df9a09`.

A subsequent MoE hotpath screen exhausted the low-level launch and algebraic
options without a production win. Exact MLX `gather_qmv` variants improved the
isolated selected-expert pair by at most 2.95%, but complete representative MoE
layers were neutral or slower. Wider work and output tiles for Nemotron's
`K=2688` expert-down fallback also failed to generalize. Specialized routing,
global-scale/score folds, and custom FP8 batches were rejected on full-path
performance or numerical drift. A GPU-lazy two-depth MTP chain reduced median
draft time from `2.874` to `2.588 ms`, but required a 0.25 GiB recursive
embedding table and improved exact decode only from `44.956` to `45.102 tok/s`.
All runtime changes were removed; multi-token benchmark support remains. The
source-build report SHA-256 is
`ddd0dd54dff2bd171717af9940c053fc7d0ae45f30ef96711738915fbc6aa526`.

Two follow-up Mamba kernel boundaries failed the end-to-end gate. Fusing
convolution, SiLU, and SSM duplicated B/C convolution work across heads and was
slower for multi-token layers. A second kernel preserved that sharing and
improved isolated captured layers by 2-4%, but made full two/three-token
captured verification `46.572/57.480 ms`, worse than the production
`44.295/55.716 ms`. Both kernels were removed. MLX's larger lazy schedule is
already hiding enough generic convolution work that this launch-level fusion
is counterproductive.

A draft-only rank-16 residual correction to recursive MTP hidden states was
also tested. It preserved first-depth output and improved depth-two matches from
46/85 to 52/85 on held-out training prompts, then from 59/141 to 68/141 on an
independent eight-prompt remove400 trace. That local gain did not survive the
resident gate: the adapter measured `44.222 tok/s` with 297/318 accepted drafts
and `53.343 GiB` peak, while the matched no-adapter control measured
`44.253 tok/s` with 295/315 accepted drafts and `53.342 GiB` peak. Both outputs
were exact. The runtime hook was removed because the throughput delta is noise
and aggregate acceptance did not improve; the trainer and independent evaluator
remain diagnostic evidence against promoting local MTP fitting without an
end-to-end gate. Log SHA-256 values are
`98673de7f543f172042d5fb314b421d2155b491138bc51d47793989e5f5427bc` and
`8077e2644a5d375921320f62955d280b63f7fa713273f17d7e0755dcd1bfa99b`.

Depth-aware expert selection was screened next without changing the e256
budget. Full-512 BF16 replay captured independently normalized route mass at
each recursive depth. Equal weighting replaced 27 of the current 256 experts
and improved BF16 accepted drafts from 379 to 385 on the ranking trace and from
256 to 259 independently; the more recursive 1/2/3 weighting reached 387 but
only 257 independently. The conservative equal-depth candidate was therefore
the only plan quantized. Its NVFP4 sidecar preserved depth-one/depth-two matches
at 186/110 but reduced depth-three matches from 48 to 47. Both temporary BF16
and NVFP4 sidecars were deleted before resident testing. The full route report,
equal-depth plan, BF16 ranking report, independent report, and NVFP4 report have
SHA-256 values `431fbacd0cb4d7baef6fd12bfcb69f65207479dc10d4656bd650f134dce27a90`,
`d05496e9c35baf017188cf93c0199a544c6b043ae9315d04fd631e0c90b3eac3`,
`ba58a2cdc2bd889b04abeca250882cd2bf1e5c322ead123fb65aaa7b62fd0fe9`,
`deca96b996bacbaeea7c8af363f693bcaaf88de327b455fbafa69b25b6c2e3af`,
and `a0c51939ca7606380fa0be31550d8324b7932f96d34088cfddbb20ec011a38df`.
Keep the original e256 plan and sidecar.

Selective BF16 retention was then evaluated within the same e256 draft. The
mixed-format quantizer records every exact BF16 matrix and the runtime rejects
any disagreement between config and payload. Restoring `o_proj` was the best
ranking-trace result: first/second-depth matches improved from 186/110 to
192/112 for 23 MiB above the 0.804 GiB sidecar, but median draft latency rose
from about 1.30 ms to 1.38 ms. It did not generalize: the independent trace
changed from 161/83/30 to 161/80/32. Restoring all attention projections was
slower and regressed independently to 159/79/31. `fc1_latent_proj` only moved
depth-three matches, while `eh_proj`, `fc2_latent_proj`, shared experts, and the
remaining individual attention projections were neutral or harmful. No mixed
artifact was promoted; all generated sidecars and the reproducible BF16
intermediate were deleted. Matched control report SHA-256 values are
`c6d8eb4353c2bf8a9f2566d203d020d700427834f4eb3304be19b08b512c2d69`
and `783a69e3755d171627984468fc653bcfb5a24899333561f15f5b48b9f80d18e5`;
the `o_proj` independent report is
`3d43c7b81e6e24cd31f4fd8e3e3d7f31b886311da11a309a4de01dd64585cc77`.

A Metal System Trace then isolated five ordinary block-two target forwards.
The steady GPU spans were 40.72-41.18 ms with 40.56-40.98 ms active, over
99.5% utilization inside each call. The apparent command-buffer fragmentation
was therefore not a GPU idle problem; about 4.6-5.9 ms remained in Python graph
construction and call setup. Mamba layers were converted to lazily compiled
pure graphs with convolution and SSM state passed explicitly. A real layer was
bit-exact and improved from 0.547 to 0.507 ms. Full eager/compiled vocabulary
logits and all 40 layers' recurrent arrays were exactly equal, while rollback
and captured-cache drift stayed below `1.526e-5`.

Matched ten-repeat block-two medians improved from 45.495 to 44.368 ms. In the
complete e256/32K depth-two generator, a paired 256-token run improved from
45.167 to 46.123 tok/s (2.12%); ordinary decode improved from 25.356 to 26.026
tok/s, verifier median from 57.163 to 56.023 ms, output remained exact, and
peak memory was 53.689 GiB. Compiled/eager log SHA-256 values are
`7068fffd6787e1069a5e513688068fba6575284bcf30dfcac054b4f662b69dd5`
and `2cbc55ea2c49c3349c09cbc151cb27a7feb7769b7a064fdd34c6d3d0d507c7d3`.
Generation CLIs enable this path by default; `--no-compile-mamba` is the eager
control. Whole-MoE compilation was bit-exact in isolation but exhausted Metal
command-buffer memory in the resident model and was removed.

Uniform routed-expert-count reduction was then screened without rewriting the
remove400 candidate. `nemotron_mlx_topk_sweep.py` streams one layer at a time,
binds reports to the source revision, packed plan/report, model index, corpus,
and tool, and compares complete vocabulary logits against native top-22.
Top-20 retained 8/8 winners in the initial eight-category screen and 16/16
winners across longer sensitivity and disjoint validation prompts. Their mean
KL values were `0.00765` and `0.00552`. Top-18 changed the multilingual winner
in the initial screen; top-16/14/12 happened to retain those eight winners but
raised mean KL to `0.0215/0.0391/0.0747`.

Matched ten-repeat resident block-two medians were 45.288 ms at top-22, 43.666
ms at top-20, 42.510 ms at top-18, 41.527 ms at top-16, and 39.475 ms at
top-12. Thus the only credible first quality candidate buys just 3.58% in the
target pass, while an already aggressive and visibly drifting top-12 buys only
12.84%. Mamba and fixed target work impose a high floor. This direction is
rejected as the next large-gain path; production remains native top-22 and the
runtime override remains confined to analysis/profiling.

The candidate-specific artifact SHA-256 is
`a42b4f167183c313ca5130e0c8800955fb8ea2ea0b25ebd00554d1a88a82ee75`.
Its offline NVFP4/32K report improved remove400 top-1 from 68.75% to 69.14% and
top-5 from 87.50% to 88.67%; report SHA-256 is
`e50ca8919c59b9ec6e87a1206541ae9e162a51a3345605de3c0b4520fd417fc4`.
Fixed-budget 8/16-expert blends at adaptation weights 0.5, 0.75, and 0.9 failed
to dominate both original and remove400 traces, so no blend was materialized.

Use a 256 MiB MLX allocator cache for this path. A 512 MiB allocator cache fit
the nominal payload calculation but triggered allocator pressure and collapsed
throughput to `22.146 tok/s`. This is separate from the small MTP attention KV
cache, which occupied 1.5 MiB after 523 prompt/generated transitions.

Correct prompt-prefilled MTP cache semantics reverse the old depth-three
decision. On the matched 256-token control, cacheless depth two measured
`45.980 tok/s`, prompt-cached depth two measured `47.065 tok/s`, and
prompt-cached confidence-gated depth three measured `49.379 tok/s`, a 7.39%
paired gain over cacheless. All outputs exactly matched ordinary greedy decode.
A 512-token depth-three run reached `50.221 tok/s`, `1.943x` its normal
`25.844 tok/s` ordinary control, with 97.75% accepted drafts and a
`53.705 GiB` peak. Lowering the third-output margin from 1.0 to 0.5 regressed
to `48.877 tok/s` and is rejected. Generated-history-only caching also lost to
prompt prefill and is not the production mode.

The 25,600-cycle disjoint trace improved from `1.35836` accepted drafts per
cycle cacheless to `2.64254` with generated history, a 94.5% increase before
end-to-end verifier costs. Reports and SHA-256 values are:

```text
quality/mtp-cacheless-e256-heldout-odd-depth8.json
da6029de6e813cf08ce7838b689b49a0caf044bfdf6895ecf60f09abd947ee00
quality/mtp-kv-cache-e256-heldout-odd-depth8.json
c7959f230ae7386480cede6aa3887df1e70fe42f3548a83301231969c970e7ed
```

A cache-aware e256 expert reselection was trained on disjoint even prompts. It
slightly improved aggregate held-out depth-three acceptance but reduced
depth-one acceptance by 2.19 percentage points and regressed the independent
short coding trace from 1.820 to 1.676 accepted drafts per cycle. Its temporary
sidecars were deleted; retain the current broadly tested e256 sidecar.

Use `--cycle-trace PATH` only for atomic, exact-output-validated policy
diagnostics. Substitute `mtp-sidecar-e128-remove400-nvfp4` below when the extra
0.370 GiB is needed.

```sh
MODEL_ROOT=/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
PYTHONPATH=nemotron/tools "$MODEL_ROOT/mlx-env/bin/python" \
  nemotron/tools/nemotron_mlx_speculative.py \
  --model-dir "$MODEL_ROOT/candidate-r25-nested-remove400-mlx" \
  --mtp-sidecar "$MODEL_ROOT/mtp-sidecar-e256-remove400-nvfp4" \
  --mtp-lm-head \
    "$MODEL_ROOT/mtp-vocab-map-bf16-e32768-r25-nested-remove400" \
  --max-new-tokens 512 --warmup-cycles 10 --margin-gib 0.5 \
  --cache-limit-mib 256 --capture-rollback --mtp-cache-mode prompt \
  --max-draft-tokens 3 \
  --draft-margin-threshold 1.5 --second-draft-margin-threshold 1.0 \
  --third-attempt-margin-threshold 2.0 --third-draft-margin-threshold 1.0 \
  --paged-embeddings --embedding-cache-rows 256
```

The next nested tier removes 1,000 experts total while preserving the
preferred r25 expert identities wherever retained. Its plan is:

```text
plans/r25-nested-frontier/plan-remove1000-total.json
SHA-256: 3f57e51f2fede9571c6d8f58ac755c37462c12b7c8689afe855c4c7b0e06e0ce
```

The complete eight-category virtual gate retained 7/8 source top tokens. Mean
KL was `0.07787`, worst KL `0.20778`, mean centered relative-L2 `0.07894`, and
mean top-64 overlap 55.0. Report SHA-256 is
`c72231cad4bb53aaf387de7d8aa50549cf1d312702fb392dcb0b75fefde33388`.
Incremental materialization rewrote and validated 34 groups, hard-linked 55,
and produced `51.6059 GiB` logical payload. With paged embeddings, projected
resident payload is `50.6059 GiB`; the 0.5 GiB-margin requirement is
`51.1059 GiB`, or 79.85% of physical memory. Pack-report SHA-256 is
`44e0ab2c61737414f0be44b3d46b7d0de30a2d0c12a32a43fb5b83b72262cdcb`.

Physical and virtual logits subsequently matched byte-for-byte: relative L2,
maximum absolute error, KL, and top-token drift were all zero. A 64-token
resident generation reached `24.423 tok/s`, with `40.878 ms` median token time
and a `50.837 GiB` peak. The first extended MBPP attempt exposed additional
prompt/generation workspace not represented by payload-only preflight. It was
stopped at 39/100 after active peak reached `52.128 GiB`, only 128 MiB below
the 55 GiB cap's allocator-GC boundary. The retained partial scored 25/39;
the preferred swap400 candidate scored 27/39 on those exact tasks.

The 56 GiB continuation reached 98/100 before a rarer prompt raised active
peak to `53.134 GiB`; the live guard stopped 66 MiB below the 53.20 GiB
allocator boundary. The final two tasks resumed under a 57 GiB cap. This was
the whole-prompt path; later bounded-prefill measurements supersede its
transient estimate. Every quality gate still enforces a live 128 MiB reserve.

The complete MBPP result rejects this pruning plan: 70/100 versus 74/100 for
the preferred swap400-r25size candidate on the same tasks, with candidate-only
wins on tasks 351 and 376 but control-only passes on 125, 277, 286, 342, 39,
and 501. The report SHA-256 is
`3c63a6da71b21f5198ea4f85346e5f9fd17d948bcd8a4d88b67d557dab46dff0`.
The five-point gap to the 75/100 guard400 quality-headroom candidate is also
material. HumanEval and LiveCodeBench are skipped because MBPP already failed.

An additional 200-expert cut is rejected before materialization. Its first six
independent categories included a tool-calling top-1 flip, baseline-token rank
8, and KL `2.28239`. The deeper remove1400 plan is therefore not evaluated.

Decision: reject plain activation-ranked remove1000 as a production runtime.
Its exactness, speed, and memory target pass, but the coding regression does
not. Six source-teacher regression trajectories produced fixed-size repair20,
repair40, and repair80 plans. Local route-weighted lost-output ratios improved
from `1.0` to `0.89095`, `0.85523`, and `0.80910`, respectively, but the
independent eight-category full-logit gate rejected the direction: mean KL
worsened from remove1000's `0.077868` to `0.081856` for repair20 and `0.083312`
for repair40, with no top-1 recovery. The comparison report SHA-256 is
`e9a9d4ebee8e6f5ef5757ba8887def3224aeb3517bf725f826139e9c5d03206c`.
None was materialized. Preserve the compact reports and plans; the reproducible
physical remove1000 artifact was deleted, recovering about 35 GiB. The old
remove400 report remains crash evidence; do not resume it unattended.

Decision: retain remove400 as the preferred memory-first fallback, not as the
balanced default. Its 1.1566 GiB saving costs one MBPP pass versus the current
quality-headroom runtime and one LiveCodeBench task-pass-any in a cross-runtime
comparison, while HumanEval remains tied and hard-sample pass count improves by
one. Most importantly, the completed hidden gate proves stable operation on the
64 GB M4 Max at a 57 GiB wired cap.

### Rejected Shared-Subspace Expert Formats

The post-training shared-subspace study follows the primary Sub-MoE principle
of clustering experts by same-input output similarity, then tests three
model-specific representations: an NVFP4 prototype plus one shared low-rank
difference per pair, a shared output basis with expert-specific coefficients,
and the transposed shared input basis. `nemotron_mlx_shared_subspace.py` binds
the screen to the source revision, retained-expert plan, proxy calibration, and
independent validation corpus. It evaluates reconstructed expert outputs on
real routed latent inputs and reports projected storage at BF16 and FP8 factor
precision before any kernel work.

Layer 1 failed at every economically useful rank. A global scan then selected
the strongest retained pair in all 40 MoE layers: layer 14 experts 104 and 392,
with `0.98753` output cosine and 16 calibration observations. Across eight
independent categories, the pair received 32 and 13 held-out candidate-route
samples.
The apparently ideal pair still failed:

| Representation | Rank | Optimistic storage saving | Mean output rel-L2 |
| --- | ---: | ---: | ---: |
| Prototype + shared difference | 256 | 19.31% at FP8 | 0.5806 |
| Shared output basis | 256 | 53.97% at FP8 | 0.9637 |
| Shared input basis | 256 | 53.97% at FP8 | 0.9593 |
| Shared input basis | 512 | 7.94% at FP8 | 0.7085 |

The rank-512 prototype format improved error to `0.3169` but was already
11.38% larger than the original NVFP4 pair even under optimistic FP8 factors.
These are float32 reconstruction errors; factor quantization cannot rescue
them. Functional output similarity therefore does not imply a low-rank shared
weight representation for Nemotron's tiny latent experts. The family is
rejected before materialization or a fused kernel. Reopening it requires
training the shared representation rather than another post-training SVD.

```text
retained-route report: fad10eb3d50a496c318d3dac30f15cef9c8ed4482bd2bc9b66958922fbd2aed8
```

## Activation-Fitted Mixed Low-Bit MTP

The MTP head is the bounded safety sandbox for sub-NVFP4 representation work.
Its routed experts are available in BF16 inside the pinned checkpoint, so the
experiment does not double-quantize the target's NVFP4 backbone weights. The
public Prism ML branch supplied a working one-bit affine representation and
Metal decoder, but not a public conversion/QAT recipe. We therefore treat its
release as evidence that the format can execute and independently derive the
quality method.

`nemotron_mlx_mtp_binary_fit.py` starts from group-128 one-bit endpoints and
fits them against routed teacher activations. Weight-only binary PTQ reached
57.42% independent top-1 acceptance; activation fitting raised it to 60.94%.
The gain transferred outside the fitting trace, while aggregate compensation
and small projection corrections did not. This establishes the first rule for
backbone work: binary experts require activation-aware fitting, not another
blind quantization pass.

`nemotron_mlx_mtp_lowbit_sensitivity.py` then evaluates causal precision
promotion. For each expert it substitutes the actual affine 3-bit up/down pair,
propagates the result through `fc2_latent` and final normalization, and scores
target cross-entropy plus teacher KL over a reduced teacher/baseline token set.
The plan used coding and swap traces; the remove400-adapter trace remained an
independent gate. `nemotron_mlx_mtp_lowbit_overlay.py --split-base` writes two
disjoint banks so no expert is duplicated in memory.

The quality point promotes 256 of 512 experts to affine 3-bit and keeps the
others at fitted affine 1-bit:

| Trace | Native NVFP4 top-1/top-5 | Mixed 1b/3b top-1/top-5 |
| --- | ---: | ---: |
| Coding plan | 81.25% / 98.44% | 83.59% / 98.44% |
| Independent | 68.75% / 94.14% | 68.36% / 94.14% |
| Swap plan | 69.14% / 95.31% | 73.05% / 94.92% |

The mixed payload is `0.805948 GiB` versus `1.5442 GiB` for native NVFP4, a
47.8% reduction. Its payload SHA-256 is
`890d13a435f4a3c0da0cb52367c1981869b4540b2135399530269e8238f9689e`.
The independent top-1 difference is one row out of 256, while top-5 is exact.
A smaller 128-promoted control occupies `0.641886 GiB` and reaches
64.84%/93.75% on the independent trace; it is a performance control, not the
quality point.

`nemotron_mlx_mtp_mixed.py` fuses bank selection and both affine decoders into
one Metal dispatch. It is enabled by default for compatible 1-bit/3-bit banks;
`NEMOTRON_MTP_MIXED_METAL=0` restores the two-`gather_qmm` reference. Across
512 full-vocabulary comparisons, every top-1 and top-5 result matched. Relative
L2 remained below `6.4e-8` and maximum absolute logit drift below `2e-5`.
Three matched isolated runs improved median latency from 3.811 to 3.642 ms and
reduced peak allocation by about 169 MiB.

On the preferred swap400 target, a matched cacheless depth-one control reached
`41.590 tok/s`, 96.49% acceptance, and `54.686 GiB` peak versus
`39.783 tok/s`, 85.0%, and `54.314 GiB` for e128 NVFP4. The two-bank reference
reached `40.362 tok/s`, so the fused kernel contributes a separate 3.0%
resident gain. Cacheless depth two reached `42.081 tok/s`, only 1.18% above
depth one.

The current prompt-cached three-draft production policy exposes the remaining
kernel/bandwidth boundary. The 256-promoted sidecar reached `47.790 tok/s`; the
128-promoted sidecar reached `48.538 tok/s`. Both preserve exact target output,
but neither beats the established e256 path's `50.221 tok/s`, so production
keeps e256. This low-bit result is accepted for representation quality and as a
backbone research method, not promoted as the fastest draft.

The custom environment is `$MODEL_ROOT/mlx-prism-env`, built from MLX 0.32.0
plus local forward-port revision
`155198dccf00ab7a0f5962806e91a5d0c91a3a1b`. The exact wheel hash is recorded
in `source-notes/revisions.json`. Stock `$MODEL_ROOT/mlx-env` remains the
ordinary runtime and cleanly skips one-bit-only tests.

This result does not establish a 21 GiB target model. MTP accounts for only
about 1.5 GiB in its native NVFP4 form, and its source experts were BF16. A
valid backbone rollout must obtain representative BF16 expert tensors, fit and
rank each of the 40 MoE layers independently, retain sensitive non-expert
tensors at measured precision, and pass held-out layer-output plus full-logit
gates before projecting or materializing a whole checkpoint.

## BF16-Derived Backbone Low-Bit Pilot

The backbone rollout has crossed its explicitly approved two-shard BF16 pilot
boundary; the transfer is in progress and no complete shard is yet claimed as
verified. Official BF16 metadata is pinned to revision
`d51eab0d1f979ebc26b546e634a04f450d99158e`: 50 immutable shard identities,
230.2487 GiB of shard files, and 230.24 GiB of indexed tensor payload. The
layer-1 contract is bound by SHA-256
`bf8fd57e70b92dcfba0f26e1a49dc21054b47ce07c47724132b8c61a0e7a01e0`.
It requires two source shards totaling 9,994,713,832 bytes (9.3083 GiB) and
contains exactly 5.25 GiB of BF16 routed-expert tensors. Expert 336 crosses the
shard boundary, proving that the real stream must retain a two-shard window.

The native-QAT context teacher is already complete at
`backbone-lowbit-work/context-layer1-balanced200-v1`. It contains 22,534
routed rows over 200 prompts and ten categories. The prompt-hash split keeps
all chunks from one prompt on one side: 17,658 training rows and 4,876 held-out
rows. Every one of the 512 experts is represented; train route counts range
from 331 to 1,643 and held-out counts from 74 to 409.

The representation is not blind PTQ. BF16 supplies the initial binary code
structure, while exact dequantized ModelOpt NVFP4 is the behavioral teacher.
`nemotron_mlx_backbone_lowbit.py` first solves group-128 affine endpoints in
function order, then optionally runs bounded straight-through code training.
The latter uses compact Gefen state, prompt-disjoint held-out checkpoint
selection, and returns the analytical fit unchanged when training does not
transfer. A synthetic distinct-target gate proves the function-teacher path;
the real native projection reconstruction matches MLX's NVFP4 qmm at relative
L2 `5.54e-7` for `up_proj` and `3.45e-7` for `down_proj`.

Planning keeps exact native NVFP4 as the safety tier. It forces every skipped,
non-finite, or held-out-regressing binary expert native and ranks the remaining
experts from score-weighted residuals. Reports distinguish error relative to
the routed branch from error relative to the residual-dominated full layer.
The physical format stores disjoint fitted-binary and exact-native banks plus
original-to-local maps. Its Metal path selects the bank inside one dispatch;
at real expert dimensions a four-route synthetic gate matched the reference at
relative L2 `3.59e-7`, maximum absolute error `1.29e-5`, and 1.056 ms. The
materializer rereads every retained payload for exact equality and is
integration-tested for atomic resumption.

The corrected packed storage points for one 512-expert layer are:

| Binary / exact native experts | Layer payload |
| ---: | ---: |
| 512 / 0 | 0.4102 GiB |
| 448 / 64 | 0.5435 GiB |
| 384 / 128 | 0.6768 GiB |
| 320 / 192 | 0.8101 GiB |
| 256 / 256 | 0.9434 GiB |

One stacked runtime NVFP4 expert has 3,096,584 tensor bytes. The immutable
source inventory reports 3,096,592 payload bytes because its two separate F32
scalar tensors are each aligned to eight bytes; the packed runtime stacks
those scalars and removes that per-expert container padding. A previous
projection incorrectly charged one global scale per output row and has been
corrected.

With MTP omitted and every other target tensor unchanged, uniform 64-native
and 128-native projections would place the model near 31.98 and 37.31 GiB of
weight payload, respectively. Fully affine-binary experts project to about
26.64 GiB. Combining the already measured 25% structural cut with the current
asymmetric binary format projects to roughly 22.54 GiB, but that is only a size
calculation: pruning and low-bit errors must be accepted independently before
they can be composed.

Prism ML's July 2026 [Bonsai 27B release](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
is relevant evidence, not a recipe. Its published group-128 binary model
retains 89.5% of its FP16 benchmark average, while its
[ternary quality point](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit)
retains 94.6%, and both transform an existing pretrained model end to end. The
[whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
explicitly calls the representation transformation proprietary and does not
publish the training algorithm. This validates investing in trained low-bit
codes and ternary or mixed operating points; it does not validate our Nemotron
candidate. The first real layer pilot, full-layer causal plan, independent
full-logit gates, and coding evaluations still decide promotion.

`nemotron/run_backbone_lowbit_pilot.sh` is the approval-gated entry point. It
isolates Hugging Face caches inside the job, disables persistent Xet chunks,
and downloads through authenticated, deterministic 128 MiB Xet ranges. Each
range is fsynced, SHA-256-bound into atomic state, and rehashed on restart; a
300-second no-progress watchdog aborts the active range without advancing
state. The final shard is rehashed against its immutable LFS identity before
atomic rename. The first two independent live ranges committed correctly, and
the second process resumed exactly at byte `134217728` after revalidating the
first range. The launcher then fits eight route-coverage representatives,
validates every artifact, and writes one human-readable stdout log plus durable
operation logs. It is safe to rerun; raw BF16 shards remain until explicit
deletion approval.

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

## MTP-Assisted Teacher Capture

`nemotron_mlx_mtp_teacher_capture.py` creates the training boundary for a
larger learned multi-depth draft. It runs the deployed resident candidate with
its exact speculative verifier, persists only on-trajectory target rows, and
stores final normalized hidden states as BF16. Every row contains the hidden
state before an accepted token, that accepted token, and the target's next
token. The resulting sequence is therefore compatible with both the cacheless
control and authoritative cache replay: each contiguous row can replace a
speculative MTP K/V entry with the target-conditioned transition used by
vLLM-style draft prefill.

The output is one atomic safetensors shard per prompt plus a provenance-bound
`state.json`. Interrupted runs resume at the first incomplete prompt after
hash- and schema-validating completed shards. Use `--validate-only` to audit an
artifact without loading 54 GiB of weights. The initial remove400/e256 real
smoke captured eight contiguous rows, peaked at `53.690 GiB`, and validated
successfully. This is a capture-mechanism certificate, not a training corpus.
The first training pilot should collect 10K diverse coding/reasoning rows; a
larger 250K run is justified only after the predictor beats the packed MTP
control on held-out exact acceptance and end-to-end tokens per second.

### Learned Multi-Depth Pilot

The 10K pilot is complete. `nemotron_mtp_teacher_prompts.py` selected 80
deterministic prompts balanced across ten oMLX calibration categories, and the
resident verifier produced 10,240 contiguous BF16 rows. The complete capture
occupies 84,145,220 bytes, peaked at `53.348 GiB`, and is bound by state hash
`8e90a543a6b0bf7f133465cfa0cdd8e87a432c77e7a3728ec8f5116d6a5ce8f4`.
MTP did accelerate data collection, but it never supplied unverified labels.

`nemotron_mlx_mtp_predictor.py` trains a compact residual predictor entirely
with MLX/Metal after unloading the resident target. A rank-1024 direct model
uses 20,974,592 parameters and a 40.0 MiB BF16 artifact. Its held-out
conditional acceptance was 44.56%, 74.16%, and 85.03% over three depths, but
resident verification exposed the distribution mismatch: depth-three decode
fell to `11.978 tok/s`. One draft remained exact and reached `26.657 tok/s`,
only 1.031x ordinary decode.

`nemotron_mlx_mtp_recursive_features.py` separately records the official MTP
next hidden and proposal for every teacher row. Training only depth-two and
depth-three continuation improved the learned conditional second-draft rate to
39.07%. It remained inferior in the decisive matched gate: `28.735 tok/s`
versus `38.918 tok/s` for the unmodified recursive e128 MTP path at the same
two-draft budget. Both paths preserve exact target output, so throughput and
acceptance decide the result. Learned drafting is rejected for production and
is available only through explicit `--learned-mtp-predictor` diagnostics; its
loader binds the artifact to the exact packed target and reduced head.

`nemotron_mlx_gefen.py` is a state-representation port of Gefen revision
`704034f0d62871cc651a5ebae7b5547c55e0fc37`. It reduces optimizer state from
167,796,736 bytes for AdamW to 21,091,328 bytes at the direct model size, a
7.96x reduction. The current unfused MLX implementation is about 2x slower per
steady epoch, while AdamW training peaks at only `2.621 GiB`. Gefen therefore
proves a useful capacity option for larger future adapters, not a speedup for
this pilot. Its Apple port substitutes a documented 4096-bin weighted Lloyd
codebook for upstream's exact CPU dynamic-programming solver; it does not claim
bitwise optimizer equivalence.

### Recursive MTP Distillation Follow-Up

The follow-up expanded the exact resident trace to 51,200 rows over 200 prompts.
`mtp-teacher-capture-balanced50k-v2` occupies 420,612,901 bytes, peaked at
`53.376 GiB`, and is bound by state SHA-256
`e2a4cc3f691ea64d5d4fdb495f2a5530c0149572e74a249dd979c613b15cab97`.
`nemotron_mlx_mtp_distill_features.py` then replayed official recursive MTP to
depth three and stored exact float32 hidden states plus top-32 reduced-head
logits. The 200-shard, 51,200-row feature artifact occupies 2,556,609,890 bytes
and is bound by state SHA-256
`0a13b75aa987ba126b9c97512b77e75ab70eea09c9800b66b192a65b04465721`.

The first distillation implementation incorrectly used rejected official MTP
proposals as hard labels. The corrected trainer filters for an accepted first
draft and uses the authoritative target's next token as the hard reduced-head
label; teacher hidden states and top-k logits remain soft evidence. The fused
rank-1024 token student has 50,366,464 parameters, a 100,733,449-byte BF16
artifact, and peaked at `5.153 GiB` during AdamW training. Its held-out
conditional second-token acceptance is 35.72%, versus 50.87% for official MTP.

The initial generic MLX layout took 7.707 ms per learned draft. Saving both
matrices transposed and contiguous for `bf16_matvec` reduced median draft time
to 2.137 ms. With a margin-2 second-draft gate, exact 512-token resident decode
reached `40.612 tok/s`, `1.574x` ordinary decode, at `53.421 GiB` peak. The
matched official recursive path reached `44.236 tok/s`, `1.715x` ordinary, at
`53.327 GiB`. The remaining 8.2% deficit is acceptance, not kernel speed, so
this student remains an explicit diagnostic. Do not replace official MTP or
collect a still larger corpus without a materially stronger student design.

### Official MTP Final-Norm Calibration

A zero-runtime-cost follow-up trained only the official MTP head's 4096 BF16
final RMSNorm values. It improved prompt-disjoint physical recursive acceptance
from 1.02785 to 1.28301 drafts per cycle, but the self-conditioned depth-three
rate regressed and the gain did not transfer to resident generation. Full
calibration reached `43.342 tok/s` and a conservative 0.4 interpolation reached
`44.107 tok/s`, both below the matched unmodified `44.236 tok/s` control with
the same `53.327 GiB` peak. The hash-bound override and capture-aware recursive
gate remain diagnostics; production retains the official norm unchanged.

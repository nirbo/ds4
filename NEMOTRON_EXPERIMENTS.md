# Nemotron Experimental Roadmap

This file is the durable ledger for size and performance experiments beyond the
current Nemotron runtime. Work through the experiments deliberately. Do not
mark an item complete merely because code exists or a smoke test passes.

## Tracking Rules

- An unchecked box means the experiment has not reached a defensible result.
- A checked box means the experiment was implemented, tested, measured, and
  classified as `SUCCESS`, `PARTIAL`, or `REJECTED`.
- Checking a box does not imply promotion into the default runtime.
- Replace each `Result: PENDING` line with the outcome, measurements, artifact
  paths, commit, and concise reason for promotion or rejection.
- Keep rejected findings. They prevent repeated work and constrain later ideas.
- Develop each item on its own `feature/nemotron-*` branch, merge verified work
  into `nemotron-main`, and do not couple unrelated experiments.
- Draft-only changes must preserve exact target-generated token IDs.
- Any change to authoritative target computation requires full-logit drift,
  greedy-token agreement, coding-quality, reasoning-quality, and downstream
  evaluation before promotion.
- Report resident memory, peak memory, artifact size, ordinary decode, proposed
  decode, acceptance, and all relevant latency distributions separately.
- Bind every durable artifact and result to source revision, plan and policy
  hashes, tool commit, MLX version, hardware, and Metal wired limit.

## Frozen Baseline

Use this baseline until a completed experiment explicitly replaces it:

- Source revision: `4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6`
- Integration baseline: `nemotron-main` at merge `0f147a6`
- Target: `candidate-oqe512-r20-mlx`, `57.504646 GiB` payload
- Ordinary decode: approximately `23.6 tok/s`
- Default speculative runtime: mmap-paged exact BF16 input embeddings,
  NVFP4-128 MTP sidecar, and the shared-target 32K BF16 vocabulary map
- Default speculative result: `34.748 tok/s` mean over paired paged controls,
  approximately `1.49x`, `57.34 GiB` peak, exact token identity
- Full 131K BF16 MTP projection remains the acceptance-oriented fallback
- Lower-memory fallback: NVFP4-64 MTP sidecar plus draft-only NVFP4 head
- Fallback result: `30.885 tok/s`, `1.302x`, `58.436 GiB` peak, exact token
  identity
- Measured kernel limit: `iogpu.wired_limit_mb=60672` (`59.25 GiB`)
- Uniform target projections before additional global-table savings:
  - 25% expert reduction: approximately `54.50 GiB`
  - 30% expert reduction: approximately `51.49 GiB`
  - 35% expert reduction: approximately `48.60 GiB`

## Phase 1: Exact Or Draft-Only Runtime Gains

### [x] 1. Reduced-Vocabulary BF16 MTP Head

**Goal:** Replace the full 131,072-token MTP output projection with a compact
draft-only BF16 vocabulary and a draft-to-target token map.

**Hypothesis:** Most accepted coding drafts use a small vocabulary. A BF16 head
over 8K, 16K, or 32K selected tokens costs approximately 64, 128, or 256 MiB,
respectively. It may be faster and more accurate within its vocabulary than the
full 288 MiB NVFP4 draft head. Tokens outside the draft vocabulary only reduce
acceptance; the target verifier remains authoritative.

**Work:**

- Build vocabularies from diverse coding, reasoning, prose, tool-call, control,
  and tokenizer-special-token evidence rather than the existing small trace.
- Always include syntax, whitespace, byte-fallback, control, and special tokens.
- Evaluate static 8K/16K/32K vocabularies and an optional prompt-augmented set.
- Materialize exact BF16 rows with revision and source-row hashes.
- Add narrow ID remapping inside MTP; target code must not know draft internals.
- Measure offline top-1/top-5 acceptance and full resident generation.

**Success gate:** Exact final token identity, no target-weight changes, and a
repeatable resident throughput or memory improvement over the default. Report
acceptance loss separately from projection latency.

**Result:** SUCCESS

- Commit/branch: `099cb51`, `feature/nemotron-reduced-mtp-vocab`
- Artifact:
  `/Users/nir/dev/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/mtp-vocab-map-bf16-e32768`
- Artifact SHA-256: `83c6d25815c89946f5701927c61820049d2fc0d244426cba63d99a9bf031e7e0`
- Report SHA-256: `0d516de7326bcfed553d56ec12e4bb51f49b35d45006aa4f1fcc9c4460a4fb05`
- Representation: 32,768 sorted target token IDs (`128 KiB`) and no copied
  weights; a custom Metal kernel gathers exact rows from the resident BF16
  target head.
- Held-out corpus: 96.46% overall token coverage over 611,402 tokens, with
  92.56% minimum category coverage.
- Offline quality: 75.00% top-1 acceptance versus 77.73% full-head; all eight
  per-prompt results are recorded in `mtp-reference/`.
- Offline performance: 1.417 ms median MTP versus 3.114 ms full-head.
- Resident performance: repeated controls averaged `33.994 tok/s` versus
  `32.981 tok/s`, a 3.1% gain. Three additional coding prompts were exact and
  showed workload-dependent changes from approximately -1.3% to a material
  positive gain.
- Resident memory: at most `58.335 GiB` peak, effectively unchanged from the
  shared full-head route.
- Decision: promoted as the performance default. Omitting `--mtp-lm-head`
  retains the full-head fallback for workloads where reduced-vocabulary
  acceptance outweighs projection savings.
- Follow-on constraint: recursive and lookup drafting must compare both the
  32K performance default and full-head acceptance fallback.

**Reference:** llama.cpp documents reduced-vocabulary EAGLE draft heads with a
draft-to-target map in its
[speculative decoding runtime](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

### [x] 2. Recursive Multi-Token MTP Drafting

**Goal:** Draft two to four tokens before one target block verification.

**Hypothesis:** The target already verifies multiple rows much faster than
sequential decode. Returning MTP's final hidden state and recursively applying
the one-depth MTP layer may produce useful short chains even though only one MTP
depth was trained.

**Work:**

- Expose MTP's final normalized hidden state without changing one-draft output.
- Measure independent chain-position acceptance for depths 2, 3, and 4.
- Add confidence-controlled early stopping based on calibrated margins.
- Verify the complete chain with the existing exact block verifier.
- Restore the cache after the first rejected draft, not merely at block end.
- Compare fixed and adaptive chain lengths with identical target output.

**Success gate:** Exact token identity and at least a repeatable 5% end-to-end
gain over the current default after all drafting, verification, rollback, and
memory costs. Reject recursive use if later-position acceptance collapses.

**Result:** PARTIAL

- Branch/implementation commit: `feature/nemotron-recursive-mtp`, `b15d4b5`
- Provenance-bound reports: `mtp-reference/recursive-e128-map32k-coding-8x32.json`
  (`4a1afe29e4ed654abf7c0a35c8956b53c7f8ba4c30ace19505d3b07438d99103`)
  and `mtp-reference/recursive-e128-full-coding-8x32.json`
  (`f42d6651c86702cabf1428736e71e663f35fd9144396bc36576c520a7fdfd6d6`).
- The shared 32K head accepted recursive drafts conditionally at 75.00%,
  64.21%, 47.06%, and 46.30% for depths one through four. The full head
  measured 77.73%, 63.45%, 46.72%, and 47.27%.
- Exact block verification remained numerically stable. Captured continuation
  logits after every prefix differed from incremental execution by at most
  `1.14440918e-05`, with identical top-1 tokens.
- Adaptive depth two uses first- and second-step logit margins, captures the
  likely partial-accept state, and falls back to exact replay after the rare
  earlier rejection. Depth one remains the default.
- On the 128-token coding control, adaptive depth two measured
  `34.872/34.982 tok/s` in repeated runs versus `34.137 tok/s` for the
  identical depth-one control and `33.994 tok/s` for the frozen repeated
  baseline. Output token IDs exactly matched ordinary greedy decode.
- The best run used a 1.5 first-margin threshold, a 1.0 recursive-margin
  threshold, a 128 MiB MLX cache, and reached approximately `58.50 GiB` peak.
  The full-head fallback was slower at `33.679 tok/s` on its 64-token control.
- Decision: retain as an opt-in exact experiment, but do not promote it. The
  repeatable gain is only about 2-3%, below the 5% success gate, and ungated
  recursive verification regressed to `29.840 tok/s`.

### [x] 3. Prompt And N-Gram Lookup Drafting

**Goal:** Generate zero-weight speculative drafts from repeated token sequences.

**Hypothesis:** Code contains repeated identifiers, indentation, delimiters,
imports, boilerplate, and local patterns. Prompt lookup can propose longer
drafts without another model or meaningful Metal memory.

**Work:**

- Implement a bounded dynamic n-gram map over prompt and generated tokens.
- Test key lengths and draft lengths independently on coding workloads.
- Give high-confidence lookup drafts priority and fall back to MTP otherwise.
- Verify drafts with existing block-2/4/8 target paths and exact rollback.
- Measure hit rate, accepted tokens per hit, lookup CPU cost, and total speed.
- Include non-repetitive reasoning prompts to quantify neutral and adverse cases.

**Success gate:** Exact output, negligible regression when no useful match
exists, and a repeatable coding-workload throughput gain without material
resident memory.

**Result:** SUCCESS

- Branch/implementation commit: `feature/nemotron-ngram-drafting`, `6b87f7d`
- A bounded LRU index searches longest suffixes over prompt plus committed
  output. The accepted policy requires two prior occurrences with the same
  complete continuation and agreement with the first MTP draft token.
- Four-token lookup blocks were the best tested horizon. Two-token blocks did
  not amortize verification, while ungated eight-token blocks caused expensive
  partial-rejection replay and were rejected.
- On the repetitive Python coding control, two final alternating pairs averaged
  `26.711 tok/s` with lookup versus `22.983 tok/s` with MTP alone, a 16.2%
  workload gain. Lookup proposed 20 measured tokens per run and accepted 19
  (95.0%). All final token IDs exactly matched ordinary greedy target decode.
- Peak MLX memory was approximately `58.54 GiB`, about 0.21 GiB above the
  one-draft path. Median lookup CPU time was 0.019-0.023 ms.
- A separate templated-test prompt that had produced harmful low-confidence
  matches emitted no lookup blocks after consensus gating. A non-repetitive
  reasoning control also emitted none and retained exact MTP behavior.
- Decision: promote as an opt-in workload accelerator with
  `--lookup-max-draft-tokens 4`; keep it disabled by default because gains
  depend on repeated token structure.

**References:** llama.cpp supports several n-gram speculative implementations
and mixing draft sources in its
[speculative decoding runtime](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).
Training-free exact parallel decoding is also explored by
[Lookahead Decoding](https://arxiv.org/abs/2402.02057).

### [x] 4. Exact Paged Input Embeddings

**Goal:** Stop holding the 1 GiB BF16 input embedding table in active Metal
memory when decode reads only one 8 KiB row per token.

**Hypothesis:** An exact mmap-backed row provider plus a small persistent shared
Metal staging buffer can recover nearly 1 GiB of resident headroom at negligible
decode cost.

**Work:**

- Build a strict safetensors row-offset catalog bound to source hashes.
- mmap the exact BF16 embedding payload without copying the complete table.
- Gather requested rows into a page-aligned shared buffer for decode and batch
  prompt rows for prefill.
- Add a small measured row cache, including MTP accepted-token embedding use.
- Compare CPU staging, Metal `bytesNoCopy`, and placement sparse-buffer options.
- Prove exact embedding bytes and full-logit/token identity.
- Measure active, cache, peak, page faults, staging latency, and thermal behavior.

**Success gate:** Exact logits within existing numerical gates, approximately
1 GiB lower active Metal memory, and no more than 2% standalone decode loss. A
small direct loss may be accepted only if recovered headroom enables a larger
net speculative gain.

**Result:** SUCCESS

- Branch/implementation commit: `feature/nemotron-paged-embeddings`, `f20aff9`
- `nemotron_paged_embeddings.py` strictly parses the packed safetensors layout,
  mmaps the exact 1 GiB BF16 input table, stages requested 8 KiB rows directly
  into MLX, and retains a bounded 256-row cache.
- The revision-bound catalog is
  `candidate-oqe512-r20-mlx/nemotron_paged_embedding_catalog.json` with SHA-256
  `bb87f39a19edac3929d8118d93f7e79e4e93cc915f2fbefbfe68bfc101cb1a8a`.
  Its exact 1 GiB payload SHA-256 is
  `5b7c77eafa74560858ee1358773defc6d778546d96a539e0a33d4ae950a662ee`.
- Unit bit-pattern validation passes. Separate-process full-vocabulary logits
  after the 12-token coding prompt were exactly equal: max absolute drift `0`,
  relative L2 `0`, and identical top-1 token `1293`.
- Ordinary paged runtime measured `56.677 GiB` active and `56.736 GiB` peak,
  exactly 1 GiB below the resident-table path. In paired speculative controls,
  ordinary decode averaged `23.310 tok/s` paged versus `23.342 tok/s` resident,
  a 0.14% difference within the 2% gate.
- Paired 128-token speculative controls averaged `34.748 tok/s` paged versus
  `32.816 tok/s` resident, a 5.9% gain. Paged peak was `57.34 GiB` versus
  `58.33 GiB`; every generated token ID was exact.
- Recovered headroom made adaptive depth two stable at `35.977 tok/s` and
  improved four-token lookup to `30.459 tok/s` on its repetitive control.
  Combining lookup and recursion was slightly slower than lookup alone.
- Rejected implementation: copying mmap rows through a temporary NumPy matrix
  cost about 0.39 ms per row. Direct mmap-to-MLX construction measured about
  0.03 ms on warm pages and is retained.
- Decision: promote paged embeddings into the performance default. Keep the
  resident-table path available as a control and fallback.

**References:** Apple documents
[no-copy Metal buffers](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))
and placement sparse buffers for recent Apple GPU families in the
[Metal feature tables](https://developer.apple.com/metal/limits/).

### [x] 5. NVFP4 Target Head With Exact BF16 Candidate Re-Ranking

**Goal:** Avoid a resident 1 GiB BF16 vocabulary projection while recovering
the BF16 greedy winner from a fast full-vocabulary NVFP4 candidate pass.

**Measured premise:** On the current 256 scored coding transitions, the BF16
winner appeared in the NVFP4 candidate set at these rates:

| Candidate set | BF16 winner recall |
| ---: | ---: |
| Top 1 | 94.92% |
| Top 2 | 99.22% |
| Top 4 | 99.61% |
| Top 8 | 100.00% |

The worst observed BF16-winner rank was five. This is evidence, not a global
correctness guarantee.

**Work:**

- Run the full NVFP4 head to identify a small candidate set.
- Fetch exact BF16 candidate rows from mmap-backed storage and re-score them.
- Develop an uncertainty certificate or conservative expansion rule.
- Provide a rare exact BF16 fallback when the winner cannot be certified.
- Test greedy, temperature, top-k, and top-p semantics separately; do not claim
  sampling equivalence from greedy evidence.
- Measure candidate recall on substantial coding, reasoning, prose, and
  adversarial low-margin traces.
- Account for mmap pages and fallback spikes, not only MLX active arrays.

**Success gate:** For greedy mode, exact BF16 target tokens under all acceptance
tests with at least 0.5 GiB resident savings and no throughput regression. Other
sampling modes remain unsupported unless independently proven correct.

**Result:** REJECTED

- Branch/implementation commit: `feature/nemotron-reranked-head`, `1077834`
- Reproducible report: `head-rerank/certificate-coding-256.json`, SHA-256
  `6d4c30a8cb9d878f39ae753562874f9af2393e7c4a4d66f80108a4ecfc734b0e`.
- The benchmark dequantizes the existing 281 MiB NVFP4 head, computes exact
  BF16-vs-NVFP4 error norms for every 16-value group, and applies a conservative
  sum-of-group-Cauchy upper bound to every non-candidate token.
- Empirical recall remained strong: top-1/2/4 recalled the BF16 winner on
  95.31%, 99.22%, and 100% of 256 coding transitions. Recall is not a proof.
- Exact certification was unusably weak at practical candidate sizes: top-4
  certified 1/256, top-64 17/256, and top-512 47/256 transitions. Thus a
  512-candidate path would still require full BF16 fallback over 81% of tokens.
- Even 32,768 candidates, requiring 256 MiB of exact BF16 row reads per token,
  certified only 232/256 (90.63%), leaving a 9.37% full fallback rate.
- Decision: reject. Heuristic top-k re-ranking would usually be correct but
  cannot preserve authoritative target semantics. Conservative certification
  requires enough exact work that it defeats the memory-bandwidth objective.

## Phase 2: Structural MoE Compression

### [x] 6. Full-Router Proxy Experts

**Goal:** Preserve the original 512-way router while storing fewer physical
experts.

**Hypothesis:** Hard pruning removes router choices and changes competition.
Mapping removed expert IDs to functionally similar retained prototypes may
preserve more behavior. When multiple selected IDs map to one prototype, their
routing weights can be summed and the prototype computed once, improving both
size and selected-expert work.

**Work:**

- Capture per-layer routing co-occurrence and expert-output signatures.
- Cluster by functional output similarity, routing behavior, and coding/general
  coverage; do not cluster solely by weight distance.
- Keep all router rows and correction biases.
- Add an immutable original-expert-to-prototype map per layer.
- Aggregate routing scores after mapping and execute each unique prototype once.
- Compare 25%, 30%, and 35% physical-expert reductions against hard pruning.
- Measure unique selected prototypes per token and actual expert-kernel traffic.

**Success gate:** Better logits and downstream quality than hard pruning at the
same physical expert budget, strict map/router validation, and no decode loss.
Promotion requires diverse evidence rather than coding-only calibration.

**Result:** REJECTED

- Branch: `feature/nemotron-proxy-experts`
- The source-only observer captured same-input output cosine, co-selection,
  route-score product, and category coverage for all 40 LatentMoE layers over
  512 diverse tokens. Calibration arrays are bound to the pinned source and
  stored at `proxy-calibration-oqe512/observations.npz`, SHA-256
  `03c31b8a30e01c35bdb082241efc72399048f7c384edac218d43b5929d8fe36`.
- Functional substitutes are weak. Across observed experts, the best retained
  co-selected neighbor with support of at least two had median output cosine
  `0.1144`; only 14.4% exceeded `0.2`, 3.19% exceeded `0.3`, and 0.238%
  exceeded `0.5`.
- The revision-bound 20% map was tested on two held-out 32-token batches at
  early, middle, and final MoE layers. Proxy routed-branch relative-L2 error
  had geometric mean `0.11326` versus `0.07783` for exact hard pruning at the
  same 410-expert payload, a 45.5% regression. Proxy complete-output error was
  also worse: `0.01652` versus `0.01136`.
- Reproducible comparison:
  `proxy-calibration-oqe512/comparison-r20-2x32.json`, SHA-256
  `e9246c34132dfd5fb03f62ef375651324ee341c6d4f0aad2361cfb1b8adf22de`.
- Mapping reduced 22 selected IDs to only `21.359` unique prototypes on
  average, a 2.91% expert-dispatch reduction. This is too small to offset the
  quality regression or justify a specialized aggregation kernel.
- Decision: reject nearest-prototype substitution. The implementation stops
  before resident-runtime integration because its core quality gate already
  fails. Layerwise fitting or distillation remains a distinct later experiment.

**Reference:** [MergeMoE](https://arxiv.org/abs/2510.14436) formulates expert
compression through merged outputs and summed routing contributions rather than
only parameter averaging.

### [x] 7. Nonuniform Per-Layer Expert Budgets

**Goal:** Spend the expert-memory budget where it produces the most quality.

**Hypothesis:** Uniform 20%, 30%, or 35% reductions assume all 40 LatentMoE
layers have equal redundancy. Per-layer sensitivity curves should permit more
aggressive compression in redundant layers while protecting sensitive layers.

**Work:**

- Generate held-out layer inputs from a diverse calibration corpus.
- Measure each layer at several expert budgets with all other layers unchanged.
- Score output error, router coverage, downstream logit drift, and category
  coverage rather than selection count alone.
- Solve a constrained allocation problem for target payloads near 55, 52, and
  49 GiB.
- Materialize plans with per-layer budgets and strict remapping validation.
- Compare against uniform plans at exactly matched payload.

**Success gate:** Better quality than a uniform plan at the same bytes, with no
unobserved-expert removal and no category-specific collapse.

**Result:** PARTIAL

- Branch/implementation commit: `feature/nemotron-layer-budgets`, `df73c9e`.
- Dynamic masked-source execution is exactly equal to a physically packed
  candidate (`relative_l2=0`, `max_abs=0`) and sweeps all 40 MoE layers without
  materializing every budget combination.
- Eight independent categories were measured at 10-45% layer cuts. Report
  `layer-sensitivity/heldout-8x32-budgets10-45.json` has SHA-256
  `758c3ddd917aee2d251c06705b41211ea4cd74f59b5ee34cd2c587a0113cf30b`.
- Exact dynamic programming produced byte-matched 25/30/35% allocations. The
  local robust-output objective improved 41.7%, 39.4%, and 18.9%; every
  calibration category improved over its uniform counterpart.
- The r25 plan keeps 308-512 experts per layer, averaging exactly 384. Its
  SHA-256 is `977828d95e930948c8b0e3d55da0253bb0a5da3bd1a6a69539ddd42dbd5211a6`.
  The exact-preserving candidate is `54.4974 GiB`; its pack report SHA-256 is
  `4579b1449cd1f1079ad2f1c2334139e7229d970094645d811d9f76286fc103f6`.
- On a second untouched eight-category logit set, nonuniform r25 retained the
  source top token on 7/8 cases versus 5/8 for uniform r25. Mean KL improved
  from `0.45654` to `0.08358`, worst KL from `3.23938` to `0.21477`, and
  top-64 overlap from `54.75` to `55.125`. Mean centered drift was slightly
  worse (`0.07554` versus `0.07385`), and coding/general cases were mixed.
  Report SHA-256:
  `3d115f7208ca9fd139050c9eead20fe511836df70cbf5d6df04dafd654cb948f`.
- Resident paged-embedding decode peaked at `53.729 GiB` and measured
  `23.610 tok/s` over 63 transitions, preserving r20 ordinary throughput while
  saving about 3 GiB. The coding continuation was coherent but is only a smoke.
- The candidate-bound 32K MTP map produced exact speculative output at
  `34.195 tok/s`, 76.19% acceptance, and `1.419x` speedup over its measured
  `24.099 tok/s` ordinary control. Peak memory was `54.333 GiB`. Log SHA-256:
  `2a4b041ff8f2bbb0ffdcdb446f7e8b9eab7cfa94f630c5c6514e1bdcad65997c`.
- Decision: retain as promising completed infrastructure, but do not promote it
  over r20 until substantial coding and instruction evaluations confirm the
  mixed per-category logit result.

### [ ] 8. Layerwise Expert Merging And Distillation

**Goal:** Recover behavior after proxying, merging, or deeper pruning without
ever loading the complete BF16 teacher into memory.

**Hypothesis:** The official model can be streamed one layer at a time. Captured
teacher inputs and outputs can train only the replacement experts, shared
projections, correction scalars, or router biases for that layer.

**Work:**

- Create a revision-bound layer-input/output capture format with bounded disk.
- Stream one official teacher layer and one candidate layer at a time.
- Begin with closed-form least squares or tiny correction parameters before
  full gradient training.
- Distill output vectors and router-weighted aggregate outputs, not expert
  weights alone.
- Hold out prompts and categories from every fitting pass.
- Test whether 30-35% physical expert reduction approaches the 20% candidate's
  quality.
- Quantize only after the merged/distilled BF16 candidate is accepted.

**Success gate:** A material downstream-quality recovery over the matching
training-free candidate, no held-out regression hidden by calibration fit, and
a reproducible bounded-memory pipeline.

**Result:** PENDING

**Progress:**

- `nemotron_mlx_layer_distill.py` now captures bounded teacher inputs/outputs,
  fits correction sidecars without changing retained NVFP4 tensors, and tests
  them on a separate corpus.
- Per-channel affine fitting overfit badly in the bounded smoke: mean held-out
  output error increased from `0.06382` to `0.07777` and worst error more than
  doubled. It is rejected.
- A two-parameter scalar affine correction per layer was stable over eight
  training and eight validation categories, but improved mean local output
  error only 1.87% (`0.07499` to `0.07358`) while worst error was effectively
  flat and slightly worse. Its 5.8 KiB sidecar is diagnostic, not promoted.
- Report: `layer-distill/r35-scalar-affine-8x24/report.json`, SHA-256
  `eee22e0e8d514b542b6a4defcf5d48dbe0bfb64152f3cd4f1c930417e2364f78`.
- Rank-4 hidden-to-routed-residual regression was tested both with and without
  per-channel bias. The unbiased full eight-by-eight run worsened mean output
  error from `0.07499` to `0.07563` and routed worst-case error from `0.879` to
  `1.406`. Report SHA-256:
  `32687f16ec34bda41abc6233c85b7a0bde3b5327263a451d5dd74cb3cac17d14`.
- Decision for this subfamily: reject all post-layer linear corrections. The
  next recovery attempt must train actual replacement expert outputs or router
  behavior; affine and low-rank residuals are in diminishing-returns territory.
- A rank-4 ReLU-squared adapter was then fitted inside the 1024-dimensional
  latent expert space, before `fc2_latent`, using 189 diverse training tokens
  and a separate eight-category validation set. It also failed: mean output
  error increased from `0.07499` to `0.07526`, and routed worst error increased
  from `0.879` to `1.356`. Only 8/40 layers improved both mean and worst error,
  with a best layer gain of 1.09%, too small for selective promotion. Report
  SHA-256: `ae74e5fe03df3b022d9a7b3b83ce836efe0ba5865d99807df8eddf49c4340dbb`.
- Small correction sidecars are now exhausted. Continuing this item requires
  training or constructing replacement expert parameters from a substantially
  larger teacher corpus and independently validating downstream logits.

**References:** [Sub-MoE](https://arxiv.org/abs/2506.23266) clusters experts by
functional outputs and merges shared subspaces. [MoE-Pruner](https://arxiv.org/abs/2410.12013)
reports gains from router-aware pruning and expert-wise knowledge distillation.

### [ ] 9. Shared Expert Subspaces With Small Residuals

**Goal:** Store common expert structure once while retaining expert-specific
behavior through compact coefficients or low-rank residuals.

**Hypothesis:** Functionally related LatentMoE experts may share substantial
input/output subspaces even when direct averaging destroys specialization.

**Work:**

- Cluster experts using held-out activation/output signatures.
- Measure joint SVD and shared-basis reconstruction curves independently for up
  and down projections.
- Compare shared basis plus per-expert coefficients, prototype plus low-rank
  residual, and direct merged-expert baselines.
- Include packed runtime size and extra arithmetic in every comparison.
- Build a fused selected-expert kernel only after the representation passes
  layer-output tests.
- Validate end-to-end quality before mixing this with lower precision.

**Success gate:** Better quality per resident byte than proxy-only experts and
no end-to-end throughput regression after the custom runtime cost. Reject the
format if decompression or extra matrix passes erase its memory benefit.

**Result:** PENDING

### [ ] 10. Aligned Routed-Expert Width Pruning

**Goal:** Preserve every expert specialization and the original 512-way router
while reducing routed payload and expert arithmetic in exact NVFP4 blocks.

**Hypothesis:** Removing low-contribution 16-neuron groups within each expert
can outperform deleting complete experts on layers where routing diversity is
more valuable than full per-expert width.

**Work:**

- Rank aligned neuron groups from route-weighted exact block-output energy.
- Slice matching up-projection rows and down-projection columns together.
- Preserve retained packed nibbles, block scales, and global scales exactly.
- Choose width pruning or whole-expert pruning independently per layer under an
  equal routed-byte budget.
- Materialize the hybrid only after independent layer gates, then require
  full-logit, generation-quality, memory, and throughput validation.

**Success gate:** Better downstream quality than the same-size nonuniform
whole-expert candidate, no retained-byte drift, and no decode regression.

**Result:** PENDING, PROMISING

- At 25%, width changes `168 -> 126` groups and `2688 -> 2016` neurons while
  retaining all 512 experts and the original router.
- The real NVFP4 dequantized reference matches packed gather-QMM at
  `2.05e-7` relative L2, and the existing kernel accepts the narrowed shape.
- The 40-layer screening report found 13 width wins. Report SHA-256:
  `2dcc1046673a5e8682eaf9c5eb7acbd153ef5603d721926c6cfb0d5b8234d573`.
- Independent 128-token calibration reconfirmed 10/13 screened winners with a
  mean width-to-hard local output-error ratio of `0.9233`. Layers 1, 3, 8, 59,
  and 70 were strongest. Report SHA-256:
  `4508624ecfe6586595e81d46e446e4f8f976549e8ec0a826e5b1521d18630f00`.
- This is a hybrid signal, not evidence for narrowing all layers. Layers 12,
  63, and 65 failed the independent mean-error gate and revert to hard pruning.
- `nemotron_mlx_hybrid_plan.py` and the streamed source runtime then tested
  mixed plans without materializing another 55 GiB artifact. The budget-neutral
  uniform hybrid was rejected by math KL. An eight-layer nonuniform hybrid was
  reduced through coding-logit ablation to layers 1, 8, 19, and 54.
- The four-layer plan improved eight-category mean KL from `0.07049` to
  `0.05254` and mean centered drift from `0.09597` to `0.09133`, but retained
  only 7/8 top-1 versus 8/8 for nonuniform r25. Report SHA-256:
  `4743e479cc5b298f290fa93d7a8150908388f5acc8462c3ed55a6b618a065b05`.
- Tool-calling ablation found that layers 1, 8, and 19 each flip the baseline
  top token when substituted independently. Layer 54 alone preserved top-1;
  its one-layer plan is the only remaining width candidate and needs the full
  quality gate before materialization. Ablation report SHA-256:
  `57069ba94aea45673d1f59f5d30f4b1b50123087cf296679da72d07ee54729af`.
- The final layer-54-only plan passed all eight categories with 8/8 top-1. Mean
  KL improved from `0.07049` to `0.06130`; mean centered relative-L2 improved
  from `0.09597` to `0.09381`; worst KL improved slightly from `0.11927` to
  `0.11916`. Report SHA-256:
  `3ec6777754a1f9c02769de9e54729d39a6edc1f6db5484ad0911aa0add3a4006`.
- Decision: promote layer 54 to the physical-pack implementation gate. The
  projected base payload is approximately 54.8 GiB without MTP, so full
  materialization must wait for more than the current 53 GiB disk headroom.

## Combined Candidates

Do not create combined candidates until their individual components have
completed results. Record combinations here when justified:

- [ ] Runtime combination: paged embeddings plus reduced-vocabulary MTP plus
  adaptive multi-token/n-gram speculation.
  - **Result:** BLOCKED ON ITEMS 2-4; ITEM 1 SUCCEEDED
- [ ] Compression combination: nonuniform proxy experts plus layerwise
  distillation.
  - **Result:** BLOCKED ON ITEMS 6-8
- [ ] Aggressive candidate: approximately 30-35% physical expert reduction,
  paged embeddings, and an accepted compact target-head strategy.
  - **Result:** BLOCKED ON ITEMS 4-8

## Result Template

Use this compact form when closing an item:

```text
**Result:** SUCCESS | PARTIAL | REJECTED

- Commit/branch:
- Artifacts and hashes:
- Quality evidence:
- Performance:
- Resident and peak memory:
- Disk payload:
- Decision and reason:
- Follow-on constraints:
```

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
- Integration baseline: `nemotron-main` through merge `022052b` before item 14
- Preferred balanced target: `candidate-mbpp-success-swap400-r25size-mlx`,
  `54.497424 GiB` payload
- Quality-headroom rollback: `candidate-mbpp-success-guard400-mlx`,
  `55.6540 GiB` payload
- Memory-first performance target: `candidate-r25-nested-remove400-mlx`,
  `53.340801 GiB` logical payload
- Ordinary decode baseline: `25.372 tok/s` on the current coding control
- Default memory-first speculative runtime: mmap-paged exact BF16 input
  embeddings, candidate-specific NVFP4-256 MTP sidecar, and the shared-target
  32K BF16 vocabulary map
- Default memory-first speculative result: `45.751 tok/s` over two 512-token
  controls, `1.803x`, `53.712 GiB` peak, exact token identity
- Full 131K BF16 MTP projection remains the acceptance-oriented fallback
- Smaller draft fallback: candidate-specific NVFP4-128 MTP sidecar at
  `43.888 tok/s` and `53.342 GiB` peak
- Lower-memory fallback: NVFP4-64 MTP sidecar plus draft-only NVFP4 head
- Fallback result: `30.885 tok/s`, `1.302x`, `58.436 GiB` peak, exact token
  identity
- Measured kernel limits: `iogpu.wired_limit_mb=58368` (`57 GiB`) for remove400
  and `60672` (`59.25 GiB`) for larger candidates
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

**Result:** REJECTED

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
- A fixed-size success-aware planner subsequently replaced 400 retained expert
  identities while preserving every per-layer r25 count. It combines broad
  calibration, a disjoint specialist corpus, successful MBPP recovery
  trajectories, and explicit regression guards. The guard50 plan SHA-256 is
  `8a67086c830ee87d267c8c4296c890b6621c904b9fb5d1c297eb3560a67894cf`.
- The untouched eight-category gate retained 7/8 source top tokens and improved
  mean centered drift from `0.07554` to `0.07183`, with mean KL `0.08318`
  versus r25's `0.08358`. Physical and virtual 131,072-way logits are
  bit-exact; the materialized runtime peaks at `53.729 GiB`.
- Deterministic MBPP improved from 74/100 to 75/100 and exactly matched all 100
  guard400 outcomes. Complete HumanEval improved from 154/164 to 155/164. It
  gained tasks 108 and 54 versus r25 but lost task 130 through a 768-token
  overlong response. Report SHA-256 values are
  `63c6aa95140a6d35684304e479a514cb7b1fe2d0c6d15275cff9c719e0a4b7a6`
  and `ebecdd794146ca7e7a67934f934a3708b916d018c775af024292f9428def5a63`.
- A matched 64-token ordinary control measured `24.462 tok/s`, `40.856 ms`
  median, and `53.730 GiB` peak. The candidate-bound shared 32K MTP map then
  reached `36.538 tok/s`, 85.0% draft acceptance, `1.491x` speedup, and
  `54.333 GiB` peak with exact output identity.
- The matched hidden LiveCodeBench replay scored 36/60 samples and 22/30 tasks,
  versus r25's 37/60 and 22/30 and guard400's 36/60 and 23/30. Against r25 the
  strict matrix is 30 both-pass, seven r25-only, six candidate-only, and 17
  both-fail. Easy/medium/hard scores are 19/14/3 versus r25's 18/15/4. Paired
  report SHA-256 values are
  `96bf57ce1d12c6ff01bcba7c83a1e94b4092e9a8f8c06036a6b78135c80199ef`
  and `56a33bc838e088d3eb19a55b8110fe3d73a1da7fb439c61a3fb0ac822299f90d`.
- Decision: promote the fixed-budget plan as the preferred balanced 64 GB
  runtime. It is not a strict quality dominance, but the +1 MBPP, +1 HumanEval,
  same r25 LiveCodeBench task coverage, exact payload integrity, and guard400
  quality at 1.1566 GiB less memory justify the trade. Keep guard400 as the
  quality-headroom rollback.

### [x] 8. Layerwise Expert Merging And Distillation

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

**Result:** REJECTED FOR PROMOTION. The bounded end-to-end training mechanism
is proven, but the completed multi-sample run regressed both downstream coding
suites.

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
- Router-only local distillation was implemented as a bounded follow-up. It
  updates retained BF16 router rows through sparse selected-score gradients,
  freezes every expert/projection, and rolls each layer back unless a disjoint
  local validation split improves. On r30 success200 it changed 23/40 layers
  and improved mean held-out layer output error by only 0.75%. Report SHA-256:
  `d293317760aa42e9ee5d6ab5e2cd8fee5601d1ffdbe51d7c8b0e7b5cca0c650a`.
- The full-logit gate rejected that local surrogate after two cases. Coding
  completion KL worsened 12.0%; coding-debug KL improved 21.5% but changed the
  source top token, leaving 1/2 top-token agreement versus 2/2 for unmodified
  r30. Partial report SHA-256:
  `70c46a90d01eb2957cc8f4a4ae6d735c2ece72c5b99ecce82d6532127c70cfaf`.
- This does not test the published end-to-end Router KD objective, which uses
  next-token KL through the complete student. Do not substitute local layer
  MSE for that objective or materialize the rejected router artifact. A future
  attempt requires complete-graph training hardware or a proven streamed
  backward/checkpointing design.
- A two-token gradient audit subsequently proved that all three NemotronH block
  types can participate in a bounded backward pass. Mamba's production path is
  bit-exact; native-BF16 training fallbacks for attention and MoE preserve
  forward output within `8.58e-7` and `3.34e-7` relative-L2 while producing
  finite nonzero input gradients. Peak memory was `5.266 GiB`; report SHA-256:
  `123dfc079c7f802ab0ea01f1f76256538f67c0f77c410aabd116e974f1141006`.
- `nemotron_mlx_streamed_router_kd.py` completes the manual layer-streamed
  backpropagation gate. It atomically saves bounded activations/cotangents,
  forms true full-vocabulary next-token KL, reloads one frozen layer at a time
  in reverse, and exports only BF16 retained-router rows after an improving
  line search. Its exact-value BF16/FP8 training fallbacks match the production
  virtual student at `5.77e-9` KL; the full run peaks at `4.195 GiB`.
- The final two-token r30 proof has finite nonzero gradients in all 40 MoE
  layers. The accepted `5e-5` step reduces teacher KL from `0.00532214` to
  `0.00319917`, preserves teacher top-1, raises top-64 overlap from 59 to 62,
  and certifies every zero-gradient row remained exact. Report SHA-256:
  `9898268f825d06e016aaebb466aa4ff7688ea8f405879ad9403ce6573850e170`.
- The two-token mechanism proof is `SUCCESS`, but its router is deliberately
  overfit and must not be packed. The completed multi-sample result below is
  the experiment-level promotion decision.
- The required multi-sample run used 24 objectives across eight categories and
  three prefix lengths, plus eight disjoint full-prompt validations. Its
  accepted step improved validation mean KL by 7.88% and maximum KL by 14.95%.
  Reverting the last 10 MoE routers then passed both bounded logit gates.
- The resulting 51.6059 GiB physical runtime was exact: 89/89 groups rehashed,
  all 40 routers matched the composed sidecar byte-for-byte, virtual/physical
  full logits were bit-identical, and resident decode measured 24.139 tok/s at
  a 50.838 GiB peak.
- Downstream quality rejected it. MBPP was 73/100 versus unmodified r30's
  74/100, losing only task 125; HumanEval was 149/164 versus 150/164, losing
  only HumanEval/147. There were no candidate-only wins. Paired report SHA-256:
  `c20cf1909fca87f2ebcf211376971b6f062ca61a7b45e9f26c98872e4760c0b1`.
- Router-layer reversion localized MBPP 125 to layers 23, 28, or 30. Full
  layer-30 reversion and 0.75 BF16 delta damping repaired that task but failed
  broader gates; damping reached `0.619237` worst KL on the disjoint coding
  set. Decision: stop post-hoc router fitting for this corpus. Any new attempt
  needs expert-capacity training or a new pruning allocation with downstream
  success evidence.
- The rejected 51.6059 GiB physical pack was deleted after preserving its
  provenance, parity, performance, and quality reports. It is reproducible
  from the immutable source, r30 plan, and retained composed router sidecar.

**References:** [Sub-MoE](https://arxiv.org/abs/2506.23266) clusters experts by
functional outputs and merges shared subspaces. [MoE-Pruner](https://arxiv.org/abs/2410.12013)
reports gains from router-aware pruning and expert-wise knowledge distillation.
[Router KD](https://arxiv.org/abs/2603.02217) updates only compressed-model
routers from teacher next-token distributions and is not equivalent to a
teacher-forced local layer loss.

### [x] 9. Shared Expert Subspaces With Small Residuals

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

**Result:** REJECTED FOR POST-TRAINING NEMOTRON COMPRESSION

- `nemotron_mlx_shared_subspace.py` implements a bounded real-layer screen for
  three representations: an NVFP4 pair prototype plus one shared low-rank
  difference, a shared output basis with expert-specific coefficients, and the
  transposed shared input basis. It clusters only retained experts using
  same-input output cosine and validates against independently captured routed
  latent inputs.
- Layer 1's two strongest disjoint supported pairs failed immediately. At rank
  64, prototype-plus-residual expert-output relative-L2 remained approximately
  `1.0` while projected BF16 savings were 34.66%. Raising rank to 192 reduced
  savings to 3.97% without improving mean error below `0.997`.
- A global 40-layer scan found layer 14 experts 104 and 392 as the strongest
  retained pair: output cosine `0.98753` with 16 calibration observations.
  Eight independent categories routed 32 and 13 held-out candidate-route
  samples to the pair.
- Despite that unusually strong functional match, rank-256 output/input union
  bases produced `0.9637/0.9593` mean output relative-L2. Rank 512 still
  produced `0.8225/0.7085`; it saves only 7.94% if every factor is stored in
  lossless-equivalent FP8. Prototype-plus-residual reached `0.3169` at rank
  512 but was already 11.38% larger than the original NVFP4 pair under the
  same optimistic FP8-factor assumption.
- Final retained-route report SHA-256 is
  `fad10eb3d50a496c318d3dac30f15cef9c8ed4482bd2bc9b66958922fbd2aed8`.
- Decision: do not build a fused runtime or materialize this representation.
  Functional similarity does not imply a sufficiently low-rank shared weight
  space for Nemotron's tiny latent experts. Quantizing the factors would only
  worsen an already failed float32 reconstruction. Reopening this item requires
  training the shared representation, not another post-training decomposition.

### [x] 10. Aligned Routed-Expert Width Pruning

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

**Result:** REJECTED AS DEFAULT

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
- Incremental materialization succeeded by hard-linking unchanged groups and
  writing only layer 54. Logical payload is `54.7182 GiB`; incremental disk use
  is about 1.2 GiB. Physical and virtual eight-token logits are bit-exact.
- Ordinary paged decode measured `23.997 tok/s` at `53.953 GiB` peak. Exact MTP
  decode measured `35.068 tok/s`, 76.92% acceptance, `1.433x` speedup, and
  `54.553 GiB` peak. The candidate is now usable and advances to substantial
  coding/instruction evaluation rather than more representation changes.
- A deterministic 20-task MBPP pass@1 gate scored 15/20 for both the hybrid and
  nonuniform-r25 control. Pass/fail matched on every task; 15/20 generated
  responses were byte-identical, and all five divergences preserved the same
  test result. The layer-54 substitution therefore passes the initial coding
  parity gate but has not yet shown the downstream-quality improvement required
  to close this item as SUCCESS.
- Expanding to 100 deterministic tasks scored 73/100 for the hybrid and 74/100
  for nonuniform r25. Task 376 was the only pass/fail delta, and it favored the
  control; there were no hybrid-only wins. Nonuniform r25 also tied r20 at
  74/100 with four paired wins in each direction. A separate 20-task HumanEval
  gate tied nonuniform r25 and r20 at 17/20 with all task outcomes aligned.
  The full corrected r25 HumanEval run scored 154/164 (93.90%) with zero
  timeouts, syntax failures, or output truncations.
  A separate 10-task LiveCodeBench public-test gate scored 5/10, including 1/5
  hard problems and one explicit 2,048-token truncation. This limits the quality
  claim to strong function-level coding rather than broad competitive coding.
  The balanced 30-task follow-up scored 16/30: easy 9/10, medium 5/10, and hard
  2/10, again with one hard truncation. The larger gate confirms that conclusion.
  The width hybrid therefore fails its downstream-quality success gate and is
  retained only as experimental evidence, not as the preferred candidate.
- Disk cleanup removed the reproducible r20 and width-hybrid artifacts after
  their provenance-bound reports completed, recovering about 59 GiB. The
  immutable NVFP4 source and preferred nonuniform-r25 candidate remain local.

## Phase 3: Alternative Kernel Toolchains

### [x] 11. Mojo Selected-Expert NVFP4 Kernel

**Goal:** Determine whether Mojo can produce a faster portable Nemotron
selected-expert kernel than the current MLX/Metal path while retaining exact
ModelOpt NVFP4 semantics.

**Result:** REJECTED FOR APPLE RUNTIME

- Branch: `feature/nemotron-mojo-moe-spike`.
- Toolchains: stable Modular 26.2 / Mojo 0.26.2.0 and nightly Modular 26.4 /
  Mojo 1.0.0b2. The retained isolated environment is `mojo-env-26.4` under the
  model root. Modular source is pinned at `2a5c98bacfd69a2d4edb5e6eecb56ed58735bcb1`;
  the MLX control source remains pinned at
  `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`.
- Correctness: the full production shape uses 22 selected experts, 1,024
  latent dimensions, and 2,688 routed hidden dimensions. Constant-pattern
  ModelOpt NVFP4 input agrees with the analytic result at zero maximum error
  on the best row-parallel path. E2M1 is decoded through MLX's exact
  half-bit construction and E4M3FN through Mojo's native bitcast.
- Size and quality: unchanged. The spike consumes the existing packed NVFP4
  representation and does not requantize, prune, or modify model tensors.
- Implemented variants: one threadgroup per expert with the activated hidden
  vector in threadgroup memory; one SIMD group per output row; fused down and
  router reduction; and MLX-inspired four-row SIMD tiles. All buffers and
  intermediates remain GPU-owned within one Mojo `DeviceContext`.
- Stable Mojo was not competitive: best measured row path was approximately
  `0.77 ms`. Mojo 1.0 beta 2 improved the same source to approximately
  `0.29-0.30 ms`, or about `224-232 GB/s` of effective packed payload.
- Matched current-runtime control on real packed layer 1 was `0.175186 ms` for
  the complete MLX up/ReLU-squared/down/reduction path, with zero drift from
  its direct selected reference. Mojo remains approximately 67% slower.
- Decision: do not integrate Mojo into the Apple resident runtime. It would
  add a second Metal owner with no documented zero-copy MLX custom-op boundary
  and would regress the isolated hot path. Retain the spike as compiler and
  CUDA/RTX reference evidence; NVIDIA NVFP4 grouped kernels require a separate
  5090 benchmark and must not be inferred from this Apple result.
- Reproduce with `NEMOTRON_MOJO_REAL_MOE=1 nemotron/check.sh` or directly with
  `nemotron/run_mojo_moe_spike.sh`. The routine check skips GPU compilation.

## Phase 4: Safe-Memory Quality Frontier

### [x] 12. Nested R25 Frontier Below The MLX Allocator Boundary

**Goal:** Retain the preferred r25 quality profile while moving sustained
resident inference below both the guarded physical-memory fraction and MLX's allocator-GC
boundary on the 64 GB M4 Max.

**Result:** PARTIAL

- Remove400 reached `53.3408 GiB` logical, 74/100 MBPP, 155/164 HumanEval,
  and `23.434 tok/s`, but its `52.572 GiB` peak exceeded the 52.25 GiB MLX
  allocator-GC threshold under a 55 GiB wired cap.
- Its hidden LiveCodeBench run kernel-panicked after 4/60 samples in
  `IOGPUGroupMemory::remove_memory_object()`. The report is retained as crash
  evidence and must not be resumed unattended.
- Resident sequence reset now synchronizes Metal and reuses recurrent/KV
  storage. Extended-run preflight checks physical-memory fraction and the
  allocator-GC boundary; all resident quality gates log active/cache/peak
  memory per sample.
- Remove1000 passed the complete virtual gate at 7/8 top-1, mean KL `0.07787`,
  and worst KL `0.20778`. Its physical runtime is `51.6059 GiB` logical and
  `50.6059 GiB` resident with paged embeddings.
- Remove1000 physical and virtual logits are bit-exact. A 64-token resident
  run reached `24.423 tok/s` with a `50.837 GiB` short-run peak.
- Its first 55 GiB-cap MBPP attempt was stopped safely at 39/100 after a
  512-token completion raised the process peak to `52.128 GiB`, only 128 MiB
  below MLX's allocator-GC boundary. The partial score was 25/39 versus 27/39
  for the preferred candidate on the same tasks.
- A second attended run at 56 GiB reached 98/100 before a rarer prompt raised
  peak active memory to `53.134 GiB`; the live guard again stopped before the
  `53.20 GiB` allocator boundary. The final two tasks resumed at 57 GiB.
- The completed MBPP gate scored 70/100 versus 74/100 for the preferred
  swap400-r25size candidate: two candidate-only passes and six control-only
  passes. It scored 70/100 versus 75/100 against the quality-headroom guard400
  candidate. This is a material regression, so HumanEval and LiveCodeBench
  were not run.
- Extended-run preflight now reserves `3.25 GiB` above resident payload for
  measured transient work and applies a live 128 MiB stop reserve. At least a
  57 GiB wired cap is required for long quality gates at this payload.
- The 793-token task-380 prompt isolated the transient spike. Report-bound
  stateful 128-token prefill reduced remove400 peak from `54.520 GiB` to
  `53.461 GiB`; final logits retained the same top-10 with KL `1.89e-7`, and
  the end-to-end completion remained byte-identical and passing.
- The complete 100-task MBPP rerun remained 74/100 with all 100 responses and
  extracted programs byte-identical to the prior whole-prompt report. Peak
  stayed at `53.466 GiB`.
- The formerly crashing hidden LiveCodeBench gate completed 60/60 samples and
  generated 104,225 tokens at a `53.591 GiB` peak. It scored 36/60 samples and
  21/30 tasks: easy 19/20, medium 13/20, hard 4/20. The prior balanced control
  scored 36/60 and 22/30 under the older runtime, so this is a mixed
  cross-runtime comparison with no category collapse, not a strict matched
  pruning estimate.
- Measured bounded-prefill preflight now uses `1.625 GiB` transient workspace
  and an 85% physical-memory ceiling while retaining the 95% allocator and live
  128 MiB reserve gates. Remove400 passes at exactly a 57 GiB wired cap; larger
  candidates remain blocked there. Generic paths keep the 3.25 GiB allowance.
- Remove1200 is rejected: tool-calling KL reached `2.28239`, changed top-1,
  and ranked the source token eighth. Remove1400 is not worth evaluating.
- The reproducible r27.5 repair150 artifact was deleted after its reports and
  plan were verified, recovering about 36 GiB before remove1000 materialization.
- Fixed-size repair20/repair40 plans lowered trajectory-local error but raised
  independent mean KL from `0.077868` to `0.081856`/`0.083312`; neither was
  materialized. The remove1000 physical artifact was deleted, recovering about
  35 GiB while preserving reproducible plans and reports.
- Decision: reject the plain activation-ranked remove1000 plan and its tested
  trajectory-repair direction. Continue from the high-quality remove400/r25
  controls with bounded prefill rather than deeper post-training expert cuts.
- Decision: retain remove400 as the stable 64 GB memory-first fallback. It saves
  `1.1566 GiB` while preserving strong MBPP/HumanEval and a mixed, near-control
  hidden coding profile; the balanced candidate remains the quality default.
- A candidate-bound shared-target 32K MTP map was materialized at
  `mtp-vocab-map-bf16-e32768-r25-nested-remove400`; its 128 KiB artifact SHA-256
  is `83c6d25815c89946f5701927c61820049d2fc0d244426cba63d99a9bf031e7e0`.
  It is bound to remove400's pack-report SHA-256
  `d2b09fd6014147dcd0fc32eca116da7575045be28c144e429dcde5e64543f09d`.
- Fusing MTP token/confidence reductions and batching every target-verifier
  row-wise argmax removed redundant Metal synchronizations without changing
  greedy semantics. With adaptive depth two and a 256 MiB MLX cache, the
  two final-code 512-token coding controls averaged `40.535 tok/s` versus
  `24.191 tok/s` ordinary (`1.676x`), accepted 90.22% of drafts, and peaked at
  `53.358 GiB`. Output token IDs were exact. Log SHA-256 values are
  `6161e12daf3a11cabf483b369e2a4372424fa142a857bc2804aadf62c9ea27d7`
  and `2fc2d472b6cb655a4f0f8547e2935c7d55858c8ae79e4fa0aa60c226c370d612`.
- Distinct 256-token reasoning, coding, and technical-instruction controls
  remained exact and measured `36.628`, `37.334`, and `27.099 tok/s`, or
  `1.515x`, `1.538x`, and `1.124x` over their paired ordinary runs. This is the
  lowest measured control, not a universal workload floor or a ceiling on
  useful repetitive coding output.
- Candidate-specific teacher replay selected a different fixed-size 128-expert
  MTP plan with 103/128 experts shared with the generic sidecar. Its NVFP4
  artifact at `mtp-sidecar-e128-remove400-nvfp4` is `0.4339 GiB`, preserves the
  target weights, and has SHA-256
  `a42b4f167183c313ca5130e0c8800955fb8ea2ea0b25ebd00554d1a88a82ee75`.
  On two exact 512-token coding controls it averaged `41.683 tok/s` versus
  `24.212 tok/s` ordinary (`1.722x`), accepted 93.65% of drafts, and retained
  the same `53.358 GiB` peak. Log SHA-256 values are
  `dc5802679880b664bcb16c927d3de958a28fed2c8b18360ac10dbbcc2701b94c`
  and `c5698342749e0dc7a20038ba504cbda1bf99b1ff9d0b050ca5e656dc2c0eb151`.
- The candidate-specific sidecar also improved the exact reasoning control to
  `38.208 tok/s` and independent coding control to `39.887 tok/s`. Its
  technical-instruction trace accepted one fewer draft than the generic
  sidecar (42.63% versus 43.39%), so the candidate-specific artifact is the
  remove400 coding default while the generic sidecar remains the broad fallback.
- Fixed-budget 8/16-expert blends at candidate weights 0.5, 0.75, and 0.9 were
  screened against both the original and remove400 traces. None dominated the
  full candidate-specific plan: the best preserved 76.95% original top-1 but
  reached at most 75.78% on remove400, versus 76.56% for the candidate plan.
  No blended sidecar was materialized. The provenance-bound blend planner and
  its focused test are retained for future adaptation corpora.
- A model-specific short-sequence SSM Metal kernel now executes 2-8 recurrent
  steps per launch while retaining the one-token update order and optional
  accepted-prefix state capture. Against the matched token-loop verifier,
  two-token batching improved from `47.479` to `45.651 ms` (3.85%) and
  three-token batching from `60.250` to `57.612 ms` (4.38%). Full-vocabulary
  top-1, rollback, and captured continuation checks passed for 2/3/4/8-token
  blocks; full-logit relative L2 remained below `1.45e-7` and maximum absolute
  drift below `3.06e-5`.
- With that kernel enabled by default, two exact 512-token resident controls
  measured `42.737` and `41.869 tok/s`, averaging `42.303 tok/s` versus
  `24.131 tok/s` ordinary (`1.753x`) at a `53.342 GiB` peak. This is 1.49%
  above the pre-kernel candidate-sidecar mean. Log SHA-256 values are
  `a29010eebb769d8fc51359830fe7eeeee786d6eec5d24d7c6a3450bc6966006d`
  and `b409fe37798269691d3e1c28c59b835e998acc5d06c9d16f8ed15290c77c44df`.
- One weight-parameterized compiled MoE tail now covers the
  BF16/BF16/FP8/FP8 projection layout used by 34/40 target MoE layers. It does
  not capture layer weights and remains bit-exact against the eager equation
  for one-, two-, three-, and eight-token inputs. A representative layer
  improved 34.6% at one token and 22.7% at two tokens.
- Full-model two-token verification improved from `45.651` to `44.840 ms`
  (1.78%), and three-token verification improved from `57.612` to `56.032 ms`
  (2.74%). Top-1, rollback, and captured continuation checks passed; maximum
  full-logit drift remained `1.53e-5`.
- Two exact 512-token controls measured `43.786` and `43.747 tok/s`, averaging
  `43.767 tok/s` versus `25.223 tok/s` ordinary (`1.735x`) at a `53.344 GiB`
  peak. This is 3.46% above the short-SSM mean with the same 93.65% draft
  acceptance and identical output tokens. Log SHA-256 values are
  `709efd53ea2770ca4877ee347d784777396c95ce53ed52f65bcbaa3c93014e3e`
  and `9df8ba12dbbb2b5677262dca8ea2a5319ea6f74e092415c4ed293adde2582fcb`.
- Five static precision-signature graphs extend the same dynamic-weight design
  to the six uncommon MoE layers. Every real FP8/BF16/NVFP4 assignment remains
  bit-exact at one, two, three, and eight tokens. Isolated one-token tail gains
  ranged from 11% to 58%; three-token gains ranged from 7% to 21%.
- Full-model two-token verification improved again from `44.840` to
  `44.295 ms` (1.22%), and three-token verification from `56.032` to
  `55.716 ms` (0.56%), retaining the prior top-1, logit, rollback, and capture
  envelope.
- Three exact 512-token controls measured `44.240`, `43.543`, and
  `43.880 tok/s`, averaging `43.888 tok/s` versus `25.394 tok/s` ordinary
  (`1.728x`) at no more than `53.342 GiB` peak. The incremental speculative
  gain is 0.28% over the dominant-only mean; promotion is based on its exact,
  memory-neutral completion of all precision layouts plus the clearer 0.68%
  ordinary-decode and verifier gains. Log SHA-256 values are
  `e67a212004237bb7cfb409b67d2d44fdeef5331912ca29e44273a623632d5531`,
  `9272abd38b28562640e3dd0e2f706ad44e8d997fcfa9c638119d21f4e61dd8b6`, and
  `c37c900edfaa1930720a7d078cd0b714b7980222cc620f22dd450dd8fc267699`.
- Compiling RMSNorm and routing around each compiled tail is rejected. It was
  bit-exact across all 40 real MoE layers and appeared 25% faster in an
  isolated one-token layer screen, but full-model verification was unchanged
  at `44.270/55.673 ms` for two/three tokens. Two exact 512-token controls
  averaged `43.977 tok/s` versus the tail-only `43.888 tok/s`, while ordinary
  decode averaged `25.382 tok/s` versus `25.394 tok/s`. The extra compiler
  boundary was removed because its end-to-end change is noise. Rejected log
  SHA-256 values are
  `f2109ed7d297ab15a1ca465601d9cb3ad8e91646f0062f0b017563fe01ed6364` and
  `5b301f57ca41044318338e2938d0aedd343b21e63b84c58c60a5c2628715b0f1`.
- Compiling the post-recurrence gated normalization and FP8 output projection
  of each Mamba layer is also rejected. Native-equation comparison over all 40
  Mamba layers after four recurrent steps had zero output and state drift, and
  the isolated layer-0 tail improved by about 33%. Full verification remained
  effectively unchanged at `44.255/55.768 ms` for two/three tokens. Two exact
  512-token speculative controls averaged `43.993 tok/s`, while alternating
  256-token ordinary A/B runs averaged `25.3425 tok/s` eager and
  `25.3995 tok/s` compiled, a 0.23% difference. That is below the promotion
  floor and does not justify a second Mamba composition or compiler boundary,
  so the implementation was removed. Speculative log SHA-256 values are
  `3a9059691f0159fc840ff5ad7ef9d10fc1e9e4ed24408bb65ffed3a4aabbe352` and
  `ad5544ffab6fe7cc5ff3c489e3574f57598eb7f38117fbea4783d563607d321c`;
  alternating A/B log hashes are
  `b2397709b859181fa8e2a3c97b08b2392eedea72e355378191588a8f3e80d938`,
  `8ef33bb3282433d9a0f976fa2448d0996789742134d55c140ff633c457221ad5`,
  `183fc397706a487ac217cae7f5c62f0e5ef78488dbf8eccc7bc6f0874fc29344`, and
  `b112d19ef9c7e684ea7dcc8b8f7ca1a73f0c1ccdc71f39196ed47f0ff627b5de`.
- A bounded resident layer profile reproduced uninstrumented one/two/three-token
  latency at `38.881/44.552/55.584 ms`. For two tokens, forced synchronization
  attributed `24.927 ms` to Mamba, `29.367 ms` to MoE, `3.627 ms` to attention,
  and `2.575 ms` to the target vocabulary head. Its 1.379x synchronization
  inflation is reported explicitly. The next kernel investigation should
  target Mamba/MoE composition or verifier amortization, not the BF16 head.
  Profile log SHA-256 is
  `f3876bb289e44269d8932ac4523fd00ac75a9f4208c8d367ed26c57795df9a09`.
- Two larger Mamba fusion boundaries are rejected. A single convolution/SILU/
  SSM kernel retained bit-exact convolution state and stayed within the
  existing `3.05e-5` recurrent-state envelope, but recomputing shared B/C
  convolution values made synchronized two/three-token layers 5-16% slower.
  A sharing-preserving depthwise-convolution kernel improved isolated captured
  layer latency by 2-4%, yet full two/three-token captured verification
  regressed to `46.572/57.480 ms` from the production `44.295/55.716 ms`.
  Both implementations were removed. The full-verifier log SHA-256 is
  `7518f5efe2f258cc63dcde8d72ae932af31b13d9a168a3d159dc05556acadde7`.
- Additional policy searches are rejected. First-draft margin 1.0 and 2.0
  stayed within noise of the 1.5 default; replay-only rollback fell to
  `39.935 tok/s`; one-token lookup agreement fell to `37.831 tok/s`; and the
  stricter three-consensus/two-token-agreement lookup reached `41.583 tok/s`.
  Candidate-specific recursive acceptance was only 35.56% at depth three,
  below its verifier-cost break-even point. Keep capture rollback, thresholds
  1.5/1.0, depth two, and lookup disabled for the broad coding default.
- A 512 MiB MLX cache is rejected despite nominally fitting: allocator pressure
  collapsed speculative decode to `22.146 tok/s`. Recursive depth three is also
  rejected for this sidecar; even a high-confidence attempt gate accepted only
  7/14 third drafts and reduced throughput to `39.093 tok/s`. Keep the stable
  default at a 256 MiB cache and at most two MTP drafts.

### [x] 13. Recursive MTP Hidden Correction

**Goal:** Improve second-depth MTP acceptance with a tiny draft-only learned
correction while leaving the authoritative target and its output unchanged.

**Result:** REJECTED

- A rank-16 residual adapter was trained on 83 prompt-disjoint recursive pairs;
  78 held-out pairs selected epoch 1. The 500 KiB artifact preserves first-depth
  MTP exactly and raised held-out depth-two matches from 46/85 to 52/85.
- An independent eight-prompt remove400 trace confirmed the direction: aggregate
  depth-two matches rose from 59/141 to 68/141 and depth-three matches from
  19/56 to 28/65. The independent report SHA-256 is
  `6bb963f76faaad2e2b5bece07032cf272943562097b2d9a12298d21d73122838`.
- The exact resident gate did not reproduce an end-to-end gain. The adapter
  reached `44.222 tok/s`, 297/318 accepted drafts, and `53.343 GiB` peak; its
  matched no-adapter control reached `44.253 tok/s`, 295/315 accepted drafts,
  and `53.342 GiB` peak. The 0.07% throughput difference is noise, while total
  acceptance was slightly lower despite two additional accepted drafts.
- Adapter and control log SHA-256 values are
  `98673de7f543f172042d5fb314b421d2155b491138bc51d47793989e5f5427bc`
  and `8077e2644a5d375921320f62955d280b63f7fa713273f17d7e0755dcd1bfa99b`.
  Both runs reported `integrity=exact`.
- The production loader and CLI hook were removed. Retain the bounded trainer
  and independent evaluator as diagnostic tools; revisit only with a materially
  larger and more diverse adaptation corpus plus a matched resident acceptance
  win, not another fit to the same short trace.

### [x] 14. Larger Candidate-Specific NVFP4 MTP Sidecar

**Goal:** Spend bounded memory on a stronger draft model when the resulting
acceptance gain improves complete resident decode rather than only offline MTP.

**Result:** SUCCESS at depth two; depth three REJECTED

- The remove400 ranking was screened at 192 and 256 BF16 experts before
  materialization. Against its 256-row target trace, top-1 rose from 76.56% at
  128 experts to 77.73% at 192 and 80.47% at 256; the official 512-expert MTP
  reached 81.25%. Only the 256-expert candidate justified materialization.
- `mtp-sidecar-e256-remove400-nvfp4` is `0.803994 GiB`, 0.370118 GiB larger
  than e128. Its artifact SHA-256 is
  `fc1075d04fa822bb6027f9bb49d357952283b527685035a1d7cb13a5abe63bba`;
  pack-report SHA-256 is
  `7e7725ba8a2295786a70db5551b914237589e382fcadc71dfefe2692876a7079`.
  The reproducible 2.854 GiB BF16 intermediate was deleted after validation.
- On the ranking trace, NVFP4/32K recursive matches improved from e128's
  177/93/32 at depths one/two/three to 186/110/48. On an independent eight-prompt
  trace, they improved from 146/59/19 to 161/83/30. Independent e128/e256 report
  SHA-256 values are
  `b48bb201181571a117c3bb2d13c0fe38d33ddeca0a68e236eed371a18163b5a1`
  and `03b8ba9ed0e6bbe1c61cc9254a74e5a0d9513238c959e65121fe43369ad5e12c`.
- Two exact 512-token resident controls measured `45.515` and `45.986 tok/s`,
  averaging `45.751 tok/s` versus `25.372 tok/s` ordinary (`1.803x`). This is
  4.24% above the prior e128 production mean. Both accepted 305/315 drafts
  (96.83%) and peaked at `53.712 GiB`. Log SHA-256 values are
  `b62ee2eade746d5fb824fea3f472017beacd34cf3f9ce593a1cdf01a5503c015`
  and `d3d02a505f9284344a63a676dc8261678655998eaf15285964de98cdcdd48338`.
- Selective depth three regressed to `43.366 tok/s` and raised peak memory to
  `54.025 GiB`, about 128 MiB below the allocator boundary. Its implementation
  was removed; the rejected log SHA-256 is
  `397a7c47328a65b72d4c8bbd1fc764ca701f1e3fcbff032c9aef82e1c8804424`.
- Decision: e256 is the remove400 coding-performance sidecar at depth two.
  Keep e128 as the 0.370 GiB smaller fallback. The e256 preflight is suitable
  for bounded generation at a 57 GiB wired cap but intentionally fails the
  conservative unattended extended-run gate.

### [x] 15. E256 Adaptive Margin Retuning

**Goal:** Convert the stronger sidecar's additional low-margin correct drafts
into higher end-to-end throughput without changing target output.

**Result:** REJECTED

- Lowering both first/second thresholds from 1.5/1.0 to 1.0/0.5 increased
  second-draft rate from 69.35% to 77.60% but also raised rollback count from 10
  to 16. Exact throughput was `45.693 tok/s`, below the `45.751 tok/s` baseline
  mean. Log SHA-256 is
  `4643dc806c8dcef53844ebb060d4c34155194fd45c7d5fac94b5822508e23dc0`.
- Holding the first threshold at 1.5 and lowering only the second threshold to
  0.5 reached `45.332 tok/s` with 14 rollbacks. Log SHA-256 is
  `577feb9fd083a8d704eea7663525e332ce6c97c20a0ff9a4568940d595bda396`.
- Both runs retained exact token identity and the `53.712 GiB` peak. Keep the
  existing 1.5/1.0 policy; the remaining margin buckets do not justify more
  long resident sweeps without a new selection signal.

### [x] 16. Adaptive Expanded MTP Vocabulary

**Goal:** Recover correct e256 drafts that fall outside the production 32K
draft vocabulary without paying the larger projection cost on every cycle.

**Result:** REJECTED

- The nested 64K token map is only 256 KiB and covers 98.4431% of its held-out
  corpus, with no category below 97.1179%. Its payload SHA-256 is
  `440d96eedc4bb208e0da61449b95fe18294789b36f999fdb78c05cd35a2d8514`;
  report SHA-256 is
  `6cb4ab9cb06dc5f554d077b67614f4b9a5430ff5c5a589cc43b0b063c9a9c1b2`.
- Direct 64K projection improved first-depth offline matches from 186/256 to
  193/256 on the ranking trace and from 161/256 to 167/256 on an independent
  trace. The full target head reached 202/256, confirming that vocabulary
  exclusion accounts for some remaining draft misses.
- The larger projection did not improve resident acceptance. A direct 64K
  exact run accepted the same 305/315 drafts as production 32K and reached
  `44.778 tok/s`, below the `45.751 tok/s` 32K mean. Its log SHA-256 is
  `39211fabfa4f0b5f892bd980f854c8fa8298a33e61308ac39d2c7f8c418617a5`.
- A temporary adaptive path invoked 64K only when the 32K margin was below
  1.0. On the production control it made 45 fallback projections, changed one
  token, gained no accepted draft, and reached `45.283 tok/s`. On an
  independent C++ prompt it made 65 fallback projections, changed no tokens,
  retained 124/156 accepted drafts, and reached `38.206 tok/s` versus
  `38.812 tok/s` for 32K. Adaptive/control log SHA-256 values are
  `a8d000db7b318282beda27ca90c2f354ec9cd02ef7ada37bf1df1105d4a39d16`,
  `049dc1370ae38066947167c634acc96efa26b313dfdea65af829f007c6a0be39`,
  and `8dd963cfeca98558686280b6ab9f531271283237ad55edc6c2d6d3a30c720ab8`.
- The adaptive loader, CLI, and projection path were removed completely. Keep
  the tiny 64K artifact and reports as diagnostic evidence; production remains
  the shared-target 32K map with e256 depth-two speculation.

### [x] 17. Confidence-Gated Rollback Capture

**Goal:** Avoid accepted-state writes on high-confidence speculative cycles
while preserving exact replay for every rejected draft.

**Result:** REJECTED

- Target-level measurements show that Mamba accepted-state capture adds about
  1.1-1.4 ms per verifier cycle, so a margin-gated implementation was tested.
  It always retained exact target verification; uncaptured misses restored the
  pre-verification snapshot and replayed the accepted prefix.
- First/second boundaries 1.0/4.0 reduced production-trace captures from 186
  to 99. Two exact controls reached `46.443` and `44.812 tok/s`, averaging
  `45.628 tok/s`, below the established `45.751 tok/s` full-capture mean. Log
  SHA-256 values are
  `ee4082048dd072abb3f5317b7120001075d7e1e353436bdd962d4620326bb640`
  and `51a046cc977e303c03d0f4f0bb934febad40a9cfc372d2e7385ba3193a26ed6f`.
- The same 1.0/4.0 policy failed to generalize to an independent C++ prompt:
  six misses between first-margin 1.0 and 2.0 caused 247.56 ms of replay and
  reduced throughput to `37.735 tok/s`. Raising the safe first boundary to 2.0
  removed that replay class but reached only `38.902 tok/s` versus the prior
  `38.812 tok/s` full-capture control. The corrected production trace reached
  `45.809 tok/s`, just 0.13% above the existing mean. Corrected independent and
  production log SHA-256 values are
  `8209c0b567a0d0495f3a3135b8af268bd1c7ce60c27c3dc8dfb900e2647935aa`
  and `a2f2430cf72098e08974876965705cf04c28c991a3d75809e51693c2f45ce88b`.
- A more aggressive 1.0/2.0 policy reached `46.349 tok/s` but exposed a 49 ms
  medium-confidence replay and does not provide a conservative boundary.
- The safe gain is below run-to-run variance and the policy adds two tuning
  parameters. The implementation and tests were removed; full accepted-state
  capture remains production.

### [x] 18. MoE And Recursive-Draft Hotpath Tuning

**Goal:** Reduce the dominant MoE and speculative-cycle costs without changing
target weights, selected experts, or authoritative target output.

**Result:** REJECTED; benchmark instrumentation retained

- Algebraic expert scale and router-score folds were rejected. The down-scale
  fold introduced up to `1.22e-4` drift and slowed a real layer; folding router
  scores into down inputs introduced up to `2.44e-4` drift and regressed some
  layer layouts by 6-16%.
- A specialized BF16 router preserved selected IDs and kept score drift below
  `6e-8`, but matched full verification was unchanged: `44.756` versus
  `44.547 ms` for two tokens and `55.895` versus `55.892 ms` for three. The
  specialized/generic verifier log SHA-256 values are
  `0f279f894949469cf37cd6f4de9468be7ebdc2217f8c7321d1f5adb76ade4677`
  and `f21075e0fe71851daa6b57c857b5d4a559ccc5aac223ac8fceafe5d3d61b577a`.
- A custom FP8 batch projection matched native MLX at one token but was about
  1.5x slower at two tokens on representative Mamba projections. It and all
  production hooks were removed.
- A pinned MLX source build screened selected-expert `gather_qmv` occupancy.
  The exact one-SIMD fast-up layout improved the isolated expert pair by 2.95%
  but was neutral or slower across complete real MoE layers. Four-SIMD and
  eight-row variants did not beat stock. For the `K=2688` expert-down fallback,
  wider per-thread work slowed by about 6.3%; larger output tiles were neutral.
- GPU-lazy recursive MTP required a 32K-by-4096 exact BF16 embedding subset
  because the production full embedding table is mmap-paged. The 0.25 GiB table
  reduced median MTP time from `2.874` to `2.588 ms`, but complete exact decode
  moved only from `44.956` to `45.102 tok/s` while peak memory rose from
  `53.704` to `53.954 GiB`. The implementation was removed. Lazy/serial log
  SHA-256 values are
  `ee42b773d7cfd71a06c363b24769104fdd269ac696fe7b3e50b79a977b75abda`
  and `e699f9c1b89a7d6000eab68d78d3b56d94e555ceb74568305e5777b49a9176c4`.
- The complete source-build and numerical screen is recorded at
  `quality/remove400-speculation/moe-hotpath-native-kernel-screen.txt`, SHA-256
  `ddd0dd54dff2bd171717af9940c053fc7d0ae45f30ef96711738915fbc6aa526`.
  Multi-token FP8 and MoE benchmarks remain to support future architectural
  work; production kernels, equations, paging, and serial MTP recursion remain
  unchanged.

### [x] 19. Confidence-Gated Third MTP Draft

**Goal:** Determine whether the e256 sidecar has a profitable high-confidence
third-draft region after accounting for recursive MTP, four-position target
verification, rollback, and resident memory.

**Result:** REJECTED; exact diagnostic path retained

- The runtime now supports an opt-in third recursive draft with separate
  second-margin attempt and third-margin emission gates. `--cycle-trace` writes
  every margin, target outcome, timing, and policy decision atomically only
  after exact output identity has passed.
- A 256-token observation run computed third proposals without emitting them.
  Overall conditional accuracy was only 23/51, but margins at least 2.0 were
  correct 12/15. This established a real confidence region while avoiding a
  guessed policy. Trace SHA-256 is
  `e031fdc4816c06e5fe6f0bc70ec15a5ba51c20f0bb2823d3ac534a974ce49bbe`.
- With a 2.0 attempt gate and 2.0 emission gate, the exact 512-token run
  accepted 31/36 emitted third drafts and reached `45.300 tok/s`. Lowering the
  emission gate to 1.5 accepted 37/45 and reached `45.317 tok/s`. Both remained
  below the matched depth-two controls at `45.515` and `45.986 tok/s`.
- Both policies peaked at `53.714 GiB`, effectively the same as depth two, and
  preserved exact ordinary-greedy token IDs. Log SHA-256 values are
  `a869b6df55efbf0a1480f07659118dcab5e2d8e8dbb6995866d7e98ee8e5b4bd`
  and `8fb5de2152f4d8a257d4066ba49d2e4a6abd41808c75fb019ca78ffa36d3efb6`.
  Corresponding trace SHA-256 values are
  `f2dbec3b49f5ecf5a518b1da51082b4958a4ec9fff881a64563d377cc3297150`
  and `6258885882dea7d93c21c638ed9f46c9295063b3c8431e1ee1c0198124b1ecce`.
- Decision: keep depth two as production. Retain depth three and cycle tracing
  solely for future sidecars or materially different workloads; do not spend
  more runs tuning the current e256 sidecar inside measurement noise.

### [x] 20. Recursive-Depth MTP Expert Planning

**Goal:** Reallocate the fixed e256 sidecar budget using route evidence from
recursive positions without increasing payload or target memory.

**Result:** REJECTED; planning tooling retained

- The full official BF16 MTP now exposes the same stateless draft-step boundary
  as packed sidecars. Recursive replay records expert counts and score mass for
  each depth, and `nemotron_mlx_mtp_depth_plan.py` combines normalized depth
  evidence into deterministic fixed-budget plans.
- Full-512 replay reached 208/135/53 matches at depths one/two/three. The
  current e256 BF16 plan reached 206/122/51. Equal-depth weighting retained the
  same 206 first-depth matches while improving later depths to 126/53; a 1/2/3
  weighting reached 206/127/54.
- On an independent 256-cycle trace, equal weighting improved total accepted
  drafts 256→259, while 1/2/3 weighting reached 257. This selected the more
  conservative equal-depth plan for the only materialization.
- NVFP4 erased the gain: the candidate matched production at 186/110 for the
  first two depths and regressed the third from 48 to 47. No resident run was
  justified. The 2.8536 GiB BF16 intermediate and 0.8040 GiB NVFP4 candidate
  were deleted; disk use returned to baseline.
- Full-route, equal-depth-plan, BF16-ranking, BF16-independent, and NVFP4 report
  SHA-256 values are
  `431fbacd0cb4d7baef6fd12bfcb69f65207479dc10d4656bd650f134dce27a90`,
  `d05496e9c35baf017188cf93c0199a544c6b043ae9315d04fd631e0c90b3eac3`,
  `ba58a2cdc2bd889b04abeca250882cd2bf1e5c322ead123fb65aaa7b62fd0fe9`,
  `deca96b996bacbaeea7c8af363f693bcaaf88de327b455fbafa69b25b6c2e3af`,
  and `a0c51939ca7606380fa0be31550d8324b7932f96d34088cfddbb20ec011a38df`.
- Decision: retain the full-reference route capture and planner for future MTP
  formats, but keep the original e256 plan and production sidecar.

### [x] 21. Selective BF16 MTP Projections

**Goal:** Recover draft acceptance lost by uniform NVFP4 while spending only
the measured resident-memory headroom on sensitive fixed projections.

**Result:** REJECTED; exact mixed-format tooling retained

- The MTP quantizer now accepts repeatable exact tensor names for BF16
  retention. Its report records the retained names and bytes, and the runtime
  rejects config/payload disagreement before constructing a mixed graph.
- The 2.853555 GiB e256 BF16 source was reconstructed from the immutable source.
  `eh_proj`, attention, latent projections, and shared experts were screened as
  groups, then every attention and latent projection was isolated.
- On the ranking trace, `o_proj` was best for production depth two: matches rose
  from 186/110 to 192/112 for about 23 MiB additional payload. Median draft time
  rose from roughly 1.30 ms to 1.38 ms. `fc1_latent_proj` improved only the
  unused third depth; the other isolated projections were neutral or harmful.
- The apparent `o_proj` gain did not generalize. The independent trace changed
  from control 161/83/30 to 161/80/32. Full BF16 attention was worse at
  159/79/31 and raised median draft time from 1.275-1.289 ms to
  1.523-1.531 ms.
- Matched coding/independent control report SHA-256 values are
  `c6d8eb4353c2bf8a9f2566d203d020d700427834f4eb3304be19b08b512c2d69`
  and `783a69e3755d171627984468fc653bcfb5a24899333561f15f5b48b9f80d18e5`.
  Independent `o_proj` and full-attention report SHA-256 values are
  `3d43c7b81e6e24cd31f4fd8e3e3d7f31b886311da11a309a4de01dd64585cc77`
  and `35d343e208899d6ef4c86cf609ba94ba59688c12e6a2330c3f84fd18e98c191f`.
- All generated mixed sidecars and the reproducible BF16 intermediate were
  deleted. Keep the original uniform-NVFP4 e256 sidecar; no resident benchmark
  was justified.

### [x] 22. Resident Graph Reuse And Metal Trace

**Goal:** Remove target-verifier CPU/command overhead after isolated kernel
tuning stopped producing end-to-end gains.

**Result:** SUCCESS for Mamba; whole-MoE compilation REJECTED

- A sleep-delimited Xcode Metal System Trace isolated five block-two forwards.
  Each steady call kept the GPU active for more than 99.5% of its 40.72-41.18
  ms span. Command-buffer merging therefore has little GPU-side ceiling, while
  roughly 4.6-5.9 ms per call remained in Python graph construction/setup.
- `CompiledMambaRunner` captures immutable layer state and exposes convolution
  and SSM arrays as explicit inputs/outputs. It falls back to eager execution
  for uninitialized, padded, or multi-prefix-capture states. Compiled signatures
  are cached separately by token count and capture position.
- A real Mamba layer was bit-exact and improved from 0.547 to 0.507 ms. The full
  verifier produced exactly equal eager/compiled full-vocabulary logits and
  every recurrent array. Existing rollback and captured-prefix checks remained
  within `1.526e-5`.
- Matched ten-repeat block-two medians improved 45.495 to 44.368 ms. Paired
  256-token resident generation improved 45.167 to 46.123 tok/s (2.12%),
  ordinary decode improved 25.356 to 26.026 tok/s, verifier median improved
  57.163 to 56.023 ms, and integrity remained exact at 53.689 GiB peak.
- Compiled/eager log SHA-256 values are
  `7068fffd6787e1069a5e513688068fba6575284bcf30dfcac054b4f662b69dd5`
  and `2cbc55ea2c49c3349c09cbc151cb27a7feb7769b7a064fdd34c6d3d0d507c7d3`.
  The trace-window stdout SHA-256 is
  `0aa0ef57557e519e87623052c7e0d95c89217e3d1b13d2b24f06cbba55745a0d`.
- Whole-MoE compilation improved an isolated layer by about 3% but exhausted
  Metal command-buffer memory when all 40 resident MoE layers compiled. It was
  removed. Generation CLIs now default to compiled Mamba and retain
  `--no-compile-mamba` for controls.

### [x] 23. Routed-Expert Top-K Sensitivity

**Goal:** Determine whether evaluating fewer than 22 routed experts per token
can produce a large target/verifier gain without material quality drift.

**Result:** REJECTED AS A LARGE-GAIN PATH; diagnostic tooling retained

- `nemotron_mlx_topk_sweep.py` performs low-memory layer-streamed comparisons
  without modifying packed weights. Reports bind the source revision, packed
  plan/report, model config/index, corpus, and tool hashes. Production
  generation remains native top-22; only the profiler exposes an experimental
  override.
- The initial eight-category final-logit sweep measured top-20/18/16/14/12 at
  mean KL `0.00443/0.01111/0.02150/0.03912/0.07467`. Top-18 changed the
  multilingual winner. The other counts happened to retain 8/8 winners, but
  the monotonic distribution drift rejects treating that as quality parity.
- Top-20 then retained 16/16 final winners over the longer sensitivity and
  disjoint validation corpora, with mean KL `0.00765` and `0.00552` and worst
  KL `0.01860` and `0.01341`. This makes top-20 the only plausible follow-up,
  not an accepted runtime policy.
- Matched block-two medians were 45.288 ms at top-22, then 43.666/42.510/
  41.527/39.475 ms at top-20/18/16/12. Top-20 gained only 3.58%; even the
  highly drifting top-12 gained only 12.84%. The expected speculative gain for
  top-20 is about 2%, too small to justify downstream coding gates or target
  quality loss.
- Longer top-20 report SHA-256 values are
  `2683a6c5a7eab35c178ce819af02a0754b0d81253c968fcef764ccbea348dd95`
  and `a0abf12b6bcf6699548b2f40c509d18114e07cc6aff9af1301f534daf55f5bfb`.
  Top-22/20/18/16/12 resident log SHA-256 values are
  `4ce07ceb604af71895c1fd7f7a88c86f98568a3ca9198aa8101e76e7bb522520`,
  `f654ae0957453566505b5349688a402f797452819aa4befd40b3902ba478649a`,
  `7aa2394ddf15c72ee48da2b3f64613f1faf500146bff0b906b8347b647c29dbf`,
  `a33bf92422ce0ff574d6e2db972cafb576c373f9b969f5ea5d5fe41eabaaf819`,
  and `364da9ef880ded6cb04323de6efe56ea51f3e2b98a0c05ac1cbeb20298832859`.
- Decision: do not pursue static or adaptive top-k as the main acceleration
  project. A per-layer policy can only recover a fraction of top-12's 12.84%
  ceiling. Move the large-gain effort to separately trained multi-depth drafts,
  where exact target verification preserves output quality.

## Combined Candidates

Do not create combined candidates until their individual components have
completed results. Record combinations here when justified:

- [x] Runtime combination: paged embeddings plus reduced-vocabulary MTP plus
  adaptive multi-token/n-gram speculation.
  - **Result:** SUCCESS for paged embeddings and reduced-vocabulary MTP;
    recursive and n-gram extensions remain exact opt-in accelerators.
- [ ] Compression combination: nonuniform proxy experts plus layerwise
  distillation.
  - **Result:** BLOCKED ON ITEMS 6-8
- [x] Aggressive candidate: approximately 25% physical expert reduction,
  paged embeddings, and an accepted compact target-head strategy.
  - **Result:** SUCCESS at `54.4974 GiB`; matched hidden LiveCodeBench is a
    documented mixed trade and candidate-bound exact MTP reaches 36.538 tok/s.

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

# Ornith Runtime Notes

This branch is the integration branch for the Ornith local-runtime work.
Feature work should branch from `ornith-main`, then merge back here after it is
tested for correctness, numerical integrity, and performance.

DS4 remains an upstream-syncable reference. Do not make Ornith depend on DS4
source files for model-specific runtime logic. If DS4 code is useful, copy the
needed piece into `ornith_*` files and adapt it there.

Performance changes are welcome only with checks that show output integrity and
numerical accuracy are preserved.

## Current Tools

Run all current Ornith metadata checks:

```sh
./ornith/check.sh
```

Run repeatable Metal decode cache benchmarks:

```sh
ornith/tools/bench_metal_decode.sh
```

The benchmark writes `ornith-metal-bench.log` by default and compares selected
expert cache budgets for fixed raw-token prompts. Override `MAX_NEW_LIST`,
`PROMPT`, `OUT`, `CATALOG`, or `SHARDS` as needed.

`ornith/tools/ornith_memory_plan.py` estimates compression targets from local
metadata:

```sh
python3 ornith/tools/ornith_memory_plan.py \
  --config /path/to/config.json \
  --index /path/to/model.safetensors.index.json
```

If the safetensors shards are available locally, add `--safetensors-dir DIR`.
The tool reads only safetensors headers to bucket tensor bytes; it does not load
full model tensors into memory.

`ornith/tools/ornith_prompt.py` is the executable reference for text-only chat
rendering and tokenizer special-token inspection:

```sh
python3 ornith/tools/ornith_prompt.py \
  --tokenizer /Users/nir/dev/models/Ornith-1.0-397B/tokenizer.json \
  --print-specials
```

`ornith/tools/ornith_decode_tokens.py` is a no-dependency tokenizer helper for
local smoke tests. It decodes token IDs and can encode simple text/special-token
prompts through the tokenizer JSON byte-BPE vocabulary:

```sh
python3 ornith/tools/ornith_decode_tokens.py \
  --tokenizer /Users/nir/dev/models/Ornith-1.0-397B/tokenizer.json \
  --encode '2+2='
```

It intentionally avoids a runtime dependency on Hugging Face `tokenizers`.
The model tokenizer has a regex pre-tokenizer, so this helper is exact enough
for current simple smoke prompts and special-token boundaries, but it should
not be treated as the final prompt encoder.

`ornith/tools/ornith_layout_check.py` validates the text tensor names expected
by the Ornith runtime against a local safetensors index:

```sh
python3 ornith/tools/ornith_layout_check.py \
  --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

`ornith/tools/ornith_storage_manifest.py` creates a small shard manifest for
future network-storage downloads without downloading any weights:

```sh
python3 ornith/tools/ornith_storage_manifest.py \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

Add `--text-only` to exclude vision tensors from the manifest.
Add `--safetensors-dir DIR` to read local shard headers and report selected
bytes without loading tensor data.

`ornith/tools/ornith_shard_scope_report.py` classifies shards as language,
visual, other, or mixed from the local index:

```sh
python3 ornith/tools/ornith_shard_scope_report.py \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

`ornith/tools/ornith_layer_catalog.py` builds a checked per-layer map of text
tensors to shards:

```sh
python3 ornith/tools/ornith_layer_catalog.py \
  --config /Users/nir/dev/models/Ornith-1.0-397B/config.json \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json
```

`ornith/tools/ornith_iq1.py` is a synthetic reference harness for 1-bit and
ternary expert quantization, weighted block scaling, and packed dot-product
checks. It does not read model weights.

```sh
python3 ornith/tools/ornith_iq1.py
```

`ornith/tools/ornith_safetensors_filter.py` copies selected tensors from a
safetensors shard into a new shard. Use `--text-only` to exclude vision tensors.

`ornith/tools/ornith_text_repack_plan.py` reports which shards can be copied,
filtered, or skipped for a text-only repack. It reads only the local index.
Add `--dry-run --src-dir SRC --dst-dir DST --allowlist-dir DIR` to print the
future copy/filter actions without touching weights.
Add `--execute` to run those actions against already-local shard files.

`ornith/tools/ornith_text_tensor_allowlist.py` writes the text tensor names for
one shard, suitable for `ornith_safetensors_filter.py --allowlist`.

`ornith/tools/ornith_stream_state.py` tracks resumable shard streaming:
pending, downloading, downloaded, processing, done, failed, output size, raw
deletion, and sha256 verification. It supports one standby downloaded shard
while another shard processes.

`ornith/tools/ornith_download_shard.py` downloads one shard to `*.part` with
human-readable byte progress logs. `ornith/tools/ornith_process_shard.py`
copies, filters, or quantizes one local shard with progress logs. Add
`--benchmark-only` to process once, report throughput, and delete the benchmark
output.

`ornith/tools/ornith_stream_run.py` runs the one-shard processing loop with one
background prefetch shard, state updates, output verification, and raw deletion.
Use `--max-shards N` for bounded smoke tests. Add `--processor quantize` to
write compact `.ornq` outputs, validate them against the still-local raw shard,
then delete raw after state verification unless `--keep-raw` is set.
Use `--download-method hf` for Hugging Face CLI/Xet downloads instead of the
stdlib fallback downloader.

`ornith/run_quant_stream.sh` starts the standard quantized streaming job and
tees live stdout to `quant-full/stdout.log` while `ornith_stream_run.py` keeps
writing its structured log to `quant-full/run.log`:

```sh
ornith/run_quant_stream.sh --max-shards 2
```

Omit `--max-shards` only after approving the full weight-download run.
Quantized outputs go to local disk at `quant-full/out` by default, or
`LOCAL_OUT_DIR=/local/path` if overridden. Downloaded source shards are
temporary and the launcher rejects `--keep-raw`; raw files are deleted after
`.ornq` validation and state verification.

Resume behavior is state-file driven. Re-run the same command after an
interrupted job:

```sh
ornith/run_quant_stream.sh
```

Completed shards stay `done` and are not repeated. A caught download/process
failure is marked `failed` and retried. A hard stop during `processing` retries
from the already-downloaded raw shard if it still exists. A hard stop during
`downloading`, or processing without a raw shard, retries the download for that
shard. Raw shards are deleted only after `.ornq` validation and state
verification. If interruption leaves a raw shard behind, rerun uses or replaces
it according to the state file.

`ornith/tools/ornith_quantize_safetensors.py` writes the experimental `.ornq`
smoke quantization format. Vision tensors are skipped. Routed expert tensors
use IQ1 blocks, small/sensitive tensors are copied as BF16, and remaining BF16
matrix tensors use symmetric Q4 blocks. The C helper uses pthread workers,
chunked I/O, and fixed output offsets.
Add `--policy ornith/policies/ornith-routed-last6-q4.policy.json` or another
JSON policy to override tensor quantization by exact name, substring, regex,
and optional layer range.

`ornith/tools/ornith_quant_policy_report.py` applies a policy to the local
runtime catalog without reading raw weights:

```sh
python3 ornith/tools/ornith_quant_policy_report.py \
  --policy ornith/policies/ornith-routed-last6-q4.policy.json
```

Current checked policies:

- `ornith-routed-q4.policy.json`: routed experts at Q4. Projected from the
  local catalog at `187.45 GiB`, too large for 64 GB but useful as a quality
  ceiling.
- `ornith-routed-last6-q4.policy.json`: current IQ1 routed experts except
  layers 54-59 at Q4. Projected at `65.95 GiB`, `+13.50 GiB` over the current
  `.ornq` set.

## REAP Path

Upstream REAP source notes live at
`/Users/nir/dev/models/Ornith-1.0-397B/source-notes/reap`. This is code only,
not model weights. REAP's useful path for Ornith is pruning, not expert
merging: collect per-layer expert saliency, remove low-saliency experts, patch
router rows/config, then quantize the reduced routed tensors.

Important upstream behavior:

- REAP saliency is the mean of `router_weight * expert_output_l2_norm` over
  tokens where an expert is selected.
- Router weights should be renormalized over selected top-k experts; upstream
  enabled this by default after its March 2026 fix.
- The memory-efficient observer replays one layer/block at a time and computes
  all expert outputs for that block, so calibration is expensive but bounded.
- Upstream safety rail: high max-activation "super/outlier" experts are
  protected by setting their saliency to infinity before pruning.
- Pruning is per-layer: for each MoE layer, keep retained expert modules and
  remove the matching router rows.

Local planning starts with `ornith/tools/ornith_reap_plan.py`. It does not
touch weights. It consumes future REAP-style observer JSON, prunes lowest
saliency per layer, and applies visible safety limits:

```sh
python3 ornith/tools/ornith_reap_plan.py \
  --observer observations.json \
  --compression-ratio 0.5 \
  --min-retained 16 \
  --out reap-plan.json
```

Default guards preserve high max-activation super experts from the first 75% of
layers, preserve the top 2% by frequency and REAP score, preserve unobserved
experts by default, and never prune below `--min-retained`. Use
`--allow-prune-unobserved` only for explicit experiments where the observer is
known to have broad enough coverage. This matters because the current fast
observer records selected experts; unselected experts have no REAP evidence,
not necessarily low importance. Plan manifests include `observed_count`,
`unobserved_count`, `observed_fraction`, and `candidate_count` per layer; check
these before treating any REAP plan as quality-safe.

`ornith/ornith_reap_observe.c` is the first native observer. It runs the normal
decode path through `ornith_generate_greedy_limited_with_decode_hooks`, replaces
only the MoE hook, emits REAP-style JSON, and preserves the generated hidden
state by reproducing the same MoE output. Example:

```sh
cc -O2 -std=c11 -Iornith ornith/ornith.c ornith/ornith_reap_observe.c \
  -lm -o /tmp/ornith_reap_observe
/tmp/ornith_reap_observe \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0,1 1 1 1 32 /tmp/ornith-reap-observe.json
```

The current observer records selected experts only: frequency, selected router
weight sum, expert-output norm mean, REAP score, and max activation. This is
enough to exercise the end-to-end observe-to-plan path and matches the experts
that can actually receive REAP score under top-k routing. Upstream's exhaustive
layerwise observer also evaluates every expert output for each block; add that
only when we need a slower full calibration pass. Until then, keep the planner's
default unobserved-expert preservation enabled for quality-sensitive plans.

`ornith/tools/ornith_reap_calibrate.py` runs the native observer over a prompt
file and writes one observer report:

```sh
python3 ornith/tools/ornith_reap_calibrate.py \
  --catalog /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  --shards /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  --prompts prompts.tokenids.txt \
  --out observations.json \
  --max-prompt-tokens 4 \
  --max-new 1 \
  --layers 1 \
  --expert-top-k 1 \
  --vocab-limit 32
```

Prompt files are line-based comma-separated token IDs. Add `--text-prompts
--tokenizer tokenizer.json` for the local fallback tokenizer encoder. The
calibrator now uses the observer's native `--prompts` mode, so the model is
loaded and mapped once while all prompt lines are observed. The observer tracks
expert-array lengths per layer, which keeps it compatible with future
REAP-pruned layers that retain different expert counts. Use
`--max-prompt-tokens N` for bounded coverage probes; it applies after optional
text tokenization and before native observation.

`ornith/tools/ornith_reap_size_report.py` estimates storage after applying a
REAP keep/drop plan plus a quant policy:

```sh
python3 ornith/tools/ornith_reap_size_report.py \
  --plan reap-plan.json \
  --policy ornith/policies/ornith-routed-last6-q4.policy.json
```

It shrinks routed expert tensors and matching router rows for layers present in
the plan; unobserved layers stay unchanged. A two-prompt, one-layer smoke with
25% layer-0 pruning and `ornith-routed-last6-q4` projected `65.75 GiB`, only
`0.20 GiB` smaller than the same policy without REAP because only layer 0 was
planned. Real size projections require observing/planning all 60 layers.

A two-prompt all-layer native calibration smoke (`0,1` and `17,10,17`,
`max_new=1`, `top_k=10`) completed in about 86 seconds with one model mapping.
With `--min-retained 384` and default unobserved-expert preservation, the plan
pruned 1,630 of 30,720 experts, retained 476-494 experts per layer, and
projected `62.80 GiB` under `ornith-routed-last6-q4`. Running the same tiny
observation with unobserved pruning allowed would project `50.60 GiB`, but that
is not quality-safe evidence because most experts were never selected.

A bounded four-prompt coding-token calibration used four-token prefixes from
local tokenizer output and completed in about 3.5 minutes. It produced 20
events per layer, observed 5,780 of 30,720 expert slots (`18.8%` coverage),
pruned 4,602 experts with the same safe defaults, retained 403-474 experts per
layer, and projected `56.83 GiB` under `ornith-routed-last6-q4`. This is better
but still below the coverage needed for a quality-sensitive REAP cut.

A bounded twelve-prompt coding-token calibration, also with four-token
prefixes, produced 60 events per layer and observed 10,359 of 30,720 expert
slots (`33.7%` coverage). With the same safe defaults it pruned 7,490 experts,
retained 384-405 experts per layer, and projected `50.92 GiB` under
`ornith-routed-last6-q4`. This is the current best disk-light REAP probe, but a
real cut still needs broader calibration and downstream quality checks.

`ornith/tools/ornith_reap_merge_observations.py` merges multiple observer JSON
reports so calibration can be accumulated in small batches:

```sh
python3 ornith/tools/ornith_reap_merge_observations.py \
  obs-a.json obs-b.json \
  --out observations-merged.json
```

Merging the two-token all-layer smoke with the twelve-prompt/four-token coding
smoke produced 11,129 of 30,720 observed expert slots (`36.2%` coverage),
pruned 7,632 experts with safe defaults, retained 384-394 experts per layer,
and projected `50.68 GiB` under `ornith-routed-last6-q4`.

Adding one more bounded 24-prompt/four-token coding batch raised merged
coverage to 15,449 of 30,720 expert slots (`50.3%`). With `min_retained=384`,
the safe plan reached the full requested 25% prune in every layer: 7,680
experts pruned, 384 retained per layer, projected `50.60 GiB` under
`ornith-routed-last6-q4`.

The current stronger calibration set lives at
`/Users/nir/dev/models/Ornith-1.0-397B/reap-calibration-77pct`. It merges the
earlier smokes plus two bounded 24-prompt/six-token coding batches: 23,665 of
30,720 expert slots observed (`77.0%`). Plan stability versus the previous
72.7% set improved to average Jaccard `0.794`, with about 30 changed pruned
experts per layer. It includes `observations.json`, `summary.json`, and
25/30/35/40% prune plans. Current-quant projected sizes are
`40.48`/`38.14`/`35.71`/`33.37 GiB`.

`ornith/tools/ornith_reap_repack_ornq.py` materializes a REAP retention plan
against already-quantized `.ornq` shards:

```sh
python3 ornith/tools/ornith_reap_repack_ornq.py \
  --src-dir /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  --dst-dir /path/to/reap-ornq-out \
  --plan reap-plan.json \
  --max-shards 1 \
  --report reap-repack-report.json
```

It classifies tensors by Ornith names, copies unplanned tensors unchanged, and
for planned layers slices routed expert tensors plus matching router rows along
the leading expert dimension. The output headers record
`reap_retained_experts`. This is the practical local test path because it does
not need raw HF weights. The final higher-quality path should apply REAP to raw
weights first, then quantize the reduced tensors. Re-running skips destination
shards that already validate, and `--max-shards N` is available for bounded
smokes.

Earlier REAP-repacked artifact, now removed for disk recovery:

- directory: `/Users/nir/dev/models/Ornith-1.0-397B/reap-keep384-50pct`
- source: `/Users/nir/dev/models/Ornith-1.0-397B/quant-full/out`
- plan: `plan.json`
- output shards: `out`
- runtime catalogs: `catalog.json`, `catalog.tsv`
- repack report: `repack-report.json`
- result: 122 shards, 1038 tensors, 60 layers, `40.48 GiB` output versus
  `52.45 GiB` source, `11.97 GiB` saved, 180 tensors sliced
- status: deleted on 2026-07-04 to recover about 40G; recreate from
  `quant-full/out` and the recorded plan if needed

Validation smokes on the reduced set:

- runtime catalog validates exact text tensor coverage
- native loader validates 122 shards, 1038 tensors, 60 layers
- 4-layer decode smoke passes
- full 60-layer CPU capped-vocab generation: token `1`, 47.21s
- full 60-layer Metal capped-vocab generation: token `1`, 6.90s
- full 60-layer Metal full-vocab generation: token `198`, 11.48s
- old full quantized set generated the same first token for the capped and
  full-vocab Metal probes, so these smokes did not expose an immediate first
  token regression

Keep `quant-full/out` until the reduced set has enough quality validation or a
new source/output location is approved. It is still the fallback and the source
for repacking alternate REAP plans.

For the quality path, `ornith/tools/ornith_quantize_safetensors.py` now accepts
`--reap-plan PLAN.json` and applies REAP to raw BF16 safetensors before
quantization. Pruned routed expert tensors and matching router rows are sliced
along the leading expert dimension, output headers record
`reap_retained_experts`, and validation maps samples back to the original raw
source tensor. `ornith/run_quant_stream.sh` passes this through with
`REAP_PLAN=/path/to/plan.json`; quant policies pass through with
`QUANT_POLICY=/path/to/policy.json`.

Raw shard smoke: using preserved shard 2 and
`reap-calibration-77pct/plan-r0.25.json`, raw `gate_up_proj` was sliced from
512 to 384 experts before IQ1 quantization, output validated against the raw
source, and the temporary `.ornq` was deleted. This proves the overnight path
can derive reduced shards from raw weights rather than from degraded `.ornq`
files.

Historical overnight-quality candidate:

```sh
JOB_DIR=/Users/nir/dev/models/Ornith-1.0-397B/quant-reap35-last19-q4 \
LOCAL_OUT_DIR=/Users/nir/dev/models/Ornith-1.0-397B/quant-reap35-last19-q4/out \
REAP_PLAN=/Users/nir/dev/models/Ornith-1.0-397B/reap-calibration-77pct/plan-r0.35.json \
QUANT_POLICY=ornith/policies/ornith-reap35-routed-last19-q4.policy.json \
ornith/run_quant_stream.sh
```

Projected `63.52 GiB`: 35% REAP prune, 333 routed experts retained/layer, last
19 routed layers at Q4 and earlier routed layers at IQ1.

Completed full run, deleted on 2026-07-07 for disk recovery:
`/Users/nir/dev/models/Ornith-1.0-397B/quant-reap35-last19-q4` contained 122
`.ornq` shards, no remaining raw `.safetensors` in `raw/`, and
generated `catalog.json`/`catalog.tsv`. State file reports all 122 shards
`done`; the final log ends with shard 122 verified and `run-done
processed=121` because shard 1 was already completed by the preflight. Actual
catalog bytes are `22,816,456,704` IQ1, `45,381,617,780` Q4, and `1,018,112`
BF16 payload bytes, about `63.52 GiB` total.

Post-run validation: native catalog loader passes with 122 shards, 1038
tensors, 60 layers. 4-layer decode and CPU generation smokes agree on top token
`10` at score `1.53235245`. Metal generation now handles the mixed IQ1/Q4
routed-expert policy after adding a Q4 3D slice fallback; synthetic Metal tests
cover that path. Real REAP-tail smokes pass through 59 layers with
`expert_top_k=1`, including the Q4 routed layers, but the full 60-layer
Metal capped-vocab probe is still too slow with the generic Q4 routed fallback.
This was a runtime performance bottleneck, not evidence that the full
quantization failed.

Follow-up: the fused/staged routed-MoE Metal path now accepts Q4 expert
tensors as well as IQ1, fixing the full-tail timeout (`top_k=1`, 60 layers,
vocab 32 now completes in about 10s). That made quality probes possible:
non-REAP full quant returns token `19` (`4`) for raw `2+2=`, while
`quant-reap35-last19-q4` returns token `241784` (`Золо`). A lighter diagnostic
REAP repack from the existing full `.ornq` set,
`/Users/nir/dev/models/Ornith-1.0-397B/reap-r10-min448`, prunes 10%, retains
461 experts/layer, is 48G, and still returns `4` plus the normal continuation
`2+2=4`. Its short fizzbuzz probe still loops without code, matching the
older quality blocker. Current conclusion: 35% expert pruning is too
aggressive for this calibration/selection recipe; 10% pruning preserves the
basic arithmetic smoke but does not solve the underlying IQ1 coding-quality
problem. The `reap-r10-min448` directory was also deleted on 2026-07-04 to
recover about 48G; recreate it from `quant-full/out` and
`reap-calibration-77pct/plan-r0.10-min448.json` if another diagnostic run is
needed.

Earlier writable candidate: `ornith/policies/ornith-reap10-routed-last6-q4.policy.json`
with `reap-calibration-77pct/plan-r0.10-min448.json`. It keeps the 10% REAP
cut that preserved the arithmetic smoke and raises the last 6 routed layers to
Q4, projecting about `59.83 GiB`. Preserved raw shard 2 smoke passed: layer-0
`gate_up_proj` was sliced to 461 experts, quantized as IQ1, and validated
against raw (`mse=6.3994e-07`, `max_abs=0.00500488`). This is the next full
overnight candidate that was produced with the then-current writer. DS4-style
candidate error on the same raw tensor said `q2_k` is much better than
`iq2_xxs` (`relative_l2` `0.297` versus `0.657`), which became the next
format-level option after this candidate still failed coding quality.

Full run result for that first REAP10 policy:
`/Users/nir/dev/models/Ornith-1.0-397B/quant-reap10-last6-q4` completed 122
shards, 122 done, 60G output, no raw safetensors left, and cataloged to
`59.83 GiB`. It failed the basic raw `2+2=` probe, returning token `85557`
(`aab`) instead of `19` (`4`). Root cause candidate: the policy default forced
150 small/sensitive tensors from BF16 to Q4 (`linear_attn.A_log`,
`linear_attn.dt_bias`, and `mlp.shared_expert_gate.weight`), unlike the
passing local REAP10 repack. The failed artifact was deleted on 2026-07-05 to
recover about 60G; logs and this note are the retained evidence.

Corrected rerun candidate:
`ornith/policies/ornith-reap10-last6-q4-sensitive-bf16.policy.json`. It keeps
the same 10% REAP and last-6 Q4 routed experts, but preserves norms,
`A_log`, `dt_bias`, and `shared_expert_gate` as BF16. Projected size remains
about `59.83 GiB`. Preserved raw shard 2 smoke passed with the same routed
tensor validation (`mse=6.3994e-07`, `max_abs=0.00500488`).

Full corrected run result:
`/Users/nir/dev/models/Ornith-1.0-397B/quant-reap10-sensitive-last6-q4`
completed with 122 state entries `done`, 122 `.ornq` shards, no partials, no
raw `.safetensors` left in `raw/`, and catalogs present. Catalog payload is
`59.83 GiB`: `38.74 GiB` IQ1, `21.09 GiB` Q4, plus BF16 sensitive tensors.
CPU/Metal golden smoke passed. Raw `2+2=` generated token `19` (`4`) and the
8-token continuation was coherent arithmetic text: `4，4+4=8，`. Preserved raw
shard 2 validation passed against `raw-cache/model-00002-of-00122.safetensors`
with 4096 IQ1 samples (`mse=4.00805e-07`, `max_abs=0.00958252`). The short
fizzbuzz coding probe still repeated the request instead of writing code, so
this candidate is file-valid and arithmetic-safe but not coding-quality-safe.

Current q2_k format result:
`.ornq` now supports `q2_k` write/read/validate/reference decode via the copied
DS4 quantizer. A preserved raw shard-2 full-tensor q2_k smoke ran in 12.41s
for 4.29B params, validated with 4096 samples
(`mse=1.05952e-07`, `max_abs=0.00323486`), and full-tensor error measured
relative L2 `0.297341`, `rmse` `0.000313371`, and `max_abs` `0.0197601`.
The native C q2_k tensor-value and slice-matvec path was probed on a nonzero
decoded value. The temporary 1.3G q2-smoke `.ornq` was deleted after
validation; tiny logs and permanent reports remain in `quant-error/`.
Full routed q2_k is not size-viable: 10% REAP plus last-6 Q4 projects to
`116.81 GiB`; routed-down-only q2_k plus last-6 Q4 projects to `78.83 GiB`.

Completed q2_k heavy candidate:
`ornith/policies/ornith-reap-routed-down-q2k-sensitive-bf16.policy.json` with
`/Users/nir/dev/models/Ornith-1.0-397B/reap-calibration-77pct/plan-r0.25.json`.
It keeps routed gate/up at IQ1, raises routed down-proj to q2_k, preserves
small/sensitive tensors as BF16, and skips the Q4 tail that made the output too
large. Size projections from the 77% calibration are `68.78 GiB` at 10% REAP,
`58.06 GiB` at 25% REAP, and `50.96 GiB` at 35% REAP. The full
`quant-reap25-down-q2k` run completed on 2026-07-07 with 122 state entries
`done`, 122 `.ornq` shards, no raw `.safetensors` or `.part` files left, and
catalog files present. Actual catalog payload is `58.058 GiB`: `25.67 GiB`
IQ1, `31.71 GiB` q2_k, `4.96 GiB` Q4, plus BF16 sensitive tensors. Native
loader passed, 1-layer and 4-layer CPU decode smokes passed, and a full
60-layer raw `2+2=` CPU probe with `expert_top_k=1`, `vocab_limit=32` returned
token `17` (`2`) in 38.47s. Treat this as execution-valid but not
quality-positive. q2_k routed down is CPU-only today, so the next blocker is a
Metal q2_k down-proj/slice path before meaningful `top_k=10` quality probes.

```sh
JOB_DIR=/Users/nir/dev/models/Ornith-1.0-397B/quant-reap40-last22-q4 \
LOCAL_OUT_DIR=/Users/nir/dev/models/Ornith-1.0-397B/quant-reap40-last22-q4/out \
REAP_PLAN=/Users/nir/dev/models/Ornith-1.0-397B/reap-calibration-77pct/plan-r0.40.json \
QUANT_POLICY=ornith/policies/ornith-reap40-routed-last22-q4.policy.json \
ornith/run_quant_stream.sh
```

Projected `63.15 GiB`: 40% REAP prune, 308 routed experts retained/layer, last
22 routed layers at Q4 and earlier routed layers at IQ1.

`ornith/tools/ornith_ds4_quant_candidate_error.py` measures DS4-style
candidate quantization formats directly from raw BF16 safetensors without
writing candidate shards. It copies the DS4 quantizer into Ornith-named
`ornith_ds4_*` files, quantizes block by block, immediately dequantizes, and
writes exact JSON/Markdown error reports. It currently tests `iq2_xxs`,
`q2_k`, and `q4_k`.

`ornith/tools/ornith_ornq_validate.py` validates `.ornq` headers and can sample
dequantized values against a source safetensors shard.

`ornith/tools/ornith_runtime.py` is the current reference runtime foundation:
it mmap-loads `.ornq` shards, validates tensor spans/sizes, classifies tensor
roles, reports quantized memory by mode/group, and provides CPU reference
dequant/matvec helpers. It is not the final inference runtime.

`ornith/tools/ornith_runtime_catalog.py` builds the compact tensor catalog used
as the bridge from quantized shards to native runtime work:

```sh
python3 ornith/tools/ornith_runtime_catalog.py \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  --index /Users/nir/dev/models/Ornith-1.0-397B/model.safetensors.index.json \
  --out /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.json \
  --native-out /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv
```

The current generated catalog is text-only, validates exact non-vision tensor
coverage against the safetensors index, and contains 122 shards, 1038 tensors,
and 60 layers. The TSV sidecar is consumed by `ornith.c` so the native runtime
does not need a JSON parser.

`ornith.h` and `ornith.c` are the first native runtime boundary. They load the
TSV catalog, validate `.ornq` shard magic/sizes, check tensor payload ranges,
mmap shards, and provide tensor lookup, layer-aware tensor lookup,
BF16/Q4/IQ1 scalar decode, reference matvec, 3D expert-slice matvec, RMSNorm,
top-k helpers, and a narrow MoE layer smoke path. The layer smoke path performs
input RMSNorm, router matvec/top-k, routed expert gate/up/down, and shared
expert contribution for correctness composition; it is not the final optimized
token loop. The native boundary also includes reference embedding lookup and
lm-head top-k scoring. Native checks validate MoE tensor shape compatibility
across all layers when the full local catalog is present. Current real-output
probe:

```sh
cc -O2 -std=c11 -I. ornith/ornith.c tests/ornith_native_catalog_loader_test.c \
  -o /tmp/ornith_native_catalog_loader_test
/tmp/ornith_native_catalog_loader_test \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out
```

`ornith/ornith_step_smoke.c` is a bounded native token-step probe:

```sh
cc -O2 -std=c11 -Iornith ornith/ornith.c ornith/ornith_step_smoke.c \
  -lm -o /tmp/ornith_step_smoke
/tmp/ornith_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0 1 1 5 32
```

Arguments are `TOKEN_ID LAYERS EXPERT_TOP_K OUT_TOP_K VOCAB_LIMIT`. The current
real one-layer capped smoke returns top rows from the first 32 lm-head rows:

```text
0  12  2.24126315
1   9  1.6809659
2  10  1.45022964
3   5  1.41391909
4   1  1.39811707
```

Add `REPEATS` as the final argument to measure repeated smoke steps. Current
fast native kernels decode BF16/Q4/IQ1 directly from mapped payloads while
keeping scalar decode as the test reference:

```sh
/tmp/ornith_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0 1 1 5 32 20
```

Current CPU wall-clock sample: 5 one-layer capped steps with `EXPERT_TOP_K=10`
and `VOCAB_LIMIT=32` in 0.771488 seconds.

Add `decode` after `REPEATS` to run the newer decode smoke path. It validates
attention tensor layout and uses decoder ordering (`input_layernorm` reserved
for attention, `post_attention_layernorm` before MoE). Full-attention layers
implement the first-token causal shortcut through `v_proj` and `o_proj`;
linear-attention layers implement the exact zero-prior-state first-token
Gated DeltaNet path through q/k/v conv, q/k L2 norm, headwise beta gate,
per-value-head gated RMSNorm, and output projection.
`ornith_decode_sequence_smoke_limited` extends that CPU reference to short
token sequences by keeping per-linear-layer conv and SSM state plus
per-full-attention-layer KV state. The full-attention CPU reference applies
q/k RMSNorm, text-only partial RoPE, causal softmax over cached keys/values,
per-head q/gate unpacking, optional q-gate, and output projection.

`ornith/ornith_generate.c` is the current greedy-generation CLI over the native
CPU reference path. Build with `ORNITH_WITH_METAL` to use the narrow Metal
hybrid path for attention projection/output matvecs, linear-attention GDN
recurrence, post-attention MoE, and lm-head scoring. Recurrent conv/KV/SSM
state orchestration, RoPE/softmax, residuals, and CPU fallback logic still live
on the CPU side:

```sh
cc -O2 -std=c11 -Iornith ornith/ornith.c ornith/ornith_generate.c \
  -lm -o /tmp/ornith_generate
clang -DORNITH_WITH_METAL -O2 -std=c11 -Iornith \
  ornith/ornith.c ornith/ornith_metal.m ornith/ornith_generate.c \
  -framework Foundation -framework Metal -lm -o /tmp/ornith_generate_metal
PROMPT=$(python3 ornith/tools/ornith_decode_tokens.py \
  --tokenizer /Users/nir/dev/models/Ornith-1.0-397B/tokenizer.json \
  --encode '2+2=')
/tmp/ornith_generate \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  "$PROMPT" 1 60 10 0
/tmp/ornith_generate_metal \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  "$PROMPT" 1 60 10 0 metal
```

Arguments are `PROMPT_TOKEN_IDS MAX_NEW LAYERS EXPERT_TOP_K VOCAB_LIMIT`.
The same binary also has a persistent worker mode for frontends that need more
than one request without remapping the model each turn:

```sh
printf '1\t0,1\nquit\n' | /tmp/ornith_generate_metal --worker \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  4 1 32 metal
```

Worker requests are `MAX_NEW<TAB>PROMPT_TOKEN_IDS`; an empty line in the output
separates responses, and `quit` exits. The worker keeps a native
`ornith_session` with linear-attention SSM/conv state and full-attention KV
state. It reuses that state only when the next request's prompt is an exact
extension of the stepped token history; otherwise it resets safely. Worker
headers report `session=reset` or `session=reuse`, the reused prefix length,
current session token count, and session capacity.

`ornith/tools/ornith_chat.py` is the first non-smoke text CLI. It wraps the
tokenizer, chat renderer, and native generator. If the optional Hugging Face
`tokenizers` package is installed, it uses that exact tokenizer; otherwise it
falls back to the local byte-BPE helper. One-shot calls launch the native
generator once. Interactive mode keeps a native worker process alive so the
catalog, quantized shard mmaps, and exact-prefix native session state are
reused across turns. The Python history stores the assistant prefill scaffold
plus decoded completion so the next rendered prompt matches the token state
the model actually saw.

```sh
python3 ornith/tools/ornith_chat.py \
  --max-new 64 \
  --nothink \
  "Write a tiny C function that adds two ints."
python3 ornith/tools/ornith_chat.py --prompt-file prompt.txt --max-new 64 --nothink
python3 ornith/tools/ornith_chat.py --interactive --max-new 128 --nothink
python3 ornith/tools/ornith_chat.py --interactive --messages chat.json --save-messages chat.json
python3 ornith/tools/ornith_chat.py --max-new 256 --temperature 0.6 --top-p 0.95 "Write fizzbuzz in C."
```

Use `--prompt-file -` to read a one-shot prompt from stdin. In interactive
mode, `--prompt-file FILE` can seed the first user message, but stdin remains
reserved for the interactive turn loop.

Use `VOCAB_LIMIT=0` for the full lm-head. Set `ORNITH_METAL_ATTN_MATVEC=0`,
`ORNITH_METAL_BATCH_MATVEC=0`, `ORNITH_METAL_GDN=0`,
`ORNITH_METAL_LINEAR_ATTN=0`, `ORNITH_METAL_SELF_ATTN=0`, or
`ORNITH_METAL_ROUTER=0` to disable those Metal decode hooks for A/B checks.
Set `ORNITH_METAL_Q4_ROW8=0` to restore the older row-4 Q4 block-256
projection matvec path. Leave it unset, or set it to `1`, for the current
row-8 default. Set `ORNITH_METAL_PROFILE=1` to print real-generation Metal
timing buckets to stderr; it is useful for proportions but adds timing
overhead.
The routed profile splits `routed_fused` into `routed_stage` for CPU selected
expert-slice staging and `routed_kernel` for the routed Metal command/wait.
The profile also prints `ornith_metal_linear_profile`, which intentionally
splits the linear-attention command into separate synchronized QKV/Z/B/A
projection, conv, GDN, and out-projection commands. Use that second line for
attribution, not absolute throughput. On a 16-token, 60-layer, top_k=10
raw-token `0,1` sample, linear attention was projection-bound: with the extra
per-projection sync, QKV was about 1.37 s, Z about 0.20 s, B/A about 0.15 s
each, conv about 0.12 s, GDN recurrence about 0.16 s, and out-proj about
0.20 s. B/A are small matrices, so their split timing mostly exposes command
overhead rather than arithmetic.
`ornith_metal_route_profile` reports consecutive per-layer routed-expert reuse
during profiled generation. `ORNITH_METAL_SELECTED_EXPERT_CACHE_MB` defaults to
`512` and caches compact selected routed-expert IQ1 slices per layer. Set it to
`0` to restore direct per-token selected-slice copies, or raise it for longer
generations. With 16 slots per layer, `512` MiB covers roughly the first 19
layers and `2048` MiB covers all 60 layers. Raw-token `0,1`, 60-layer,
top_k=10, full-vocab max_new=64 improved from `9.944324` seconds with the
cache disabled to `9.464343` seconds at the default 512 MiB and `8.734374`
seconds at 2048 MiB; token IDs and scores were unchanged.
`ORNITH_METAL_SELECTED_EXPERT_CACHE_SLOTS` defaults to `16`; `32` slots did
not help that sample.
The repeatable benchmark harness later showed the tradeoff more clearly:
max_new=16 was neutral (`3.941200` off, `3.945488` at 512 MiB, `3.962769` at
2048 MiB), max_new=64 was noisy (`9.272274` off, `9.747852` at 512 MiB,
`8.852295` at 2048 MiB), and max_new=128 strongly favored the cache
(`26.361671` off, `18.804899` at 512 MiB, `15.386105` at 2048 MiB).
A second run adding 1024 MiB kept 2048 MiB as the best high-memory setting:
max_new=64 was `9.271396`, `12.929347`, `8.974977`, and `8.779632` seconds for
0/512/1024/2048 MiB; max_new=128 was `28.012862`, `16.252663`, `24.914015`,
and `15.596003` seconds. The current policy stays conservative: 512 MiB is the
default, 2048 MiB is the useful long-decode/high-memory mode, and 1024 MiB did
not justify a special policy.
The router default is the specialized block-256 Q4 Metal router.
`ORNITH_METAL_ROUTER=serial` restores the old serial Metal accumulation path,
`ORNITH_METAL_ROUTER=parallel` uses the generic parallel Metal matvec, and
`ORNITH_METAL_ROUTER=0` uses the CPU router. The default can also be selected
explicitly with `ORNITH_METAL_ROUTER=specialized` or `q4`.
The Metal GDN hook fuses linear-attention GDN with `linear_attn.out_proj.weight`
when the out projection is the Ornith Q4/block-256 layout; unsupported layouts
fall back to the older GDN-then-matvec path.
The Metal linear-attention hook fuses QKV/Z/A/B projections, depthwise
conv+SiLU, GDN recurrence, and out-proj into one command buffer when
predecoded linear constants are available. One-shot generation keeps per-layer
conv/SSM recurrence state and linear-attention constants in resident Metal
shared buffers for the generation call. Public session calls copy recurrence
state back so later session calls remain correct.
The Metal self-attention hook covers Ornith's periodic full-attention layers
for decode sessions with `token_cap <= 256`: it fuses q/k/v projections, q/k
norm, RoPE, resident KV append, causal softmax/value mix, gate, and out-proj.
Longer contexts fall back before the hook takes ownership of KV state.
`ORNITH_METAL_LINEAR_RESIDENT_MB` is opt-in for keeping linear-attention Q4
projection weights resident. A 3072 MiB budget helped capped top_k=4 samples
but was not the best default for warm top_k=10 full-vocab generation, so leave
it unset unless that specific workload benefits.
Set `ORNITH_METAL_RESIDENT_LAYER_MB=1024` to keep the first full routed-expert
layer that fits in resident Metal shared buffers. This is opt-in because it
spends about 1 GiB; use `0` or leave it unset to keep the compact per-token
selected-slice staging path.
`ORNITH_METAL_SHARED_RESIDENT_MB` defaults to `512`, keeping shared-expert Q4
matrices in Metal shared buffers instead of copying them every layer/token. Set
it to `0` to keep the lower-memory copy path.
Current real-model smokes on the fully quantized 122-shard `.ornq` set:

```text
raw prompt "2+2=", max_new=1: token 19 -> "4" in 69.475699 s
raw prompt "2+2=", max_new=1, old Metal MoE/lm-head hybrid: token 19 -> "4" in ~35 s
raw prompt "2+2=", max_new=1, Metal attention matvec + GDN hooks: token 19 -> "4" in 8.951971 s
raw prompt "2+2=", max_new=1, batched Metal projection matvecs + fast BF16 RMSNorm
  + fast conv scalar decode: token 19 -> "4" in 3.843622 s
raw prompt "2+2=", max_new=2, fast conv scalar decode:
  tokens 19,11 in 4.221559 s
raw prompt "2+2=", max_new=4, fast attention scalar decode:
  tokens 19,11,19,10 in 4.713793 s
raw prompt "2+2=", max_new=8, conditional predecoded linear constants:
  tokens 19,11,19,10,17,28,19,11 in 6.117132 s
raw prompt "2+2=", max_new=1, serial Metal router:
  token 19 -> "4" in 3.178888 s
raw prompt "2+2=", max_new=8, serial Metal router, full vocab:
  tokens 19,198,17,10,17,28,19,198 in 4.909367 s
raw prompt "2+2=", max_new=16, serial Metal router, full vocab:
  tokens 19,198,17,10,17,28,19,198,17,10,17,28,19,198,17,10 in 6.867080 s
raw prompt "2+2=", max_new=16, specialized Q4 router, full vocab:
  same token ids as serial Metal router in ~6.35-6.49 s
raw prompt "2+2=", max_new=16, fused GDN+out-proj:
  same token ids and scores as specialized Q4 router in ~7.61-7.63 s on noisy paired samples
raw prompt "2+2=", max_new=16, serial Metal router, vocab_limit=32:
  tokens 19,11,19,10,17,28,19,11,17,10,17,28,19,11,19,10 in 6.314424 s
raw prompt "2+2=", max_new=32, parallel Metal router, vocab_limit=32:
  tokens unchanged from CPU-router baseline in 9.583022 s, but with larger score drift than serial router
raw prompt "2+2=", `-O3` generator build, max_new=16/32/64:
  same token ids as `-O2`; 16 tokens in 7.953053 s, 32 in 13.908750 s,
  64 in 15.974482 s on paired local samples
raw token prompt 17,10,17, Q4 block-256 row-4 Metal matvec, max_new=8:
  same token ids and scores as the previous Q4 matvec path; capped vocab
  60-layer samples improved from 5.118766/4.863442 s to 4.699666/4.646261 s,
  and full-vocab max_new=8 improved from 4.856976 s to 4.640597 s
raw token prompt 17,10,17, IQ1 block-256 row-8 routed-expert Metal matvec:
  same token ids and scores as the previous IQ1 slice path; full-vocab
  max_new=16 improved from 7.308135 s to 5.665501 s with row-4, then
  5.961183/5.650345 s to 5.724127/5.505277 s with row-8 on paired samples
raw token prompt 17,10,17, Q4 block-256 row-8 router:
  same token ids and scores as the row-4 specialized router; full-vocab
  max_new=32 improved from 7.595937 s to 7.425162 s on a paired local sample
raw token prompt 0,1, Q4 block-256 row-8 projection matvec default:
  same token ids as row-4 projections, with small score drift from reduction
  order; full-vocab max_new=32 improved from 7.036358 s to 6.921315 s, and
  max_new=64 improved from 11.850986 s to 11.376880 s on paired local samples
raw token prompt 0,1, `ORNITH_METAL_SHARED_RESIDENT_MB=512`:
  same token ids and scores as shared copy staging; max_new=64 improved from
  11.384548/11.297883 s to 11.172463/11.123517 s, and max_new=128 improved
  from 20.940052 s to 19.985033 s on paired local samples
raw token prompt 0,1, Metal generation MoE scratch reuse, vocab_limit=32:
  same token ids and scores as the previous Metal path; 60-layer max_new=16
  paired samples were 4.842849/4.465718 s baseline vs 4.554837/4.635285 s
  with scratch reuse, and max_new=32 was 7.044319/6.002555 s baseline vs
  6.238517/6.036583 s with scratch reuse
raw token prompt 0,1, `ORNITH_METAL_RESIDENT_LAYER_MB=1024`, top_k=10:
  same token ids and scores as compact staging; max_new=32 was neutral
  (7.135451 s off vs 7.136932 s resident), max_new=64 improved from
  13.673331 s to 12.096104 s, and max_new=128 improved from 22.734660 s to
  21.534458 s in local samples. A 4096 MiB budget was slower on these short
  runs because first-use tensor copies dominated.
raw token prompt 0,1, fused resident Metal linear attention:
  same token ids as the older Metal batch-projection + GDN path, with expected
  small score drift from changed GPU reduction/order. On a warm sequential
  60-layer capped-vocab sample (`max_new=10`, `top_k=4`, `vocab_limit=128`),
  `ORNITH_METAL_LINEAR_ATTN=0` took 6.584314 s and the default resident fused
  path took 3.033750 s. The profile shifted `batch_matvec+gdn` from about
  5.54 s to `linear_attn` about 1.61 s plus 0.43 s residual batch matvec for
  unsupported attention layers.
raw token prompt 0,1, fused resident Metal self attention:
  same token ids as the older self-attention Metal matvec path, with expected
  small score drift. On a warm sequential 60-layer capped-vocab sample
  (`max_new=10`, `top_k=4`, `vocab_limit=128`), `ORNITH_METAL_SELF_ATTN=0`
  took 3.021742 s and the default self hook took 2.983191 s. On a longer
  supported capped sample (`max_new=64`, `vocab_limit=32`) it was neutral
  within noise: 7.969136 s off versus 7.975130 s on.
raw token prompt 0,1, `ORNITH_METAL_LINEAR_RESIDENT_MB=3072`:
  same token ids as the default mapped-weight path. It helped a 60-layer
  capped-vocab top_k=4 sample (`max_new=64`, `vocab_limit=32`) from 12.879188 s
  to 8.551661 s, but did not help the warm realistic top_k=10 full-vocab
  sample: 10.493942 s default mapped weights versus 10.777743 s resident.
  Default remains off.
raw prompt "2+2=", max_new=3: tokens 19,198,17 -> "4\n2" in 98.032124 s
chat prompt "<|im_start|>user\n2+2=<|im_end|>\n<|im_start|>assistant\n":
  token 248068 -> "<think>" in 178.751386 s
interactive `--nothink --max-new 1`, full layers/vocab, two turns:
  first turn `session=reset` in 7.690712 s, second turn `session=reuse`
  with 17-token reused prefix in 3.524433 s
```

These are correctness/usability smokes, not final performance numbers. CPU and
hybrid generation are still too slow for sustained interactive use; the next
useful speed target is keeping more decode state resident on Metal and fusing
the remaining CPU-side recurrence/softmax/orchestration work.

## Linear Attention Notes

Small source references used for the Ornith/Qwen3.5 Gated DeltaNet path are
stored outside the repo at:

```sh
/Users/nir/dev/models/Ornith-1.0-397B/source-notes
```

These are implementation source notes only, not model weights. They include the
vLLM Qwen3.5 wrapper, Qwen3-Next attention source, Qwen Gated DeltaNet layer,
recurrent decode kernel, causal conv helper, gated RMSNorm path, and the
upstream Hugging Face Qwen3.5 MoE model source used to verify raw checkpoint
layouts. The HF source confirms `linear_attn.in_proj_qkv` is contiguous
`[query, key, value]`; full-attention `q_proj` is per-head `[query, gate]`.

Primary references:

- Hugging Face Qwen3.5 docs:
  `https://huggingface.co/docs/transformers/en/model_doc/qwen3_5`
- vLLM Qwen3.5 model docs:
  `https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_5/`
- vLLM Qwen Gated DeltaNet docs:
  `https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn/`

Source-backed facts now encoded in the CPU decode smoke:

- Qwen3.5 uses a 3:1 hybrid stack: three Gated DeltaNet linear-attention layers
  for every full-attention layer.
- Ornith text config uses 60 layers with linear layers at 0,1,2 then full
  attention at 3, repeating.
- Linear attention tensors are non-interleaved Qwen3.5 layout:
  `in_proj_qkv` is `[q, k, v]`, `in_proj_z` is the output gate, and
  `in_proj_b`/`in_proj_a` are separate headwise gates.
- For Ornith, `q/k` are 16 heads x 128 dims, `v/z` are 64 heads x 128 dims,
  and `conv1d.weight` is `[12288, 1, 4]`.
- Decode recurrence uses q/k L2 normalization, `q *= 1/sqrt(head_k_dim)`,
  `beta = sigmoid(b)`, and `g = -exp(A_log) * softplus(a + dt_bias)`.
- Ornith/Qwen3.5 layer norms and full-attention q/k norms are Gemma-style
  RMSNorm: normalized activations are multiplied by `1 + weight`.
- With zero initial recurrent state, `g` has no first-token effect; first-token
  output is `beta * v * dot(k, q/sqrt(head_k_dim))` per value head after the
  causal conv current-column transform.
- Output projection input is normalized per value head with
  `linear_attn.norm.weight` over the 128 value dimension, then gated by
  `silu(z)`, flattened to 8192, and projected by `linear_attn.out_proj.weight`.

What remains for real generation: multi-token prefill/chunk support, a real
session API, tokenizer/prompt-to-token plumbing, and Metal/CUDA kernels for the
full recurrent path. The current native implementation is intentionally a CPU
correctness bridge for wiring, layout validation, and numerical smoke tests.

Run a real two-token sequence through the first full-attention layer:

```sh
/tmp/ornith_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0,1 4 1 5 32 1 decode
```

Current CPU reference samples with local quantized Ornith artifacts:

```text
2 tokens, 4 layers, vocab 32:  1.107560 seconds
2 tokens, 8 layers, vocab 32:  2.098592 seconds
2 tokens, 60 layers, vocab 32: 15.341629 seconds
```

`ornith/ornith_metal.m` adds narrow Metal BF16/Q4/IQ1 matvec kernels over
mapped `.ornq` shard spans plus a Metal-backed token-step smoke CLI:

```sh
clang -O3 -std=c11 -Iornith \
  ornith/ornith.c ornith/ornith_metal.m ornith/ornith_metal_step_smoke.m \
  -framework Foundation -framework Metal -lm -o /tmp/ornith_metal_step_smoke
/tmp/ornith_metal_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0 1 1 5 32 20
```

Current wall-clock one-layer capped timing sample:

```text
CPU:   5 repeats in 0.771488 seconds
Metal: 5 repeats in 0.336660 seconds
```

The Metal and CPU top-k order matches; scores differ only by small float-order
rounding.

For the realistic Ornith routed count (`EXPERT_TOP_K=10`), Metal batches the
selected IQ1 expert slices and scores capped/full lm-head rows on Metal before
CPU top-k selection:

```text
Metal full vocab, 1 layer: 1 repeat in 0.218928 seconds
Metal full vocab, 5 layers: 1 repeat in 0.573513 seconds
```

The 248,320-row lm-head uses a one-thread-per-row Metal policy instead of
one-threadgroup-per-row; the latter was faster for smaller matrices but too
expensive at vocab scale.

Add `trace` after `REPEATS` to print the Metal smoke timing breakdown:

```sh
/tmp/ornith_metal_step_smoke \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0 10 10 5 0 1 trace
```

Current Metal kernels include block-256-specialized Q4 and routed IQ1 paths for
the Ornith `.ornq` layout. General Q4 block-256 projection matvecs default to
the row-8 kernel; fused attention output projections and shared-expert down
projection use the same row-8 shape. Use `ORNITH_METAL_Q4_ROW8=0` for the
older row-4 path during A/B checks outside the fused output/down sites. The
routed MLP smoke path fuses selected-expert
gate/up, SiLU, down, and weighted mix into one Metal command buffer when both
routed tensors are IQ1 block-256. To avoid expensive GPU sparse-mmap faults,
the fused routed path stages only the selected expert slices into compact shared
Metal buffers before dispatch. `ORNITH_METAL_PARALLEL_STAGE=1` enables
parallel host copies for selected routed slice staging; it is off by default
because it helped the 16-token sample but lost on the 128-token sample. Router
scoring defaults to the specialized
one-threadgroup-per-row block-256 Q4 Metal kernel. Use
`ORNITH_METAL_ROUTER=serial` for the older serial Metal accumulation path,
`ORNITH_METAL_ROUTER=parallel` for the generic parallel Metal matvec,
`ORNITH_METAL_ROUTER=0` for the CPU-router fallback, or
`ORNITH_METAL_ROUTER=specialized`/`q4` for the default specialized Q4 Metal
router. A step-smoke profile can make CPU router scoring look faster, but
paired full-generation runs kept the specialized router as the better default.
The linear-attention Metal hook also fuses GDN recurrence with the Q4 out-proj,
avoiding the intermediate gated-vector CPU round trip when the shape matches
Ornith.

The shared-expert path now also stages its Q4 matrices into compact Metal
buffers and fuses gate/up Q4 matvecs with the SiLU product before the Q4 down
projection. Only the small shared gate scalar stays on the CPU path. By
default, `ORNITH_METAL_SHARED_RESIDENT_MB=512` keeps those staged Q4 matrices
resident across tokens/layers; set it to `0` to disable. On the full quantized
60-layer raw-token `0,1` sample (`max_new=128`, `top_k=10`, full vocab), the
fusion preserved token output and reduced the shared-expert profile bucket from
about 1.97 s to 1.89 s. Moving fused output/down Q4 matvecs to the row-8
kernel kept token IDs unchanged on the same sample, with mean score drift about
0.0077 from Q4 reduction-order changes, and improved total time from 18.40 s
to 17.96 s.
Shared-expert Metal work now starts before routed expert-slice staging and is
waited after the routed output is available, overlapping independent shared
GPU work with CPU staging. Set `ORNITH_METAL_OVERLAP_SHARED=0` to restore the
older sequential order. On the full quantized 60-layer raw-token `0,1` sample
(`max_new=128`, `top_k=10`, full vocab), overlap preserved token IDs and
scores exactly and improved total time from 17.90 s to 16.67 s; the visible
non-overlapped shared wait fell to about 0.009 s.
The Metal generation hook keeps routed-MoE host scratch and selected-expert
index buffers in the hook context so full generation does not allocate/free
that workspace once per layer.
`ORNITH_METAL_RESIDENT_LAYER_MB` can instead keep full routed-expert layer
tensors resident and feed the existing selected-slice kernel with original
expert ids, avoiding compact staging for layers that fit the budget.
`ORNITH_METAL_BUFFER_MOE=1` enables the first architectural buffer-resident MoE
path: RMSNorm writes to a Metal buffer, router reads that buffer, routed MoE
writes to a Metal output buffer, and the shared expert is combined with a
Metal sigmoid-scaled add. It is intentionally not the default yet because the
per-layer RMSNorm command boundary costs more than it saves. On a 16-token,
60-layer, top_k=10 raw-token `0,1` run, IDs matched the default path and score
drift stayed around 1e-3; after fusing RMSNorm/router into one command buffer
and moving shared-expert start to GPU buffers, time was 4.15 s vs 4.01 s for
the default path. The next step is to fold routed MoE, shared combine, and
residual updates into a single layer command sequence with hidden state already
resident.
With `ORNITH_METAL_BUFFER_MOE=1`, the post-attention finish hook is enabled by
default. Metal fuses `x + attn` with post-attention RMSNorm, runs buffer MoE,
then does the final `attn + mlp` during the required CPU copy-out. Set
`ORNITH_METAL_LAYER_FINISH=0` to disable it. On the same 16-token sample this
stayed token-stable and improved the buffer-MoE path to about 4.09 s, close to
the default path's 4.01 s.
`ORNITH_METAL_ATTN_BUFFER=1` additionally keeps Metal attention output in a
buffer for the finish hook instead of copying it through CPU first. It is off
by default because the current 16-token sample stayed token-stable but slowed
to about 4.3 s; revisit when the whole hidden state is resident.

Warm local samples after those optimizations:

```text
Metal capped vocab, 10 layers, top_k=10: 0.142864 seconds
Metal full vocab, 60 layers, top_k=10:   0.437375 seconds
Metal full vocab, 60 layers, 5 repeats:  1.542250 seconds
```

Before selected-slice staging and router/shared staging work, the pure sparse
mmap Metal path took roughly 29 seconds for a 60-layer capped smoke step and
two 60-layer repeats in one process took 139.303659 seconds. Deeper performance
work should focus on a final inference graph around this staged expert layout,
router/top-k policy, and attention integration. Lm-head scoring and shared
experts are no longer meaningful bottlenecks in this smoke path.

The old smoke artifacts were deleted on 2026-07-07 for disk recovery:

```sh
/Users/nir/dev/models/Ornith-1.0-397B/quant-smoke
```

The two checked outputs were text-only:

- `model-00001-of-00122.ornq`: 16 tensors, Q4 plus BF16 passthrough, no
  `model.visual.*` tensors
- `model-00002-of-00122.ornq`: routed expert `gate_up_proj` in IQ1

The latest smoke log is `quant-text-current.log`; sampled validation logs are
`validate-00001.log` and `validate-00002.log`.

Do not download Hugging Face files on this machine without explicit approval.

## GPU-Resident Decode Branch

Branch `ornith-gpu-resident-decode` changes:

- `ornith/ornith.h`
- `ornith/ornith.c`
- `ornith/ornith_metal.m`

Goal of this branch: move Ornith decode toward a GPU-resident hot path and
avoid CPU/GPU boundaries. Current implementation adds a narrow
`ornith_layer_decode_fn` hook plus state views, and a gated Metal resident
layer path behind `ORNITH_METAL_RESIDENT_LAYER=1`.

What is in the working tree:

- Metal attention kernels can consume a caller-owned norm `MTLBuffer`.
- `ornith_metal_rmsnorm_buffer` computes RMSNorm directly from one Metal buffer
  into another.
- `metal_layer_decode_hook` computes input RMSNorm, attention, post-attention
  buffer MoE, and delta assembly in the Metal-side layer hook.
- `ORNITH_METAL_RESIDENT_LAYER=1` enables the path. It is not default.

Root cause found during testing: the first resident implementation used global
`temp_buffer` slots `26` and `27` for hidden-sized attention/MLP buffers, while
the shared-expert path reused those slots for argument buffers. That corrupted
resident attention before finish. The resident hook now keeps dedicated
context-owned Metal buffers, released with the hook context.

Verified:

```text
./ornith/check.sh -> ok
ornith_metal_matvec_test -> ok after each compile
```

Real-model findings on the full quantized catalog:

```sh
/tmp/ornith_generate_metal \
  /Users/nir/dev/models/Ornith-1.0-397B/ornith-runtime-catalog.tsv \
  /Users/nir/dev/models/Ornith-1.0-397B/quant-full/out \
  0,1 16 60 10 0 metal
```

Baseline:

```text
seconds=3.931035
tokens: 198,12,198,12,198,12,198,12,198,12,198,12,198,12,198,12
```

Resident layer:

```text
ORNITH_METAL_RESIDENT_LAYER=1
seconds=10.598745
tokens: 198,12,198,12,198,12,198,12,198,12,198,12,198,12,198,12
```

Resident mode is token-stable with small score drift, but slower. Keep it
gated; do not default it. Next useful step is not more hook layering. The next
step is a narrow Metal-owned hidden-state loop that keeps `x` on GPU across
layers and only copies back for final norm/lm-head until those are also moved.

`ORNITH_METAL_TOKEN_LOOP=1` enables that first token-level GPU-owned hidden
state loop. It embeds into a Metal buffer once per token, runs all layers with
`x` resident, applies residual updates on GPU, then scores from the resident
hidden buffer through a Metal final norm/lm-head hook. A fused post-attention
add-RMSNorm-router kernel avoids one norm/router dispatch pair in this path.
Following the same command-buffer scheduling lesson used by llama.cpp/ggml
Metal and MLX Metal, the token-loop path also folds input RMSNorm into the
linear/self-attention command buffer instead of launching and waiting on a
separate command per layer. Profiling mode keeps the old split so timing
buckets remain readable.
On the 16-token, 60-layer, top_k=10 raw-token `0,1` sample, token IDs match
the default path with small score drift, but it is still slower:

```text
default:                 3.997789 seconds
ORNITH_METAL_TOKEN_LOOP: 4.507006 seconds
```

On a later paired full-vocab run after fused input RMSNorm, token IDs stayed
identical and scores drifted only in the existing Metal reduction range; wall
time was effectively flat on the 60-layer sample, so this is a correctness-safe
wait removal rather than a major speed win by itself.

`ORNITH_METAL_ROUTER_TOPK=1` computes selected router top-k and softmax weights
with a small Metal kernel in the same command buffer that writes router scores.
It still copies the selected IDs/weights back because the current expert
staging path is CPU-owned, but it avoids copying all router scores and gives the
future resident-expert path a tested GPU-side router output contract. A
60-layer, top_k=10, full-vocab raw-token `0,1` sample kept token IDs and scores
unchanged and moved from `3.979984` seconds to `3.960835` seconds in one
sequential A/B run; keep it opt-in until it is part of a larger GPU-owned
routing path.

The token-loop final scoring path now encodes final RMSNorm, lm-head matvec,
and raw top-k selection into one Metal command buffer when `k <= 64`. This
removes one CPU wait plus the full-vocab CPU score scan/copy from that path and
adds a tested raw-value GPU top-k primitive. A 60-layer, top_k=10, full-vocab
raw-token `0,1`, max_new=16 token-loop sample kept token IDs unchanged and ran
in `4.541286` seconds. The default non-token-loop path is still faster on the
same sample (`3.938896` seconds in the paired run), so token loop remains
gated until routed expert staging is GPU-owned.
Token-loop decode also no longer copies the final hidden vector back to CPU by
default; hidden top-k reads the resident Metal buffer. Set
`ORNITH_METAL_TOKEN_X_COPYBACK=1` for A/B or CPU fallback diagnostics. A
60-layer, top_k=10, full-vocab raw-token `0,1`, max_new=16 token-loop sample
kept token IDs unchanged and moved from `4.497411` seconds with copyback to
`4.468652` seconds without it.
`ORNITH_METAL_LMHEAD_GPU_TOPK=1` is an opt-in default-path lm-head experiment:
lm-head scoring and raw top-k run in one Metal command buffer, copying back
only `k` results. The greedy `k=1` path now uses a 256-lane Metal reduction
instead of scanning the full vocab on one GPU thread. Tokens and scores stayed
unchanged. A paired 60-layer, top_k=10, full-vocab raw-token `0,1` sample moved
max_new=16 from `4.393940` seconds to `4.003468` seconds; max_new=64 was
neutral (`9.007978` seconds off, `9.045478` seconds on), and max_new=128 was
slower (`15.990855` seconds off, `19.102951` seconds on). Keep it off by
default until it wins longer samples consistently.

`ORNITH_METAL_GPU_SELECTED_ROUTE=1` is an experimental resident-expert route.
When `ORNITH_METAL_ROUTER_TOPK=1` and `ORNITH_METAL_RESIDENT_LAYER_MB` make the
selected layer's routed expert tensors resident, the routed MoE kernel consumes
GPU top-k IDs and weights directly instead of copying them back to CPU and
rebuilding selected-slice buffers. The route is appended to the same command
buffer as router top-k when possible, removing the router-to-routed CPU wait
for this opt-in path. It preserved token IDs on 60-layer, top_k=10, full-vocab
raw-token `0,1` token-loop samples. max_new=8 moved from `3.303609` seconds to
`3.279938` seconds, while max_new=16 was neutral (`4.681458` seconds default
resident/top-k path versus `4.693680` seconds with GPU-selected routing), so it
remains opt-in.

Actual full-catalog text prompt smoke:

```sh
ORNITH_METAL_ROUTER_TOPK=1 python3 ornith/tools/ornith_chat.py \
  --backend metal --raw --max-new 8 --layers 60 --expert-top-k 10 \
  --show-tokens '2+2='
```

The quantized model loaded from `quant-full/out` and generated token `19`
(`4`) first, then continued with `2+2=4`. The chat-shaped `--nothink` prompt
currently repeats thinking delimiters, so it proves execution but not useful
assistant quality yet.

A first coding-quality probe on 2026-07-04 is negative. Command:

```sh
python3 ornith/tools/ornith_chat.py --backend metal --nothink --max-new 48 \
  --layers 60 --expert-top-k 10 --show-tokens 'Write fizzbuzz in C.'
```

It took `35.996584` seconds and produced no code, starting with "The word
\"f\" is not a category. This is a meaningless task." before repeating
thinking delimiters. Treat this as a quality blocker: more performance work is
useful only after comparing against a BF16/fp16 reference run or changing the
compression recipe. The BF16 reference is not local; getting it requires
approved storage/cloud because the full upstream weights are too large for this
machine's current free disk.

Follow-up localization on 2026-07-04:

- The local Python environment does not have the Hugging Face `tokenizers`
  package, so `ornith_chat.py` uses its fallback byte-BPE encoder. For the
  fizzbuzz prompt this is probably not the primary bug: the rendered prompt
  encodes special tokens as single IDs (`<|im_start|>` = `248045`,
  `<|im_end|>` = `248046`, `<think>` = `248068`, `</think>` = `248069`).
- CPU full-vocab first-token generation for the same chat prompt matched Metal:
  token `248068` (`<think>`) with score `12.0539274` on CPU versus
  `12.0539665` on Metal. CPU took `248.677914` seconds for that one token, so
  full 60-layer/full-vocab CPU multi-token checks are too slow for default
  `check.sh`.
- `tests/ornith_cpu_metal_golden_test.py` now has an opt-in operating-point
  case. Set `ORNITH_OPERATING_GOLDEN=1` to compare CPU and Metal on the exact
  fizzbuzz chat token IDs with 60 layers, top_k=10, and full vocab. The default
  check remains the fast 4-layer case.
- The empty `--nothink` think scaffold is not harmless. A raw prompt ending at
  `<|im_start|>assistant\n` avoided the "meaningless task" start and instead
  began a coherent but still failing thought: `The user wants me to write
  "fizz" in C...`; by 48 tokens it repeated that phrase and still produced no
  code. `ornith_chat.py --no-think-scaffold --nothink` now exposes this A/B
  without hand-rendered raw prompts.
- External reference options exist and should be checked before downloading
  397B BF16 locally: DeepReinforce documents vLLM/SGLang/Transformers serving
  for `deepreinforce-ai/Ornith-1.0-397B`, there is an official FP8 model, and
  community GGUF/MLX repos are visible on Hugging Face. These can provide a
  known-good first-token/top-k trace or a better quantization baseline.
- Local quant-error measurement against BF16 was blocked at that point because
  raw safetensors shards were deleted after quantization as intended and
  `quant-smoke` had only `.ornq` files plus logs. Shard 2 is now preserved
  separately under `raw-cache/` for repeated quant experiments.

Sampling probe on 2026-07-04:

- One-shot native generation now accepts sampled decoding via
  `ornith_chat.py --temperature T --sample-top-k K --top-p P --seed S`. This is
  intentionally not wired through the persistent interactive worker yet.
- With normal thinking enabled, `--temperature 0.6 --top-p 0.95
  --sample-top-k 64 --max-new 256` generated 71 tokens in 29.55s, finished a
  short `</think>` trace, and answered `# I will write "fizzb" in C.` rather
  than code.
- With `--nothink --no-think-scaffold`, the same sampled path generated 128
  tokens in 31.74s and looped inside `<think>` around `I'm C`.
- Greedy thinking-enabled `--max-new 256` generated 235 tokens in 65.01s and
  repeated `The user wants me to write "fizz"/"fzz" in C.` with no code.
  Decoding/harness artifacts are therefore not enough to explain the failure;
  quantization degradation or a shared runtime math/layout bug remain live.

Quant-error probe on 2026-07-04:

- `ornith/tools/ornith_quant_error.py` compares raw BF16 safetensors against
  dequantized `.ornq` tensors and writes JSON/Markdown reports. It shells out
  to `ornith_quant_error_raw.c` for full-tensor scans, so the large checks are
  exact, not sampled.
- Shard 2 (`model.language_model.layers.0.mlp.experts.gate_up_proj`, IQ1,
  4.29B params) was scanned end-to-end: mean abs error `0.000170829`, RMSE
  `0.000636297`, relative L2 `0.603748`, max abs `0.13446`.
- Shard 3 was scanned end-to-end. BF16 passthrough tensors were exact
  (`relative_l2=0`). Q4 tensors were moderate (`relative_l2=0.13132`). The IQ1
  routed down projection was much worse: mean abs error `0.000720259`, RMSE
  `0.00147477`, relative L2 `0.950618`, max abs `0.161346`.
- Re-quantizing shard 3 from the downloaded raw safetensors produced a
  byte-identical `.ornq`, so this is not evidence of a corrupt quantization run.
  It points at the current IQ1 recipe being too lossy for routed experts.
- Reports live outside the repo in
  `/Users/nir/dev/models/Ornith-1.0-397B/quant-error/reports/`.

DS4-style candidate probe on 2026-07-04:

- DS4's published 2-bit recipe is asymmetric rather than blanket one-bit:
  routed gate/up uses `IQ2_XXS`, routed down uses `Q2_K`, and non-routed
  tensors stay higher precision. `IQ2_XXS` uses imatrix importance; without a
  real activation imatrix DS4 falls back to per-column weight energy
  `sum(row[column]^2)`.
- Shard 2 raw is intentionally preserved outside the transient quantizer path
  at
  `/Users/nir/dev/models/Ornith-1.0-397B/raw-cache/model-00002-of-00122.safetensors`
  for repeated experiments. The transient
  `/Users/nir/dev/models/Ornith-1.0-397B/quant-error/raw` directory remains
  disposable.
- On layer-0 routed `gate_up_proj` (`[512, 2048, 4096]`), synthetic-imatrix
  `IQ2_XXS` measured relative L2 `0.657291`, worse than current IQ1's
  `0.603748`. `Q2_K` measured `0.297341`; `Q4_K` measured `0.0716374`.
- On layer-0 routed `down_proj` (`[512, 4096, 1024]`), synthetic-imatrix
  `IQ2_XXS` measured relative L2 `0.743402`, current IQ1 measured `0.950618`,
  `Q2_K` measured `0.441085`, and `Q4_K` measured `0.0546557`.
- Current evidence: Ornith's IQ1 recipe is too lossy. DS4's exact
  `IQ2_XXS` gate/up choice does not transfer cleanly with only synthetic
  weight-energy importance, so the smallest promising measured candidate is
  `Q2_K`, with `Q4_K` as the current quality ceiling. `.ornq` supports `q2_k`
  now, but full-routed q2_k is too large; use it selectively on routed
  down-proj unless a later imatrix makes IQ2 practical. Real Ornith activation
  imatrix collection could still make `IQ2_XXS` viable, but it should not be
  assumed.

Keep token loop gated until final norm/lm-head and more layer work are resident
enough to recover the extra GPU command overhead.

## DS4 Scheduler Map

DS4's useful lesson for Ornith is scheduling discipline, not model math.
`ds4_metal.m` states that C owns model semantics and graph scheduling while
Metal functions append work. The important pieces to copy into `ornith_*` code
are:

- one initialized Metal library plus cached pipelines;
- caller-owned command buffers via `ds4_gpu_begin_commands`,
  `ds4_gpu_flush_commands`, and `ds4_gpu_finish_command_buffer`;
- encode helpers that take an existing command buffer and only wait when they
  own that buffer;
- transient shared buffers retained until command completion;
- optional shared-event readback for the unavoidable selected-expert boundary.

The first Ornith port should not be a generic graph engine. Add a tiny
caller-owned command context to `ornith_metal.m`, then convert one already-safe
path at a time from "create/commit/wait" into "encode into caller buffer".
Useful first candidates are final norm + lm-head top-k and fused
linear-attention, because their dependencies are already Metal-owned inside a
token. Routed expert staging is the hard boundary; do not pretend it is
GPU-resident until selected expert IDs and weights stop round-tripping through
CPU slice buffers.

## Metal API Findings

Apple Metal docs checked on 2026-07-03 point to practical runtime work, not a
single missing magic kernel.

Useful APIs and where they fit:

- `MTLCommandBuffer waitUntilCompleted()` blocks the CPU until the GPU and
  completion handlers finish. This confirms the main performance rule for
  Ornith: stop treating GPU helpers as synchronous functions. Encode larger
  token/layer chunks and wait only when CPU-owned state is actually needed.
- Argument buffers group buffers, textures, samplers, and constants into one
  bindable buffer. Apple specifically calls out arrays inside argument buffers,
  often combined with heaps, to reduce CPU overhead. This fits resident
  per-layer resources: router weights, shared experts, linear/self-attention
  weights, and eventually routed expert tables.
- `MTLHeap` can back resident resource arenas. Use it after the hot weights are
  stable enough to avoid ad hoc buffer churn; do not introduce it before the
  current resident-buffer paths show a clear need.
- `MTLIndirectCommandBuffer` can encode repeated dispatches once and reuse
  them. This is useful only after expert weights and argument buffers are
  resident. It does not help much while CPU staging still picks slices.
- `MTLSharedEvent`/`MTLFence` are for synchronization without broad CPU waits.
  Within a single command queue, normal command order handles most dependencies;
  events matter when overlapping I/O, staging, and compute queues.
- `MTLIOCommandQueue` can load filesystem data directly into GPU resources. It
  may help a future streaming/runtime format, but current decode is bottlenecked
  by synchronization and CPU staging, not file reads.
- `MTLBinaryArchive` and precompiled `.metallib` reduce startup/pipeline
  creation time. They are worth doing for cold start, but they will not improve
  per-token throughput after pipelines are created.
- Metal 4 machine-learning passes target Core ML model execution inside a Metal
  timeline. They are not a direct fit for custom `.ornq` IQ1/Q4 MoE kernels.
- Metal 4 placement sparse buffers and mapping updates are interesting for a
  future paged/resident model arena, but they are OS/API-version gated and too
  much machinery for the current hot path.

Leverage order:

1. Build a narrow command scheduler for the existing Metal decode path. Helpers
   should encode into caller-owned command buffers/encoders instead of creating,
   committing, and waiting internally.
2. Keep router top-k output on GPU and remove the CPU copyback by pairing it
   with resident expert resources. Argument buffers are the likely smallest API
   step for this.
3. Move routed expert selection from CPU slice staging to GPU-indexed resident
   tables. Only then consider indirect command buffers for repeated per-layer
   dispatch shape.
4. Add binary archive/precompiled library support for cold-start cleanup once
   the runtime hot path stops moving.

Reference docs:

- https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29
- https://developer.apple.com/documentation/metal/improving-cpu-performance-by-using-argument-buffers
- https://developer.apple.com/documentation/metal/using-argument-buffers-with-resource-heaps
- https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer
- https://developer.apple.com/documentation/metal/mtliocommandqueue
- https://developer.apple.com/documentation/metal/metal-binary-archives
- https://developer.apple.com/documentation/metal/resource-synchronization
- https://developer.apple.com/documentation/metal/machine-learning-passes

REAP notes from `CerebrasResearch/reap`: REAP is directly relevant to reducing
the model footprint, not to fixing Metal runtime synchronization. It prunes
experts by saliency from router weights and expert activation norms, updates
router rows and expert lists in the saved model, and now includes a layer-wise
calibration path intended to fit larger pruning runs on one GPU. The useful
follow-up here is an Ornith pruning/repack path that consumes REAP-style
retained expert IDs before `.ornq` quantization, then re-runs quality and size
checks.

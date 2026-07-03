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
during profiled generation. On a 128-token, 60-layer, top_k=10 raw-token `0,1`
sample, 24,958 of 77,400 comparable expert selections repeated from the same
layer's previous call, a 0.322 hit rate. That is a plausible signal for an
Ornith selected-expert cache, but not strong enough to assume a cache wins
without measuring its staging overhead and memory budget.
A simple 512 MiB shared-buffer selected-expert cache prototype was
token-stable but not a clear win on the 128-token sample: one cold run lost
badly, a repeat run tied/slightly beat baseline within noise, and the cache
added substantial code. It was discarded; revisit only with better cache-hit
instrumentation and a design that accounts cache fill/lookup time inside the
stage profile.
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

Current smoke artifacts live in:

```sh
/Users/nir/dev/models/Ornith-1.0-397B/quant-smoke
```

The two checked outputs are text-only:

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
On the 16-token, 60-layer, top_k=10 raw-token `0,1` sample, token IDs match
the default path with small score drift, but it is still slower:

```text
default:                 3.997789 seconds
ORNITH_METAL_TOKEN_LOOP: 4.507006 seconds
```

Keep token loop gated until final norm/lm-head and more layer work are resident
enough to recover the extra GPU command overhead.

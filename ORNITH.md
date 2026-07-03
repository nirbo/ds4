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
`ornith/tools/ornith_chat.py` is the first non-smoke text CLI. It wraps the
tokenizer, chat renderer, and native generator:

```sh
python3 ornith/tools/ornith_chat.py \
  --max-new 64 \
  --nothink \
  "Write a tiny C function that adds two ints."
python3 ornith/tools/ornith_chat.py --interactive --max-new 128 --nothink
```

Use `VOCAB_LIMIT=0` for the full lm-head. Set `ORNITH_METAL_ATTN_MATVEC=0`,
`ORNITH_METAL_BATCH_MATVEC=0`, `ORNITH_METAL_GDN=0`, or
`ORNITH_METAL_ROUTER=0` to disable those Metal decode hooks for A/B checks.
The router default is the specialized block-256 Q4 Metal router.
`ORNITH_METAL_ROUTER=serial` restores the old serial Metal accumulation path,
`ORNITH_METAL_ROUTER=parallel` uses the generic parallel Metal matvec, and
`ORNITH_METAL_ROUTER=0` uses the CPU router.
The Metal GDN hook fuses linear-attention GDN with `linear_attn.out_proj.weight`
when the out projection is the Ornith Q4/block-256 layout; unsupported layouts
fall back to the older GDN-then-matvec path.
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
raw prompt "2+2=", max_new=3: tokens 19,198,17 -> "4\n2" in 98.032124 s
chat prompt "<|im_start|>user\n2+2=<|im_end|>\n<|im_start|>assistant\n":
  token 248068 -> "<think>" in 178.751386 s
```

These are correctness/usability smokes, not final performance numbers. CPU
and hybrid generation are still too slow for interactive use; the next useful
speed target is keeping more decode state resident on Metal and fusing the
remaining CPU-side recurrence/softmax/orchestration work.

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
the Ornith `.ornq` layout. The routed MLP smoke path fuses selected-expert
gate/up, SiLU, down, and weighted mix into one Metal command buffer when both
routed tensors are IQ1 block-256. To avoid expensive GPU sparse-mmap faults,
the fused routed path stages only the selected expert slices into compact shared
Metal buffers before dispatch. Router scoring now defaults to a specialized
block-256 Q4 Metal kernel. Use `ORNITH_METAL_ROUTER=serial` for the older
serial Metal accumulation path, `ORNITH_METAL_ROUTER=parallel` for the generic
parallel Metal matvec, or `ORNITH_METAL_ROUTER=0` for the CPU-router fallback.
The linear-attention Metal hook also fuses GDN recurrence with the Q4 out-proj,
avoiding the intermediate gated-vector CPU round trip when the shape matches
Ornith.

The shared-expert path now also stages its Q4 matrices into compact Metal
buffers and runs gate/up, SiLU product, and down projection on Metal. Only the
small shared gate scalar stays on the CPU path.

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

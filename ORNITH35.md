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
Metal kernel inside a lazy MLX graph. Synthetic CPU/GPU parity is a mechanism
check only; promotion requires real gate/up/down expert projections to pass
drift and bandwidth gates after the source is verified.

The GatedDeltaNet equations are pinned to Transformers `v5.10.1` commit
`90c3ae54d448d4906b6167317ea5a7f5d48a232d`; hashes for the copied upstream
references live in external `source-notes/transformers-5.10.1/source-state.json`.
`ornith35_gdn_reference.py` is a dependency-free scalar oracle, while
`ornith35_mlx_gdn.py` implements immutable one-token MLX state transitions.
Three-token synthetic output/state parity is a mechanism check. Promotion of a
complete layer remains pending real BF16 source loading and an independent
checkpoint-derived numerical comparison.

The full-attention counterpart follows Qwen3.5's per-head interleaved
query/gate projection layout, `(1 + weight)` Q/K RMSNorm, 64 rotary dimensions,
two KV heads repeated across sixteen query heads, FP32 softmax, and post-
attention sigmoid gating. Text tokens use the same position in all three mRoPE
axes, so the interleaving reduces exactly to the standard partial-RoPE
calculation implemented by the scalar and MLX paths. Their three-token
synthetic output and K/V-state parity is not yet a real-weight acceptance.

The MoE decode boundary keeps router softmax, sorted top-8 IDs, retained score
renormalization, selected packed expert projection, shared-expert gating, and
the final reduction in one lazy MLX graph. A selected-expert Metal kernel
broadcasts one token across gate/up matrices and consumes one vector per down
matrix, so Python never reads router IDs. The scalar oracle decodes the actual
E2M1/E4M3FN/global-scale representation; synthetic parity still requires a
real layer and full-logit comparison before promotion.

`ornith35_mlx_layer.py` composes these boundaries in checkpoint order: centered
input RMSNorm, GDN or gated GQA, first residual, centered post-attention
RMSNorm, routed plus shared MoE, and second residual. It propagates immutable
GDN or K/V state and router observations without host synchronization.

`ornith35_mlx_model.py` is the first complete text graph boundary. It loads
only the explicit embedding, 40 decoder layers, final norm, and untied LM head;
there is no vision field or wildcard tensor load. Its aggregate state binds the
next position to every attention cache, and each token produces the complete
248,320-entry target logit vector. The synthetic two-layer model proves layer
ordering and sequential state plumbing; production loading awaits the verified
source.

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

- native MLX Steel attention specialized for Ornith's GQA shape
- fused RMSNorm, QKV, RoPE, and K/V writes
- chunk-parallel GatedDeltaNet Metal kernels
- token-grouped NVFP4 expert GEMMs during prefill
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

No target or draft weights have been downloaded by the bootstrap feature.

After the target download completes, accept it with:

```sh
python3 ornith35/tools/ornith35_source_verify.py
```

The verifier compares the complete file size, raw safetensors header, payload
decomposition, and full SHA-256 against the pinned metadata. It reports hash
throughput at 1 GiB intervals and writes `source-nvfp4-state.json` atomically
only after every check passes.

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

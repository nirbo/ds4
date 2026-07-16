# Ornith 35B Experiment Ledger

An item is checked only after implementation, correctness validation, quality
measurement, and relevant memory/performance measurement. Every completed item
must record `SUCCESS`, `PARTIAL`, or `REJECTED` with evidence.

## Bootstrap

- [x] Pin and validate target, DSpark, and MTP metadata without weight payloads.
  `SUCCESS` (2026-07-16): pinned revisions and LFS identities; bounded range
  reads cataloged the 93,346-tensor target, 44-tensor DSpark draft, and 785 MTP
  tensors without downloading weight payloads. Ten corruption/shape/index
  tests pass, as does strict validation of all 92,520 target NVFP4 companions.
- [ ] Download and hash the immutable AEON NVFP4 source after explicit approval.
- [ ] Prove complete text-only tensor coverage and exact vision exclusion.
- [ ] Establish authoritative source logits and coding-quality controls.

## Target Runtime

- [ ] Decode ModelOpt NVFP4 experts accurately in MLX on Apple Silicon.
- [ ] Compose one complete GatedDeltaNet layer against an independent reference.
- [ ] Compose one complete full-attention layer against an independent reference.
- [ ] Compose one complete MoE layer with exact top-8 routing and shared expert.
- [ ] Run the complete 40-layer text target with full-vocabulary logits.
- [ ] Materialize or directly load the text-only resident runtime.

## Context And Cache

- [ ] Validate native 262,144-token RoPE and cache semantics.
- [ ] Implement YaRN factor-2 loading for 524,288 tokens.
- [ ] Prove native and YaRN cache profiles cannot be mixed.
- [ ] Implement exact in-memory prefix reuse for K/V and GatedDeltaNet state.
- [ ] Implement atomic persistent prompt-cache checkpoints.
- [ ] Prove save/restore and incremental append logit parity.
- [ ] Add content-addressed workspace cache lookup and bounded disk LRU.
- [ ] Add background cache warming without blocking foreground decode.
- [ ] Evaluate eight-bit K/V against BF16 long-context quality and speed.

## Prefill Performance

- [ ] Establish 2K, 32K, 128K, 262K, and bounded 524K TTFT baselines.
- [ ] Use native Steel flash attention with Ornith-specific GQA tuning.
- [ ] Fuse RMSNorm, QKV, RoPE, and cache writes where numerically safe.
- [ ] Implement chunk-parallel GatedDeltaNet prefill on Metal.
- [ ] Group routed tokens into batched NVFP4 expert GEMMs.
- [ ] Tune chunk scheduling for throughput, scratch memory, and watchdog safety.
- [ ] Measure cold prefill, restored-prefix, and incremental-suffix paths separately.

## Speculative Decode

- [ ] Measure the public matched DSpark draft against the authoritative target.
- [ ] Extract and validate the official Qwen3.5 MTP bootstrap tensors.
- [ ] Distill an Ornith-targeted MTP sidecar if bootstrap acceptance is inadequate.
- [ ] Train or extend an Ornith-targeted DSpark draft if needed.
- [ ] Implement exact block verification with GDN/KV snapshot and rollback.
- [ ] Tune adaptive MTP-versus-DSpark scheduling by context and acceptance.
- [ ] Measure exact generation speed at 2K, 128K, 262K, and 524K context.

## Optional Semantic Changes

- [ ] Evaluate repository-aware AST/symbol retrieval before model changes.
- [ ] Evaluate CacheBlend-style non-prefix reuse under matched coding gates.
- [ ] Evaluate a target-trained sparse-attention indexer for cold 524K prefill.
- [ ] Consider further quantization only after the source runtime is accepted.
- [ ] Consider expert pruning only after quant-only quality is established.

#!/usr/bin/env python3
"""Complete one-token decoder-layer composition for Ornith-35."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_gdn as gdn
import ornith35_mlx_moe as moe
from ornith35_moe_reference import require
from ornith35_nvfp4 import SafetensorsFile


# The reduction layout follows MLX 0.32.0's MIT-licensed `all_reduce` Metal
# path: four contiguous FP32 values per thread and two ordered simd reductions.
RESIDUAL_MEAN_KERNEL_SOURCE = r"""
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
threadgroup float local_sums[32];
float total = 0.0f;
uint base = lid * 4u;
for (uint offset = 0u; offset < 4u; ++offset) {
    uint index = base + offset;
    volatile float added = float(hidden[index]) + float(delta[index]);
    bfloat16_t rounded = bfloat16_t(added);
    output_hidden[index] = rounded;
    volatile float square = float(rounded) * float(rounded);
    total += square;
}
total = simd_sum(total);
if (group == 0u) local_sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0u) local_sums[group] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    float value = lid < 16u ? local_sums[lid] : 0.0f;
    value = simd_sum(value);
    if (lane == 0u) output_mean_square[0] = value / 2048.0f;
}
"""


_residual_mean_kernel = mx.fast.metal_kernel(
    name="ornith35_residual_mean_bf16_2048",
    input_names=["hidden", "delta"],
    output_names=["output_hidden", "output_mean_square"],
    source=RESIDUAL_MEAN_KERNEL_SOURCE,
)


RESIDUAL_RMSNORM_KERNEL_SOURCE = r"""
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
threadgroup float local_sums[32];
threadgroup float inverse_mean[1];
bfloat16_t values[4];
float total = 0.0f;
uint base = lid * 4u;
for (uint offset = 0u; offset < 4u; ++offset) {
    uint index = base + offset;
    volatile float added = float(hidden[index]) + float(delta[index]);
    values[offset] = bfloat16_t(added);
    output_hidden[index] = values[offset];
    volatile float square = float(values[offset]) * float(values[offset]);
    total += square;
}
total = simd_sum(total);
if (group == 0u) local_sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0u) local_sums[group] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    float value = lid < 16u ? local_sums[lid] : 0.0f;
    value = simd_sum(value);
    if (lane == 0u) {
        volatile float mean = value / 2048.0f;
        volatile float adjusted = mean + 1.0e-6f;
        inverse_mean[0] = metal::precise::rsqrt(adjusted);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint offset = 0u; offset < 4u; ++offset) {
    uint index = base + offset;
    volatile float normalized = float(values[offset]) * inverse_mean[0];
    volatile float centered_weight = 1.0f + float(weight[index]);
    volatile float weighted = normalized * centered_weight;
    output_normalized[index] = bfloat16_t(weighted);
}
"""


_residual_rmsnorm_kernel = mx.fast.metal_kernel(
    name="ornith35_residual_rmsnorm_bf16_2048",
    input_names=["hidden", "delta", "weight"],
    output_names=["output_hidden", "output_normalized"],
    source=RESIDUAL_RMSNORM_KERNEL_SOURCE,
)


RESIDUAL_RMSNORM_BATCH_KERNEL_SOURCE = r"""
uint token = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
threadgroup float local_sums[32];
threadgroup float inverse_mean[1];
bfloat16_t values[4];
float total = 0.0f;
uint local_base = lid * 4u;
uint base = token * 2048u + local_base;
for (uint offset = 0u; offset < 4u; ++offset) {
    uint index = base + offset;
    volatile float added = float(hidden[index]) + float(delta[index]);
    values[offset] = bfloat16_t(added);
    output_hidden[index] = values[offset];
    volatile float square = float(values[offset]) * float(values[offset]);
    total += square;
}
total = simd_sum(total);
if (group == 0u) local_sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0u) local_sums[group] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    float value = lid < 16u ? local_sums[lid] : 0.0f;
    value = simd_sum(value);
    if (lane == 0u) {
        volatile float mean = value / 2048.0f;
        volatile float adjusted = mean + 1.0e-6f;
        inverse_mean[0] = metal::precise::rsqrt(adjusted);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint offset = 0u; offset < 4u; ++offset) {
    uint local_index = local_base + offset;
    uint index = base + offset;
    volatile float normalized = float(values[offset]) * inverse_mean[0];
    volatile float centered_weight = 1.0f + float(weight[local_index]);
    volatile float weighted = normalized * centered_weight;
    output_normalized[index] = bfloat16_t(weighted);
}
"""


_residual_rmsnorm_batch_kernel = mx.fast.metal_kernel(
    name="ornith35_residual_rmsnorm_batch_bf16_2048",
    input_names=["hidden", "delta", "weight"],
    output_names=["output_hidden", "output_normalized"],
    source=RESIDUAL_RMSNORM_BATCH_KERNEL_SOURCE,
)


@dataclass(frozen=True)
class LayerNorms:
    input_layernorm: mx.array
    post_attention_layernorm: mx.array


@dataclass(frozen=True)
class GDNLayerWeights:
    token_mixer: gdn.MLXGDNWeights
    moe: moe.MLXMoEWeights
    norms: LayerNorms


@dataclass(frozen=True)
class AttentionLayerWeights:
    token_mixer: attention.MLXAttentionWeights
    moe: moe.MLXMoEWeights
    norms: LayerNorms


@dataclass(frozen=True)
class LayerResult:
    output: mx.array
    state: (
        gdn.MLXGDNState
        | attention.MLXAttentionState
        | attention.MLXLinearAttentionState
    )
    selected_experts: mx.array
    routing_weights: mx.array
    normalized_output: mx.array | None


def fused_residual_mean_square(
    hidden: mx.array,
    delta: mx.array,
) -> tuple[mx.array, mx.array]:
    """Round a production residual and reproduce MLX's FP32 mean exactly."""
    require(
        hidden.dtype == mx.bfloat16 and hidden.shape == (2048,),
        "fused residual hidden mismatch",
    )
    require(
        delta.dtype == mx.bfloat16 and delta.shape == (2048,),
        "fused residual delta mismatch",
    )
    output, mean_square = _residual_mean_kernel(
        inputs=[hidden, delta],
        grid=(512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(2048,), (1,)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return output, mean_square


def fused_residual_rms_norm(
    hidden: mx.array,
    delta: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply the exact production residual and following centered RMSNorm."""
    require(
        hidden.dtype == mx.bfloat16 and hidden.shape == (2048,),
        "fused residual hidden mismatch",
    )
    require(
        delta.dtype == mx.bfloat16 and delta.shape == (2048,),
        "fused residual delta mismatch",
    )
    require(
        weight.dtype == mx.bfloat16 and weight.shape == (2048,),
        "fused residual RMSNorm weight mismatch",
    )
    output, normalized = _residual_rmsnorm_kernel(
        inputs=[hidden, delta, weight],
        grid=(512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[(2048,), (2048,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return output, normalized


def fused_residual_rms_norm_batch(
    hidden: mx.array,
    delta: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply the exact production residual/norm independently per token."""
    require(
        hidden.dtype == mx.bfloat16 and hidden.ndim == 2 and hidden.shape[0] > 0
        and hidden.shape[1] == 2048,
        "fused residual batch hidden mismatch",
    )
    require(
        delta.dtype == mx.bfloat16 and delta.shape == hidden.shape,
        "fused residual batch delta mismatch",
    )
    require(
        weight.dtype == mx.bfloat16 and weight.shape == (2048,),
        "fused residual batch RMSNorm weight mismatch",
    )
    tokens = hidden.shape[0]
    output, normalized = _residual_rmsnorm_batch_kernel(
        inputs=[hidden, delta, weight],
        grid=(tokens * 512, 1, 1),
        threadgroup=(512, 1, 1),
        output_shapes=[hidden.shape, hidden.shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return output, normalized


def residual_and_mean_square(
    hidden: mx.array,
    delta: mx.array,
    *,
    fused: bool,
) -> tuple[mx.array, mx.array | None]:
    """Apply a residual and optionally retain its exact production mean."""
    dtype = hidden.dtype
    require(delta.dtype == dtype and delta.shape == hidden.shape, "residual mismatch")
    if fused and dtype == mx.bfloat16 and hidden.shape == (2048,):
        return fused_residual_mean_square(hidden, delta)
    return (hidden + delta).astype(dtype), None


def residual_and_rms_norm(
    hidden: mx.array,
    delta: mx.array,
    weight: mx.array,
    eps: float,
    *,
    fused_rmsnorm: bool,
    fused_mean_square: bool,
) -> tuple[mx.array, mx.array]:
    """Apply a residual and its following norm with two exact fallbacks."""
    production = (
        hidden.dtype == mx.bfloat16
        and hidden.shape == (2048,)
        and weight.dtype == mx.bfloat16
        and weight.shape == (2048,)
        and eps == 1e-6
    )
    if fused_rmsnorm and production:
        return fused_residual_rms_norm(hidden, delta, weight)
    output, mean_square = residual_and_mean_square(
        hidden,
        delta,
        fused=fused_mean_square,
    )
    return output, qwen_rms_norm(
        output,
        weight,
        eps,
        mean_square=mean_square,
    )


def qwen_rms_norm(
    hidden: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
    *,
    mean_square: mx.array | None = None,
) -> mx.array:
    """Apply Qwen3.5's centered `(1 + weight)` RMSNorm."""
    require(hidden.ndim == 1 and weight.shape == hidden.shape, "RMSNorm shape mismatch")
    require(hidden.dtype == weight.dtype, "RMSNorm dtype mismatch")
    hidden32 = hidden.astype(mx.float32)
    if mean_square is None:
        mean_square = mx.mean(hidden32 * hidden32)
    else:
        require(
            mean_square.dtype == mx.float32 and mean_square.shape == (1,),
            "RMSNorm mean-square mismatch",
        )
    normalized = hidden32 * mx.rsqrt(mean_square + eps)
    return (normalized * (1.0 + weight.astype(mx.float32))).astype(hidden.dtype)


def qwen_rms_norm_batch(
    hidden: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Vectorize the authoritative one-token centered RMSNorm."""
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0 and weight.shape == (hidden.shape[1],),
        "batched RMSNorm shape mismatch",
    )
    require(hidden.dtype == weight.dtype, "batched RMSNorm dtype mismatch")
    return mx.vmap(lambda token: qwen_rms_norm(token, weight, eps))(hidden)


def residual_and_rms_norm_batch(
    hidden: mx.array,
    delta: mx.array,
    weight: mx.array,
    eps: float,
) -> tuple[mx.array, mx.array]:
    """Apply the production batch fusion or its exact generic composition."""
    require(hidden.ndim == 2 and hidden.shape[0] > 0, "residual batch is empty")
    require(delta.dtype == hidden.dtype and delta.shape == hidden.shape, "residual batch mismatch")
    production = (
        hidden.dtype == mx.bfloat16
        and hidden.shape[1] == 2048
        and weight.dtype == mx.bfloat16
        and weight.shape == (2048,)
        and eps == 1e-6
    )
    if production:
        return fused_residual_rms_norm_batch(hidden, delta, weight)
    output = (hidden + delta).astype(hidden.dtype)
    return output, qwen_rms_norm_batch(output, weight, eps)


def _validate_norms(norms: LayerNorms, hidden_size: int, dtype: mx.Dtype) -> None:
    require(norms.input_layernorm.shape == (hidden_size,), "input RMSNorm shape mismatch")
    require(
        norms.post_attention_layernorm.shape == (hidden_size,),
        "post-attention RMSNorm shape mismatch",
    )
    require(norms.input_layernorm.dtype == dtype, "input RMSNorm dtype mismatch")
    require(
        norms.post_attention_layernorm.dtype == dtype,
        "post-attention RMSNorm dtype mismatch",
    )


def forward_gdn(
    hidden: mx.array,
    state: gdn.MLXGDNState,
    weights: GDNLayerWeights,
    gdn_config: gdn.GDNConfig = gdn.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
    *,
    normalized_input: mx.array | None = None,
    next_input_norm: mx.array | None = None,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    fused_gdn_convolution: bool = True,
    fused_gdn_recurrence: bool = True,
    fused_gdn_core_gate: bool = True,
    fused_gdn_recurrence_inputs: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
    _validated: bool = False,
) -> LayerResult:
    require(gdn_config.hidden_size == moe_config.hidden_size, "layer hidden-size mismatch")
    dtype = weights.token_mixer.in_proj_qkv.dtype
    require(weights.moe.router.dtype == dtype, "GDN/MoE dtype mismatch")
    if not _validated:
        _validate_norms(weights.norms, gdn_config.hidden_size, dtype)
    hidden = hidden.astype(dtype)
    if normalized_input is None:
        mixed_input = qwen_rms_norm(
            hidden,
            weights.norms.input_layernorm,
            gdn_config.rms_norm_eps,
        )
    else:
        require(
            normalized_input.dtype == dtype and normalized_input.shape == hidden.shape,
            "normalized GDN input mismatch",
        )
        mixed_input = normalized_input
    mixed, next_state = gdn.decode_step(
        mixed_input,
        state,
        weights.token_mixer,
        gdn_config,
        fused_convolution=fused_gdn_convolution,
        fused_recurrence=fused_gdn_recurrence,
        fused_core_gate_output=fused_gdn_core_gate,
        fused_recurrence_inputs=fused_gdn_recurrence_inputs,
        _validated=_validated,
    )
    hidden, moe_input = residual_and_rms_norm(
        hidden,
        mixed,
        weights.norms.post_attention_layernorm,
        gdn_config.rms_norm_eps,
        fused_rmsnorm=fused_residual_rmsnorm,
        fused_mean_square=fused_residual_mean_square,
    )
    moe_result = moe.forward(
        moe_input,
        weights.moe,
        moe_config,
        paired_gate_up=paired_moe_gate_up,
        fused_shared_gate=fused_moe_shared_gate,
        fused_routed_down=fused_moe_routed_down,
        _validated=_validated,
    )
    if next_input_norm is None:
        output, _ = residual_and_mean_square(
            hidden,
            moe_result.output,
            fused=fused_residual_mean_square,
        )
        normalized_output = None
    else:
        output, normalized_output = residual_and_rms_norm(
            hidden,
            moe_result.output,
            next_input_norm,
            gdn_config.rms_norm_eps,
            fused_rmsnorm=fused_residual_rmsnorm,
            fused_mean_square=fused_residual_mean_square,
        )
    return LayerResult(
        output=output,
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
        normalized_output=normalized_output,
    )


def forward_attention(
    hidden: mx.array,
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState,
    weights: AttentionLayerWeights,
    attention_config: attention.AttentionConfig = attention.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
    *,
    normalized_input: mx.array | None = None,
    next_input_norm: mx.array | None = None,
    attention_rope: attention.MLXTextRoPE | None = None,
    fused_attention_qk_norm_rope: bool = True,
    grouped_attention_gqa: bool = True,
    fused_residual_mean_square: bool = True,
    fused_residual_rmsnorm: bool = True,
    paired_moe_gate_up: bool = True,
    fused_moe_shared_gate: bool = True,
    fused_moe_routed_down: bool = True,
    _validated: bool = False,
) -> LayerResult:
    require(
        attention_config.hidden_size == moe_config.hidden_size,
        "layer hidden-size mismatch",
    )
    dtype = weights.token_mixer.q_proj.dtype
    require(weights.moe.router.dtype == dtype, "attention/MoE dtype mismatch")
    if not _validated:
        _validate_norms(weights.norms, attention_config.hidden_size, dtype)
    hidden = hidden.astype(dtype)
    if normalized_input is None:
        mixed_input = qwen_rms_norm(
            hidden,
            weights.norms.input_layernorm,
            attention_config.rms_norm_eps,
        )
    else:
        require(
            normalized_input.dtype == dtype and normalized_input.shape == hidden.shape,
            "normalized attention input mismatch",
        )
        mixed_input = normalized_input
    mixed, next_state = attention.decode_step(
        mixed_input,
        state,
        weights.token_mixer,
        attention_config,
        rope=attention_rope,
        fused_qk_norm_rope=fused_attention_qk_norm_rope,
        grouped_gqa=grouped_attention_gqa,
        _validated=_validated,
    )
    hidden, moe_input = residual_and_rms_norm(
        hidden,
        mixed,
        weights.norms.post_attention_layernorm,
        attention_config.rms_norm_eps,
        fused_rmsnorm=fused_residual_rmsnorm,
        fused_mean_square=fused_residual_mean_square,
    )
    moe_result = moe.forward(
        moe_input,
        weights.moe,
        moe_config,
        paired_gate_up=paired_moe_gate_up,
        fused_shared_gate=fused_moe_shared_gate,
        fused_routed_down=fused_moe_routed_down,
        _validated=_validated,
    )
    if next_input_norm is None:
        output, _ = residual_and_mean_square(
            hidden,
            moe_result.output,
            fused=fused_residual_mean_square,
        )
        normalized_output = None
    else:
        output, normalized_output = residual_and_rms_norm(
            hidden,
            moe_result.output,
            next_input_norm,
            attention_config.rms_norm_eps,
            fused_rmsnorm=fused_residual_rmsnorm,
            fused_mean_square=fused_residual_mean_square,
        )
    return LayerResult(
        output=output,
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
        normalized_output=normalized_output,
    )


def prefill_gdn(
    hidden: mx.array,
    state: gdn.MLXGDNState,
    weights: GDNLayerWeights,
    gdn_config: gdn.GDNConfig = gdn.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
    *,
    normalized_input: mx.array | None = None,
    next_input_norm: mx.array | None = None,
    fused_moe_shared_gate: bool = True,
) -> LayerResult:
    """Compose a nonempty GatedDeltaNet decoder-layer prefill chunk."""
    require(gdn_config.hidden_size == moe_config.hidden_size, "layer hidden-size mismatch")
    dtype = weights.token_mixer.in_proj_qkv.dtype
    require(weights.moe.router.dtype == dtype, "GDN/MoE dtype mismatch")
    _validate_norms(weights.norms, gdn_config.hidden_size, dtype)
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0 and hidden.shape[1] == gdn_config.hidden_size,
        "GDN prefill layer input mismatch",
    )
    hidden = hidden.astype(dtype)
    if normalized_input is None:
        mixed_input = qwen_rms_norm_batch(
            hidden,
            weights.norms.input_layernorm,
            gdn_config.rms_norm_eps,
        )
    else:
        require(
            normalized_input.dtype == dtype and normalized_input.shape == hidden.shape,
            "normalized GDN prefill input mismatch",
        )
        mixed_input = normalized_input
    mixed, next_state = gdn.prefill_chunk(
        mixed_input,
        state,
        weights.token_mixer,
        gdn_config,
    )
    hidden, moe_input = residual_and_rms_norm_batch(
        hidden,
        mixed,
        weights.norms.post_attention_layernorm,
        gdn_config.rms_norm_eps,
    )
    moe_result = moe.forward_batch(
        moe_input,
        weights.moe,
        moe_config,
        fused_shared_gate=fused_moe_shared_gate,
    )
    if next_input_norm is None:
        output = (hidden + moe_result.output).astype(dtype)
        normalized_output = None
    else:
        output, normalized_output = residual_and_rms_norm_batch(
            hidden,
            moe_result.output,
            next_input_norm,
            gdn_config.rms_norm_eps,
        )
    return LayerResult(
        output=output,
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
        normalized_output=normalized_output,
    )


def prefill_attention(
    hidden: mx.array,
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState,
    weights: AttentionLayerWeights,
    attention_config: attention.AttentionConfig = attention.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
    *,
    normalized_input: mx.array | None = None,
    next_input_norm: mx.array | None = None,
    use_steel: bool = True,
    attention_rope: attention.MLXTextRoPE | None = None,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
) -> LayerResult:
    """Compose a nonempty full-attention decoder-layer prefill chunk."""
    require(
        attention_config.hidden_size == moe_config.hidden_size,
        "layer hidden-size mismatch",
    )
    dtype = weights.token_mixer.q_proj.dtype
    require(weights.moe.router.dtype == dtype, "attention/MoE dtype mismatch")
    _validate_norms(weights.norms, attention_config.hidden_size, dtype)
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0
        and hidden.shape[1] == attention_config.hidden_size,
        "attention prefill layer input mismatch",
    )
    hidden = hidden.astype(dtype)
    if normalized_input is None:
        mixed_input = qwen_rms_norm_batch(
            hidden,
            weights.norms.input_layernorm,
            attention_config.rms_norm_eps,
        )
    else:
        require(
            normalized_input.dtype == dtype and normalized_input.shape == hidden.shape,
            "normalized attention prefill input mismatch",
        )
        mixed_input = normalized_input
    mixed, next_state = attention.prefill_chunk(
        mixed_input,
        state,
        weights.token_mixer,
        attention_config,
        use_steel=use_steel,
        rope=attention_rope,
        grouped_gqa=grouped_attention_gqa,
        exact_long_prefill=exact_long_attention,
        fused_long_softmax_value=fused_long_attention,
    )
    hidden, moe_input = residual_and_rms_norm_batch(
        hidden,
        mixed,
        weights.norms.post_attention_layernorm,
        attention_config.rms_norm_eps,
    )
    moe_result = moe.forward_batch(
        moe_input,
        weights.moe,
        moe_config,
        fused_shared_gate=fused_moe_shared_gate,
    )
    if next_input_norm is None:
        output = (hidden + moe_result.output).astype(dtype)
        normalized_output = None
    else:
        output, normalized_output = residual_and_rms_norm_batch(
            hidden,
            moe_result.output,
            next_input_norm,
            attention_config.rms_norm_eps,
        )
    return LayerResult(
        output=output,
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
        normalized_output=normalized_output,
    )


def prefill_attention_last(
    hidden: mx.array,
    state: attention.MLXAttentionState | attention.MLXLinearAttentionState,
    weights: AttentionLayerWeights,
    attention_config: attention.AttentionConfig = attention.PRODUCTION_CONFIG,
    moe_config: moe.MoEConfig = moe.PRODUCTION_CONFIG,
    *,
    normalized_input: mx.array | None = None,
    next_input_norm: mx.array | None = None,
    attention_rope: attention.MLXTextRoPE | None = None,
    grouped_attention_gqa: bool = True,
    fused_moe_shared_gate: bool = True,
    exact_long_attention: bool = True,
    fused_long_attention: bool | None = None,
) -> LayerResult:
    """Advance a chunk while evaluating only its final observable layer output."""
    require(
        attention_config.hidden_size == moe_config.hidden_size,
        "layer hidden-size mismatch",
    )
    dtype = weights.token_mixer.q_proj.dtype
    require(weights.moe.router.dtype == dtype, "attention/MoE dtype mismatch")
    _validate_norms(weights.norms, attention_config.hidden_size, dtype)
    require(
        hidden.ndim == 2 and hidden.shape[0] > 0
        and hidden.shape[1] == attention_config.hidden_size,
        "attention final prefill layer input mismatch",
    )
    hidden = hidden.astype(dtype)
    if normalized_input is None:
        mixed_input = qwen_rms_norm_batch(
            hidden,
            weights.norms.input_layernorm,
            attention_config.rms_norm_eps,
        )
    else:
        require(
            normalized_input.dtype == dtype and normalized_input.shape == hidden.shape,
            "normalized attention final prefill input mismatch",
        )
        mixed_input = normalized_input
    mixed, next_state = attention.prefill_last_query_chunk(
        mixed_input,
        state,
        weights.token_mixer,
        attention_config,
        rope=attention_rope,
        grouped_gqa=grouped_attention_gqa,
        exact_long_prefill=exact_long_attention,
        fused_long_softmax_value=fused_long_attention,
    )
    hidden_tail, moe_input = residual_and_rms_norm_batch(
        hidden[-1:],
        mixed[None, :],
        weights.norms.post_attention_layernorm,
        attention_config.rms_norm_eps,
    )
    moe_result = moe.forward_batch(
        moe_input,
        weights.moe,
        moe_config,
        fused_shared_gate=fused_moe_shared_gate,
    )
    if next_input_norm is None:
        output = (hidden_tail + moe_result.output).astype(dtype)
        normalized_output = None
    else:
        output, normalized_output = residual_and_rms_norm_batch(
            hidden_tail,
            moe_result.output,
            next_input_norm,
            attention_config.rms_norm_eps,
        )
    return LayerResult(
        output=output,
        state=next_state,
        selected_experts=moe_result.selected_experts,
        routing_weights=moe_result.routing_weights,
        normalized_output=normalized_output,
    )


def _load_bf16(source: SafetensorsFile, name: str, shape: tuple[int, ...]) -> mx.array:
    entry = source.entry(name)
    require(entry.get("dtype") == "BF16", f"expected BF16 tensor: {name}")
    require(entry.get("shape") == list(shape), f"tensor shape mismatch: {name}")
    payload = source.tensor_bytes(name)
    expected_bytes = 2
    for size in shape:
        expected_bytes *= size
    require(len(payload) == expected_bytes, f"tensor payload mismatch: {name}")
    return mx.array(memoryview(payload), dtype=mx.uint8).view(mx.bfloat16).reshape(shape)


def _load_norms(source_path: Path, layer: int) -> LayerNorms:
    prefix = f"model.language_model.layers.{layer}"
    with SafetensorsFile(source_path) as source:
        norms = LayerNorms(
            input_layernorm=_load_bf16(source, f"{prefix}.input_layernorm.weight", (2048,)),
            post_attention_layernorm=_load_bf16(
                source,
                f"{prefix}.post_attention_layernorm.weight",
                (2048,),
            ),
        )
        mx.eval(norms.input_layernorm, norms.post_attention_layernorm)
    return norms


def load_layer(source_path: Path, layer: int) -> GDNLayerWeights | AttentionLayerWeights:
    require(0 <= layer < 40, "layer is outside the Ornith text model")
    norms = _load_norms(source_path, layer)
    moe_weights = moe.load_layer(source_path, layer)
    if layer % 4 == 3:
        return AttentionLayerWeights(
            token_mixer=attention.load_layer(source_path, layer),
            moe=moe_weights,
            norms=norms,
        )
    return GDNLayerWeights(
        token_mixer=gdn.load_layer(source_path, layer),
        moe=moe_weights,
        norms=norms,
    )

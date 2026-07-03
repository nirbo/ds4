#ifndef ORNITH_METAL_H
#define ORNITH_METAL_H

#include "ornith.h"

int ornith_metal_available(void);
typedef struct ornith_metal_step_profile {
    double embed_seconds;
    double layer_seconds;
    double layer_norm_seconds;
    double router_seconds;
    double routed_fused_seconds;
    double routed_gate_up_seconds;
    double routed_activation_seconds;
    double routed_down_seconds;
    double routed_mix_seconds;
    double routed_stage_seconds;
    double routed_kernel_seconds;
    double shared_expert_seconds;
    double final_norm_seconds;
    double lm_head_seconds;
    double max_layer_seconds;
    size_t max_layer_index;
} ornith_metal_step_profile;

int ornith_metal_tensor_matvec(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    uint64_t slice,
    const float *x,
    size_t x_count,
    float *out,
    char *err,
    size_t errcap);
int ornith_metal_gdn_recurrent_step(
    const float *qkv,
    const float *z,
    const float *a,
    const float *b,
    const float *alog,
    const float *dt,
    const float *norm_w,
    float *ssm,
    size_t value_heads,
    size_t head_v,
    size_t key_heads,
    size_t head_k,
    float *gated,
    char *err,
    size_t errcap);
int ornith_metal_layer_moe_smoke(const ornith_model *model, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out, char *err, size_t errcap);
int ornith_metal_lm_head_topk_limited(const ornith_model *model, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values, char *err, size_t errcap);
int ornith_metal_generate_greedy_limited(const ornith_model *model, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, char *err, size_t errcap);
int ornith_metal_session_generate_greedy_limited(ornith_session *session, const uint64_t *prompt_suffix_ids, size_t prompt_suffix_count, size_t max_new, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, char *err, size_t errcap);
int ornith_metal_step_smoke_limited(const ornith_model *model, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values, char *err, size_t errcap);
int ornith_metal_step_smoke_profiled_limited(
    const ornith_model *model,
    uint64_t token_id,
    size_t layer_count,
    size_t expert_top_k,
    size_t out_top_k,
    size_t vocab_limit,
    size_t *indices,
    float *values,
    ornith_metal_step_profile *profile,
    char *err,
    size_t errcap);
int ornith_metal_step_smoke_hybrid_limited(
    const ornith_model *model,
    uint64_t token_id,
    size_t layer_count,
    size_t expert_top_k,
    size_t out_top_k,
    size_t vocab_limit,
    size_t *indices,
    float *values,
    ornith_metal_step_profile *profile,
    char *err,
    size_t errcap);

#endif

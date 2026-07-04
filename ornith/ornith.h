#ifndef ORNITH_H
#define ORNITH_H

#include <stddef.h>
#include <stdint.h>

#define ORNITH_MAX_DIMS 8

typedef enum {
    ORNITH_QUANT_BF16 = 1,
    ORNITH_QUANT_Q4 = 2,
    ORNITH_QUANT_IQ1 = 3,
} ornith_quant;

typedef struct {
    char *file;
    uint64_t size;
    uint64_t data_start;
    uint32_t block_size;
    uint32_t tensor_count;
    int fd;
    const unsigned char *map;
} ornith_shard_info;

typedef struct {
    char *name;
    char *shard;
    char *group;
    char *kind;
    ornith_quant quant;
    uint64_t payload_offset;
    uint64_t nbytes;
    uint64_t nparams;
    int64_t layer;
    uint32_t ndim;
    int64_t shape[ORNITH_MAX_DIMS];
} ornith_tensor_info;

typedef struct ornith_model ornith_model;
typedef struct ornith_session ornith_session;
typedef struct {
    float temperature;
    float top_p;
    size_t top_k;
    uint64_t seed;
} ornith_sampling;
typedef struct {
    size_t qkv_dim;
    size_t value_heads;
    size_t head_v;
    size_t key_heads;
    size_t head_k;
    size_t conv_width;
    float *conv;
    float *conv_w;
    float *ssm;
    float *alog;
    float *dt;
    float *gated_norm;
} ornith_linear_state_view;
typedef struct {
    size_t token_cap;
    size_t token_count;
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t kv_dim;
    float *k;
    float *v;
} ornith_full_state_view;
typedef int (*ornith_moe_with_norm_fn)(const ornith_model *model, int64_t layer, const char *norm_kind, const float *x, size_t hidden, size_t top_k, float *out, void *ctx);
typedef int (*ornith_lm_head_topk_fn)(const ornith_model *model, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values, void *ctx);
typedef int (*ornith_tensor_matvec_fn)(const ornith_model *model, const ornith_tensor_info *tensor, const float *x, size_t x_count, float *out, void *ctx);
typedef int (*ornith_tensor_matvec_batch_fn)(const ornith_model *model, const ornith_tensor_info * const *tensors, size_t count, const float *x, size_t x_count, float **outs, void *ctx);
typedef int (*ornith_gdn_recurrent_fn)(const float *qkv, const float *z, const float *a, const float *b, const float *alog, const float *dt, const float *norm_w, float *ssm, size_t value_heads, size_t head_v, size_t key_heads, size_t head_k, float *gated, const ornith_model *model, const ornith_tensor_info *out_w, float *out, void *ctx);
typedef int (*ornith_linear_attention_fn)(const ornith_model *model, int64_t layer, const float *norm, size_t hidden, float *conv_state, const float *conv_w, float *ssm, const float *alog, const float *dt, const float *gated_norm, size_t qkv_dim, size_t value_heads, size_t head_v, size_t key_heads, size_t head_k, size_t conv_width, float *out, void *ctx);
typedef int (*ornith_self_attention_fn)(const ornith_model *model, int64_t layer, const float *norm, size_t hidden, float *k_state, float *v_state, size_t *token_count, size_t token_cap, size_t q_heads, size_t kv_heads, size_t head_dim, size_t q_rows, size_t pos, float *out, void *ctx);
typedef int (*ornith_decode_token_fn)(const ornith_model *model, uint64_t token_id, size_t pos, size_t layer_count, size_t hidden, size_t top_k, const ornith_linear_state_view *linear, ornith_full_state_view *full, float *x_out, void *ctx);
typedef int (*ornith_hidden_topk_fn)(const ornith_model *model, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values, void *ctx);
typedef int (*ornith_layer_decode_fn)(const ornith_model *model, int64_t layer, const float *x, size_t hidden, size_t pos, size_t top_k, const ornith_linear_state_view *linear, ornith_full_state_view *full, float *out, void *ctx);
typedef int (*ornith_layer_finish_fn)(const ornith_model *model, int64_t layer, const float *x, const float *attn, size_t hidden, size_t top_k, float *out, void *ctx);

int ornith_model_open(const char *catalog_tsv, const char *shard_dir, ornith_model **out, char *err, size_t errcap);
void ornith_model_close(ornith_model *model);
size_t ornith_model_shard_count(const ornith_model *model);
size_t ornith_model_tensor_count(const ornith_model *model);
size_t ornith_model_layer_count(const ornith_model *model);
const ornith_tensor_info *ornith_model_find_tensor(const ornith_model *model, const char *name);
const ornith_tensor_info *ornith_model_find_layer_tensor(const ornith_model *model, int64_t layer, const char *kind);
int ornith_model_validate_shards(const ornith_model *model, char *err, size_t errcap);
int ornith_model_validate_moe_layout(const ornith_model *model, char *err, size_t errcap);
int ornith_model_validate_attention_layout(const ornith_model *model, char *err, size_t errcap);
int ornith_model_map_shards(ornith_model *model, char *err, size_t errcap);
const unsigned char *ornith_tensor_payload(const ornith_model *model, const ornith_tensor_info *tensor, uint32_t *block_size);
const unsigned char *ornith_tensor_mapped_span(const ornith_model *model, const ornith_tensor_info *tensor, uint64_t *payload_offset, uint64_t *span_size, uint32_t *block_size);
int ornith_tensor_value(const ornith_model *model, const ornith_tensor_info *tensor, uint64_t i, float *out);
int ornith_tensor_matvec(const ornith_model *model, const ornith_tensor_info *tensor, const float *x, size_t x_count, float *out);
int ornith_tensor_slice_matvec(const ornith_model *model, const ornith_tensor_info *tensor, uint64_t slice, const float *x, size_t x_count, float *out);
int ornith_rmsnorm(const ornith_model *model, const ornith_tensor_info *weight, const float *x, size_t n, float eps, float *out);
int ornith_topk(const float *scores, size_t n, size_t k, size_t *indices, float *values);
int ornith_layer_moe_smoke(const ornith_model *model, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out);
int ornith_layer_decode_smoke(const ornith_model *model, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out);
int ornith_embed_token(const ornith_model *model, uint64_t token_id, float *out, size_t hidden);
int ornith_lm_head_topk(const ornith_model *model, const float *x, size_t hidden, size_t k, size_t *indices, float *values);
int ornith_lm_head_topk_limited(const ornith_model *model, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values);
int ornith_step_smoke(const ornith_model *model, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t *indices, float *values);
int ornith_step_smoke_limited(const ornith_model *model, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values);
int ornith_decode_smoke_limited(const ornith_model *model, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values);
int ornith_decode_sequence_smoke_limited(const ornith_model *model, const uint64_t *token_ids, size_t token_count, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values);
int ornith_generate_greedy_limited(const ornith_model *model, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count);
int ornith_generate_greedy_limited_with_hooks(const ornith_model *model, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, void *hook_ctx);
int ornith_generate_greedy_limited_with_decode_hooks(const ornith_model *model, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_matvec_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_decode_token_fn token_decode_hook, ornith_hidden_topk_fn hidden_topk_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx);
int ornith_generate_sampled_limited_with_decode_hooks(const ornith_model *model, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, const ornith_sampling *sampling, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_matvec_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_decode_token_fn token_decode_hook, ornith_hidden_topk_fn hidden_topk_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx);
int ornith_session_open(const ornith_model *model, size_t layer_count, size_t expert_top_k, size_t token_cap, ornith_session **out);
void ornith_session_close(ornith_session *session);
size_t ornith_session_token_count(const ornith_session *session);
size_t ornith_session_token_cap(const ornith_session *session);
int ornith_session_generate_greedy_limited(ornith_session *session, const uint64_t *prompt_suffix_ids, size_t prompt_suffix_count, size_t max_new, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count);
int ornith_session_generate_greedy_limited_with_decode_hooks(ornith_session *session, const uint64_t *prompt_suffix_ids, size_t prompt_suffix_count, size_t max_new, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_matvec_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_decode_token_fn token_decode_hook, ornith_hidden_topk_fn hidden_topk_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx);
int ornith_session_generate_sampled_limited_with_decode_hooks(ornith_session *session, const uint64_t *prompt_suffix_ids, size_t prompt_suffix_count, size_t max_new, size_t vocab_limit, const ornith_sampling *sampling, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_matvec_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_decode_token_fn token_decode_hook, ornith_hidden_topk_fn hidden_topk_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx);

#endif

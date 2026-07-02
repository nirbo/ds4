#ifndef ORNITH_METAL_H
#define ORNITH_METAL_H

#include "ornith.h"

int ornith_metal_available(void);
int ornith_metal_tensor_matvec(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    uint64_t slice,
    const float *x,
    size_t x_count,
    float *out,
    char *err,
    size_t errcap);
int ornith_metal_layer_moe_smoke(const ornith_model *model, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out, char *err, size_t errcap);
int ornith_metal_lm_head_topk_limited(const ornith_model *model, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values, char *err, size_t errcap);
int ornith_metal_step_smoke_limited(const ornith_model *model, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values, char *err, size_t errcap);

#endif

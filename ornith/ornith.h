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

int ornith_model_open(const char *catalog_tsv, const char *shard_dir, ornith_model **out, char *err, size_t errcap);
void ornith_model_close(ornith_model *model);
size_t ornith_model_shard_count(const ornith_model *model);
size_t ornith_model_tensor_count(const ornith_model *model);
size_t ornith_model_layer_count(const ornith_model *model);
const ornith_tensor_info *ornith_model_find_tensor(const ornith_model *model, const char *name);
const ornith_tensor_info *ornith_model_find_layer_tensor(const ornith_model *model, int64_t layer, const char *kind);
int ornith_model_validate_shards(const ornith_model *model, char *err, size_t errcap);
int ornith_model_map_shards(ornith_model *model, char *err, size_t errcap);
int ornith_tensor_value(const ornith_model *model, const ornith_tensor_info *tensor, uint64_t i, float *out);
int ornith_tensor_matvec(const ornith_model *model, const ornith_tensor_info *tensor, const float *x, size_t x_count, float *out);
int ornith_rmsnorm(const ornith_model *model, const ornith_tensor_info *weight, const float *x, size_t n, float eps, float *out);
int ornith_topk(const float *scores, size_t n, size_t k, size_t *indices, float *values);

#endif

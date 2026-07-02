#include "ornith.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long arg_u64(const char *s)
{
    return strtoull(s, NULL, 10);
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main(int argc, char **argv)
{
    if (argc < 7 || argc > 9) {
        fprintf(stderr, "usage: %s CATALOG.tsv SHARD_DIR TOKEN_ID LAYERS EXPERT_TOP_K OUT_TOP_K [VOCAB_LIMIT] [REPEATS]\n", argv[0]);
        return 2;
    }

    const char *catalog = argv[1];
    const char *shard_dir = argv[2];
    unsigned long long token_id = arg_u64(argv[3]);
    size_t layers = (size_t)arg_u64(argv[4]);
    size_t expert_top_k = (size_t)arg_u64(argv[5]);
    size_t out_top_k = (size_t)arg_u64(argv[6]);
    size_t vocab_limit = argc >= 8 ? (size_t)arg_u64(argv[7]) : 0;
    size_t repeats = argc == 9 ? (size_t)arg_u64(argv[8]) : 1;
    if (repeats == 0) repeats = 1;

    char err[512] = {0};
    ornith_model *model = NULL;
    if (!ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)) ||
        !ornith_model_validate_moe_layout(model, err, sizeof(err)) ||
        !ornith_model_map_shards(model, err, sizeof(err))) {
        fprintf(stderr, "ornith_step_smoke: %s\n", err[0] ? err : "open failed");
        ornith_model_close(model);
        return 1;
    }

    size_t *indices = calloc(out_top_k, sizeof(size_t));
    float *values = calloc(out_top_k, sizeof(float));
    if (!indices || !values) {
        fprintf(stderr, "ornith_step_smoke: out of memory\n");
        free(indices);
        free(values);
        ornith_model_close(model);
        return 1;
    }
    double start = now_seconds();
    for (size_t i = 0; i < repeats; i++) {
        if (!ornith_step_smoke_limited(model, token_id, layers, expert_top_k, out_top_k, vocab_limit, indices, values)) {
            fprintf(stderr, "ornith_step_smoke: step failed\n");
            free(indices);
            free(values);
            ornith_model_close(model);
            return 1;
        }
    }
    double seconds = now_seconds() - start;

    printf("shards=%zu tensors=%zu layers=%zu token=%llu step_layers=%zu vocab_limit=%zu repeats=%zu seconds=%.6f\n",
           ornith_model_shard_count(model), ornith_model_tensor_count(model), ornith_model_layer_count(model),
           token_id, layers, vocab_limit, repeats, seconds);
    for (size_t i = 0; i < out_top_k; i++) {
        printf("%zu\t%zu\t%.9g\n", i, indices[i], values[i]);
    }

    free(indices);
    free(values);
    ornith_model_close(model);
    return 0;
}

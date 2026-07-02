#include "ornith.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long arg_u64(const char *s)
{
    return strtoull(s, NULL, 10);
}

static uint64_t *parse_tokens(const char *s, size_t *count)
{
    size_t n = 1;
    for (const char *p = s; *p; p++) {
        if (*p == ',') n++;
    }
    uint64_t *ids = calloc(n, sizeof(*ids));
    if (!ids) return NULL;
    char *tmp = malloc(strlen(s) + 1);
    if (!tmp) {
        free(ids);
        return NULL;
    }
    strcpy(tmp, s);
    size_t i = 0;
    for (char *tok = strtok(tmp, ","); tok; tok = strtok(NULL, ",")) {
        ids[i++] = strtoull(tok, NULL, 10);
    }
    free(tmp);
    *count = i;
    return i ? ids : NULL;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main(int argc, char **argv)
{
    if (argc < 7 || argc > 10) {
        fprintf(stderr, "usage: %s CATALOG.tsv SHARD_DIR TOKEN_ID[,TOKEN_ID...] LAYERS EXPERT_TOP_K OUT_TOP_K [VOCAB_LIMIT] [REPEATS] [decode]\n", argv[0]);
        return 2;
    }

    const char *catalog = argv[1];
    const char *shard_dir = argv[2];
    const char *token_arg = argv[3];
    unsigned long long token_id = arg_u64(token_arg);
    size_t layers = (size_t)arg_u64(argv[4]);
    size_t expert_top_k = (size_t)arg_u64(argv[5]);
    size_t out_top_k = (size_t)arg_u64(argv[6]);
    size_t vocab_limit = argc >= 8 ? (size_t)arg_u64(argv[7]) : 0;
    size_t repeats = argc >= 9 ? (size_t)arg_u64(argv[8]) : 1;
    int decode = argc == 10 && strcmp(argv[9], "decode") == 0;
    size_t token_count = 0;
    uint64_t *tokens = (decode && strchr(token_arg, ',')) ? parse_tokens(token_arg, &token_count) : NULL;
    if (decode && strchr(token_arg, ',') && !tokens) {
        fprintf(stderr, "ornith_step_smoke: bad token list\n");
        return 2;
    }
    if (repeats == 0) repeats = 1;

    char err[512] = {0};
    ornith_model *model = NULL;
    if (!ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)) ||
        !ornith_model_validate_moe_layout(model, err, sizeof(err)) ||
        (decode && !ornith_model_validate_attention_layout(model, err, sizeof(err))) ||
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
        free(tokens);
        ornith_model_close(model);
        return 1;
    }
    double start = now_seconds();
    for (size_t i = 0; i < repeats; i++) {
        int ok = decode ?
            (tokens ?
                ornith_decode_sequence_smoke_limited(model, tokens, token_count, layers, expert_top_k, out_top_k, vocab_limit, indices, values) :
                ornith_decode_smoke_limited(model, token_id, layers, expert_top_k, out_top_k, vocab_limit, indices, values)) :
            ornith_step_smoke_limited(model, token_id, layers, expert_top_k, out_top_k, vocab_limit, indices, values);
        if (!ok) {
            fprintf(stderr, "ornith_step_smoke: step failed\n");
            free(indices);
            free(values);
            free(tokens);
            ornith_model_close(model);
            return 1;
        }
    }
    double seconds = now_seconds() - start;

    printf("backend=%s shards=%zu tensors=%zu layers=%zu token=%llu step_layers=%zu vocab_limit=%zu repeats=%zu seconds=%.6f\n",
           decode ? "decode-smoke" : "moe-smoke",
           ornith_model_shard_count(model), ornith_model_tensor_count(model), ornith_model_layer_count(model),
           token_id, layers, vocab_limit, repeats, seconds);
    for (size_t i = 0; i < out_top_k; i++) {
        printf("%zu\t%zu\t%.9g\n", i, indices[i], values[i]);
    }

    free(indices);
    free(values);
    free(tokens);
    ornith_model_close(model);
    return 0;
}

#define _POSIX_C_SOURCE 200809L

#include "ornith.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    char *tmp = malloc(strlen(s) + 1);
    if (!ids || !tmp) {
        free(ids);
        free(tmp);
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
    if (argc != 8) {
        fprintf(stderr, "usage: %s CATALOG.tsv SHARD_DIR PROMPT_TOKEN_IDS MAX_NEW LAYERS EXPERT_TOP_K VOCAB_LIMIT\n", argv[0]);
        return 2;
    }
    const char *catalog = argv[1];
    const char *shard_dir = argv[2];
    size_t prompt_count = 0;
    uint64_t *prompt = parse_tokens(argv[3], &prompt_count);
    size_t max_new = (size_t)arg_u64(argv[4]);
    size_t layers = (size_t)arg_u64(argv[5]);
    size_t expert_top_k = (size_t)arg_u64(argv[6]);
    size_t vocab_limit = (size_t)arg_u64(argv[7]);
    if (!prompt || !max_new) {
        fprintf(stderr, "ornith_generate: bad prompt or max_new\n");
        free(prompt);
        return 2;
    }

    char err[512] = {0};
    ornith_model *model = NULL;
    if (!ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)) ||
        !ornith_model_validate_moe_layout(model, err, sizeof(err)) ||
        !ornith_model_validate_attention_layout(model, err, sizeof(err)) ||
        !ornith_model_map_shards(model, err, sizeof(err))) {
        fprintf(stderr, "ornith_generate: %s\n", err[0] ? err : "open failed");
        free(prompt);
        ornith_model_close(model);
        return 1;
    }

    uint64_t *out = calloc(max_new, sizeof(*out));
    float *scores = calloc(max_new, sizeof(*scores));
    size_t out_count = 0;
    double start = now_seconds();
    int ok = out && scores &&
             ornith_generate_greedy_limited(model, prompt, prompt_count, max_new, layers, expert_top_k, vocab_limit, out, scores, &out_count);
    double seconds = now_seconds() - start;
    if (!ok) {
        fprintf(stderr, "ornith_generate: generation failed\n");
        free(scores);
        free(out);
        free(prompt);
        ornith_model_close(model);
        return 1;
    }
    printf("generated=%zu layers=%zu expert_top_k=%zu vocab_limit=%zu seconds=%.6f\n", out_count, layers, expert_top_k, vocab_limit, seconds);
    for (size_t i = 0; i < out_count; i++) {
        printf("%zu\t%llu\t%.9g\n", i, (unsigned long long)out[i], scores[i]);
    }

    free(scores);
    free(out);
    free(prompt);
    ornith_model_close(model);
    return 0;
}

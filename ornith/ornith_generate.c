#define _POSIX_C_SOURCE 200809L

#include "ornith.h"
#ifdef ORNITH_WITH_METAL
#include "ornith_metal.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned long long arg_u64(const char *s)
{
    return strtoull(s, NULL, 10);
}

static float arg_f32(const char *s)
{
    return strtof(s, NULL);
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

static int run_generation(
    ornith_model *model,
    const uint64_t *prompt,
    size_t prompt_count,
    size_t max_new,
    size_t layers,
    size_t expert_top_k,
    size_t vocab_limit,
    const ornith_sampling *sampling,
    int use_metal,
    FILE *out_fp)
{
    uint64_t *out = calloc(max_new, sizeof(*out));
    float *scores = calloc(max_new, sizeof(*scores));
    size_t out_count = 0;
    char err[512] = {0};
    double start = now_seconds();
    int ok = 0;
    if (out && scores) {
#ifdef ORNITH_WITH_METAL
        ok = use_metal ?
             ornith_metal_generate_sampled_limited(model, prompt, prompt_count, max_new, layers, expert_top_k, vocab_limit, sampling, out, scores, &out_count, err, sizeof(err)) :
             ornith_generate_sampled_limited_with_decode_hooks(model, prompt, prompt_count, max_new, layers, expert_top_k, vocab_limit, sampling, out, scores, &out_count, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
#else
        if (use_metal) {
            snprintf(err, sizeof(err), "metal backend not compiled in");
        } else {
            ok = ornith_generate_sampled_limited_with_decode_hooks(model, prompt, prompt_count, max_new, layers, expert_top_k, vocab_limit, sampling, out, scores, &out_count, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        }
#endif
    }
    double seconds = now_seconds() - start;
    if (!ok) {
        fprintf(out_fp, "error\t%s\n", err[0] ? err : "generation failed");
        free(scores);
        free(out);
        return 0;
    }
    fprintf(out_fp, "backend=%s generated=%zu layers=%zu expert_top_k=%zu vocab_limit=%zu seconds=%.6f", use_metal ? "metal" : "cpu", out_count, layers, expert_top_k, vocab_limit, seconds);
    if (sampling && sampling->temperature > 0.0f) {
        fprintf(out_fp, " sample=1 temperature=%.6g top_k=%zu top_p=%.6g seed=%llu", sampling->temperature, sampling->top_k, sampling->top_p, (unsigned long long)sampling->seed);
    }
    fprintf(out_fp, "\n");
    for (size_t i = 0; i < out_count; i++) {
        fprintf(out_fp, "%zu\t%llu\t%.9g\n", i, (unsigned long long)out[i], scores[i]);
    }
    free(scores);
    free(out);
    return 1;
}

typedef struct {
    ornith_session *session;
    uint64_t *tokens;
    size_t token_count;
    size_t token_cap;
    size_t decode_cap;
} worker_session;

static void worker_session_reset(worker_session *s)
{
    if (!s) return;
    ornith_session_close(s->session);
    s->session = NULL;
    s->token_count = 0;
    s->decode_cap = 0;
}

static void worker_session_free(worker_session *s)
{
    if (!s) return;
    worker_session_reset(s);
    free(s->tokens);
    memset(s, 0, sizeof(*s));
}

static int worker_session_reserve_tokens(worker_session *s, size_t count)
{
    if (count > s->token_cap) {
        size_t next = s->token_cap ? s->token_cap : 256;
        while (next < count) next *= 2;
        uint64_t *p = realloc(s->tokens, next * sizeof(*p));
        if (!p) return 0;
        s->tokens = p;
        s->token_cap = next;
    }
    return 1;
}

static int worker_session_store_tokens(worker_session *s, const uint64_t *tokens, size_t count)
{
    if (!worker_session_reserve_tokens(s, count)) return 0;
    if (count) memcpy(s->tokens, tokens, count * sizeof(*tokens));
    s->token_count = count;
    return 1;
}

static int worker_session_store_generated(worker_session *s, const uint64_t *prompt, size_t prompt_count, const uint64_t *out, size_t out_count)
{
    size_t count = prompt_count + out_count;
    if (!worker_session_reserve_tokens(s, count)) return 0;
    if (prompt_count) memcpy(s->tokens, prompt, prompt_count * sizeof(*prompt));
    if (out_count) memcpy(s->tokens + prompt_count, out, out_count * sizeof(*out));
    s->token_count = count;
    return 1;
}

static int worker_session_can_reuse(const worker_session *s, const uint64_t *prompt, size_t prompt_count, size_t needed_cap)
{
    return s->session &&
           s->token_count <= prompt_count &&
           needed_cap <= s->decode_cap &&
           (!s->token_count || memcmp(s->tokens, prompt, s->token_count * sizeof(*prompt)) == 0);
}

static size_t worker_decode_cap(size_t prompt_count, size_t max_new)
{
    size_t slack = max_new > 64 ? max_new : 64;
    return prompt_count + max_new + slack;
}

static int run_session_generation(
    ornith_model *model,
    worker_session *session,
    const uint64_t *prompt,
    size_t prompt_count,
    size_t max_new,
    size_t layers,
    size_t expert_top_k,
    size_t vocab_limit,
    int use_metal,
    FILE *out_fp)
{
    uint64_t *out = calloc(max_new, sizeof(*out));
    float *scores = calloc(max_new, sizeof(*scores));
    size_t out_count = 0;
    char err[512] = {0};
    size_t needed_cap = prompt_count + max_new;
    int reused = worker_session_can_reuse(session, prompt, prompt_count, needed_cap);
    if (!reused) {
        worker_session_reset(session);
        session->decode_cap = worker_decode_cap(prompt_count, max_new);
        if (!ornith_session_open(model, layers, expert_top_k, session->decode_cap, &session->session)) {
            fprintf(out_fp, "error\t%s\n", "session open failed");
            free(scores);
            free(out);
            return 0;
        }
        if (!worker_session_store_tokens(session, NULL, 0)) {
            fprintf(out_fp, "error\t%s\n", "out of memory");
            free(scores);
            free(out);
            return 0;
        }
    }
    size_t suffix_start = reused ? session->token_count : 0;
    double start = now_seconds();
    int ok = 0;
    if (out && scores) {
#ifdef ORNITH_WITH_METAL
        ok = use_metal ?
             ornith_metal_session_generate_greedy_limited(session->session, prompt + suffix_start, prompt_count - suffix_start, max_new, vocab_limit, out, scores, &out_count, err, sizeof(err)) :
             ornith_session_generate_greedy_limited(session->session, prompt + suffix_start, prompt_count - suffix_start, max_new, vocab_limit, out, scores, &out_count);
#else
        if (use_metal) {
            snprintf(err, sizeof(err), "metal backend not compiled in");
        } else {
            ok = ornith_session_generate_greedy_limited(session->session, prompt + suffix_start, prompt_count - suffix_start, max_new, vocab_limit, out, scores, &out_count);
        }
#endif
    }
    double seconds = now_seconds() - start;
    if (!ok) {
        fprintf(out_fp, "error\t%s\n", err[0] ? err : "generation failed");
        worker_session_reset(session);
        free(scores);
        free(out);
        return 0;
    }
    size_t stepped = ornith_session_token_count(session->session);
    size_t generated_stepped = stepped > prompt_count ? stepped - prompt_count : 0;
    ok = worker_session_store_generated(session, prompt, prompt_count, out, generated_stepped);
    if (!ok) {
        fprintf(out_fp, "error\t%s\n", "out of memory");
        worker_session_reset(session);
        free(scores);
        free(out);
        return 0;
    }
    fprintf(out_fp, "backend=%s generated=%zu layers=%zu expert_top_k=%zu vocab_limit=%zu seconds=%.6f session=%s reused_prefix=%zu session_tokens=%zu session_cap=%zu\n", use_metal ? "metal" : "cpu", out_count, layers, expert_top_k, vocab_limit, seconds, reused ? "reuse" : "reset", suffix_start, session->token_count, session->decode_cap);
    for (size_t i = 0; i < out_count; i++) {
        fprintf(out_fp, "%zu\t%llu\t%.9g\n", i, (unsigned long long)out[i], scores[i]);
    }
    free(scores);
    free(out);
    return 1;
}

static int run_worker(int argc, char **argv)
{
    if (argc < 7 || argc > 8) {
        fprintf(stderr, "usage: %s --worker CATALOG.tsv SHARD_DIR LAYERS EXPERT_TOP_K VOCAB_LIMIT [metal]\n", argv[0]);
        return 2;
    }
    const char *catalog = argv[2];
    const char *shard_dir = argv[3];
    size_t layers = (size_t)arg_u64(argv[4]);
    size_t expert_top_k = (size_t)arg_u64(argv[5]);
    size_t vocab_limit = (size_t)arg_u64(argv[6]);
    int use_metal = argc == 8 && strcmp(argv[7], "metal") == 0;
    if (argc == 8 && !use_metal) {
        fprintf(stderr, "ornith_generate: unknown backend '%s'\n", argv[7]);
        return 2;
    }

    char err[512] = {0};
    ornith_model *model = NULL;
    if (!ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)) ||
        !ornith_model_validate_moe_layout(model, err, sizeof(err)) ||
        !ornith_model_validate_attention_layout(model, err, sizeof(err)) ||
        !ornith_model_map_shards(model, err, sizeof(err))) {
        fprintf(stderr, "ornith_generate: %s\n", err[0] ? err : "open failed");
        ornith_model_close(model);
        return 1;
    }

    worker_session session = {0};
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, stdin) >= 0) {
        line[strcspn(line, "\r\n")] = '\0';
        if (strcmp(line, "quit") == 0) break;
        char *tab = strchr(line, '\t');
        if (!tab) {
            printf("error\tbad request\n\n");
            fflush(stdout);
            continue;
        }
        *tab++ = '\0';
        size_t max_new = (size_t)arg_u64(line);
        size_t prompt_count = 0;
        uint64_t *prompt = parse_tokens(tab, &prompt_count);
        if (!prompt || !max_new) {
            printf("error\tbad prompt or max_new\n\n");
            fflush(stdout);
            free(prompt);
            continue;
        }
        (void)run_session_generation(model, &session, prompt, prompt_count, max_new, layers, expert_top_k, vocab_limit, use_metal, stdout);
        printf("\n");
        fflush(stdout);
        free(prompt);
    }
    worker_session_free(&session);
    free(line);
    ornith_model_close(model);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--worker") == 0) {
        return run_worker(argc, argv);
    }
    if (argc < 8 || argc > 14) {
        fprintf(stderr, "usage: %s CATALOG.tsv SHARD_DIR PROMPT_TOKEN_IDS MAX_NEW LAYERS EXPERT_TOP_K VOCAB_LIMIT [metal] [sample TEMP TOP_K TOP_P SEED]\n", argv[0]);
        fprintf(stderr, "       %s --worker CATALOG.tsv SHARD_DIR LAYERS EXPERT_TOP_K VOCAB_LIMIT [metal]\n", argv[0]);
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
    int argi = 8;
    int use_metal = 0;
    ornith_sampling sampling = {0};
    ornith_sampling *sampling_ptr = NULL;
    if (argi < argc && strcmp(argv[argi], "metal") == 0) {
        use_metal = 1;
        argi++;
    }
    if (argi < argc && strcmp(argv[argi], "sample") == 0 && argi + 4 < argc) {
        sampling.temperature = arg_f32(argv[argi + 1]);
        sampling.top_k = (size_t)arg_u64(argv[argi + 2]);
        sampling.top_p = arg_f32(argv[argi + 3]);
        sampling.seed = (uint64_t)arg_u64(argv[argi + 4]);
        sampling_ptr = &sampling;
        argi += 5;
    }
    if (!prompt || !max_new) {
        fprintf(stderr, "ornith_generate: bad prompt or max_new\n");
        free(prompt);
        return 2;
    }
    if (argi != argc || (sampling_ptr && (sampling.temperature <= 0.0f || sampling.top_k == 0 || sampling.top_k > 64 || sampling.top_p <= 0.0f))) {
        fprintf(stderr, "ornith_generate: bad backend or sampling args\n");
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

    int ok = run_generation(model, prompt, prompt_count, max_new, layers, expert_top_k, vocab_limit, sampling_ptr, use_metal, stdout);
    if (!ok) {
        fprintf(stderr, "ornith_generate: generation failed\n");
        free(prompt);
        ornith_model_close(model);
        return 1;
    }
    free(prompt);
    ornith_model_close(model);
    return 0;
}

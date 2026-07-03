#include "ornith.h"
#include "ornith_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned long long arg_u64(const char *s)
{
    return strtoull(s, NULL, 10);
}

static int arg_trace(const char *s)
{
    return s && (strcmp(s, "trace") == 0 || strcmp(s, "1") == 0 || strcmp(s, "true") == 0 || strcmp(s, "hybrid-trace") == 0);
}

static int arg_hybrid(const char *s)
{
    return s && (strcmp(s, "hybrid") == 0 || strcmp(s, "hybrid-trace") == 0);
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
        fprintf(stderr, "usage: %s CATALOG.tsv SHARD_DIR TOKEN_ID LAYERS EXPERT_TOP_K OUT_TOP_K [VOCAB_LIMIT] [REPEATS] [trace|hybrid|hybrid-trace]\n", argv[0]);
        return 2;
    }

    const char *catalog = argv[1];
    const char *shard_dir = argv[2];
    unsigned long long token_id = arg_u64(argv[3]);
    size_t layers = (size_t)arg_u64(argv[4]);
    size_t expert_top_k = (size_t)arg_u64(argv[5]);
    size_t out_top_k = (size_t)arg_u64(argv[6]);
    size_t vocab_limit = argc >= 8 ? (size_t)arg_u64(argv[7]) : 0;
    size_t repeats = argc >= 9 ? (size_t)arg_u64(argv[8]) : 1;
    int trace = argc == 10 && arg_trace(argv[9]);
    int hybrid = argc == 10 && arg_hybrid(argv[9]);
    if (repeats == 0) repeats = 1;

    char err[512] = {0};
    ornith_model *model = NULL;
    if (!ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)) ||
        !ornith_model_validate_moe_layout(model, err, sizeof(err)) ||
        !ornith_model_map_shards(model, err, sizeof(err))) {
        fprintf(stderr, "ornith_metal_step_smoke: %s\n", err[0] ? err : "open failed");
        ornith_model_close(model);
        return 1;
    }

    size_t *indices = calloc(out_top_k, sizeof(size_t));
    float *values = calloc(out_top_k, sizeof(float));
    if (!indices || !values) {
        fprintf(stderr, "ornith_metal_step_smoke: out of memory\n");
        free(indices);
        free(values);
        ornith_model_close(model);
        return 1;
    }

    double start = now_seconds();
    ornith_metal_step_profile profile_sum = {0};
    for (size_t i = 0; i < repeats; i++) {
        ornith_metal_step_profile profile = {0};
        int ok = 0;
        if (hybrid) {
            ok = ornith_metal_step_smoke_hybrid_limited(model, token_id, layers, expert_top_k, out_top_k, vocab_limit, indices, values, trace ? &profile : NULL, err, sizeof(err));
        } else if (trace) {
            ok = ornith_metal_step_smoke_profiled_limited(model, token_id, layers, expert_top_k, out_top_k, vocab_limit, indices, values, &profile, err, sizeof(err));
        } else {
            ok = ornith_metal_step_smoke_limited(model, token_id, layers, expert_top_k, out_top_k, vocab_limit, indices, values, err, sizeof(err));
        }
        if (!ok) {
            fprintf(stderr, "ornith_metal_step_smoke: %s\n", err[0] ? err : "step failed");
            free(indices);
            free(values);
            ornith_model_close(model);
            return 1;
        }
        if (trace) {
            profile_sum.embed_seconds += profile.embed_seconds;
            profile_sum.layer_seconds += profile.layer_seconds;
            profile_sum.layer_norm_seconds += profile.layer_norm_seconds;
            profile_sum.router_seconds += profile.router_seconds;
            profile_sum.routed_fused_seconds += profile.routed_fused_seconds;
            profile_sum.routed_gate_up_seconds += profile.routed_gate_up_seconds;
            profile_sum.routed_activation_seconds += profile.routed_activation_seconds;
            profile_sum.routed_down_seconds += profile.routed_down_seconds;
            profile_sum.routed_mix_seconds += profile.routed_mix_seconds;
            profile_sum.routed_stage_seconds += profile.routed_stage_seconds;
            profile_sum.routed_kernel_seconds += profile.routed_kernel_seconds;
            profile_sum.shared_expert_seconds += profile.shared_expert_seconds;
            profile_sum.final_norm_seconds += profile.final_norm_seconds;
            profile_sum.lm_head_seconds += profile.lm_head_seconds;
            if (profile.max_layer_seconds > profile_sum.max_layer_seconds) {
                profile_sum.max_layer_seconds = profile.max_layer_seconds;
                profile_sum.max_layer_index = profile.max_layer_index;
            }
        }
    }
    double seconds = now_seconds() - start;

    printf("backend=%s shards=%zu tensors=%zu layers=%zu token=%llu step_layers=%zu vocab_limit=%zu repeats=%zu seconds=%.6f\n",
           hybrid ? "hybrid" : "metal",
           ornith_model_shard_count(model), ornith_model_tensor_count(model), ornith_model_layer_count(model),
           token_id, layers, vocab_limit, repeats, seconds);
    if (trace) {
        printf("trace embed=%.6f layers=%.6f layer_norm=%.6f router=%.6f routed_fused=%.6f routed_gate_up=%.6f routed_activation=%.6f routed_down=%.6f routed_mix=%.6f routed_stage=%.6f routed_kernel=%.6f shared_expert=%.6f final_norm=%.6f lm_head=%.6f max_layer=%zu max_layer_seconds=%.6f\n",
               profile_sum.embed_seconds, profile_sum.layer_seconds, profile_sum.layer_norm_seconds,
               profile_sum.router_seconds, profile_sum.routed_fused_seconds,
               profile_sum.routed_gate_up_seconds, profile_sum.routed_activation_seconds,
               profile_sum.routed_down_seconds, profile_sum.routed_mix_seconds,
               profile_sum.routed_stage_seconds, profile_sum.routed_kernel_seconds,
               profile_sum.shared_expert_seconds,
               profile_sum.final_norm_seconds, profile_sum.lm_head_seconds, profile_sum.max_layer_index,
               profile_sum.max_layer_seconds);
    }
    for (size_t i = 0; i < out_top_k; i++) {
        printf("%zu\t%zu\t%.9g\n", i, indices[i], values[i]);
    }

    free(indices);
    free(values);
    ornith_model_close(model);
    return 0;
}

#include "ornith.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t total_tokens;
    uint64_t *freq;
    double *weighted_freq;
    double *ean_sum;
    double *reap_sum;
    float *max_activations;
} layer_obs;

typedef struct {
    size_t layers;
    size_t experts;
    layer_obs *layer;
} reap_ctx;

static float sigf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static float siluf_local(float x) { return x * sigf_local(x); }

static uint64_t *parse_tokens(const char *s, size_t *count)
{
    size_t n = 1;
    for (const char *p = s; *p; p++) if (*p == ',') n++;
    uint64_t *ids = calloc(n, sizeof(*ids));
    char *tmp = malloc(strlen(s) + 1);
    if (!ids || !tmp) {
        free(ids);
        free(tmp);
        return NULL;
    }
    strcpy(tmp, s);
    size_t i = 0;
    for (char *tok = strtok(tmp, ","); tok; tok = strtok(NULL, ",")) ids[i++] = strtoull(tok, NULL, 10);
    free(tmp);
    *count = i;
    return i ? ids : NULL;
}

static int softmax_selected(float *v, size_t n)
{
    float maxv = v[0], sum = 0.0f;
    for (size_t i = 1; i < n; i++) if (v[i] > maxv) maxv = v[i];
    for (size_t i = 0; i < n; i++) {
        v[i] = expf(v[i] - maxv);
        sum += v[i];
    }
    if (sum <= 0.0f) return 0;
    for (size_t i = 0; i < n; i++) v[i] /= sum;
    return 1;
}

static int add_shared_expert_obs(const ornith_model *m, int64_t layer, const float *norm, size_t hidden, float *out)
{
    const ornith_tensor_info *gate = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.gate_proj.weight");
    const ornith_tensor_info *up = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.up_proj.weight");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.down_proj.weight");
    const ornith_tensor_info *sgate = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert_gate.weight");
    if (!gate || !up || !down || !sgate || gate->ndim != 2 || up->ndim != 2 || down->ndim != 2 ||
        gate->shape[1] != (int64_t)hidden || up->shape[1] != (int64_t)hidden ||
        gate->shape[0] != up->shape[0] || down->shape[1] != gate->shape[0]) {
        return 0;
    }
    size_t inter = (size_t)gate->shape[0];
    float *buf = calloc(inter * 3 + 1, sizeof(float));
    if (!buf) return 0;
    float *g = buf, *u = g + inter, *mid = u + inter, *tmp = mid + inter, s = 0.0f;
    int ok = ornith_tensor_matvec(m, gate, norm, hidden, g) &&
             ornith_tensor_matvec(m, up, norm, hidden, u) &&
             ornith_tensor_matvec(m, sgate, norm, hidden, &s);
    for (size_t i = 0; ok && i < inter; i++) mid[i] = siluf_local(g[i]) * u[i];
    ok = ok && ornith_tensor_matvec(m, down, mid, inter, tmp);
    float w = sigf_local(s);
    for (size_t i = 0; ok && i < hidden; i++) out[i] += w * tmp[i];
    free(buf);
    return ok;
}

static int ensure_layer(reap_ctx *ctx, size_t layer, size_t experts)
{
    if (layer >= ctx->layers) return 0;
    layer_obs *o = &ctx->layer[layer];
    if (o->freq) return experts == ctx->experts;
    o->freq = calloc(experts, sizeof(*o->freq));
    o->weighted_freq = calloc(experts, sizeof(*o->weighted_freq));
    o->ean_sum = calloc(experts, sizeof(*o->ean_sum));
    o->reap_sum = calloc(experts, sizeof(*o->reap_sum));
    o->max_activations = calloc(experts, sizeof(*o->max_activations));
    return o->freq && o->weighted_freq && o->ean_sum && o->reap_sum && o->max_activations;
}

static int reap_moe_hook(const ornith_model *m, int64_t layer, const char *norm_kind, const float *x, size_t hidden, size_t top_k, float *out, void *vctx)
{
    reap_ctx *ctx = vctx;
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, norm_kind);
    const ornith_tensor_info *router = ornith_model_find_layer_tensor(m, layer, "mlp.gate.weight");
    const ornith_tensor_info *gate_up = ornith_model_find_layer_tensor(m, layer, "mlp.experts.gate_up_proj");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.experts.down_proj");
    if (!ctx || !norm_w || !router || !gate_up || !down || router->ndim != 2 ||
        gate_up->ndim != 3 || down->ndim != 3 || top_k == 0 || top_k > (size_t)router->shape[0]) return 0;
    size_t experts = (size_t)router->shape[0], inter = (size_t)down->shape[2], gu = (size_t)gate_up->shape[1];
    if (!ensure_layer(ctx, (size_t)layer, experts)) return 0;
    float *scratch = calloc(hidden + experts + top_k + gu + inter + hidden, sizeof(float));
    size_t *idx = calloc(top_k, sizeof(size_t));
    if (!scratch || !idx) {
        free(scratch);
        free(idx);
        return 0;
    }
    float *norm = scratch, *scores = norm + hidden, *weights = scores + experts;
    float *gate_up_out = weights + top_k, *mid = gate_up_out + gu, *tmp = mid + inter;
    memset(out, 0, hidden * sizeof(float));
    int ok = ornith_rmsnorm(m, norm_w, x, hidden, 1e-6f, norm) &&
             ornith_tensor_matvec(m, router, norm, hidden, scores) &&
             ornith_topk(scores, experts, top_k, idx, weights) &&
             softmax_selected(weights, top_k);
    layer_obs *obs = &ctx->layer[layer];
    if (ok) obs->total_tokens++;
    for (size_t k = 0; ok && k < top_k; k++) {
        size_t e = idx[k];
        ok = ornith_tensor_slice_matvec(m, gate_up, e, norm, hidden, gate_up_out);
        for (size_t i = 0; ok && i < inter; i++) mid[i] = siluf_local(gate_up_out[i]) * gate_up_out[i + inter];
        ok = ok && ornith_tensor_slice_matvec(m, down, e, mid, inter, tmp);
        double ss = 0.0;
        float maxv = 0.0f;
        for (size_t i = 0; ok && i < hidden; i++) {
            out[i] += weights[k] * tmp[i];
            ss += (double)tmp[i] * (double)tmp[i];
            float av = fabsf(tmp[i]);
            if (av > maxv) maxv = av;
        }
        if (ok) {
            double norm2 = sqrt(ss);
            obs->freq[e]++;
            obs->weighted_freq[e] += weights[k];
            obs->ean_sum[e] += norm2;
            obs->reap_sum[e] += norm2 * weights[k];
            if (maxv > obs->max_activations[e]) obs->max_activations[e] = maxv;
        }
    }
    ok = ok && add_shared_expert_obs(m, layer, norm, hidden, out);
    free(idx);
    free(scratch);
    return ok;
}

static void write_array_u64(FILE *fp, const char *name, const uint64_t *v, size_t n)
{
    fprintf(fp, "\"%s\":[", name);
    for (size_t i = 0; i < n; i++) fprintf(fp, "%s%llu", i ? "," : "", (unsigned long long)v[i]);
    fprintf(fp, "]");
}

static void write_array_double(FILE *fp, const char *name, const double *v, size_t n)
{
    fprintf(fp, "\"%s\":[", name);
    for (size_t i = 0; i < n; i++) fprintf(fp, "%s%.9g", i ? "," : "", v[i]);
    fprintf(fp, "]");
}

static int write_observer_json(const char *path, const reap_ctx *ctx)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return 0;
    fprintf(fp, "{\n\"format\":\"ornith-reap-observer-v1\",\n\"layers\":{\n");
    int first_layer = 1;
    for (size_t l = 0; l < ctx->layers; l++) {
        const layer_obs *o = &ctx->layer[l];
        if (!o->freq) continue;
        double *ean_mean = calloc(ctx->experts, sizeof(double));
        double *reap = calloc(ctx->experts, sizeof(double));
        double *maxa = calloc(ctx->experts, sizeof(double));
        if (!ean_mean || !reap || !maxa) {
            free(ean_mean); free(reap); free(maxa); fclose(fp); return 0;
        }
        for (size_t e = 0; e < ctx->experts; e++) {
            if (o->freq[e]) {
                ean_mean[e] = o->ean_sum[e] / (double)o->freq[e];
                reap[e] = o->reap_sum[e] / (double)o->freq[e];
            }
            maxa[e] = o->max_activations[e];
        }
        fprintf(fp, "%s\"%zu\":{\"total_tokens\":%llu,", first_layer ? "" : ",\n", l, (unsigned long long)o->total_tokens);
        write_array_u64(fp, "expert_frequency", o->freq, ctx->experts); fprintf(fp, ",");
        write_array_double(fp, "weighted_expert_frequency_sum", o->weighted_freq, ctx->experts); fprintf(fp, ",");
        write_array_double(fp, "ean_mean", ean_mean, ctx->experts); fprintf(fp, ",");
        write_array_double(fp, "reap", reap, ctx->experts); fprintf(fp, ",");
        write_array_double(fp, "max_activations", maxa, ctx->experts); fprintf(fp, "}");
        first_layer = 0;
        free(ean_mean); free(reap); free(maxa);
    }
    fprintf(fp, "\n}}\n");
    fclose(fp);
    return 1;
}

static void free_ctx(reap_ctx *ctx)
{
    if (!ctx || !ctx->layer) return;
    for (size_t l = 0; l < ctx->layers; l++) {
        free(ctx->layer[l].freq);
        free(ctx->layer[l].weighted_freq);
        free(ctx->layer[l].ean_sum);
        free(ctx->layer[l].reap_sum);
        free(ctx->layer[l].max_activations);
    }
    free(ctx->layer);
}

int main(int argc, char **argv)
{
    if (argc != 9) {
        fprintf(stderr, "usage: %s CATALOG.tsv SHARD_DIR TOKEN_ID[,TOKEN_ID...] MAX_NEW LAYERS EXPERT_TOP_K VOCAB_LIMIT OUT.json\n", argv[0]);
        return 2;
    }
    size_t prompt_count = 0;
    uint64_t *prompt = parse_tokens(argv[3], &prompt_count);
    size_t max_new = (size_t)strtoull(argv[4], NULL, 10);
    size_t layers = (size_t)strtoull(argv[5], NULL, 10);
    size_t top_k = (size_t)strtoull(argv[6], NULL, 10);
    size_t vocab_limit = (size_t)strtoull(argv[7], NULL, 10);
    const char *out_json = argv[8];
    if (!prompt || !prompt_count || !max_new || !layers || !top_k) {
        free(prompt);
        return 2;
    }
    char err[512] = {0};
    ornith_model *model = NULL;
    if (!ornith_model_open(argv[1], argv[2], &model, err, sizeof(err)) ||
        !ornith_model_validate_moe_layout(model, err, sizeof(err)) ||
        !ornith_model_validate_attention_layout(model, err, sizeof(err)) ||
        !ornith_model_map_shards(model, err, sizeof(err))) {
        fprintf(stderr, "ornith_reap_observe: %s\n", err[0] ? err : "open failed");
        free(prompt);
        ornith_model_close(model);
        return 1;
    }
    const ornith_tensor_info *router = ornith_model_find_layer_tensor(model, 0, "mlp.gate.weight");
    reap_ctx ctx = {.layers = layers, .experts = router ? (size_t)router->shape[0] : 0, .layer = calloc(layers, sizeof(layer_obs))};
    uint64_t *out_ids = calloc(max_new, sizeof(*out_ids));
    float *scores = calloc(max_new, sizeof(*scores));
    size_t out_count = 0;
    int ok = ctx.layer && out_ids && scores &&
        ornith_generate_greedy_limited_with_decode_hooks(model, prompt, prompt_count, max_new, layers, top_k, vocab_limit, out_ids, scores, &out_count,
            reap_moe_hook, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &ctx) &&
        write_observer_json(out_json, &ctx);
    printf("observed_layers=%zu experts=%zu prompt=%zu generated=%zu out=%s\n", layers, ctx.experts, prompt_count, out_count, out_json);
    free(out_ids);
    free(scores);
    free_ctx(&ctx);
    free(prompt);
    ornith_model_close(model);
    return ok ? 0 : 1;
}

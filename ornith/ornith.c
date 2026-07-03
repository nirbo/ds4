#define _POSIX_C_SOURCE 200809L

#include "ornith.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct ornith_model {
    char *shard_dir;
    ornith_shard_info *shards;
    size_t shard_count;
    size_t shard_cap;
    ornith_tensor_info *tensors;
    size_t tensor_count;
    size_t tensor_cap;
};

typedef struct {
    size_t qkv_dim;
    size_t value_heads;
    size_t head_v;
    size_t key_heads;
    size_t head_k;
    size_t conv_width;
    float *conv;
    float *conv_w;
    float *alog;
    float *dt;
    float *gated_norm;
    float *ssm;
} ornith_linear_state;

typedef struct {
    size_t token_cap;
    size_t token_count;
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t kv_dim;
    float *k;
    float *v;
    float *scores;
} ornith_full_state;

typedef struct {
    size_t layer_count;
    size_t hidden;
    size_t expert_top_k;
    ornith_linear_state *linear;
    ornith_full_state *full;
    float *x;
    float *delta;
    float *norm;
} ornith_decode_state;

struct ornith_session {
    const ornith_model *model;
    ornith_decode_state decode;
    size_t token_count;
    size_t token_cap;
};

static void set_err(char *err, size_t errcap, const char *msg)
{
    if (err && errcap) {
        snprintf(err, errcap, "%s", msg);
    }
}

static char *dupstr(const char *s)
{
    size_t n = strlen(s) + 1;
    char *out = malloc(n);
    if (out) {
        memcpy(out, s, n);
    }
    return out;
}

static int split_tab(char *line, char **fields, int max_fields)
{
    int n = 0;
    line[strcspn(line, "\r\n")] = 0;
    while (n < max_fields) {
        fields[n++] = line;
        char *tab = strchr(line, '\t');
        if (!tab) {
            break;
        }
        *tab = 0;
        line = tab + 1;
    }
    return n;
}

static uint64_t u64(const char *s)
{
    return strtoull(s, NULL, 10);
}

static int64_t i64(const char *s)
{
    return strtoll(s, NULL, 10);
}

static ornith_quant parse_quant(const char *s)
{
    if (strcmp(s, "bf16") == 0) return ORNITH_QUANT_BF16;
    if (strcmp(s, "q4") == 0) return ORNITH_QUANT_Q4;
    if (strcmp(s, "iq1") == 0) return ORNITH_QUANT_IQ1;
    return 0;
}

static int grow(void **ptr, size_t *cap, size_t elem_size)
{
    size_t next = *cap ? *cap * 2 : 128;
    void *p = realloc(*ptr, next * elem_size);
    if (!p) {
        return 0;
    }
    *ptr = p;
    *cap = next;
    return 1;
}

static int parse_shape(char *s, ornith_tensor_info *t)
{
    t->ndim = 0;
    for (;;) {
        if (t->ndim == ORNITH_MAX_DIMS) {
            return 0;
        }
        t->shape[t->ndim++] = i64(s);
        char *comma = strchr(s, ',');
        if (!comma) {
            return 1;
        }
        *comma = 0;
        s = comma + 1;
    }
}

static int add_shard(ornith_model *m, char **f, int n, char *err, size_t errcap)
{
    if (n != 6) {
        set_err(err, errcap, "bad shard row");
        return 0;
    }
    if (m->shard_count == m->shard_cap && !grow((void **)&m->shards, &m->shard_cap, sizeof(m->shards[0]))) {
        set_err(err, errcap, "out of memory");
        return 0;
    }
    ornith_shard_info *s = &m->shards[m->shard_count++];
    memset(s, 0, sizeof(*s));
    s->fd = -1;
    s->file = dupstr(f[1]);
    s->size = u64(f[2]);
    s->data_start = u64(f[3]);
    s->block_size = (uint32_t)u64(f[4]);
    s->tensor_count = (uint32_t)u64(f[5]);
    if (!s->file || !s->size || !s->data_start || !s->block_size) {
        set_err(err, errcap, "invalid shard row");
        return 0;
    }
    return 1;
}

static int add_tensor(ornith_model *m, char **f, int n, char *err, size_t errcap)
{
    if (n != 11) {
        set_err(err, errcap, "bad tensor row");
        return 0;
    }
    if (m->tensor_count == m->tensor_cap && !grow((void **)&m->tensors, &m->tensor_cap, sizeof(m->tensors[0]))) {
        set_err(err, errcap, "out of memory");
        return 0;
    }
    ornith_tensor_info *t = &m->tensors[m->tensor_count++];
    memset(t, 0, sizeof(*t));
    t->name = dupstr(f[1]);
    t->shard = dupstr(f[2]);
    t->quant = parse_quant(f[3]);
    t->payload_offset = u64(f[4]);
    t->nbytes = u64(f[5]);
    t->nparams = u64(f[6]);
    t->layer = i64(f[7]);
    t->group = dupstr(f[8]);
    t->kind = dupstr(f[9]);
    if (!t->name || !t->shard || !t->group || !t->kind || !t->quant || !t->nbytes || !t->nparams ||
        !parse_shape(f[10], t)) {
        set_err(err, errcap, "invalid tensor row");
        return 0;
    }
    return 1;
}

static int has_tensor(const ornith_model *m, const char *name)
{
    return ornith_model_find_tensor(m, name) != NULL;
}

static const ornith_shard_info *find_shard(const ornith_model *m, const char *file)
{
    for (size_t i = 0; i < m->shard_count; i++) {
        if (strcmp(m->shards[i].file, file) == 0) {
            return &m->shards[i];
        }
    }
    return NULL;
}

static int validate_tensor_ranges(const ornith_model *m, char *err, size_t errcap)
{
    for (size_t i = 0; i < m->tensor_count; i++) {
        const ornith_tensor_info *t = &m->tensors[i];
        const ornith_shard_info *s = find_shard(m, t->shard);
        if (!s || t->payload_offset < s->data_start || t->nbytes > s->size ||
            t->payload_offset > s->size - t->nbytes) {
            set_err(err, errcap, "tensor range outside shard");
            return 0;
        }
    }
    return 1;
}

int ornith_model_open(const char *catalog_tsv, const char *shard_dir, ornith_model **out, char *err, size_t errcap)
{
    FILE *fp = fopen(catalog_tsv, "r");
    if (!fp) {
        set_err(err, errcap, strerror(errno));
        return 0;
    }
    ornith_model *m = calloc(1, sizeof(*m));
    if (!m) {
        fclose(fp);
        set_err(err, errcap, "out of memory");
        return 0;
    }
    m->shard_dir = dupstr(shard_dir ? shard_dir : ".");

    char *line = NULL;
    size_t cap = 0;
    int saw_header = 0;
    while (getline(&line, &cap, fp) != -1) {
        char *fields[12];
        if (line[0] == '#') {
            saw_header = strstr(line, "ornith-runtime-catalog-tsv-v1") != NULL;
            continue;
        }
        int n = split_tab(line, fields, 12);
        if (n == 0 || fields[0][0] == 0) {
            continue;
        }
        if (strcmp(fields[0], "shard") == 0) {
            if (!add_shard(m, fields, n, err, errcap)) goto fail;
        } else if (strcmp(fields[0], "tensor") == 0) {
            if (!add_tensor(m, fields, n, err, errcap)) goto fail;
        } else {
            set_err(err, errcap, "unknown catalog row");
            goto fail;
        }
    }
    free(line);
    fclose(fp);
    if (!saw_header || !m->shard_count || !m->tensor_count) {
        set_err(err, errcap, "incomplete catalog");
        ornith_model_close(m);
        return 0;
    }
    if (!has_tensor(m, "model.language_model.embed_tokens.weight") ||
        !has_tensor(m, "model.language_model.norm.weight") ||
        !has_tensor(m, "lm_head.weight")) {
        set_err(err, errcap, "missing required global tensor");
        ornith_model_close(m);
        return 0;
    }
    if (!validate_tensor_ranges(m, err, errcap)) {
        ornith_model_close(m);
        return 0;
    }
    *out = m;
    return 1;

fail:
    free(line);
    fclose(fp);
    ornith_model_close(m);
    return 0;
}

void ornith_model_close(ornith_model *m)
{
    if (!m) return;
    for (size_t i = 0; i < m->shard_count; i++) {
        if (m->shards[i].map) {
            munmap((void *)m->shards[i].map, m->shards[i].size);
        }
        if (m->shards[i].fd >= 0) {
            close(m->shards[i].fd);
        }
        free(m->shards[i].file);
    }
    for (size_t i = 0; i < m->tensor_count; i++) {
        free(m->tensors[i].name);
        free(m->tensors[i].shard);
        free(m->tensors[i].group);
        free(m->tensors[i].kind);
    }
    free(m->shard_dir);
    free(m->shards);
    free(m->tensors);
    free(m);
}

size_t ornith_model_shard_count(const ornith_model *m)
{
    return m ? m->shard_count : 0;
}

size_t ornith_model_tensor_count(const ornith_model *m)
{
    return m ? m->tensor_count : 0;
}

size_t ornith_model_layer_count(const ornith_model *m)
{
    int64_t max_layer = -1;
    if (!m) return 0;
    for (size_t i = 0; i < m->tensor_count; i++) {
        if (m->tensors[i].layer > max_layer) {
            max_layer = m->tensors[i].layer;
        }
    }
    return (size_t)(max_layer + 1);
}

const ornith_tensor_info *ornith_model_find_tensor(const ornith_model *m, const char *name)
{
    if (!m) return NULL;
    for (size_t i = 0; i < m->tensor_count; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

const ornith_tensor_info *ornith_model_find_layer_tensor(const ornith_model *m, int64_t layer, const char *kind)
{
    if (!m) return NULL;
    for (size_t i = 0; i < m->tensor_count; i++) {
        if (m->tensors[i].layer == layer && strcmp(m->tensors[i].kind, kind) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

static int path_join(char *out, size_t outcap, const char *dir, const char *file)
{
    int n = snprintf(out, outcap, "%s%s%s", dir, dir[0] && dir[strlen(dir) - 1] == '/' ? "" : "/", file);
    return n > 0 && (size_t)n < outcap;
}

int ornith_model_validate_shards(const ornith_model *m, char *err, size_t errcap)
{
    static const unsigned char magic[8] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    if (!m) {
        set_err(err, errcap, "null model");
        return 0;
    }
    for (size_t i = 0; i < m->shard_count; i++) {
        char path[4096];
        if (!path_join(path, sizeof(path), m->shard_dir, m->shards[i].file)) {
            set_err(err, errcap, "path too long");
            return 0;
        }
        FILE *fp = fopen(path, "rb");
        if (!fp) {
            set_err(err, errcap, path);
            return 0;
        }
        unsigned char got[8];
        size_t nr = fread(got, 1, sizeof(got), fp);
        fclose(fp);
        struct stat st;
        if (nr != sizeof(got) || memcmp(got, magic, sizeof(magic)) != 0 || stat(path, &st) != 0 ||
            (uint64_t)st.st_size != m->shards[i].size) {
            set_err(err, errcap, path);
            return 0;
        }
    }
    return 1;
}

static int require_tensor(const ornith_model *m, int64_t layer, const char *kind, const ornith_tensor_info **out, char *err, size_t errcap)
{
    *out = ornith_model_find_layer_tensor(m, layer, kind);
    if (!*out) {
        char msg[256];
        snprintf(msg, sizeof(msg), "layer %lld missing %s", (long long)layer, kind);
        set_err(err, errcap, msg);
        return 0;
    }
    return 1;
}

int ornith_model_validate_moe_layout(const ornith_model *m, char *err, size_t errcap)
{
    size_t layers = ornith_model_layer_count(m);
    for (size_t layer = 0; layer < layers; layer++) {
        const ornith_tensor_info *norm = NULL;
        const ornith_tensor_info *router = NULL;
        const ornith_tensor_info *gate_up = NULL;
        const ornith_tensor_info *down = NULL;
        const ornith_tensor_info *sgate = NULL;
        const ornith_tensor_info *sup = NULL;
        const ornith_tensor_info *sdown = NULL;
        const ornith_tensor_info *srouter = NULL;
        int64_t l = (int64_t)layer;
        if (!require_tensor(m, l, "input_layernorm.weight", &norm, err, errcap) ||
            !require_tensor(m, l, "mlp.gate.weight", &router, err, errcap) ||
            !require_tensor(m, l, "mlp.experts.gate_up_proj", &gate_up, err, errcap) ||
            !require_tensor(m, l, "mlp.experts.down_proj", &down, err, errcap) ||
            !require_tensor(m, l, "mlp.shared_expert.gate_proj.weight", &sgate, err, errcap) ||
            !require_tensor(m, l, "mlp.shared_expert.up_proj.weight", &sup, err, errcap) ||
            !require_tensor(m, l, "mlp.shared_expert.down_proj.weight", &sdown, err, errcap) ||
            !require_tensor(m, l, "mlp.shared_expert_gate.weight", &srouter, err, errcap)) {
            return 0;
        }
        int64_t hidden = (int64_t)norm->nparams;
        if (norm->ndim != 1 || router->ndim != 2 || gate_up->ndim != 3 || down->ndim != 3 ||
            sgate->ndim != 2 || sup->ndim != 2 || sdown->ndim != 2 || srouter->ndim != 2 ||
            hidden <= 0 || router->shape[0] <= 0 || down->shape[2] <= 0 ||
            router->shape[1] != hidden || gate_up->shape[0] != router->shape[0] ||
            gate_up->shape[1] != down->shape[2] * 2 || gate_up->shape[2] != hidden ||
            down->shape[0] != router->shape[0] || down->shape[1] != hidden ||
            sgate->shape[0] != sup->shape[0] || sgate->shape[1] != hidden || sup->shape[1] != hidden ||
            sdown->shape[0] != hidden || sdown->shape[1] != sgate->shape[0] ||
            srouter->shape[0] != 1 || srouter->shape[1] != hidden) {
            char msg[128];
            snprintf(msg, sizeof(msg), "layer %zu has incompatible MoE shapes", layer);
            set_err(err, errcap, msg);
            return 0;
        }
    }
    return 1;
}

static int layer_has_linear_attention(const ornith_model *m, int64_t layer)
{
    return ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_qkv.weight") != NULL;
}

static int layer_has_self_attention(const ornith_model *m, int64_t layer)
{
    return ornith_model_find_layer_tensor(m, layer, "self_attn.q_proj.weight") != NULL;
}

int ornith_model_validate_attention_layout(const ornith_model *m, char *err, size_t errcap)
{
    size_t layers = ornith_model_layer_count(m);
    for (size_t layer = 0; layer < layers; layer++) {
        int64_t l = (int64_t)layer;
        const ornith_tensor_info *input_norm = NULL;
        const ornith_tensor_info *post_norm = NULL;
        if (!require_tensor(m, l, "input_layernorm.weight", &input_norm, err, errcap) ||
            !require_tensor(m, l, "post_attention_layernorm.weight", &post_norm, err, errcap)) {
            return 0;
        }
        int64_t hidden = (int64_t)input_norm->nparams;
        if (input_norm->ndim != 1 || post_norm->ndim != 1 || post_norm->nparams != (uint64_t)hidden) {
            char msg[128];
            snprintf(msg, sizeof(msg), "layer %zu has incompatible norm shapes", layer);
            set_err(err, errcap, msg);
            return 0;
        }

        if (layer_has_linear_attention(m, l)) {
            const ornith_tensor_info *qkv = ornith_model_find_layer_tensor(m, l, "linear_attn.in_proj_qkv.weight");
            const ornith_tensor_info *z = ornith_model_find_layer_tensor(m, l, "linear_attn.in_proj_z.weight");
            const ornith_tensor_info *out = ornith_model_find_layer_tensor(m, l, "linear_attn.out_proj.weight");
            const ornith_tensor_info *a = ornith_model_find_layer_tensor(m, l, "linear_attn.in_proj_a.weight");
            const ornith_tensor_info *b = ornith_model_find_layer_tensor(m, l, "linear_attn.in_proj_b.weight");
            const ornith_tensor_info *alog = ornith_model_find_layer_tensor(m, l, "linear_attn.A_log");
            const ornith_tensor_info *dt = ornith_model_find_layer_tensor(m, l, "linear_attn.dt_bias");
            const ornith_tensor_info *norm = ornith_model_find_layer_tensor(m, l, "linear_attn.norm.weight");
            const ornith_tensor_info *conv = ornith_model_find_layer_tensor(m, l, "linear_attn.conv1d.weight");
            if (!qkv || !z || !out || !a || !b || !alog || !dt || !norm || !conv ||
                qkv->ndim != 2 || z->ndim != 2 || out->ndim != 2 || a->ndim != 2 || b->ndim != 2 ||
                alog->ndim != 1 || dt->ndim != 1 || norm->ndim != 1 || conv->ndim != 3 ||
                qkv->shape[1] != hidden || qkv->shape[0] != hidden * 3 ||
                z->shape[1] != hidden || z->shape[0] != hidden * 2 ||
                out->shape[0] != hidden || out->shape[1] != z->shape[0] ||
                a->shape[1] != hidden || b->shape[1] != hidden || a->shape[0] != b->shape[0] ||
                alog->nparams != (uint64_t)a->shape[0] || dt->nparams != (uint64_t)a->shape[0] ||
                z->shape[0] % a->shape[0] != 0 || norm->nparams != (uint64_t)(z->shape[0] / a->shape[0]) ||
                conv->shape[0] != qkv->shape[0] || conv->shape[1] != 1 || conv->shape[2] <= 0) {
                char msg[128];
                snprintf(msg, sizeof(msg), "layer %zu has incompatible linear attention shapes", layer);
                set_err(err, errcap, msg);
                return 0;
            }
        } else if (layer_has_self_attention(m, l)) {
            const ornith_tensor_info *q = ornith_model_find_layer_tensor(m, l, "self_attn.q_proj.weight");
            const ornith_tensor_info *k = ornith_model_find_layer_tensor(m, l, "self_attn.k_proj.weight");
            const ornith_tensor_info *v = ornith_model_find_layer_tensor(m, l, "self_attn.v_proj.weight");
            const ornith_tensor_info *o = ornith_model_find_layer_tensor(m, l, "self_attn.o_proj.weight");
            const ornith_tensor_info *qn = ornith_model_find_layer_tensor(m, l, "self_attn.q_norm.weight");
            const ornith_tensor_info *kn = ornith_model_find_layer_tensor(m, l, "self_attn.k_norm.weight");
            if (!q || !k || !v || !o || !qn || !kn ||
                q->ndim != 2 || k->ndim != 2 || v->ndim != 2 || o->ndim != 2 || qn->ndim != 1 || kn->ndim != 1 ||
                q->shape[1] != hidden || k->shape[1] != hidden || v->shape[1] != hidden ||
                k->shape[0] != v->shape[0] || qn->nparams != kn->nparams || qn->nparams <= 0 ||
                q->shape[0] % (int64_t)qn->nparams != 0 || k->shape[0] % (int64_t)kn->nparams != 0 ||
                o->shape[0] != hidden || o->shape[1] <= 0) {
                char msg[128];
                snprintf(msg, sizeof(msg), "layer %zu has incompatible self attention shapes", layer);
                set_err(err, errcap, msg);
                return 0;
            }
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "layer %zu missing attention tensors", layer);
            set_err(err, errcap, msg);
            return 0;
        }
    }
    return 1;
}

int ornith_model_map_shards(ornith_model *m, char *err, size_t errcap)
{
    if (!ornith_model_validate_shards(m, err, errcap)) {
        return 0;
    }
    for (size_t i = 0; i < m->shard_count; i++) {
        if (m->shards[i].map) {
            continue;
        }
        char path[4096];
        if (!path_join(path, sizeof(path), m->shard_dir, m->shards[i].file)) {
            set_err(err, errcap, "path too long");
            return 0;
        }
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            set_err(err, errcap, path);
            return 0;
        }
        void *map = mmap(NULL, m->shards[i].size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) {
            close(fd);
            set_err(err, errcap, path);
            return 0;
        }
        m->shards[i].fd = fd;
        m->shards[i].map = map;
    }
    return 1;
}

static float bf16_at(const unsigned char *p)
{
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)p[0] | ((uint32_t)p[1] << 8)) << 16;
    return v.f;
}

static uint64_t block_bytes(ornith_quant quant, uint32_t block)
{
    if (quant == ORNITH_QUANT_IQ1) return 2 + (block + 7) / 8;
    if (quant == ORNITH_QUANT_Q4) return 2 + (block + 1) / 2;
    return 2;
}

static int slice_matvec_iq1(const unsigned char *payload, uint32_t block, uint64_t offset, size_t rows, size_t cols, const float *x, float *out);

static const ornith_shard_info *mapped_shard_for_tensor(const ornith_model *m, const ornith_tensor_info *t)
{
    const ornith_shard_info *s = find_shard(m, t->shard);
    return s && s->map ? s : NULL;
}

const unsigned char *ornith_tensor_payload(const ornith_model *m, const ornith_tensor_info *t, uint32_t *block_size)
{
    const ornith_shard_info *s = (!m || !t) ? NULL : mapped_shard_for_tensor(m, t);
    if (!s) return NULL;
    if (block_size) *block_size = s->block_size;
    return s->map + t->payload_offset;
}

const unsigned char *ornith_tensor_mapped_span(const ornith_model *m, const ornith_tensor_info *t, uint64_t *payload_offset, uint64_t *span_size, uint32_t *block_size)
{
    const ornith_shard_info *s = (!m || !t) ? NULL : mapped_shard_for_tensor(m, t);
    if (!s) return NULL;
    if (payload_offset) *payload_offset = t->payload_offset;
    if (span_size) *span_size = s->size;
    if (block_size) *block_size = s->block_size;
    return s->map;
}

static float tensor_payload_value(const unsigned char *payload, ornith_quant quant, uint32_t block, uint64_t i)
{
    if (quant == ORNITH_QUANT_BF16) {
        return bf16_at(payload + i * 2);
    }
    uint64_t block_idx = i / block;
    uint32_t in_block = (uint32_t)(i % block);
    const unsigned char *base = payload + block_idx * block_bytes(quant, block);
    float scale = bf16_at(base);
    if (quant == ORNITH_QUANT_IQ1) {
        unsigned char bits = base[2 + in_block / 8];
        return (bits & (1u << (in_block % 8))) ? scale : -scale;
    }
    unsigned char packed = base[2 + in_block / 2];
    int q = (in_block & 1) ? (packed >> 4) : (packed & 15);
    if (q >= 8) q -= 16;
    return scale * (float)q;
}

int ornith_tensor_value(const ornith_model *m, const ornith_tensor_info *t, uint64_t i, float *out)
{
    if (!m || !t || !out || i >= t->nparams) {
        return 0;
    }
    const ornith_shard_info *s = mapped_shard_for_tensor(m, t);
    if (!s) {
        return 0;
    }
    const unsigned char *payload = s->map + t->payload_offset;
    *out = tensor_payload_value(payload, t->quant, s->block_size, i);
    return 1;
}

static int matvec_bf16(const unsigned char *payload, size_t rows, size_t cols, const float *x, float *out)
{
    for (size_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        const unsigned char *row = payload + r * cols * 2;
        for (size_t c = 0; c < cols; c++) {
            acc += bf16_at(row + c * 2) * x[c];
        }
        out[r] = acc;
    }
    return 1;
}

static int matvec_q4(const unsigned char *payload, uint32_t block, size_t rows, size_t cols, const float *x, float *out)
{
    for (size_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        uint64_t row_base = (uint64_t)r * cols;
        for (size_t c = 0; c < cols;) {
            uint64_t i = row_base + c;
            uint32_t in_block = (uint32_t)(i % block);
            uint32_t take = block - in_block;
            if (take > cols - c) take = (uint32_t)(cols - c);
            const unsigned char *base = payload + (i / block) * block_bytes(ORNITH_QUANT_Q4, block);
            float scale = bf16_at(base);
            for (uint32_t j = 0; j < take; j++, c++) {
                uint32_t qidx = in_block + j;
                unsigned char packed = base[2 + qidx / 2];
                int q = (qidx & 1) ? (packed >> 4) : (packed & 15);
                if (q >= 8) q -= 16;
                acc += scale * (float)q * x[c];
            }
        }
        out[r] = acc;
    }
    return 1;
}

static int matvec_iq1(const unsigned char *payload, uint32_t block, size_t rows, size_t cols, const float *x, float *out)
{
    return slice_matvec_iq1(payload, block, 0, rows, cols, x, out);
}

int ornith_tensor_matvec(const ornith_model *m, const ornith_tensor_info *t, const float *x, size_t x_count, float *out)
{
    if (!m || !t || !x || !out || t->ndim != 2 || x_count != (size_t)t->shape[1]) {
        return 0;
    }
    const ornith_shard_info *s = mapped_shard_for_tensor(m, t);
    if (!s) return 0;
    const unsigned char *payload = s->map + t->payload_offset;
    size_t rows = (size_t)t->shape[0];
    size_t cols = (size_t)t->shape[1];
    if (t->quant == ORNITH_QUANT_BF16) return matvec_bf16(payload, rows, cols, x, out);
    if (t->quant == ORNITH_QUANT_Q4) return matvec_q4(payload, s->block_size, rows, cols, x, out);
    if (t->quant == ORNITH_QUANT_IQ1) return matvec_iq1(payload, s->block_size, rows, cols, x, out);
    for (size_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float v = 0.0f;
            if (!ornith_tensor_value(m, t, r * cols + c, &v)) {
                return 0;
            }
            acc += v * x[c];
        }
        out[r] = acc;
    }
    return 1;
}

static int slice_matvec_iq1(const unsigned char *payload, uint32_t block, uint64_t offset, size_t rows, size_t cols, const float *x, float *out)
{
    for (size_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        uint64_t row_base = offset + (uint64_t)r * cols;
        for (size_t c = 0; c < cols;) {
            uint64_t i = row_base + c;
            uint32_t in_block = (uint32_t)(i % block);
            uint32_t take = block - in_block;
            if (take > cols - c) take = (uint32_t)(cols - c);
            const unsigned char *base = payload + (i / block) * block_bytes(ORNITH_QUANT_IQ1, block);
            float scale = bf16_at(base);
            for (uint32_t j = 0; j < take; j++, c++) {
                uint32_t b = in_block + j;
                unsigned char bits = base[2 + b / 8];
                acc += ((bits & (1u << (b % 8))) ? scale : -scale) * x[c];
            }
        }
        out[r] = acc;
    }
    return 1;
}

int ornith_tensor_slice_matvec(const ornith_model *m, const ornith_tensor_info *t, uint64_t slice, const float *x, size_t x_count, float *out)
{
    if (!m || !t || !x || !out || t->ndim != 3 || slice >= (uint64_t)t->shape[0] || x_count != (size_t)t->shape[2]) {
        return 0;
    }
    const ornith_shard_info *s = mapped_shard_for_tensor(m, t);
    if (!s) return 0;
    const unsigned char *payload = s->map + t->payload_offset;
    size_t rows = (size_t)t->shape[1];
    size_t cols = (size_t)t->shape[2];
    uint64_t base = slice * rows * cols;
    if (t->quant == ORNITH_QUANT_BF16) return matvec_bf16(payload + base * 2, rows, cols, x, out);
    if (t->quant == ORNITH_QUANT_IQ1) return slice_matvec_iq1(payload, s->block_size, base, rows, cols, x, out);
    for (size_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float v = 0.0f;
            if (!ornith_tensor_value(m, t, base + r * cols + c, &v)) {
                return 0;
            }
            acc += v * x[c];
        }
        out[r] = acc;
    }
    return 1;
}

int ornith_rmsnorm(const ornith_model *m, const ornith_tensor_info *weight, const float *x, size_t n, float eps, float *out)
{
    if (!m || !weight || !x || !out || weight->nparams != n) {
        return 0;
    }
    float mean_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        mean_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(mean_sq / (float)n + eps);
    const ornith_shard_info *s = mapped_shard_for_tensor(m, weight);
    if (s && weight->quant == ORNITH_QUANT_BF16) {
        const unsigned char *payload = s->map + weight->payload_offset;
        for (size_t i = 0; i < n; i++) {
            out[i] = x[i] * scale * (1.0f + bf16_at(payload + i * 2));
        }
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        float w = 0.0f;
        if (!ornith_tensor_value(m, weight, i, &w)) {
            return 0;
        }
        out[i] = x[i] * scale * (1.0f + w);
    }
    return 1;
}

static float sigmoidf_local(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float siluf(float x)
{
    return x * sigmoidf_local(x);
}

static float softplusf_local(float x)
{
    return x <= 20.0f ? log1pf(expf(x)) : x;
}

static int softmax_selected(float *values, size_t n)
{
    if (!values || !n) return 0;
    float maxv = values[0];
    for (size_t i = 1; i < n; i++) {
        if (values[i] > maxv) maxv = values[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        values[i] = expf(values[i] - maxv);
        sum += values[i];
    }
    if (sum == 0.0f) return 0;
    for (size_t i = 0; i < n; i++) {
        values[i] /= sum;
    }
    return 1;
}

static int add_shared_expert(
    const ornith_model *m,
    int64_t layer,
    const float *norm,
    size_t hidden,
    float *out)
{
    const ornith_tensor_info *gate = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.gate_proj.weight");
    const ornith_tensor_info *up = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.up_proj.weight");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.down_proj.weight");
    const ornith_tensor_info *sgate = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert_gate.weight");
    if (!gate || !up || !down || !sgate || gate->ndim != 2 || up->ndim != 2 || down->ndim != 2 ||
        sgate->ndim != 2 || sgate->shape[0] != 1 || sgate->shape[1] != (int64_t)hidden ||
        gate->shape[0] != up->shape[0] || gate->shape[1] != (int64_t)hidden || up->shape[1] != (int64_t)hidden ||
        down->shape[0] != (int64_t)hidden || down->shape[1] != gate->shape[0] || gate->shape[0] <= 0) {
        return 0;
    }
    size_t inter = (size_t)gate->shape[0];
    float *g = calloc(inter * 3 + hidden + 1, sizeof(float));
    if (!g) return 0;
    float *u = g + inter;
    float *mid = u + inter;
    float *tmp = mid + inter;
    float s = 0.0f;
    int ok = ornith_tensor_matvec(m, gate, norm, hidden, g) &&
             ornith_tensor_matvec(m, up, norm, hidden, u) &&
             ornith_tensor_matvec(m, sgate, norm, hidden, &s);
    if (ok) {
        for (size_t i = 0; i < inter; i++) {
            mid[i] = siluf(g[i]) * u[i];
        }
        ok = ornith_tensor_matvec(m, down, mid, inter, tmp);
    }
    if (ok) {
        float w = sigmoidf_local(s);
        for (size_t i = 0; i < hidden; i++) {
            out[i] += w * tmp[i];
        }
    }
    free(g);
    return ok;
}

static int layer_moe_smoke_with_norm(const ornith_model *m, int64_t layer, const char *norm_kind, const float *x, size_t hidden, size_t top_k, float *out)
{
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, norm_kind);
    const ornith_tensor_info *router = ornith_model_find_layer_tensor(m, layer, "mlp.gate.weight");
    const ornith_tensor_info *gate_up = ornith_model_find_layer_tensor(m, layer, "mlp.experts.gate_up_proj");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.experts.down_proj");
    if (!m || !x || !out || !norm_w || !router || !gate_up || !down || router->ndim != 2 ||
        gate_up->ndim != 3 || down->ndim != 3 || router->shape[1] != (int64_t)hidden ||
        gate_up->shape[0] != router->shape[0] || gate_up->shape[2] != (int64_t)hidden ||
        down->shape[0] != router->shape[0] || down->shape[1] != (int64_t)hidden ||
        gate_up->shape[1] != down->shape[2] * 2 || router->shape[0] <= 0 || down->shape[2] <= 0 ||
        top_k == 0 || top_k > (size_t)router->shape[0]) {
        return 0;
    }
    size_t experts = (size_t)router->shape[0];
    size_t inter = (size_t)down->shape[2];
    float *scratch = calloc(hidden * 3 + experts + top_k + gate_up->shape[1] + inter, sizeof(float));
    size_t *idx = calloc(top_k, sizeof(size_t));
    if (!scratch || !idx) {
        free(scratch);
        free(idx);
        return 0;
    }
    float *norm = scratch;
    float *scores = norm + hidden;
    float *weights = scores + experts;
    float *gate_up_out = weights + top_k;
    float *mid = gate_up_out + gate_up->shape[1];
    float *tmp = mid + inter;

    memset(out, 0, hidden * sizeof(float));
    int ok = ornith_rmsnorm(m, norm_w, x, hidden, 1e-6f, norm) &&
             ornith_tensor_matvec(m, router, norm, hidden, scores) &&
             ornith_topk(scores, experts, top_k, idx, weights) &&
             softmax_selected(weights, top_k);
    for (size_t k = 0; ok && k < top_k; k++) {
        ok = ornith_tensor_slice_matvec(m, gate_up, idx[k], norm, hidden, gate_up_out);
        if (!ok) break;
        for (size_t i = 0; i < inter; i++) {
            mid[i] = siluf(gate_up_out[i]) * gate_up_out[i + inter];
        }
        ok = ornith_tensor_slice_matvec(m, down, idx[k], mid, inter, tmp);
        for (size_t i = 0; ok && i < hidden; i++) {
            out[i] += weights[k] * tmp[i];
        }
    }
    ok = ok && add_shared_expert(m, layer, norm, hidden, out);
    free(idx);
    free(scratch);
    return ok;
}

static int tensor_matvec_hooked(const ornith_model *m, const ornith_tensor_info *t, const float *x, size_t x_count, float *out, ornith_tensor_matvec_fn matvec_hook, void *hook_ctx);

static int tensor_matvec_batch_hooked(const ornith_model *m, const ornith_tensor_info * const *tensors, size_t count, const float *x, size_t x_count, float **outs, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, void *hook_ctx);

static int self_attention_first_token_hooked(const ornith_model *m, int64_t layer, const float *x, size_t hidden, float *out, ornith_tensor_matvec_fn matvec_hook, void *hook_ctx)
{
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, "input_layernorm.weight");
    const ornith_tensor_info *v_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.v_proj.weight");
    const ornith_tensor_info *o_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.o_proj.weight");
    if (!norm_w || !v_proj || !o_proj || v_proj->ndim != 2 || o_proj->ndim != 2 ||
        v_proj->shape[1] != (int64_t)hidden || o_proj->shape[0] != (int64_t)hidden ||
        v_proj->shape[0] <= 0 || o_proj->shape[1] <= 0 || o_proj->shape[1] % v_proj->shape[0] != 0) {
        return 0;
    }
    size_t kv = (size_t)v_proj->shape[0];
    size_t out_cols = (size_t)o_proj->shape[1];
    float *buf = malloc((hidden + kv + out_cols) * sizeof(float));
    if (!buf) return 0;
    float *norm = buf;
    float *v = norm + hidden;
    float *expanded = v + kv;
    int ok = ornith_rmsnorm(m, norm_w, x, hidden, 1e-6f, norm) &&
             tensor_matvec_hooked(m, v_proj, norm, hidden, v, matvec_hook, hook_ctx);
    for (size_t i = 0; ok && i < out_cols; i++) {
        expanded[i] = v[i % kv];
    }
    ok = ok && tensor_matvec_hooked(m, o_proj, expanded, out_cols, out, matvec_hook, hook_ctx);
    free(buf);
    return ok;
}

static void full_state_free(ornith_full_state *s)
{
    if (!s) return;
    free(s->k);
    free(s->v);
    free(s->scores);
    memset(s, 0, sizeof(*s));
}

static int full_state_init(const ornith_model *m, int64_t layer, size_t token_cap, ornith_full_state *s)
{
    const ornith_tensor_info *q_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.q_proj.weight");
    const ornith_tensor_info *k_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.k_proj.weight");
    const ornith_tensor_info *v_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.v_proj.weight");
    const ornith_tensor_info *o_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.o_proj.weight");
    const ornith_tensor_info *q_norm = ornith_model_find_layer_tensor(m, layer, "self_attn.q_norm.weight");
    if (!q_proj || !k_proj || !v_proj || !o_proj || !q_norm || q_proj->ndim != 2 || k_proj->ndim != 2 ||
        v_proj->ndim != 2 || o_proj->ndim != 2 || q_norm->ndim != 1 || !token_cap ||
        o_proj->shape[1] <= 0 || q_norm->nparams == 0 || o_proj->shape[1] % (int64_t)q_norm->nparams != 0 ||
        k_proj->shape[0] != v_proj->shape[0] || k_proj->shape[0] % (int64_t)q_norm->nparams != 0) {
        return 0;
    }
    size_t q_size = (size_t)o_proj->shape[1];
    if ((size_t)q_proj->shape[0] != q_size && (size_t)q_proj->shape[0] != q_size * 2) {
        return 0;
    }
    s->token_cap = token_cap;
    s->q_heads = q_size / (size_t)q_norm->nparams;
    s->head_dim = (size_t)q_norm->nparams;
    s->kv_dim = (size_t)k_proj->shape[0];
    s->kv_heads = s->kv_dim / s->head_dim;
    if (!s->q_heads || !s->kv_heads || s->q_heads % s->kv_heads != 0) {
        return 0;
    }
    s->k = calloc(token_cap * s->kv_dim, sizeof(float));
    s->v = calloc(token_cap * s->kv_dim, sizeof(float));
    s->scores = malloc(token_cap * sizeof(float));
    if (!s->k || !s->v || !s->scores) {
        full_state_free(s);
        return 0;
    }
    return 1;
}

static int rmsnorm_head(const ornith_model *m, const ornith_tensor_info *w, float *x, size_t n)
{
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + 1e-6f);
    uint32_t block = 0;
    const unsigned char *payload = ornith_tensor_payload(m, w, &block);
    if (!payload) return 0;
    for (size_t i = 0; i < n; i++) {
        x[i] *= scale * (1.0f + tensor_payload_value(payload, w->quant, block, i));
    }
    return 1;
}

static int tensor_matvec_hooked(const ornith_model *m, const ornith_tensor_info *t, const float *x, size_t x_count, float *out, ornith_tensor_matvec_fn matvec_hook, void *hook_ctx)
{
    return matvec_hook ? matvec_hook(m, t, x, x_count, out, hook_ctx) : ornith_tensor_matvec(m, t, x, x_count, out);
}

static int tensor_matvec_batch_hooked(const ornith_model *m, const ornith_tensor_info * const *tensors, size_t count, const float *x, size_t x_count, float **outs, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, void *hook_ctx)
{
    if (batch_hook) {
        return batch_hook(m, tensors, count, x, x_count, outs, hook_ctx);
    }
    for (size_t i = 0; i < count; i++) {
        if (!tensor_matvec_hooked(m, tensors[i], x, x_count, outs[i], matvec_hook, hook_ctx)) {
            return 0;
        }
    }
    return 1;
}

static void apply_text_rope(float *x, size_t head_dim, size_t pos)
{
    size_t rotary = head_dim / 4;
    if (rotary < 2) return;
    rotary &= ~(size_t)1;
    for (size_t i = 0; i < rotary / 2; i++) {
        double theta = pow(10000000.0, -(double)(2 * i) / (double)rotary);
        float c = (float)cos((double)pos * theta);
        float s = (float)sin((double)pos * theta);
        float a = x[i];
        float b = x[i + rotary / 2];
        x[i] = a * c - b * s;
        x[i + rotary / 2] = b * c + a * s;
    }
}

static int unpack_attention_q_gate_interleaved(const float *mixed, size_t heads, size_t head_dim, float *q, float *gate);

static int self_attention_step_hooked(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t pos, ornith_full_state *state, float *out, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, ornith_self_attention_fn self_attn_hook, void *hook_ctx)
{
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, "input_layernorm.weight");
    const ornith_tensor_info *q_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.q_proj.weight");
    const ornith_tensor_info *k_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.k_proj.weight");
    const ornith_tensor_info *v_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.v_proj.weight");
    const ornith_tensor_info *o_proj = ornith_model_find_layer_tensor(m, layer, "self_attn.o_proj.weight");
    const ornith_tensor_info *q_norm = ornith_model_find_layer_tensor(m, layer, "self_attn.q_norm.weight");
    const ornith_tensor_info *k_norm = ornith_model_find_layer_tensor(m, layer, "self_attn.k_norm.weight");
    if (!state || !norm_w || !q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm ||
        q_proj->ndim != 2 || k_proj->ndim != 2 || v_proj->ndim != 2 || o_proj->ndim != 2 ||
        q_proj->shape[1] != (int64_t)hidden || k_proj->shape[1] != (int64_t)hidden ||
        v_proj->shape[1] != (int64_t)hidden || o_proj->shape[0] != (int64_t)hidden ||
        state->token_count >= state->token_cap) {
        return 0;
    }
    size_t q_size = state->q_heads * state->head_dim;
    size_t q_rows = (size_t)q_proj->shape[0];
    size_t scratch_n = hidden + q_rows + state->kv_dim * 2 + q_size * 3;
    float *scratch = malloc(scratch_n * sizeof(float));
    if (!scratch) return 0;
    float *norm = scratch;
    float *q_raw = norm + hidden;
    float *k = q_raw + q_rows;
    float *v = k + state->kv_dim;
    float *q_all = v + state->kv_dim;
    float *attn = q_all + q_size;
    float *gate = attn + q_size;
    const ornith_tensor_info *proj_tensors[3] = { q_proj, k_proj, v_proj };
    float *proj_outs[3] = { q_raw, k, v };
    int ok = ornith_rmsnorm(m, norm_w, x, hidden, 1e-6f, norm);
    if (ok && self_attn_hook) {
        int handled = self_attn_hook(m, layer, norm, hidden, state->k, state->v, &state->token_count, state->token_cap, state->q_heads, state->kv_heads, state->head_dim, q_rows, pos, out, hook_ctx);
        if (handled >= 0) {
            free(scratch);
            return handled;
        }
    }
    ok = ok && tensor_matvec_batch_hooked(m, proj_tensors, 3, norm, hidden, proj_outs, matvec_hook, batch_hook, hook_ctx);
    if (ok) {
        if (q_rows == q_size * 2) {
            ok = unpack_attention_q_gate_interleaved(q_raw, state->q_heads, state->head_dim, q_all, gate);
        } else {
            memcpy(q_all, q_raw, q_size * sizeof(float));
        }
    }
    for (size_t h = 0; ok && h < state->q_heads; h++) {
        ok = rmsnorm_head(m, q_norm, q_all + h * state->head_dim, state->head_dim);
        apply_text_rope(q_all + h * state->head_dim, state->head_dim, pos);
    }
    for (size_t h = 0; ok && h < state->kv_heads; h++) {
        ok = rmsnorm_head(m, k_norm, k + h * state->head_dim, state->head_dim);
        apply_text_rope(k + h * state->head_dim, state->head_dim, pos);
    }
    if (ok) {
        memcpy(state->k + state->token_count * state->kv_dim, k, state->kv_dim * sizeof(float));
        memcpy(state->v + state->token_count * state->kv_dim, v, state->kv_dim * sizeof(float));
        state->token_count++;
    }
    for (size_t qh = 0; ok && qh < state->q_heads; qh++) {
        size_t kvh = qh / (state->q_heads / state->kv_heads);
        const float *q = q_all + qh * state->head_dim;
        float *head_out = attn + qh * state->head_dim;
        float *scores = state->scores;
        float maxv = -INFINITY;
        for (size_t t = 0; t < state->token_count; t++) {
            const float *kk = state->k + t * state->kv_dim + kvh * state->head_dim;
            float dot = 0.0f;
            for (size_t i = 0; i < state->head_dim; i++) dot += q[i] * kk[i];
            scores[t] = dot / sqrtf((float)state->head_dim);
            if (scores[t] > maxv) maxv = scores[t];
        }
        float sum = 0.0f;
        for (size_t t = 0; t < state->token_count; t++) {
            scores[t] = expf(scores[t] - maxv);
            sum += scores[t];
        }
        memset(head_out, 0, state->head_dim * sizeof(float));
        for (size_t t = 0; sum != 0.0f && t < state->token_count; t++) {
            const float *vv = state->v + t * state->kv_dim + kvh * state->head_dim;
            float w = scores[t] / sum;
            for (size_t i = 0; i < state->head_dim; i++) head_out[i] += w * vv[i];
        }
        if (q_rows == q_size * 2) {
            for (size_t i = 0; i < state->head_dim; i++) {
                head_out[i] *= sigmoidf_local(gate[qh * state->head_dim + i]);
            }
        }
    }
    ok = ok && tensor_matvec_hooked(m, o_proj, attn, q_size, out, matvec_hook, hook_ctx);
    free(scratch);
    return ok;
}

static void linear_state_free(ornith_linear_state *s)
{
    if (!s) return;
    free(s->conv);
    free(s->conv_w);
    free(s->alog);
    free(s->dt);
    free(s->gated_norm);
    free(s->ssm);
    memset(s, 0, sizeof(*s));
}

static int linear_state_init(const ornith_model *m, int64_t layer, int predecode, ornith_linear_state *s)
{
    const ornith_tensor_info *qkv_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_qkv.weight");
    const ornith_tensor_info *z_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_z.weight");
    const ornith_tensor_info *b_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_b.weight");
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.norm.weight");
    const ornith_tensor_info *alog_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.A_log");
    const ornith_tensor_info *dt_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.dt_bias");
    const ornith_tensor_info *conv_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.conv1d.weight");
    if (!qkv_w || !z_w || !b_w || !norm_w || !alog_w || !dt_w || !conv_w || qkv_w->ndim != 2 || z_w->ndim != 2 ||
        b_w->ndim != 2 || norm_w->ndim != 1 || alog_w->ndim != 1 || dt_w->ndim != 1 || conv_w->ndim != 3 || conv_w->shape[2] < 1) {
        return 0;
    }
    size_t value_dim = (size_t)z_w->shape[0];
    s->qkv_dim = (size_t)qkv_w->shape[0];
    s->value_heads = (size_t)b_w->shape[0];
    s->head_v = (size_t)norm_w->nparams;
    s->conv_width = (size_t)conv_w->shape[2];
    if (!s->value_heads || !s->head_v || value_dim != s->value_heads * s->head_v ||
        alog_w->nparams != (uint64_t)s->value_heads || dt_w->nparams != (uint64_t)s->value_heads ||
        s->value_heads % 4 != 0 || s->qkv_dim <= value_dim) {
        return 0;
    }
    s->key_heads = s->value_heads / 4;
    size_t key_dim = (s->qkv_dim - value_dim) / 2;
    if (!s->key_heads || key_dim * 2 + value_dim != s->qkv_dim || key_dim % s->key_heads != 0) {
        return 0;
    }
    s->head_k = key_dim / s->key_heads;
    if (s->conv_width > 1) {
        s->conv = calloc(s->qkv_dim * (s->conv_width - 1), sizeof(float));
    }
    if (predecode) {
        s->conv_w = malloc(s->qkv_dim * s->conv_width * sizeof(float));
        s->alog = malloc(s->value_heads * sizeof(float));
        s->dt = malloc(s->value_heads * sizeof(float));
        s->gated_norm = malloc(s->head_v * sizeof(float));
    }
    s->ssm = calloc(s->value_heads * s->head_v * s->head_k, sizeof(float));
    if ((s->conv_width > 1 && !s->conv) || (predecode && (!s->conv_w || !s->alog || !s->dt || !s->gated_norm)) || !s->ssm) {
        linear_state_free(s);
        return 0;
    }
    if (predecode) {
        uint32_t conv_block = 0, alog_block = 0, dt_block = 0, norm_block = 0;
        const unsigned char *conv_payload = ornith_tensor_payload(m, conv_w, &conv_block);
        const unsigned char *alog_payload = ornith_tensor_payload(m, alog_w, &alog_block);
        const unsigned char *dt_payload = ornith_tensor_payload(m, dt_w, &dt_block);
        const unsigned char *norm_payload = ornith_tensor_payload(m, norm_w, &norm_block);
        if (!conv_payload || !alog_payload || !dt_payload || !norm_payload) {
            linear_state_free(s);
            return 0;
        }
        for (size_t i = 0; i < s->qkv_dim * s->conv_width; i++) {
            s->conv_w[i] = tensor_payload_value(conv_payload, conv_w->quant, conv_block, i);
        }
        for (size_t i = 0; i < s->value_heads; i++) {
            s->alog[i] = tensor_payload_value(alog_payload, alog_w->quant, alog_block, i);
            s->dt[i] = tensor_payload_value(dt_payload, dt_w->quant, dt_block, i);
        }
        for (size_t i = 0; i < s->head_v; i++) {
            s->gated_norm[i] = tensor_payload_value(norm_payload, norm_w->quant, norm_block, i);
        }
    }
    return 1;
}

static int unpack_attention_q_gate_interleaved(const float *mixed, size_t heads, size_t head_dim, float *q, float *gate)
{
    if (!mixed || !q || !gate || !heads || !head_dim) {
        return 0;
    }
    for (size_t h = 0; h < heads; h++) {
        const float *src = mixed + h * head_dim * 2;
        memcpy(q + h * head_dim, src, head_dim * sizeof(float));
        memcpy(gate + h * head_dim, src + head_dim, head_dim * sizeof(float));
    }
    return 1;
}

#ifdef ORNITH_TESTING
int ornith_test_unpack_attention_q_gate_interleaved(const float *mixed, size_t heads, size_t head_dim, float *q, float *gate)
{
    return unpack_attention_q_gate_interleaved(mixed, heads, head_dim, q, gate);
}
#endif

static int linear_attention_step_hooked(const ornith_model *m, int64_t layer, const float *x, size_t hidden, ornith_linear_state *state, float *out, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, void *hook_ctx)
{
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, "input_layernorm.weight");
    const ornith_tensor_info *qkv_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_qkv.weight");
    const ornith_tensor_info *z_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_z.weight");
    const ornith_tensor_info *a_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_a.weight");
    const ornith_tensor_info *b_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.in_proj_b.weight");
    const ornith_tensor_info *alog_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.A_log");
    const ornith_tensor_info *dt_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.dt_bias");
    const ornith_tensor_info *conv_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.conv1d.weight");
    const ornith_tensor_info *gated_norm_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.norm.weight");
    const ornith_tensor_info *out_w = ornith_model_find_layer_tensor(m, layer, "linear_attn.out_proj.weight");
    if (!norm_w || !qkv_w || !z_w || !a_w || !b_w || !alog_w || !dt_w || !conv_w || !gated_norm_w || !out_w ||
        qkv_w->ndim != 2 || z_w->ndim != 2 || a_w->ndim != 2 || b_w->ndim != 2 || conv_w->ndim != 3 ||
        gated_norm_w->ndim != 1 || out_w->ndim != 2 || qkv_w->shape[1] != (int64_t)hidden ||
        z_w->shape[1] != (int64_t)hidden || a_w->shape[1] != (int64_t)hidden || b_w->shape[1] != (int64_t)hidden ||
        out_w->shape[0] != (int64_t)hidden || out_w->shape[1] != z_w->shape[0] ||
        conv_w->shape[0] != qkv_w->shape[0] || conv_w->shape[1] != 1 ||
        a_w->shape[0] != b_w->shape[0] || alog_w->nparams != (uint64_t)b_w->shape[0] ||
        dt_w->nparams != (uint64_t)b_w->shape[0]) {
        return 0;
    }
    size_t value_heads = (size_t)b_w->shape[0];
    size_t head_v = (size_t)gated_norm_w->nparams;
    size_t value_dim = (size_t)z_w->shape[0];
    if (!value_heads || !head_v || value_dim != value_heads * head_v ||
        value_heads % 4 != 0 || (size_t)qkv_w->shape[0] <= value_dim) {
        return 0;
    }
    size_t key_heads = value_heads / 4;
    size_t key_dim = ((size_t)qkv_w->shape[0] - value_dim) / 2;
    if (!key_heads || key_dim * 2 + value_dim != (size_t)qkv_w->shape[0] || key_dim % key_heads != 0) {
        return 0;
    }
    size_t head_k = key_dim / key_heads;
    size_t conv_width = (size_t)conv_w->shape[2];
    if (state && (state->qkv_dim != (size_t)qkv_w->shape[0] || state->value_heads != value_heads ||
        state->head_v != head_v || state->key_heads != key_heads || state->head_k != head_k ||
        state->conv_width != conv_width)) {
        return 0;
    }
    size_t qkv_dim = (size_t)qkv_w->shape[0];
    size_t scratch_n = hidden + qkv_dim * 2 + value_dim * 3 + value_heads * 4 + head_v;
    float *scratch = malloc(scratch_n * sizeof(float));
    if (!scratch) return 0;
    float *norm = scratch;
    float *raw_qkv = norm + hidden;
    float *qkv = raw_qkv + qkv_dim;
    float *z = qkv + qkv_dim;
    float *beta_in = z + value_dim;
    float *a_in = beta_in + value_heads;
    float *core = a_in + value_heads;
    float *gated = core + value_dim;
    float *alog = gated + value_dim;
    float *dt = alog + value_heads;
    float *gated_norm = dt + value_heads;
    const ornith_tensor_info *proj_tensors[4] = { qkv_w, z_w, b_w, a_w };
    float *proj_outs[4] = { raw_qkv, z, beta_in, a_in };
    int ok = ornith_rmsnorm(m, norm_w, x, hidden, 1e-6f, norm);
    if (ok && linear_attn_hook && state && state->conv_w && state->alog && state->dt && state->gated_norm) {
        int handled = linear_attn_hook(m, layer, norm, hidden, state->conv, state->conv_w, state->ssm, state->alog, state->dt, state->gated_norm, qkv_dim, value_heads, head_v, key_heads, head_k, conv_width, out, hook_ctx);
        if (handled >= 0) {
            free(scratch);
            return handled;
        }
    }
    ok = ok && tensor_matvec_batch_hooked(m, proj_tensors, 4, norm, hidden, proj_outs, matvec_hook, batch_hook, hook_ctx);
    uint32_t conv_block = 0;
    const unsigned char *conv_payload = ok ? ornith_tensor_payload(m, conv_w, &conv_block) : NULL;
    const float *conv_weights = state ? state->conv_w : NULL;
    ok = ok && (conv_weights || conv_payload);

    for (size_t i = 0; ok && i < qkv_dim; i++) {
        float acc = 0.0f;
        for (size_t j = 0; j + 1 < conv_width; j++) {
            size_t wi = i * conv_width + j;
            float w = conv_weights ? conv_weights[wi] : tensor_payload_value(conv_payload, conv_w->quant, conv_block, wi);
            acc += (state ? state->conv[i * (conv_width - 1) + j] : 0.0f) * w;
        }
        size_t wi = i * conv_width + (conv_width - 1);
        float w = conv_weights ? conv_weights[wi] : tensor_payload_value(conv_payload, conv_w->quant, conv_block, wi);
        qkv[i] = siluf(acc + raw_qkv[i] * w);
    }
    if (ok && state && conv_width > 1) {
        for (size_t i = 0; i < qkv_dim; i++) {
            float *s = state->conv + i * (conv_width - 1);
            memmove(s, s + 1, (conv_width - 2) * sizeof(float));
            s[conv_width - 2] = raw_qkv[i];
        }
    }
    for (size_t h = 0; ok && h < key_heads; h++) {
        float qss = 0.0f;
        float kss = 0.0f;
        float *q = qkv + h * head_k;
        float *k = qkv + key_dim + h * head_k;
        for (size_t i = 0; i < head_k; i++) {
            qss += q[i] * q[i];
            kss += k[i] * k[i];
        }
        qss = 1.0f / sqrtf(qss + 1e-6f);
        kss = 1.0f / sqrtf(kss + 1e-6f);
        for (size_t i = 0; i < head_k; i++) {
            q[i] *= qss;
            k[i] *= kss;
        }
    }
    int use_gdn_hook = gdn_hook && state;
    int gdn_wrote_out = 0;
    if (ok && use_gdn_hook) {
        if (state->alog && state->dt && state->gated_norm) {
            memcpy(alog, state->alog, value_heads * sizeof(float));
            memcpy(dt, state->dt, value_heads * sizeof(float));
            memcpy(gated_norm, state->gated_norm, head_v * sizeof(float));
        } else {
            uint32_t alog_block = 0, dt_block = 0, gated_norm_block = 0;
            const unsigned char *alog_payload = ornith_tensor_payload(m, alog_w, &alog_block);
            const unsigned char *dt_payload = ornith_tensor_payload(m, dt_w, &dt_block);
            const unsigned char *gated_norm_payload = ornith_tensor_payload(m, gated_norm_w, &gated_norm_block);
            ok = alog_payload && dt_payload && gated_norm_payload;
            for (size_t hv = 0; ok && hv < value_heads; hv++) {
                alog[hv] = tensor_payload_value(alog_payload, alog_w->quant, alog_block, hv);
                dt[hv] = tensor_payload_value(dt_payload, dt_w->quant, dt_block, hv);
            }
            for (size_t i = 0; ok && i < head_v; i++) {
                gated_norm[i] = tensor_payload_value(gated_norm_payload, gated_norm_w->quant, gated_norm_block, i);
            }
        }
        if (ok) {
            int gdn_ok = gdn_hook(qkv, z, a_in, beta_in, alog, dt, gated_norm, state->ssm, value_heads, head_v, key_heads, head_k, gated, m, out_w, out, hook_ctx);
            ok = gdn_ok != 0;
            gdn_wrote_out = gdn_ok == 2;
        }
    }
    for (size_t hv = 0; ok && !use_gdn_hook && hv < value_heads; hv++) {
        size_t h = hv / 4;
        const float *q = qkv + h * head_k;
        const float *k = qkv + key_dim + h * head_k;
        const float *v = qkv + key_dim * 2 + hv * head_v;
        float beta = sigmoidf_local(beta_in[hv]);
        if (state) {
            float alog = 0.0f;
            float dt = 0.0f;
            ok = ornith_tensor_value(m, alog_w, hv, &alog) && ornith_tensor_value(m, dt_w, hv, &dt);
            float decay = ok ? expf(-expf(alog) * softplusf_local(a_in[hv] + dt)) : 0.0f;
            float *hs = state->ssm + hv * head_v * head_k;
            for (size_t vi = 0; ok && vi < head_v; vi++) {
                float *row = hs + vi * head_k;
                float proj = 0.0f;
                for (size_t ki = 0; ki < head_k; ki++) {
                    row[ki] *= decay;
                    proj += row[ki] * k[ki];
                }
                float vv = (v[vi] - proj) * beta;
                for (size_t ki = 0; ki < head_k; ki++) {
                    row[ki] += vv * k[ki];
                }
                float sum = 0.0f;
                for (size_t ki = 0; ki < head_k; ki++) {
                    sum += row[ki] * q[ki];
                }
                core[hv * head_v + vi] = sum / sqrtf((float)head_k);
            }
        } else {
            float dot = 0.0f;
            for (size_t i = 0; i < head_k; i++) {
                dot += q[i] * k[i];
            }
            dot *= 1.0f / sqrtf((float)head_k);
            for (size_t i = 0; i < head_v; i++) {
                core[hv * head_v + i] = beta * v[i] * dot;
            }
        }
    }
    for (size_t hv = 0; ok && !use_gdn_hook && hv < value_heads; hv++) {
        float ss = 0.0f;
        float *head = core + hv * head_v;
        for (size_t i = 0; i < head_v; i++) {
            ss += head[i] * head[i];
        }
        float scale = 1.0f / sqrtf(ss / (float)head_v + 1e-6f);
        for (size_t i = 0; i < head_v; i++) {
            float w = 0.0f;
            ok = ornith_tensor_value(m, gated_norm_w, i, &w);
            gated[hv * head_v + i] = head[i] * scale * w * siluf(z[hv * head_v + i]);
        }
    }
    if (!gdn_wrote_out) {
        ok = ok && tensor_matvec_hooked(m, out_w, gated, value_dim, out, matvec_hook, hook_ctx);
    }
    free(scratch);
    return ok;
}

int ornith_layer_moe_smoke(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out)
{
    return layer_moe_smoke_with_norm(m, layer, "input_layernorm.weight", x, hidden, top_k, out);
}

static int layer_decode_smoke_with_state_hooked(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t pos, size_t top_k, ornith_linear_state *linear_state, ornith_full_state *full_state, float *out, ornith_moe_with_norm_fn moe_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx)
{
    if (!m || !x || !out || (!layer_has_linear_attention(m, layer) && !layer_has_self_attention(m, layer))) {
        return 0;
    }
    if (layer_decode_hook) {
        ornith_linear_state_view linear_view = {0};
        ornith_full_state_view full_view = {0};
        if (linear_state) {
            linear_view.qkv_dim = linear_state->qkv_dim;
            linear_view.value_heads = linear_state->value_heads;
            linear_view.head_v = linear_state->head_v;
            linear_view.key_heads = linear_state->key_heads;
            linear_view.head_k = linear_state->head_k;
            linear_view.conv_width = linear_state->conv_width;
            linear_view.conv = linear_state->conv;
            linear_view.conv_w = linear_state->conv_w;
            linear_view.ssm = linear_state->ssm;
            linear_view.alog = linear_state->alog;
            linear_view.dt = linear_state->dt;
            linear_view.gated_norm = linear_state->gated_norm;
        }
        if (full_state) {
            full_view.token_cap = full_state->token_cap;
            full_view.token_count = full_state->token_count;
            full_view.q_heads = full_state->q_heads;
            full_view.kv_heads = full_state->kv_heads;
            full_view.head_dim = full_state->head_dim;
            full_view.kv_dim = full_state->kv_dim;
            full_view.k = full_state->k;
            full_view.v = full_state->v;
        }
        int handled = layer_decode_hook(m, layer, x, hidden, pos, top_k, linear_state ? &linear_view : NULL, full_state ? &full_view : NULL, out, hook_ctx);
        if (handled >= 0) {
            if (handled && full_state) full_state->token_count = full_view.token_count;
            return handled;
        }
    }
    float *attn_x = malloc(hidden * 3 * sizeof(float));
    if (!attn_x) {
        return 0;
    }
    float *attn = attn_x + hidden;
    float *mlp = attn + hidden;
    int ok = 1;
    if (layer_has_self_attention(m, layer)) {
        if (full_state) {
            ok = self_attention_step_hooked(m, layer, x, hidden, pos, full_state, attn, matvec_hook, batch_hook, self_attn_hook, hook_ctx);
        } else {
            ok = self_attention_first_token_hooked(m, layer, x, hidden, attn, matvec_hook, hook_ctx);
        }
    } else {
        ok = linear_attention_step_hooked(m, layer, x, hidden, linear_state, attn, matvec_hook, batch_hook, gdn_hook, linear_attn_hook, hook_ctx);
    }
    if (ok && layer_finish_hook) {
        int handled = layer_finish_hook(m, layer, x, attn, hidden, top_k, out, hook_ctx);
        if (handled >= 0) {
            free(attn_x);
            return handled;
        }
    }
    for (size_t i = 0; ok && i < hidden; i++) {
        attn_x[i] = x[i] + attn[i];
    }
    ok = ok && (moe_hook ?
        moe_hook(m, layer, "post_attention_layernorm.weight", attn_x, hidden, top_k, mlp, hook_ctx) :
        layer_moe_smoke_with_norm(m, layer, "post_attention_layernorm.weight", attn_x, hidden, top_k, mlp));
    for (size_t i = 0; ok && i < hidden; i++) {
        out[i] = attn[i] + mlp[i];
    }
    free(attn_x);
    return ok;
}

static int layer_decode_smoke_with_state(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t pos, size_t top_k, ornith_linear_state *linear_state, ornith_full_state *full_state, float *out)
{
    return layer_decode_smoke_with_state_hooked(m, layer, x, hidden, pos, top_k, linear_state, full_state, out, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

int ornith_layer_decode_smoke(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out)
{
    return layer_decode_smoke_with_state(m, layer, x, hidden, 0, top_k, NULL, NULL, out);
}

int ornith_embed_token(const ornith_model *m, uint64_t token_id, float *out, size_t hidden)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    if (!embed || !out || embed->ndim != 2 || token_id >= (uint64_t)embed->shape[0] ||
        hidden != (size_t)embed->shape[1]) {
        return 0;
    }
    uint64_t base = token_id * hidden;
    const unsigned char *payload = ornith_tensor_payload(m, embed, NULL);
    if (payload && embed->quant == ORNITH_QUANT_BF16) {
        const unsigned char *row = payload + base * 2;
        for (size_t i = 0; i < hidden; i++) {
            out[i] = bf16_at(row + i * 2);
        }
        return 1;
    }
    for (size_t i = 0; i < hidden; i++) {
        if (!ornith_tensor_value(m, embed, base + i, &out[i])) {
            return 0;
        }
    }
    return 1;
}

static int lm_head_topk_rows(const ornith_model *m, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values)
{
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!head || !x || !indices || !values || head->ndim != 2 || hidden != (size_t)head->shape[1] ||
        rows == 0 || rows > (size_t)head->shape[0] || k == 0 || k > rows) {
        return 0;
    }
    float *scores = malloc(rows * sizeof(float));
    if (!scores) {
        return 0;
    }
    int ok = 1;
    if (rows == (size_t)head->shape[0]) {
        ok = ornith_tensor_matvec(m, head, x, hidden, scores);
    } else {
        for (size_t r = 0; ok && r < rows; r++) {
            float acc = 0.0f;
            for (size_t c = 0; c < hidden; c++) {
                float v = 0.0f;
                ok = ornith_tensor_value(m, head, r * hidden + c, &v);
                acc += v * x[c];
            }
            scores[r] = acc;
        }
    }
    ok = ok && ornith_topk(scores, rows, k, indices, values);
    free(scores);
    return ok;
}

int ornith_lm_head_topk(const ornith_model *m, const float *x, size_t hidden, size_t k, size_t *indices, float *values)
{
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    return head ? lm_head_topk_rows(m, x, hidden, (size_t)head->shape[0], k, indices, values) : 0;
}

int ornith_lm_head_topk_limited(const ornith_model *m, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values)
{
    return lm_head_topk_rows(m, x, hidden, rows, k, indices, values);
}

int ornith_step_smoke_limited(const ornith_model *m, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    const ornith_tensor_info *final_norm = ornith_model_find_tensor(m, "model.language_model.norm.weight");
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!m || !embed || !final_norm || !indices || !values || embed->ndim != 2 ||
        !head || head->ndim != 2 || final_norm->nparams != (uint64_t)embed->shape[1] ||
        layer_count > ornith_model_layer_count(m)) {
        return 0;
    }
    size_t hidden = (size_t)embed->shape[1];
    size_t rows = vocab_limit ? vocab_limit : (size_t)head->shape[0];
    float *x = calloc(hidden * 3, sizeof(float));
    if (!x) {
        return 0;
    }
    float *delta = x + hidden;
    float *norm = delta + hidden;

    int ok = ornith_embed_token(m, token_id, x, hidden);
    for (size_t layer = 0; ok && layer < layer_count; layer++) {
        ok = ornith_layer_moe_smoke(m, (int64_t)layer, x, hidden, expert_top_k, delta);
        for (size_t i = 0; ok && i < hidden; i++) {
            x[i] += delta[i];
        }
    }
    ok = ok &&
         ornith_rmsnorm(m, final_norm, x, hidden, 1e-6f, norm) &&
         lm_head_topk_rows(m, norm, hidden, rows, out_top_k, indices, values);
    free(x);
    return ok;
}

int ornith_decode_smoke_limited(const ornith_model *m, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    const ornith_tensor_info *final_norm = ornith_model_find_tensor(m, "model.language_model.norm.weight");
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!m || !embed || !final_norm || !indices || !values || embed->ndim != 2 ||
        !head || head->ndim != 2 || final_norm->nparams != (uint64_t)embed->shape[1] ||
        layer_count > ornith_model_layer_count(m)) {
        return 0;
    }
    size_t hidden = (size_t)embed->shape[1];
    size_t rows = vocab_limit ? vocab_limit : (size_t)head->shape[0];
    float *x = calloc(hidden * 3, sizeof(float));
    if (!x) {
        return 0;
    }
    float *delta = x + hidden;
    float *norm = delta + hidden;

    int ok = ornith_embed_token(m, token_id, x, hidden);
    for (size_t layer = 0; ok && layer < layer_count; layer++) {
        ok = layer_decode_smoke_with_state(m, (int64_t)layer, x, hidden, 0, expert_top_k, NULL, NULL, delta);
        for (size_t i = 0; ok && i < hidden; i++) {
            x[i] += delta[i];
        }
    }
    ok = ok &&
         ornith_rmsnorm(m, final_norm, x, hidden, 1e-6f, norm) &&
         lm_head_topk_rows(m, norm, hidden, rows, out_top_k, indices, values);
    free(x);
    return ok;
}

static void decode_state_free(ornith_decode_state *s)
{
    if (!s) return;
    for (size_t layer = 0; layer < s->layer_count; layer++) {
        linear_state_free(&s->linear[layer]);
        full_state_free(&s->full[layer]);
    }
    free(s->linear);
    free(s->full);
    free(s->x);
    memset(s, 0, sizeof(*s));
}

static int decode_state_init(const ornith_model *m, size_t layer_count, size_t expert_top_k, size_t token_cap, ornith_decode_state *s)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    if (!m || !embed || embed->ndim != 2 || !token_cap || layer_count > ornith_model_layer_count(m)) {
        return 0;
    }
    s->layer_count = layer_count;
    s->hidden = (size_t)embed->shape[1];
    s->expert_top_k = expert_top_k;
    s->linear = calloc(layer_count ? layer_count : 1, sizeof(s->linear[0]));
    s->full = calloc(layer_count ? layer_count : 1, sizeof(s->full[0]));
    s->x = calloc(s->hidden * 3, sizeof(float));
    if (!s->linear || !s->full || !s->x) {
        decode_state_free(s);
        return 0;
    }
    s->delta = s->x + s->hidden;
    s->norm = s->delta + s->hidden;
    int ok = 1;
    /* ponytail: predecode pays back around prompt+8 tokens; tune when a real sampler owns sessions. */
    int predecode_linear = token_cap >= 12;
    for (size_t layer = 0; ok && layer < layer_count; layer++) {
        if (layer_has_linear_attention(m, (int64_t)layer)) {
            ok = linear_state_init(m, (int64_t)layer, predecode_linear, &s->linear[layer]);
        } else if (layer_has_self_attention(m, (int64_t)layer)) {
            ok = full_state_init(m, (int64_t)layer, token_cap, &s->full[layer]);
        }
    }
    if (!ok) decode_state_free(s);
    return ok;
}

static void fill_linear_view(const ornith_linear_state *src, ornith_linear_state_view *dst)
{
    memset(dst, 0, sizeof(*dst));
    if (!src || !src->ssm) return;
    dst->qkv_dim = src->qkv_dim;
    dst->value_heads = src->value_heads;
    dst->head_v = src->head_v;
    dst->key_heads = src->key_heads;
    dst->head_k = src->head_k;
    dst->conv_width = src->conv_width;
    dst->conv = src->conv;
    dst->conv_w = src->conv_w;
    dst->ssm = src->ssm;
    dst->alog = src->alog;
    dst->dt = src->dt;
    dst->gated_norm = src->gated_norm;
}

static void fill_full_view(const ornith_full_state *src, ornith_full_state_view *dst)
{
    memset(dst, 0, sizeof(*dst));
    if (!src || !src->k) return;
    dst->token_cap = src->token_cap;
    dst->token_count = src->token_count;
    dst->q_heads = src->q_heads;
    dst->kv_heads = src->kv_heads;
    dst->head_dim = src->head_dim;
    dst->kv_dim = src->kv_dim;
    dst->k = src->k;
    dst->v = src->v;
}

static int decode_state_step_hooked(const ornith_model *m, ornith_decode_state *s, uint64_t token_id, size_t pos, ornith_moe_with_norm_fn moe_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_decode_token_fn token_decode_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx)
{
    if (token_decode_hook) {
        ornith_linear_state_view *linear = calloc(s->layer_count ? s->layer_count : 1, sizeof(*linear));
        ornith_full_state_view *full = calloc(s->layer_count ? s->layer_count : 1, sizeof(*full));
        if (!linear || !full) {
            free(linear);
            free(full);
            return 0;
        }
        for (size_t layer = 0; layer < s->layer_count; layer++) {
            fill_linear_view(&s->linear[layer], &linear[layer]);
            fill_full_view(&s->full[layer], &full[layer]);
        }
        int handled = token_decode_hook(m, token_id, pos, s->layer_count, s->hidden, s->expert_top_k, linear, full, s->x, hook_ctx);
        if (handled >= 0) {
            if (handled) {
                for (size_t layer = 0; layer < s->layer_count; layer++) {
                    if (s->full[layer].k) s->full[layer].token_count = full[layer].token_count;
                }
            }
            free(linear);
            free(full);
            return handled;
        }
        free(linear);
        free(full);
    }
    int ok = ornith_embed_token(m, token_id, s->x, s->hidden);
    for (size_t layer = 0; ok && layer < s->layer_count; layer++) {
        ornith_linear_state *lst = s->linear[layer].ssm ? &s->linear[layer] : NULL;
        ornith_full_state *fst = s->full[layer].k ? &s->full[layer] : NULL;
        ok = layer_decode_smoke_with_state_hooked(m, (int64_t)layer, s->x, s->hidden, pos, s->expert_top_k, lst, fst, s->delta, moe_hook, matvec_hook, batch_hook, gdn_hook, linear_attn_hook, self_attn_hook, layer_decode_hook, layer_finish_hook, hook_ctx);
        for (size_t i = 0; ok && i < s->hidden; i++) {
            s->x[i] += s->delta[i];
        }
    }
    return ok;
}

static int decode_state_step(const ornith_model *m, ornith_decode_state *s, uint64_t token_id, size_t pos)
{
    return decode_state_step_hooked(m, s, token_id, pos, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

int ornith_session_open(const ornith_model *m, size_t layer_count, size_t expert_top_k, size_t token_cap, ornith_session **out)
{
    if (!m || !out || !token_cap || layer_count > ornith_model_layer_count(m)) {
        return 0;
    }
    ornith_session *s = calloc(1, sizeof(*s));
    if (!s) {
        return 0;
    }
    s->model = m;
    s->token_cap = token_cap;
    if (!decode_state_init(m, layer_count, expert_top_k, token_cap, &s->decode)) {
        free(s);
        return 0;
    }
    *out = s;
    return 1;
}

void ornith_session_close(ornith_session *s)
{
    if (!s) return;
    decode_state_free(&s->decode);
    free(s);
}

size_t ornith_session_token_count(const ornith_session *s)
{
    return s ? s->token_count : 0;
}

size_t ornith_session_token_cap(const ornith_session *s)
{
    return s ? s->token_cap : 0;
}

int ornith_session_generate_greedy_limited_with_decode_hooks(ornith_session *s, const uint64_t *prompt_suffix_ids, size_t prompt_suffix_count, size_t max_new, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_decode_token_fn token_decode_hook, ornith_hidden_topk_fn hidden_topk_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx)
{
    if (out_count) *out_count = 0;
    if (!s || !s->model || !out_ids || !out_count || !max_new ||
        s->token_count + prompt_suffix_count + max_new > s->token_cap ||
        (prompt_suffix_count && !prompt_suffix_ids)) {
        return 0;
    }
    const ornith_model *m = s->model;
    const ornith_tensor_info *final_norm = ornith_model_find_tensor(m, "model.language_model.norm.weight");
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!final_norm || !head || head->ndim != 2) {
        return 0;
    }
    for (size_t i = 0; i < prompt_suffix_count; i++) {
        if (!decode_state_step_hooked(m, &s->decode, prompt_suffix_ids[i], s->token_count, moe_hook, matvec_hook, batch_hook, gdn_hook, linear_attn_hook, self_attn_hook, token_decode_hook, layer_decode_hook, layer_finish_hook, hook_ctx)) {
            return 0;
        }
        s->token_count++;
    }
    size_t rows = vocab_limit ? vocab_limit : (size_t)head->shape[0];
    size_t idx = 0;
    float score = 0.0f;
    size_t made = 0;
    for (; made < max_new; made++) {
        int ok = -1;
        if (hidden_topk_hook) {
            ok = hidden_topk_hook(m, s->decode.x, s->decode.hidden, rows, 1, &idx, &score, hook_ctx);
        }
        if (ok < 0) {
            ok = ornith_rmsnorm(m, final_norm, s->decode.x, s->decode.hidden, 1e-6f, s->decode.norm) &&
                 (lm_head_hook ?
                    lm_head_hook(m, s->decode.norm, s->decode.hidden, rows, 1, &idx, &score, hook_ctx) :
                    lm_head_topk_rows(m, s->decode.norm, s->decode.hidden, rows, 1, &idx, &score));
        }
        if (!ok) break;
        out_ids[made] = (uint64_t)idx;
        if (out_scores) out_scores[made] = score;
        if (idx == 248046 || idx == 248044) {
            made++;
            break;
        }
        if (!decode_state_step_hooked(m, &s->decode, (uint64_t)idx, s->token_count, moe_hook, matvec_hook, batch_hook, gdn_hook, linear_attn_hook, self_attn_hook, token_decode_hook, layer_decode_hook, layer_finish_hook, hook_ctx)) {
            break;
        }
        s->token_count++;
    }
    *out_count = made;
    return made > 0;
}

int ornith_session_generate_greedy_limited(ornith_session *s, const uint64_t *prompt_suffix_ids, size_t prompt_suffix_count, size_t max_new, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count)
{
    return ornith_session_generate_greedy_limited_with_decode_hooks(s, prompt_suffix_ids, prompt_suffix_count, max_new, vocab_limit, out_ids, out_scores, out_count, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

int ornith_decode_sequence_smoke_limited(const ornith_model *m, const uint64_t *token_ids, size_t token_count, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    const ornith_tensor_info *final_norm = ornith_model_find_tensor(m, "model.language_model.norm.weight");
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!m || !token_ids || !token_count || !embed || !final_norm || !indices || !values || embed->ndim != 2 ||
        !head || head->ndim != 2 || final_norm->nparams != (uint64_t)embed->shape[1] ||
        layer_count > ornith_model_layer_count(m)) {
        return 0;
    }
    ornith_decode_state state = {0};
    int ok = decode_state_init(m, layer_count, expert_top_k, token_count, &state);
    size_t rows = vocab_limit ? vocab_limit : (size_t)head->shape[0];
    for (size_t tok = 0; ok && tok < token_count; tok++) {
        ok = decode_state_step(m, &state, token_ids[tok], tok);
    }
    ok = ok &&
         ornith_rmsnorm(m, final_norm, state.x, state.hidden, 1e-6f, state.norm) &&
         lm_head_topk_rows(m, state.norm, state.hidden, rows, out_top_k, indices, values);
    decode_state_free(&state);
    return ok;
}

int ornith_generate_greedy_limited_with_decode_hooks(const ornith_model *m, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, ornith_tensor_matvec_fn matvec_hook, ornith_tensor_matvec_batch_fn batch_hook, ornith_gdn_recurrent_fn gdn_hook, ornith_linear_attention_fn linear_attn_hook, ornith_self_attention_fn self_attn_hook, ornith_decode_token_fn token_decode_hook, ornith_hidden_topk_fn hidden_topk_hook, ornith_layer_decode_fn layer_decode_hook, ornith_layer_finish_fn layer_finish_hook, void *hook_ctx)
{
    if (!m || !prompt_ids || !prompt_count || !out_ids || !out_count || !max_new ||
        layer_count > ornith_model_layer_count(m)) {
        return 0;
    }
    ornith_session *session = NULL;
    int ok = ornith_session_open(m, layer_count, expert_top_k, prompt_count + max_new, &session) &&
             ornith_session_generate_greedy_limited_with_decode_hooks(session, prompt_ids, prompt_count, max_new, vocab_limit, out_ids, out_scores, out_count, moe_hook, lm_head_hook, matvec_hook, batch_hook, gdn_hook, linear_attn_hook, self_attn_hook, token_decode_hook, hidden_topk_hook, layer_decode_hook, layer_finish_hook, hook_ctx);
    ornith_session_close(session);
    if (!ok && out_count) {
        *out_count = 0;
    }
    return ok;
}

int ornith_generate_greedy_limited_with_hooks(const ornith_model *m, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, ornith_moe_with_norm_fn moe_hook, ornith_lm_head_topk_fn lm_head_hook, void *hook_ctx)
{
    return ornith_generate_greedy_limited_with_decode_hooks(m, prompt_ids, prompt_count, max_new, layer_count, expert_top_k, vocab_limit, out_ids, out_scores, out_count, moe_hook, lm_head_hook, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, hook_ctx);
}

int ornith_generate_greedy_limited(const ornith_model *m, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count)
{
    return ornith_generate_greedy_limited_with_hooks(m, prompt_ids, prompt_count, max_new, layer_count, expert_top_k, vocab_limit, out_ids, out_scores, out_count, NULL, NULL, NULL);
}

int ornith_step_smoke(const ornith_model *m, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t *indices, float *values)
{
    return ornith_step_smoke_limited(m, token_id, layer_count, expert_top_k, out_top_k, 0, indices, values);
}

int ornith_topk(const float *scores, size_t n, size_t k, size_t *indices, float *values)
{
    if (!scores || !indices || !values || k > n) {
        return 0;
    }
    for (size_t i = 0; i < k; i++) {
        indices[i] = n;
        values[i] = -INFINITY;
    }
    for (size_t i = 0; i < n; i++) {
        float v = scores[i];
        if (k == 0 || v <= values[k - 1]) {
            continue;
        }
        size_t pos = k - 1;
        while (pos > 0 && v > values[pos - 1]) {
            values[pos] = values[pos - 1];
            indices[pos] = indices[pos - 1];
            pos--;
        }
        values[pos] = v;
        indices[pos] = i;
    }
    for (size_t i = 0; i < k; i++) {
        if (indices[i] == n) return 0;
    }
    return 1;
}

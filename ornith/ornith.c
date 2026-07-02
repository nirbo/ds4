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
                norm->nparams != (uint64_t)(a->shape[0] * 2) ||
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
    if (t->quant == ORNITH_QUANT_BF16) {
        *out = bf16_at(payload + i * 2);
        return 1;
    }

    uint32_t block = s->block_size;
    uint64_t block_idx = i / block;
    uint32_t in_block = (uint32_t)(i % block);
    const unsigned char *base = payload + block_idx * block_bytes(t->quant, block);
    float scale = bf16_at(base);
    if (t->quant == ORNITH_QUANT_IQ1) {
        unsigned char bits = base[2 + in_block / 8];
        *out = (bits & (1u << (in_block % 8))) ? scale : -scale;
        return 1;
    }
    unsigned char packed = base[2 + in_block / 2];
    int q = (in_block & 1) ? (packed >> 4) : (packed & 15);
    if (q >= 8) q -= 16;
    *out = scale * (float)q;
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
    for (size_t i = 0; i < n; i++) {
        float w = 0.0f;
        if (!ornith_tensor_value(m, weight, i, &w)) {
            return 0;
        }
        out[i] = x[i] * scale * w;
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

int ornith_layer_moe_smoke(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out)
{
    return layer_moe_smoke_with_norm(m, layer, "input_layernorm.weight", x, hidden, top_k, out);
}

int ornith_layer_decode_smoke(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out)
{
    if (!m || !x || !out || (!layer_has_linear_attention(m, layer) && !layer_has_self_attention(m, layer))) {
        return 0;
    }
    /* ponytail: attention delta is intentionally zero until Ornith attention kernels exist. */
    return layer_moe_smoke_with_norm(m, layer, "post_attention_layernorm.weight", x, hidden, top_k, out);
}

int ornith_embed_token(const ornith_model *m, uint64_t token_id, float *out, size_t hidden)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    if (!embed || !out || embed->ndim != 2 || token_id >= (uint64_t)embed->shape[0] ||
        hidden != (size_t)embed->shape[1]) {
        return 0;
    }
    uint64_t base = token_id * hidden;
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
    for (size_t r = 0; ok && r < rows; r++) {
        float acc = 0.0f;
        for (size_t c = 0; c < hidden; c++) {
            float v = 0.0f;
            ok = ornith_tensor_value(m, head, r * hidden + c, &v);
            acc += v * x[c];
        }
        scores[r] = acc;
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
        ok = ornith_layer_decode_smoke(m, (int64_t)layer, x, hidden, expert_top_k, delta);
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

int ornith_step_smoke(const ornith_model *m, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t *indices, float *values)
{
    return ornith_step_smoke_limited(m, token_id, layer_count, expert_top_k, out_top_k, 0, indices, values);
}

int ornith_topk(const float *scores, size_t n, size_t k, size_t *indices, float *values)
{
    if (!scores || !indices || !values || k > n) {
        return 0;
    }
    for (size_t out_i = 0; out_i < k; out_i++) {
        size_t best = n;
        for (size_t i = 0; i < n; i++) {
            int used = 0;
            for (size_t j = 0; j < out_i; j++) {
                used = used || indices[j] == i;
            }
            if (!used && (best == n || scores[i] > scores[best])) {
                best = i;
            }
        }
        if (best == n) {
            return 0;
        }
        indices[out_i] = best;
        values[out_i] = scores[best];
    }
    return 1;
}

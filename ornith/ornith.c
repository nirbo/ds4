#define _POSIX_C_SOURCE 200809L

#include "ornith.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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

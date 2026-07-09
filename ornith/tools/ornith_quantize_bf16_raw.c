#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ornith_ds4_quants.h"

#define LEGACY_BLOCKS_PER_CHUNK 8192
#define ROWS_PER_CHUNK 64

typedef struct {
    int in_fd;
    int out_fd;
    const char *mode;
    ds4q_type ds4_type;
    uint64_t in_base;
    uint64_t out_base;
    uint64_t nparams;
    uint64_t start_unit;
    uint64_t end_unit;
    uint64_t ncols;
    uint64_t rows_per_expert;
    int block;
    uint64_t full_block_bytes;
    const float *imatrix;
    uint64_t imatrix_experts;
    uint64_t progress_params;
    uint64_t *done;
    uint64_t *next_progress;
    struct timespec started;
    pthread_mutex_t *lock;
} Ctx;

static float bf16_to_float(uint16_t v)
{
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t float_to_bf16(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void fail(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(2);
}

static void die(const char *msg)
{
    perror(msg);
    exit(1);
}

static void read_full(int fd, void *buf, size_t n, off_t off)
{
    uint8_t *p = buf;
    while (n) {
        ssize_t r = pread(fd, p, n, off);
        if (r <= 0) die("pread");
        p += r;
        off += r;
        n -= (size_t)r;
    }
}

static void write_full(int fd, const void *buf, size_t n, off_t off)
{
    const uint8_t *p = buf;
    while (n) {
        ssize_t r = pwrite(fd, p, n, off);
        if (r <= 0) die("pwrite");
        p += r;
        off += r;
        n -= (size_t)r;
    }
}

static void progress(Ctx *ctx, uint64_t count)
{
    if (!ctx->progress_params) return;
    pthread_mutex_lock(ctx->lock);
    *ctx->done += count;
    if (*ctx->done >= *ctx->next_progress) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (double)(now.tv_sec - ctx->started.tv_sec) +
                         (double)(now.tv_nsec - ctx->started.tv_nsec) / 1000000000.0;
        if (elapsed < 0.001) elapsed = 0.001;
        fprintf(stderr, "quant mode=%s params=%llu/%llu rate=%.1fMparams/s\n",
                ctx->mode, (unsigned long long)*ctx->done,
                (unsigned long long)ctx->nparams,
                (double)*ctx->done / elapsed / 1000000.0);
        while (*ctx->next_progress <= *ctx->done) *ctx->next_progress += ctx->progress_params;
    }
    pthread_mutex_unlock(ctx->lock);
}

static size_t quant_iq1_chunk(Ctx *ctx, uint64_t block_idx, uint64_t blocks,
                              uint16_t *buf, uint8_t *out, uint64_t *done_params)
{
    size_t out_pos = 0;
    uint64_t in_pos = 0;
    *done_params = 0;
    for (uint64_t local = 0; local < blocks; local++) {
        uint64_t param = (block_idx + local) * (uint64_t)ctx->block;
        int count = (int)((ctx->nparams - param) < (uint64_t)ctx->block ?
                          (ctx->nparams - param) : (uint64_t)ctx->block);
        uint16_t *values = buf + in_pos;
        uint8_t *packed = out + out_pos + 2;
        size_t packed_bytes = (size_t)(count + 7) / 8;
        float sum = 0.0f;
        memset(packed, 0, packed_bytes);
        for (int i = 0; i < count; i++) {
            float v = bf16_to_float(values[i]);
            sum += fabsf(v);
            if (!signbit(v)) packed[i >> 3] |= (uint8_t)(1u << (i & 7));
        }
        uint16_t scale = float_to_bf16(sum / (float)count);
        memcpy(out + out_pos, &scale, sizeof(scale));
        in_pos += (uint64_t)count;
        out_pos += 2 + packed_bytes;
        *done_params += (uint64_t)count;
    }
    return out_pos;
}

static size_t quant_q4_chunk(Ctx *ctx, uint64_t block_idx, uint64_t blocks,
                             uint16_t *buf, uint8_t *out, uint64_t *done_params)
{
    size_t out_pos = 0;
    uint64_t in_pos = 0;
    *done_params = 0;
    for (uint64_t local = 0; local < blocks; local++) {
        uint64_t param = (block_idx + local) * (uint64_t)ctx->block;
        int count = (int)((ctx->nparams - param) < (uint64_t)ctx->block ?
                          (ctx->nparams - param) : (uint64_t)ctx->block);
        uint16_t *values = buf + in_pos;
        uint8_t *packed = out + out_pos + 2;
        size_t packed_bytes = (size_t)(count + 1) / 2;
        float max_abs = 0.0f;
        for (int i = 0; i < count; i++) {
            float a = fabsf(bf16_to_float(values[i]));
            if (a > max_abs) max_abs = a;
        }
        float scale_f = max_abs > 0.0f ? max_abs / 7.0f : 0.0f;
        uint16_t scale = float_to_bf16(scale_f);
        memset(packed, 0, packed_bytes);
        for (int i = 0; i < count; i++) {
            int q = scale_f > 0.0f ? (int)lrintf(bf16_to_float(values[i]) / scale_f) : 0;
            if (q < -8) q = -8;
            if (q > 7) q = 7;
            uint8_t nibble = (uint8_t)(q & 15);
            if (i & 1) packed[i >> 1] |= (uint8_t)(nibble << 4);
            else packed[i >> 1] |= nibble;
        }
        memcpy(out + out_pos, &scale, sizeof(scale));
        in_pos += (uint64_t)count;
        out_pos += 2 + packed_bytes;
        *done_params += (uint64_t)count;
    }
    return out_pos;
}

static void *legacy_worker(void *arg)
{
    Ctx *ctx = arg;
    uint64_t max_blocks = ctx->end_unit - ctx->start_unit;
    if (max_blocks > LEGACY_BLOCKS_PER_CHUNK) max_blocks = LEGACY_BLOCKS_PER_CHUNK;
    uint16_t *buf = malloc((size_t)max_blocks * (size_t)ctx->block * sizeof(uint16_t));
    uint8_t *out = malloc((size_t)max_blocks * (size_t)ctx->full_block_bytes);
    if (!buf || !out) die("alloc legacy worker");
    for (uint64_t b = ctx->start_unit; b < ctx->end_unit;) {
        uint64_t blocks = ctx->end_unit - b;
        if (blocks > LEGACY_BLOCKS_PER_CHUNK) blocks = LEGACY_BLOCKS_PER_CHUNK;
        uint64_t first_param = b * (uint64_t)ctx->block;
        uint64_t available = ctx->nparams - first_param;
        uint64_t params = blocks * (uint64_t)ctx->block;
        if (params > available) params = available;
        uint64_t done_params = 0;
        read_full(ctx->in_fd, buf, (size_t)params * sizeof(uint16_t),
                  (off_t)(ctx->in_base + first_param * 2));
        size_t out_bytes = strcmp(ctx->mode, "iq1") == 0
            ? quant_iq1_chunk(ctx, b, blocks, buf, out, &done_params)
            : quant_q4_chunk(ctx, b, blocks, buf, out, &done_params);
        write_full(ctx->out_fd, out, out_bytes,
                   (off_t)(ctx->out_base + b * ctx->full_block_bytes));
        progress(ctx, done_params);
        b += blocks;
    }
    free(buf);
    free(out);
    return NULL;
}

static const float *row_imatrix(const Ctx *ctx, uint64_t row)
{
    if (!ctx->imatrix) return NULL;
    uint64_t expert = ctx->rows_per_expert ? row / ctx->rows_per_expert : 0;
    if (expert >= ctx->imatrix_experts) fail("imatrix expert index out of range");
    return ctx->imatrix + expert * ctx->ncols;
}

static void *row_worker(void *arg)
{
    Ctx *ctx = arg;
    const size_t row_size = ds4q_row_size(ctx->ds4_type, (int64_t)ctx->ncols);
    uint16_t *raw = malloc(ROWS_PER_CHUNK * (size_t)ctx->ncols * sizeof(uint16_t));
    float *f32 = malloc(ROWS_PER_CHUNK * (size_t)ctx->ncols * sizeof(float));
    uint8_t *out = malloc(ROWS_PER_CHUNK * row_size);
    if (!raw || !f32 || !out) die("alloc row worker");

    for (uint64_t row = ctx->start_unit; row < ctx->end_unit;) {
        uint64_t rows = ctx->end_unit - row;
        if (rows > ROWS_PER_CHUNK) rows = ROWS_PER_CHUNK;
        if (ctx->rows_per_expert) {
            uint64_t until_boundary = ctx->rows_per_expert - (row % ctx->rows_per_expert);
            if (rows > until_boundary) rows = until_boundary;
        }
        const uint64_t params = rows * ctx->ncols;
        read_full(ctx->in_fd, raw, (size_t)params * sizeof(uint16_t),
                  (off_t)(ctx->in_base + row * ctx->ncols * 2));
        for (uint64_t i = 0; i < params; i++) f32[i] = bf16_to_float(raw[i]);
        const float *importance = row_imatrix(ctx, row);
        size_t written = ds4q_quantize_chunk(ctx->ds4_type, f32, out, 0,
                                             (int64_t)rows, (int64_t)ctx->ncols,
                                             importance);
        if (written != rows * row_size) fail("DS4 quantizer wrote unexpected byte count");
        write_full(ctx->out_fd, out, written, (off_t)(ctx->out_base + row * row_size));
        progress(ctx, params);
        row += rows;
    }
    free(raw);
    free(f32);
    free(out);
    return NULL;
}

static ds4q_type parse_ds4_type(const char *mode)
{
    if (strcmp(mode, "q8_0") == 0) return DS4Q_TYPE_Q8_0;
    if (strcmp(mode, "q2_k") == 0) return DS4Q_TYPE_Q2_K;
    if (strcmp(mode, "q4_k") == 0) return DS4Q_TYPE_Q4_K;
    if (strcmp(mode, "iq2_xxs") == 0) return DS4Q_TYPE_IQ2_XXS;
    return DS4Q_TYPE_COUNT;
}

static float *load_imatrix(const char *path, uint64_t ncols, uint64_t experts)
{
    if (strcmp(path, "-") == 0) return NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("open imatrix");
    struct stat st;
    if (fstat(fd, &st) != 0) die("stat imatrix");
    uint64_t expected = ncols * experts * sizeof(float);
    if ((uint64_t)st.st_size != expected) fail("imatrix byte size mismatch");
    float *values = malloc((size_t)expected);
    if (!values) die("alloc imatrix");
    read_full(fd, values, (size_t)expected, 0);
    close(fd);
    for (uint64_t i = 0; i < ncols * experts; i++) {
        if (!isfinite(values[i]) || values[i] < 0.0f) fail("invalid imatrix value");
    }
    return values;
}

int main(int argc, char **argv)
{
    if (argc != 13) {
        fprintf(stderr,
                "usage: %s IN OUT MODE IN_BYTE_OFFSET OUT_BYTE_OFFSET NPARAMS BLOCK THREADS "
                "PROGRESS_PARAMS NCOLS ROWS_PER_EXPERT IMATRIX_OR_DASH\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    const char *mode = argv[3];
    uint64_t in_base = strtoull(argv[4], NULL, 10);
    uint64_t out_base = strtoull(argv[5], NULL, 10);
    uint64_t nparams = strtoull(argv[6], NULL, 10);
    int block = atoi(argv[7]);
    int threads = atoi(argv[8]);
    uint64_t progress_params = strtoull(argv[9], NULL, 10);
    uint64_t ncols = strtoull(argv[10], NULL, 10);
    uint64_t rows_per_expert = strtoull(argv[11], NULL, 10);
    const char *imatrix_path = argv[12];
    if (threads < 1) threads = 1;

    const int legacy = strcmp(mode, "iq1") == 0 || strcmp(mode, "q4") == 0;
    ds4q_type ds4_type = parse_ds4_type(mode);
    if (!legacy && ds4_type == DS4Q_TYPE_COUNT) fail("unknown quant mode");
    if (!legacy && (block != 256 || !ncols || ncols % ds4q_block_size(ds4_type) || nparams % ncols)) {
        fail("DS4 quant modes require block=256 and aligned complete rows");
    }
    if (rows_per_expert && (nparams / ncols) % rows_per_expert) fail("rows_per_expert does not divide tensor rows");
    uint64_t imatrix_experts = rows_per_expert ? (nparams / ncols) / rows_per_expert : 1;
    float *imatrix = load_imatrix(imatrix_path, ncols, imatrix_experts);
    if (ds4_type == DS4Q_TYPE_IQ2_XXS && !imatrix) fail("iq2_xxs requires a per-expert imatrix");
    if (imatrix && legacy) fail("legacy quant mode cannot consume an imatrix");
    if (!legacy) ds4q_quantize_init(ds4_type);

    int in_fd = open(in_path, O_RDONLY);
    if (in_fd < 0) die("open input");
    int out_fd = open(out_path, O_RDWR);
    if (out_fd < 0) die("open output");

    uint64_t units = legacy ? (nparams + (uint64_t)block - 1) / (uint64_t)block : nparams / ncols;
    uint64_t full_block_bytes = legacy
        ? (strcmp(mode, "iq1") == 0 ? 2 + ((uint64_t)block + 7) / 8 : 2 + ((uint64_t)block + 1) / 2)
        : ds4q_row_size(ds4_type, (int64_t)ncols) / (ncols / (uint64_t)ds4q_block_size(ds4_type));
    if ((uint64_t)threads > units) threads = (int)units;
    pthread_t *tids = calloc((size_t)threads, sizeof(*tids));
    Ctx *ctxs = calloc((size_t)threads, sizeof(*ctxs));
    if (!tids || !ctxs) die("alloc threads");
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    uint64_t done = 0, next_progress = progress_params;
    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);

    for (int t = 0; t < threads; t++) {
        uint64_t start = units * (uint64_t)t / (uint64_t)threads;
        uint64_t end = units * (uint64_t)(t + 1) / (uint64_t)threads;
        ctxs[t] = (Ctx){
            .in_fd = in_fd, .out_fd = out_fd, .mode = mode, .ds4_type = ds4_type,
            .in_base = in_base, .out_base = out_base, .nparams = nparams,
            .start_unit = start, .end_unit = end, .ncols = ncols,
            .rows_per_expert = rows_per_expert, .block = block,
            .full_block_bytes = full_block_bytes, .imatrix = imatrix,
            .imatrix_experts = imatrix_experts, .progress_params = progress_params,
            .done = &done, .next_progress = &next_progress, .started = started, .lock = &lock,
        };
        void *(*worker)(void *) = legacy ? legacy_worker : row_worker;
        if (pthread_create(&tids[t], NULL, worker, &ctxs[t]) != 0) die("pthread_create");
    }
    for (int t = 0; t < threads; t++) {
        if (pthread_join(tids[t], NULL) != 0) die("pthread_join");
    }
    close(in_fd);
    close(out_fd);
    free(imatrix);
    free(tids);
    free(ctxs);
    return 0;
}

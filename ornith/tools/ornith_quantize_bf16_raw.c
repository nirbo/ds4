#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

typedef struct {
    int in_fd;
    int out_fd;
    const char *mode;
    uint64_t in_base;
    uint64_t out_base;
    uint64_t nparams;
    uint64_t start_block;
    uint64_t end_block;
    int block;
    uint64_t full_block_bytes;
    uint64_t progress_params;
    uint64_t *done;
    uint64_t *next_progress;
    time_t started;
    pthread_mutex_t *lock;
} Ctx;

static float bf16_to_float(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t float_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static void read_full(int fd, void *buf, size_t n, off_t off) {
    uint8_t *p = (uint8_t *)buf;
    while (n) {
        ssize_t r = pread(fd, p, n, off);
        if (r <= 0) die("pread");
        p += r;
        off += r;
        n -= (size_t)r;
    }
}

static void write_full(int fd, const void *buf, size_t n, off_t off) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n) {
        ssize_t r = pwrite(fd, p, n, off);
        if (r <= 0) die("pwrite");
        p += r;
        off += r;
        n -= (size_t)r;
    }
}

static void progress(Ctx *ctx, uint64_t count) {
    if (!ctx->progress_params) return;
    pthread_mutex_lock(ctx->lock);
    *ctx->done += count;
    if (*ctx->done >= *ctx->next_progress) {
        double elapsed = difftime(time(NULL), ctx->started);
        if (elapsed < 1.0) elapsed = 1.0;
        fprintf(stderr, "quant mode=%s params=%llu/%llu rate=%.1fMparams/s\n",
                ctx->mode, (unsigned long long)*ctx->done,
                (unsigned long long)ctx->nparams, (double)*ctx->done / elapsed / 1000000.0);
        *ctx->next_progress += ctx->progress_params;
    }
    pthread_mutex_unlock(ctx->lock);
}

static void quant_iq1_block(Ctx *ctx, uint64_t block_idx, uint16_t *buf, uint8_t *packed) {
    uint64_t param = block_idx * (uint64_t)ctx->block;
    int count = (int)((ctx->nparams - param) < (uint64_t)ctx->block ? (ctx->nparams - param) : (uint64_t)ctx->block);
    read_full(ctx->in_fd, buf, (size_t)count * sizeof(uint16_t), (off_t)(ctx->in_base + param * 2));
    float sum = 0.0f;
    memset(packed, 0, (size_t)(count + 7) / 8);
    for (int i = 0; i < count; i++) {
        float v = bf16_to_float(buf[i]);
        sum += fabsf(v);
        if (!signbit(v)) packed[i >> 3] |= (uint8_t)(1u << (i & 7));
    }
    uint16_t scale = float_to_bf16(sum / (float)count);
    uint64_t out_off = ctx->out_base + block_idx * ctx->full_block_bytes;
    write_full(ctx->out_fd, &scale, sizeof(scale), (off_t)out_off);
    write_full(ctx->out_fd, packed, (size_t)(count + 7) / 8, (off_t)(out_off + 2));
    progress(ctx, (uint64_t)count);
}

static void quant_q4_block(Ctx *ctx, uint64_t block_idx, uint16_t *buf, uint8_t *packed) {
    uint64_t param = block_idx * (uint64_t)ctx->block;
    int count = (int)((ctx->nparams - param) < (uint64_t)ctx->block ? (ctx->nparams - param) : (uint64_t)ctx->block);
    read_full(ctx->in_fd, buf, (size_t)count * sizeof(uint16_t), (off_t)(ctx->in_base + param * 2));
    float max_abs = 0.0f;
    for (int i = 0; i < count; i++) {
        float a = fabsf(bf16_to_float(buf[i]));
        if (a > max_abs) max_abs = a;
    }
    float scale_f = max_abs > 0.0f ? max_abs / 7.0f : 0.0f;
    uint16_t scale = float_to_bf16(scale_f);
    memset(packed, 0, (size_t)(count + 1) / 2);
    for (int i = 0; i < count; i++) {
        int q = scale_f > 0.0f ? (int)lrintf(bf16_to_float(buf[i]) / scale_f) : 0;
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        uint8_t nibble = (uint8_t)(q & 15);
        if (i & 1) packed[i >> 1] |= (uint8_t)(nibble << 4);
        else packed[i >> 1] |= nibble;
    }
    uint64_t out_off = ctx->out_base + block_idx * ctx->full_block_bytes;
    write_full(ctx->out_fd, &scale, sizeof(scale), (off_t)out_off);
    write_full(ctx->out_fd, packed, (size_t)(count + 1) / 2, (off_t)(out_off + 2));
    progress(ctx, (uint64_t)count);
}

static void *worker(void *arg) {
    Ctx *ctx = (Ctx *)arg;
    uint16_t *buf = malloc((size_t)ctx->block * sizeof(uint16_t));
    uint8_t *packed = calloc((size_t)ctx->full_block_bytes, 1);
    if (!buf || !packed) die("alloc");
    for (uint64_t b = ctx->start_block; b < ctx->end_block; b++) {
        if (strcmp(ctx->mode, "iq1") == 0) quant_iq1_block(ctx, b, buf, packed);
        else quant_q4_block(ctx, b, buf, packed);
    }
    free(buf);
    free(packed);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 10) {
        fprintf(stderr, "usage: %s IN OUT MODE IN_BYTE_OFFSET OUT_BYTE_OFFSET NPARAMS BLOCK THREADS PROGRESS_PARAMS\n", argv[0]);
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
    if (threads < 1) threads = 1;
    if (strcmp(mode, "iq1") != 0 && strcmp(mode, "q4") != 0) {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }
    int in_fd = open(in_path, O_RDONLY);
    if (in_fd < 0) die("open input");
    int out_fd = open(out_path, O_RDWR);
    if (out_fd < 0) die("open output");

    uint64_t blocks = (nparams + (uint64_t)block - 1) / (uint64_t)block;
    uint64_t full_block_bytes = strcmp(mode, "iq1") == 0 ? 2 + ((uint64_t)block + 7) / 8 : 2 + ((uint64_t)block + 1) / 2;
    if ((uint64_t)threads > blocks) threads = (int)blocks;
    pthread_t *tids = calloc((size_t)threads, sizeof(pthread_t));
    Ctx *ctxs = calloc((size_t)threads, sizeof(Ctx));
    if (!tids || !ctxs) die("alloc threads");
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    uint64_t done = 0, next_progress = progress_params;
    time_t started = time(NULL);

    for (int t = 0; t < threads; t++) {
        uint64_t start = blocks * (uint64_t)t / (uint64_t)threads;
        uint64_t end = blocks * (uint64_t)(t + 1) / (uint64_t)threads;
        ctxs[t] = (Ctx){
            .in_fd = in_fd, .out_fd = out_fd, .mode = mode,
            .in_base = in_base, .out_base = out_base, .nparams = nparams,
            .start_block = start, .end_block = end, .block = block,
            .full_block_bytes = full_block_bytes, .progress_params = progress_params,
            .done = &done, .next_progress = &next_progress, .started = started, .lock = &lock,
        };
        if (pthread_create(&tids[t], NULL, worker, &ctxs[t]) != 0) die("pthread_create");
    }
    for (int t = 0; t < threads; t++) {
        if (pthread_join(tids[t], NULL) != 0) die("pthread_join");
    }
    close(in_fd);
    close(out_fd);
    free(tids);
    free(ctxs);
    return 0;
}

#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BLOCKS_PER_CHUNK 8192

typedef struct {
    uint64_t count;
    double sum_abs_src;
    double sum_abs_err;
    double sum_sq_src;
    double sum_sq_err;
    double max_abs;
} Stats;

typedef struct {
    int src_fd;
    int ornq_fd;
    const char *mode;
    uint64_t src_base;
    uint64_t ornq_base;
    uint64_t nparams;
    uint64_t start_block;
    uint64_t end_block;
    int block;
    uint64_t full_block_bytes;
    uint64_t progress_params;
    uint64_t *done;
    uint64_t *next_progress;
    struct timespec started;
    pthread_mutex_t *lock;
    Stats stats;
} Ctx;

static void die(const char *msg)
{
    perror(msg);
    exit(1);
}

static float bf16_to_float(uint16_t v)
{
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float f16_to_float(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    int exp = (int)((h >> 10) & 31u);
    uint32_t mant = (uint32_t)h & 1023u;
    uint32_t bits;
    if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else if (exp == 0) {
        if (!mant) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void read_full(int fd, void *buf, size_t n, off_t off)
{
    uint8_t *p = (uint8_t *)buf;
    while (n) {
        ssize_t r = pread(fd, p, n, off);
        if (r <= 0) die("pread");
        p += r;
        off += r;
        n -= (size_t)r;
    }
}

static uint64_t mode_block_bytes(const char *mode, uint64_t count)
{
    if (strcmp(mode, "bf16") == 0) return count * 2;
    if (strcmp(mode, "q2_k") == 0) return count == 256 ? 84 : 0;
    if (strcmp(mode, "iq1") == 0) return 2 + (count + 7) / 8;
    if (strcmp(mode, "q4") == 0) return 2 + (count + 1) / 2;
    return 0;
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
        fprintf(stderr, "compare mode=%s params=%llu/%llu rate=%.1fMparams/s\n",
                ctx->mode, (unsigned long long)*ctx->done,
                (unsigned long long)ctx->nparams, (double)*ctx->done / elapsed / 1000000.0);
        *ctx->next_progress += ctx->progress_params;
    }
    pthread_mutex_unlock(ctx->lock);
}

static void stats_add(Stats *s, float src, float got)
{
    double err = (double)src - (double)got;
    double abs_err = fabs(err);
    s->count++;
    s->sum_abs_src += fabs((double)src);
    s->sum_abs_err += abs_err;
    s->sum_sq_src += (double)src * (double)src;
    s->sum_sq_err += err * err;
    if (abs_err > s->max_abs) s->max_abs = abs_err;
}

static float read_quant_value(const char *mode, const uint8_t *q, int in_block)
{
    if (strcmp(mode, "q2_k") == 0) {
        int group = in_block / 16;
        int rem = in_block & 127;
        int v = (q[16 + (in_block / 128) * 32 + (rem & 31)] >> ((rem / 32) * 2)) & 3;
        float d = f16_to_float((uint16_t)q[80] | ((uint16_t)q[81] << 8));
        float dmin = f16_to_float((uint16_t)q[82] | ((uint16_t)q[83] << 8));
        return d * (float)(q[group] & 15) * (float)v - dmin * (float)(q[group] >> 4);
    }
    uint16_t raw_scale;
    memcpy(&raw_scale, q, sizeof(raw_scale));
    float scale = bf16_to_float(raw_scale);
    if (strcmp(mode, "iq1") == 0) {
        return (q[2 + in_block / 8] & (uint8_t)(1u << (in_block & 7))) ? scale : -scale;
    }
    if (strcmp(mode, "q4") == 0) {
        uint8_t packed = q[2 + in_block / 2];
        int v = (in_block & 1) ? (packed >> 4) : (packed & 15);
        if (v >= 8) v -= 16;
        return scale * (float)v;
    }
    uint16_t raw;
    memcpy(&raw, q + (uint64_t)in_block * 2, sizeof(raw));
    return bf16_to_float(raw);
}

static void *worker(void *arg)
{
    Ctx *ctx = (Ctx *)arg;
    uint64_t max_blocks = ctx->end_block - ctx->start_block;
    if (max_blocks > BLOCKS_PER_CHUNK) max_blocks = BLOCKS_PER_CHUNK;
    uint16_t *src = malloc((size_t)max_blocks * (size_t)ctx->block * sizeof(uint16_t));
    uint8_t *ornq = malloc((size_t)max_blocks * (size_t)ctx->full_block_bytes);
    if (!src || !ornq) die("alloc");
    for (uint64_t b = ctx->start_block; b < ctx->end_block;) {
        uint64_t blocks = ctx->end_block - b;
        if (blocks > BLOCKS_PER_CHUNK) blocks = BLOCKS_PER_CHUNK;
        uint64_t first_param = b * (uint64_t)ctx->block;
        uint64_t available = ctx->nparams - first_param;
        uint64_t max_params = blocks * (uint64_t)ctx->block;
        uint64_t params = available < max_params ? available : max_params;
        uint64_t qbytes = 0;
        for (uint64_t local = 0; local < blocks; local++) {
            uint64_t param = (b + local) * (uint64_t)ctx->block;
            uint64_t count = ctx->nparams - param < (uint64_t)ctx->block ? ctx->nparams - param : (uint64_t)ctx->block;
            qbytes += mode_block_bytes(ctx->mode, count);
        }
        read_full(ctx->src_fd, src, (size_t)params * sizeof(uint16_t), (off_t)(ctx->src_base + first_param * 2));
        read_full(ctx->ornq_fd, ornq, (size_t)qbytes, (off_t)(ctx->ornq_base + b * ctx->full_block_bytes));
        uint64_t src_pos = 0;
        uint64_t qpos = 0;
        for (uint64_t local = 0; local < blocks; local++) {
            uint64_t param = (b + local) * (uint64_t)ctx->block;
            int count = (int)(ctx->nparams - param < (uint64_t)ctx->block ? ctx->nparams - param : (uint64_t)ctx->block);
            for (int i = 0; i < count; i++) {
                stats_add(&ctx->stats, bf16_to_float(src[src_pos + (uint64_t)i]), read_quant_value(ctx->mode, ornq + qpos, i));
            }
            src_pos += (uint64_t)count;
            qpos += mode_block_bytes(ctx->mode, (uint64_t)count);
        }
        progress(ctx, params);
        b += blocks;
    }
    free(src);
    free(ornq);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 10) {
        fprintf(stderr, "usage: %s SOURCE ORNQ MODE SOURCE_BYTE_OFFSET ORNQ_BYTE_OFFSET NPARAMS BLOCK THREADS PROGRESS_PARAMS\n", argv[0]);
        return 2;
    }
    const char *src_path = argv[1];
    const char *ornq_path = argv[2];
    const char *mode = argv[3];
    uint64_t src_base = strtoull(argv[4], NULL, 10);
    uint64_t ornq_base = strtoull(argv[5], NULL, 10);
    uint64_t nparams = strtoull(argv[6], NULL, 10);
    int block = atoi(argv[7]);
    int threads = atoi(argv[8]);
    uint64_t progress_params = strtoull(argv[9], NULL, 10);
    if (threads < 1) threads = 1;
    if (strcmp(mode, "bf16") != 0 && strcmp(mode, "iq1") != 0 && strcmp(mode, "q4") != 0 && strcmp(mode, "q2_k") != 0) {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }
    if (strcmp(mode, "q2_k") == 0 && (block != 256 || nparams % 256 != 0)) {
        fprintf(stderr, "q2_k requires block=256 and 256-aligned nparams\n");
        return 2;
    }
    int src_fd = open(src_path, O_RDONLY);
    if (src_fd < 0) die("open source");
    int ornq_fd = open(ornq_path, O_RDONLY);
    if (ornq_fd < 0) die("open ornq");

    uint64_t blocks = (nparams + (uint64_t)block - 1) / (uint64_t)block;
    uint64_t full_block_bytes = mode_block_bytes(mode, (uint64_t)block);
    if ((uint64_t)threads > blocks) threads = (int)blocks;
    pthread_t *tids = calloc((size_t)threads, sizeof(*tids));
    Ctx *ctxs = calloc((size_t)threads, sizeof(*ctxs));
    if (!tids || !ctxs) die("alloc threads");
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    uint64_t done = 0, next_progress = progress_params;
    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);

    for (int t = 0; t < threads; t++) {
        uint64_t start = blocks * (uint64_t)t / (uint64_t)threads;
        uint64_t end = blocks * (uint64_t)(t + 1) / (uint64_t)threads;
        ctxs[t] = (Ctx){
            .src_fd = src_fd, .ornq_fd = ornq_fd, .mode = mode,
            .src_base = src_base, .ornq_base = ornq_base, .nparams = nparams,
            .start_block = start, .end_block = end, .block = block,
            .full_block_bytes = full_block_bytes, .progress_params = progress_params,
            .done = &done, .next_progress = &next_progress, .started = started, .lock = &lock,
        };
        if (pthread_create(&tids[t], NULL, worker, &ctxs[t]) != 0) die("pthread_create");
    }
    Stats total = {0};
    for (int t = 0; t < threads; t++) {
        if (pthread_join(tids[t], NULL) != 0) die("pthread_join");
        total.count += ctxs[t].stats.count;
        total.sum_abs_src += ctxs[t].stats.sum_abs_src;
        total.sum_abs_err += ctxs[t].stats.sum_abs_err;
        total.sum_sq_src += ctxs[t].stats.sum_sq_src;
        total.sum_sq_err += ctxs[t].stats.sum_sq_err;
        if (ctxs[t].stats.max_abs > total.max_abs) total.max_abs = ctxs[t].stats.max_abs;
    }
    printf("count=%llu sum_abs_src=%.17g sum_abs_err=%.17g sum_sq_src=%.17g sum_sq_err=%.17g max_abs=%.17g\n",
           (unsigned long long)total.count, total.sum_abs_src, total.sum_abs_err,
           total.sum_sq_src, total.sum_sq_err, total.max_abs);
    close(src_fd);
    close(ornq_fd);
    free(tids);
    free(ctxs);
    return 0;
}

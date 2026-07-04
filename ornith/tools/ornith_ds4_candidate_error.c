#define _FILE_OFFSET_BITS 64

#include "ornith_ds4_quants.h"

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

#define QK_K 256
#define CHUNK_ROWS 256

typedef struct {
    uint64_t count;
    double sum_abs_src;
    double sum_abs_err;
    double sum_sq_src;
    double sum_sq_err;
    double max_abs;
} Stats;

typedef struct {
    int fd;
    ds4q_type type;
    uint64_t src_base;
    uint64_t nrows;
    uint64_t ncols;
    uint64_t row_start;
    uint64_t row_end;
    const float *imatrix;
    uint64_t progress_rows;
    uint64_t *done_rows;
    uint64_t *next_progress;
    struct timespec started;
    pthread_mutex_t *lock;
    Stats stats;
    double *imatrix_accum;
} Ctx;

static void die(const char *msg)
{
    perror(msg);
    exit(1);
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

static void stats_add(Stats *s, float src, float got)
{
    const double err = (double)src - (double)got;
    const double abs_err = fabs(err);
    s->count++;
    s->sum_abs_src += fabs((double)src);
    s->sum_abs_err += abs_err;
    s->sum_sq_src += (double)src * (double)src;
    s->sum_sq_err += err * err;
    if (abs_err > s->max_abs) s->max_abs = abs_err;
}

static void progress(Ctx *ctx, uint64_t rows)
{
    if (!ctx->progress_rows) return;
    pthread_mutex_lock(ctx->lock);
    *ctx->done_rows += rows;
    if (*ctx->done_rows >= *ctx->next_progress) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (double)(now.tv_sec - ctx->started.tv_sec) +
                         (double)(now.tv_nsec - ctx->started.tv_nsec) / 1000000000.0;
        if (elapsed < 0.001) elapsed = 0.001;
        double params = (double)(*ctx->done_rows) * (double)ctx->ncols;
        fprintf(stderr, "candidate type=%s rows=%llu/%llu rate=%.1fMparams/s\n",
                ds4q_type_name(ctx->type),
                (unsigned long long)*ctx->done_rows,
                (unsigned long long)ctx->nrows,
                params / elapsed / 1000000.0);
        *ctx->next_progress += ctx->progress_rows;
    }
    pthread_mutex_unlock(ctx->lock);
}

static void dequant_q2_k_block(const uint8_t *block, float *out)
{
    const uint8_t *scales = block;
    const uint8_t *qs = block + 16;
    const float d = ds4q_f16_to_f32((uint16_t)block[80] | ((uint16_t)block[81] << 8));
    const float dmin = ds4q_f16_to_f32((uint16_t)block[82] | ((uint16_t)block[83] << 8));
    for (int i = 0; i < QK_K; i++) {
        const int group = i / 16;
        const int rem = i & 127;
        const int q = (qs[(i / 128) * 32 + (rem & 31)] >> ((rem / 32) * 2)) & 3;
        out[i] = d * (float)(scales[group] & 15) * (float)q -
                 dmin * (float)(scales[group] >> 4);
    }
}

static void get_scale_min_k4(int j, const uint8_t *scales, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *d = (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

static void dequant_q4_k_block(const uint8_t *block, float *out)
{
    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;
    const float d = ds4q_f16_to_f32((uint16_t)block[0] | ((uint16_t)block[1] << 8));
    const float dmin = ds4q_f16_to_f32((uint16_t)block[2] | ((uint16_t)block[3] << 8));
    for (int i = 0; i < QK_K; i++) {
        uint8_t sc, mn;
        get_scale_min_k4(i / 32, scales, &sc, &mn);
        const int rem = i & 63;
        const int q = (qs[(i / 64) * 32 + (rem & 31)] >> (rem >= 32 ? 4 : 0)) & 15;
        out[i] = d * (float)sc * (float)q - dmin * (float)mn;
    }
}

static const uint16_t kgrid_iq2_xxs[256] = {
        0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
      100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
     1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
     1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
     2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
     4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
     5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
     8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
};

static void dequant_iq2_xxs_block(const uint8_t *block, float *out)
{
    const float db = ds4q_f16_to_f32((uint16_t)block[0] | ((uint16_t)block[1] << 8));
    const uint32_t *q2 = (const uint32_t *)(const void *)(block + 2);
    for (int ib = 0; ib < 8; ib++) {
        const uint32_t grids = q2[2 * ib + 0];
        const uint32_t aux = q2[2 * ib + 1];
        const float d = db * (0.5f + (float)(aux >> 28));
        for (int k = 0; k < 4; k++) {
            const uint8_t grid_index = (uint8_t)(grids >> (8 * k));
            const uint8_t signs = (uint8_t)((aux >> (7 * k)) & 127);
            const uint16_t grid = kgrid_iq2_xxs[grid_index];
            for (int j = 0; j < 8; j++) {
                const int l = (grid >> (2 * j)) & 3;
                float v = d * (float)(2 * l + 1);
                if (signs & (uint8_t)(1u << j)) v = -v;
                out[32 * ib + 8 * k + j] = v;
            }
        }
    }
}

static void dequant_block(ds4q_type type, const uint8_t *block, float *out)
{
    if (type == DS4Q_TYPE_Q2_K) {
        dequant_q2_k_block(block, out);
    } else if (type == DS4Q_TYPE_Q4_K) {
        dequant_q4_k_block(block, out);
    } else if (type == DS4Q_TYPE_IQ2_XXS) {
        dequant_iq2_xxs_block(block, out);
    } else {
        fprintf(stderr, "unsupported dequant type\n");
        exit(2);
    }
}

static void bf16_rows_to_f32(const uint16_t *src, float *dst, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) dst[i] = ds4q_bf16_to_f32(src[i]);
}

static void *imatrix_worker(void *arg)
{
    Ctx *ctx = (Ctx *)arg;
    const uint64_t ncols = ctx->ncols;
    uint16_t *raw = malloc((size_t)CHUNK_ROWS * (size_t)ncols * sizeof(uint16_t));
    float *f32 = malloc((size_t)CHUNK_ROWS * (size_t)ncols * sizeof(float));
    ctx->imatrix_accum = calloc((size_t)ncols, sizeof(double));
    if (!raw || !f32 || !ctx->imatrix_accum) die("alloc imatrix");
    for (uint64_t row = ctx->row_start; row < ctx->row_end;) {
        uint64_t rows = ctx->row_end - row;
        if (rows > CHUNK_ROWS) rows = CHUNK_ROWS;
        read_full(ctx->fd, raw, (size_t)(rows * ncols) * sizeof(uint16_t),
                  (off_t)(ctx->src_base + row * ncols * 2));
        bf16_rows_to_f32(raw, f32, rows * ncols);
        for (uint64_t r = 0; r < rows; r++) {
            float *x = f32 + r * ncols;
            for (uint64_t c = 0; c < ncols; c++) ctx->imatrix_accum[c] += (double)x[c] * (double)x[c];
        }
        progress(ctx, rows);
        row += rows;
    }
    free(raw);
    free(f32);
    return NULL;
}

static void *quant_worker(void *arg)
{
    Ctx *ctx = (Ctx *)arg;
    const uint64_t ncols = ctx->ncols;
    const size_t row_size = ds4q_row_size(ctx->type, (int64_t)ncols);
    const uint64_t blocks_per_row = ncols / QK_K;
    uint16_t *raw = malloc((size_t)CHUNK_ROWS * (size_t)ncols * sizeof(uint16_t));
    float *f32 = malloc((size_t)CHUNK_ROWS * (size_t)ncols * sizeof(float));
    uint8_t *quant = malloc((size_t)CHUNK_ROWS * row_size);
    float deq[QK_K];
    if (!raw || !f32 || !quant) die("alloc quant");
    for (uint64_t row = ctx->row_start; row < ctx->row_end;) {
        uint64_t rows = ctx->row_end - row;
        if (rows > CHUNK_ROWS) rows = CHUNK_ROWS;
        read_full(ctx->fd, raw, (size_t)(rows * ncols) * sizeof(uint16_t),
                  (off_t)(ctx->src_base + row * ncols * 2));
        bf16_rows_to_f32(raw, f32, rows * ncols);
        size_t written = ds4q_quantize_chunk(ctx->type, f32, quant, 0, (int64_t)rows,
                                             (int64_t)ncols, ctx->imatrix);
        if (written != (size_t)rows * row_size) {
            fprintf(stderr, "quant wrote unexpected byte count\n");
            exit(1);
        }
        for (uint64_t r = 0; r < rows; r++) {
            const float *src = f32 + r * ncols;
            const uint8_t *qrow = quant + r * row_size;
            for (uint64_t b = 0; b < blocks_per_row; b++) {
                dequant_block(ctx->type, qrow + b * ds4q_row_size(ctx->type, QK_K), deq);
                for (int i = 0; i < QK_K; i++) stats_add(&ctx->stats, src[b * QK_K + i], deq[i]);
            }
        }
        progress(ctx, rows);
        row += rows;
    }
    free(raw);
    free(f32);
    free(quant);
    return NULL;
}

static ds4q_type parse_type(const char *s)
{
    if (strcmp(s, "q2_k") == 0 || strcmp(s, "Q2_K") == 0) return DS4Q_TYPE_Q2_K;
    if (strcmp(s, "q4_k") == 0 || strcmp(s, "Q4_K") == 0) return DS4Q_TYPE_Q4_K;
    if (strcmp(s, "iq2_xxs") == 0 || strcmp(s, "IQ2_XXS") == 0) return DS4Q_TYPE_IQ2_XXS;
    fprintf(stderr, "unknown type: %s\n", s);
    exit(2);
}

static void run_threads(Ctx *ctxs, pthread_t *tids, int threads, void *(*fn)(void *))
{
    for (int t = 0; t < threads; t++) {
        if (pthread_create(&tids[t], NULL, fn, &ctxs[t]) != 0) die("pthread_create");
    }
    for (int t = 0; t < threads; t++) {
        if (pthread_join(tids[t], NULL) != 0) die("pthread_join");
    }
}

int main(int argc, char **argv)
{
    if (argc != 8) {
        fprintf(stderr, "usage: %s SOURCE TYPE SOURCE_BYTE_OFFSET NROWS NCOLS THREADS PROGRESS_ROWS\n", argv[0]);
        return 2;
    }
    const char *source = argv[1];
    ds4q_type type = parse_type(argv[2]);
    uint64_t src_base = strtoull(argv[3], NULL, 10);
    uint64_t nrows = strtoull(argv[4], NULL, 10);
    uint64_t ncols = strtoull(argv[5], NULL, 10);
    int threads = atoi(argv[6]);
    uint64_t progress_rows = strtoull(argv[7], NULL, 10);
    if (threads < 1) threads = 1;
    if ((uint64_t)threads > nrows) threads = (int)nrows;
    if (ncols == 0 || ncols % QK_K != 0) {
        fprintf(stderr, "ncols must be divisible by 256\n");
        return 2;
    }
    if (!ds4q_can_quantize(type) || !ds4q_row_size(type, (int64_t)ncols)) {
        fprintf(stderr, "unsupported type or shape\n");
        return 2;
    }
    ds4q_quantize_init(type);
    int fd = open(source, O_RDONLY);
    if (fd < 0) die("open source");
    pthread_t *tids = calloc((size_t)threads, sizeof(*tids));
    Ctx *ctxs = calloc((size_t)threads, sizeof(*ctxs));
    if (!tids || !ctxs) die("alloc threads");
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    uint64_t done_rows = 0, next_progress = progress_rows;
    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);

    for (int t = 0; t < threads; t++) {
        uint64_t start = nrows * (uint64_t)t / (uint64_t)threads;
        uint64_t end = nrows * (uint64_t)(t + 1) / (uint64_t)threads;
        ctxs[t] = (Ctx){ .fd = fd, .type = type, .src_base = src_base, .nrows = nrows,
                         .ncols = ncols, .row_start = start, .row_end = end,
                         .progress_rows = progress_rows, .done_rows = &done_rows,
                         .next_progress = &next_progress, .started = started, .lock = &lock };
    }

    float *imatrix = NULL;
    if (ds4q_requires_imatrix(type)) {
        fprintf(stderr, "candidate type=%s computing synthetic imatrix\n", ds4q_type_name(type));
        run_threads(ctxs, tids, threads, imatrix_worker);
        imatrix = malloc((size_t)ncols * sizeof(float));
        if (!imatrix) die("alloc imatrix final");
        for (uint64_t c = 0; c < ncols; c++) {
            double v = 0.0;
            for (int t = 0; t < threads; t++) v += ctxs[t].imatrix_accum[c];
            imatrix[c] = (float)v;
        }
        for (int t = 0; t < threads; t++) {
            free(ctxs[t].imatrix_accum);
            ctxs[t].imatrix_accum = NULL;
        }
        done_rows = 0;
        next_progress = progress_rows;
        clock_gettime(CLOCK_MONOTONIC, &started);
        for (int t = 0; t < threads; t++) {
            ctxs[t].imatrix = imatrix;
            ctxs[t].done_rows = &done_rows;
            ctxs[t].next_progress = &next_progress;
            ctxs[t].started = started;
        }
    }

    run_threads(ctxs, tids, threads, quant_worker);
    Stats total = {0};
    for (int t = 0; t < threads; t++) {
        total.count += ctxs[t].stats.count;
        total.sum_abs_src += ctxs[t].stats.sum_abs_src;
        total.sum_abs_err += ctxs[t].stats.sum_abs_err;
        total.sum_sq_src += ctxs[t].stats.sum_sq_src;
        total.sum_sq_err += ctxs[t].stats.sum_sq_err;
        if (ctxs[t].stats.max_abs > total.max_abs) total.max_abs = ctxs[t].stats.max_abs;
    }
    printf("count=%llu sum_abs_src=%.17g sum_abs_err=%.17g sum_sq_src=%.17g sum_sq_err=%.17g max_abs=%.17g row_size=%zu\n",
           (unsigned long long)total.count, total.sum_abs_src, total.sum_abs_err,
           total.sum_sq_src, total.sum_sq_err, total.max_abs,
           ds4q_row_size(type, (int64_t)ncols));
    free(imatrix);
    free(ctxs);
    free(tids);
    close(fd);
    return 0;
}

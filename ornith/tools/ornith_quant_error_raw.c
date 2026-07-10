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
    double sum_weighted_sq_src;
    double sum_weighted_sq_err;
    double sum_weight;
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
    const uint32_t *retained;
    uint64_t retained_count;
    uint64_t source_slice_params;
    const float *imatrix;
    uint64_t imatrix_experts;
    uint64_t ncols;
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
    if (strcmp(mode, "q8_0") == 0) return count == 32 ? 34 : 0;
    if (strcmp(mode, "q2_k") == 0) return count == 256 ? 84 : 0;
    if (strcmp(mode, "q4_k") == 0) return count == 256 ? 144 : 0;
    if (strcmp(mode, "iq2_xxs") == 0) return count == 256 ? 66 : 0;
    if (strcmp(mode, "iq1") == 0) return 2 + (count + 7) / 8;
    if (strcmp(mode, "q4") == 0) return 2 + (count + 1) / 2;
    return 0;
}

static int mode_qk(const char *mode, int container_block)
{
    if (strcmp(mode, "q8_0") == 0) return 32;
    if (strcmp(mode, "q2_k") == 0 || strcmp(mode, "q4_k") == 0 || strcmp(mode, "iq2_xxs") == 0) return 256;
    return container_block;
}

static const uint16_t iq2_xxs_grid[256] = {
    0,2,5,8,10,17,20,32,34,40,42,65,68,80,88,97,100,128,130,138,162,257,260,272,277,320,388,408,512,514,546,642,
    1025,1028,1040,1057,1060,1088,1090,1096,1120,1153,1156,1168,1188,1280,1282,1288,1312,1350,1385,1408,1425,1545,1552,1600,1668,1700,2048,2053,2056,2068,2088,2113,
    2116,2128,2130,2184,2308,2368,2562,2580,4097,4100,4112,4129,4160,4192,4228,4240,4245,4352,4360,4384,4432,4442,4480,4644,4677,5120,5128,5152,5157,5193,5248,5400,
    5474,5632,5654,6145,6148,6160,6208,6273,6400,6405,6560,6737,8192,8194,8202,8260,8289,8320,8322,8489,8520,8704,8706,9217,9220,9232,9280,9302,9472,9537,9572,9872,
    10248,10272,10388,10820,16385,16388,16400,16408,16417,16420,16448,16456,16470,16480,16513,16516,16528,16640,16672,16737,16768,16773,16897,16912,16968,16982,17000,17408,17416,17440,17536,17561,
    17682,17700,17920,18433,18436,18448,18496,18501,18688,18776,18785,18818,19013,19088,20480,20488,20497,20505,20512,20608,20616,20740,20802,20900,21137,21648,21650,21770,22017,22100,22528,22545,
    22553,22628,22848,23048,24580,24592,24640,24680,24832,24917,25112,25184,25600,25605,25872,25874,25988,26690,32768,32770,32778,32833,32898,33028,33048,33088,33297,33793,33796,33808,33813,33856,
    33888,34048,34118,34196,34313,34368,34400,34818,35076,35345,36868,36880,36900,36928,37025,37142,37248,37445,37888,37922,37956,38225,39041,39200,40962,41040,41093,41225,41472,42008,43088,43268,
};

static void q4_k_scale_min(int group, const uint8_t *scales, uint8_t *scale, uint8_t *minimum)
{
    if (group < 4) {
        *scale = scales[group] & 63;
        *minimum = scales[group + 4] & 63;
    } else {
        *scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        *minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
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
        fprintf(stderr, "compare mode=%s params=%llu/%llu rate=%.1fMparams/s\n",
                ctx->mode, (unsigned long long)*ctx->done,
                (unsigned long long)ctx->nparams, (double)*ctx->done / elapsed / 1000000.0);
        *ctx->next_progress += ctx->progress_params;
    }
    pthread_mutex_unlock(ctx->lock);
}

static void stats_add(Stats *s, float src, float got, float weight)
{
    double err = (double)src - (double)got;
    double abs_err = fabs(err);
    s->count++;
    s->sum_abs_src += fabs((double)src);
    s->sum_abs_err += abs_err;
    s->sum_sq_src += (double)src * (double)src;
    s->sum_sq_err += err * err;
    if (weight >= 0.0f) {
        s->sum_weighted_sq_src += (double)weight * (double)src * (double)src;
        s->sum_weighted_sq_err += (double)weight * err * err;
        s->sum_weight += weight;
    }
    if (abs_err > s->max_abs) s->max_abs = abs_err;
}

static void read_source_params(Ctx *ctx, uint16_t *dst, uint64_t first_param, uint64_t params)
{
    if (!ctx->retained) {
        read_full(ctx->src_fd, dst, (size_t)params * sizeof(uint16_t),
                  (off_t)(ctx->src_base + first_param * 2));
        return;
    }
    uint64_t written = 0;
    while (written < params) {
        uint64_t output_param = first_param + written;
        uint64_t output_expert = output_param / ctx->source_slice_params;
        uint64_t within = output_param % ctx->source_slice_params;
        if (output_expert >= ctx->retained_count) {
            fprintf(stderr, "retained expert mapping out of range\n");
            exit(2);
        }
        uint64_t take = ctx->source_slice_params - within;
        if (take > params - written) take = params - written;
        uint64_t source_param = (uint64_t)ctx->retained[output_expert] * ctx->source_slice_params + within;
        read_full(ctx->src_fd, dst + written, (size_t)take * sizeof(uint16_t),
                  (off_t)(ctx->src_base + source_param * 2));
        written += take;
    }
}

static float read_quant_value(const char *mode, const uint8_t *q, int in_block)
{
    if (strcmp(mode, "q8_0") == 0) {
        return f16_to_float((uint16_t)q[0] | ((uint16_t)q[1] << 8)) * (float)(int8_t)q[2 + in_block];
    }
    if (strcmp(mode, "q2_k") == 0) {
        int group = in_block / 16;
        int rem = in_block & 127;
        int v = (q[16 + (in_block / 128) * 32 + (rem & 31)] >> ((rem / 32) * 2)) & 3;
        float d = f16_to_float((uint16_t)q[80] | ((uint16_t)q[81] << 8));
        float dmin = f16_to_float((uint16_t)q[82] | ((uint16_t)q[83] << 8));
        return d * (float)(q[group] & 15) * (float)v - dmin * (float)(q[group] >> 4);
    }
    if (strcmp(mode, "q4_k") == 0) {
        uint8_t scale, minimum;
        q4_k_scale_min(in_block / 32, q + 4, &scale, &minimum);
        int rem = in_block & 63;
        int v = (q[16 + (in_block / 64) * 32 + (rem & 31)] >> (rem >= 32 ? 4 : 0)) & 15;
        float d = f16_to_float((uint16_t)q[0] | ((uint16_t)q[1] << 8));
        float dmin = f16_to_float((uint16_t)q[2] | ((uint16_t)q[3] << 8));
        return d * scale * v - dmin * minimum;
    }
    if (strcmp(mode, "iq2_xxs") == 0) {
        int group = in_block / 32;
        int subgroup = (in_block % 32) / 8;
        int item = in_block % 8;
        uint32_t grids, aux;
        memcpy(&grids, q + 2 + group * 8, sizeof(grids));
        memcpy(&aux, q + 6 + group * 8, sizeof(aux));
        uint16_t grid = iq2_xxs_grid[(grids >> (8 * subgroup)) & 255];
        int v = (grid >> (2 * item)) & 3;
        float d = f16_to_float((uint16_t)q[0] | ((uint16_t)q[1] << 8));
        float value = d * (0.5f + (float)(aux >> 28)) * (float)(2 * v + 1);
        return aux & (1u << (7 * subgroup + item)) ? -value : value;
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
        read_source_params(ctx, src, first_param, params);
        read_full(ctx->ornq_fd, ornq, (size_t)qbytes, (off_t)(ctx->ornq_base + b * ctx->full_block_bytes));
        uint64_t src_pos = 0;
        uint64_t qpos = 0;
        for (uint64_t local = 0; local < blocks; local++) {
            uint64_t param = (b + local) * (uint64_t)ctx->block;
            int count = (int)(ctx->nparams - param < (uint64_t)ctx->block ? ctx->nparams - param : (uint64_t)ctx->block);
            for (int i = 0; i < count; i++) {
                uint64_t output_param = param + (uint64_t)i;
                float weight = -1.0f;
                if (ctx->imatrix) {
                    uint64_t output_expert = output_param / ctx->source_slice_params;
                    uint64_t source_expert = ctx->retained ? ctx->retained[output_expert] : output_expert;
                    uint64_t column = output_param % ctx->ncols;
                    if (source_expert >= ctx->imatrix_experts) {
                        fprintf(stderr, "imatrix expert mapping out of range\n");
                        exit(2);
                    }
                    weight = ctx->imatrix[source_expert * ctx->ncols + column];
                }
                stats_add(&ctx->stats, bf16_to_float(src[src_pos + (uint64_t)i]),
                          read_quant_value(ctx->mode, ornq + qpos, i), weight);
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
    if (argc != 12 && argc != 14) {
        fprintf(stderr, "usage: %s SOURCE ORNQ MODE SOURCE_BYTE_OFFSET ORNQ_BYTE_OFFSET NPARAMS BLOCK THREADS PROGRESS_PARAMS SOURCE_SLICE_PARAMS RETAINED_CSV_OR_DASH [IMATRIX_F32 NCOLS]\n", argv[0]);
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
    uint64_t source_slice_params = strtoull(argv[10], NULL, 10);
    uint32_t *retained = NULL;
    uint64_t retained_count = 0;
    if (strcmp(argv[11], "-") != 0) {
        char *copy = strdup(argv[11]);
        if (!copy) die("strdup retained");
        for (char *p = copy; *p; p++) if (*p == ',') retained_count++;
        retained_count++;
        retained = calloc((size_t)retained_count, sizeof(*retained));
        if (!retained) die("alloc retained");
        uint64_t index = 0;
        char *save = NULL;
        for (char *item = strtok_r(copy, ",", &save); item; item = strtok_r(NULL, ",", &save)) {
            retained[index++] = (uint32_t)strtoul(item, NULL, 10);
        }
        free(copy);
        if (!source_slice_params || retained_count * source_slice_params != nparams) {
            fprintf(stderr, "invalid retained expert mapping\n");
            return 2;
        }
    }
    float *imatrix = NULL;
    uint64_t imatrix_experts = 0;
    uint64_t ncols = 0;
    if (argc == 14) {
        ncols = strtoull(argv[13], NULL, 10);
        if (!ncols || !source_slice_params || source_slice_params % ncols) {
            fprintf(stderr, "invalid imatrix tensor geometry\n");
            return 2;
        }
        int fd = open(argv[12], O_RDONLY);
        if (fd < 0) die("open imatrix");
        off_t size = lseek(fd, 0, SEEK_END);
        if (size <= 0 || (uint64_t)size % (ncols * sizeof(float))) {
            fprintf(stderr, "invalid imatrix byte size\n");
            return 2;
        }
        imatrix_experts = (uint64_t)size / (ncols * sizeof(float));
        imatrix = malloc((size_t)size);
        if (!imatrix) die("alloc imatrix");
        read_full(fd, imatrix, (size_t)size, 0);
        close(fd);
        for (uint64_t i = 0; i < imatrix_experts * ncols; i++) {
            if (!isfinite(imatrix[i]) || imatrix[i] < 0.0f) {
                fprintf(stderr, "invalid imatrix value\n");
                return 2;
            }
        }
    }
    if (threads < 1) threads = 1;
    if (strcmp(mode, "bf16") != 0 && strcmp(mode, "iq1") != 0 && strcmp(mode, "q4") != 0 &&
        strcmp(mode, "q8_0") != 0 && strcmp(mode, "q2_k") != 0 && strcmp(mode, "q4_k") != 0 &&
        strcmp(mode, "iq2_xxs") != 0) {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }
    int qk = mode_qk(mode, block);
    if ((strcmp(mode, "q8_0") == 0 || strcmp(mode, "q2_k") == 0 || strcmp(mode, "q4_k") == 0 || strcmp(mode, "iq2_xxs") == 0) &&
        (block != 256 || nparams % (uint64_t)qk != 0)) {
        fprintf(stderr, "%s requires block=256 and aligned nparams\n", mode);
        return 2;
    }
    int src_fd = open(src_path, O_RDONLY);
    if (src_fd < 0) die("open source");
    int ornq_fd = open(ornq_path, O_RDONLY);
    if (ornq_fd < 0) die("open ornq");

    block = qk;
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
            .retained = retained, .retained_count = retained_count,
            .source_slice_params = source_slice_params,
            .imatrix = imatrix, .imatrix_experts = imatrix_experts, .ncols = ncols,
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
        total.sum_weighted_sq_src += ctxs[t].stats.sum_weighted_sq_src;
        total.sum_weighted_sq_err += ctxs[t].stats.sum_weighted_sq_err;
        total.sum_weight += ctxs[t].stats.sum_weight;
        if (ctxs[t].stats.max_abs > total.max_abs) total.max_abs = ctxs[t].stats.max_abs;
    }
    printf("count=%llu sum_abs_src=%.17g sum_abs_err=%.17g sum_sq_src=%.17g sum_sq_err=%.17g max_abs=%.17g sum_weighted_sq_src=%.17g sum_weighted_sq_err=%.17g sum_weight=%.17g\n",
           (unsigned long long)total.count, total.sum_abs_src, total.sum_abs_err,
           total.sum_sq_src, total.sum_sq_err, total.max_abs, total.sum_weighted_sq_src,
           total.sum_weighted_sq_err, total.sum_weight);
    close(src_fd);
    close(ornq_fd);
    free(tids);
    free(ctxs);
    free(retained);
    free(imatrix);
    return 0;
}

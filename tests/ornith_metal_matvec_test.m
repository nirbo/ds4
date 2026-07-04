#include "ornith/ornith.h"
#include "ornith/ornith_metal.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef ORNITH_TESTING
int ornith_metal_test_iq1_slice_many(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    const size_t *slices,
    size_t nslices,
    const float *x,
    size_t x_count,
    size_t x_stride,
    float *out,
    char *err,
    size_t errcap);
int ornith_metal_test_router_q4_b256(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    const float *x,
    size_t x_count,
    size_t rows,
    float *out,
    char *err,
    size_t errcap);
int ornith_metal_test_topk_softmax(const float *scores, size_t n, size_t k, size_t *indices, float *values, char *err, size_t errcap);
int ornith_metal_test_topk_values(const float *scores, size_t n, size_t k, size_t *indices, float *values, char *err, size_t errcap);
int ornith_metal_test_add_sigmoid_scaled_inplace(float *dst, const float *src, float scale, size_t n, char *err, size_t errcap);
int ornith_metal_test_vector_add(const float *a, const float *b, float *out, size_t n, int inplace, char *err, size_t errcap);
int ornith_metal_test_add_rmsnorm(const ornith_model *model, const ornith_tensor_info *weight, const float *x, const float *y, size_t n, float eps, float *out, char *err, size_t errcap);
#endif

static void put_bf16(unsigned char *p, unsigned short raw)
{
    p[0] = (unsigned char)(raw & 255);
    p[1] = (unsigned char)(raw >> 8);
}

static void write_file(const char *path, const unsigned char *data, size_t n)
{
    FILE *fp = fopen(path, "wb");
    assert(fp);
    assert(fwrite(data, 1, n, fp) == n);
    assert(fclose(fp) == 0);
}

static void fill_q4_b256(unsigned char *payload, size_t rows, size_t cols)
{
    assert((cols % 256) == 0);
    size_t off = 0;
    for (size_t r = 0; r < rows; r++) {
        for (size_t b = 0; b < cols / 256; b++) {
            put_bf16(payload + off, 0x3f80);
            for (size_t i = 0; i < 128; i++) {
                size_t c0 = b * 256 + i * 2;
                int q0 = (int)((r * 7 + c0 * 3) % 15) - 7;
                int q1 = (int)((r * 7 + (c0 + 1) * 3) % 15) - 7;
                payload[off + 2 + i] = (unsigned char)((q0 & 15) | ((q1 & 15) << 4));
            }
            off += 130;
        }
    }
}

static void fill_iq1_b256(unsigned char *payload, size_t rows, size_t cols)
{
    assert((cols % 256) == 0);
    size_t off = 0;
    for (size_t r = 0; r < rows; r++) {
        for (size_t b = 0; b < cols / 256; b++) {
            put_bf16(payload + off, 0x3f80);
            for (size_t i = 0; i < 32; i++) {
                unsigned char bits = 0;
                for (size_t bit = 0; bit < 8; bit++) {
                    size_t c = b * 256 + i * 8 + bit;
                    if (((r * 11 + c * 5) & 7) >= 3) bits |= (unsigned char)(1u << bit);
                }
                payload[off + 2 + i] = bits;
            }
            off += 34;
        }
    }
}

static void near_array(const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        assert(fabsf(a[i] - b[i]) < 0.0001f);
    }
}

static void softmax_inplace(float *values, size_t n)
{
    float maxv = values[0];
    for (size_t i = 1; i < n; i++) if (values[i] > maxv) maxv = values[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        values[i] = expf(values[i] - maxv);
        sum += values[i];
    }
    for (size_t i = 0; i < n; i++) values[i] /= sum;
}

int main(void)
{
    if (!ornith_metal_available()) {
        puts("ornith_metal_matvec_test: skipped");
        return 0;
    }

    char dir[] = "/tmp/ornith-metal-matvec-XXXXXX";
    assert(mkdtemp(dir));
    char shard[512];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00122.ornq", dir);
    char shard2[512];
    snprintf(shard2, sizeof(shard2), "%s/model-00002-of-00122.ornq", dir);
    unsigned char bytes[128] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    put_bf16(bytes + 16, 0x3f80);
    put_bf16(bytes + 18, 0x4000);
    put_bf16(bytes + 20, 0x4040);
    put_bf16(bytes + 22, 0x4080);
    put_bf16(bytes + 24, 0x3f80);
    put_bf16(bytes + 26, 0x3f80);
    put_bf16(bytes + 28, 0x4080);
    put_bf16(bytes + 30, 0x4040);
    put_bf16(bytes + 32, 0x4000);
    put_bf16(bytes + 34, 0x3f80);
    put_bf16(bytes + 40, 0x3f80); bytes[42] = 0x21; bytes[43] = 0x43;
    put_bf16(bytes + 44, 0x3f80); bytes[46] = 0x0f; bytes[47] = 0x21;
    put_bf16(bytes + 56, 0x3f80); bytes[58] = 0x05;
    put_bf16(bytes + 64, 0x3f80);
    put_bf16(bytes + 66, 0x0000);
    put_bf16(bytes + 68, 0x4000);
    put_bf16(bytes + 70, 0x0000);
    put_bf16(bytes + 72, 0x0000);
    put_bf16(bytes + 74, 0x4040);
    write_file(shard, bytes, sizeof(bytes));
    enum {
        r4_rows = 5,
        r4_cols = 256,
        q4_r4_payload = r4_rows * 130,
        iq1_r4_payload = r4_rows * 34,
        q4_r4_offset = 16,
        iq1_r4_offset = q4_r4_offset + q4_r4_payload,
        r4_size = iq1_r4_offset + iq1_r4_payload,
    };
    unsigned char *bytes2 = calloc(r4_size, 1);
    assert(bytes2);
    memcpy(bytes2, "ORNQ1", 5);
    fill_q4_b256(bytes2 + q4_r4_offset, r4_rows, r4_cols);
    fill_iq1_b256(bytes2 + iq1_r4_offset, r4_rows, r4_cols);
    write_file(shard2, bytes2, r4_size);
    free(bytes2);

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t128\t16\t4\t6\n");
    fprintf(fp, "shard\tmodel-00002-of-00122.ornq\t%d\t16\t256\t2\n", r4_size);
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t8\t4\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t4\t2\t-1\tglobal\tmodel.language_model.norm.weight\t2\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t28\t8\t4\t-1\tglobal\tlm_head.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.out_proj.weight\tmodel-00001-of-00122.ornq\tq4\t40\t8\t8\t0\tattention\tlinear_attn.out_proj.weight\t2,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tiq1\t56\t3\t4\t0\trouted_expert\tmlp.experts.gate_up_proj\t1,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t64\t12\t6\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t3,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.test.q4_b256_r4\tmodel-00002-of-00122.ornq\tq4\t%d\t%d\t%d\t0\tattention\ttest.q4_b256_r4\t%d,%d\n", q4_r4_offset, q4_r4_payload, r4_rows * r4_cols, r4_rows, r4_cols);
    fprintf(fp, "tensor\tmodel.language_model.layers.0.test.iq1_b256_r4\tmodel-00002-of-00122.ornq\tiq1\t%d\t%d\t%d\t0\trouted_expert\ttest.iq1_b256_r4\t1,%d,%d\n", iq1_r4_offset, iq1_r4_payload, r4_rows * r4_cols, r4_rows, r4_cols);
    assert(fclose(fp) == 0);

    char err[512] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));

#ifdef ORNITH_TESTING
    const float top_scores[9] = {1, -2, 3, 3, 0.5f, 8, -1, 7, 4};
    size_t top_cpu_idx[4] = {0}, top_gpu_idx[4] = {0};
    float top_cpu_val[4] = {0}, top_gpu_val[4] = {0};
    assert(ornith_topk(top_scores, 9, 4, top_cpu_idx, top_cpu_val));
    softmax_inplace(top_cpu_val, 4);
    assert(ornith_metal_test_topk_softmax(top_scores, 9, 4, top_gpu_idx, top_gpu_val, err, sizeof(err)));
    for (size_t i = 0; i < 4; i++) assert(top_cpu_idx[i] == top_gpu_idx[i]);
    near_array(top_cpu_val, top_gpu_val, 4);
    memset(top_gpu_idx, 0, sizeof(top_gpu_idx));
    memset(top_gpu_val, 0, sizeof(top_gpu_val));
    assert(ornith_topk(top_scores, 9, 4, top_cpu_idx, top_cpu_val));
    assert(ornith_metal_test_topk_values(top_scores, 9, 4, top_gpu_idx, top_gpu_val, err, sizeof(err)));
    for (size_t i = 0; i < 4; i++) assert(top_cpu_idx[i] == top_gpu_idx[i]);
    near_array(top_cpu_val, top_gpu_val, 4);
    memset(top_gpu_idx, 0, sizeof(top_gpu_idx));
    memset(top_gpu_val, 0, sizeof(top_gpu_val));
    assert(ornith_topk(top_scores, 9, 1, top_cpu_idx, top_cpu_val));
    assert(ornith_metal_test_topk_values(top_scores, 9, 1, top_gpu_idx, top_gpu_val, err, sizeof(err)));
    assert(top_cpu_idx[0] == top_gpu_idx[0]);
    near_array(top_cpu_val, top_gpu_val, 1);

    float add_dst[3] = {1, -2, 0.5f};
    const float add_src[3] = {4, 8, -2};
    const float add_scale = -0.25f;
    float add_expect[3] = {add_dst[0], add_dst[1], add_dst[2]};
    float add_w = 1.0f / (1.0f + expf(-add_scale));
    for (size_t i = 0; i < 3; i++) add_expect[i] += add_w * add_src[i];
    assert(ornith_metal_test_add_sigmoid_scaled_inplace(add_dst, add_src, add_scale, 3, err, sizeof(err)));
    near_array(add_expect, add_dst, 3);
    const float va[4] = {1, -2, 3.5f, 0};
    const float vb[4] = {4, 8, -1.5f, 7};
    const float vsum[4] = {5, 6, 2, 7};
    float vout[4] = {0};
    assert(ornith_metal_test_vector_add(va, vb, vout, 4, 0, err, sizeof(err)));
    near_array(vsum, vout, 4);
    memset(vout, 0, sizeof(vout));
    assert(ornith_metal_test_vector_add(va, vb, vout, 4, 1, err, sizeof(err)));
    near_array(vsum, vout, 4);
#endif

    const float xn[2] = {3, 4};
    float cpun[2] = {0}, gpun[2] = {0};
    const ornith_tensor_info *norm_t = ornith_model_find_tensor(model, "model.language_model.norm.weight");
    assert(norm_t);
    assert(ornith_rmsnorm(model, norm_t, xn, 2, 1e-6f, cpun));
    assert(ornith_metal_rmsnorm(model, norm_t, xn, 2, 1e-6f, gpun, err, sizeof(err)));
    near_array(cpun, gpun, 2);
#ifdef ORNITH_TESTING
    const float yn[2] = {-1, 5};
    const float sum_n[2] = {xn[0] + yn[0], xn[1] + yn[1]};
    assert(ornith_rmsnorm(model, norm_t, sum_n, 2, 1e-6f, cpun));
    assert(ornith_metal_test_add_rmsnorm(model, norm_t, xn, yn, 2, 1e-6f, gpun, err, sizeof(err)));
    near_array(cpun, gpun, 2);
#endif

    const float x4[4] = {1, 1, 1, 1};
    float cpu4[2] = {0}, gpu4[2] = {0};
    const ornith_tensor_info *t = ornith_model_find_layer_tensor(model, 0, "linear_attn.out_proj.weight");
    assert(t);
    assert(ornith_tensor_matvec(model, t, x4, 4, cpu4));
    assert(ornith_metal_tensor_matvec(model, t, 0, x4, 4, gpu4, err, sizeof(err)));
    near_array(cpu4, gpu4, 2);

    float cpu1[1] = {0}, gpu1[1] = {0};
    t = ornith_model_find_layer_tensor(model, 0, "mlp.experts.gate_up_proj");
    assert(t);
    assert(ornith_tensor_matvec(model, t, x4, 4, cpu1));
    assert(ornith_metal_tensor_matvec(model, t, 0, x4, 4, gpu1, err, sizeof(err)));
    near_array(cpu1, gpu1, 1);

    const float x2[2] = {1, 2};
    float cpu2[3] = {0}, gpu2[3] = {0};
    t = ornith_model_find_layer_tensor(model, 0, "mlp.shared_expert.up_proj.weight");
    assert(t);
    assert(ornith_tensor_matvec(model, t, x2, 2, cpu2));
    assert(ornith_metal_tensor_matvec(model, t, 0, x2, 2, gpu2, err, sizeof(err)));
    near_array(cpu2, gpu2, 3);

    float x256[r4_cols];
    for (size_t i = 0; i < r4_cols; i++) x256[i] = (float)((int)(i % 17) - 8) / 16.0f;
    float cpu5[r4_rows] = {0}, gpu5[r4_rows] = {0};
    t = ornith_model_find_layer_tensor(model, 0, "test.q4_b256_r4");
    assert(t);
    assert(ornith_tensor_matvec(model, t, x256, r4_cols, cpu5));
    assert(ornith_metal_tensor_matvec(model, t, 0, x256, r4_cols, gpu5, err, sizeof(err)));
    near_array(cpu5, gpu5, r4_rows);
#ifdef ORNITH_TESTING
    memset(gpu5, 0, sizeof(gpu5));
    assert(ornith_metal_test_router_q4_b256(model, t, x256, r4_cols, r4_rows, gpu5, err, sizeof(err)));
    near_array(cpu5, gpu5, r4_rows);
#endif

    memset(cpu5, 0, sizeof(cpu5));
    memset(gpu5, 0, sizeof(gpu5));
    t = ornith_model_find_layer_tensor(model, 0, "test.iq1_b256_r4");
    assert(t);
    assert(ornith_tensor_slice_matvec(model, t, 0, x256, r4_cols, cpu5));
    size_t slice0[1] = {0};
#ifdef ORNITH_TESTING
    assert(ornith_metal_test_iq1_slice_many(model, t, slice0, 1, x256, r4_cols, 0, gpu5, err, sizeof(err)));
#else
    assert(ornith_metal_tensor_matvec(model, t, 0, x256, r4_cols, gpu5, err, sizeof(err)));
#endif
    near_array(cpu5, gpu5, r4_rows);

    ornith_model_close(model);
    remove(catalog);
    remove(shard);
    remove(shard2);
    rmdir(dir);
    puts("ornith_metal_matvec_test: ok");
    return 0;
}

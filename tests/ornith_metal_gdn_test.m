#include "ornith/ornith_metal.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float sigmoidf_ref(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float siluf_ref(float x)
{
    return x * sigmoidf_ref(x);
}

static float softplusf_ref(float x)
{
    return x <= 20.0f ? log1pf(expf(x)) : x;
}

static void ref_gdn(
    const float *qkv,
    const float *z,
    const float *a,
    const float *b,
    const float *alog,
    const float *dt,
    const float *norm_w,
    float *ssm,
    size_t value_heads,
    size_t head_v,
    size_t key_heads,
    size_t head_k,
    float *gated)
{
    size_t key_dim = key_heads * head_k;
    size_t value_dim = value_heads * head_v;
    float core[32] = {0};
    assert(value_dim <= 32);
    for (size_t hv = 0; hv < value_heads; hv++) {
        size_t h = hv / (value_heads / key_heads);
        const float *q = qkv + h * head_k;
        const float *k = qkv + key_dim + h * head_k;
        const float *v = qkv + key_dim * 2 + hv * head_v;
        float qss = 0.0f, kss = 0.0f;
        for (size_t i = 0; i < head_k; i++) {
            qss += q[i] * q[i];
            kss += k[i] * k[i];
        }
        float qscale = 1.0f / sqrtf(qss + 1e-6f);
        float kscale = 1.0f / sqrtf(kss + 1e-6f);
        float decay = expf(-expf(alog[hv]) * softplusf_ref(a[hv] + dt[hv]));
        float beta = sigmoidf_ref(b[hv]);
        float ss = 0.0f;
        for (size_t vi = 0; vi < head_v; vi++) {
            float *row = ssm + (hv * head_v + vi) * head_k;
            float proj = 0.0f;
            for (size_t ki = 0; ki < head_k; ki++) {
                row[ki] *= decay;
                proj += row[ki] * k[ki] * kscale;
            }
            float vv = (v[vi] - proj) * beta;
            float sum = 0.0f;
            for (size_t ki = 0; ki < head_k; ki++) {
                row[ki] += vv * k[ki] * kscale;
                sum += row[ki] * q[ki] * qscale;
            }
            float c = sum / sqrtf((float)head_k);
            core[hv * head_v + vi] = c;
            ss += c * c;
        }
        float scale = 1.0f / sqrtf(ss / (float)head_v + 1e-6f);
        for (size_t vi = 0; vi < head_v; vi++) {
            size_t o = hv * head_v + vi;
            gated[o] = core[o] * scale * norm_w[vi] * siluf_ref(z[o]);
        }
    }
}

static void near_array(const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        assert(fabsf(a[i] - b[i]) < 0.0002f);
    }
}

static void perturb(float *dst, const float *src, size_t n, float step_scale)
{
    for (size_t i = 0; i < n; i++) {
        float sign = (i & 1) ? -1.0f : 1.0f;
        dst[i] = src[i] + sign * step_scale * (float)((i % 5) + 1);
    }
}

int main(void)
{
    if (!ornith_metal_available()) {
        puts("ornith_metal_gdn_test: skipped");
        return 0;
    }
    enum { value_heads = 2, head_v = 4, key_heads = 1, head_k = 3 };
    float qkv[key_heads * head_k * 2 + value_heads * head_v] = {
        0.3f, -0.4f, 0.7f,
        -0.2f, 0.5f, 0.6f,
        0.1f, -0.3f, 0.8f, 0.4f,
        -0.5f, 0.2f, 0.9f, -0.7f,
    };
    float z[value_heads * head_v] = {0.2f, -0.1f, 0.6f, 0.9f, -0.4f, 0.3f, 0.7f, -0.8f};
    float a[value_heads] = {0.25f, -0.35f};
    float b[value_heads] = {-0.2f, 0.45f};
    float alog[value_heads] = {-1.2f, -0.7f};
    float dt[value_heads] = {0.1f, -0.05f};
    float norm_w[head_v] = {0.9f, 1.1f, 0.7f, 1.3f};
    float cpu_ssm[value_heads * head_v * head_k];
    float gpu_ssm[value_heads * head_v * head_k];
    for (size_t i = 0; i < value_heads * head_v * head_k; i++) {
        cpu_ssm[i] = gpu_ssm[i] = ((float)(int)(i % 7) - 3.0f) * 0.05f;
    }
    float cpu[value_heads * head_v] = {0};
    float gpu[value_heads * head_v] = {0};
    ref_gdn(qkv, z, a, b, alog, dt, norm_w, cpu_ssm, value_heads, head_v, key_heads, head_k, cpu);
    char err[512] = {0};
    assert(ornith_metal_gdn_recurrent_step(qkv, z, a, b, alog, dt, norm_w, gpu_ssm, value_heads, head_v, key_heads, head_k, gpu, err, sizeof(err)));
    near_array(cpu, gpu, value_heads * head_v);
    near_array(cpu_ssm, gpu_ssm, value_heads * head_v * head_k);

    memcpy(cpu_ssm, gpu_ssm, sizeof(cpu_ssm));
    for (size_t step = 1; step <= 4; step++) {
        float qkv_step[key_heads * head_k * 2 + value_heads * head_v];
        float z_step[value_heads * head_v];
        float a_step[value_heads];
        float b_step[value_heads];
        perturb(qkv_step, qkv, sizeof(qkv) / sizeof(qkv[0]), 0.01f * (float)step);
        perturb(z_step, z, sizeof(z) / sizeof(z[0]), 0.015f * (float)step);
        perturb(a_step, a, sizeof(a) / sizeof(a[0]), 0.02f * (float)step);
        perturb(b_step, b, sizeof(b) / sizeof(b[0]), 0.025f * (float)step);
        memset(cpu, 0, sizeof(cpu));
        memset(gpu, 0, sizeof(gpu));
        ref_gdn(qkv_step, z_step, a_step, b_step, alog, dt, norm_w, cpu_ssm, value_heads, head_v, key_heads, head_k, cpu);
        assert(ornith_metal_gdn_recurrent_step(qkv_step, z_step, a_step, b_step, alog, dt, norm_w, gpu_ssm, value_heads, head_v, key_heads, head_k, gpu, err, sizeof(err)));
        near_array(cpu, gpu, value_heads * head_v);
        near_array(cpu_ssm, gpu_ssm, value_heads * head_v * head_k);
    }
    puts("ornith_metal_gdn_test: ok");
    return 0;
}

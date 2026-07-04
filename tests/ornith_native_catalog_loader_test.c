#include "ornith/ornith.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef ORNITH_TESTING
int ornith_test_unpack_attention_q_gate_interleaved(const float *mixed, size_t heads, size_t head_dim, float *q, float *gate);
#endif

static void write_file(const char *path, const unsigned char *data, size_t n)
{
    FILE *fp = fopen(path, "wb");
    assert(fp);
    assert(fwrite(data, 1, n, fp) == n);
    assert(fclose(fp) == 0);
}

static void put_bf16(unsigned char *p, unsigned short raw)
{
    p[0] = (unsigned char)(raw & 255);
    p[1] = (unsigned char)(raw >> 8);
}

static void nearf(float got, float want)
{
    assert(fabsf(got - want) < 0.0001f);
}

static float sigf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void ref_matvec(const ornith_model *model, const ornith_tensor_info *t, const float *x, float *out)
{
    assert(t->ndim == 2);
    for (int64_t r = 0; r < t->shape[0]; r++) {
        float acc = 0.0f;
        for (int64_t c = 0; c < t->shape[1]; c++) {
            float v = 0.0f;
            assert(ornith_tensor_value(model, t, (uint64_t)(r * t->shape[1] + c), &v));
            acc += v * x[c];
        }
        out[r] = acc;
    }
}

static void ref_slice_matvec(const ornith_model *model, const ornith_tensor_info *t, uint64_t slice, const float *x, float *out)
{
    assert(t->ndim == 3);
    uint64_t base = slice * (uint64_t)t->shape[1] * (uint64_t)t->shape[2];
    for (int64_t r = 0; r < t->shape[1]; r++) {
        float acc = 0.0f;
        for (int64_t c = 0; c < t->shape[2]; c++) {
            float v = 0.0f;
            assert(ornith_tensor_value(model, t, base + (uint64_t)(r * t->shape[2] + c), &v));
            acc += v * x[c];
        }
        out[r] = acc;
    }
}

static int probe_real(const char *catalog, const char *shard_dir)
{
    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_moe_layout(model, err, sizeof(err)));
    assert(ornith_model_validate_attention_layout(model, err, sizeof(err)));
    assert(ornith_model_validate_shards(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));
    printf("ornith_native_catalog_loader_test: shards=%zu tensors=%zu layers=%zu\n",
           ornith_model_shard_count(model), ornith_model_tensor_count(model), ornith_model_layer_count(model));
    ornith_model_close(model);
    return 0;
}

static void test_layout_validator(void)
{
    char dir[] = "/tmp/ornith-native-layout-XXXXXX";
    assert(mkdtemp(dir));

    char shard[512];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00122.ornq", dir);
    unsigned char bytes[128] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    write_file(shard, bytes, sizeof(bytes));

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t128\t16\t4\t11\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t8\t4\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t4\t2\t-1\tglobal\tmodel.language_model.norm.weight\t2\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t28\t8\t4\t-1\tglobal\tlm_head.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.input_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t36\t4\t2\t0\tnorm\tinput_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.gate.weight\tmodel-00001-of-00122.ornq\tbf16\t40\t8\t4\t0\trouter\tmlp.gate.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tbf16\t48\t32\t16\t0\trouted_expert\tmlp.experts.gate_up_proj\t2,4,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.down_proj\tmodel-00001-of-00122.ornq\tbf16\t80\t16\t8\t0\trouted_expert\tmlp.experts.down_proj\t2,2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.gate_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t96\t8\t4\t0\tshared_expert\tmlp.shared_expert.gate_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t104\t8\t4\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.down_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t112\t8\t4\t0\tshared_expert\tmlp.shared_expert.down_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert_gate.weight\tmodel-00001-of-00122.ornq\tbf16\t120\t4\t2\t0\trouter\tmlp.shared_expert_gate.weight\t1,2\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_moe_layout(model, err, sizeof(err)));
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
}

static void test_attention_layout_validator(void)
{
    char dir[] = "/tmp/ornith-native-attention-XXXXXX";
    assert(mkdtemp(dir));

    char shard[512];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00122.ornq", dir);
    unsigned char bytes[2048] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    write_file(shard, bytes, sizeof(bytes));

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t2048\t16\t4\t22\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t8\t4\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t4\t2\t-1\tglobal\tmodel.language_model.norm.weight\t2\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t28\t8\t4\t-1\tglobal\tlm_head.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.input_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t36\t4\t2\t0\tnorm\tinput_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.post_attention_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t40\t4\t2\t0\tnorm\tpost_attention_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.A_log\tmodel-00001-of-00122.ornq\tbf16\t44\t4\t2\t0\tattention\tlinear_attn.A_log\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.dt_bias\tmodel-00001-of-00122.ornq\tbf16\t48\t4\t2\t0\tattention\tlinear_attn.dt_bias\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t52\t4\t2\t0\tattention\tlinear_attn.norm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.conv1d.weight\tmodel-00001-of-00122.ornq\tq4\t60\t16\t12\t0\tattention\tlinear_attn.conv1d.weight\t6,1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_a.weight\tmodel-00001-of-00122.ornq\tq4\t76\t16\t4\t0\tattention\tlinear_attn.in_proj_a.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_b.weight\tmodel-00001-of-00122.ornq\tq4\t92\t16\t4\t0\tattention\tlinear_attn.in_proj_b.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_qkv.weight\tmodel-00001-of-00122.ornq\tq4\t108\t16\t12\t0\tattention\tlinear_attn.in_proj_qkv.weight\t6,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_z.weight\tmodel-00001-of-00122.ornq\tq4\t124\t16\t8\t0\tattention\tlinear_attn.in_proj_z.weight\t4,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.out_proj.weight\tmodel-00001-of-00122.ornq\tq4\t140\t16\t8\t0\tattention\tlinear_attn.out_proj.weight\t2,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.gate.weight\tmodel-00001-of-00122.ornq\tbf16\t156\t8\t4\t0\trouter\tmlp.gate.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tbf16\t164\t32\t16\t0\trouted_expert\tmlp.experts.gate_up_proj\t2,4,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.down_proj\tmodel-00001-of-00122.ornq\tbf16\t196\t16\t8\t0\trouted_expert\tmlp.experts.down_proj\t2,2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.gate_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t212\t8\t4\t0\tshared_expert\tmlp.shared_expert.gate_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t220\t8\t4\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.down_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t228\t8\t4\t0\tshared_expert\tmlp.shared_expert.down_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert_gate.weight\tmodel-00001-of-00122.ornq\tbf16\t236\t4\t2\t0\trouter\tmlp.shared_expert_gate.weight\t1,2\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_moe_layout(model, err, sizeof(err)));
    assert(ornith_model_validate_attention_layout(model, err, sizeof(err)));
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
}

static void test_decode_self_attention_first_token(void)
{
    char dir[] = "/tmp/ornith-native-decode-XXXXXX";
    assert(mkdtemp(dir));

    char shard[512];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00122.ornq", dir);
    unsigned char bytes[512] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    for (size_t off = 16; off + 1 < sizeof(bytes); off += 2) put_bf16(bytes + off, 0x0000);
    put_bf16(bytes + 16, 0x3f80);
    put_bf16(bytes + 20, 0x3f80);
    put_bf16(bytes + 24, 0x3f80);
    put_bf16(bytes + 28, 0x3f80);
    put_bf16(bytes + 32, 0x3f80);
    put_bf16(bytes + 36, 0x3f80);
    put_bf16(bytes + 38, 0x3f80);
    put_bf16(bytes + 56, 0x3f80);
    put_bf16(bytes + 62, 0x3f80);
    put_bf16(bytes + 64, 0x3f80);
    put_bf16(bytes + 70, 0x3f80);
    put_bf16(bytes + 72, 0x3f80);
    put_bf16(bytes + 80, 0x3f80);
    put_bf16(bytes + 84, 0x3f80);
    put_bf16(bytes + 88, 0x3f80);
    write_file(shard, bytes, sizeof(bytes));

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t512\t16\t4\t23\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t4\t2\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t20\t4\t2\t-1\tglobal\tmodel.language_model.norm.weight\t2\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t4\t2\t-1\tglobal\tlm_head.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.input_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t28\t4\t2\t0\tnorm\tinput_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.post_attention_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t32\t4\t2\t0\tnorm\tpost_attention_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.q_norm.weight\tmodel-00001-of-00122.ornq\tbf16\t36\t2\t1\t0\tattention\tself_attn.q_norm.weight\t1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.k_norm.weight\tmodel-00001-of-00122.ornq\tbf16\t38\t2\t1\t0\tattention\tself_attn.k_norm.weight\t1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.q_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t40\t8\t4\t0\tattention\tself_attn.q_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.k_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t48\t8\t4\t0\tattention\tself_attn.k_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.v_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t56\t8\t4\t0\tattention\tself_attn.v_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.o_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t64\t8\t4\t0\tattention\tself_attn.o_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.gate.weight\tmodel-00001-of-00122.ornq\tbf16\t72\t4\t2\t0\trouter\tmlp.gate.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tbf16\t80\t8\t4\t0\trouted_expert\tmlp.experts.gate_up_proj\t1,2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.down_proj\tmodel-00001-of-00122.ornq\tbf16\t88\t4\t2\t0\trouted_expert\tmlp.experts.down_proj\t1,2,1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.gate_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t96\t4\t2\t0\tshared_expert\tmlp.shared_expert.gate_proj.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t104\t4\t2\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.down_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t112\t4\t2\t0\tshared_expert\tmlp.shared_expert.down_proj.weight\t2,1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert_gate.weight\tmodel-00001-of-00122.ornq\tbf16\t120\t4\t2\t0\trouter\tmlp.shared_expert_gate.weight\t1,2\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_moe_layout(model, err, sizeof(err)));
    assert(ornith_model_validate_attention_layout(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));
    float x[2] = {1.0f, 0.0f};
    float out[2] = {0.0f, 0.0f};
    assert(ornith_layer_decode_smoke(model, 0, x, 2, 1, out));
    assert(out[0] > 1.4f);
    nearf(out[1], 0.0f);
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
}

static void test_decode_self_attention_sequence(void)
{
    char dir[] = "/tmp/ornith-native-self-seq-XXXXXX";
    assert(mkdtemp(dir));

    char shard[512];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00122.ornq", dir);
    unsigned char bytes[512] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    for (size_t off = 16; off + 1 < sizeof(bytes); off += 2) put_bf16(bytes + off, 0x0000);
    put_bf16(bytes + 16, 0x3f80);
    put_bf16(bytes + 22, 0x3f80);
    put_bf16(bytes + 24, 0x3f80);
    put_bf16(bytes + 26, 0x3f80);
    put_bf16(bytes + 28, 0x3f80);
    put_bf16(bytes + 32, 0x3f80);
    put_bf16(bytes + 34, 0x3f80);
    put_bf16(bytes + 36, 0x3f80);
    put_bf16(bytes + 38, 0x3f80);
    put_bf16(bytes + 46, 0x3f80);
    put_bf16(bytes + 52, 0x3f80);
    put_bf16(bytes + 60, 0x3f80);
    put_bf16(bytes + 68, 0x3f80);
    write_file(shard, bytes, sizeof(bytes));

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t512\t16\t4\t21\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t8\t4\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t4\t2\t-1\tglobal\tmodel.language_model.norm.weight\t2\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t28\t4\t2\t-1\tglobal\tlm_head.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.input_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t32\t4\t2\t0\tnorm\tinput_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.post_attention_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t36\t4\t2\t0\tnorm\tpost_attention_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.q_norm.weight\tmodel-00001-of-00122.ornq\tbf16\t40\t2\t1\t0\tattention\tself_attn.q_norm.weight\t1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.k_norm.weight\tmodel-00001-of-00122.ornq\tbf16\t42\t2\t1\t0\tattention\tself_attn.k_norm.weight\t1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.q_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t44\t8\t4\t0\tattention\tself_attn.q_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.k_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t52\t8\t4\t0\tattention\tself_attn.k_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.v_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t60\t8\t4\t0\tattention\tself_attn.v_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.self_attn.o_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t68\t8\t4\t0\tattention\tself_attn.o_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.gate.weight\tmodel-00001-of-00122.ornq\tbf16\t76\t4\t2\t0\trouter\tmlp.gate.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tbf16\t80\t8\t4\t0\trouted_expert\tmlp.experts.gate_up_proj\t1,2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.down_proj\tmodel-00001-of-00122.ornq\tbf16\t88\t4\t2\t0\trouted_expert\tmlp.experts.down_proj\t1,2,1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.gate_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t96\t4\t2\t0\tshared_expert\tmlp.shared_expert.gate_proj.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t104\t4\t2\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t1,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.down_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t112\t4\t2\t0\tshared_expert\tmlp.shared_expert.down_proj.weight\t2,1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert_gate.weight\tmodel-00001-of-00122.ornq\tbf16\t120\t4\t2\t0\trouter\tmlp.shared_expert_gate.weight\t1,2\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_moe_layout(model, err, sizeof(err)));
    assert(ornith_model_validate_attention_layout(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));
    uint64_t seq[2] = {0, 1};
    size_t idx[1] = {0};
    float val[1] = {0};
    assert(ornith_decode_sequence_smoke_limited(model, seq, 2, 1, 1, 1, 1, idx, val));
    assert(idx[0] == 0);
    assert(val[0] > 0.3f);
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
}

static void test_decode_linear_attention_first_token(void)
{
    char dir[] = "/tmp/ornith-native-linear-decode-XXXXXX";
    assert(mkdtemp(dir));

    char shard[512];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00122.ornq", dir);
    unsigned char bytes[2048] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    for (size_t off = 16; off + 1 < sizeof(bytes); off += 2) put_bf16(bytes + off, 0x0000);
    for (size_t i = 0; i < 8; i++) put_bf16(bytes + 16 + i * 2, 0x3f80);
    for (size_t i = 0; i < 4; i++) put_bf16(bytes + 64 + i * 2, 0x3f80);
    put_bf16(bytes + 72 + (0 * 4 + 3) * 2, 0x3f80);
    put_bf16(bytes + 72 + (4 * 4 + 3) * 2, 0x3f80);
    put_bf16(bytes + 72 + (8 * 4 + 3) * 2, 0x3f80);
    put_bf16(bytes + 392 + (0 * 8 + 0) * 2, 0x3f80);
    put_bf16(bytes + 392 + (4 * 8 + 0) * 2, 0x3f80);
    put_bf16(bytes + 392 + (8 * 8 + 0) * 2, 0x3f80);
    put_bf16(bytes + 392 + (0 * 8 + 1) * 2, 0x3f80);
    put_bf16(bytes + 392 + (4 * 8 + 1) * 2, 0x3f80);
    put_bf16(bytes + 776 + (0 * 8 + 0) * 2, 0x3f80);
    put_bf16(bytes + 776 + (0 * 8 + 1) * 2, 0x3f80);
    put_bf16(bytes + 1032 + (0 * 16 + 0) * 2, 0x3f80);
    put_bf16(bytes + 1416, 0x3f80);
    put_bf16(bytes + 1416 + (8 + 1) * 2, 0x3f80);
    for (size_t i = 0; i < 8; i++) put_bf16(bytes + 1448 + i * 2, 0x3f80);
    put_bf16(bytes + 1464, 0x3f80);
    write_file(shard, bytes, sizeof(bytes));

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t2048\t16\t4\t21\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t1416\t32\t16\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,8\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t1448\t16\t8\t-1\tglobal\tmodel.language_model.norm.weight\t8\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t1464\t16\t8\t-1\tglobal\tlm_head.weight\t1,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.input_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t16\t8\t0\tnorm\tinput_layernorm.weight\t8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.post_attention_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t32\t16\t8\t0\tnorm\tpost_attention_layernorm.weight\t8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.A_log\tmodel-00001-of-00122.ornq\tbf16\t48\t8\t4\t0\tattention\tlinear_attn.A_log\t4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.dt_bias\tmodel-00001-of-00122.ornq\tbf16\t56\t8\t4\t0\tattention\tlinear_attn.dt_bias\t4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t64\t8\t4\t0\tattention\tlinear_attn.norm.weight\t4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.conv1d.weight\tmodel-00001-of-00122.ornq\tbf16\t72\t192\t96\t0\tattention\tlinear_attn.conv1d.weight\t24,1,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_a.weight\tmodel-00001-of-00122.ornq\tbf16\t264\t64\t32\t0\tattention\tlinear_attn.in_proj_a.weight\t4,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_b.weight\tmodel-00001-of-00122.ornq\tbf16\t328\t64\t32\t0\tattention\tlinear_attn.in_proj_b.weight\t4,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_qkv.weight\tmodel-00001-of-00122.ornq\tbf16\t392\t384\t192\t0\tattention\tlinear_attn.in_proj_qkv.weight\t24,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.in_proj_z.weight\tmodel-00001-of-00122.ornq\tbf16\t776\t256\t128\t0\tattention\tlinear_attn.in_proj_z.weight\t16,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.out_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t1032\t256\t128\t0\tattention\tlinear_attn.out_proj.weight\t8,16\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.gate.weight\tmodel-00001-of-00122.ornq\tbf16\t1288\t16\t8\t0\trouter\tmlp.gate.weight\t1,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tbf16\t1304\t32\t16\t0\trouted_expert\tmlp.experts.gate_up_proj\t1,2,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.down_proj\tmodel-00001-of-00122.ornq\tbf16\t1336\t16\t8\t0\trouted_expert\tmlp.experts.down_proj\t1,8,1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.gate_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t1352\t16\t8\t0\tshared_expert\tmlp.shared_expert.gate_proj.weight\t1,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t1368\t16\t8\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t1,8\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.down_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t1384\t16\t8\t0\tshared_expert\tmlp.shared_expert.down_proj.weight\t8,1\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert_gate.weight\tmodel-00001-of-00122.ornq\tbf16\t1400\t16\t8\t0\trouter\tmlp.shared_expert_gate.weight\t1,8\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_moe_layout(model, err, sizeof(err)));
    assert(ornith_model_validate_attention_layout(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));
    float x[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float out[8] = {0.0f};
    assert(ornith_layer_decode_smoke(model, 0, x, 8, 1, out));
    assert(out[0] > 3.0f);
    for (size_t i = 1; i < 8; i++) nearf(out[i], 0.0f);
    uint64_t seq1[1] = {1};
    uint64_t seq2[2] = {0, 1};
    size_t idx[1] = {0};
    float one[1] = {0};
    float two[1] = {0};
    assert(ornith_decode_sequence_smoke_limited(model, seq1, 1, 1, 1, 1, 1, idx, one));
    assert(ornith_decode_sequence_smoke_limited(model, seq2, 2, 1, 1, 1, 1, idx, two));
    assert(idx[0] == 0);
    nearf(one[0], 0.0f);
    assert(two[0] > 2.0f);
    uint64_t gen_id[1] = {99};
    float gen_score[1] = {0};
    size_t gen_count = 0;
    assert(ornith_generate_greedy_limited(model, seq1, 1, 1, 1, 1, 1, gen_id, gen_score, &gen_count));
    assert(gen_count == 1);
    assert(gen_id[0] == 0);
    ornith_session *session = NULL;
    uint64_t sess_id[1] = {99};
    float sess_score[1] = {0};
    size_t sess_count = 0;
    assert(ornith_session_open(model, 1, 1, 4, &session));
    assert(ornith_session_token_count(session) == 0);
    assert(ornith_session_token_cap(session) == 4);
    assert(ornith_session_generate_greedy_limited(session, seq1, 1, 1, 1, sess_id, sess_score, &sess_count));
    assert(sess_count == gen_count);
    assert(sess_id[0] == gen_id[0]);
    nearf(sess_score[0], gen_score[0]);
    assert(ornith_session_token_count(session) == 2);
    assert(ornith_session_generate_greedy_limited(session, NULL, 0, 1, 1, sess_id, sess_score, &sess_count));
    assert(sess_count == 1);
    assert(ornith_session_token_count(session) == 3);
    ornith_session_close(session);
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
}

static void test_attention_q_gate_interleaved_unpack(void)
{
#ifdef ORNITH_TESTING
    float mixed[8] = {
        10.0f, 11.0f, 20.0f, 21.0f,
        12.0f, 13.0f, 22.0f, 23.0f,
    };
    float q[4] = {0};
    float gate[4] = {0};
    assert(ornith_test_unpack_attention_q_gate_interleaved(mixed, 2, 2, q, gate));
    nearf(q[0], 10.0f);
    nearf(q[1], 11.0f);
    nearf(q[2], 12.0f);
    nearf(q[3], 13.0f);
    nearf(gate[0], 20.0f);
    nearf(gate[1], 21.0f);
    nearf(gate[2], 22.0f);
    nearf(gate[3], 23.0f);
#endif
}

int main(int argc, char **argv)
{
    if (argc == 3) {
        return probe_real(argv[1], argv[2]);
    }

    char dir[] = "/tmp/ornith-native-loader-XXXXXX";
    assert(mkdtemp(dir));

    char shard[512];
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00122.ornq", dir);
    unsigned char bytes[256] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    put_bf16(bytes + 16, 0x3f80);
    put_bf16(bytes + 18, 0x4000);
    put_bf16(bytes + 20, 0x4040);
    put_bf16(bytes + 22, 0x4080);
    put_bf16(bytes + 24, 0x3f80);
    put_bf16(bytes + 26, 0x3f80);
    put_bf16(bytes + 28, 0x3f80);
    put_bf16(bytes + 30, 0x3f80);
    put_bf16(bytes + 32, 0x4080);
    put_bf16(bytes + 34, 0x4040);
    put_bf16(bytes + 36, 0x4000);
    put_bf16(bytes + 38, 0x3f80);
    put_bf16(bytes + 40, 0x3f80);
    bytes[42] = 0x05;
    put_bf16(bytes + 48, 0x3f80);
    bytes[50] = 0x21;
    bytes[51] = 0x43;
    put_bf16(bytes + 52, 0x3f80);
    bytes[54] = 0x0f;
    bytes[55] = 0x21;
    put_bf16(bytes + 64, 0x3f80);
    put_bf16(bytes + 66, 0x3f80);
    put_bf16(bytes + 68, 0x4000);
    put_bf16(bytes + 70, 0x0000);
    put_bf16(bytes + 72, 0x0000);
    put_bf16(bytes + 74, 0x3f80);
    unsigned short layer_gate_up[] = {
        0x3f80, 0x0000, 0x0000, 0x3f80, 0x4000, 0x0000, 0x0000, 0x4040,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    };
    for (size_t i = 0; i < sizeof(layer_gate_up) / sizeof(layer_gate_up[0]); i++) {
        put_bf16(bytes + 76 + i * 2, layer_gate_up[i]);
    }
    unsigned short layer_down[] = {
        0x3f80, 0x0000, 0x0000, 0x3f80,
        0x0000, 0x0000, 0x0000, 0x0000,
    };
    for (size_t i = 0; i < sizeof(layer_down) / sizeof(layer_down[0]); i++) {
        put_bf16(bytes + 108 + i * 2, layer_down[i]);
    }
    write_file(shard, bytes, sizeof(bytes));

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t256\t16\t4\t13\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t8\t4\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t4\t2\t-1\tglobal\tmodel.language_model.norm.weight\t2\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t32\t8\t4\t-1\tglobal\tlm_head.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.2.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tiq1\t40\t3\t4\t2\trouted_expert\tmlp.experts.gate_up_proj\t1,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.2.linear_attn.out_proj.weight\tmodel-00001-of-00122.ornq\tq4\t48\t8\t8\t2\tattention\tlinear_attn.out_proj.weight\t2,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.input_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t64\t4\t2\t0\tnorm\tinput_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.gate.weight\tmodel-00001-of-00122.ornq\tbf16\t68\t8\t4\t0\trouter\tmlp.gate.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tbf16\t76\t32\t16\t0\trouted_expert\tmlp.experts.gate_up_proj\t2,4,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.down_proj\tmodel-00001-of-00122.ornq\tbf16\t108\t16\t8\t0\trouted_expert\tmlp.experts.down_proj\t2,2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.gate_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t124\t8\t4\t0\tshared_expert\tmlp.shared_expert.gate_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t132\t8\t4\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.down_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t140\t8\t4\t0\tshared_expert\tmlp.shared_expert.down_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert_gate.weight\tmodel-00001-of-00122.ornq\tbf16\t148\t4\t2\t0\trouter\tmlp.shared_expert_gate.weight\t1,2\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_shard_count(model) == 1);
    assert(ornith_model_tensor_count(model) == 13);
    assert(ornith_model_layer_count(model) == 3);
    assert(ornith_model_validate_shards(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));

    float embedding[2] = {0, 0};
    assert(ornith_embed_token(model, 1, embedding, 2));
    assert(embedding[0] == 3.0f);
    assert(embedding[1] == 4.0f);
    size_t token_idx[1] = {0};
    float token_vals[1] = {0};
    assert(ornith_lm_head_topk(model, embedding, 2, 1, token_idx, token_vals));
    assert(token_idx[0] == 0);
    assert(token_vals[0] == 24.0f);

    const ornith_tensor_info *t = ornith_model_find_tensor(model, "model.language_model.layers.2.mlp.experts.gate_up_proj");
    assert(t);
    assert(t->quant == ORNITH_QUANT_IQ1);
    assert(t->layer == 2);
    assert(t->ndim == 2);
    assert(t->shape[0] == 1);
    assert(t->shape[1] == 4);
    assert(strcmp(t->group, "routed_expert") == 0);
    float v = 0.0f;
    assert(ornith_tensor_value(model, t, 0, &v) && v == 1.0f);
    assert(ornith_tensor_value(model, t, 1, &v) && v == -1.0f);
    float iq1_x[4] = {1, 2, 3, 4};
    float iq1_y[1] = {0};
    float iq1_ref[1] = {0};
    assert(ornith_tensor_matvec(model, t, iq1_x, 4, iq1_y));
    ref_matvec(model, t, iq1_x, iq1_ref);
    assert(iq1_y[0] == iq1_ref[0]);

    t = ornith_model_find_tensor(model, "model.language_model.layers.2.linear_attn.out_proj.weight");
    assert(t);
    assert(ornith_model_find_layer_tensor(model, 2, "linear_attn.out_proj.weight") == t);
    assert(ornith_model_find_layer_tensor(model, 1, "linear_attn.out_proj.weight") == NULL);
    assert(t->quant == ORNITH_QUANT_Q4);
    assert(ornith_tensor_value(model, t, 2, &v) && v == 3.0f);
    assert(ornith_tensor_value(model, t, 4, &v) && v == -1.0f);
    float x[4] = {1, 1, 1, 1};
    float y[2] = {0, 0};
    float y_ref[2] = {0, 0};
    assert(ornith_tensor_matvec(model, t, x, 4, y));
    ref_matvec(model, t, x, y_ref);
    assert(y[0] == 10.0f);
    assert(y[1] == 2.0f);
    assert(y[0] == y_ref[0]);
    assert(y[1] == y_ref[1]);

    t = ornith_model_find_layer_tensor(model, 0, "mlp.experts.gate_up_proj");
    assert(t);
    float routed_gu[4] = {0, 0, 0, 0};
    float routed_gu_ref[4] = {0, 0, 0, 0};
    float small_x[2] = {1, 1};
    assert(ornith_tensor_slice_matvec(model, t, 0, small_x, 2, routed_gu));
    ref_slice_matvec(model, t, 0, small_x, routed_gu_ref);
    assert(routed_gu[0] == 1.0f);
    assert(routed_gu[1] == 1.0f);
    assert(routed_gu[2] == 2.0f);
    assert(routed_gu[3] == 3.0f);
    for (size_t i = 0; i < 4; i++) assert(routed_gu[i] == routed_gu_ref[i]);

    t = ornith_model_find_layer_tensor(model, 0, "input_layernorm.weight");
    assert(t);
    float norm_in[2] = {3, 4};
    float norm_out[2] = {0, 0};
    assert(ornith_rmsnorm(model, t, norm_in, 2, 0.0f, norm_out));
    nearf(norm_out[0], 1.6970563f);
    nearf(norm_out[1], 2.2627418f);

    float scores[4] = {0.1f, 5.0f, 3.0f, 5.0f};
    size_t idx[2] = {0, 0};
    float vals[2] = {0, 0};
    assert(ornith_topk(scores, 4, 2, idx, vals));
    assert(idx[0] == 1);
    assert(idx[1] == 3);
    assert(vals[0] == 5.0f);
    assert(vals[1] == 5.0f);

    float moe_out[2] = {0, 0};
    assert(ornith_layer_moe_smoke(model, 0, small_x, 2, 1, moe_out));
    float rms = 1.0f / sqrtf(1.0f + 0.000001f);
    float gemma_rms = 2.0f * rms;
    nearf(moe_out[0], (gemma_rms * sigf(gemma_rms)) * (2.0f * gemma_rms));
    nearf(moe_out[1], (gemma_rms * sigf(gemma_rms)) * (3.0f * gemma_rms));
    assert(ornith_step_smoke(model, 1, 0, 1, 1, token_idx, token_vals));
    assert(token_idx[0] == 0);
    assert(token_vals[0] > 4.9f);
    assert(ornith_step_smoke(model, 1, 1, 1, 1, token_idx, token_vals));
    assert(token_idx[0] == 0);
    assert(token_vals[0] > 4.9f);
    uint64_t prompt_ids[1] = {1};
    uint64_t sampled_ids[2] = {99, 99};
    float sampled_scores[2] = {0, 0};
    size_t sampled_count = 0;
    ornith_sampling sampling = {10.0f, 1.0f, 2, 1};
    assert(ornith_generate_sampled_limited_with_decode_hooks(model, prompt_ids, 1, 2, 0, 1, 2, &sampling, sampled_ids, sampled_scores, &sampled_count, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    assert(sampled_count == 2);
    assert(sampled_ids[0] < 2);
    assert(sampled_ids[1] < 2);
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
    test_layout_validator();
    test_attention_layout_validator();
    test_decode_self_attention_first_token();
    test_decode_self_attention_sequence();
    test_decode_linear_attention_first_token();
    test_attention_q_gate_interleaved_unpack();
    puts("ornith_native_catalog_loader_test: ok");
    return 0;
}

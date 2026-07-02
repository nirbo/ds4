#include "ornith/ornith.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static int probe_real(const char *catalog, const char *shard_dir)
{
    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_shards(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));
    printf("ornith_native_catalog_loader_test: shards=%zu tensors=%zu layers=%zu\n",
           ornith_model_shard_count(model), ornith_model_tensor_count(model), ornith_model_layer_count(model));
    ornith_model_close(model);
    return 0;
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
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t8\t4\t-1\tglobal\tmodel.language_model.norm.weight\t4\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t32\t8\t4\t-1\tglobal\tlm_head.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tiq1\t40\t3\t4\t0\trouted_expert\tmlp.experts.gate_up_proj\t1,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.out_proj.weight\tmodel-00001-of-00122.ornq\tq4\t48\t8\t8\t0\tattention\tlinear_attn.out_proj.weight\t2,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.input_layernorm.weight\tmodel-00001-of-00122.ornq\tbf16\t64\t4\t2\t1\tnorm\tinput_layernorm.weight\t2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.mlp.gate.weight\tmodel-00001-of-00122.ornq\tbf16\t68\t8\t4\t1\trouter\tmlp.gate.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tbf16\t76\t32\t16\t1\trouted_expert\tmlp.experts.gate_up_proj\t2,4,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.mlp.experts.down_proj\tmodel-00001-of-00122.ornq\tbf16\t108\t16\t8\t1\trouted_expert\tmlp.experts.down_proj\t2,2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.mlp.shared_expert.gate_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t124\t8\t4\t1\tshared_expert\tmlp.shared_expert.gate_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t132\t8\t4\t1\tshared_expert\tmlp.shared_expert.up_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.mlp.shared_expert.down_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t140\t8\t4\t1\tshared_expert\tmlp.shared_expert.down_proj.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.1.mlp.shared_expert_gate.weight\tmodel-00001-of-00122.ornq\tbf16\t148\t4\t2\t1\trouter\tmlp.shared_expert_gate.weight\t1,2\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_shard_count(model) == 1);
    assert(ornith_model_tensor_count(model) == 13);
    assert(ornith_model_layer_count(model) == 2);
    assert(ornith_model_validate_shards(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));

    const ornith_tensor_info *t = ornith_model_find_tensor(model, "model.language_model.layers.0.mlp.experts.gate_up_proj");
    assert(t);
    assert(t->quant == ORNITH_QUANT_IQ1);
    assert(t->layer == 0);
    assert(t->ndim == 2);
    assert(t->shape[0] == 1);
    assert(t->shape[1] == 4);
    assert(strcmp(t->group, "routed_expert") == 0);
    float v = 0.0f;
    assert(ornith_tensor_value(model, t, 0, &v) && v == 1.0f);
    assert(ornith_tensor_value(model, t, 1, &v) && v == -1.0f);

    t = ornith_model_find_tensor(model, "model.language_model.layers.0.linear_attn.out_proj.weight");
    assert(t);
    assert(ornith_model_find_layer_tensor(model, 0, "linear_attn.out_proj.weight") == t);
    assert(ornith_model_find_layer_tensor(model, 1, "linear_attn.out_proj.weight") == NULL);
    assert(t->quant == ORNITH_QUANT_Q4);
    assert(ornith_tensor_value(model, t, 2, &v) && v == 3.0f);
    assert(ornith_tensor_value(model, t, 4, &v) && v == -1.0f);
    float x[4] = {1, 1, 1, 1};
    float y[2] = {0, 0};
    assert(ornith_tensor_matvec(model, t, x, 4, y));
    assert(y[0] == 10.0f);
    assert(y[1] == 2.0f);

    t = ornith_model_find_layer_tensor(model, 1, "mlp.experts.gate_up_proj");
    assert(t);
    float routed_gu[4] = {0, 0, 0, 0};
    float small_x[2] = {1, 1};
    assert(ornith_tensor_slice_matvec(model, t, 0, small_x, 2, routed_gu));
    assert(routed_gu[0] == 1.0f);
    assert(routed_gu[1] == 1.0f);
    assert(routed_gu[2] == 2.0f);
    assert(routed_gu[3] == 3.0f);

    t = ornith_model_find_tensor(model, "model.language_model.norm.weight");
    assert(t);
    float norm_in[4] = {1, 2, 3, 4};
    float norm_out[4] = {0, 0, 0, 0};
    assert(ornith_rmsnorm(model, t, norm_in, 4, 0.0f, norm_out));
    nearf(norm_out[0], 0.36514837f);
    nearf(norm_out[3], 1.4605935f);

    float scores[4] = {0.1f, 5.0f, 3.0f, 5.0f};
    size_t idx[2] = {0, 0};
    float vals[2] = {0, 0};
    assert(ornith_topk(scores, 4, 2, idx, vals));
    assert(idx[0] == 1);
    assert(idx[1] == 3);
    assert(vals[0] == 5.0f);
    assert(vals[1] == 5.0f);

    float moe_out[2] = {0, 0};
    assert(ornith_layer_moe_smoke(model, 1, small_x, 2, 1, moe_out));
    float rms = 1.0f / sqrtf(1.0f + 0.000001f);
    nearf(moe_out[0], (rms * sigf(rms)) * (2.0f * rms));
    nearf(moe_out[1], (rms * sigf(rms)) * (3.0f * rms));
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
    puts("ornith_native_catalog_loader_test: ok");
    return 0;
}

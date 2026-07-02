#include "ornith/ornith.h"
#include "ornith/ornith_metal.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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

static void near_array(const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        assert(fabsf(a[i] - b[i]) < 0.0001f);
    }
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

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t128\t16\t4\t6\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t8\t4\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t4\t2\t-1\tglobal\tmodel.language_model.norm.weight\t2\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t28\t8\t4\t-1\tglobal\tlm_head.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.linear_attn.out_proj.weight\tmodel-00001-of-00122.ornq\tq4\t40\t8\t8\t0\tattention\tlinear_attn.out_proj.weight\t2,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tiq1\t56\t3\t4\t0\trouted_expert\tmlp.experts.gate_up_proj\t1,4\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.shared_expert.up_proj.weight\tmodel-00001-of-00122.ornq\tbf16\t64\t12\t6\t0\tshared_expert\tmlp.shared_expert.up_proj.weight\t3,2\n");
    assert(fclose(fp) == 0);

    char err[512] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));

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

    ornith_model_close(model);
    remove(catalog);
    remove(shard);
    rmdir(dir);
    puts("ornith_metal_matvec_test: ok");
    return 0;
}

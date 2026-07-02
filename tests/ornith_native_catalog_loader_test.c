#include "ornith/ornith.h"

#include <assert.h>
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

static int probe_real(const char *catalog, const char *shard_dir)
{
    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, shard_dir, &model, err, sizeof(err)));
    assert(ornith_model_validate_shards(model, err, sizeof(err)));
    printf("ornith_native_catalog_loader_test: shards=%zu tensors=%zu\n",
           ornith_model_shard_count(model), ornith_model_tensor_count(model));
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
    unsigned char bytes[64] = {'O', 'R', 'N', 'Q', '1', 0, 0, 0};
    write_file(shard, bytes, sizeof(bytes));

    char catalog[512];
    snprintf(catalog, sizeof(catalog), "%s/catalog.tsv", dir);
    FILE *fp = fopen(catalog, "w");
    assert(fp);
    fprintf(fp, "# ornith-runtime-catalog-tsv-v1\n");
    fprintf(fp, "shard\tmodel-00001-of-00122.ornq\t64\t16\t128\t4\n");
    fprintf(fp, "tensor\tmodel.language_model.embed_tokens.weight\tmodel-00001-of-00122.ornq\tbf16\t16\t8\t4\t-1\tglobal\tmodel.language_model.embed_tokens.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.norm.weight\tmodel-00001-of-00122.ornq\tbf16\t24\t8\t4\t-1\tglobal\tmodel.language_model.norm.weight\t4\n");
    fprintf(fp, "tensor\tlm_head.weight\tmodel-00001-of-00122.ornq\tbf16\t32\t8\t4\t-1\tglobal\tlm_head.weight\t2,2\n");
    fprintf(fp, "tensor\tmodel.language_model.layers.0.mlp.experts.gate_up_proj\tmodel-00001-of-00122.ornq\tiq1\t40\t4\t4\t0\trouted_expert\tmlp.experts.gate_up_proj\t1,4\n");
    assert(fclose(fp) == 0);

    char err[256] = {0};
    ornith_model *model = NULL;
    assert(ornith_model_open(catalog, dir, &model, err, sizeof(err)));
    assert(ornith_model_shard_count(model) == 1);
    assert(ornith_model_tensor_count(model) == 4);
    assert(ornith_model_validate_shards(model, err, sizeof(err)));

    const ornith_tensor_info *t = ornith_model_find_tensor(model, "model.language_model.layers.0.mlp.experts.gate_up_proj");
    assert(t);
    assert(t->quant == ORNITH_QUANT_IQ1);
    assert(t->layer == 0);
    assert(t->ndim == 2);
    assert(t->shape[0] == 1);
    assert(t->shape[1] == 4);
    assert(strcmp(t->group, "routed_expert") == 0);
    ornith_model_close(model);

    remove(catalog);
    remove(shard);
    rmdir(dir);
    puts("ornith_native_catalog_loader_test: ok");
    return 0;
}

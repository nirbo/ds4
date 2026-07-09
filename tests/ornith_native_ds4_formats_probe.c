#include "ornith/ornith.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    char err[256] = {0};
    ornith_model *model = NULL;
    if (!ornith_model_open(argv[1], argv[2], &model, err, sizeof(err))) {
        fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    assert(ornith_model_validate_shards(model, err, sizeof(err)));
    assert(ornith_model_map_shards(model, err, sizeof(err)));
    const char *names[] = {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        const ornith_tensor_info *tensor = ornith_model_find_tensor(model, names[i]);
        assert(tensor);
        float value = 0.0f;
        assert(ornith_tensor_value(model, tensor, tensor->nparams / 2, &value));
        assert(isfinite(value));
        size_t cols = (size_t)tensor->shape[tensor->ndim - 1];
        size_t rows = (size_t)tensor->shape[tensor->ndim - 2];
        float *x = calloc(cols, sizeof(*x));
        float *out = calloc(rows, sizeof(*out));
        assert(x && out);
        x[0] = 1.0f;
        int ok = tensor->ndim == 2
            ? ornith_tensor_matvec(model, tensor, x, cols, out)
            : ornith_tensor_slice_matvec(model, tensor, 0, x, cols, out);
        assert(ok);
        for (size_t row = 0; row < rows; row++) assert(isfinite(out[row]));
        free(x);
        free(out);
    }
    ornith_model_close(model);
    puts("ornith_native_ds4_formats_probe: ok");
    return 0;
}

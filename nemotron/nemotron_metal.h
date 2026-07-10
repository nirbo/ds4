#ifndef NEMOTRON_METAL_H
#define NEMOTRON_METAL_H

#include <stddef.h>
#include <stdint.h>

int nemotron_metal_nvfp4_matvec(
    const uint8_t *packed_weight,
    const uint8_t *block_scale,
    float global_scale,
    const float *input,
    float *output,
    uint32_t rows,
    uint32_t columns,
    uint32_t repeats,
    double *elapsed_ms,
    char *error,
    size_t error_cap);

#endif

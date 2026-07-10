#import <Foundation/Foundation.h>

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../nemotron/nemotron_metal.h"

static const float e2m1_values[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};

static float decode_e2m1(uint8_t value) {
    float magnitude = e2m1_values[value & 7];
    return (value & 8) ? -magnitude : magnitude;
}

static float decode_e4m3fn(uint8_t value) {
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    float decoded;
    if (exponent == 0) decoded = ldexpf((float) mantissa, -9);
    else if (exponent == 15 && mantissa == 7) decoded = NAN;
    else decoded = ldexpf(1.0f + (float) mantissa / 8.0f, exponent - 7);
    return (value & 128) ? -decoded : decoded;
}

static float reference_row(
    const uint8_t *weight,
    const uint8_t *scale,
    float global_scale,
    const float *input,
    uint32_t row,
    uint32_t columns)
{
    uint32_t packed_columns = columns / 2;
    uint32_t blocks_per_row = columns / 16;
    double sum = 0;
    for (uint32_t column = 0; column < columns; column++) {
        uint8_t packed = weight[(size_t) row * packed_columns + column / 2];
        uint8_t nibble = (column & 1) ? packed >> 4 : packed & 15;
        float block = decode_e4m3fn(scale[(size_t) row * blocks_per_row + column / 16]);
        sum += (double) decode_e2m1(nibble) * block * global_scale * input[column];
    }
    return (float) sum;
}

static int compare_outputs(
    const uint8_t *weight,
    const uint8_t *scale,
    float global_scale,
    const float *input,
    const float *actual,
    uint32_t rows,
    uint32_t columns)
{
    double error2 = 0;
    double reference2 = 0;
    float max_abs = 0;
    for (uint32_t row = 0; row < rows; row++) {
        float expected = reference_row(weight, scale, global_scale, input, row, columns);
        float error = actual[row] - expected;
        error2 += (double) error * error;
        reference2 += (double) expected * expected;
        if (fabsf(error) > max_abs) max_abs = fabsf(error);
    }
    double relative_l2 = sqrt(error2 / fmax(reference2, 1e-30));
    printf("numeric: relative_l2=%.9g max_abs=%.9g\n", relative_l2, max_abs);
    return relative_l2 <= 2e-5 && max_abs <= 2e-4 ? 0 : -1;
}

static int synthetic_test(void) {
    enum { rows = 7, columns = 64 };
    uint8_t weight[rows * columns / 2];
    uint8_t scale[rows * columns / 16];
    float input[columns];
    float output[rows];
    for (size_t i = 0; i < sizeof(weight); i++) {
        uint8_t low = (uint8_t) ((i * 5 + 1) & 15);
        uint8_t high = (uint8_t) ((i * 7 + 3) & 15);
        weight[i] = low | (high << 4);
    }
    const uint8_t scales[] = {0x01, 0x20, 0x38, 0x3c, 0x40, 0x58, 0x70, 0x7e};
    for (size_t i = 0; i < sizeof(scale); i++) scale[i] = scales[i % (sizeof(scales))];
    for (uint32_t i = 0; i < columns; i++) input[i] = sinf((float) i * 0.17f) * 0.5f;
    char error[512] = {0};
    double elapsed_ms = 0;
    if (nemotron_metal_nvfp4_matvec(weight, scale, 0.03125f, input, output,
            rows, columns, 1, &elapsed_ms, error, sizeof(error)) != 0) {
        fprintf(stderr, "synthetic Metal failure: %s\n", error);
        return -1;
    }
    printf("synthetic: rows=%u columns=%u elapsed_ms=%.6f\n", rows, columns, elapsed_ms);
    return compare_outputs(weight, scale, 0.03125f, input, output, rows, columns);
}

static int pread_exact(int fd, void *buffer, size_t size, off_t offset) {
    uint8_t *bytes = buffer;
    size_t complete = 0;
    while (complete < size) {
        ssize_t count = pread(fd, bytes + complete, size - complete, offset + (off_t) complete);
        if (count <= 0) return -1;
        complete += (size_t) count;
    }
    return 0;
}

static NSDictionary *load_header(int fd, uint64_t *header_size) {
    if (pread_exact(fd, header_size, sizeof(*header_size), 0) != 0) return nil;
    NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger) *header_size];
    if (pread_exact(fd, data.mutableBytes, (size_t) *header_size, 8) != 0) return nil;
    return [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
}

static int load_tensor(
    int fd,
    NSDictionary *header,
    uint64_t header_size,
    NSString *name,
    uint8_t **bytes,
    size_t *size,
    NSArray **shape)
{
    NSDictionary *entry = header[name];
    if (![entry isKindOfClass:[NSDictionary class]]) return -1;
    NSArray *offsets = entry[@"data_offsets"];
    uint64_t start = [offsets[0] unsignedLongLongValue];
    uint64_t end = [offsets[1] unsignedLongLongValue];
    if (end < start || end - start > SIZE_MAX) return -1;
    *size = (size_t) (end - start);
    *bytes = malloc(*size);
    if (*bytes == NULL || pread_exact(fd, *bytes, *size, (off_t) (8 + header_size + start)) != 0) {
        free(*bytes);
        *bytes = NULL;
        return -1;
    }
    if (shape != NULL) *shape = entry[@"shape"];
    return 0;
}

static int real_test(const char *path, const char *prefix) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) return -1;
    uint64_t header_size = 0;
    NSDictionary *header = load_header(fd, &header_size);
    if (header == nil) {
        close(fd);
        return -1;
    }
    NSString *base = [NSString stringWithUTF8String:prefix];
    uint8_t *weight = NULL, *scale = NULL, *global_bytes = NULL;
    size_t weight_size = 0, scale_size = 0, global_size = 0;
    NSArray *weight_shape = nil;
    int result = 0;
    if (load_tensor(fd, header, header_size, [base stringByAppendingString:@".weight"],
            &weight, &weight_size, &weight_shape) != 0 ||
        load_tensor(fd, header, header_size, [base stringByAppendingString:@".weight_scale"],
            &scale, &scale_size, NULL) != 0 ||
        load_tensor(fd, header, header_size, [base stringByAppendingString:@".weight_scale_2"],
            &global_bytes, &global_size, NULL) != 0 || global_size != sizeof(float)) {
        result = -1;
        goto cleanup;
    }
    uint32_t rows = [weight_shape[0] unsignedIntValue];
    uint32_t columns = [weight_shape[1] unsignedIntValue] * 2;
    if (weight_size != (size_t) rows * columns / 2 ||
        scale_size != (size_t) rows * columns / 16) {
        result = -1;
        goto cleanup;
    }
    float global_scale;
    memcpy(&global_scale, global_bytes, sizeof(global_scale));
    float *input = malloc((size_t) columns * sizeof(float));
    float *output = malloc((size_t) rows * sizeof(float));
    if (input == NULL || output == NULL) {
        free(input); free(output);
        result = -1;
        goto cleanup;
    }
    for (uint32_t i = 0; i < columns; i++) input[i] = sinf((float) i * 0.013f) + cosf((float) i * 0.007f) * 0.25f;
    char error[512] = {0};
    double elapsed_ms = 0;
    const uint32_t repeats = 100;
    if (nemotron_metal_nvfp4_matvec(weight, scale, global_scale, input, output,
            rows, columns, repeats, &elapsed_ms, error, sizeof(error)) != 0) {
        fprintf(stderr, "real Metal failure: %s\n", error);
        free(input); free(output);
        result = -1;
        goto cleanup;
    }
    double per_call_ms = elapsed_ms / repeats;
    double bytes_per_call = (double) weight_size + scale_size +
        (double) (rows + columns) * sizeof(float);
    printf("real: prefix=%s rows=%u columns=%u repeats=%u ms=%.6f bandwidth=%.2fGB/s\n",
        prefix, rows, columns, repeats, per_call_ms, bytes_per_call / (per_call_ms * 1e6));
    result = compare_outputs(weight, scale, global_scale, input, output, rows, columns);
    free(input); free(output);

cleanup:
    free(weight); free(scale); free(global_bytes);
    close(fd);
    return result;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (synthetic_test() != 0) return 1;
        if (argc == 3 && real_test(argv[1], argv[2]) != 0) return 1;
        return 0;
    }
}

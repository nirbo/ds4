#include <metal_stdlib>
using namespace metal;

constant float nemotron_e2m1_values[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

inline float nemotron_decode_e2m1(uchar nibble) {
    float value = nemotron_e2m1_values[nibble & 7u];
    return (nibble & 8u) ? -value : value;
}

inline float nemotron_decode_e4m3fn(uchar bits) {
    uint exponent = (bits >> 3) & 15u;
    uint mantissa = bits & 7u;
    float value;
    if (exponent == 0u) {
        value = float(mantissa) * 0.001953125f;
    } else if (exponent == 15u && mantissa == 7u) {
        value = NAN;
    } else {
        value = (1.0f + float(mantissa) * 0.125f) * exp2(float(int(exponent) - 7));
    }
    return (bits & 128u) ? -value : value;
}

kernel void nemotron_nvfp4_matvec_f32(
    device const uchar *packed_weight [[buffer(0)]],
    device const uchar *block_scale [[buffer(1)]],
    device const float *global_scale [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint2 &dimensions [[buffer(5)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint row = group_position.x * 8u + simd_group;
    uint rows = dimensions.x;
    uint columns = dimensions.y;
    if (row >= rows) return;
    uint packed_columns = columns >> 1;
    uint blocks_per_row = columns >> 4;
    float sum = 0.0f;
    for (uint block = lane; block < blocks_per_row; block += 32u) {
        float scale = nemotron_decode_e4m3fn(block_scale[row * blocks_per_row + block])
            * global_scale[0];
        uint column_base = block << 4;
        uint packed_base = row * packed_columns + (column_base >> 1);
        for (uint pair = 0; pair < 8u; pair++) {
            uchar packed = packed_weight[packed_base + pair];
            uint column = column_base + (pair << 1);
            sum += nemotron_decode_e2m1(packed & 15u) * scale * input[column];
            sum += nemotron_decode_e2m1(packed >> 4) * scale * input[column + 1u];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

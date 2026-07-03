#import "ornith_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static NSString *const ORNITH_METAL_SRC =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Args { ulong byte_base; ulong elem_offset; uint rows; uint cols; uint block; uint x_stride; uint total_rows; };\n"
"struct GdnArgs { uint value_heads; uint head_v; uint key_heads; uint head_k; };\n"
"static inline float bf16_at(device const uchar *p, ulong i) {\n"
"    uint lo = p[i]; uint hi = p[i + 1]; return as_type<float>((hi << 24) | (lo << 16));\n"
"}\n"
"kernel void ornith_bf16_matvec(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; ulong base = a.byte_base + (a.elem_offset + (ulong)row * a.cols) * 2;\n"
"    for (uint c = 0; c < a.cols; c++) acc += bf16_at(payload, base + (ulong)c * 2) * x[c]; out[row] = acc;\n"
"}\n"
"kernel void ornith_bf16_matvec_tg(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
"    threadgroup float partial[256]; float acc = 0.0f; ulong base = a.byte_base + (a.elem_offset + (ulong)row * a.cols) * 2;\n"
"    for (uint c = tid; c < a.cols; c += nt) acc += bf16_at(payload, base + (ulong)c * 2) * x[c]; partial[tid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = nt >> 1; s > 0; s >>= 1) { if (tid < s) partial[tid] += partial[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); } if (tid == 0) out[row] = partial[0];\n"
"}\n"
"kernel void ornith_q4_matvec(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; ulong row_base = a.elem_offset + (ulong)row * a.cols;\n"
"    for (uint c = 0; c < a.cols;) { ulong i = row_base + c; uint inb = (uint)(i % a.block); uint take = min(a.block - inb, a.cols - c); ulong bb = a.byte_base + (i / a.block) * (2 + (a.block + 1) / 2); float scale = bf16_at(payload, bb);\n"
"        for (uint j = 0; j < take; j++, c++) { uint qidx = inb + j; uchar packed = payload[bb + 2 + qidx / 2]; int q = (qidx & 1) ? (packed >> 4) : (packed & 15); if (q >= 8) q -= 16; acc += scale * (float)q * x[c]; }} out[row] = acc;\n"
"}\n"
"kernel void ornith_q4_matvec_b256(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; ulong elem = a.elem_offset + (ulong)row * a.cols; ulong block_base = a.byte_base + (elem >> 8) * 130;\n"
"    for (uint b = 0; b < (a.cols >> 8); b++) { ulong bb = block_base + (ulong)b * 130; float scale = bf16_at(payload, bb); uint xbase = b << 8; for (uint j = 0; j < 256; j++) { uchar packed = payload[bb + 2 + j / 2]; int q = (j & 1) ? (packed >> 4) : (packed & 15); if (q >= 8) q -= 16; acc += scale * (float)q * x[xbase + j]; }} out[row] = acc;\n"
"}\n"
"kernel void ornith_q4_matvec_tg(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
"    threadgroup float partial[256]; float acc = 0.0f; ulong row_base = a.elem_offset + (ulong)row * a.cols;\n"
"    for (uint c = tid; c < a.cols; c += nt) { ulong i = row_base + c; uint inb = (uint)(i % a.block); ulong bb = a.byte_base + (i / a.block) * (2 + (a.block + 1) / 2); float scale = bf16_at(payload, bb); uchar packed = payload[bb + 2 + inb / 2]; int q = (inb & 1) ? (packed >> 4) : (packed & 15); if (q >= 8) q -= 16; acc += scale * (float)q * x[c]; }\n"
"    partial[tid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup); for (uint s = nt >> 1; s > 0; s >>= 1) { if (tid < s) partial[tid] += partial[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); } if (tid == 0) out[row] = partial[0];\n"
"}\n"
"kernel void ornith_q4_matvec_b256_r4_tg(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row_group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
"    threadgroup float p0[256]; threadgroup float p1[256]; threadgroup float p2[256]; threadgroup float p3[256]; uint row0 = row_group << 2; float acc0 = 0.0f; float acc1 = 0.0f; float acc2 = 0.0f; float acc3 = 0.0f;\n"
"    ulong elem0 = a.elem_offset + (ulong)row0 * a.cols; ulong base0 = a.byte_base + (elem0 >> 8) * 130; ulong row_stride = ((ulong)a.cols >> 8) * 130;\n"
"    for (uint b = 0; b < (a.cols >> 8); b++) { ulong off = (ulong)b * 130; float xv = x[(b << 8) + tid]; ulong bb0 = base0 + off; float s0 = bf16_at(payload, bb0); uchar pk0 = payload[bb0 + 2 + tid / 2]; int q0 = (tid & 1) ? (pk0 >> 4) : (pk0 & 15); if (q0 >= 8) q0 -= 16; acc0 += s0 * (float)q0 * xv;\n"
"        if (row0 + 1 < a.rows) { ulong bb1 = bb0 + row_stride; float s1 = bf16_at(payload, bb1); uchar pk1 = payload[bb1 + 2 + tid / 2]; int q1 = (tid & 1) ? (pk1 >> 4) : (pk1 & 15); if (q1 >= 8) q1 -= 16; acc1 += s1 * (float)q1 * xv; }\n"
"        if (row0 + 2 < a.rows) { ulong bb2 = bb0 + row_stride * 2; float s2 = bf16_at(payload, bb2); uchar pk2 = payload[bb2 + 2 + tid / 2]; int q2 = (tid & 1) ? (pk2 >> 4) : (pk2 & 15); if (q2 >= 8) q2 -= 16; acc2 += s2 * (float)q2 * xv; }\n"
"        if (row0 + 3 < a.rows) { ulong bb3 = bb0 + row_stride * 3; float s3 = bf16_at(payload, bb3); uchar pk3 = payload[bb3 + 2 + tid / 2]; int q3 = (tid & 1) ? (pk3 >> 4) : (pk3 & 15); if (q3 >= 8) q3 -= 16; acc3 += s3 * (float)q3 * xv; }}\n"
"    p0[tid] = acc0; p1[tid] = acc1; p2[tid] = acc2; p3[tid] = acc3; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { p0[tid] += p0[tid + s]; p1[tid] += p1[tid + s]; p2[tid] += p2[tid + s]; p3[tid] += p3[tid + s]; } threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    if (tid == 0) { out[row0] = p0[0]; if (row0 + 1 < a.rows) out[row0 + 1] = p1[0]; if (row0 + 2 < a.rows) out[row0 + 2] = p2[0]; if (row0 + 3 < a.rows) out[row0 + 3] = p3[0]; }\n"
"}\n"
"kernel void ornith_q4_router_b256_r8_tg(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row_group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
"    threadgroup float p0[64]; threadgroup float p1[64]; threadgroup float p2[64]; threadgroup float p3[64]; threadgroup float p4[64]; threadgroup float p5[64]; threadgroup float p6[64]; threadgroup float p7[64]; uint row0 = row_group << 3; float acc0 = 0.0f; float acc1 = 0.0f; float acc2 = 0.0f; float acc3 = 0.0f; float acc4 = 0.0f; float acc5 = 0.0f; float acc6 = 0.0f; float acc7 = 0.0f; ulong elem0 = a.elem_offset + (ulong)row0 * a.cols; ulong base0 = a.byte_base + (elem0 >> 8) * 130; ulong row_stride = ((ulong)a.cols >> 8) * 130;\n"
"    for (uint b = 0; b < (a.cols >> 8); b++) { ulong off = (ulong)b * 130; uint qbase = tid << 1; uint xbase = (b << 8) + (tid << 2); float x0 = x[xbase]; float x1 = x[xbase + 1]; float x2 = x[xbase + 2]; float x3 = x[xbase + 3]; ulong bb0 = base0 + off; float s0 = bf16_at(payload, bb0); uchar a0 = payload[bb0 + 2 + qbase]; uchar a1 = payload[bb0 + 3 + qbase]; int q0 = a0 & 15; if (q0 >= 8) q0 -= 16; int q1 = a0 >> 4; if (q1 >= 8) q1 -= 16; int q2 = a1 & 15; if (q2 >= 8) q2 -= 16; int q3 = a1 >> 4; if (q3 >= 8) q3 -= 16; acc0 += s0 * ((float)q0 * x0 + (float)q1 * x1 + (float)q2 * x2 + (float)q3 * x3);\n"
"        if (row0 + 1 < a.rows) { ulong bb = bb0 + row_stride; float ss = bf16_at(payload, bb); uchar p = payload[bb + 2 + qbase]; uchar q = payload[bb + 3 + qbase]; int r0 = p & 15; if (r0 >= 8) r0 -= 16; int r1 = p >> 4; if (r1 >= 8) r1 -= 16; int r2 = q & 15; if (r2 >= 8) r2 -= 16; int r3 = q >> 4; if (r3 >= 8) r3 -= 16; acc1 += ss * ((float)r0 * x0 + (float)r1 * x1 + (float)r2 * x2 + (float)r3 * x3); }\n"
"        if (row0 + 2 < a.rows) { ulong bb = bb0 + row_stride * 2; float ss = bf16_at(payload, bb); uchar p = payload[bb + 2 + qbase]; uchar q = payload[bb + 3 + qbase]; int r0 = p & 15; if (r0 >= 8) r0 -= 16; int r1 = p >> 4; if (r1 >= 8) r1 -= 16; int r2 = q & 15; if (r2 >= 8) r2 -= 16; int r3 = q >> 4; if (r3 >= 8) r3 -= 16; acc2 += ss * ((float)r0 * x0 + (float)r1 * x1 + (float)r2 * x2 + (float)r3 * x3); }\n"
"        if (row0 + 3 < a.rows) { ulong bb = bb0 + row_stride * 3; float ss = bf16_at(payload, bb); uchar p = payload[bb + 2 + qbase]; uchar q = payload[bb + 3 + qbase]; int r0 = p & 15; if (r0 >= 8) r0 -= 16; int r1 = p >> 4; if (r1 >= 8) r1 -= 16; int r2 = q & 15; if (r2 >= 8) r2 -= 16; int r3 = q >> 4; if (r3 >= 8) r3 -= 16; acc3 += ss * ((float)r0 * x0 + (float)r1 * x1 + (float)r2 * x2 + (float)r3 * x3); }\n"
"        if (row0 + 4 < a.rows) { ulong bb = bb0 + row_stride * 4; float ss = bf16_at(payload, bb); uchar p = payload[bb + 2 + qbase]; uchar q = payload[bb + 3 + qbase]; int r0 = p & 15; if (r0 >= 8) r0 -= 16; int r1 = p >> 4; if (r1 >= 8) r1 -= 16; int r2 = q & 15; if (r2 >= 8) r2 -= 16; int r3 = q >> 4; if (r3 >= 8) r3 -= 16; acc4 += ss * ((float)r0 * x0 + (float)r1 * x1 + (float)r2 * x2 + (float)r3 * x3); }\n"
"        if (row0 + 5 < a.rows) { ulong bb = bb0 + row_stride * 5; float ss = bf16_at(payload, bb); uchar p = payload[bb + 2 + qbase]; uchar q = payload[bb + 3 + qbase]; int r0 = p & 15; if (r0 >= 8) r0 -= 16; int r1 = p >> 4; if (r1 >= 8) r1 -= 16; int r2 = q & 15; if (r2 >= 8) r2 -= 16; int r3 = q >> 4; if (r3 >= 8) r3 -= 16; acc5 += ss * ((float)r0 * x0 + (float)r1 * x1 + (float)r2 * x2 + (float)r3 * x3); }\n"
"        if (row0 + 6 < a.rows) { ulong bb = bb0 + row_stride * 6; float ss = bf16_at(payload, bb); uchar p = payload[bb + 2 + qbase]; uchar q = payload[bb + 3 + qbase]; int r0 = p & 15; if (r0 >= 8) r0 -= 16; int r1 = p >> 4; if (r1 >= 8) r1 -= 16; int r2 = q & 15; if (r2 >= 8) r2 -= 16; int r3 = q >> 4; if (r3 >= 8) r3 -= 16; acc6 += ss * ((float)r0 * x0 + (float)r1 * x1 + (float)r2 * x2 + (float)r3 * x3); }\n"
"        if (row0 + 7 < a.rows) { ulong bb = bb0 + row_stride * 7; float ss = bf16_at(payload, bb); uchar p = payload[bb + 2 + qbase]; uchar q = payload[bb + 3 + qbase]; int r0 = p & 15; if (r0 >= 8) r0 -= 16; int r1 = p >> 4; if (r1 >= 8) r1 -= 16; int r2 = q & 15; if (r2 >= 8) r2 -= 16; int r3 = q >> 4; if (r3 >= 8) r3 -= 16; acc7 += ss * ((float)r0 * x0 + (float)r1 * x1 + (float)r2 * x2 + (float)r3 * x3); }}\n"
"    p0[tid] = acc0; p1[tid] = acc1; p2[tid] = acc2; p3[tid] = acc3; p4[tid] = acc4; p5[tid] = acc5; p6[tid] = acc6; p7[tid] = acc7; threadgroup_barrier(mem_flags::mem_threadgroup); for (uint s = 32; s > 0; s >>= 1) { if (tid < s) { p0[tid] += p0[tid + s]; p1[tid] += p1[tid + s]; p2[tid] += p2[tid + s]; p3[tid] += p3[tid + s]; p4[tid] += p4[tid + s]; p5[tid] += p5[tid + s]; p6[tid] += p6[tid + s]; p7[tid] += p7[tid + s]; } threadgroup_barrier(mem_flags::mem_threadgroup); } if (tid == 0) { out[row0] = p0[0]; if (row0 + 1 < a.rows) out[row0 + 1] = p1[0]; if (row0 + 2 < a.rows) out[row0 + 2] = p2[0]; if (row0 + 3 < a.rows) out[row0 + 3] = p3[0]; if (row0 + 4 < a.rows) out[row0 + 4] = p4[0]; if (row0 + 5 < a.rows) out[row0 + 5] = p5[0]; if (row0 + 6 < a.rows) out[row0 + 6] = p6[0]; if (row0 + 7 < a.rows) out[row0 + 7] = p7[0]; }\n"
"}\n"
"kernel void ornith_iq1_matvec(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; ulong row_base = a.elem_offset + (ulong)row * a.cols;\n"
"    for (uint c = 0; c < a.cols;) { ulong i = row_base + c; uint inb = (uint)(i % a.block); uint take = min(a.block - inb, a.cols - c); ulong bb = a.byte_base + (i / a.block) * (2 + (a.block + 7) / 8); float scale = bf16_at(payload, bb);\n"
"        for (uint j = 0; j < take; j++, c++) { uint b = inb + j; uchar bits = payload[bb + 2 + b / 8]; acc += ((bits & (1 << (b % 8))) ? scale : -scale) * x[c]; }} out[row] = acc;\n"
"}\n"
"kernel void ornith_iq1_matvec_tg(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
"    threadgroup float partial[256]; float acc = 0.0f; ulong row_base = a.elem_offset + (ulong)row * a.cols;\n"
"    for (uint c = tid; c < a.cols; c += nt) { ulong i = row_base + c; uint inb = (uint)(i % a.block); ulong bb = a.byte_base + (i / a.block) * (2 + (a.block + 7) / 8); float scale = bf16_at(payload, bb); uchar bits = payload[bb + 2 + inb / 8]; acc += ((bits & (1 << (inb % 8))) ? scale : -scale) * x[c]; }\n"
"    partial[tid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup); for (uint s = nt >> 1; s > 0; s >>= 1) { if (tid < s) partial[tid] += partial[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); } if (tid == 0) out[row] = partial[0];\n"
"}\n"
"kernel void ornith_iq1_slice_many_tg(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], device const uint *slices [[buffer(4)]], uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
"    uint slice_i = gid / a.rows; uint row = gid - slice_i * a.rows; uint slice = slices[slice_i]; threadgroup float partial[256]; float acc = 0.0f; ulong row_base = a.elem_offset + ((ulong)slice * a.rows + row) * a.cols; ulong xbase = (ulong)slice_i * a.x_stride;\n"
"    for (uint c = tid; c < a.cols; c += nt) { ulong i = row_base + c; uint inb = (uint)(i % a.block); ulong bb = a.byte_base + (i / a.block) * (2 + (a.block + 7) / 8); float scale = bf16_at(payload, bb); uchar bits = payload[bb + 2 + inb / 8]; acc += ((bits & (1 << (inb % 8))) ? scale : -scale) * x[xbase + c]; }\n"
"    partial[tid] = acc; threadgroup_barrier(mem_flags::mem_threadgroup); for (uint s = nt >> 1; s > 0; s >>= 1) { if (tid < s) partial[tid] += partial[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); } if (tid == 0) out[gid] = partial[0];\n"
"}\n"
"kernel void ornith_iq1_slice_many_b256_r8_tg(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], device const uint *slices [[buffer(4)]], uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
"    threadgroup float p0[256]; threadgroup float p1[256]; threadgroup float p2[256]; threadgroup float p3[256]; threadgroup float p4[256]; threadgroup float p5[256]; threadgroup float p6[256]; threadgroup float p7[256]; uint row_groups = (a.rows + 7) >> 3; uint slice_i = group / row_groups; uint row0 = (group - slice_i * row_groups) << 3; uint slice = slices[slice_i]; ulong xbase = (ulong)slice_i * a.x_stride;\n"
"    float acc0 = 0.0f; float acc1 = 0.0f; float acc2 = 0.0f; float acc3 = 0.0f; float acc4 = 0.0f; float acc5 = 0.0f; float acc6 = 0.0f; float acc7 = 0.0f; ulong elem0 = a.elem_offset + ((ulong)slice * a.rows + row0) * a.cols; ulong base0 = a.byte_base + (elem0 >> 8) * 34; ulong row_stride = ((ulong)a.cols >> 8) * 34;\n"
"    for (uint b = 0; b < (a.cols >> 8); b++) { ulong off = (ulong)b * 34; float xv = x[xbase + (b << 8) + tid]; ulong bb0 = base0 + off; float s0 = bf16_at(payload, bb0); uchar bits0 = payload[bb0 + 2 + tid / 8]; acc0 += ((bits0 & (1 << (tid & 7))) ? s0 : -s0) * xv;\n"
"        if (row0 + 1 < a.rows) { ulong bb = bb0 + row_stride; float ss = bf16_at(payload, bb); uchar bits = payload[bb + 2 + tid / 8]; acc1 += ((bits & (1 << (tid & 7))) ? ss : -ss) * xv; }\n"
"        if (row0 + 2 < a.rows) { ulong bb = bb0 + row_stride * 2; float ss = bf16_at(payload, bb); uchar bits = payload[bb + 2 + tid / 8]; acc2 += ((bits & (1 << (tid & 7))) ? ss : -ss) * xv; }\n"
"        if (row0 + 3 < a.rows) { ulong bb = bb0 + row_stride * 3; float ss = bf16_at(payload, bb); uchar bits = payload[bb + 2 + tid / 8]; acc3 += ((bits & (1 << (tid & 7))) ? ss : -ss) * xv; }\n"
"        if (row0 + 4 < a.rows) { ulong bb = bb0 + row_stride * 4; float ss = bf16_at(payload, bb); uchar bits = payload[bb + 2 + tid / 8]; acc4 += ((bits & (1 << (tid & 7))) ? ss : -ss) * xv; }\n"
"        if (row0 + 5 < a.rows) { ulong bb = bb0 + row_stride * 5; float ss = bf16_at(payload, bb); uchar bits = payload[bb + 2 + tid / 8]; acc5 += ((bits & (1 << (tid & 7))) ? ss : -ss) * xv; }\n"
"        if (row0 + 6 < a.rows) { ulong bb = bb0 + row_stride * 6; float ss = bf16_at(payload, bb); uchar bits = payload[bb + 2 + tid / 8]; acc6 += ((bits & (1 << (tid & 7))) ? ss : -ss) * xv; }\n"
"        if (row0 + 7 < a.rows) { ulong bb = bb0 + row_stride * 7; float ss = bf16_at(payload, bb); uchar bits = payload[bb + 2 + tid / 8]; acc7 += ((bits & (1 << (tid & 7))) ? ss : -ss) * xv; }}\n"
"    p0[tid] = acc0; p1[tid] = acc1; p2[tid] = acc2; p3[tid] = acc3; p4[tid] = acc4; p5[tid] = acc5; p6[tid] = acc6; p7[tid] = acc7; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { p0[tid] += p0[tid + s]; p1[tid] += p1[tid + s]; p2[tid] += p2[tid + s]; p3[tid] += p3[tid + s]; p4[tid] += p4[tid + s]; p5[tid] += p5[tid + s]; p6[tid] += p6[tid + s]; p7[tid] += p7[tid + s]; } threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"    if (tid == 0) { ulong o = (ulong)slice_i * a.rows + row0; out[o] = p0[0]; if (row0 + 1 < a.rows) out[o + 1] = p1[0]; if (row0 + 2 < a.rows) out[o + 2] = p2[0]; if (row0 + 3 < a.rows) out[o + 3] = p3[0]; if (row0 + 4 < a.rows) out[o + 4] = p4[0]; if (row0 + 5 < a.rows) out[o + 5] = p5[0]; if (row0 + 6 < a.rows) out[o + 6] = p6[0]; if (row0 + 7 < a.rows) out[o + 7] = p7[0]; }\n"
"}\n"
"kernel void ornith_gate_up_silu(device const float *gate_up [[buffer(0)]], device float *mid [[buffer(1)]], constant Args &a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= a.total_rows) return; uint k = gid / a.rows; uint i = gid - k * a.rows; ulong base = (ulong)k * a.rows * 2; float g = gate_up[base + i]; float u = gate_up[base + a.rows + i]; mid[gid] = (g / (1.0f + exp(-g))) * u;\n"
"}\n"
"kernel void ornith_pair_silu_product(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device float *mid [[buffer(2)]], constant Args &a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= a.total_rows) return; float v = g[gid]; mid[gid] = (v / (1.0f + exp(-v))) * u[gid];\n"
"}\n"
"kernel void ornith_weighted_mix(device const float *down [[buffer(0)]], device const float *weights [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; for (uint k = 0; k < a.cols; k++) acc += weights[k] * down[(ulong)k * a.rows + row]; out[row] = acc;\n"
"}\n"
"static inline float ornith_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }\n"
"static inline float ornith_silu(float x) { return x * ornith_sigmoid(x); }\n"
"static inline float ornith_softplus(float x) { return x <= 20.0f ? log(1.0f + exp(x)) : x; }\n"
"kernel void ornith_gdn_recurrent_step(device const float *qkv [[buffer(0)]], device const float *z [[buffer(1)]], device const float *a_in [[buffer(2)]], device const float *b_in [[buffer(3)]], device const float *alog [[buffer(4)]], device const float *dt [[buffer(5)]], device const float *norm_w [[buffer(6)]], device float *ssm [[buffer(7)]], device float *gated [[buffer(8)]], constant GdnArgs &ga [[buffer(9)]], uint hv [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
"    threadgroup float core[256]; threadgroup float partial[256]; uint values_per_key = ga.value_heads / ga.key_heads; uint h = hv / values_per_key; uint key_dim = ga.key_heads * ga.head_k; float qss = 0.0f; float kss = 0.0f;\n"
"    for (uint ki = 0; ki < ga.head_k; ki++) { float q = qkv[h * ga.head_k + ki]; float k = qkv[key_dim + h * ga.head_k + ki]; qss += q * q; kss += k * k; }\n"
"    float qscale = rsqrt(qss + 1.0e-6f); float kscale = rsqrt(kss + 1.0e-6f); float decay = exp(-exp(alog[hv]) * ornith_softplus(a_in[hv] + dt[hv])); float beta = ornith_sigmoid(b_in[hv]); float c = 0.0f;\n"
"    if (tid < ga.head_v) { ulong row_base = ((ulong)hv * ga.head_v + tid) * ga.head_k; float proj = 0.0f; for (uint ki = 0; ki < ga.head_k; ki++) { float kval = qkv[key_dim + h * ga.head_k + ki] * kscale; float old = ssm[row_base + ki] * decay; ssm[row_base + ki] = old; proj += old * kval; } float v = qkv[key_dim * 2 + hv * ga.head_v + tid]; float vv = (v - proj) * beta; float sum = 0.0f; for (uint ki = 0; ki < ga.head_k; ki++) { float kval = qkv[key_dim + h * ga.head_k + ki] * kscale; float updated = ssm[row_base + ki] + vv * kval; ssm[row_base + ki] = updated; sum += updated * qkv[h * ga.head_k + ki] * qscale; } c = sum * rsqrt((float)ga.head_k); }\n"
"    core[tid] = c; partial[tid] = tid < ga.head_v ? c * c : 0.0f; threadgroup_barrier(mem_flags::mem_threadgroup); for (uint s = nt >> 1; s > 0; s >>= 1) { if (tid < s) partial[tid] += partial[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); } if (tid < ga.head_v) { float scale = rsqrt(partial[0] / (float)ga.head_v + 1.0e-6f); ulong o = (ulong)hv * ga.head_v + tid; gated[o] = core[tid] * scale * norm_w[tid] * ornith_silu(z[o]); }\n"
"}\n";

typedef struct {
    uint64_t byte_base;
    uint64_t elem_offset;
    uint32_t rows;
    uint32_t cols;
    uint32_t block;
    uint32_t x_stride;
    uint32_t total_rows;
} ornith_metal_args;

typedef struct {
    uint32_t value_heads;
    uint32_t head_v;
    uint32_t key_heads;
    uint32_t head_k;
} ornith_metal_gdn_args;

static double ornith_now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void set_err(char *err, size_t errcap, NSString *msg)
{
    if (err && errcap) snprintf(err, errcap, "%s", msg.UTF8String);
}

static int router_mode(void)
{
    static int mode = -1;
    if (mode >= 0) return mode;
    const char *env = getenv("ORNITH_METAL_ROUTER");
    if (env && strcmp(env, "0") == 0) mode = 0;
    else if (env && strcmp(env, "serial") == 0) mode = 1;
    else if (env && strcmp(env, "parallel") == 0) mode = 2;
    else if (env && (strcmp(env, "specialized") == 0 || strcmp(env, "q4") == 0)) mode = 3;
    else mode = 3;
    return mode;
}

static int q4_row8_mode(void)
{
    static int mode = -1;
    if (mode >= 0) return mode;
    const char *env = getenv("ORNITH_METAL_Q4_ROW8");
    mode = !env || strcmp(env, "0") != 0;
    return mode;
}

static id<MTLDevice> device(void)
{
    static id<MTLDevice> d;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ d = MTLCreateSystemDefaultDevice(); });
    return d;
}

static id<MTLCommandQueue> command_queue(void)
{
    static id<MTLCommandQueue> q;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ q = [device() newCommandQueue]; });
    return q;
}

static id<MTLBuffer> span_buffer(const unsigned char *span, uint64_t span_size)
{
    static NSMutableDictionary<NSValue *, id<MTLBuffer>> *cache;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ cache = [[NSMutableDictionary alloc] init]; });
    NSValue *key = [NSValue valueWithPointer:span];
    id<MTLBuffer> b = cache[key];
    if (b) return b;
    b = [device() newBufferWithBytesNoCopy:(void *)span length:(NSUInteger)span_size options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) b = [device() newBufferWithBytes:span length:(NSUInteger)span_size options:MTLResourceStorageModeShared];
    if (b) cache[key] = b;
    return b;
}

static id<MTLBuffer> temp_buffer(int slot, NSUInteger length)
{
    static id<MTLBuffer> buffers[16];
    static NSUInteger caps[16];
    if (slot < 0 || slot >= 16 || length == 0) return nil;
    if (!buffers[slot] || caps[slot] < length) {
        NSUInteger cap = 4096;
        while (cap < length) cap *= 2;
        buffers[slot] = [device() newBufferWithLength:cap options:MTLResourceStorageModeShared];
        caps[slot] = cap;
    }
    return buffers[slot];
}

typedef struct {
    const ornith_tensor_info *tensor;
    id<MTLBuffer> buffer;
    size_t bytes;
} ornith_metal_resident_tensor;

static size_t resident_layer_limit(void)
{
    static int init;
    static size_t limit;
    if (init) return limit;
    init = 1;
    const char *env = getenv("ORNITH_METAL_RESIDENT_LAYER_MB");
    unsigned long mb = env && env[0] ? strtoul(env, NULL, 10) : 0;
    limit = (size_t)mb * 1024u * 1024u;
    return limit;
}

static id<MTLBuffer> resident_layer_tensor_buffer(const ornith_tensor_info *tensor, const unsigned char *src)
{
    enum { max_layers = 128 };
    static ornith_metal_resident_tensor cache[max_layers][2];
    static size_t used;
    if (!tensor || tensor->layer < 0 || tensor->layer >= max_layers || !src || resident_layer_limit() == 0 ||
        (uint64_t)(NSUInteger)tensor->nbytes != tensor->nbytes) return nil;
    int slot = strcmp(tensor->kind, "mlp.experts.gate_up_proj") == 0 ? 0 :
               strcmp(tensor->kind, "mlp.experts.down_proj") == 0 ? 1 : -1;
    if (slot < 0) return nil;
    ornith_metal_resident_tensor *e = &cache[tensor->layer][slot];
    if (e->tensor == tensor && e->buffer && e->bytes == (size_t)tensor->nbytes) return e->buffer;
    if (e->buffer || used + (size_t)tensor->nbytes > resident_layer_limit()) return nil;
    id<MTLBuffer> b = [device() newBufferWithBytes:src length:(NSUInteger)tensor->nbytes options:MTLResourceStorageModeShared];
    if (!b) return nil;
    e->tensor = tensor;
    e->buffer = b;
    e->bytes = (size_t)tensor->nbytes;
    used += e->bytes;
    return b;
}

static id<MTLComputePipelineState> pipeline(NSString *name, char *err, size_t errcap)
{
    static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *cache;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ cache = [[NSMutableDictionary alloc] init]; });
    id<MTLComputePipelineState> p = cache[name];
    if (p) return p;
    NSError *e = nil;
    id<MTLLibrary> lib = [device() newLibraryWithSource:ORNITH_METAL_SRC options:nil error:&e];
    if (!lib) {
        set_err(err, errcap, e.localizedDescription ?: @"metal library compile failed");
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        set_err(err, errcap, @"metal function missing");
        return nil;
    }
    p = [device() newComputePipelineStateWithFunction:fn error:&e];
    if (!p) {
        set_err(err, errcap, e.localizedDescription ?: @"metal pipeline failed");
        return nil;
    }
    cache[name] = p;
    return p;
}

int ornith_metal_available(void)
{
    return device() != nil;
}

static NSString *matvec_kernel(const ornith_tensor_info *tensor, uint32_t block, size_t rows, size_t cols, BOOL *use_tg, char *err, size_t errcap)
{
    *use_tg = cols >= 128 && rows <= 16384;
    if (tensor->quant == ORNITH_QUANT_BF16) return *use_tg ? @"ornith_bf16_matvec_tg" : @"ornith_bf16_matvec";
    if (tensor->quant == ORNITH_QUANT_Q4 && tensor->ndim == 2 && block == 256 && (cols % 256) == 0) return *use_tg ? (q4_row8_mode() ? @"ornith_q4_router_b256_r8_tg" : @"ornith_q4_matvec_b256_r4_tg") : @"ornith_q4_matvec_b256";
    if (tensor->quant == ORNITH_QUANT_Q4 && tensor->ndim == 2) return *use_tg ? @"ornith_q4_matvec_tg" : @"ornith_q4_matvec";
    if (tensor->quant == ORNITH_QUANT_IQ1) return *use_tg ? @"ornith_iq1_matvec_tg" : @"ornith_iq1_matvec";
    set_err(err, errcap, @"unsupported quant mode");
    return nil;
}

static BOOL matvec_kernel_rows4(NSString *kernel)
{
    return [kernel isEqualToString:@"ornith_q4_matvec_b256_r4_tg"];
}

static BOOL matvec_kernel_rows8(NSString *kernel)
{
    return [kernel isEqualToString:@"ornith_q4_router_b256_r8_tg"];
}

static NSUInteger matvec_kernel_thread_count(NSString *kernel, BOOL use_tg, NSUInteger max_threads)
{
    if (matvec_kernel_rows8(kernel)) return 64;
    return use_tg ? 256 : MIN(max_threads, (NSUInteger)256);
}

static NSUInteger matvec_kernel_group_count(NSString *kernel, size_t rows)
{
    if (matvec_kernel_rows8(kernel)) return (rows + 7) / 8;
    if (matvec_kernel_rows4(kernel)) return (rows + 3) / 4;
    return rows;
}

int ornith_metal_tensor_matvec(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    uint64_t slice,
    const float *x,
    size_t x_count,
    float *out,
    char *err,
    size_t errcap)
{
    @autoreleasepool {
        if (!ornith_metal_available()) {
            set_err(err, errcap, @"no Metal device");
            return 0;
        }
        if (!model || !tensor || !x || !out) {
            set_err(err, errcap, @"bad arguments");
            return 0;
        }
        uint64_t byte_base = 0, span_size = 0;
        uint32_t block = 0;
        const unsigned char *span = ornith_tensor_mapped_span(model, tensor, &byte_base, &span_size, &block);
        if (!span) {
            set_err(err, errcap, @"tensor is not mapped");
            return 0;
        }

        uint64_t elem_offset = 0;
        size_t rows = 0, cols = 0;
        if (tensor->ndim == 2) {
            rows = (size_t)tensor->shape[0];
            cols = (size_t)tensor->shape[1];
        } else if (tensor->ndim == 3 && slice < (uint64_t)tensor->shape[0]) {
            rows = (size_t)tensor->shape[1];
            cols = (size_t)tensor->shape[2];
            elem_offset = slice * rows * cols;
        } else {
            set_err(err, errcap, @"unsupported tensor rank");
            return 0;
        }
        if (x_count != cols || rows > UINT32_MAX || cols > UINT32_MAX) {
            set_err(err, errcap, @"shape mismatch");
            return 0;
        }

        BOOL use_tg = NO;
        NSString *kernel = matvec_kernel(tensor, block, rows, cols, &use_tg, err, errcap);
        if (!kernel) return 0;

        id<MTLComputePipelineState> p = pipeline(kernel, err, errcap);
        if (!p) return 0;
        id<MTLCommandQueue> q = command_queue();
        id<MTLBuffer> payload_buf = span_buffer(span, span_size);
        ornith_metal_args args = { byte_base, elem_offset, (uint32_t)rows, (uint32_t)cols, block, 0, (uint32_t)rows };
        id<MTLBuffer> x_buf = temp_buffer(0, cols * sizeof(float));
        id<MTLBuffer> out_buf = temp_buffer(1, rows * sizeof(float));
        id<MTLBuffer> args_buf = temp_buffer(2, sizeof(args));
        if (!q || !payload_buf || !x_buf || !out_buf || !args_buf) {
            set_err(err, errcap, @"metal buffer allocation failed");
            return 0;
        }
        memcpy(x_buf.contents, x, cols * sizeof(float));
        memcpy(args_buf.contents, &args, sizeof(args));
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:payload_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];
        [enc setBuffer:args_buf offset:0 atIndex:3];
        NSUInteger tg = matvec_kernel_thread_count(kernel, use_tg, (NSUInteger)p.maxTotalThreadsPerThreadgroup);
        if (use_tg) {
            NSUInteger groups = matvec_kernel_group_count(kernel, rows);
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        } else {
            [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            set_err(err, errcap, cb.error.localizedDescription ?: @"metal command failed");
            return 0;
        }
        memcpy(out, out_buf.contents, rows * sizeof(float));
        return 1;
    }
}

static int ornith_metal_tensor_matvec_batch(
    const ornith_model *model,
    const ornith_tensor_info * const *tensors,
    size_t count,
    const float *x,
    size_t x_count,
    float **outs,
    char *err,
    size_t errcap)
{
    @autoreleasepool {
        if (!ornith_metal_available()) {
            set_err(err, errcap, @"no Metal device");
            return 0;
        }
        if (!model || !tensors || !count || count > 4 || !x || !outs || x_count > UINT32_MAX) {
            set_err(err, errcap, @"bad batch matvec arguments");
            return 0;
        }
        id<MTLComputePipelineState> pipes[4] = {nil, nil, nil, nil};
        NSString *kernels[4] = {nil, nil, nil, nil};
        id<MTLBuffer> payloads[4] = {nil, nil, nil, nil};
        id<MTLBuffer> out_bufs[4] = {nil, nil, nil, nil};
        id<MTLBuffer> arg_bufs[4] = {nil, nil, nil, nil};
        ornith_metal_args args[4];
        size_t rows[4] = {0, 0, 0, 0};
        BOOL use_tg[4] = {NO, NO, NO, NO};
        for (size_t i = 0; i < count; i++) {
            const ornith_tensor_info *t = tensors[i];
            if (!t || !outs[i] || t->ndim != 2 || x_count != (size_t)t->shape[1] ||
                (size_t)t->shape[0] > UINT32_MAX || (size_t)t->shape[1] > UINT32_MAX) {
                set_err(err, errcap, @"bad batch matvec shape");
                return 0;
            }
            uint64_t byte_base = 0, span_size = 0;
            uint32_t block = 0;
            const unsigned char *span = ornith_tensor_mapped_span(model, t, &byte_base, &span_size, &block);
            if (!span) {
                set_err(err, errcap, @"batch tensor is not mapped");
                return 0;
            }
            rows[i] = (size_t)t->shape[0];
            kernels[i] = matvec_kernel(t, block, rows[i], x_count, &use_tg[i], err, errcap);
            if (!kernels[i]) return 0;
            pipes[i] = pipeline(kernels[i], err, errcap);
            payloads[i] = span_buffer(span, span_size);
            out_bufs[i] = temp_buffer(1 + (int)i * 2, rows[i] * sizeof(float));
            arg_bufs[i] = temp_buffer(2 + (int)i * 2, sizeof(args[i]));
            args[i] = (ornith_metal_args){ byte_base, 0, (uint32_t)rows[i], (uint32_t)x_count, block, 0, (uint32_t)rows[i] };
            if (!pipes[i] || !payloads[i] || !out_bufs[i] || !arg_bufs[i]) {
                set_err(err, errcap, @"metal batch allocation failed");
                return 0;
            }
            memcpy(arg_bufs[i].contents, &args[i], sizeof(args[i]));
        }
        id<MTLBuffer> x_buf = temp_buffer(0, x_count * sizeof(float));
        if (!x_buf) {
            set_err(err, errcap, @"metal batch x allocation failed");
            return 0;
        }
        memcpy(x_buf.contents, x, x_count * sizeof(float));
        id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (size_t i = 0; i < count; i++) {
            [enc setComputePipelineState:pipes[i]];
            [enc setBuffer:payloads[i] offset:0 atIndex:0];
            [enc setBuffer:x_buf offset:0 atIndex:1];
            [enc setBuffer:out_bufs[i] offset:0 atIndex:2];
            [enc setBuffer:arg_bufs[i] offset:0 atIndex:3];
            NSUInteger tg = matvec_kernel_thread_count(kernels[i], use_tg[i], (NSUInteger)pipes[i].maxTotalThreadsPerThreadgroup);
            if (use_tg[i]) {
                NSUInteger groups = matvec_kernel_group_count(kernels[i], rows[i]);
                [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            } else {
                [enc dispatchThreads:MTLSizeMake(rows[i], 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            }
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            set_err(err, errcap, cb.error.localizedDescription ?: @"metal batch command failed");
            return 0;
        }
        for (size_t i = 0; i < count; i++) {
            memcpy(outs[i], out_bufs[i].contents, rows[i] * sizeof(float));
        }
        return 1;
    }
}

int ornith_metal_gdn_recurrent_step(
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
    float *gated,
    char *err,
    size_t errcap)
{
    @autoreleasepool {
        if (!ornith_metal_available()) {
            set_err(err, errcap, @"no Metal device");
            return 0;
        }
        if (!qkv || !z || !a || !b || !alog || !dt || !norm_w || !ssm || !gated ||
            !value_heads || !head_v || !key_heads || !head_k || value_heads % key_heads != 0 ||
            value_heads > UINT32_MAX || head_v > 256 || key_heads > UINT32_MAX || head_k > 256) {
            set_err(err, errcap, @"bad gdn recurrent args");
            return 0;
        }
        size_t key_dim = key_heads * head_k;
        size_t value_dim = value_heads * head_v;
        size_t qkv_dim = key_dim * 2 + value_dim;
        size_t ssm_count = value_heads * head_v * head_k;
        id<MTLComputePipelineState> p = pipeline(@"ornith_gdn_recurrent_step", err, errcap);
        id<MTLBuffer> qkv_buf = temp_buffer(0, qkv_dim * sizeof(float));
        id<MTLBuffer> z_buf = temp_buffer(1, value_dim * sizeof(float));
        id<MTLBuffer> a_buf = temp_buffer(2, value_heads * sizeof(float));
        id<MTLBuffer> b_buf = temp_buffer(3, value_heads * sizeof(float));
        id<MTLBuffer> alog_buf = temp_buffer(4, value_heads * sizeof(float));
        id<MTLBuffer> dt_buf = temp_buffer(5, value_heads * sizeof(float));
        id<MTLBuffer> norm_buf = temp_buffer(6, head_v * sizeof(float));
        id<MTLBuffer> ssm_buf = temp_buffer(7, ssm_count * sizeof(float));
        id<MTLBuffer> gated_buf = temp_buffer(8, value_dim * sizeof(float));
        id<MTLBuffer> args_buf = temp_buffer(9, sizeof(ornith_metal_gdn_args));
        if (!p || !qkv_buf || !z_buf || !a_buf || !b_buf || !alog_buf || !dt_buf || !norm_buf || !ssm_buf || !gated_buf || !args_buf) {
            set_err(err, errcap, @"metal buffer allocation failed");
            return 0;
        }
        ornith_metal_gdn_args args = { (uint32_t)value_heads, (uint32_t)head_v, (uint32_t)key_heads, (uint32_t)head_k };
        memcpy(qkv_buf.contents, qkv, qkv_dim * sizeof(float));
        memcpy(z_buf.contents, z, value_dim * sizeof(float));
        memcpy(a_buf.contents, a, value_heads * sizeof(float));
        memcpy(b_buf.contents, b, value_heads * sizeof(float));
        memcpy(alog_buf.contents, alog, value_heads * sizeof(float));
        memcpy(dt_buf.contents, dt, value_heads * sizeof(float));
        memcpy(norm_buf.contents, norm_w, head_v * sizeof(float));
        memcpy(ssm_buf.contents, ssm, ssm_count * sizeof(float));
        memcpy(args_buf.contents, &args, sizeof(args));
        id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:qkv_buf offset:0 atIndex:0];
        [enc setBuffer:z_buf offset:0 atIndex:1];
        [enc setBuffer:a_buf offset:0 atIndex:2];
        [enc setBuffer:b_buf offset:0 atIndex:3];
        [enc setBuffer:alog_buf offset:0 atIndex:4];
        [enc setBuffer:dt_buf offset:0 atIndex:5];
        [enc setBuffer:norm_buf offset:0 atIndex:6];
        [enc setBuffer:ssm_buf offset:0 atIndex:7];
        [enc setBuffer:gated_buf offset:0 atIndex:8];
        [enc setBuffer:args_buf offset:0 atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            set_err(err, errcap, cb.error.localizedDescription ?: @"metal command failed");
            return 0;
        }
        memcpy(gated, gated_buf.contents, value_dim * sizeof(float));
        memcpy(ssm, ssm_buf.contents, ssm_count * sizeof(float));
        return 1;
    }
}

static int ornith_metal_gdn_recurrent_out_proj(
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
    const ornith_model *model,
    const ornith_tensor_info *out_w,
    float *out,
    char *err,
    size_t errcap)
{
    @autoreleasepool {
        if (!out_w || !out || out_w->quant != ORNITH_QUANT_Q4 || out_w->ndim != 2 ||
            !value_heads || !head_v || !key_heads || !head_k || value_heads % key_heads != 0 ||
            out_w->shape[1] != (int64_t)(value_heads * head_v)) {
            return -1;
        }
        size_t out_rows = (size_t)out_w->shape[0];
        size_t value_dim = value_heads * head_v;
        if (out_rows > UINT32_MAX || value_dim > UINT32_MAX || (value_dim % 256) != 0) return -1;

        uint64_t out_base = 0, out_span_size = 0;
        uint32_t out_block = 0;
        const unsigned char *out_span = ornith_tensor_mapped_span(model, out_w, &out_base, &out_span_size, &out_block);
        if (!out_span || out_block != 256) return -1;

        if (!qkv || !z || !a || !b || !alog || !dt || !norm_w || !ssm ||
            value_heads > UINT32_MAX || head_v > 256 || key_heads > UINT32_MAX || head_k > 256) {
            set_err(err, errcap, @"bad fused gdn args");
            return 0;
        }
        size_t key_dim = key_heads * head_k;
        size_t qkv_dim = key_dim * 2 + value_dim;
        size_t ssm_count = value_heads * head_v * head_k;
        id<MTLComputePipelineState> gdn_p = pipeline(@"ornith_gdn_recurrent_step", err, errcap);
        id<MTLComputePipelineState> out_p = pipeline(@"ornith_q4_matvec_b256_r4_tg", err, errcap);
        id<MTLBuffer> qkv_buf = temp_buffer(0, qkv_dim * sizeof(float));
        id<MTLBuffer> z_buf = temp_buffer(1, value_dim * sizeof(float));
        id<MTLBuffer> a_buf = temp_buffer(2, value_heads * sizeof(float));
        id<MTLBuffer> b_buf = temp_buffer(3, value_heads * sizeof(float));
        id<MTLBuffer> alog_buf = temp_buffer(4, value_heads * sizeof(float));
        id<MTLBuffer> dt_buf = temp_buffer(5, value_heads * sizeof(float));
        id<MTLBuffer> norm_buf = temp_buffer(6, head_v * sizeof(float));
        id<MTLBuffer> ssm_buf = temp_buffer(7, ssm_count * sizeof(float));
        id<MTLBuffer> gated_buf = temp_buffer(8, value_dim * sizeof(float));
        id<MTLBuffer> gdn_args_buf = temp_buffer(9, sizeof(ornith_metal_gdn_args));
        id<MTLBuffer> out_buf = temp_buffer(10, out_rows * sizeof(float));
        id<MTLBuffer> out_args_buf = temp_buffer(11, sizeof(ornith_metal_args));
        id<MTLBuffer> out_payload = span_buffer(out_span, out_span_size);
        if (!gdn_p || !out_p || !qkv_buf || !z_buf || !a_buf || !b_buf || !alog_buf || !dt_buf ||
            !norm_buf || !ssm_buf || !gated_buf || !gdn_args_buf || !out_buf || !out_args_buf ||
            !out_payload) {
            set_err(err, errcap, @"metal fused gdn allocation failed");
            return 0;
        }

        ornith_metal_gdn_args gdn_args = { (uint32_t)value_heads, (uint32_t)head_v, (uint32_t)key_heads, (uint32_t)head_k };
        ornith_metal_args out_args = { out_base, 0, (uint32_t)out_rows, (uint32_t)value_dim, out_block, 0, (uint32_t)out_rows };
        memcpy(qkv_buf.contents, qkv, qkv_dim * sizeof(float));
        memcpy(z_buf.contents, z, value_dim * sizeof(float));
        memcpy(a_buf.contents, a, value_heads * sizeof(float));
        memcpy(b_buf.contents, b, value_heads * sizeof(float));
        memcpy(alog_buf.contents, alog, value_heads * sizeof(float));
        memcpy(dt_buf.contents, dt, value_heads * sizeof(float));
        memcpy(norm_buf.contents, norm_w, head_v * sizeof(float));
        memcpy(ssm_buf.contents, ssm, ssm_count * sizeof(float));
        memcpy(gdn_args_buf.contents, &gdn_args, sizeof(gdn_args));
        memcpy(out_args_buf.contents, &out_args, sizeof(out_args));

        id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gdn_p];
        [enc setBuffer:qkv_buf offset:0 atIndex:0];
        [enc setBuffer:z_buf offset:0 atIndex:1];
        [enc setBuffer:a_buf offset:0 atIndex:2];
        [enc setBuffer:b_buf offset:0 atIndex:3];
        [enc setBuffer:alog_buf offset:0 atIndex:4];
        [enc setBuffer:dt_buf offset:0 atIndex:5];
        [enc setBuffer:norm_buf offset:0 atIndex:6];
        [enc setBuffer:ssm_buf offset:0 atIndex:7];
        [enc setBuffer:gated_buf offset:0 atIndex:8];
        [enc setBuffer:gdn_args_buf offset:0 atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:out_p];
        [enc setBuffer:out_payload offset:0 atIndex:0];
        [enc setBuffer:gated_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];
        [enc setBuffer:out_args_buf offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((out_rows + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            set_err(err, errcap, cb.error.localizedDescription ?: @"metal fused gdn command failed");
            return 0;
        }
        memcpy(out, out_buf.contents, out_rows * sizeof(float));
        memcpy(ssm, ssm_buf.contents, ssm_count * sizeof(float));
        return 2;
    }
}

static int ornith_metal_tensor_matvec_rows(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    const float *x,
    size_t x_count,
    size_t rows,
    float *out,
    char *err,
    size_t errcap)
{
    if (!model || !tensor || !x || !out || tensor->ndim != 2 || rows == 0 || rows > (size_t)tensor->shape[0] ||
        x_count != (size_t)tensor->shape[1]) {
        set_err(err, errcap, @"bad limited matvec args");
        return 0;
    }
    uint64_t byte_base = 0, span_size = 0;
    uint32_t block = 0;
    const unsigned char *span = ornith_tensor_mapped_span(model, tensor, &byte_base, &span_size, &block);
    if (!span || rows > UINT32_MAX || x_count > UINT32_MAX) {
        set_err(err, errcap, @"bad limited matvec shape");
        return 0;
    }
    BOOL use_tg = x_count >= 128 && rows <= 16384;
    NSString *kernel = nil;
    if (tensor->quant == ORNITH_QUANT_BF16) kernel = use_tg ? @"ornith_bf16_matvec_tg" : @"ornith_bf16_matvec";
    else if (tensor->quant == ORNITH_QUANT_Q4 && block == 256 && (x_count % 256) == 0) kernel = use_tg ? (q4_row8_mode() ? @"ornith_q4_router_b256_r8_tg" : @"ornith_q4_matvec_b256_r4_tg") : @"ornith_q4_matvec_b256";
    else if (tensor->quant == ORNITH_QUANT_Q4) kernel = use_tg ? @"ornith_q4_matvec_tg" : @"ornith_q4_matvec";
    else if (tensor->quant == ORNITH_QUANT_IQ1) kernel = use_tg ? @"ornith_iq1_matvec_tg" : @"ornith_iq1_matvec";
    else {
        set_err(err, errcap, @"unsupported quant mode");
        return 0;
    }
    id<MTLComputePipelineState> p = pipeline(kernel, err, errcap);
    if (!p) return 0;
    id<MTLBuffer> payload_buf = span_buffer(span, span_size);
    id<MTLBuffer> x_buf = temp_buffer(0, x_count * sizeof(float));
    id<MTLBuffer> out_buf = temp_buffer(1, rows * sizeof(float));
    ornith_metal_args args = { byte_base, 0, (uint32_t)rows, (uint32_t)x_count, block, 0, (uint32_t)rows };
    id<MTLBuffer> args_buf = temp_buffer(2, sizeof(args));
    if (!payload_buf || !x_buf || !out_buf || !args_buf) {
        set_err(err, errcap, @"metal buffer allocation failed");
        return 0;
    }
    memcpy(x_buf.contents, x, x_count * sizeof(float));
    memcpy(args_buf.contents, &args, sizeof(args));
    id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:payload_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBuffer:args_buf offset:0 atIndex:3];
    NSUInteger tg = matvec_kernel_thread_count(kernel, use_tg, (NSUInteger)p.maxTotalThreadsPerThreadgroup);
    if (use_tg) {
        NSUInteger groups = matvec_kernel_group_count(kernel, rows);
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    } else {
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        set_err(err, errcap, cb.error.localizedDescription ?: @"metal command failed");
        return 0;
    }
    memcpy(out, out_buf.contents, rows * sizeof(float));
    return 1;
}

static int ornith_metal_tensor_matvec_serial_rows(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    const float *x,
    size_t x_count,
    size_t rows,
    float *out,
    char *err,
    size_t errcap)
{
    if (!model || !tensor || !x || !out || tensor->ndim != 2 || rows == 0 || rows > (size_t)tensor->shape[0] ||
        x_count != (size_t)tensor->shape[1] || rows > UINT32_MAX || x_count > UINT32_MAX) {
        set_err(err, errcap, @"bad serial matvec args");
        return 0;
    }
    uint64_t byte_base = 0, span_size = 0;
    uint32_t block = 0;
    const unsigned char *span = ornith_tensor_mapped_span(model, tensor, &byte_base, &span_size, &block);
    if (!span) {
        set_err(err, errcap, @"bad serial matvec shape");
        return 0;
    }
    NSString *kernel = nil;
    if (tensor->quant == ORNITH_QUANT_BF16) kernel = @"ornith_bf16_matvec";
    else if (tensor->quant == ORNITH_QUANT_Q4 && block == 256 && (x_count % 256) == 0) kernel = @"ornith_q4_matvec_b256";
    else if (tensor->quant == ORNITH_QUANT_Q4) kernel = @"ornith_q4_matvec";
    else if (tensor->quant == ORNITH_QUANT_IQ1) kernel = @"ornith_iq1_matvec";
    else {
        set_err(err, errcap, @"unsupported quant mode");
        return 0;
    }
    id<MTLComputePipelineState> p = pipeline(kernel, err, errcap);
    id<MTLBuffer> payload_buf = span_buffer(span, span_size);
    id<MTLBuffer> x_buf = temp_buffer(0, x_count * sizeof(float));
    id<MTLBuffer> out_buf = temp_buffer(1, rows * sizeof(float));
    ornith_metal_args args = { byte_base, 0, (uint32_t)rows, (uint32_t)x_count, block, 0, (uint32_t)rows };
    id<MTLBuffer> args_buf = temp_buffer(2, sizeof(args));
    if (!p || !payload_buf || !x_buf || !out_buf || !args_buf) {
        set_err(err, errcap, @"metal serial allocation failed");
        return 0;
    }
    memcpy(x_buf.contents, x, x_count * sizeof(float));
    memcpy(args_buf.contents, &args, sizeof(args));
    id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:payload_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBuffer:args_buf offset:0 atIndex:3];
    NSUInteger tg = MIN((NSUInteger)p.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        set_err(err, errcap, cb.error.localizedDescription ?: @"metal serial command failed");
        return 0;
    }
    memcpy(out, out_buf.contents, rows * sizeof(float));
    return 1;
}

static int ornith_metal_router_q4_b256(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    const float *x,
    size_t x_count,
    size_t rows,
    float *out,
    char *err,
    size_t errcap)
{
    if (!model || !tensor || !x || !out || tensor->quant != ORNITH_QUANT_Q4 ||
        tensor->ndim != 2 || rows == 0 || rows > (size_t)tensor->shape[0] ||
        x_count != (size_t)tensor->shape[1] || rows > UINT32_MAX || x_count > UINT32_MAX ||
        (x_count % 256) != 0) {
        return -1;
    }
    uint64_t byte_base = 0, span_size = 0;
    uint32_t block = 0;
    const unsigned char *span = ornith_tensor_mapped_span(model, tensor, &byte_base, &span_size, &block);
    if (!span || block != 256) return -1;

    id<MTLComputePipelineState> p = pipeline(@"ornith_q4_router_b256_r8_tg", err, errcap);
    id<MTLBuffer> payload_buf = span_buffer(span, span_size);
    id<MTLBuffer> x_buf = temp_buffer(0, x_count * sizeof(float));
    id<MTLBuffer> out_buf = temp_buffer(1, rows * sizeof(float));
    ornith_metal_args args = { byte_base, 0, (uint32_t)rows, (uint32_t)x_count, block, 0, (uint32_t)rows };
    id<MTLBuffer> args_buf = temp_buffer(2, sizeof(args));
    if (!p || !payload_buf || !x_buf || !out_buf || !args_buf) {
        set_err(err, errcap, @"metal router allocation failed");
        return 0;
    }
    memcpy(x_buf.contents, x, x_count * sizeof(float));
    memcpy(args_buf.contents, &args, sizeof(args));

    id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:payload_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBuffer:args_buf offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((rows + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        set_err(err, errcap, cb.error.localizedDescription ?: @"metal router command failed");
        return 0;
    }
    memcpy(out, out_buf.contents, rows * sizeof(float));
    return 1;
}

#ifdef ORNITH_TESTING
int ornith_metal_test_router_q4_b256(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    const float *x,
    size_t x_count,
    size_t rows,
    float *out,
    char *err,
    size_t errcap)
{
    return ornith_metal_router_q4_b256(model, tensor, x, x_count, rows, out, err, errcap);
}
#endif

static int ornith_metal_iq1_slice_many(
    const ornith_model *model,
    const ornith_tensor_info *tensor,
    const size_t *slices,
    size_t nslices,
    const float *x,
    size_t x_count,
    size_t x_stride,
    float *out,
    char *err,
    size_t errcap)
{
    if (!model || !tensor || !slices || !nslices || !x || !out || tensor->quant != ORNITH_QUANT_IQ1 || tensor->ndim != 3) {
        set_err(err, errcap, @"bad batched iq1 args");
        return 0;
    }
    size_t rows = (size_t)tensor->shape[1];
    size_t cols = (size_t)tensor->shape[2];
    if ((x_stride == 0 && x_count != cols) || (x_stride != 0 && x_count != nslices * cols) ||
        rows > UINT32_MAX || cols > UINT32_MAX || nslices * rows > UINT32_MAX) {
        set_err(err, errcap, @"bad batched iq1 shape");
        return 0;
    }
    uint64_t byte_base = 0, span_size = 0;
    uint32_t block = 0;
    const unsigned char *span = ornith_tensor_mapped_span(model, tensor, &byte_base, &span_size, &block);
    if (!span) {
        set_err(err, errcap, @"tensor is not mapped");
        return 0;
    }
    uint32_t *slice32 = malloc(nslices * sizeof(uint32_t));
    if (!slice32) {
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    for (size_t i = 0; i < nslices; i++) {
        if (slices[i] >= (size_t)tensor->shape[0]) {
            free(slice32);
            set_err(err, errcap, @"expert slice out of range");
            return 0;
        }
        slice32[i] = (uint32_t)slices[i];
    }

    BOOL use_b256 = block == 256 && (cols % 256) == 0;
    id<MTLComputePipelineState> p = pipeline(use_b256 ? @"ornith_iq1_slice_many_b256_r8_tg" : @"ornith_iq1_slice_many_tg", err, errcap);
    if (!p) {
        free(slice32);
        return 0;
    }
    id<MTLBuffer> payload_buf = span_buffer(span, span_size);
    id<MTLBuffer> x_buf = temp_buffer(0, x_count * sizeof(float));
    id<MTLBuffer> out_buf = temp_buffer(1, nslices * rows * sizeof(float));
    ornith_metal_args args = { byte_base, 0, (uint32_t)rows, (uint32_t)cols, block, (uint32_t)x_stride, (uint32_t)(nslices * rows) };
    id<MTLBuffer> args_buf = temp_buffer(2, sizeof(args));
    id<MTLBuffer> slices_buf = temp_buffer(3, nslices * sizeof(uint32_t));
    if (!payload_buf || !x_buf || !out_buf || !args_buf || !slices_buf) {
        free(slice32);
        set_err(err, errcap, @"metal buffer allocation failed");
        return 0;
    }
    memcpy(x_buf.contents, x, x_count * sizeof(float));
    memcpy(args_buf.contents, &args, sizeof(args));
    memcpy(slices_buf.contents, slice32, nslices * sizeof(uint32_t));
    free(slice32);

    id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:payload_buf offset:0 atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBuffer:args_buf offset:0 atIndex:3];
    [enc setBuffer:slices_buf offset:0 atIndex:4];
    NSUInteger groups = use_b256 ? nslices * ((rows + 7) / 8) : nslices * rows;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        set_err(err, errcap, cb.error.localizedDescription ?: @"metal command failed");
        return 0;
    }
    memcpy(out, out_buf.contents, nslices * rows * sizeof(float));
    return 1;
}

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
    size_t errcap)
{
    return ornith_metal_iq1_slice_many(model, tensor, slices, nslices, x, x_count, x_stride, out, err, errcap);
}
#endif

static int ornith_metal_routed_mlp_b256(
    const ornith_model *model,
    const ornith_tensor_info *gate_up,
    const ornith_tensor_info *down,
    const size_t *slices,
    const float *weights,
    size_t nslices,
    const float *norm,
    size_t hidden,
    float *out,
    char *err,
    size_t errcap)
{
    if (!model || !gate_up || !down || !slices || !weights || !nslices || !norm || !out ||
        gate_up->quant != ORNITH_QUANT_IQ1 || down->quant != ORNITH_QUANT_IQ1 ||
        gate_up->ndim != 3 || down->ndim != 3 || gate_up->shape[0] != down->shape[0] ||
        gate_up->shape[2] != (int64_t)hidden || down->shape[1] != (int64_t)hidden ||
        gate_up->shape[1] != down->shape[2] * 2) {
        return -1;
    }
    size_t gate_up_rows = (size_t)gate_up->shape[1];
    size_t inter = (size_t)down->shape[2];
    if (hidden > UINT32_MAX || gate_up_rows > UINT32_MAX || inter > UINT32_MAX ||
        nslices * gate_up_rows > UINT32_MAX || nslices * inter > UINT32_MAX || nslices * hidden > UINT32_MAX) {
        return -1;
    }

    uint64_t gate_byte_base = 0, gate_span_size = 0, down_byte_base = 0, down_span_size = 0;
    uint32_t gate_block = 0, down_block = 0;
    const unsigned char *gate_span = ornith_tensor_mapped_span(model, gate_up, &gate_byte_base, &gate_span_size, &gate_block);
    const unsigned char *down_span = ornith_tensor_mapped_span(model, down, &down_byte_base, &down_span_size, &down_block);
    if (!gate_span || !down_span || gate_block != 256 || down_block != 256 || (hidden % 256) != 0 || (inter % 256) != 0) {
        return -1;
    }

    uint32_t *slice32 = malloc(nslices * sizeof(uint32_t));
    if (!slice32) {
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    for (size_t i = 0; i < nslices; i++) {
        if (slices[i] >= (size_t)gate_up->shape[0]) {
            free(slice32);
            set_err(err, errcap, @"expert slice out of range");
            return 0;
        }
        slice32[i] = (uint32_t)slices[i];
    }
    size_t gate_slice_elems = gate_up_rows * hidden;
    size_t down_slice_elems = hidden * inter;
    size_t gate_slice_bytes = (gate_slice_elems / 256) * 34;
    size_t down_slice_bytes = (down_slice_elems / 256) * 34;
    id<MTLBuffer> resident_gate = resident_layer_tensor_buffer(gate_up, gate_span + gate_byte_base);
    id<MTLBuffer> resident_down = resident_layer_tensor_buffer(down, down_span + down_byte_base);
    int use_resident = resident_gate && resident_down;

    id<MTLComputePipelineState> slice_p = pipeline(@"ornith_iq1_slice_many_b256_r8_tg", err, errcap);
    id<MTLComputePipelineState> act_p = pipeline(@"ornith_gate_up_silu", err, errcap);
    id<MTLComputePipelineState> mix_p = pipeline(@"ornith_weighted_mix", err, errcap);
    id<MTLBuffer> gate_payload = use_resident ? resident_gate : temp_buffer(11, nslices * gate_slice_bytes);
    id<MTLBuffer> down_payload = use_resident ? resident_down : temp_buffer(12, nslices * down_slice_bytes);
    id<MTLBuffer> norm_buf = temp_buffer(0, hidden * sizeof(float));
    id<MTLBuffer> gate_up_buf = temp_buffer(1, nslices * gate_up_rows * sizeof(float));
    id<MTLBuffer> gate_args_buf = temp_buffer(2, sizeof(ornith_metal_args));
    id<MTLBuffer> slices_buf = temp_buffer(3, nslices * sizeof(uint32_t));
    id<MTLBuffer> mid_buf = temp_buffer(4, nslices * inter * sizeof(float));
    id<MTLBuffer> down_buf = temp_buffer(5, nslices * hidden * sizeof(float));
    id<MTLBuffer> down_args_buf = temp_buffer(6, sizeof(ornith_metal_args));
    id<MTLBuffer> weights_buf = temp_buffer(7, nslices * sizeof(float));
    id<MTLBuffer> mix_out_buf = temp_buffer(8, hidden * sizeof(float));
    id<MTLBuffer> act_args_buf = temp_buffer(9, sizeof(ornith_metal_args));
    id<MTLBuffer> mix_args_buf = temp_buffer(10, sizeof(ornith_metal_args));
    if (!slice_p || !act_p || !mix_p || !gate_payload || !down_payload || !norm_buf || !gate_up_buf ||
        !gate_args_buf || !slices_buf || !mid_buf || !down_buf || !down_args_buf || !weights_buf ||
        !mix_out_buf || !act_args_buf || !mix_args_buf) {
        free(slice32);
        set_err(err, errcap, @"metal buffer allocation failed");
        return 0;
    }

    for (size_t i = 0; i < nslices; i++) {
        if (!use_resident) {
            const unsigned char *gate_src = gate_span + gate_byte_base + ((uint64_t)slices[i] * gate_slice_elems / 256) * 34;
            const unsigned char *down_src = down_span + down_byte_base + ((uint64_t)slices[i] * down_slice_elems / 256) * 34;
            memcpy((unsigned char *)gate_payload.contents + i * gate_slice_bytes, gate_src, gate_slice_bytes);
            memcpy((unsigned char *)down_payload.contents + i * down_slice_bytes, down_src, down_slice_bytes);
        }
        slice32[i] = (uint32_t)(use_resident ? slices[i] : i);
    }

    ornith_metal_args gate_args = { 0, 0, (uint32_t)gate_up_rows, (uint32_t)hidden, 256, 0, (uint32_t)(nslices * gate_up_rows) };
    ornith_metal_args act_args = { 0, 0, (uint32_t)inter, 0, 0, 0, (uint32_t)(nslices * inter) };
    ornith_metal_args down_args = { 0, 0, (uint32_t)hidden, (uint32_t)inter, 256, (uint32_t)inter, (uint32_t)(nslices * hidden) };
    ornith_metal_args mix_args = { 0, 0, (uint32_t)hidden, (uint32_t)nslices, 0, 0, (uint32_t)hidden };
    memcpy(norm_buf.contents, norm, hidden * sizeof(float));
    memcpy(gate_args_buf.contents, &gate_args, sizeof(gate_args));
    memcpy(act_args_buf.contents, &act_args, sizeof(act_args));
    memcpy(down_args_buf.contents, &down_args, sizeof(down_args));
    memcpy(mix_args_buf.contents, &mix_args, sizeof(mix_args));
    memcpy(slices_buf.contents, slice32, nslices * sizeof(uint32_t));
    memcpy(weights_buf.contents, weights, nslices * sizeof(float));
    free(slice32);

    id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:slice_p];
    [enc setBuffer:gate_payload offset:0 atIndex:0];
    [enc setBuffer:norm_buf offset:0 atIndex:1];
    [enc setBuffer:gate_up_buf offset:0 atIndex:2];
    [enc setBuffer:gate_args_buf offset:0 atIndex:3];
    [enc setBuffer:slices_buf offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(nslices * ((gate_up_rows + 7) / 8), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:act_p];
    [enc setBuffer:gate_up_buf offset:0 atIndex:0];
    [enc setBuffer:mid_buf offset:0 atIndex:1];
    [enc setBuffer:act_args_buf offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(nslices * inter, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:slice_p];
    [enc setBuffer:down_payload offset:0 atIndex:0];
    [enc setBuffer:mid_buf offset:0 atIndex:1];
    [enc setBuffer:down_buf offset:0 atIndex:2];
    [enc setBuffer:down_args_buf offset:0 atIndex:3];
    [enc setBuffer:slices_buf offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(nslices * ((hidden + 7) / 8), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:mix_p];
    [enc setBuffer:down_buf offset:0 atIndex:0];
    [enc setBuffer:weights_buf offset:0 atIndex:1];
    [enc setBuffer:mix_out_buf offset:0 atIndex:2];
    [enc setBuffer:mix_args_buf offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        set_err(err, errcap, cb.error.localizedDescription ?: @"metal command failed");
        return 0;
    }
    memcpy(out, mix_out_buf.contents, hidden * sizeof(float));
    return 1;
}

static float sigmoidf_local(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float siluf(float x)
{
    return x * sigmoidf_local(x);
}

static int add_shared_expert_staged_metal(
    const ornith_model *m,
    const ornith_tensor_info *gate,
    const ornith_tensor_info *up,
    const ornith_tensor_info *down,
    const ornith_tensor_info *sgate,
    const float *norm,
    size_t hidden,
    size_t inter,
    float *out,
    char *err,
    size_t errcap)
{
    if (!m || !gate || !up || !down || !sgate || !norm || !out ||
        gate->quant != ORNITH_QUANT_Q4 || up->quant != ORNITH_QUANT_Q4 || down->quant != ORNITH_QUANT_Q4 ||
        gate->ndim != 2 || up->ndim != 2 || down->ndim != 2 ||
        gate->shape[0] != (int64_t)inter || gate->shape[1] != (int64_t)hidden ||
        up->shape[0] != (int64_t)inter || up->shape[1] != (int64_t)hidden ||
        down->shape[0] != (int64_t)hidden || down->shape[1] != (int64_t)inter ||
        (hidden % 256) != 0 || (inter % 256) != 0 || hidden > UINT32_MAX || inter > UINT32_MAX) {
        return -1;
    }

    uint64_t gate_base = 0, gate_span_size = 0, up_base = 0, up_span_size = 0, down_base = 0, down_span_size = 0;
    uint32_t gate_block = 0, up_block = 0, down_block = 0;
    const unsigned char *gate_span = ornith_tensor_mapped_span(m, gate, &gate_base, &gate_span_size, &gate_block);
    const unsigned char *up_span = ornith_tensor_mapped_span(m, up, &up_base, &up_span_size, &up_block);
    const unsigned char *down_span = ornith_tensor_mapped_span(m, down, &down_base, &down_span_size, &down_block);
    if (!gate_span || !up_span || !down_span || gate_block != 256 || up_block != 256 || down_block != 256) return -1;

    id<MTLComputePipelineState> q4_p = pipeline(@"ornith_q4_matvec_b256_r4_tg", err, errcap);
    id<MTLComputePipelineState> act_p = pipeline(@"ornith_pair_silu_product", err, errcap);
    id<MTLBuffer> gate_payload = temp_buffer(11, (NSUInteger)gate->nbytes);
    id<MTLBuffer> up_payload = temp_buffer(12, (NSUInteger)up->nbytes);
    id<MTLBuffer> down_payload = temp_buffer(13, (NSUInteger)down->nbytes);
    id<MTLBuffer> g_buf = temp_buffer(0, inter * sizeof(float));
    id<MTLBuffer> u_buf = temp_buffer(1, inter * sizeof(float));
    id<MTLBuffer> mid_buf = temp_buffer(2, inter * sizeof(float));
    id<MTLBuffer> tmp_buf = temp_buffer(3, hidden * sizeof(float));
    id<MTLBuffer> gate_args_buf = temp_buffer(4, sizeof(ornith_metal_args));
    id<MTLBuffer> up_args_buf = temp_buffer(5, sizeof(ornith_metal_args));
    id<MTLBuffer> act_args_buf = temp_buffer(6, sizeof(ornith_metal_args));
    id<MTLBuffer> down_args_buf = temp_buffer(7, sizeof(ornith_metal_args));
    id<MTLBuffer> norm_buf = temp_buffer(8, hidden * sizeof(float));
    if (!q4_p || !act_p || !gate_payload || !up_payload || !down_payload || !g_buf || !u_buf || !mid_buf ||
        !tmp_buf || !gate_args_buf || !up_args_buf || !act_args_buf || !down_args_buf || !norm_buf) {
        set_err(err, errcap, @"metal buffer allocation failed");
        return 0;
    }

    float s = 0.0f;
    if (!ornith_tensor_matvec(m, sgate, norm, hidden, &s)) return -1;
    memcpy(gate_payload.contents, gate_span + gate_base, (size_t)gate->nbytes);
    memcpy(up_payload.contents, up_span + up_base, (size_t)up->nbytes);
    memcpy(down_payload.contents, down_span + down_base, (size_t)down->nbytes);
    memcpy(norm_buf.contents, norm, hidden * sizeof(float));
    ornith_metal_args gate_args = { 0, 0, (uint32_t)inter, (uint32_t)hidden, 256, 0, (uint32_t)inter };
    ornith_metal_args up_args = { 0, 0, (uint32_t)inter, (uint32_t)hidden, 256, 0, (uint32_t)inter };
    ornith_metal_args act_args = { 0, 0, (uint32_t)inter, 0, 0, 0, (uint32_t)inter };
    ornith_metal_args down_args = { 0, 0, (uint32_t)hidden, (uint32_t)inter, 256, 0, (uint32_t)hidden };
    memcpy(gate_args_buf.contents, &gate_args, sizeof(gate_args));
    memcpy(up_args_buf.contents, &up_args, sizeof(up_args));
    memcpy(act_args_buf.contents, &act_args, sizeof(act_args));
    memcpy(down_args_buf.contents, &down_args, sizeof(down_args));

    id<MTLCommandBuffer> cb = [command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:q4_p];
    [enc setBuffer:gate_payload offset:0 atIndex:0];
    [enc setBuffer:norm_buf offset:0 atIndex:1];
    [enc setBuffer:g_buf offset:0 atIndex:2];
    [enc setBuffer:gate_args_buf offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((inter + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setBuffer:up_payload offset:0 atIndex:0];
    [enc setBuffer:norm_buf offset:0 atIndex:1];
    [enc setBuffer:u_buf offset:0 atIndex:2];
    [enc setBuffer:up_args_buf offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((inter + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:act_p];
    [enc setBuffer:g_buf offset:0 atIndex:0];
    [enc setBuffer:u_buf offset:0 atIndex:1];
    [enc setBuffer:mid_buf offset:0 atIndex:2];
    [enc setBuffer:act_args_buf offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(inter, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:q4_p];
    [enc setBuffer:down_payload offset:0 atIndex:0];
    [enc setBuffer:mid_buf offset:0 atIndex:1];
    [enc setBuffer:tmp_buf offset:0 atIndex:2];
    [enc setBuffer:down_args_buf offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((hidden + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        set_err(err, errcap, cb.error.localizedDescription ?: @"metal command failed");
        return 0;
    }

    float w = sigmoidf_local(s);
    const float *tmp = tmp_buf.contents;
    for (size_t i = 0; i < hidden; i++) out[i] += w * tmp[i];
    return 1;
}

static int softmax_selected(float *values, size_t n)
{
    if (!values || !n) return 0;
    float maxv = values[0];
    for (size_t i = 1; i < n; i++) if (values[i] > maxv) maxv = values[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        values[i] = expf(values[i] - maxv);
        sum += values[i];
    }
    if (sum == 0.0f) return 0;
    for (size_t i = 0; i < n; i++) values[i] /= sum;
    return 1;
}

static int add_shared_expert_metal(
    const ornith_model *m,
    int64_t layer,
    const float *norm,
    size_t hidden,
    float *out,
    char *err,
    size_t errcap)
{
    const ornith_tensor_info *gate = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.gate_proj.weight");
    const ornith_tensor_info *up = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.up_proj.weight");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert.down_proj.weight");
    const ornith_tensor_info *sgate = ornith_model_find_layer_tensor(m, layer, "mlp.shared_expert_gate.weight");
    if (!gate || !up || !down || !sgate || gate->ndim != 2 || up->ndim != 2 || down->ndim != 2 ||
        gate->shape[0] <= 0 || gate->shape[1] != (int64_t)hidden || up->shape[0] != gate->shape[0] ||
        up->shape[1] != (int64_t)hidden || down->shape[0] != (int64_t)hidden || down->shape[1] != gate->shape[0]) {
        set_err(err, errcap, @"bad shared expert shape");
        return 0;
    }
    size_t inter = (size_t)gate->shape[0];
    int staged = add_shared_expert_staged_metal(m, gate, up, down, sgate, norm, hidden, inter, out, err, errcap);
    if (staged == 1) {
        return 1;
    }
    if (staged == 0) {
        return 0;
    }
    float *buf = calloc(inter * 3 + hidden + 1, sizeof(float));
    if (!buf) {
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    float *g = buf;
    float *u = g + inter;
    float *mid = u + inter;
    float *tmp = mid + inter;
    float s = 0.0f;
    int ok = ornith_tensor_matvec(m, gate, norm, hidden, g) &&
             ornith_tensor_matvec(m, up, norm, hidden, u) &&
             ornith_tensor_matvec(m, sgate, norm, hidden, &s);
    if (ok) {
        for (size_t i = 0; i < inter; i++) mid[i] = siluf(g[i]) * u[i];
        ok = ornith_tensor_matvec(m, down, mid, inter, tmp);
    }
    if (ok) {
        float w = sigmoidf_local(s);
        for (size_t i = 0; i < hidden; i++) out[i] += w * tmp[i];
    }
    free(buf);
    return ok;
}

static int ornith_metal_moe_workspace_counts(
    const ornith_model *m,
    int64_t layer,
    const char *norm_kind,
    size_t hidden,
    size_t top_k,
    size_t *scratch_count,
    size_t *idx_count,
    char *err,
    size_t errcap)
{
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, norm_kind);
    const ornith_tensor_info *router = ornith_model_find_layer_tensor(m, layer, "mlp.gate.weight");
    const ornith_tensor_info *gate_up = ornith_model_find_layer_tensor(m, layer, "mlp.experts.gate_up_proj");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.experts.down_proj");
    if (!norm_w || !router || !gate_up || !down || router->shape[0] <= 0 || down->shape[2] <= 0 || top_k == 0 || top_k > (size_t)router->shape[0]) {
        set_err(err, errcap, @"bad layer shape");
        return 0;
    }
    size_t experts = (size_t)router->shape[0];
    size_t inter = (size_t)down->shape[2];
    size_t gate_up_rows = (size_t)gate_up->shape[1];
    if (scratch_count) *scratch_count = hidden + experts + top_k + top_k * gate_up_rows + top_k * inter + top_k * hidden;
    if (idx_count) *idx_count = top_k;
    return 1;
}

static int ornith_metal_layer_moe_smoke_profiled_workspace(
    const ornith_model *m,
    int64_t layer,
    const char *norm_kind,
    const float *x,
    size_t hidden,
    size_t top_k,
    float *out,
    float *scratch,
    size_t scratch_count,
    size_t *idx,
    size_t idx_count,
    ornith_metal_step_profile *profile,
    char *err,
    size_t errcap)
{
    double layer_start = profile ? ornith_now_seconds() : 0.0;
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, norm_kind);
    const ornith_tensor_info *router = ornith_model_find_layer_tensor(m, layer, "mlp.gate.weight");
    const ornith_tensor_info *gate_up = ornith_model_find_layer_tensor(m, layer, "mlp.experts.gate_up_proj");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.experts.down_proj");
    size_t need = 0;
    size_t idx_need = 0;
    if (!ornith_metal_moe_workspace_counts(m, layer, norm_kind, hidden, top_k, &need, &idx_need, err, errcap)) {
        return 0;
    }
    if (!scratch || scratch_count < need || !idx || idx_count < idx_need) {
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    size_t experts = (size_t)router->shape[0];
    size_t inter = (size_t)down->shape[2];
    size_t gate_up_rows = (size_t)gate_up->shape[1];
    float *norm = scratch;
    float *scores = norm + hidden;
    float *weights = scores + experts;
    float *gate_up_out = weights + top_k;
    float *mid = gate_up_out + top_k * gate_up_rows;
    float *down_out = mid + top_k * inter;

    memset(out, 0, hidden * sizeof(float));
    double phase = profile ? ornith_now_seconds() : 0.0;
    int ok = ornith_rmsnorm(m, norm_w, x, hidden, 1e-6f, norm);
    if (profile) {
        double now = ornith_now_seconds();
        profile->layer_norm_seconds += now - phase;
        phase = now;
    }
    int rmode = router_mode();
    int router_ok = 0;
    if (rmode == 0) {
        router_ok = ornith_tensor_matvec(m, router, norm, hidden, scores);
    } else if (rmode == 2) {
        router_ok = ornith_metal_tensor_matvec(m, router, 0, norm, hidden, scores, err, errcap);
    } else if (rmode == 3) {
        router_ok = ornith_metal_router_q4_b256(m, router, norm, hidden, experts, scores, err, errcap);
        if (router_ok < 0) router_ok = ornith_metal_tensor_matvec_serial_rows(m, router, norm, hidden, experts, scores, err, errcap);
    } else {
        router_ok = ornith_metal_tensor_matvec_serial_rows(m, router, norm, hidden, experts, scores, err, errcap);
    }
    ok = ok && router_ok &&
         ornith_topk(scores, experts, top_k, idx, weights) &&
         softmax_selected(weights, top_k);
    if (profile) {
        double now = ornith_now_seconds();
        profile->router_seconds += now - phase;
        phase = now;
    }
    int fused = ok ? ornith_metal_routed_mlp_b256(m, gate_up, down, idx, weights, top_k, norm, hidden, out, err, errcap) : 0;
    if (fused == 1) {
        if (profile) {
            double now = ornith_now_seconds();
            profile->routed_fused_seconds += now - phase;
            phase = now;
        }
    } else {
        ok = ok && fused != 0;
        ok = ok && ornith_metal_iq1_slice_many(m, gate_up, idx, top_k, norm, hidden, 0, gate_up_out, err, errcap);
        if (profile) {
            double now = ornith_now_seconds();
            profile->routed_gate_up_seconds += now - phase;
            phase = now;
        }
        for (size_t k = 0; ok && k < top_k; k++) {
            float *gu = gate_up_out + k * gate_up_rows;
            float *mids = mid + k * inter;
            for (size_t i = 0; i < inter; i++) mids[i] = siluf(gu[i]) * gu[i + inter];
        }
        if (profile) {
            double now = ornith_now_seconds();
            profile->routed_activation_seconds += now - phase;
            phase = now;
        }
        ok = ok && ornith_metal_iq1_slice_many(m, down, idx, top_k, mid, top_k * inter, inter, down_out, err, errcap);
        if (profile) {
            double now = ornith_now_seconds();
            profile->routed_down_seconds += now - phase;
            phase = now;
        }
        for (size_t k = 0; ok && k < top_k; k++) {
            float *d = down_out + k * hidden;
            for (size_t i = 0; i < hidden; i++) out[i] += weights[k] * d[i];
        }
        if (profile) {
            double now = ornith_now_seconds();
            profile->routed_mix_seconds += now - phase;
            phase = now;
        }
    }
    ok = ok && add_shared_expert_metal(m, layer, norm, hidden, out, err, errcap);
    if (profile) {
        double now = ornith_now_seconds();
        profile->shared_expert_seconds += now - phase;
        double layer_seconds = now - layer_start;
        profile->layer_seconds += layer_seconds;
        if (layer_seconds > profile->max_layer_seconds) {
            profile->max_layer_seconds = layer_seconds;
            profile->max_layer_index = (size_t)layer;
        }
    }
    return ok;
}

static int ornith_metal_layer_moe_smoke_profiled(
    const ornith_model *m,
    int64_t layer,
    const char *norm_kind,
    const float *x,
    size_t hidden,
    size_t top_k,
    float *out,
    ornith_metal_step_profile *profile,
    char *err,
    size_t errcap)
{
    size_t scratch_count = 0;
    size_t idx_count = 0;
    if (!ornith_metal_moe_workspace_counts(m, layer, norm_kind, hidden, top_k, &scratch_count, &idx_count, err, errcap)) {
        return 0;
    }
    float *scratch = malloc(scratch_count * sizeof(*scratch));
    size_t *idx = malloc(idx_count * sizeof(*idx));
    if (!scratch || !idx) {
        free(scratch);
        free(idx);
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    int ok = ornith_metal_layer_moe_smoke_profiled_workspace(m, layer, norm_kind, x, hidden, top_k, out, scratch, scratch_count, idx, idx_count, profile, err, errcap);
    free(idx);
    free(scratch);
    return ok;
}

int ornith_metal_layer_moe_smoke(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out, char *err, size_t errcap)
{
    return ornith_metal_layer_moe_smoke_profiled(m, layer, "input_layernorm.weight", x, hidden, top_k, out, NULL, err, errcap);
}

int ornith_metal_lm_head_topk_limited(const ornith_model *m, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values, char *err, size_t errcap)
{
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!head || !x || !indices || !values || head->ndim != 2 || hidden != (size_t)head->shape[1] ||
        rows == 0 || rows > (size_t)head->shape[0] || k == 0 || k > rows) {
        set_err(err, errcap, @"bad lm-head shape");
        return 0;
    }
    float *scores = malloc(rows * sizeof(float));
    if (!scores) {
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    int ok = ornith_metal_tensor_matvec_rows(m, head, x, hidden, rows, scores, err, errcap) &&
             ornith_topk(scores, rows, k, indices, values);
    free(scores);
    return ok;
}

typedef struct {
    char *err;
    size_t errcap;
    float *moe_scratch;
    size_t moe_scratch_count;
    size_t *moe_idx;
    size_t moe_idx_count;
    int profile_enabled;
    ornith_metal_step_profile moe_profile;
    double lm_head_seconds;
    double matvec_seconds;
    double batch_matvec_seconds;
    double gdn_seconds;
} ornith_metal_hook_ctx;

static void metal_hook_ctx_free(ornith_metal_hook_ctx *ctx)
{
    if (!ctx) return;
    free(ctx->moe_scratch);
    free(ctx->moe_idx);
    ctx->moe_scratch = NULL;
    ctx->moe_idx = NULL;
    ctx->moe_scratch_count = 0;
    ctx->moe_idx_count = 0;
}

static int metal_hook_ctx_reserve_moe(ornith_metal_hook_ctx *ctx, size_t scratch_count, size_t idx_count)
{
    if (!ctx) return 0;
    if (scratch_count > ctx->moe_scratch_count) {
        float *p = realloc(ctx->moe_scratch, scratch_count * sizeof(*p));
        if (!p) return 0;
        ctx->moe_scratch = p;
        ctx->moe_scratch_count = scratch_count;
    }
    if (idx_count > ctx->moe_idx_count) {
        size_t *p = realloc(ctx->moe_idx, idx_count * sizeof(*p));
        if (!p) return 0;
        ctx->moe_idx = p;
        ctx->moe_idx_count = idx_count;
    }
    return 1;
}

static int metal_moe_hook(const ornith_model *m, int64_t layer, const char *norm_kind, const float *x, size_t hidden, size_t top_k, float *out, void *ctx)
{
    ornith_metal_hook_ctx *h = ctx;
    if (!h) {
        return ornith_metal_layer_moe_smoke_profiled(m, layer, norm_kind, x, hidden, top_k, out, NULL, NULL, 0);
    }
    size_t scratch_count = 0;
    size_t idx_count = 0;
    if (!ornith_metal_moe_workspace_counts(m, layer, norm_kind, hidden, top_k, &scratch_count, &idx_count, h->err, h->errcap)) {
        return 0;
    }
    if (!metal_hook_ctx_reserve_moe(h, scratch_count, idx_count)) {
        set_err(h->err, h->errcap, @"out of memory");
        return 0;
    }
    ornith_metal_step_profile one = {0};
    int ok = ornith_metal_layer_moe_smoke_profiled_workspace(m, layer, norm_kind, x, hidden, top_k, out, h->moe_scratch, h->moe_scratch_count, h->moe_idx, h->moe_idx_count, h->profile_enabled ? &one : NULL, h->err, h->errcap);
    if (h->profile_enabled) {
        h->moe_profile.layer_seconds += one.layer_seconds;
        h->moe_profile.layer_norm_seconds += one.layer_norm_seconds;
        h->moe_profile.router_seconds += one.router_seconds;
        h->moe_profile.routed_fused_seconds += one.routed_fused_seconds;
        h->moe_profile.routed_gate_up_seconds += one.routed_gate_up_seconds;
        h->moe_profile.routed_activation_seconds += one.routed_activation_seconds;
        h->moe_profile.routed_down_seconds += one.routed_down_seconds;
        h->moe_profile.routed_mix_seconds += one.routed_mix_seconds;
        h->moe_profile.shared_expert_seconds += one.shared_expert_seconds;
        if (one.max_layer_seconds > h->moe_profile.max_layer_seconds) {
            h->moe_profile.max_layer_seconds = one.max_layer_seconds;
            h->moe_profile.max_layer_index = one.max_layer_index;
        }
    }
    return ok;
}

static int metal_lm_head_hook(const ornith_model *m, const float *x, size_t hidden, size_t rows, size_t k, size_t *indices, float *values, void *ctx)
{
    ornith_metal_hook_ctx *h = ctx;
    double start = h && h->profile_enabled ? ornith_now_seconds() : 0.0;
    int ok = ornith_metal_lm_head_topk_limited(m, x, hidden, rows, k, indices, values, h ? h->err : NULL, h ? h->errcap : 0);
    if (h && h->profile_enabled) h->lm_head_seconds += ornith_now_seconds() - start;
    return ok;
}

static int metal_matvec_hook(const ornith_model *m, const ornith_tensor_info *t, const float *x, size_t x_count, float *out, void *ctx)
{
    ornith_metal_hook_ctx *h = ctx;
    double start = h && h->profile_enabled ? ornith_now_seconds() : 0.0;
    int ok = ornith_metal_tensor_matvec(m, t, 0, x, x_count, out, h ? h->err : NULL, h ? h->errcap : 0);
    if (h && h->profile_enabled) h->matvec_seconds += ornith_now_seconds() - start;
    return ok;
}

static int metal_batch_matvec_hook(const ornith_model *m, const ornith_tensor_info * const *tensors, size_t count, const float *x, size_t x_count, float **outs, void *ctx)
{
    ornith_metal_hook_ctx *h = ctx;
    double start = h && h->profile_enabled ? ornith_now_seconds() : 0.0;
    int ok = ornith_metal_tensor_matvec_batch(m, tensors, count, x, x_count, outs, h ? h->err : NULL, h ? h->errcap : 0);
    if (h && h->profile_enabled) h->batch_matvec_seconds += ornith_now_seconds() - start;
    return ok;
}

static int metal_gdn_hook(const float *qkv, const float *z, const float *a, const float *b, const float *alog, const float *dt, const float *norm_w, float *ssm, size_t value_heads, size_t head_v, size_t key_heads, size_t head_k, float *gated, const ornith_model *model, const ornith_tensor_info *out_w, float *out, void *ctx)
{
    ornith_metal_hook_ctx *h = ctx;
    double start = h && h->profile_enabled ? ornith_now_seconds() : 0.0;
    int fused = ornith_metal_gdn_recurrent_out_proj(qkv, z, a, b, alog, dt, norm_w, ssm, value_heads, head_v, key_heads, head_k, model, out_w, out, h ? h->err : NULL, h ? h->errcap : 0);
    if (h && h->profile_enabled && fused >= 0) h->gdn_seconds += ornith_now_seconds() - start;
    if (fused >= 0) return fused;
    int ok = ornith_metal_gdn_recurrent_step(qkv, z, a, b, alog, dt, norm_w, ssm, value_heads, head_v, key_heads, head_k, gated, h ? h->err : NULL, h ? h->errcap : 0);
    if (h && h->profile_enabled) h->gdn_seconds += ornith_now_seconds() - start;
    return ok;
}

static int metal_profile_enabled(void)
{
    const char *env = getenv("ORNITH_METAL_PROFILE");
    return env && env[0] && strcmp(env, "0") != 0;
}

static void metal_hook_ctx_print_profile(const ornith_metal_hook_ctx *ctx, const char *scope)
{
    if (!ctx || !ctx->profile_enabled) return;
    fprintf(stderr,
            "ornith_metal_profile scope=%s moe_layers=%.6f moe_norm=%.6f router=%.6f routed_fused=%.6f shared=%.6f lm_head=%.6f matvec=%.6f batch_matvec=%.6f gdn=%.6f max_layer=%zu max_layer_seconds=%.6f\n",
            scope ? scope : "generation",
            ctx->moe_profile.layer_seconds,
            ctx->moe_profile.layer_norm_seconds,
            ctx->moe_profile.router_seconds,
            ctx->moe_profile.routed_fused_seconds,
            ctx->moe_profile.shared_expert_seconds,
            ctx->lm_head_seconds,
            ctx->matvec_seconds,
            ctx->batch_matvec_seconds,
            ctx->gdn_seconds,
            ctx->moe_profile.max_layer_index,
            ctx->moe_profile.max_layer_seconds);
}

int ornith_metal_generate_greedy_limited(const ornith_model *m, const uint64_t *prompt_ids, size_t prompt_count, size_t max_new, size_t layer_count, size_t expert_top_k, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, char *err, size_t errcap)
{
    ornith_metal_hook_ctx ctx = { err, errcap };
    ctx.profile_enabled = metal_profile_enabled();
    const char *gdn_env = getenv("ORNITH_METAL_GDN");
    const char *matvec_env = getenv("ORNITH_METAL_ATTN_MATVEC");
    const char *batch_env = getenv("ORNITH_METAL_BATCH_MATVEC");
    ornith_tensor_matvec_fn matvec_hook = (matvec_env && strcmp(matvec_env, "0") == 0) ? NULL : metal_matvec_hook;
    ornith_tensor_matvec_batch_fn batch_hook = (batch_env && strcmp(batch_env, "0") == 0) ? NULL : metal_batch_matvec_hook;
    ornith_gdn_recurrent_fn gdn_hook = (gdn_env && strcmp(gdn_env, "0") == 0) ? NULL : metal_gdn_hook;
    int ok = ornith_generate_greedy_limited_with_decode_hooks(m, prompt_ids, prompt_count, max_new, layer_count, expert_top_k, vocab_limit, out_ids, out_scores, out_count, metal_moe_hook, metal_lm_head_hook, matvec_hook, batch_hook, gdn_hook, &ctx);
    metal_hook_ctx_print_profile(&ctx, "generation");
    metal_hook_ctx_free(&ctx);
    return ok;
}

int ornith_metal_session_generate_greedy_limited(ornith_session *session, const uint64_t *prompt_suffix_ids, size_t prompt_suffix_count, size_t max_new, size_t vocab_limit, uint64_t *out_ids, float *out_scores, size_t *out_count, char *err, size_t errcap)
{
    ornith_metal_hook_ctx ctx = { err, errcap };
    ctx.profile_enabled = metal_profile_enabled();
    const char *gdn_env = getenv("ORNITH_METAL_GDN");
    const char *matvec_env = getenv("ORNITH_METAL_ATTN_MATVEC");
    const char *batch_env = getenv("ORNITH_METAL_BATCH_MATVEC");
    ornith_tensor_matvec_fn matvec_hook = (matvec_env && strcmp(matvec_env, "0") == 0) ? NULL : metal_matvec_hook;
    ornith_tensor_matvec_batch_fn batch_hook = (batch_env && strcmp(batch_env, "0") == 0) ? NULL : metal_batch_matvec_hook;
    ornith_gdn_recurrent_fn gdn_hook = (gdn_env && strcmp(gdn_env, "0") == 0) ? NULL : metal_gdn_hook;
    int ok = ornith_session_generate_greedy_limited_with_decode_hooks(session, prompt_suffix_ids, prompt_suffix_count, max_new, vocab_limit, out_ids, out_scores, out_count, metal_moe_hook, metal_lm_head_hook, matvec_hook, batch_hook, gdn_hook, &ctx);
    metal_hook_ctx_print_profile(&ctx, "session");
    metal_hook_ctx_free(&ctx);
    return ok;
}

int ornith_metal_step_smoke_limited(const ornith_model *m, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values, char *err, size_t errcap)
{
    return ornith_metal_step_smoke_profiled_limited(m, token_id, layer_count, expert_top_k, out_top_k, vocab_limit, indices, values, NULL, err, errcap);
}

int ornith_metal_step_smoke_profiled_limited(
    const ornith_model *m,
    uint64_t token_id,
    size_t layer_count,
    size_t expert_top_k,
    size_t out_top_k,
    size_t vocab_limit,
    size_t *indices,
    float *values,
    ornith_metal_step_profile *profile,
    char *err,
    size_t errcap)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    const ornith_tensor_info *final_norm = ornith_model_find_tensor(m, "model.language_model.norm.weight");
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!embed || !final_norm || !head || embed->ndim != 2 || layer_count > ornith_model_layer_count(m)) {
        set_err(err, errcap, @"bad step shape");
        return 0;
    }
    size_t hidden = (size_t)embed->shape[1];
    size_t rows = vocab_limit ? vocab_limit : (size_t)head->shape[0];
    float *x = calloc(hidden * 3, sizeof(float));
    if (!x) {
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    float *delta = x + hidden;
    float *norm = delta + hidden;
    if (profile) memset(profile, 0, sizeof(*profile));
    double phase = profile ? ornith_now_seconds() : 0.0;
    int ok = ornith_embed_token(m, token_id, x, hidden);
    if (profile) {
        double now = ornith_now_seconds();
        profile->embed_seconds += now - phase;
        phase = now;
    }
    for (size_t layer = 0; ok && layer < layer_count; layer++) {
        ok = ornith_metal_layer_moe_smoke_profiled(m, (int64_t)layer, "input_layernorm.weight", x, hidden, expert_top_k, delta, profile, err, errcap);
        for (size_t i = 0; ok && i < hidden; i++) x[i] += delta[i];
    }
    phase = profile ? ornith_now_seconds() : 0.0;
    ok = ok && ornith_rmsnorm(m, final_norm, x, hidden, 1e-6f, norm);
    if (profile) {
        double now = ornith_now_seconds();
        profile->final_norm_seconds += now - phase;
        phase = now;
    }
    ok = ok && ornith_metal_lm_head_topk_limited(m, norm, hidden, rows, out_top_k, indices, values, err, errcap);
    if (profile) {
        double now = ornith_now_seconds();
        profile->lm_head_seconds += now - phase;
    }
    free(x);
    return ok;
}

int ornith_metal_step_smoke_hybrid_limited(
    const ornith_model *m,
    uint64_t token_id,
    size_t layer_count,
    size_t expert_top_k,
    size_t out_top_k,
    size_t vocab_limit,
    size_t *indices,
    float *values,
    ornith_metal_step_profile *profile,
    char *err,
    size_t errcap)
{
    const ornith_tensor_info *embed = ornith_model_find_tensor(m, "model.language_model.embed_tokens.weight");
    const ornith_tensor_info *final_norm = ornith_model_find_tensor(m, "model.language_model.norm.weight");
    const ornith_tensor_info *head = ornith_model_find_tensor(m, "lm_head.weight");
    if (!embed || !final_norm || !head || embed->ndim != 2 || layer_count > ornith_model_layer_count(m)) {
        set_err(err, errcap, @"bad hybrid step shape");
        return 0;
    }
    size_t hidden = (size_t)embed->shape[1];
    size_t rows = vocab_limit ? vocab_limit : (size_t)head->shape[0];
    float *x = calloc(hidden * 3, sizeof(float));
    if (!x) {
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    float *delta = x + hidden;
    float *norm = delta + hidden;
    if (profile) memset(profile, 0, sizeof(*profile));
    double phase = profile ? ornith_now_seconds() : 0.0;
    int ok = ornith_embed_token(m, token_id, x, hidden);
    if (profile) {
        double now = ornith_now_seconds();
        profile->embed_seconds += now - phase;
    }
    for (size_t layer = 0; ok && layer < layer_count; layer++) {
        double layer_start = profile ? ornith_now_seconds() : 0.0;
        ok = ornith_layer_moe_smoke(m, (int64_t)layer, x, hidden, expert_top_k, delta);
        for (size_t i = 0; ok && i < hidden; i++) x[i] += delta[i];
        if (profile) {
            double layer_seconds = ornith_now_seconds() - layer_start;
            profile->layer_seconds += layer_seconds;
            if (layer_seconds > profile->max_layer_seconds) {
                profile->max_layer_seconds = layer_seconds;
                profile->max_layer_index = layer;
            }
        }
    }
    phase = profile ? ornith_now_seconds() : 0.0;
    ok = ok && ornith_rmsnorm(m, final_norm, x, hidden, 1e-6f, norm);
    if (profile) {
        double now = ornith_now_seconds();
        profile->final_norm_seconds += now - phase;
        phase = now;
    }
    ok = ok && ornith_metal_lm_head_topk_limited(m, norm, hidden, rows, out_top_k, indices, values, err, errcap);
    if (profile) {
        double now = ornith_now_seconds();
        profile->lm_head_seconds += now - phase;
    }
    free(x);
    return ok;
}

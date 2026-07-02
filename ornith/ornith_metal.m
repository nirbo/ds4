#import "ornith_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString *const ORNITH_METAL_SRC =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Args { ulong byte_base; ulong elem_offset; uint rows; uint cols; uint block; };\n"
"static inline float bf16_at(device const uchar *p, ulong i) {\n"
"    uint lo = p[i]; uint hi = p[i + 1]; return as_type<float>((hi << 24) | (lo << 16));\n"
"}\n"
"kernel void ornith_bf16_matvec(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; ulong base = a.byte_base + (a.elem_offset + (ulong)row * a.cols) * 2;\n"
"    for (uint c = 0; c < a.cols; c++) acc += bf16_at(payload, base + (ulong)c * 2) * x[c]; out[row] = acc;\n"
"}\n"
"kernel void ornith_q4_matvec(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; ulong row_base = a.elem_offset + (ulong)row * a.cols;\n"
"    for (uint c = 0; c < a.cols;) { ulong i = row_base + c; uint inb = (uint)(i % a.block); uint take = min(a.block - inb, a.cols - c); ulong bb = a.byte_base + (i / a.block) * (2 + (a.block + 1) / 2); float scale = bf16_at(payload, bb);\n"
"        for (uint j = 0; j < take; j++, c++) { uint qidx = inb + j; uchar packed = payload[bb + 2 + qidx / 2]; int q = (qidx & 1) ? (packed >> 4) : (packed & 15); if (q >= 8) q -= 16; acc += scale * (float)q * x[c]; }} out[row] = acc;\n"
"}\n"
"kernel void ornith_iq1_matvec(device const uchar *payload [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]], constant Args &a [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"    if (row >= a.rows) return; float acc = 0.0f; ulong row_base = a.elem_offset + (ulong)row * a.cols;\n"
"    for (uint c = 0; c < a.cols;) { ulong i = row_base + c; uint inb = (uint)(i % a.block); uint take = min(a.block - inb, a.cols - c); ulong bb = a.byte_base + (i / a.block) * (2 + (a.block + 7) / 8); float scale = bf16_at(payload, bb);\n"
"        for (uint j = 0; j < take; j++, c++) { uint b = inb + j; uchar bits = payload[bb + 2 + b / 8]; acc += ((bits & (1 << (b % 8))) ? scale : -scale) * x[c]; }} out[row] = acc;\n"
"}\n";

typedef struct {
    uint64_t byte_base;
    uint64_t elem_offset;
    uint32_t rows;
    uint32_t cols;
    uint32_t block;
} ornith_metal_args;

static void set_err(char *err, size_t errcap, NSString *msg)
{
    if (err && errcap) snprintf(err, errcap, "%s", msg.UTF8String);
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

        NSString *kernel = nil;
        if (tensor->quant == ORNITH_QUANT_BF16) kernel = @"ornith_bf16_matvec";
        else if (tensor->quant == ORNITH_QUANT_Q4 && tensor->ndim == 2) kernel = @"ornith_q4_matvec";
        else if (tensor->quant == ORNITH_QUANT_IQ1) kernel = @"ornith_iq1_matvec";
        else {
            set_err(err, errcap, @"unsupported quant mode");
            return 0;
        }

        id<MTLComputePipelineState> p = pipeline(kernel, err, errcap);
        if (!p) return 0;
        id<MTLCommandQueue> q = command_queue();
        id<MTLBuffer> payload_buf = span_buffer(span, span_size);
        id<MTLBuffer> x_buf = [device() newBufferWithBytes:x length:cols * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [device() newBufferWithLength:rows * sizeof(float) options:MTLResourceStorageModeShared];
        ornith_metal_args args = { byte_base, elem_offset, (uint32_t)rows, (uint32_t)cols, block };
        id<MTLBuffer> args_buf = [device() newBufferWithBytes:&args length:sizeof(args) options:MTLResourceStorageModeShared];
        if (!q || !payload_buf || !x_buf || !out_buf || !args_buf) {
            set_err(err, errcap, @"metal buffer allocation failed");
            return 0;
        }
        id<MTLCommandBuffer> cb = [q commandBuffer];
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
            set_err(err, errcap, cb.error.localizedDescription ?: @"metal command failed");
            return 0;
        }
        memcpy(out, out_buf.contents, rows * sizeof(float));
        return 1;
    }
}

static float sigmoidf_local(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float siluf(float x)
{
    return x * sigmoidf_local(x);
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
    int ok = ornith_metal_tensor_matvec(m, gate, 0, norm, hidden, g, err, errcap) &&
             ornith_metal_tensor_matvec(m, up, 0, norm, hidden, u, err, errcap) &&
             ornith_metal_tensor_matvec(m, sgate, 0, norm, hidden, &s, err, errcap);
    if (ok) {
        for (size_t i = 0; i < inter; i++) mid[i] = siluf(g[i]) * u[i];
        ok = ornith_metal_tensor_matvec(m, down, 0, mid, inter, tmp, err, errcap);
    }
    if (ok) {
        float w = sigmoidf_local(s);
        for (size_t i = 0; i < hidden; i++) out[i] += w * tmp[i];
    }
    free(buf);
    return ok;
}

int ornith_metal_layer_moe_smoke(const ornith_model *m, int64_t layer, const float *x, size_t hidden, size_t top_k, float *out, char *err, size_t errcap)
{
    const ornith_tensor_info *norm_w = ornith_model_find_layer_tensor(m, layer, "input_layernorm.weight");
    const ornith_tensor_info *router = ornith_model_find_layer_tensor(m, layer, "mlp.gate.weight");
    const ornith_tensor_info *gate_up = ornith_model_find_layer_tensor(m, layer, "mlp.experts.gate_up_proj");
    const ornith_tensor_info *down = ornith_model_find_layer_tensor(m, layer, "mlp.experts.down_proj");
    if (!norm_w || !router || !gate_up || !down || router->shape[0] <= 0 || down->shape[2] <= 0 || top_k == 0 || top_k > (size_t)router->shape[0]) {
        set_err(err, errcap, @"bad layer shape");
        return 0;
    }
    size_t experts = (size_t)router->shape[0];
    size_t inter = (size_t)down->shape[2];
    float *scratch = calloc(hidden * 3 + experts + top_k + (size_t)gate_up->shape[1] + inter, sizeof(float));
    size_t *idx = calloc(top_k, sizeof(size_t));
    if (!scratch || !idx) {
        free(scratch);
        free(idx);
        set_err(err, errcap, @"out of memory");
        return 0;
    }
    float *norm = scratch;
    float *scores = norm + hidden;
    float *weights = scores + experts;
    float *gate_up_out = weights + top_k;
    float *mid = gate_up_out + gate_up->shape[1];
    float *tmp = mid + inter;

    memset(out, 0, hidden * sizeof(float));
    int ok = ornith_rmsnorm(m, norm_w, x, hidden, 1e-6f, norm) &&
             ornith_metal_tensor_matvec(m, router, 0, norm, hidden, scores, err, errcap) &&
             ornith_topk(scores, experts, top_k, idx, weights) &&
             softmax_selected(weights, top_k);
    for (size_t k = 0; ok && k < top_k; k++) {
        ok = ornith_metal_tensor_matvec(m, gate_up, idx[k], norm, hidden, gate_up_out, err, errcap);
        if (!ok) break;
        for (size_t i = 0; i < inter; i++) mid[i] = siluf(gate_up_out[i]) * gate_up_out[i + inter];
        ok = ornith_metal_tensor_matvec(m, down, idx[k], mid, inter, tmp, err, errcap);
        for (size_t i = 0; ok && i < hidden; i++) out[i] += weights[k] * tmp[i];
    }
    ok = ok && add_shared_expert_metal(m, layer, norm, hidden, out, err, errcap);
    free(idx);
    free(scratch);
    return ok;
}

int ornith_metal_step_smoke_limited(const ornith_model *m, uint64_t token_id, size_t layer_count, size_t expert_top_k, size_t out_top_k, size_t vocab_limit, size_t *indices, float *values, char *err, size_t errcap)
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
    int ok = ornith_embed_token(m, token_id, x, hidden);
    for (size_t layer = 0; ok && layer < layer_count; layer++) {
        ok = ornith_metal_layer_moe_smoke(m, (int64_t)layer, x, hidden, expert_top_k, delta, err, errcap);
        for (size_t i = 0; ok && i < hidden; i++) x[i] += delta[i];
    }
    ok = ok &&
         ornith_rmsnorm(m, final_norm, x, hidden, 1e-6f, norm) &&
         ornith_lm_head_topk_limited(m, norm, hidden, rows, out_top_k, indices, values);
    free(x);
    return ok;
}

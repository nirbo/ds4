#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <string.h>

#include "nemotron_metal.h"

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_nvfp4_matvec;

static int set_error(char *error, size_t cap, NSString *message) {
    if (error != NULL && cap > 0) {
        snprintf(error, cap, "%s", message.UTF8String ?: "unknown Metal error");
    }
    return -1;
}

static int initialize_metal(char *error, size_t error_cap) {
    if (g_nvfp4_matvec != nil) return 0;
    g_device = MTLCreateSystemDefaultDevice();
    if (g_device == nil) return set_error(error, error_cap, @"Metal device unavailable");
    g_queue = [g_device newCommandQueue];
    if (g_queue == nil) return set_error(error, error_cap, @"Metal command queue unavailable");

    const char *source_env = getenv("NEMOTRON_METAL_SOURCE");
    NSString *source_path = source_env != NULL
        ? [NSString stringWithUTF8String:source_env]
        : @"metal/nemotron_nvfp4.metal";
    NSError *read_error = nil;
    NSString *source = [NSString stringWithContentsOfFile:source_path
                                                  encoding:NSUTF8StringEncoding
                                                     error:&read_error];
    if (source == nil) {
        return set_error(error, error_cap,
            [NSString stringWithFormat:@"cannot read %@: %@", source_path, read_error]);
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    NSError *compile_error = nil;
    id<MTLLibrary> library = [g_device newLibraryWithSource:source
                                                    options:options
                                                      error:&compile_error];
    if (library == nil) return set_error(error, error_cap, compile_error.description);
    id<MTLFunction> function = [library newFunctionWithName:@"nemotron_nvfp4_matvec_f32"];
    if (function == nil) return set_error(error, error_cap, @"NVFP4 Metal function unavailable");
    NSError *pipeline_error = nil;
    g_nvfp4_matvec = [g_device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (g_nvfp4_matvec == nil) return set_error(error, error_cap, pipeline_error.description);
    if (g_nvfp4_matvec.threadExecutionWidth != 32) {
        return set_error(error, error_cap, @"NVFP4 kernel requires Metal SIMD width 32");
    }
    return 0;
}

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
    size_t error_cap)
{
    @autoreleasepool {
        if (packed_weight == NULL || block_scale == NULL || input == NULL ||
            output == NULL || rows == 0 || columns == 0 || columns % 16 != 0 ||
            repeats == 0 || !(global_scale > 0.0f)) {
            return set_error(error, error_cap, @"invalid NVFP4 matvec arguments");
        }
        if (initialize_metal(error, error_cap) != 0) return -1;

        size_t weight_bytes = (size_t) rows * columns / 2;
        size_t scale_bytes = (size_t) rows * columns / 16;
        size_t input_bytes = (size_t) columns * sizeof(float);
        size_t output_bytes = (size_t) rows * sizeof(float);
        id<MTLBuffer> weight = [g_device newBufferWithBytes:packed_weight
                                                    length:weight_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> scale = [g_device newBufferWithBytes:block_scale
                                                   length:scale_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> global = [g_device newBufferWithBytes:&global_scale
                                                    length:sizeof(global_scale)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> x = [g_device newBufferWithBytes:input
                                               length:input_bytes
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> y = [g_device newBufferWithLength:output_bytes
                                               options:MTLResourceStorageModeShared];
        if (weight == nil || scale == nil || global == nil || x == nil || y == nil) {
            return set_error(error, error_cap, @"Metal buffer allocation failed");
        }

        uint32_t dimensions[2] = {rows, columns};
        id<MTLBuffer> dims = [g_device newBufferWithBytes:dimensions
                                                   length:sizeof(dimensions)
                                                  options:MTLResourceStorageModeShared];
        if (dims == nil) return set_error(error, error_cap, @"Metal dimension buffer allocation failed");

        id<MTLCommandBuffer> command = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:g_nvfp4_matvec];
        [encoder setBuffer:weight offset:0 atIndex:0];
        [encoder setBuffer:scale offset:0 atIndex:1];
        [encoder setBuffer:global offset:0 atIndex:2];
        [encoder setBuffer:x offset:0 atIndex:3];
        [encoder setBuffer:y offset:0 atIndex:4];
        [encoder setBuffer:dims offset:0 atIndex:5];
        MTLSize groups = MTLSizeMake((rows + 7) / 8, 1, 1);
        MTLSize threads = MTLSizeMake(256, 1, 1);
        CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
        for (uint32_t iteration = 0; iteration < repeats; iteration++) {
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();
        if (command.status == MTLCommandBufferStatusError) {
            return set_error(error, error_cap, command.error.description);
        }
        memcpy(output, y.contents, output_bytes);
        if (elapsed_ms != NULL) *elapsed_ms = (end - start) * 1000.0;
        return 0;
    }
}

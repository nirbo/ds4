"""Standalone Metal spike for a fused Nemotron selected-expert NVFP4 MLP."""

from std.gpu import barrier, block_idx, thread_idx
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.memory import bitcast, stack_allocation
from std.testing import assert_true


comptime TOP_K = 22
comptime INPUT_DIMS = 1024
comptime HIDDEN_DIMS = 2688
comptime THREADS = 128
comptime REPEATS = 200


@always_inline
def decode_e2m1(nibble: UInt8) -> Float32:
    var code = nibble & 15
    var magnitude: Float32
    if (code & 7) == 0:
        magnitude = 0.0
    elif (code & 7) == 1:
        magnitude = 0.5
    elif (code & 7) == 2:
        magnitude = 1.0
    elif (code & 7) == 3:
        magnitude = 1.5
    elif (code & 7) == 4:
        magnitude = 2.0
    elif (code & 7) == 5:
        magnitude = 3.0
    elif (code & 7) == 6:
        magnitude = 4.0
    else:
        magnitude = 6.0
    return -magnitude if (code & 8) != 0 else magnitude


@always_inline
def decode_e4m3(bits: UInt8) -> Float32:
    return bitcast[DType.float8_e4m3fn](bits).cast[DType.float32]()


@always_inline
def decode_e2m1x16(packed: SIMD[DType.uint8, 8]) -> SIMD[DType.float32, 16]:
    var codes = (packed & 15).interleave(packed >> 4)
    var half_bits = (codes & 7).cast[DType.uint16]() << 9
    var magnitude = (
        bitcast[DType.float16](half_bits).cast[DType.float32]() * 16384.0
    )
    var sign = codes.lt(8).select(Float32(1.0), Float32(-1.0))
    return magnitude * sign


def selected_expert_mlp[
    input_dims: Int,
    hidden_dims: Int,
    threads: Int,
](
    up_weight: UnsafePointer[UInt8, MutAnyOrigin],
    up_scales: UnsafePointer[UInt8, MutAnyOrigin],
    up_global: UnsafePointer[Float32, MutAnyOrigin],
    down_weight: UnsafePointer[UInt8, MutAnyOrigin],
    down_scales: UnsafePointer[UInt8, MutAnyOrigin],
    down_global: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    expert_output: UnsafePointer[Float32, MutAnyOrigin],
):
    var hidden = stack_allocation[
        hidden_dims,
        Float32,
        address_space=AddressSpace.SHARED,
    ]()
    var expert = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    comptime up_packed_columns = input_dims // 2
    comptime up_blocks = input_dims // 16
    comptime down_packed_columns = hidden_dims // 2
    comptime down_blocks = hidden_dims // 16

    for row in range(tid, hidden_dims, threads):
        var total: Float32 = 0.0
        var packed_base = (expert * hidden_dims + row) * up_packed_columns
        var scale_base = (expert * hidden_dims + row) * up_blocks
        for block in range(up_blocks):
            var scale = (
                decode_e4m3(up_scales[scale_base + block]) * up_global[expert]
            )
            var input_base = block * 16
            var weight_base = packed_base + block * 8
            for pair in range(8):
                var packed = up_weight[weight_base + pair]
                var column = input_base + pair * 2
                total += decode_e2m1(packed & 15) * scale * input[column]
                total += decode_e2m1(packed >> 4) * scale * input[column + 1]
        hidden[row] = total * total if total > 0.0 else 0.0

    barrier()

    for row in range(tid, input_dims, threads):
        var total: Float32 = 0.0
        var packed_base = (expert * input_dims + row) * down_packed_columns
        var scale_base = (expert * input_dims + row) * down_blocks
        for block in range(down_blocks):
            var scale = (
                decode_e4m3(down_scales[scale_base + block])
                * down_global[expert]
            )
            var hidden_base = block * 16
            var weight_base = packed_base + block * 8
            for pair in range(8):
                var packed = down_weight[weight_base + pair]
                var column = hidden_base + pair * 2
                total += decode_e2m1(packed & 15) * scale * hidden[column]
                total += decode_e2m1(packed >> 4) * scale * hidden[column + 1]
        expert_output[expert * input_dims + row] = total * scores[expert]


def reduce_experts(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    var row = Int(thread_idx.x)
    if row < INPUT_DIMS:
        var total: Float32 = 0.0
        for expert in range(TOP_K):
            total += input[expert * INPUT_DIMS + row]
        output[row] = total


def selected_up_rows[
    input_dims: Int,
    hidden_dims: Int,
](
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[UInt8, MutAnyOrigin],
    global_scales: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime warps_per_block = THREADS // Int(WARP_SIZE)
    comptime packed_columns = input_dims // 2
    comptime blocks_per_row = input_dims // 16
    var tid = Int(thread_idx.x)
    var lane = tid % Int(WARP_SIZE)
    var warp_in_block = tid // Int(WARP_SIZE)
    var work = Int(block_idx.x) * warps_per_block + warp_in_block
    if work >= TOP_K * hidden_dims:
        return
    var expert = work // hidden_dims
    var row = work % hidden_dims
    var packed_base = work * packed_columns
    var scale_base = work * blocks_per_row
    var total: Float32 = 0.0
    for block in range(lane, blocks_per_row, Int(WARP_SIZE)):
        var scale = (
            decode_e4m3(scales[scale_base + block]) * global_scales[expert]
        )
        var input_base = block * 16
        var weight_base = packed_base + block * 8
        var values = decode_e2m1x16(weight.load[width=8](weight_base))
        total += (
            values * input.load[width=16](input_base)
        ).reduce_add() * scale
    total = warp.sum(total)
    if lane == 0:
        hidden[work] = total * total if total > 0.0 else 0.0


def selected_down_rows[
    input_dims: Int,
    hidden_dims: Int,
](
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[UInt8, MutAnyOrigin],
    global_scales: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    expert_output: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime warps_per_block = THREADS // Int(WARP_SIZE)
    comptime packed_columns = hidden_dims // 2
    comptime blocks_per_row = hidden_dims // 16
    var tid = Int(thread_idx.x)
    var lane = tid % Int(WARP_SIZE)
    var warp_in_block = tid // Int(WARP_SIZE)
    var work = Int(block_idx.x) * warps_per_block + warp_in_block
    if work >= TOP_K * input_dims:
        return
    var expert = work // input_dims
    var row = work % input_dims
    var packed_base = work * packed_columns
    var scale_base = work * blocks_per_row
    var hidden_base = expert * hidden_dims
    var total: Float32 = 0.0
    for block in range(lane, blocks_per_row, Int(WARP_SIZE)):
        var scale = (
            decode_e4m3(scales[scale_base + block]) * global_scales[expert]
        )
        var column_base = block * 16
        var weight_base = packed_base + block * 8
        var values = decode_e2m1x16(weight.load[width=8](weight_base))
        total += (
            values * hidden.load[width=16](hidden_base + column_base)
        ).reduce_add() * scale
    total = warp.sum(total)
    if lane == 0:
        expert_output[work] = total * scores[expert]


def selected_down_reduce_rows[
    input_dims: Int,
    hidden_dims: Int,
](
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[UInt8, MutAnyOrigin],
    global_scales: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime warps_per_block = THREADS // Int(WARP_SIZE)
    comptime packed_columns = hidden_dims // 2
    comptime blocks_per_row = hidden_dims // 16
    var tid = Int(thread_idx.x)
    var lane = tid % Int(WARP_SIZE)
    var warp_in_block = tid // Int(WARP_SIZE)
    var row = Int(block_idx.x) * warps_per_block + warp_in_block
    if row >= input_dims:
        return
    var mixed: Float32 = 0.0
    for expert in range(TOP_K):
        var work = expert * input_dims + row
        var packed_base = work * packed_columns
        var scale_base = work * blocks_per_row
        var hidden_base = expert * hidden_dims
        var total: Float32 = 0.0
        for block in range(lane, blocks_per_row, Int(WARP_SIZE)):
            var scale = (
                decode_e4m3(scales[scale_base + block]) * global_scales[expert]
            )
            var column_base = block * 16
            var weight_base = packed_base + block * 8
            var values = decode_e2m1x16(weight.load[width=8](weight_base))
            total += (
                values * hidden.load[width=16](hidden_base + column_base)
            ).reduce_add() * scale
        mixed += warp.sum(total) * scores[expert]
    if lane == 0:
        output[row] = mixed


def selected_up_four_rows[
    input_dims: Int,
    hidden_dims: Int,
](
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[UInt8, MutAnyOrigin],
    global_scales: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime warps_per_block = THREADS // Int(WARP_SIZE)
    comptime rows_per_warp = 4
    comptime row_groups = hidden_dims // rows_per_warp
    comptime packed_columns = input_dims // 2
    comptime blocks_per_row = input_dims // 16
    comptime columns_per_warp_step = Int(WARP_SIZE) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % Int(WARP_SIZE)
    var warp_in_block = tid // Int(WARP_SIZE)
    var work = Int(block_idx.x) * warps_per_block + warp_in_block
    if work >= TOP_K * row_groups:
        return
    var expert = work // row_groups
    var row = (work % row_groups) * rows_per_warp
    var result0: Float32 = 0.0
    var result1: Float32 = 0.0
    var result2: Float32 = 0.0
    var result3: Float32 = 0.0
    for step in range(0, input_dims, columns_per_warp_step):
        var column = step + lane * 16
        var input_values = input.load[width=16](column)
        var block = column // 16
        var packed_column = column // 2
        var base0 = (
            expert * hidden_dims + row
        ) * packed_columns + packed_column
        var scale0 = (expert * hidden_dims + row) * blocks_per_row + block
        var values0 = decode_e2m1x16(weight.load[width=8](base0))
        var values1 = decode_e2m1x16(
            weight.load[width=8](base0 + packed_columns)
        )
        var values2 = decode_e2m1x16(
            weight.load[width=8](base0 + 2 * packed_columns)
        )
        var values3 = decode_e2m1x16(
            weight.load[width=8](base0 + 3 * packed_columns)
        )
        result0 += (values0 * input_values).reduce_add() * decode_e4m3(
            scales[scale0]
        )
        result1 += (values1 * input_values).reduce_add() * decode_e4m3(
            scales[scale0 + blocks_per_row]
        )
        result2 += (values2 * input_values).reduce_add() * decode_e4m3(
            scales[scale0 + 2 * blocks_per_row]
        )
        result3 += (values3 * input_values).reduce_add() * decode_e4m3(
            scales[scale0 + 3 * blocks_per_row]
        )
    result0 = warp.sum(result0) * global_scales[expert]
    result1 = warp.sum(result1) * global_scales[expert]
    result2 = warp.sum(result2) * global_scales[expert]
    result3 = warp.sum(result3) * global_scales[expert]
    if lane == 0:
        var output_base = expert * hidden_dims + row
        hidden[output_base] = result0 * result0 if result0 > 0.0 else 0.0
        hidden[output_base + 1] = result1 * result1 if result1 > 0.0 else 0.0
        hidden[output_base + 2] = result2 * result2 if result2 > 0.0 else 0.0
        hidden[output_base + 3] = result3 * result3 if result3 > 0.0 else 0.0


def selected_down_four_rows[
    input_dims: Int,
    hidden_dims: Int,
](
    weight: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[UInt8, MutAnyOrigin],
    global_scales: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    expert_output: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime warps_per_block = THREADS // Int(WARP_SIZE)
    comptime rows_per_warp = 4
    comptime row_groups = input_dims // rows_per_warp
    comptime packed_columns = hidden_dims // 2
    comptime blocks_per_row = hidden_dims // 16
    comptime columns_per_warp_step = Int(WARP_SIZE) * 16
    var tid = Int(thread_idx.x)
    var lane = tid % Int(WARP_SIZE)
    var warp_in_block = tid // Int(WARP_SIZE)
    var work = Int(block_idx.x) * warps_per_block + warp_in_block
    if work >= TOP_K * row_groups:
        return
    var expert = work // row_groups
    var row = (work % row_groups) * rows_per_warp
    var result0: Float32 = 0.0
    var result1: Float32 = 0.0
    var result2: Float32 = 0.0
    var result3: Float32 = 0.0
    for step in range(0, hidden_dims, columns_per_warp_step):
        var column = step + lane * 16
        var input_values = SIMD[DType.float32, 16](0.0)
        if column < hidden_dims:
            input_values = hidden.load[width=16](expert * hidden_dims + column)
            var block = column // 16
            var packed_column = column // 2
            var base0 = (
                expert * input_dims + row
            ) * packed_columns + packed_column
            var scale0 = (expert * input_dims + row) * blocks_per_row + block
            var values0 = decode_e2m1x16(weight.load[width=8](base0))
            var values1 = decode_e2m1x16(
                weight.load[width=8](base0 + packed_columns)
            )
            var values2 = decode_e2m1x16(
                weight.load[width=8](base0 + 2 * packed_columns)
            )
            var values3 = decode_e2m1x16(
                weight.load[width=8](base0 + 3 * packed_columns)
            )
            result0 += (values0 * input_values).reduce_add() * decode_e4m3(
                scales[scale0]
            )
            result1 += (values1 * input_values).reduce_add() * decode_e4m3(
                scales[scale0 + blocks_per_row]
            )
            result2 += (values2 * input_values).reduce_add() * decode_e4m3(
                scales[scale0 + 2 * blocks_per_row]
            )
            result3 += (values3 * input_values).reduce_add() * decode_e4m3(
                scales[scale0 + 3 * blocks_per_row]
            )
    var mixed_scale = global_scales[expert] * scores[expert]
    result0 = warp.sum(result0) * mixed_scale
    result1 = warp.sum(result1) * mixed_scale
    result2 = warp.sum(result2) * mixed_scale
    result3 = warp.sum(result3) * mixed_scale
    if lane == 0:
        var output_base = expert * input_dims + row
        expert_output[output_base] = result0
        expert_output[output_base + 1] = result1
        expert_output[output_base + 2] = result2
        expert_output[output_base + 3] = result3


def main() raises:
    with DeviceContext(api="metal") as ctx:
        comptime up_weight_count = TOP_K * HIDDEN_DIMS * (INPUT_DIMS // 2)
        comptime up_scale_count = TOP_K * HIDDEN_DIMS * (INPUT_DIMS // 16)
        comptime down_weight_count = TOP_K * INPUT_DIMS * (HIDDEN_DIMS // 2)
        comptime down_scale_count = TOP_K * INPUT_DIMS * (HIDDEN_DIMS // 16)

        var up_weight = ctx.enqueue_create_buffer[DType.uint8](up_weight_count)
        var up_scales = ctx.enqueue_create_buffer[DType.uint8](up_scale_count)
        var up_global = ctx.enqueue_create_buffer[DType.float32](TOP_K)
        var down_weight = ctx.enqueue_create_buffer[DType.uint8](
            down_weight_count
        )
        var down_scales = ctx.enqueue_create_buffer[DType.uint8](
            down_scale_count
        )
        var down_global = ctx.enqueue_create_buffer[DType.float32](TOP_K)
        var input = ctx.enqueue_create_buffer[DType.float32](INPUT_DIMS)
        var scores = ctx.enqueue_create_buffer[DType.float32](TOP_K)
        var hidden = ctx.enqueue_create_buffer[DType.float32](
            TOP_K * HIDDEN_DIMS
        )
        var expert_output = ctx.enqueue_create_buffer[DType.float32](
            TOP_K * INPUT_DIMS
        )
        var output = ctx.enqueue_create_buffer[DType.float32](INPUT_DIMS)

        up_weight.enqueue_fill(UInt8(0x21))
        up_scales.enqueue_fill(UInt8(0x38))
        up_global.enqueue_fill(Float32(0.01))
        down_weight.enqueue_fill(UInt8(0x21))
        down_scales.enqueue_fill(UInt8(0x38))
        down_global.enqueue_fill(Float32(0.02))
        input.enqueue_fill(Float32(0.25))
        scores.enqueue_fill(Float32(1.0 / Float32(TOP_K)))

        comptime mlp = selected_expert_mlp[INPUT_DIMS, HIDDEN_DIMS, THREADS]
        var compiled_mlp = ctx.compile_function[mlp]()
        comptime up_rows = selected_up_rows[INPUT_DIMS, HIDDEN_DIMS]
        comptime down_rows = selected_down_rows[INPUT_DIMS, HIDDEN_DIMS]
        comptime down_reduce_rows = selected_down_reduce_rows[
            INPUT_DIMS, HIDDEN_DIMS
        ]
        comptime up_four_rows = selected_up_four_rows[INPUT_DIMS, HIDDEN_DIMS]
        comptime down_four_rows = selected_down_four_rows[
            INPUT_DIMS, HIDDEN_DIMS
        ]
        var compiled_up_rows = ctx.compile_function[up_rows]()
        var compiled_down_rows = ctx.compile_function[down_rows]()
        var compiled_down_reduce_rows = ctx.compile_function[down_reduce_rows]()
        var compiled_up_four_rows = ctx.compile_function[up_four_rows]()
        var compiled_down_four_rows = ctx.compile_function[down_four_rows]()
        var compiled_reduce = ctx.compile_function[reduce_experts]()

        @always_inline
        def launch(ctx: DeviceContext) raises capturing:
            ctx.enqueue_function(
                compiled_mlp,
                up_weight,
                up_scales,
                up_global,
                down_weight,
                down_scales,
                down_global,
                input,
                scores,
                expert_output,
                grid_dim=TOP_K,
                block_dim=THREADS,
            )
            ctx.enqueue_function(
                compiled_reduce,
                expert_output,
                output,
                grid_dim=1,
                block_dim=INPUT_DIMS,
            )

        launch(ctx)
        ctx.synchronize()

        var expected_up = Float32(INPUT_DIMS // 2) * 1.5 * 0.25 * 0.01
        var expected = (
            Float32(HIDDEN_DIMS // 2) * 1.5 * expected_up * expected_up * 0.02
        )
        with output.map_to_host() as host_output:
            var max_abs: Float32 = 0.0
            for index in range(INPUT_DIMS):
                var difference = abs(host_output[index] - expected)
                max_abs = difference if difference > max_abs else max_abs
            print("nemotron mojo correctness max_abs:", max_abs)
            assert_true(
                max_abs <= 0.02, "fused NVFP4 MLP exceeds numerical tolerance"
            )

        var elapsed_ns = ctx.execution_time[launch](REPEATS)
        var milliseconds = Float64(elapsed_ns) / Float64(REPEATS) / 1_000_000.0
        comptime payload_bytes = (
            up_weight_count
            + up_scale_count
            + down_weight_count
            + down_scale_count
        )
        var bandwidth = Float64(payload_bytes) / milliseconds / 1_000_000.0
        print("nemotron mojo selected-expert mlp ms:", milliseconds)
        print("nemotron mojo effective payload GB/s:", bandwidth)

        @always_inline
        def launch_rows(ctx: DeviceContext) raises capturing:
            comptime warps_per_block = THREADS // Int(WARP_SIZE)
            ctx.enqueue_function(
                compiled_up_rows,
                up_weight,
                up_scales,
                up_global,
                input,
                hidden,
                grid_dim=(TOP_K * HIDDEN_DIMS + warps_per_block - 1)
                // warps_per_block,
                block_dim=THREADS,
            )
            ctx.enqueue_function(
                compiled_down_rows,
                down_weight,
                down_scales,
                down_global,
                hidden,
                scores,
                expert_output,
                grid_dim=(TOP_K * INPUT_DIMS + warps_per_block - 1)
                // warps_per_block,
                block_dim=THREADS,
            )
            ctx.enqueue_function(
                compiled_reduce,
                expert_output,
                output,
                grid_dim=1,
                block_dim=INPUT_DIMS,
            )

        launch_rows(ctx)
        ctx.synchronize()
        with output.map_to_host() as host_output:
            var max_abs: Float32 = 0.0
            for index in range(INPUT_DIMS):
                var difference = abs(host_output[index] - expected)
                max_abs = difference if difference > max_abs else max_abs
            print("nemotron mojo row-parallel correctness max_abs:", max_abs)
            assert_true(
                max_abs <= 0.02,
                "row-parallel NVFP4 MLP exceeds numerical tolerance",
            )

        elapsed_ns = ctx.execution_time[launch_rows](REPEATS)
        milliseconds = Float64(elapsed_ns) / Float64(REPEATS) / 1_000_000.0
        bandwidth = Float64(payload_bytes) / milliseconds / 1_000_000.0
        print("nemotron mojo row-parallel mlp ms:", milliseconds)
        print("nemotron mojo row-parallel effective payload GB/s:", bandwidth)

        @always_inline
        def launch_rows_fused_reduce(ctx: DeviceContext) raises capturing:
            comptime warps_per_block = THREADS // Int(WARP_SIZE)
            ctx.enqueue_function(
                compiled_up_rows,
                up_weight,
                up_scales,
                up_global,
                input,
                hidden,
                grid_dim=(TOP_K * HIDDEN_DIMS + warps_per_block - 1)
                // warps_per_block,
                block_dim=THREADS,
            )
            ctx.enqueue_function(
                compiled_down_reduce_rows,
                down_weight,
                down_scales,
                down_global,
                hidden,
                scores,
                output,
                grid_dim=(INPUT_DIMS + warps_per_block - 1) // warps_per_block,
                block_dim=THREADS,
            )

        launch_rows_fused_reduce(ctx)
        ctx.synchronize()
        with output.map_to_host() as host_output:
            var max_abs: Float32 = 0.0
            for index in range(INPUT_DIMS):
                var difference = abs(host_output[index] - expected)
                max_abs = difference if difference > max_abs else max_abs
            print("nemotron mojo fused-reduce correctness max_abs:", max_abs)
            assert_true(
                max_abs <= 0.02,
                "fused-reduce NVFP4 MLP exceeds numerical tolerance",
            )

        elapsed_ns = ctx.execution_time[launch_rows_fused_reduce](REPEATS)
        milliseconds = Float64(elapsed_ns) / Float64(REPEATS) / 1_000_000.0
        bandwidth = Float64(payload_bytes) / milliseconds / 1_000_000.0
        print("nemotron mojo fused-reduce mlp ms:", milliseconds)
        print("nemotron mojo fused-reduce effective payload GB/s:", bandwidth)

        @always_inline
        def launch_four_rows(ctx: DeviceContext) raises capturing:
            comptime warps_per_block = THREADS // Int(WARP_SIZE)
            comptime up_work = TOP_K * (HIDDEN_DIMS // 4)
            comptime down_work = TOP_K * (INPUT_DIMS // 4)
            ctx.enqueue_function(
                compiled_up_four_rows,
                up_weight,
                up_scales,
                up_global,
                input,
                hidden,
                grid_dim=(up_work + warps_per_block - 1) // warps_per_block,
                block_dim=THREADS,
            )
            ctx.enqueue_function(
                compiled_down_four_rows,
                down_weight,
                down_scales,
                down_global,
                hidden,
                scores,
                expert_output,
                grid_dim=(down_work + warps_per_block - 1) // warps_per_block,
                block_dim=THREADS,
            )
            ctx.enqueue_function(
                compiled_reduce,
                expert_output,
                output,
                grid_dim=1,
                block_dim=INPUT_DIMS,
            )

        launch_four_rows(ctx)
        ctx.synchronize()
        with output.map_to_host() as host_output:
            var max_abs: Float32 = 0.0
            for index in range(INPUT_DIMS):
                var difference = abs(host_output[index] - expected)
                max_abs = difference if difference > max_abs else max_abs
            print("nemotron mojo four-row correctness max_abs:", max_abs)
            assert_true(
                max_abs <= 0.02,
                "four-row NVFP4 MLP exceeds numerical tolerance",
            )

        elapsed_ns = ctx.execution_time[launch_four_rows](REPEATS)
        milliseconds = Float64(elapsed_ns) / Float64(REPEATS) / 1_000_000.0
        bandwidth = Float64(payload_bytes) / milliseconds / 1_000_000.0
        print("nemotron mojo four-row mlp ms:", milliseconds)
        print("nemotron mojo four-row effective payload GB/s:", bandwidth)

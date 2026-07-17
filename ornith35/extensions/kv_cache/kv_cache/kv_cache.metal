#include <metal_stdlib>

using namespace metal;

[[kernel]] void ornith35_append_bf16(
    const device bfloat* update [[buffer(0)]],
    device bfloat* cache [[buffer(1)]],
    constant uint& position [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  uint column = index % width;
  uint row = index / width;
  uint token = row % tokens;
  uint head = row / tokens;
  cache[(head * capacity + position + token) * width + column] = update[index];
}

[[kernel]] void ornith35_append_kv_bf16(
    const device bfloat* key_update [[buffer(0)]],
    const device bfloat* value_update [[buffer(1)]],
    device bfloat* keys [[buffer(2)]],
    device bfloat* values [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  uint column = index % width;
  uint row = index / width;
  uint token = row % tokens;
  uint head = row / tokens;
  uint destination = (head * capacity + position + token) * width + column;
  keys[destination] = key_update[index];
  values[destination] = value_update[index];
}

[[kernel]] void ornith35_append_kv_transposed_bf16(
    const device bfloat* key_update [[buffer(0)]],
    const device bfloat* value_update [[buffer(1)]],
    device bfloat* keys [[buffer(2)]],
    device bfloat* values [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  uint column = index % width;
  uint row = index / width;
  uint head = row % heads;
  uint token = row / heads;
  if (token >= tokens) {
    return;
  }
  uint destination = (head * capacity + position + token) * width + column;
  keys[destination] = key_update[index];
  values[destination] = value_update[index];
}

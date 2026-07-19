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

[[kernel]] void ornith35_append_packed_mse8(
    const device uchar* packed_key_update [[buffer(0)]],
    const device bfloat* key_norm_update [[buffer(1)]],
    const device uchar* packed_value_update [[buffer(2)]],
    const device bfloat* value_norm_update [[buffer(3)]],
    device uchar* packed_keys [[buffer(4)]],
    device bfloat* key_norms [[buffer(5)]],
    device uchar* packed_values [[buffer(6)]],
    device bfloat* value_norms [[buffer(7)]],
    constant uint& position [[buffer(8)]],
    constant uint& tokens [[buffer(9)]],
    constant uint& capacity [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
  uint column = index % width;
  uint row = index / width;
  uint token = row % tokens;
  uint head = row / tokens;
  uint source = (head * tokens + token) * width + column;
  uint destination = (head * capacity + position + token) * width + column;
  packed_keys[destination] = packed_key_update[source];
  packed_values[destination] = packed_value_update[source];
  if (column == 0u) {
    uint source_norm = head * tokens + token;
    uint destination_norm = head * capacity + position + token;
    key_norms[destination_norm] = key_norm_update[source_norm];
    value_norms[destination_norm] = value_norm_update[source_norm];
  }
}

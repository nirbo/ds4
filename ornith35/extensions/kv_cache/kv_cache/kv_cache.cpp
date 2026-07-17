#include <dlfcn.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "kv_cache/kv_cache.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace ornith35 {

namespace {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("unable to locate Ornith-35 extension binary");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

void validate_append(
    const mx::array& cache,
    const mx::array& update,
    int position) {
  if (cache.dtype() != mx::bfloat16 || update.dtype() != mx::bfloat16) {
    throw std::invalid_argument("append_bf16 requires BF16 arrays");
  }
  if (cache.ndim() != 3 || update.ndim() != 3 ||
      update.shape(0) != cache.shape(0) ||
      update.shape(2) != cache.shape(2)) {
    throw std::invalid_argument("append_bf16 shape mismatch");
  }
  if (update.shape(1) <= 0 || position < 0 ||
      position > cache.shape(1) - update.shape(1)) {
    throw std::invalid_argument("append_bf16 range is outside capacity");
  }
  if (!cache.flags().row_contiguous || !update.flags().row_contiguous) {
    throw std::invalid_argument("append_bf16 requires row-contiguous arrays");
  }
}

} // namespace

mx::array append_bf16(
    const mx::array& cache,
    const mx::array& update,
    int position,
    mx::StreamOrDevice stream) {
  validate_append(cache, update, position);
  return mx::array(
      cache.shape(),
      cache.dtype(),
      std::make_shared<AppendBF16>(mx::to_stream(stream), position),
      {cache, update});
}

std::vector<mx::array> append_kv_bf16(
    const mx::array& keys,
    const mx::array& values,
    const mx::array& key_update,
    const mx::array& value_update,
    int position,
    mx::StreamOrDevice stream) {
  validate_append(keys, key_update, position);
  validate_append(values, value_update, position);
  if (keys.shape() != values.shape() ||
      key_update.shape() != value_update.shape()) {
    throw std::invalid_argument("append_kv_bf16 K/V shape mismatch");
  }
  auto primitive =
      std::make_shared<AppendKVBF16>(mx::to_stream(stream), position);
  return mx::array::make_arrays(
      {keys.shape(), values.shape()},
      {keys.dtype(), values.dtype()},
      primitive,
      {keys, values, key_update, value_update});
}

void AppendBF16::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("Ornith35AppendBF16 has no CPU implementation");
}

#ifdef _METAL_

void AppendBF16::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& cache = inputs[0];
  const auto& update = inputs[1];
  auto& output = outputs[0];
  output.copy_shared_buffer(cache);

  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  auto library = device.get_library(
      "ornith35_kv_cache_ext", current_binary_dir());
  auto kernel = device.get_kernel("ornith35_append_bf16", library);
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(update, 0);
  encoder.set_output_array(output, 1);

  uint32_t position = static_cast<uint32_t>(position_);
  uint32_t tokens = static_cast<uint32_t>(update.shape(1));
  uint32_t capacity = static_cast<uint32_t>(cache.shape(1));
  uint32_t width = static_cast<uint32_t>(cache.shape(2));
  encoder.set_bytes(position, 2);
  encoder.set_bytes(tokens, 3);
  encoder.set_bytes(capacity, 4);
  encoder.set_bytes(width, 5);

  size_t elements = update.size();
  size_t group_size = std::min(
      elements, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(
      MTL::Size(elements, 1, 1),
      MTL::Size(group_size, 1, 1));
}

void AppendKVBF16::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& keys = inputs[0];
  const auto& values = inputs[1];
  const auto& key_update = inputs[2];
  const auto& value_update = inputs[3];
  auto& output_keys = outputs[0];
  auto& output_values = outputs[1];
  output_keys.copy_shared_buffer(keys);
  output_values.copy_shared_buffer(values);

  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  auto library = device.get_library(
      "ornith35_kv_cache_ext", current_binary_dir());
  auto kernel = device.get_kernel("ornith35_append_kv_bf16", library);
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(key_update, 0);
  encoder.set_input_array(value_update, 1);
  encoder.set_output_array(output_keys, 2);
  encoder.set_output_array(output_values, 3);

  uint32_t position = static_cast<uint32_t>(position_);
  uint32_t tokens = static_cast<uint32_t>(key_update.shape(1));
  uint32_t capacity = static_cast<uint32_t>(keys.shape(1));
  uint32_t width = static_cast<uint32_t>(keys.shape(2));
  encoder.set_bytes(position, 4);
  encoder.set_bytes(tokens, 5);
  encoder.set_bytes(capacity, 6);
  encoder.set_bytes(width, 7);

  size_t elements = key_update.size();
  size_t group_size = std::min(
      elements, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(
      MTL::Size(elements, 1, 1),
      MTL::Size(group_size, 1, 1));
}

#else

void AppendBF16::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("Ornith35AppendBF16 has no Metal implementation");
}

void AppendKVBF16::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("Ornith35AppendKVBF16 has no Metal implementation");
}

#endif

void AppendKVBF16::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("Ornith35AppendKVBF16 has no CPU implementation");
}

std::vector<mx::array> AppendBF16::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("Ornith35AppendBF16 is inference-only");
}

std::vector<mx::array> AppendBF16::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("Ornith35AppendBF16 is inference-only");
}

std::pair<std::vector<mx::array>, std::vector<int>> AppendBF16::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("Ornith35AppendBF16 has no vmap implementation");
}

bool AppendBF16::is_equivalent(const mx::Primitive& other) const {
  const auto& append = static_cast<const AppendBF16&>(other);
  return position_ == append.position_;
}

std::vector<mx::array> AppendKVBF16::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("Ornith35AppendKVBF16 is inference-only");
}

std::vector<mx::array> AppendKVBF16::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error("Ornith35AppendKVBF16 is inference-only");
}

std::pair<std::vector<mx::array>, std::vector<int>> AppendKVBF16::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("Ornith35AppendKVBF16 has no vmap implementation");
}

bool AppendKVBF16::is_equivalent(const mx::Primitive& other) const {
  const auto& append = static_cast<const AppendKVBF16&>(other);
  return position_ == append.position_;
}

} // namespace ornith35

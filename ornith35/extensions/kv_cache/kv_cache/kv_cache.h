#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ornith35 {

mx::array append_bf16(
    const mx::array& cache,
    const mx::array& update,
    int position,
    mx::StreamOrDevice stream = {});

std::vector<mx::array> append_kv_bf16(
    const mx::array& keys,
    const mx::array& values,
    const mx::array& key_update,
    const mx::array& value_update,
    int position,
    mx::StreamOrDevice stream = {});

std::vector<mx::array> append_kv_transposed_bf16(
    const mx::array& keys,
    const mx::array& values,
    const mx::array& key_update,
    const mx::array& value_update,
    int position,
    mx::StreamOrDevice stream = {});

class AppendBF16 : public mx::Primitive {
 public:
  AppendBF16(mx::Stream stream, int position)
      : mx::Primitive(stream), position_(position) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "Ornith35AppendBF16";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int position_;
};

class AppendKVBF16 : public mx::Primitive {
 public:
  AppendKVBF16(mx::Stream stream, int position, bool transposed)
      : mx::Primitive(stream), position_(position), transposed_(transposed) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "Ornith35AppendKVBF16";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int position_;
  bool transposed_;
};

} // namespace ornith35

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include "kv_cache/kv_cache.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, module) {
  module.doc() = "Ornith-35 append-only MLX K/V cache primitive";
  module.def(
      "append_bf16",
      &ornith35::append_bf16,
      "cache"_a,
      "update"_a,
      "position"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  module.def(
      "append_kv_bf16",
      &ornith35::append_kv_bf16,
      "keys"_a,
      "values"_a,
      "key_update"_a,
      "value_update"_a,
      "position"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  module.def(
      "append_kv_transposed_bf16",
      &ornith35::append_kv_transposed_bf16,
      "keys"_a,
      "values"_a,
      "key_update"_a,
      "value_update"_a,
      "position"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  module.def(
      "append_packed_mse4",
      &ornith35::append_packed_mse4,
      "packed_keys"_a,
      "key_norms"_a,
      "packed_values"_a,
      "value_norms"_a,
      "packed_key_update"_a,
      "key_norm_update"_a,
      "packed_value_update"_a,
      "value_norm_update"_a,
      "position"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
}

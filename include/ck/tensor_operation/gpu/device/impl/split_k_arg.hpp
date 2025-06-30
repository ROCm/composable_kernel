// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
namespace ck {
namespace tensor_operation {
namespace device {

struct ArgumentSplitK
{
  index_t k_batch() const { return k_batch_; }
  index_t k_dim_size() const { return k_dim_size_; }
  index_t m_dim_size() const { return m_dim_size_; }
  index_t n_dim_size() const { return n_dim_size_; }
  float arithmetic_intensity() const { return arithmetic_intensity_; }
  std::string data_type() const { return data_type_; }
  protected:
        index_t k_batch_{-1};
        index_t k_dim_size_{-1};
        index_t m_dim_size_{-1};
        index_t n_dim_size_{-1};
        float arithmetic_intensity_{-1};
        std::string data_type_{""};
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

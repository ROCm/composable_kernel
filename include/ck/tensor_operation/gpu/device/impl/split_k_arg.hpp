// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

struct ArgumentSplitK
{
  index_t k_batch() const { return k_batch_; }
  index_t k_dim_size() const { return k_dim_size_; }
  protected:
        index_t k_batch_{-1};
        index_t k_dim_size_{-1};
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

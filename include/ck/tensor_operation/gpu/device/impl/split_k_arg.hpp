// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

struct ArgumentSplitK
{
  index_t k_batch() const { return k_batch_; }
  protected:
        index_t k_batch_;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

  enum class SplitKMode
{
    FixedSplitK = 0,
    BestOccupancyWithOversubscription
};

struct ParamsSplitK
{
  SplitKMode split_k_mode_{SplitKMode::FixedSplitK};
  index_t split_k_value_{1};
  index_t oversubscription_{-1};

  std::string to_string() const
  {
    const std::string str = (split_k_mode_ == SplitKMode::FixedSplitK) 
      ? "FixedSplitK = " + std::to_string(split_k_value_)
      : "BestOccupancyWithOversubscription = " + std::to_string(oversubscription_);
    return str;
  }
};

}
}
}

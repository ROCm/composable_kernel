// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

enum class SplitKStrategy
{
    FixedSplitK = 0,
    BestOccupancy,
    BestOccupancyWithMinQuantization
};

struct ParamsSplitK
{
  SplitKStrategy strategy_{SplitKStrategy::FixedSplitK};
  index_t fixed_value_{1};
};

}
}
}

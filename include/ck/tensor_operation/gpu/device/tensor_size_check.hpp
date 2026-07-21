// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DataType, typename Lengths>
bool tensor_exceeds_2gb(const Lengths& lengths)
{
    constexpr long_index_t TwoGB = (long_index_t{1} << 31);
    long_index_t total           = sizeof(DataType);
    for(const auto& l : lengths)
        total *= l;
    return total > TwoGB;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck

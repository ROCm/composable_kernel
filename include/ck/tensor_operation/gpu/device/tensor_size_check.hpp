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
    long_index_t total           = 1;
    for(const auto& l : lengths)
        total *= l;
    long_index_t total_bytes = total * sizeof(DataType);
    // Element counts of 2^31 elements cannot be represented as int32_t (">=" check).
    // A byte count of exactly 2^31 can be represented as uint32_t and is therefore valid (strict
    // ">" check). The element-count limit is an additional constraint only for 1-byte DataTypes.
    return total >= TwoGB || total_bytes > TwoGB;
}

template <typename DataType, typename Desc>
bool descriptor_exceeds_2gb(const Desc& desc)
{
    constexpr long_index_t TwoGB          = (long_index_t{1} << 31);
    const long_index_t element_space_size = desc.GetElementSpaceSize();
    return element_space_size * sizeof(DataType) > TwoGB || element_space_size >= TwoGB;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck

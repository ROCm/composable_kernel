// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>
#include <vector>

#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/**
 * Calculates the maximum number of subsequent elements of the fast changing dimension
 * that are consecutive in memory.
 *
 * Example:
 *   NumDimM = 2, NumDimK = 3
 *   A shape =   [  2,   3,  4, 5, 6]
 *   A strides = [360, 120, 30, 6, 1]
 *                |  M   |  |   K  |
 *   It follows from strides that K is FCD and all the subsequent elements of K are consecutive
 *   in memory.
 *   But if strides were [360, 120, 6, 24, 1], then only 6 subsequent elements of K would be
 *   consecutive in memory.
 *
 * Assumes that the dimensions are split into two groups of `NumDim1` and `NumDim2` dimensions.
 */
template <index_t NumDim1, index_t NumDim2>
auto CalculateMaxRead(const std::vector<index_t>& lengths, const std::vector<index_t>& strides)
{
    assert(lengths.size() == NumDim1 + NumDim2 && strides.size() == NumDim1 + NumDim2);

    // Determine the beginning and end idx of the group representing the FCD.
    index_t begin_idx, end_idx;
    if(strides[NumDim1 - 1] == 1)
    {
        begin_idx = 0;
        end_idx   = NumDim1 - 1;
    }
    else if(strides[NumDim1 + NumDim2 - 1] == 1)
    {
        begin_idx = NumDim1;
        end_idx   = NumDim1 + NumDim2 - 1;
    }
    else
    {
        // The dimension consecutive in memory is not the last dimension of any group, so only
        // one element can be read/written at once.
        return 1;
    }

    index_t consecutive_stride = 1;
    for(index_t dim_idx = end_idx; dim_idx >= begin_idx; --dim_idx)
    {
        if(strides[dim_idx] == consecutive_stride)
        {
            consecutive_stride *= lengths[dim_idx];
        }
        else
        {
            break;
        }
    }
    const index_t max_subsequent_elems = consecutive_stride;
    return max_subsequent_elems;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck

// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename InputType_,
          typename WeightType_,
          typename IndexType_,
          index_t Experts_,
          index_t IssuesPerCol_  = 1, // issue along col, to make sure block_reduce() OK
          index_t BytesPerIssue_ = sizeof(InputType_),
          index_t BlockSize_     = 256>
struct TopkSoftmaxWarpPerRowProblem
{
    // TODO: this kernel only support warp per row
    using InputType  = remove_cvref_t<InputType_>;
    using WeightType = remove_cvref_t<WeightType_>;
    using IndexType  = remove_cvref_t<IndexType_>;

    static constexpr index_t Experts       = Experts_;
    static constexpr index_t BytesPerIssue = BytesPerIssue_;
    static constexpr index_t BlockSize     = BlockSize_;
    static constexpr index_t WarpSize      = get_warp_size();

    static_assert(BytesPerIssue % sizeof(InputType) == 0);
    static constexpr index_t VectorSize = BytesPerIssue / sizeof(InputType);
    static_assert(Experts % VectorSize == 0);
    static constexpr index_t LanesPerRow = min(Experts / VectorSize, WarpSize);
    static_assert(WarpSize % LanesPerRow == 0);
    static constexpr index_t RowsPerWarp  = WarpSize / LanesPerRow;
    static constexpr index_t IssuesPerRow = Experts / (LanesPerRow * VectorSize);

    static constexpr index_t IssuesPerCol = IssuesPerCol_;

    static constexpr index_t WarpsPerBlock = BlockSize / WarpSize;
    static constexpr index_t RowsPerBlock  = RowsPerWarp * WarpsPerBlock;
};
} // namespace ck_tile

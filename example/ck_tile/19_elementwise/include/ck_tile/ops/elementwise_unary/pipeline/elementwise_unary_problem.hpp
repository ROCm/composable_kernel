// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename InputType_,
          typename OutputType_,
          typename UnaryFunctor_,
          index_t BytesPerIssue_ = sizeof(InputType_), // this is input
          index_t Chunks_        = 8,
          index_t BlockSize_     = 256>
struct ElementwiseUnaryWarpPerRowProblem
{
    // TODO: this kernel only support warp per row
    using InputType    = remove_cvref_t<InputType_>;
    using OutputType   = remove_cvref_t<OutputType_>;
    using UnaryFunctor = remove_cvref_t<UnaryFunctor_>;

    static constexpr index_t Chunks        = Chunks_;
    static constexpr index_t BytesPerIssue = BytesPerIssue_;
    static constexpr index_t BlockSize     = BlockSize_;
    static constexpr index_t WarpSize      = get_warp_size();

    static_assert(BytesPerIssue % sizeof(InputType) == 0);
    static constexpr index_t VectorSize    = BytesPerIssue / sizeof(InputType);
    static constexpr index_t LanesPerRow   = WarpSize;
    static constexpr index_t WarpsPerBlock = BlockSize / LanesPerRow;
    static constexpr index_t IssuesPerRow  = 1;

#if 0
    static_assert(Experts % VectorSize == 0);
    static constexpr index_t LanesPerRow = min(Experts / VectorSize, WarpSize);
    static_assert(WarpSize % LanesPerRow == 0);
    static constexpr index_t RowsPerWarpPerColIssue = WarpSize / LanesPerRow;
    static constexpr index_t RowsPerWarp            = IssuesPerCol * RowsPerWarpPerColIssue;
    static constexpr index_t IssuesPerRow           = Experts / (LanesPerRow * VectorSize);

    static constexpr index_t WarpsPerBlock = BlockSize / WarpSize;
    static constexpr index_t RowsPerBlock  = RowsPerWarp * WarpsPerBlock;
#endif
};
} // namespace ck_tile

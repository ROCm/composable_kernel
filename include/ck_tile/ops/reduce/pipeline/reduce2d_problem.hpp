// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename BlockShape_,
          typename ReduceOp_,
          bool OutputIndex_ = false>
struct Reduce2dProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
    using ReduceOp        = ReduceOp_;

    static constexpr bool kOutputIndex       = OutputIndex_;
    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;
};

} // namespace ck_tile

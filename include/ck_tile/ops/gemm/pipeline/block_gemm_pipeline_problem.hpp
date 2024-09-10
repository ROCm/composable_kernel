// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

#define VectorLoadSize 16

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          bool kPadA_ = false,
          bool kPadB_ = false,
          bool kPadC_ = false>
struct BlockGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();
    static constexpr bool kPadA         = kPadA_;
    static constexpr bool kPadB         = kPadB_;
    static constexpr bool kPadC         = kPadC_;

    static constexpr index_t AlignmentA = kPadA ? VectorLoadSize / sizeof(ADataType) : 1;
    static constexpr index_t AlignmentB = kPadB ? VectorLoadSize / sizeof(BDataType) : 1;
    static constexpr index_t AlignmentC = kPadC ? VectorLoadSize / sizeof(CDataType) : 1;
};

} // namespace ck_tile

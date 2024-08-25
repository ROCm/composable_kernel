// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          bool kPadA_,
          bool kPadB_,
          bool kPadC_>
struct BlockGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * 64;
    static constexpr bool kPadA         = kPadA_;
    static constexpr bool kPadB         = kPadB_;
    static constexpr bool kPadC         = kPadC_;

    static constexpr index_t AlignmentA = kPadA ? 16 / sizeof(ADataType) : 1;
    static constexpr index_t AlignmentB = kPadB ? 16 / sizeof(BDataType) : 1;
    static constexpr index_t AlignmentC = kPadC ? 16 / sizeof(CDataType) : 1;
};

} // namespace ck_tile

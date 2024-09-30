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
          typename TileGemmTraits_>
struct BlockGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;
    using GemmTraits     = remove_cvref_t<TileGemmTraits_>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();
    static constexpr bool kPadA         = GemmTraits::kPadA;
    static constexpr bool kPadB         = GemmTraits::kPadB;
    static constexpr bool kPadC         = GemmTraits::kPadC;

    using LayoutA = remove_cvref_t<typename GemmTraits::LayoutA>;
    using LayoutB = remove_cvref_t<typename GemmTraits::LayoutB>;
    using LayoutC = remove_cvref_t<typename GemmTraits::LayoutC>;

    static constexpr index_t AlignmentA = kPadA ? 1 : VectorLoadSize / sizeof(ADataType);
    static constexpr index_t AlignmentB = kPadB ? 1 : VectorLoadSize / sizeof(BDataType);
    static constexpr index_t AlignmentC = kPadC ? 1 : VectorLoadSize / sizeof(CDataType);
};

} // namespace ck_tile

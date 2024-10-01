// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

#define VectorLoadSize 16

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          bool kPadA_ = false,
          bool kPadB_ = false,
          bool kPadC_ = false>
struct BlockGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using ALayout = remove_cvref_t<ALayout_>;
    using BLayout = remove_cvref_t<BLayout_>;
    using CLayout = remove_cvref_t<CLayout_>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();
    static constexpr bool kPadA         = kPadA_;
    static constexpr bool kPadB         = kPadB_;
    static constexpr bool kPadC         = kPadC_;

    static constexpr index_t AlignmentA = kPadA ? 1 : VectorLoadSize / sizeof(ADataType);
    static constexpr index_t AlignmentB = kPadB ? 1 : VectorLoadSize / sizeof(BDataType);
    static constexpr index_t AlignmentC = kPadC ? 1 : VectorLoadSize / sizeof(CDataType);
};

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          bool kPadA_                      = false,
          bool kPadB_                      = false,
          bool kPadC_                      = false,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = false,
          TailNumber TailNum_              = TailNumber::Full>
struct UniversalGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using ALayout = remove_cvref_t<ALayout_>;
    using BLayout = remove_cvref_t<BLayout_>;
    using CLayout = remove_cvref_t<CLayout_>;

    static constexpr auto Scheduler     = Scheduler_;
    static constexpr auto HasHotLoop    = HasHotLoop_;
    static constexpr auto TailNum       = TailNum_;
    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadA = kPadA_;
    static constexpr bool kPadB = kPadB_;
    static constexpr bool kPadC = kPadC_;

    // TODO: what about vector load/store size? should we have template paramter for A/B/C ?
    static constexpr index_t AlignmentA = kPadA ? VectorLoadSize / sizeof(ADataType) : 1;
    static constexpr index_t AlignmentB = kPadB ? VectorLoadSize / sizeof(BDataType) : 1;
    static constexpr index_t AlignmentC = kPadC ? VectorLoadSize / sizeof(CDataType) : 1;
};

} // namespace ck_tile

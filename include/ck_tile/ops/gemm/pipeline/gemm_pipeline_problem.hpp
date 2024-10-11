// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

#define VectorLoadSize 16

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename TileGemmTraits_>
struct GemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;
    using GemmTraits     = remove_cvref_t<TileGemmTraits_>;

    using ALayout = remove_cvref_t<typename GemmTraits::ALayout>;
    using BLayout = remove_cvref_t<typename GemmTraits::BLayout>;
    using CLayout = remove_cvref_t<typename GemmTraits::CLayout>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();
    static constexpr bool kPadA         = GemmTraits::kPadA;
    static constexpr bool kPadB         = GemmTraits::kPadB;
    static constexpr bool kPadC         = GemmTraits::kPadC;

    static constexpr index_t AlignmentA = kPadA ? 1 : VectorLoadSize / sizeof(ADataType);
    static constexpr index_t AlignmentB = kPadB ? 1 : VectorLoadSize / sizeof(BDataType);
    static constexpr index_t AlignmentC = kPadC ? 1 : VectorLoadSize / sizeof(CDataType);
};

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename TileGemmTraits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct UniversalGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;
    using GemmTraits     = remove_cvref_t<TileGemmTraits_>;

    using ALayout = remove_cvref_t<typename GemmTraits::ALayout>;
    using BLayout = remove_cvref_t<typename GemmTraits::BLayout>;
    using CLayout = remove_cvref_t<typename GemmTraits::CLayout>;

    static constexpr auto Scheduler     = Scheduler_;
    static constexpr auto HasHotLoop    = HasHotLoop_;
    static constexpr auto TailNum       = TailNum_;
    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadA = GemmTraits::kPadA;
    static constexpr bool kPadB = GemmTraits::kPadB;
    static constexpr bool kPadC = GemmTraits::kPadC;

    // TODO: what about vector load/store size? should we have template paramter for A/B/C ?
    static constexpr index_t AlignmentA = kPadA ? VectorLoadSize / sizeof(ADataType) : 1;
    static constexpr index_t AlignmentB = kPadB ? VectorLoadSize / sizeof(BDataType) : 1;
    static constexpr index_t AlignmentC = kPadC ? VectorLoadSize / sizeof(CDataType) : 1;
};

} // namespace ck_tile

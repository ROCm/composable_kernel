// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

static constexpr int _VectorSize = 16;

template <typename TileGemmTraits_>
struct GemmPipelineProblem
{
    using GemmTraits = remove_cvref_t<TileGemmTraits_>;

    using ADataType = remove_cvref_t<typename GemmTraits::ADataType>;
    using BDataType = remove_cvref_t<typename GemmTraits::BDataType>;
    using CDataType = remove_cvref_t<typename GemmTraits::CDataType>;

    using BlockGemmShape = remove_cvref_t<typename GemmTraits::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename GemmTraits::ALayout>;
    using BLayout = remove_cvref_t<typename GemmTraits::BLayout>;
    using CLayout = remove_cvref_t<typename GemmTraits::CLayout>;

    static constexpr index_t kBlockSize = GemmTraits::kBlockSize;

    static constexpr bool kPadM = GemmTraits::kPadM;
    static constexpr bool kPadN = GemmTraits::kPadN;
    static constexpr bool kPadK = GemmTraits::kPadK;

    static constexpr index_t VectorSizeA = GemmTraits::VectorSizeA;
    static constexpr index_t VectorSizeB = GemmTraits::VectorSizeB;
    static constexpr index_t VectorSizeC = GemmTraits::VectorSizeC;
};

template <typename TileGemmTraits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct UniversalGemmPipelineProblem
{
    using GemmTraits = remove_cvref_t<TileGemmTraits_>;

    using BlockGemmShape = remove_cvref_t<typename GemmTraits::BlockGemmShape>;

    using ADataType = remove_cvref_t<typename GemmTraits::ADataType>;
    using BDataType = remove_cvref_t<typename GemmTraits::BDataType>;
    using CDataType = remove_cvref_t<typename GemmTraits::CDataType>;

    using ALayout = remove_cvref_t<typename GemmTraits::ALayout>;
    using BLayout = remove_cvref_t<typename GemmTraits::BLayout>;
    using CLayout = remove_cvref_t<typename GemmTraits::CLayout>;

    static constexpr auto Scheduler  = Scheduler_;
    static constexpr auto HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;

    static constexpr index_t kBlockSize = GemmTraits::kBlockSize;

    static constexpr bool kPadM = GemmTraits::kPadM;
    static constexpr bool kPadN = GemmTraits::kPadN;
    static constexpr bool kPadK = GemmTraits::kPadK;

    static constexpr index_t VectorSizeA = GemmTraits::VectorSizeA;
    static constexpr index_t VectorSizeB = GemmTraits::VectorSizeB;
    static constexpr index_t VectorSizeC = GemmTraits::VectorSizeC;
};

} // namespace ck_tile

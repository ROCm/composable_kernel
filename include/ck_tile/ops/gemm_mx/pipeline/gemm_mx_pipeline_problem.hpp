// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"

#include <string>

namespace ck_tile {

template <typename ADataType_,
          typename AScaleDataType_,
          typename BDataType_,
          typename BScaleDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          uint32_t BlockScaleSize_,
          typename ComputeDataType_        = BDataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct GemmMXPipelineProblemBase : public GemmPipelineProblemBase<ADataType_,
                                                                  BDataType_,
                                                                  CDataType_,
                                                                  BlockGemmShape_,
                                                                  Traits_,
                                                                  ComputeDataType_>
{
    using Base = GemmPipelineProblemBase<ADataType_,
                                         BDataType_,
                                         CDataType_,
                                         BlockGemmShape_,
                                         Traits_,
                                         ComputeDataType_>;

    using Traits = typename Base::Traits;

    using typename Base::ADataType;
    using typename Base::BDataType;
    using typename Base::CDataType;
    using typename Base::ComputeDataType;
    using AScaleDataType = remove_cvref_t<AScaleDataType_>;
    using BScaleDataType = remove_cvref_t<BScaleDataType_>;

    using BlockGemmShape = typename Base::BlockGemmShape;

    using typename Base::ALayout;
    using typename Base::BLayout;
    using typename Base::CLayout;

    static constexpr bool TransposeC = false;

    using Base::kBlockSize;

    using Base::kPadK;
    using Base::kPadM;
    using Base::kPadN;

    using Base::DoubleSmemBuffer;
    using Base::VectorLoadSize;

    using AScaleLayout = remove_cvref_t<typename Traits::AScaleLayout>;
    using BScaleLayout = remove_cvref_t<typename Traits::BScaleLayout>;

    static constexpr uint32_t kBlockScaleSize = BlockScaleSize_;
    static constexpr auto Scheduler           = Scheduler_;
    static constexpr auto HasHotLoop          = HasHotLoop_;
    static constexpr auto TailNum             = TailNum_;

    static_assert(BlockGemmShape::kK % kBlockScaleSize == 0);
    static_assert(Scheduler == GemmPipelineScheduler::Intrawave);

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_mx_problem",
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler,
                      "BlockScaleSize",
                      kBlockScaleSize);
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentAScale()
    {
        static_assert(std::is_same_v<ASclaeLayout, tensor_layout::gemm::RowMajor>);
        return VectorLoadSize / sizeof(AScaleDataType);
    }

    static constexpr index_t VectorSizeAScale = []() {
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);
        return kPadK ? 1 : GetAlignmentAScale();
    }();
};

template <typename ADataType_,
          typename AScaleDataType_,
          typename BDataType_,
          typename BScaleDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          uint32_t ScaleBlockSize_,
          typename ComputeDataType_        = BDataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
using GemmMXPipelineProblem = GemmMXPipelineProblemBase<ADataType_,
                                                        AScaleDataType_,
                                                        BDataType_,
                                                        BScaleDataType_,
                                                        CDataType_,
                                                        BlockGemmShape_,
                                                        Traits_,
                                                        BlockScaleSize_,
                                                        ComputeDataType_,
                                                        Scheduler_,
                                                        HasHotLoop_,
                                                        TailNum_>;

} // namespace ck_tile

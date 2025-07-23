// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"

#include <string>

namespace ck_tile {

template <typename ADataType_,
          typename AQDataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          uint32_t QuantGroupSize_,
          typename ComputeDataType_        = BDataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct GemmAQuantPipelineProblemBase : public GemmPipelineProblemBase<ADataType_,
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
    using AQDataType = remove_cvref_t<AQDataType_>;

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

    using AQLayout = remove_cvref_t<typename Traits::AQLayout>;

    static constexpr uint32_t kQuantGroupSize = QuantGroupSize_;
    static constexpr auto Scheduler           = Scheduler_;
    static constexpr auto HasHotLoop          = HasHotLoop_;
    static constexpr auto TailNum             = TailNum_;

    static_assert(BlockGemmShape::kK % kQuantGroupSize == 0);
    static_assert(Scheduler == GemmPipelineScheduler::Intrawave);

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_aquant_problem",
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler,
                      "QuantGroupSize",
                      kQuantGroupSize);
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentAQ()
    {
        static_assert(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>);
        return VectorLoadSize / sizeof(AQDataType);
    }

    static constexpr index_t VectorSizeAQ = []() {
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);
        return kPadK ? 1 : GetAlignmentAQ();
    }();
};

template <typename ADataType_,
          typename AQDataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          uint32_t QuantGroupSize_,
          typename ComputeDataType_        = BDataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
using GemmAQuantPipelineProblem = GemmAQuantPipelineProblemBase<ADataType_,
                                                                AQDataType_,
                                                                BDataType_,
                                                                CDataType_,
                                                                BlockGemmShape_,
                                                                Traits_,
                                                                QuantGroupSize_,
                                                                ComputeDataType_,
                                                                Scheduler_,
                                                                HasHotLoop_,
                                                                TailNum_>;

} // namespace ck_tile

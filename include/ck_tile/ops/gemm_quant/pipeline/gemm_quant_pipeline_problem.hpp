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
          typename BQDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename QuantGroupSize_,
          bool TransposeC_,
          typename ComputeDataType_        = BDataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct GemmQuantPipelineProblemBase : public GemmPipelineProblemBase<ADataType_,
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
    using BQDataType = remove_cvref_t<BQDataType_>;

    using BlockGemmShape = typename Base::BlockGemmShape;
    using QuantGroupSize = QuantGroupSize_;

    using typename Base::ALayout;
    using typename Base::BLayout;
    using typename Base::CLayout;

    static constexpr bool TransposeC       = TransposeC_;
    static constexpr bool PreshuffleB      = Traits::PreshuffleB;
    static constexpr bool DoubleSmemBuffer = Traits::DoubleSmemBuffer;
    using Base::kBlockSize;

    using Base::kPadK;
    using Base::kPadM;
    using Base::kPadN;

    using Base::VectorLoadSize;

    using AQLayout = remove_cvref_t<typename Traits::AQLayout>;
    using BQLayout = remove_cvref_t<typename Traits::BQLayout>;

    static constexpr auto Scheduler  = Scheduler_;
    static constexpr auto HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;

    static_assert(BlockGemmShape::kM % QuantGroupSize::kM == 0);
    static_assert(BlockGemmShape::kN % QuantGroupSize::kN == 0);
    static_assert(BlockGemmShape::kK % QuantGroupSize::kK == 0);

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_quant_problem",
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler,
                      QuantGroupSize::GetName());
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

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBQ()
    {
        return VectorLoadSize / sizeof(BQDataType);
    }

    static constexpr index_t VectorSizeBQ = []() { return kPadK ? 1 : GetAlignmentBQ(); }();
};

template <typename ADataType_,
          typename AQDataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename QuantGroupSize_,
          bool TransposeC_,
          typename ComputeDataType_        = BDataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
using GemmAQuantPipelineProblem = GemmQuantPipelineProblemBase<ADataType_,
                                                               AQDataType_,
                                                               BDataType_,
                                                               void, // no BQDataType for AQuant
                                                               CDataType_,
                                                               BlockGemmShape_,
                                                               Traits_,
                                                               QuantGroupSize_,
                                                               TransposeC_,
                                                               ComputeDataType_,
                                                               Scheduler_,
                                                               HasHotLoop_,
                                                               TailNum_>;

template <typename ADataType_,
          typename BDataType_,
          typename BQDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename QuantGroupSize_,
          typename ComputeDataType_        = ADataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
using GemmBQuantPipelineProblem = GemmQuantPipelineProblemBase<ADataType_,
                                                               void, // no AQDataType for BQuant
                                                               BDataType_,
                                                               BQDataType_,
                                                               CDataType_,
                                                               BlockGemmShape_,
                                                               Traits_,
                                                               QuantGroupSize_,
                                                               false, // no TransposeC
                                                               ComputeDataType_,
                                                               Scheduler_,
                                                               HasHotLoop_,
                                                               TailNum_>;

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          bool TransposeC_                 = false,
          typename ComputeDataType_        = BDataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
using GemmRowColTensorQuantPipelineProblem =
    GemmQuantPipelineProblemBase<ADataType_,
                                 AccDataType_,
                                 BDataType_,
                                 AccDataType_,
                                 CDataType_,
                                 BlockGemmShape_,
                                 Traits_,
                                 QuantGroupShape<sequence<1, 1, 1>>, // no group size applicable
                                 TransposeC_,
                                 ComputeDataType_,
                                 Scheduler_,
                                 HasHotLoop_,
                                 TailNum_>;
} // namespace ck_tile

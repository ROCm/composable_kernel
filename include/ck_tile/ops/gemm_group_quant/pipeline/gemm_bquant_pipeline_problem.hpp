#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"

#include <string>

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename BQDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          uint32_t QuantGroupSize_,
          bool TransposedWarpGemm_         = false,
          typename ComputeDataType_        = ADataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct GemmBQuantPipelineProblemBase : public GemmPipelineProblemBase<ADataType_,
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
    using BQDataType = remove_cvref_t<BQDataType_>;

    using BlockGemmShape = typename Base::BlockGemmShape;

    using typename Base::ALayout;
    using typename Base::BLayout;
    using typename Base::CLayout;

    static constexpr bool TransposeC = Traits::TransposeC;
    static constexpr bool TransposedWarpGemm = TransposedWarpGemm_;

    using Base::kBlockSize;

    using Base::kPadK;
    using Base::kPadM;
    using Base::kPadN;

    using Base::DoubleSmemBuffer;
    using Base::VectorLoadSize;

    using BQLayout = remove_cvref_t<typename Traits::BQLayout>;

    static constexpr uint32_t kQuantGroupSize = QuantGroupSize_;
    static constexpr auto Scheduler           = Scheduler_;
    static constexpr auto HasHotLoop          = HasHotLoop_;
    static constexpr auto TailNum             = TailNum_;

    static_assert(BlockGemmShape::kK % kQuantGroupSize == 0);
    static_assert(Scheduler == GemmPipelineScheduler::Intrawave);

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_bquant_problem",
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler,
                      "QuantGroupSize",
                      kQuantGroupSize);
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBQ()
    {
        return VectorLoadSize / sizeof(BQDataType);
    }

    static constexpr index_t VectorSizeBQ = []() {
        return kPadK ? 1 : GetAlignmentBQ();
    }();
};

template <typename ADataType_,
          typename BDataType_,
          typename BQDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          uint32_t QuantGroupSize_,
          bool TransposedWarpGemm_         = false,
          typename ComputeDataType_        = ADataType_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
using GemmBQuantPipelineProblem = GemmBQuantPipelineProblemBase<ADataType_,
                                                                BDataType_,
                                                                BQDataType_,
                                                                CDataType_,
                                                                BlockGemmShape_,
                                                                Traits_,
                                                                QuantGroupSize_,
                                                                TransposedWarpGemm_,
                                                                ComputeDataType_,
                                                                Scheduler_,
                                                                HasHotLoop_,
                                                                TailNum_>;

} // namespace ck_tile

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"
#include "ck_tile/core/arch/mma/mma.hpp"

#include "pipeline_tests_helper.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

// Kernel functor: constructs Pipeline internally using device-side get_compiler_target().
// Uses void* for data to avoid host/device symbol mismatches.
template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy,
          bool TransposeC>
struct WaveWisePipelineKernel
{
    static constexpr int kBlockSize = mma_pipeline_test::getCMakeWaveSize();

    __device__ void
    operator()(const void* a_per_lane, const void* b_per_lane, void* c_per_lane) const
    {
        using CompilerTarget = decltype(get_compiler_target());
        using Pipeline       = WaveWiseMmaPipeline<AType,
                                                   BType,
                                                   CType,
                                                   WaveTileM,
                                                   WaveTileN,
                                                   WaveTileK,
                                                   MmaOpFamily::DENSE,
                                                   AccumPolicy,
                                                   TransposeC,
                                                   CompilerTarget>;

        using AVecType = typename Pipeline::AVecType;
        using BVecType = typename Pipeline::BVecType;
        using CVecType = typename Pipeline::CVecType;

        const uint32_t lane = threadIdx.x;

        AVecType a;
        BVecType b;
        CVecType c;
        __builtin_memcpy(&a,
                         static_cast<const uint8_t*>(a_per_lane) + lane * sizeof(AVecType),
                         sizeof(AVecType));
        __builtin_memcpy(&b,
                         static_cast<const uint8_t*>(b_per_lane) + lane * sizeof(BVecType),
                         sizeof(BVecType));
        __builtin_memset(&c, 0, sizeof(CVecType));

        if constexpr(MmaOpTraits<typename Pipeline::MmaOp>::IsSupported)
        {
            Pipeline::exec(a, b, c);
            __builtin_memcpy(
                static_cast<uint8_t*>(c_per_lane) + lane * sizeof(CVecType), &c, sizeof(CVecType));
        }
    }
};

namespace {
const auto should_skip = [](amdgcn_target_id currentArchId) {
    bool isSupportedWmma = false;
    bool isSupportedMfma =
        (currentArchId >= amdgcn_target_id::GFX942) && (currentArchId <= amdgcn_target_id::GFX950);
    return ((currentArchId == amdgcn_target_id::HOST) || !(isSupportedWmma || isSupportedMfma));
};
} // namespace

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy>
struct WaveWisePipelineFactory
{
    template <typename Target>
    struct Create
    {
        using type = WaveWiseMmaPipeline<AType,
                                         BType,
                                         CType,
                                         WaveTileM,
                                         WaveTileN,
                                         WaveTileK,
                                         MmaOpFamily::DENSE,
                                         AccumPolicy,
                                         false,
                                         Target>;
    };
};

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          bool TransposeC            = false>
void WaveWisePipeline_Real_impl()
{
    using Factory =
        WaveWisePipelineFactory<AType, BType, CType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy>;
    using Kernel = WaveWisePipelineKernel<AType,
                                          BType,
                                          CType,
                                          WaveTileM,
                                          WaveTileN,
                                          WaveTileK,
                                          AccumPolicy,
                                          TransposeC>;

    mma_pipeline_test::
        run_pipeline_matrix_test<Factory::template Create, Kernel, AType, BType, CType>(
            WaveTileM,
            WaveTileN,
            WaveTileK,
            should_skip,
            Kernel{},
            /*isSparse=*/false,
            /*transposeExpected=*/TransposeC);
}

TEST(WaveWiseMmaPipeline, FullMatrixVerify_16x16x32)
{
    WaveWisePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u>();
}

TEST(WaveWiseMmaPipeline, FullMatrixVerify_16x16x32_SwapAB)
{
    WaveWisePipeline_Real_impl<fp16_t,
                               fp16_t,
                               fp32_t,
                               16u,
                               16u,
                               32u,
                               MmaAccumPolicy::ROW_MAJOR,
                               true>();
}

TEST(WaveWiseMmaPipeline, FullMatrixVerify_16x16x32_ColMajor)
{
    WaveWisePipeline_Real_impl<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, MmaAccumPolicy::COL_MAJOR>();
}

TEST(WaveWiseMmaPipeline, FullMatrixVerify_16x16x32_ColMajor_TransposeC)
{
    WaveWisePipeline_Real_impl<fp16_t,
                               fp16_t,
                               fp32_t,
                               16u,
                               16u,
                               32u,
                               MmaAccumPolicy::COL_MAJOR,
                               true>();
}

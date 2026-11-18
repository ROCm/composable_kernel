// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp"

using namespace ck_tile;

namespace {

// Host-only: the kernels below are only named, IsSupportedArgument and MakeKernelArgs run on the
// host, and nothing is launched.
template <template <typename> class EpilogueType>
struct BuildBwdDataKernel
{
    static constexpr index_t NDimSpatial = 2;
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 8;
    static constexpr index_t VectorSizeC = 8;
    static constexpr index_t M_Warp      = 2;
    static constexpr index_t N_Warp      = 2;
    static constexpr index_t WarpTile    = 16;

    using GemmShape = TileGemmShape<sequence<128, 128, 32>,
                                    sequence<M_Warp, N_Warp, 1>,
                                    sequence<WarpTile, WarpTile, WarpTile>>;

    using ConvTraits = GroupedConvTraits<NDimSpatial,
                                         ConvolutionSpecialization::Default,
                                         tensor_layout::convolution::NHWGC,
                                         tensor_layout::convolution::GKYXC,
                                         tuple<>,
                                         tensor_layout::convolution::NHWGK,
                                         VectorSizeA,
                                         VectorSizeB,
                                         VectorSizeC>;

    using TilePartitioner =
        GemmSpatiallyLocalTilePartitioner<GemmShape,
                                          ConvTraits::FixedGemmParams::TilePartitionerGroupNum,
                                          ConvTraits::FixedGemmParams::TilePartitionerM01>;

    using GemmUniversalTraits =
        TileGemmUniversalTraits<ConvTraits::FixedGemmParams::kPadM,
                                ConvTraits::FixedGemmParams::kPadN,
                                ConvTraits::FixedGemmParams::kPadK,
                                false, // DoubleSmemBuffer
                                typename ConvTraits::AsLayoutBwdData,
                                typename ConvTraits::BsLayoutBwdData,
                                typename ConvTraits::CLayoutBwdData,
                                ConvTraits::FixedGemmParams::TransposeC,
                                ConvTraits::FixedGemmParams::UseStructuredSparsity,
                                ConvTraits::FixedGemmParams::Persistent,
                                1, // NumWaveGroups
                                ConvTraits::FixedGemmParams::Preshuffle,
                                ConvTraits::FixedGemmParams::LDSVectorSize>;

    using UniversalGemmProblem =
        UniversalGemmPipelineProblem<half_t,
                                     half_t,
                                     float,
                                     GemmShape,
                                     GemmUniversalTraits,
                                     GemmPipelineScheduler::Intrawave,
                                     element_wise::PassThrough,
                                     element_wise::PassThrough,
                                     half_t,
                                     half_t,
                                     ConvTraits::FixedGemmParams::FixedVectorSize,
                                     ConvTraits::VectorSizeA,
                                     ConvTraits::VectorSizeB>;

    using GemmPipeline = GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using EpilogueProblem = CShuffleEpilogueProblem<half_t,
                                                    half_t,
                                                    tuple<>,
                                                    float,
                                                    half_t,
                                                    typename ConvTraits::ImplicitGemmDsLayout,
                                                    typename ConvTraits::FixedGemmParams::ELayout,
                                                    element_wise::PassThrough,
                                                    TilePartitioner::MPerBlock,
                                                    TilePartitioner::NPerBlock,
                                                    M_Warp,
                                                    N_Warp,
                                                    WarpTile,
                                                    WarpTile,
                                                    WarpTile,
                                                    ConvTraits::FixedGemmParams::TransposeC,
                                                    1, // NumWaveGroups
                                                    ConvTraits::FixedGemmParams::FixedVectorSize,
                                                    ConvTraits::VectorSizeC>;

    using type = GroupedConvolutionBackwardDataKernel<ConvTraits,
                                                      TilePartitioner,
                                                      GemmPipeline,
                                                      EpilogueType<EpilogueProblem>>;
};

GroupedConvBwdDataHostArgs create_2d_host_args(index_t k_batch)
{
    // G=2, N=2, K=8, C=8, 3x3 filter, 7x7 input, stride/dilation/padding 1.
    const auto conv_param =
        conv::ConvParam{2, 2, 2, 8, 8, {3, 3}, {7, 7}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    return GroupedConvBwdDataHostArgs{conv_param, nullptr, nullptr, {}, nullptr, k_batch};
}

template <typename Kernel>
bool is_supported(index_t k_batch)
{
    return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(create_2d_host_args(k_batch)));
}

// The control: CShuffleEpilogue with fp16 and an even VectorSizeC accumulates split-K partials
// with atomic_add, so the same arguments that TdmEpilogue must reject below are accepted here.
TEST(GroupedConvBwdDataIsSupportedArgumentTest, CShuffleEpilogueAcceptsSplitK)
{
    using Kernel = typename BuildBwdDataKernel<CShuffleEpilogue>::type;
    EXPECT_TRUE(Kernel::EpilogueSupportsSplitK);
    EXPECT_TRUE(is_supported<Kernel>(1));
    EXPECT_TRUE(is_supported<Kernel>(2));
}

// TdmEpilogue's store overwrites the input-gradient tile instead of accumulating into it.
TEST(GroupedConvBwdDataIsSupportedArgumentTest, TdmEpilogueRejectsSplitK)
{
    using Kernel = typename BuildBwdDataKernel<TdmEpilogue>::type;
    EXPECT_FALSE(Kernel::EpilogueSupportsSplitK);
    EXPECT_TRUE(is_supported<Kernel>(1));
    EXPECT_FALSE(is_supported<Kernel>(2));
}

} // namespace

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp"

using namespace ck_tile;

struct TestConvConfig
{
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 8;
    static constexpr index_t VectorSizeC = 8;

    static constexpr index_t M_Tile = 128;
    static constexpr index_t N_Tile = 128;
    static constexpr index_t K_Tile = 32;

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 16;
    static constexpr index_t N_Warp_Tile = 16;
    static constexpr index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr GemmPipeline Pipeline    = GemmPipeline::COMPUTE_V3;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
};

// ============================================================================
// V6 pipeline: num_loop check
//
// GemmPipelineAgBgCrCompV6 uses a 3-stage prefetch (PrefetchStages=3).  The
// hot loop only executes when num_loop > 3 (i.e. num_loop >= 4). When
// num_loop == 1 the Odd-tail branch unconditionally processes all 3 prefetch
// buffers, including two that contain K-data from neighbouring workgroups'
// K-partitions, causing an ~3x over-count in the accumulator.  The kernel
// must therefore reject configurations where:
//
//   num_loop = ceil(GemmK / (k_batch * KPerBlock)) < 4
//
// In the 2D bwd-weight tests below, GemmK = N * Ho * Wo.
// The V6 config tile has KPerBlock = 32.
// ============================================================================
struct TestConvConfigV6
{
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 8;
    static constexpr index_t VectorSizeC = 8;

    static constexpr index_t M_Tile = 256;
    static constexpr index_t N_Tile = 256;
    static constexpr index_t K_Tile = 32; // KPerBlock = 32

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 32;
    static constexpr index_t N_Warp_Tile = 32;
    static constexpr index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr GemmPipeline Pipeline    = GemmPipeline::COMPUTE_V6;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
};

// Helper to build full kernel type
template <typename PrecType,
          typename ConvConfig,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          index_t NDimSpatial = 2>
struct BuildKernel
{
    using GemmShape = TileGemmShape<
        sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
        sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
        sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile>>;

    using ConvTraits = GroupedConvTraits<NDimSpatial,
                                         ConvolutionSpecialization::Default,
                                         InLayout,
                                         WeiLayout,
                                         tuple<>,
                                         OutLayout,
                                         ConvConfig::VectorSizeA,
                                         ConvConfig::VectorSizeB,
                                         ConvConfig::VectorSizeC,
                                         ConvConfig::NumGroupsToMerge>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using GemmUniversalTraits =
        TileGemmUniversalTraits<ConvTraits::FixedGemmParams::kPadM,
                                ConvTraits::FixedGemmParams::kPadN,
                                ConvTraits::FixedGemmParams::kPadK,
                                ConvConfig::DoubleSmemBuffer,
                                typename ConvTraits::AsLayoutBwdWeight,
                                typename ConvTraits::BsLayoutBwdWeight,
                                typename ConvTraits::CLayoutBwdWeight,
                                ConvTraits::FixedGemmParams::TransposeC,
                                ConvTraits::FixedGemmParams::UseStructuredSparsity,
                                ConvTraits::FixedGemmParams::Persistent,
                                ConvConfig::NumWaveGroups>;

    using GemmPipelineProblem =
        GemmPipelineProblem<PrecType, // OutDataType (A in bwd weight)
                            PrecType, // InDataType (B in bwd weight)
                            float,    // AccDataType
                            GemmShape,
                            typename ConvTraits::template GroupedConvImplicitGemmTraitsBwdWeight<
                                ConvConfig::NumWaveGroups>,
                            element_wise::PassThrough,
                            element_wise::PassThrough,
                            PrecType, // WeiDataType (C in bwd weight)
                            PrecType,
                            ConvTraits::FixedGemmParams::FixedVectorSize,
                            ConvTraits::VectorSizeA,
                            ConvTraits::VectorSizeB>;

    using UniversalGemmProblem =
        UniversalGemmPipelineProblem<PrecType,
                                     PrecType,
                                     float,
                                     GemmShape,
                                     GemmUniversalTraits,
                                     ConvConfig::Scheduler,
                                     element_wise::PassThrough,
                                     element_wise::PassThrough,
                                     PrecType,
                                     PrecType,
                                     ConvTraits::FixedGemmParams::FixedVectorSize,
                                     ConvTraits::VectorSizeA,
                                     ConvTraits::VectorSizeB>;

    using GemmPipeline = std::conditional_t<ConvConfig::Pipeline == GemmPipeline::COMPUTE_V6,
                                            GemmPipelineAgBgCrCompV6<UniversalGemmProblem>,
                                            GemmPipelineAgBgCrCompV3<UniversalGemmProblem>>;

    using EpilogueProblem = CShuffleEpilogueProblem<PrecType,
                                                    PrecType,
                                                    tuple<>,
                                                    float,
                                                    PrecType,
                                                    typename ConvTraits::ImplicitGemmDsLayout,
                                                    typename ConvTraits::FixedGemmParams::ELayout,
                                                    element_wise::PassThrough,
                                                    TilePartitioner::MPerBlock,
                                                    TilePartitioner::NPerBlock,
                                                    ConvConfig::M_Warp,
                                                    ConvConfig::N_Warp,
                                                    ConvConfig::M_Warp_Tile,
                                                    ConvConfig::N_Warp_Tile,
                                                    ConvConfig::K_Warp_Tile,
                                                    ConvTraits::FixedGemmParams::TransposeC,
                                                    ConvConfig::NumWaveGroups,
                                                    ConvTraits::FixedGemmParams::FixedVectorSize,
                                                    ConvTraits::VectorSizeC>;

    using Epilogue = CShuffleEpilogue<EpilogueProblem>;

    using type =
        GroupedConvolutionBackwardWeightKernel<ConvTraits, TilePartitioner, GemmPipeline, Epilogue>;
};

// Helper to create 2D host args
static GroupedConvBwdWeightHostArgs create_2d_host_args(index_t G,
                                                        index_t N,
                                                        index_t K,
                                                        index_t C,
                                                        index_t Y,
                                                        index_t X,
                                                        index_t Hi,
                                                        index_t Wi,
                                                        index_t stride_y,
                                                        index_t stride_x,
                                                        index_t dilation_y,
                                                        index_t dilation_x,
                                                        index_t left_pad_y,
                                                        index_t left_pad_x,
                                                        index_t right_pad_y,
                                                        index_t right_pad_x,
                                                        index_t k_batch = 1)
{
    auto conv_param = conv::ConvParam{2,
                                      G,
                                      N,
                                      K,
                                      C,
                                      {Y, X},
                                      {Hi, Wi},
                                      {stride_y, stride_x},
                                      {dilation_y, dilation_x},
                                      {left_pad_y, left_pad_x},
                                      {right_pad_y, right_pad_x}};

    return GroupedConvBwdWeightHostArgs{conv_param, nullptr, nullptr, {}, nullptr, k_batch};
}

static GroupedConvBwdWeightHostArgs create_2d_host_args(index_t k_batch)
{
    return create_2d_host_args(2, 2, 8, 8, 3, 3, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, k_batch);
}

static GroupedConvBwdWeightHostArgs create_large_2d_host_args(index_t k_batch)
{
    return create_2d_host_args(2, 2, 8, 8, 3, 3, 70, 70, 1, 1, 1, 1, 1, 1, 1, 1, k_batch);
}

class GroupedConvBwdWeightIsSupportedArgumentTest : public ::testing::Test
{
};

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, ValidKBatch)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    auto host_args_kbatch_1 = create_2d_host_args(1);
    auto kargs_1 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_1);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_1));

    auto host_args_kbatch_4 = create_2d_host_args(4);
    auto kargs_4 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_4);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_4));
}

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, InvalidKBatchLessThanOne)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    auto host_args_kbatch_0 = create_2d_host_args(0);
    auto kargs = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_0);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, K0KBatchLimitation)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // k_batch = 128 should pass
    auto host_args_kbatch_6 = create_2d_host_args(7);
    auto kargs_6 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_6);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_6));

    // k_batch = 129 should fail for half_t output
    auto host_args_kbatch_7 = create_2d_host_args(8);
    auto kargs_7 = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_7);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs_7));
}

TEST_F(GroupedConvBwdWeightIsSupportedArgumentTest, NonFloatDoubleOutputLimitsKBatch)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfig,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // k_batch = 128 should pass
    auto host_args_kbatch_128 = create_large_2d_host_args(128);
    auto kargs_128 =
        typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_128);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_128));

    // k_batch = 129 should fail for half_t output
    auto host_args_kbatch_129 = create_large_2d_host_args(129);
    auto kargs_129 =
        typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_kbatch_129);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs_129));
}
class GroupedConvBwdWeightV6PipelineTest : public ::testing::Test
{
};

// KPerBlock=32. GemmK = N*Ho*Wo.
// For the default small args (N=2, Hi=Wi=7, Y=X=3, pad=1):
//   Ho = Wo = 7, GemmK = 2*7*7 = 98.
// With k_batch=1: num_loop = ceil(98/32) = 4  -> accepted.
// With k_batch=2: num_loop = ceil(98/64) = 2  -> rejected (< 4).
// With k_batch=3: num_loop = ceil(98/96) = 2  -> rejected.
// With k_batch=4: num_loop = ceil(98/128)= 1  -> rejected.

TEST_F(GroupedConvBwdWeightV6PipelineTest, AcceptsWhenNumLoopAtLeast4)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfigV6,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // k_batch=1: num_loop=ceil(98/32)=4.  Exactly on the boundary -> must pass.
    auto host_args = create_2d_host_args(1);
    auto kargs     = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs))
        << "V6 kernel must accept k_batch=1 (num_loop=4 >= 4)";
}

TEST_F(GroupedConvBwdWeightV6PipelineTest, RejectsWhenNumLoopIs2)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfigV6,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // k_batch=2: num_loop=ceil(98/64)=2 < 4. Must be rejected.
    auto host_args = create_2d_host_args(2);
    auto kargs     = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs))
        << "V6 kernel must reject k_batch=2 (num_loop=2 < 4) to avoid "
           "incorrect Odd-tail prefetch over-read";
}

TEST_F(GroupedConvBwdWeightV6PipelineTest, RejectsWhenNumLoopIs1)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfigV6,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // k_batch=4: num_loop=ceil(98/128)=1 < 4. Must be rejected.
    auto host_args = create_2d_host_args(4);
    auto kargs     = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs))
        << "V6 kernel must reject k_batch=4 (num_loop=1 < 4)";
}

TEST_F(GroupedConvBwdWeightV6PipelineTest, AcceptsLargeSpatialWithSmallKBatch)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfigV6,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // Large spatial: N=2, Hi=Wi=70 -> Ho=Wo=70, GemmK=2*70*70=9800.
    // k_batch=1: num_loop=ceil(9800/32)=307 >= 4 -> accepted.
    auto host_args = create_large_2d_host_args(1);
    auto kargs     = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs))
        << "V6 kernel must accept large spatial (num_loop=307) with k_batch=1";
}

TEST_F(GroupedConvBwdWeightV6PipelineTest, RejectsLargeSpatialWithLargeKBatch)
{
    using Kernel = typename BuildKernel<half_t,
                                        TestConvConfigV6,
                                        tensor_layout::convolution::NHWGC,
                                        tensor_layout::convolution::GKYXC,
                                        tensor_layout::convolution::NHWGK>::type;

    // GemmK=9800. With k_batch=64: num_loop=ceil(9800/(64*32))=ceil(9800/2048)=5 >= 4 -> pass.
    // With k_batch=128: num_loop=ceil(9800/4096)=3 < 4 -> rejected.
    auto host_args_pass = create_large_2d_host_args(64);
    auto kargs_pass = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_pass);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs_pass))
        << "V6 kernel must accept k_batch=64 (num_loop=5 >= 4)";

    auto host_args_fail = create_large_2d_host_args(128);
    auto kargs_fail = typename Kernel::GroupedConvBwdWeightKernelArgsSpecialized(host_args_fail);
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs_fail))
        << "V6 kernel must reject k_batch=128 (num_loop=3 < 4)";
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/grouped_conv_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"

using namespace ck_tile;

// ============================================================================
// Minimal conv config with unit vector sizes (VectorSizeA/B/C = 1) and
// BASIC_V1 pipeline, mirroring the depthwise-like instances that appear in
// the tile dispatcher and triggered the split-image bug.
// ============================================================================
struct TestFwdConvConfigBasicV1UnitVec
{
    static constexpr index_t VectorSizeA = 1;
    static constexpr index_t VectorSizeB = 1;
    static constexpr index_t VectorSizeC = 1;

    static constexpr index_t M_Tile = 16;
    static constexpr index_t N_Tile = 64;
    static constexpr index_t K_Tile = 64;

    static constexpr index_t M_Warp = 1;
    static constexpr index_t N_Warp = 4;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 16;
    static constexpr index_t N_Warp_Tile = 16;
    static constexpr index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr GemmPipeline Pipeline    = GemmPipeline::BASIC_V1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
};

// Standard config with larger vector sizes (passes all checks for normal problem sizes)
struct TestFwdConvConfigStandard
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
    static constexpr GemmPipeline Pipeline    = GemmPipeline::BASIC_V1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
};

// ============================================================================
// Helper to assemble the full forward kernel type.
// EnableSplitImage_ corresponds to the compile-time flag on GroupedConvTraits.
// ============================================================================
template <typename PrecType,
          typename ConvConfig,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          index_t NDimSpatial   = 2,
          bool EnableSplitImage = false>
struct BuildFwdKernel
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
                                         ConvConfig::NumGroupsToMerge,
                                         EnableSplitImage>;

    using TilePartitioner =
        GemmSpatiallyLocalTilePartitioner<GemmShape,
                                          ConvTraits::FixedGemmParams::TilePartitionerGroupNum,
                                          ConvTraits::FixedGemmParams::TilePartitionerM01>;

    using GemmUniversalTraits =
        TileGemmUniversalTraits<ConvTraits::FixedGemmParams::kPadM,
                                ConvTraits::FixedGemmParams::kPadN,
                                ConvTraits::FixedGemmParams::kPadK,
                                ConvConfig::DoubleSmemBuffer,
                                typename ConvTraits::AsLayoutFwd,
                                typename ConvTraits::BsLayoutFwd,
                                typename ConvTraits::CLayoutFwd,
                                ConvTraits::FixedGemmParams::TransposeC,
                                ConvTraits::FixedGemmParams::UseStructuredSparsity,
                                ConvTraits::FixedGemmParams::Persistent,
                                ConvConfig::NumWaveGroups>;

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

    using GemmPipeline = GemmPipelineAGmemBGmemCRegV1<UniversalGemmProblem,
                                                      GroupedConvUniversalPipelineAgBgCrPolicy>;

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
        GroupedConvolutionForwardKernel<ConvTraits, TilePartitioner, GemmPipeline, Epilogue>;
};

// ============================================================================
// Helper to create 2D forward host args (null device pointers, host-only).
// ============================================================================
static GroupedConvFwdHostArgs<> create_2d_fwd_host_args(index_t G,
                                                        index_t N,
                                                        index_t K,
                                                        index_t C,
                                                        index_t Y,
                                                        index_t X,
                                                        index_t Hi,
                                                        index_t Wi,
                                                        index_t stride_y   = 1,
                                                        index_t stride_x   = 1,
                                                        index_t dilation_y = 1,
                                                        index_t dilation_x = 1,
                                                        index_t lpad_y     = 0,
                                                        index_t lpad_x     = 0,
                                                        index_t rpad_y     = 0,
                                                        index_t rpad_x     = 0,
                                                        index_t k_batch    = 1)
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
                                      {lpad_y, lpad_x},
                                      {rpad_y, rpad_x}};

    return GroupedConvFwdHostArgs<>{conv_param, nullptr, nullptr, {}, nullptr, k_batch};
}

// ============================================================================
// Tests
// ============================================================================

class GroupedConvFwdIsSupportedArgumentTest : public ::testing::Test
{
};

// ---------------------------------------------------------------------------
// Split-image default initialization in MakeKernelArgs (full-image path)
// ---------------------------------------------------------------------------
// MakeKernelArgs() initializes split_image.pieces[0] as a single piece covering
// the entire output. This ensures the split-image kernel path works correctly
// even without the large-tensor invoker. The invoker can override with
// multi-piece data for large tensors.

// MakeKernelArgs initializes pieces so split-image is accepted for any valid problem.
TEST_F(GroupedConvFwdIsSupportedArgumentTest, SplitImageFullImageAfterMakeKernelArgs)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestFwdConvConfigBasicV1UnitVec,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,
                                           true /*EnableSplitImage*/>::type;

    // K=64, C=64 - MakeKernelArgs should set up a single-piece split_image.
    // 3x3 filter, stride 1, no padding => output H/W = 5x5.
    auto host_args = create_2d_fwd_host_args(1, 2, 64, 64, 3, 3, 7, 7);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Verify piece[0] was initialized by MakeKernelArgs (full-image, single piece).
    EXPECT_EQ(kargs.num_spatial_pieces, 1);
    EXPECT_EQ(kargs.split_image.pieces[0].block_start, 0);
    EXPECT_GT(kargs.split_image.pieces[0].block_end, 0)
        << "MakeKernelArgs must initialize split_image.pieces[0]";
    EXPECT_EQ(kargs.split_image.pieces[0].h_start, 0);
    EXPECT_EQ(kargs.split_image.pieces[0].w_start, 0);
    EXPECT_EQ(kargs.split_image.pieces[0].h_size, 5) << "Output H = (7 - 3)/1 + 1 = 5";
    EXPECT_EQ(kargs.split_image.pieces[0].w_size, 5) << "Output W = (7 - 3)/1 + 1 = 5";

    // Unused pieces retain the sentinel default (-1) from PieceInfo.
    EXPECT_EQ(kargs.split_image.pieces[1].block_start, -1)
        << "Unused pieces must retain sentinel default";
    EXPECT_EQ(kargs.split_image.pieces[1].block_end, -1)
        << "Unused pieces must retain sentinel default";

    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs))
        << "Split-image instance must be accepted after MakeKernelArgs initializes pieces";
}

// Split-image with depthwise-like K=1 problem is also accepted (pieces are initialized).
TEST_F(GroupedConvFwdIsSupportedArgumentTest, SplitImageFullImageSmallK)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestFwdConvConfigBasicV1UnitVec,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,
                                           true /*EnableSplitImage*/>::type;

    // K=1, C=1
    auto host_args = create_2d_fwd_host_args(1, 2, 1, 1, 3, 3, 7, 7);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    EXPECT_EQ(kargs.num_spatial_pieces, 1);
    EXPECT_EQ(kargs.split_image.pieces[0].block_start, 0);
    EXPECT_GT(kargs.split_image.pieces[0].block_end, 0)
        << "MakeKernelArgs must initialize pieces even for small K";
}

// Large K is accepted with proper initialization.
TEST_F(GroupedConvFwdIsSupportedArgumentTest, SplitImageFullImageLargeK)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestFwdConvConfigBasicV1UnitVec,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,
                                           true /*EnableSplitImage*/>::type;

    // K=96 - the case that caused flaky failures
    // 1x1 filter, stride 1, no padding => output H/W = 73x128.
    auto host_args = create_2d_fwd_host_args(3, 5, 96, 200, 1, 1, 73, 128);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    EXPECT_EQ(kargs.num_spatial_pieces, 1);
    EXPECT_EQ(kargs.split_image.pieces[0].block_start, 0);
    EXPECT_GT(kargs.split_image.pieces[0].block_end, 0)
        << "MakeKernelArgs must initialize pieces for K=96";
    EXPECT_EQ(kargs.split_image.pieces[0].h_size, 73);
    EXPECT_EQ(kargs.split_image.pieces[0].w_size, 128);
}

// ---------------------------------------------------------------------------
// Multi-piece split-image path (large-tensor invoker)
// ---------------------------------------------------------------------------
// The large-tensor invoker (grouped_convolution_forward_large_tensor_invoker.hpp in the examples
// code) calls MakeKernelArgs() first, then overrides split_image.pieces[] with multi-piece data
// computed by calculate_spatial_piece<TilePartitioner>().

TEST_F(GroupedConvFwdIsSupportedArgumentTest, SplitImageMultiPieceInvokerOverride)
{
    using Build           = BuildFwdKernel<half_t,
                                           TestFwdConvConfigBasicV1UnitVec,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,
                                           true /*EnableSplitImage*/>;
    using Kernel          = typename Build::type;
    using TilePartitioner = typename Build::TilePartitioner;

    // Large problem: G=1, N=4, K=64, C=64, filter=1x1, input=128x128
    // Output = 128x128. Split H into 2 pieces: H=[0..64) and H=[64..128).
    const index_t G = 1, N = 4, K = 64, C = 64;
    const index_t Hi = 128, Wi = 128;
    const index_t Ho = 128, Wo = 128; // 1x1 filter, stride 1, no padding

    auto host_args = create_2d_fwd_host_args(G, N, K, C, 1, 1, Hi, Wi);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Verify MakeKernelArgs set up the default single-piece first.
    ASSERT_EQ(kargs.num_spatial_pieces, 1);
    ASSERT_GT(kargs.split_image.pieces[0].block_end, 0);

    // Now simulate what the invoker does: split H into 2 pieces.
    const index_t num_h_pieces = 2;
    const index_t num_w_pieces = 1;
    const index_t num_d_pieces = 1;
    const index_t total_pieces = num_d_pieces * num_h_pieces * num_w_pieces;
    const index_t base_piece_h = Ho / num_h_pieces; // 64
    const index_t base_piece_w = Wo;                // 128
    const index_t base_piece_d = 1;

    index_t total_blocks = 0;
    std::array<SplitImagePieceInfo, 64> temp_pieces{};
    for(index_t piece = 0; piece < total_pieces; piece++)
    {
        temp_pieces[piece] = calculate_spatial_piece<TilePartitioner>(piece,
                                                                      num_d_pieces,
                                                                      num_h_pieces,
                                                                      num_w_pieces,
                                                                      base_piece_d,
                                                                      base_piece_h,
                                                                      base_piece_w,
                                                                      1, // total_d
                                                                      Ho,
                                                                      Wo,
                                                                      N,
                                                                      K,
                                                                      total_blocks);
        total_blocks       = temp_pieces[piece].block_end;
    }

    // Override kargs with multi-piece data.
    kargs.num_spatial_pieces       = total_pieces;
    kargs.split_image.num_h_pieces = num_h_pieces;
    kargs.split_image.num_w_pieces = num_w_pieces;
    kargs.split_image.num_d_pieces = num_d_pieces;
    for(index_t i = 0; i < total_pieces; i++)
    {
        kargs.split_image.pieces[i] = {temp_pieces[i].block_start,
                                       temp_pieces[i].block_end,
                                       temp_pieces[i].d_start,
                                       temp_pieces[i].h_start,
                                       temp_pieces[i].w_start,
                                       temp_pieces[i].d_size,
                                       temp_pieces[i].h_size,
                                       temp_pieces[i].w_size};
    }

    // Verify piece 0: covers H=[0..64), W=[0..128)
    EXPECT_EQ(kargs.split_image.pieces[0].block_start, 0);
    EXPECT_GT(kargs.split_image.pieces[0].block_end, 0);
    EXPECT_EQ(kargs.split_image.pieces[0].h_start, 0);
    EXPECT_EQ(kargs.split_image.pieces[0].h_size, 64);
    EXPECT_EQ(kargs.split_image.pieces[0].w_start, 0);
    EXPECT_EQ(kargs.split_image.pieces[0].w_size, 128);

    // Verify piece 1: covers H=[64..128), W=[0..128)
    EXPECT_EQ(kargs.split_image.pieces[1].block_start, kargs.split_image.pieces[0].block_end);
    EXPECT_GT(kargs.split_image.pieces[1].block_end, kargs.split_image.pieces[1].block_start);
    EXPECT_EQ(kargs.split_image.pieces[1].h_start, 64);
    EXPECT_EQ(kargs.split_image.pieces[1].h_size, 64);
    EXPECT_EQ(kargs.split_image.pieces[1].w_start, 0);
    EXPECT_EQ(kargs.split_image.pieces[1].w_size, 128);

    // Pieces must be contiguous: piece1.block_start == piece0.block_end
    EXPECT_EQ(kargs.split_image.pieces[1].block_start, kargs.split_image.pieces[0].block_end);

    // Total blocks across pieces must equal the full grid size.
    const index_t full_grid = TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN);
    EXPECT_EQ(kargs.split_image.pieces[total_pieces - 1].block_end, full_grid);

    EXPECT_EQ(kargs.num_spatial_pieces, 2);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs))
        << "Split-image instance must be accepted after invoker populates multi-piece data";
}

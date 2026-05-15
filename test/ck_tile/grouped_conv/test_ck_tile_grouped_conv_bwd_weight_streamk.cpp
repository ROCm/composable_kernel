// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp"
#include "ck_tile/host/convolution_host_tensor_descriptor_helper.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/fill.hpp"

using namespace ck_tile;

struct StreamKTestConvConfig
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

// Build a conv bwd weight kernel type from a tile partitioner.
// Works for both StreamK and Split-K partitioners.
template <typename PrecType,
          typename ConvConfig,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TilePartitioner_,
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

    using GemmPipeline_ = GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using EpilogueProblem = CShuffleEpilogueProblem<PrecType,
                                                    PrecType,
                                                    tuple<>,
                                                    float,
                                                    PrecType,
                                                    typename ConvTraits::ImplicitGemmDsLayout,
                                                    typename ConvTraits::FixedGemmParams::ELayout,
                                                    element_wise::PassThrough,
                                                    TilePartitioner_::MPerBlock,
                                                    TilePartitioner_::NPerBlock,
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

    using type = GroupedConvolutionBackwardWeightKernel<ConvTraits,
                                                        TilePartitioner_,
                                                        GemmPipeline_,
                                                        Epilogue>;
};

// Helper to create 2D host args
static GroupedConvBwdWeightHostArgs create_host_args(index_t G,
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

// Common type aliases
using InLayout  = tensor_layout::convolution::NHWGC;
using WeiLayout = tensor_layout::convolution::GKYXC;
using OutLayout = tensor_layout::convolution::NHWGK;
using PrecType  = half_t;

using TestGemmShape =
    TileGemmShape<sequence<128, 128, 32>, sequence<2, 2, 1>, sequence<16, 16, 16>>;

using SplitKPartitioner = GemmSpatiallyLocalTilePartitioner<TestGemmShape, 8, 4>;
using LinearPartitioner =
    StreamKTilePartitioner<TestGemmShape, StreamKReductionStrategy::Linear, false>;
using TreePartitioner =
    StreamKTilePartitioner<TestGemmShape, StreamKReductionStrategy::Tree, false>;
using LinearPersistentPartitioner =
    StreamKTilePartitioner<TestGemmShape, StreamKReductionStrategy::Linear, true>;
using TreePersistentPartitioner =
    StreamKTilePartitioner<TestGemmShape, StreamKReductionStrategy::Tree, true>;

template <typename Partitioner>
using TestKernel = typename BuildKernel<PrecType,
                                        StreamKTestConvConfig,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        Partitioner>::type;

// ============================================================================
// Host-side unit tests
// ============================================================================

TEST(StreamKConvBwdWeight, TypeTraitDetection)
{
    EXPECT_FALSE(is_streamk_partitioner<SplitKPartitioner>::value);
    EXPECT_TRUE(is_streamk_partitioner<LinearPartitioner>::value);
    EXPECT_TRUE(is_streamk_partitioner<TreePartitioner>::value);
}

TEST(StreamKConvBwdWeight, KernelArgsConstruction_LinearPartitioner)
{
    using Kernel = TestKernel<LinearPartitioner>;
    EXPECT_TRUE(Kernel::IsStreamK);

    auto host_args = create_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    auto kargs     = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    EXPECT_EQ(kargs.k_batch, 1);
    EXPECT_GT(kargs.GemmM, 0);
    EXPECT_GT(kargs.GemmN, 0);
    EXPECT_GT(kargs.GemmK, 0);
    EXPECT_GT(kargs.tile_partitioner.get_max_active_wgs(), 0);
}

TEST(StreamKConvBwdWeight, KernelArgsConstruction_TreePartitioner)
{
    using Kernel = TestKernel<TreePartitioner>;
    EXPECT_TRUE(Kernel::IsStreamK);

    auto host_args = create_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    auto kargs     = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    EXPECT_EQ(kargs.k_batch, 1);
    EXPECT_GT(kargs.GemmM, 0);
    EXPECT_GT(kargs.GemmN, 0);
    EXPECT_GT(kargs.GemmK, 0);
    EXPECT_GT(kargs.tile_partitioner.get_max_active_wgs(), 0);
}

TEST(StreamKConvBwdWeight, GridSize)
{
    using Kernel = TestKernel<LinearPartitioner>;

    auto host_args = create_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    auto kargs     = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);
    auto grid      = Kernel::GridSize(kargs);

    auto sk_grid = kargs.tile_partitioner.grid_size();
    EXPECT_EQ(grid.x, sk_grid.x);
    EXPECT_EQ(grid.y, static_cast<unsigned int>(kargs.GemmBatch));
    EXPECT_EQ(grid.z, 1u);
}

TEST(StreamKConvBwdWeight, WorkSpaceSize)
{
    using Kernel = TestKernel<LinearPartitioner>;

    auto host_args = create_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    auto kargs     = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    EXPECT_GT(Kernel::GetWorkSpaceSize(kargs), 0);
}

TEST(StreamKConvBwdWeight, SplitKNoWorkspace)
{
    using Kernel = TestKernel<SplitKPartitioner>;
    EXPECT_FALSE(Kernel::IsStreamK);

    auto host_args = create_host_args(1, 4, 128, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    EXPECT_EQ(Kernel::GetWorkSpaceSize(kargs), 0);
}

// ============================================================================
// GPU end-to-end tests: StreamK vs Split-K=1 reference
// ============================================================================

template <typename StreamKKernelType>
static bool run_streamk_vs_splitk_test(index_t G,
                                       index_t N,
                                       index_t K,
                                       index_t C,
                                       index_t Y,
                                       index_t X,
                                       index_t Hi,
                                       index_t Wi,
                                       index_t num_cu,
                                       index_t occupancy,
                                       index_t stride_h   = 1,
                                       index_t stride_w   = 1,
                                       index_t dilation_h = 1,
                                       index_t dilation_w = 1,
                                       index_t lpad_h     = 1,
                                       index_t lpad_w     = 1,
                                       index_t rpad_h     = 1,
                                       index_t rpad_w     = 1)
{
    using RefKernel               = TestKernel<SplitKPartitioner>;
    constexpr index_t NDimSpatial = 2;

    auto conv_param = conv::ConvParam{NDimSpatial,
                                      G,
                                      N,
                                      K,
                                      C,
                                      {Y, X},
                                      {Hi, Wi},
                                      {stride_h, stride_w},
                                      {dilation_h, dilation_w},
                                      {lpad_h, lpad_w},
                                      {rpad_h, rpad_w}};

    const auto in_desc =
        conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto wei_desc =
        conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_desc =
        conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    HostTensor<PrecType> input(in_desc);
    HostTensor<PrecType> output(out_desc);
    HostTensor<PrecType> weight_ref(wei_desc);
    HostTensor<PrecType> weight_streamk(wei_desc);

    FillUniformDistribution<PrecType>{-1.f, 1.f}(input);
    FillUniformDistribution<PrecType>{-1.f, 1.f}(output);

    DeviceMem input_dev(input.get_element_space_size_in_bytes());
    DeviceMem output_dev(output.get_element_space_size_in_bytes());
    DeviceMem weight_ref_dev(weight_ref.get_element_space_size_in_bytes());
    DeviceMem weight_streamk_dev(weight_streamk.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    output_dev.ToDevice(output.data());

    // Reference: Split-K=1
    {
        weight_ref_dev.SetZero();

        GroupedConvBwdWeightHostArgs host_args(conv_param,
                                               input_dev.GetDeviceBuffer(),
                                               weight_ref_dev.GetDeviceBuffer(),
                                               {},
                                               output_dev.GetDeviceBuffer(),
                                               /*k_batch=*/1);

        auto kargs = RefKernel::MakeKernelArgs(host_args);
        if(!RefKernel::IsSupportedArgument(kargs))
        {
            std::cout << "Split-K kernel does not support this shape, skipping\n";
            return true;
        }

        auto kernel_func = make_kernel<1>(
            RefKernel{}, RefKernel::GridSize(kargs), RefKernel::BlockSize(), 0, kargs);
        launch_kernel(stream_config{nullptr, false}, kernel_func);
        hip_check_error(hipDeviceSynchronize());
    }

    // StreamK under test
    {
        weight_streamk_dev.SetZero();

        GroupedConvBwdWeightHostArgs host_args(conv_param,
                                               input_dev.GetDeviceBuffer(),
                                               weight_streamk_dev.GetDeviceBuffer(),
                                               {},
                                               output_dev.GetDeviceBuffer(),
                                               /*k_batch=*/1);

        auto kargs   = StreamKKernelType::MakeKernelArgs(host_args, num_cu, occupancy);
        auto ws_size = StreamKKernelType::GetWorkSpaceSize(kargs);
        DeviceMem workspace_dev(ws_size);
        workspace_dev.SetZero();
        StreamKKernelType::SetWorkSpacePointer(kargs, workspace_dev.GetDeviceBuffer());

        auto kernel_func = make_kernel<1>(StreamKKernelType{},
                                          StreamKKernelType::GridSize(kargs),
                                          StreamKKernelType::BlockSize(),
                                          0,
                                          kargs);
        launch_kernel(stream_config{nullptr, false}, kernel_func);
        hip_check_error(hipDeviceSynchronize());
    }

    weight_ref_dev.FromDevice(weight_ref.data());
    weight_streamk_dev.FromDevice(weight_streamk.data());

    // Compute GemmK = N * product(output_spatial_lengths) for bwd weight
    const index_t GemmK = N * std::accumulate(conv_param.output_spatial_lengths_.begin(),
                                              conv_param.output_spatial_lengths_.end(),
                                              static_cast<index_t>(1),
                                              std::multiplies<index_t>());

    // Max accumulated value calibrates atol to the output's ULP scale.
    const float max_accumulated_value =
        *std::max_element(weight_ref.mData.begin(), weight_ref.mData.end());

    // Tolerance follows the calculate_rtol_atol pattern from conv examples:
    // (1) GEMM accumulation error: fp16 compute, fp16 output, f32 accumulator
    // (2) Reduction error: accounts for fp16 output quantization differences
    //     when two f32 results (from different accumulation orders) round to fp16
    using ComputeType        = PrecType;
    using AccType            = float;
    constexpr index_t kbatch = 1;
    const auto rtol_gemm =
        get_relative_threshold<ComputeType, PrecType, AccType>(integer_divide_ceil(GemmK, kbatch));
    const auto atol_gemm = get_absolute_threshold<ComputeType, PrecType, AccType>(
        max_accumulated_value / kbatch, integer_divide_ceil(GemmK, kbatch));
    const auto rtol_reduction = get_relative_threshold<PrecType, PrecType, PrecType>(kbatch);
    const auto atol_reduction =
        get_absolute_threshold<PrecType, PrecType, PrecType>(max_accumulated_value, kbatch);

    const double rtol = std::max(rtol_gemm, rtol_reduction);
    const double atol = std::max(atol_gemm, atol_reduction);

    return check_err(weight_streamk, weight_ref, "StreamK vs SplitK mismatch", rtol, atol);
}

// Linear Reduction
TEST(StreamKConvBwdWeight, Linear_EndToEnd_SmallShape)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 2, 1)));
}

TEST(StreamKConvBwdWeight, Linear_EndToEnd_MediumShape)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(
        1, 8, 256, 128, 3, 3, 16, 16, 4, 1)));
}

TEST(StreamKConvBwdWeight, Linear_EndToEnd_MoreSKWork)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 4, 1)));
}

TEST(StreamKConvBwdWeight, Linear_EndToEnd_MultiGroup)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(
        2, 4, 128, 128, 3, 3, 16, 16, 4, 1)));
}

// Tree Reduction
TEST(StreamKConvBwdWeight, Tree_EndToEnd_SmallShape)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 2, 1)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_MediumShape)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(
        1, 8, 256, 128, 3, 3, 16, 16, 4, 1)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_MoreSKWork)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 4, 1)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_MultiGroup)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(
        2, 4, 128, 128, 3, 3, 16, 16, 4, 1)));
}

// Stride > 1 — shrinks Ho/Wo, changing the K/tile ratio and DP/SK split.
// Hi=16, Wi=16, 3x3 filter, stride=2, pad=1 → Ho=Wo=8, GemmK=N*64
TEST(StreamKConvBwdWeight, Linear_EndToEnd_Stride2)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(1,
                                                                           4,
                                                                           128,
                                                                           128,
                                                                           3,
                                                                           3,
                                                                           16,
                                                                           16,
                                                                           4,
                                                                           1,
                                                                           /*stride=*/2,
                                                                           2,
                                                                           /*dil=*/1,
                                                                           1,
                                                                           /*pad=*/1,
                                                                           1,
                                                                           1,
                                                                           1)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_Stride2)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(1,
                                                                         4,
                                                                         128,
                                                                         128,
                                                                         3,
                                                                         3,
                                                                         16,
                                                                         16,
                                                                         4,
                                                                         1,
                                                                         /*stride=*/2,
                                                                         2,
                                                                         /*dil=*/1,
                                                                         1,
                                                                         /*pad=*/1,
                                                                         1,
                                                                         1,
                                                                         1)));
}

// Pure DP — num_tiles evenly divides grid, so sk_ctas=0.
// K=256, C=128, 3x3 → GemmM=256, GemmN=1152 → tiles=2*9=18, grid=3*1=3, 18%3=0
TEST(StreamKConvBwdWeight, Linear_EndToEnd_PureDP)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(
        1, 4, 256, 128, 3, 3, 16, 16, 3, 1)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_PureDP)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(
        1, 4, 256, 128, 3, 3, 16, 16, 3, 1)));
}

// Single output tile — all work is SK, zero DP tiles.
// K=128, C=128, 1x1 filter, stride=1, pad=0 → GemmM=128, GemmN=128, tiles=1
TEST(StreamKConvBwdWeight, Linear_EndToEnd_SingleTile)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(1,
                                                                           4,
                                                                           128,
                                                                           128,
                                                                           1,
                                                                           1,
                                                                           16,
                                                                           16,
                                                                           4,
                                                                           1,
                                                                           /*stride=*/1,
                                                                           1,
                                                                           /*dil=*/1,
                                                                           1,
                                                                           /*pad=*/0,
                                                                           0,
                                                                           0,
                                                                           0)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_SingleTile)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(1,
                                                                         4,
                                                                         128,
                                                                         128,
                                                                         1,
                                                                         1,
                                                                         16,
                                                                         16,
                                                                         4,
                                                                         1,
                                                                         /*stride=*/1,
                                                                         1,
                                                                         /*dil=*/1,
                                                                         1,
                                                                         /*pad=*/0,
                                                                         0,
                                                                         0,
                                                                         0)));
}

// Large N — GemmK = 32*16*16 = 8192, many K iterations per tile.
TEST(StreamKConvBwdWeight, Linear_EndToEnd_LargeN)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(
        1, 32, 128, 128, 3, 3, 16, 16, 4, 1)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_LargeN)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(
        1, 32, 128, 128, 3, 3, 16, 16, 4, 1)));
}

// Higher occupancy — doubles the grid, more SK CTAs share tiles.
TEST(StreamKConvBwdWeight, Linear_EndToEnd_HigherOccupancy)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 4, 2)));
}

TEST(StreamKConvBwdWeight, Tree_EndToEnd_HigherOccupancy)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 4, 2)));
}

// Persistent DP — workgroups loop over DP tiles, then do SK work.
TEST(StreamKConvBwdWeight, LinearPersistent_EndToEnd_SmallShape)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPersistentPartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 2, 1)));
}

TEST(StreamKConvBwdWeight, TreePersistent_EndToEnd_SmallShape)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePersistentPartitioner>>(
        1, 4, 128, 128, 3, 3, 16, 16, 2, 1)));
}

TEST(StreamKConvBwdWeight, LinearPersistent_EndToEnd_MultiGroup)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<LinearPersistentPartitioner>>(
        2, 4, 128, 128, 3, 3, 16, 16, 4, 1)));
}

TEST(StreamKConvBwdWeight, TreePersistent_EndToEnd_MultiGroup)
{
    EXPECT_TRUE((run_streamk_vs_splitk_test<TestKernel<TreePersistentPartitioner>>(
        2, 4, 128, 128, 3, 3, 16, 16, 4, 1)));
}

// ============================================================================
// Negative tests: IsSupportedArgument should reject invalid shapes
// ============================================================================

// C not divisible by VectorSizeB (=8) → rejected
TEST(StreamKConvBwdWeight, IsSupportedArgument_RejectsUnalignedC)
{
    using Kernel = TestKernel<LinearPartitioner>;

    auto host_args = create_host_args(1, 4, 128, 100, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    auto kargs     = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

// K not divisible by VectorSizeA (=4) → rejected
TEST(StreamKConvBwdWeight, IsSupportedArgument_RejectsUnalignedK)
{
    using Kernel = TestKernel<TreePartitioner>;

    auto host_args = create_host_args(1, 4, 103, 128, 3, 3, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    auto kargs     = Kernel::MakeKernelArgs(host_args, /*num_cu=*/4, /*occupancy=*/1);

    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs));
}

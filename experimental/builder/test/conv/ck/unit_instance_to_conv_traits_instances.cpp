// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ============================================================================
// Unit Tests for Complete Device Instance Transformations
// ============================================================================
//
// PURPOSE:
// --------
// These tests verify the complete instance_to_conv_traits transformation
// for entire Device class templates. Each test validates that all traits
// are correctly extracted from a specific Device class instantiation.
//
// TEST COVERAGE:
// --------------
// Complete transformation verification for each XDL Device class template:
// 1. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
// 2. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
// 3. DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor
//
// Each test verifies:
// - Spatial dimension extraction
// - Convolution direction
// - Data type detection
// - GEMM padding configuration
// - Tile dimensions (M, N, K per block)
// - Pipeline scheduler and version
// ============================================================================

#include <gtest/gtest.h>

#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck_tile/builder/reflect/instance_to_conv_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp>

namespace {

using ::ck_tile::builder::ConvDirection;
using ::ck_tile::builder::DataType;
using ::ck_tile::builder::GemmPadding;
using ::ck_tile::builder::PipelineScheduler;
using ::ck_tile::builder::PipelineVersion;

// ============================================================================
// Comprehensive Transformation Tests - Per Device Class Template
// ============================================================================
// These tests verify the complete InstanceTraits -> ConvTraits transformation
// for each forward convolution Device class template.
// ============================================================================

TEST(InstanceToConvTraits, TransformsFwdMultipleAbdXdlCShuffleV3)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,                                       // BlockSize
            128,                                       // MPerBlock
            128,                                       // NPerBlock
            16,                                        // KPerBlock
            8,                                         // AK1
            8,                                         // BK1
            32,                                        // MPerXDL
            32,                                        // NPerXDL
            4,                                         // MXdlPerWave
            4,                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>,                    // ABlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,                     // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,                     // ABlockTransferSrcAccessOrder
            2,                                         // ABlockTransferSrcVectorDim
            8,                                         // ABlockTransferSrcScalarPerVector
            8,                                         // ABlockTransferDstScalarPerVector_AK1
            1,                                         // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>,                    // BBlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,                     // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,                     // BBlockTransferSrcAccessOrder
            2,                                         // BBlockTransferSrcVectorDim
            8,                                         // BBlockTransferSrcScalarPerVector
            8,                                         // BBlockTransferDstScalarPerVector_BK1
            1,                                         // BBlockLdsExtraN
            1,                                         // CShuffleMXdlPerWavePerShuffle
            1,                                         // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1, 32, 1, 8>,                 // CDEBlockTransferClusterLengths
            8,                                         // CDEBlockTransferScalarPerVector_NPerBlock
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t,                                // BComputeDataType
            false,                                     // DirectLoad
            1>;                                        // NumGroupsToMerge

    using InstTraits  = ck_tile::reflect::InstanceTraits<DeviceInstance>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, InstTraits::kSpatialDim);
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.gemm_padding, GemmPadding::DEFAULT);
    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, InstTraits::kMPerBlock);
    EXPECT_EQ(traits.tile_dims.n, InstTraits::kNPerBlock);
    EXPECT_EQ(traits.tile_dims.k, InstTraits::kKPerBlock);
    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::INTRAWAVE);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);
}

TEST(InstanceToConvTraits, TransformsFwdMultipleAbdXdlCShuffle)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            1,                          // NumGemmKPrefetchStage
            256,                        // BlockSize
            128,                        // MPerBlock
            128,                        // NPerBlock
            16,                         // KPerBlock
            8,                          // AK1
            8,                          // BK1
            32,                         // MPerXDL
            32,                         // NPerXDL
            4,                          // MXdlPerWave
            4,                          // NXdlPerWave
            ck::Sequence<4, 64, 1>,     // ABlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,      // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,      // ABlockTransferSrcAccessOrder
            2,                          // ABlockTransferSrcVectorDim
            8,                          // ABlockTransferSrcScalarPerVector
            8,                          // ABlockTransferDstScalarPerVector_AK1
            1,                          // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>,     // BBlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,      // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,      // BBlockTransferSrcAccessOrder
            2,                          // BBlockTransferSrcVectorDim
            8,                          // BBlockTransferSrcScalarPerVector
            8,                          // BBlockTransferDstScalarPerVector_BK1
            1,                          // BBlockLdsExtraN
            1,                          // CShuffleMXdlPerWavePerShuffle
            1,                          // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1, 32, 1, 8>,  // CDEBlockTransferClusterLengths
            8,                          // CDEBlockTransferScalarPerVector_NPerBlock
            ck::half_t,                 // AComputeDataType
            ck::half_t,                 // BComputeDataType
            ck::LoopScheduler::Default, // LoopSched
            1>;                         // NumGroupsToMerge

    using InstTraits  = ck_tile::reflect::InstanceTraits<DeviceInstance>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, InstTraits::kSpatialDim);
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.gemm_padding, GemmPadding::DEFAULT);
    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, InstTraits::kMPerBlock);
    EXPECT_EQ(traits.tile_dims.n, InstTraits::kNPerBlock);
    EXPECT_EQ(traits.tile_dims.k, InstTraits::kKPerBlock);
    // Verify pipeline configuration (uses LoopScheduler instead of BlockGemmPipelineScheduler)
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);
}

TEST(InstanceToConvTraits, TransformsFwdMultipleDXdlLargeTensor)
{
    using DeviceInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
            2,                                               // NDimSpatial
            ck::tensor_layout::convolution::GNHWC,           // ALayout
            ck::tensor_layout::convolution::GKYXC,           // BLayout
            ck::Tuple<>,                                     // DsLayout
            ck::tensor_layout::convolution::GNHWK,           // ELayout
            ck::half_t,                                      // ADataType
            ck::half_t,                                      // BDataType
            float,                                           // AccDataType
            ck::half_t,                                      // CShuffleDataType
            ck::Tuple<>,                                     // DsDataType
            ck::half_t,                                      // EDataType
            ck::tensor_operation::element_wise::PassThrough, // AElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // BElementwiseOperation
            ck::tensor_operation::element_wise::PassThrough, // CDEElementwiseOperation
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::Default,
            1,                           // NumGemmKPrefetchStage
            256,                         // BlockSize
            128,                         // MPerBlock
            128,                         // NPerBlock
            16,                          // KPerBlock
            8,                           // AK1
            8,                           // BK1
            32,                          // MPerXDL
            32,                          // NPerXDL
            4,                           // MXdlPerWave
            4,                           // NXdlPerWave
            ck::Sequence<4, 64, 1>,      // ABlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,       // ABlockTransferSrcAccessOrder
            2,                           // ABlockTransferSrcVectorDim
            8,                           // ABlockTransferSrcScalarPerVector
            8,                           // ABlockTransferDstScalarPerVector_AK1
            1,                           // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>,      // BBlockTransferThreadClusterLengths
            ck::Sequence<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,       // BBlockTransferSrcAccessOrder
            2,                           // BBlockTransferSrcVectorDim
            8,                           // BBlockTransferSrcScalarPerVector
            8,                           // BBlockTransferDstScalarPerVector_BK1
            1,                           // BBlockLdsExtraN
            1,                           // CShuffleMXdlPerWavePerShuffle
            1,                           // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1, 32, 1, 8>,   // CDEBlockTransferClusterLengths
            8,                           // CDEBlockTransferScalarPerVector_NPerBlock
            ck::half_t,                  // AComputeDataType
            ck::half_t,                  // BComputeDataType
            ck::LoopScheduler::Default>; // LoopSched

    using InstTraits  = ck_tile::reflect::InstanceTraits<DeviceInstance>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    // Verify signature information
    EXPECT_EQ(traits.spatial_dim, InstTraits::kSpatialDim);
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
    EXPECT_EQ(traits.data_type, DataType::FP16);
    EXPECT_EQ(traits.gemm_padding, GemmPadding::DEFAULT);
    // Verify tile dimensions
    EXPECT_EQ(traits.tile_dims.m, InstTraits::kMPerBlock);
    EXPECT_EQ(traits.tile_dims.n, InstTraits::kNPerBlock);
    EXPECT_EQ(traits.tile_dims.k, InstTraits::kKPerBlock);
    // Verify pipeline configuration
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::DEFAULT);
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V1);
}

} // anonymous namespace

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ============================================================================
// Unit Tests for Individual Conversion Functions
// ============================================================================
//
// PURPOSE:
// --------
// These tests verify individual conversion and extraction functions that
// transform raw CK kernel parameters into semantic types. Each test
// focuses on a single conversion function to ensure it correctly maps
// CK types to builder enums and structures.
//
// TEST COVERAGE:
// --------------
// 1. Enum Conversions:
//    - Pipeline versions (BlockGemmPipelineVersion and PipelineVersion)
//    - Pipeline schedulers (BlockGemmPipelineScheduler and LoopScheduler)
//
// 2. Elementwise Operations (14 operations):
//    - PassThrough, Scale, Relu, Gelu, Sigmoid, Tanh, ScaleAdd
//    - Silu, Swish, Elu, LeakyRelu, UnaryConvert, ConvScale, ConvScaleAdd
//
// 3. Convolution Properties:
//    - Direction detection (Forward)
//    - Specializations (Default, Filter1x1Pad0, Filter1x1Stride1Pad0,
//      Filter3x3, OddC)
//
// 4. Layout Detection:
//    - 1D layouts (GNWC, NWGC, NGCW)
//    - 2D layouts (GNHWC, NHWGC, NGCHW with GKYXC/GKCYX)
//    - 3D layouts (GNDHWC, NDHWGC, NGCDHW)
//
// 5. Data Type Detection:
//    - FP16, BF16, FP32, I8
//
// 6. Pipeline Configuration:
//    - Pipeline versions (V2, V3)
//    - Schedulers (Interwave)
//
// 7. GEMM Padding Variations (17 types):
//    - Default, MNK, M, N, K, MN, MK, NK
//    - O, MO, NO, KO, MNO, MKO, NKO, MNKO
// ============================================================================

#include "ck/utility/scheduler_enum.hpp"
#include "ck_tile/builder/types.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck_tile/builder/reflect/instance_to_conv_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp>

namespace {

using ::ck_tile::builder::ConvDirection;
using ::ck_tile::builder::DataType;
using ::ck_tile::builder::ElementwiseOperation;
using ::ck_tile::builder::GemmPadding;
using ::ck_tile::builder::PipelineScheduler;
using ::ck_tile::builder::PipelineVersion;
using ::ck_tile::builder::TensorLayout;
using ::testing::ElementsAre;

// ============================================================================
// Test Helper Templates
// ============================================================================
// These templates reduce boilerplate by providing sensible defaults for
// template parameters that don't vary in most tests.
// ============================================================================

namespace defaults {
// Default values used across most tests
static constexpr int kBlockSize                                 = 256;
static constexpr int kMPerBlock                                 = 128;
static constexpr int kNPerBlock                                 = 128;
static constexpr int kKPerBlock                                 = 16;
static constexpr int kAK1                                       = 8;
static constexpr int kBK1                                       = 8;
static constexpr int kMPerXDL                                   = 32;
static constexpr int kNPerXDL                                   = 32;
static constexpr int kMXdlPerWave                               = 4;
static constexpr int kNXdlPerWave                               = 4;
static constexpr int kABlockTransferSrcVectorDim                = 2;
static constexpr int kABlockTransferSrcScalarPerVector          = 8;
static constexpr int kABlockTransferDstScalarPerVector_AK1      = 8;
static constexpr int kABlockLdsExtraM                           = 1;
static constexpr int kBBlockTransferSrcVectorDim                = 2;
static constexpr int kBBlockTransferSrcScalarPerVector          = 8;
static constexpr int kBBlockTransferDstScalarPerVector_BK1      = 8;
static constexpr int kBBlockLdsExtraN                           = 1;
static constexpr int kCShuffleMXdlPerWavePerShuffle             = 1;
static constexpr int kCShuffleNXdlPerWavePerShuffle             = 1;
static constexpr int kCDEBlockTransferScalarPerVector_NPerBlock = 8;
static constexpr bool kDirectLoad                               = false;
static constexpr int kNumGroupsToMerge                          = 1;

using DefaultABlockTransferThreadClusterLengths      = ck::Sequence<4, 64, 1>;
using DefaultABlockTransferThreadClusterArrangeOrder = ck::Sequence<1, 0, 2>;
using DefaultABlockTransferSrcAccessOrder            = ck::Sequence<1, 0, 2>;
using DefaultBBlockTransferThreadClusterLengths      = ck::Sequence<4, 64, 1>;
using DefaultBBlockTransferThreadClusterArrangeOrder = ck::Sequence<1, 0, 2>;
using DefaultBBlockTransferSrcAccessOrder            = ck::Sequence<1, 0, 2>;
using DefaultCDEBlockTransferClusterLengths          = ck::Sequence<1, 32, 1, 8>;
} // namespace defaults

// DeviceInstanceForTests - V3 variant with sensible defaults
template <int NDimSpatial                  = 2,
          typename ALayout                 = ck::tensor_layout::convolution::GNHWC,
          typename BLayout                 = ck::tensor_layout::convolution::GKYXC,
          typename ELayout                 = ck::tensor_layout::convolution::GNHWK,
          typename ADataType               = ck::half_t,
          typename BDataType               = ck::half_t,
          typename EDataType               = ck::half_t,
          typename AccDataType             = float,
          typename AElementwiseOperation   = ck::tensor_operation::element_wise::PassThrough,
          typename BElementwiseOperation   = ck::tensor_operation::element_wise::PassThrough,
          typename CDEElementwiseOperation = ck::tensor_operation::element_wise::PassThrough,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvForwardSpecialization =
              ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
          ck::tensor_operation::device::GemmSpecialization GemmSpec =
              ck::tensor_operation::device::GemmSpecialization::Default,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched =
              ck::BlockGemmPipelineScheduler::Intrawave,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer = ck::BlockGemmPipelineVersion::v1>
using DeviceInstanceForTests_V3 =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        NDimSpatial,
        ALayout,
        BLayout,
        ck::Tuple<>,
        ELayout,
        ADataType,
        BDataType,
        AccDataType,
        ADataType,
        ck::Tuple<>,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        ConvForwardSpecialization,
        GemmSpec,
        defaults::kBlockSize,
        defaults::kMPerBlock,
        defaults::kNPerBlock,
        defaults::kKPerBlock,
        defaults::kAK1,
        defaults::kBK1,
        defaults::kMPerXDL,
        defaults::kNPerXDL,
        defaults::kMXdlPerWave,
        defaults::kNXdlPerWave,
        defaults::DefaultABlockTransferThreadClusterLengths,
        defaults::DefaultABlockTransferThreadClusterArrangeOrder,
        defaults::DefaultABlockTransferSrcAccessOrder,
        defaults::kABlockTransferSrcVectorDim,
        defaults::kABlockTransferSrcScalarPerVector,
        defaults::kABlockTransferDstScalarPerVector_AK1,
        defaults::kABlockLdsExtraM,
        defaults::DefaultBBlockTransferThreadClusterLengths,
        defaults::DefaultBBlockTransferThreadClusterArrangeOrder,
        defaults::DefaultBBlockTransferSrcAccessOrder,
        defaults::kBBlockTransferSrcVectorDim,
        defaults::kBBlockTransferSrcScalarPerVector,
        defaults::kBBlockTransferDstScalarPerVector_BK1,
        defaults::kBBlockLdsExtraN,
        defaults::kCShuffleMXdlPerWavePerShuffle,
        defaults::kCShuffleNXdlPerWavePerShuffle,
        defaults::DefaultCDEBlockTransferClusterLengths,
        defaults::kCDEBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ADataType,
        BDataType,
        defaults::kDirectLoad,
        defaults::kNumGroupsToMerge>;

// Test case helper for specialization testing
template <ck::tensor_operation::device::ConvolutionForwardSpecialization Spec>
using SpecializationTestInstance =
    DeviceInstanceForTests_V3<2,
                              ck::tensor_layout::convolution::GNHWC,
                              ck::tensor_layout::convolution::GKYXC,
                              ck::tensor_layout::convolution::GNHWK,
                              ck::half_t,
                              ck::half_t,
                              ck::half_t,
                              float,
                              ck::tensor_operation::element_wise::PassThrough,
                              ck::tensor_operation::element_wise::PassThrough,
                              ck::tensor_operation::element_wise::PassThrough,
                              Spec>;

// Test case helper for layout testing (1D, 2D, 3D)
template <int NDim, typename ALayout, typename BLayout, typename ELayout>
using LayoutTestInstance = DeviceInstanceForTests_V3<NDim, ALayout, BLayout, ELayout>;

// Test case helper for data type testing
template <typename DataType, typename AccDataType = float>
using DataTypeTestInstance = DeviceInstanceForTests_V3<2,
                                                       ck::tensor_layout::convolution::GNHWC,
                                                       ck::tensor_layout::convolution::GKYXC,
                                                       ck::tensor_layout::convolution::GNHWK,
                                                       DataType,
                                                       DataType,
                                                       DataType,
                                                       AccDataType>;

// Test case helper for pipeline version testing
template <ck::BlockGemmPipelineVersion PipelineVer>
using PipelineVersionTestInstance = DeviceInstanceForTests_V3<
    2,
    ck::tensor_layout::convolution::GNHWC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::GNHWK,
    ck::half_t,
    ck::half_t,
    ck::half_t,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
    ck::tensor_operation::device::GemmSpecialization::Default,
    ck::BlockGemmPipelineScheduler::Intrawave,
    PipelineVer>;

// Test case helper for pipeline scheduler testing
template <ck::BlockGemmPipelineScheduler Scheduler>
using PipelineSchedulerTestInstance = DeviceInstanceForTests_V3<
    2,
    ck::tensor_layout::convolution::GNHWC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::GNHWK,
    ck::half_t,
    ck::half_t,
    ck::half_t,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
    ck::tensor_operation::device::GemmSpecialization::Default,
    Scheduler>;

// Test case helper for GEMM padding testing
template <ck::tensor_operation::device::GemmSpecialization GemmSpec>
using GemmPaddingTestInstance = DeviceInstanceForTests_V3<
    2,
    ck::tensor_layout::convolution::GNHWC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::GNHWK,
    ck::half_t,
    ck::half_t,
    ck::half_t,
    float,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
    GemmSpec>;

// ============================================================================
// Test Enum Conversion Functions
// ============================================================================

TEST(InstanceToConvTraits, ConvertsBlockGemmPipelineVersion)
{
    using ck_tile::reflect::conv::convert_pipeline_version;
    using enum ::ck::BlockGemmPipelineVersion;
    using enum ::ck_tile::builder::PipelineVersion;
    EXPECT_EQ(convert_pipeline_version<v1>(), V1);
    EXPECT_EQ(convert_pipeline_version<v2>(), V2);
    EXPECT_EQ(convert_pipeline_version<v3>(), V3);
    EXPECT_EQ(convert_pipeline_version<v4>(), V4);
    EXPECT_EQ(convert_pipeline_version<v5>(), V5);
}

TEST(InstanceToConvTraits, ConvertsPipelineVersion)
{
    using ck_tile::reflect::conv::convert_pipeline_version;
    using enum ck::PipelineVersion;
    using enum PipelineVersion;
    EXPECT_EQ(convert_pipeline_version<v1>(), V1);
    EXPECT_EQ(convert_pipeline_version<v2>(), V2);
    EXPECT_EQ(convert_pipeline_version<v4>(), V4);
    EXPECT_EQ(convert_pipeline_version<weight_only>(), WEIGHT_ONLY);
}

TEST(InstanceToConvTraits, ConvertsBlockGemmPipelineScheduler)
{
    using ck_tile::reflect::conv::convert_pipeline_scheduler;
    using enum ck::BlockGemmPipelineScheduler;
    using enum PipelineScheduler;
    EXPECT_EQ(convert_pipeline_scheduler<Intrawave>(), INTRAWAVE);
    EXPECT_EQ(convert_pipeline_scheduler<Interwave>(), INTERWAVE);
}

TEST(InstanceToConvTraits, ConvertsLoopScheduler)
{
    using ck_tile::reflect::conv::convert_pipeline_scheduler;
    using enum ck::LoopScheduler;
    using enum PipelineScheduler;
    EXPECT_EQ(convert_pipeline_scheduler<Default>(), DEFAULT);
    EXPECT_EQ(convert_pipeline_scheduler<Interwave>(), INTERWAVE);
}

// ============================================================================
// Test Elementwise Operations
// ============================================================================

TEST(InstanceToConvTraits, ExtractsPassThroughOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::PassThrough>();
    EXPECT_EQ(op, PASS_THROUGH);
}

TEST(InstanceToConvTraits, ExtractsScaleOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::Scale>();
    EXPECT_EQ(op, SCALE);
}

TEST(InstanceToConvTraits, ExtractsReluOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::Relu>();
    EXPECT_EQ(op, RELU);
}

TEST(InstanceToConvTraits, ExtractsGeluOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::Gelu>();
    EXPECT_EQ(op, GELU);
}

TEST(InstanceToConvTraits, ExtractsSigmoidOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::Sigmoid>();
    EXPECT_EQ(op, SIGMOID);
}

TEST(InstanceToConvTraits, ExtractsTanhOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::TanH>();
    EXPECT_EQ(op, TANH);
}

TEST(InstanceToConvTraits, ExtractsScaleAddOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::ScaleAdd>();
    EXPECT_EQ(op, SCALE_ADD);
}

TEST(InstanceToConvTraits, ExtractsSiluOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::Silu>();
    EXPECT_EQ(op, SILU);
}

TEST(InstanceToConvTraits, ExtractsSwishOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::Swish>();
    EXPECT_EQ(op, SWISH);
}

TEST(InstanceToConvTraits, ExtractsEluOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::Elu>();
    EXPECT_EQ(op, ELU);
}

TEST(InstanceToConvTraits, ExtractsLeakyReluOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::LeakyRelu>();
    EXPECT_EQ(op, LEAKY_RELU);
}

TEST(InstanceToConvTraits, ExtractsUnaryConvertOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::UnaryConvert>();
    EXPECT_EQ(op, UNARY_CONVERT);
}

TEST(InstanceToConvTraits, ExtractsConvScaleOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::ConvScale>();
    EXPECT_EQ(op, CONV_SCALE);
}

TEST(InstanceToConvTraits, ExtractsConvScaleAddOperation)
{
    using enum ElementwiseOperation;
    constexpr auto op =
        ck_tile::reflect::conv::elementwise_op<ck::tensor_operation::element_wise::ConvScaleAdd>();
    EXPECT_EQ(op, CONV_SCALE_ADD);
}

// ============================================================================
// Test Convolution Direction Detection
// ============================================================================

TEST(InstanceToConvTraits, DetectsForwardDirection)
{
    using DeviceInstance = DeviceInstanceForTests_V3<>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.direction, ConvDirection::FORWARD);
}

// ============================================================================
// Test Convolution Specialization Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsDefaultSpecialization)
{
    using DeviceInstance = SpecializationTestInstance<
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::DEFAULT);
}

TEST(InstanceToConvTraits, ExtractsFilter1x1Pad0Specialization)
{
    using DeviceInstance = SpecializationTestInstance<
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::FILTER_1X1_PAD0);
}

TEST(InstanceToConvTraits, ExtractsFilter1x1Stride1Pad0Specialization)
{
    using DeviceInstance = SpecializationTestInstance<
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.conv_specialization,
              ck_tile::builder::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);
}

TEST(InstanceToConvTraits, ExtractsFilter3x3Specialization)
{
    using DeviceInstance = SpecializationTestInstance<
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::FILTER_3x3);
}

TEST(InstanceToConvTraits, ExtractsOddCSpecialization)
{
    using DeviceInstance = SpecializationTestInstance<
        ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.conv_specialization, ck_tile::builder::ConvSpecialization::ODD_C);
}

// ============================================================================
// Test 1D Convolution Layout Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsGnwcLayout)
{
    using DeviceInstance = LayoutTestInstance<1,
                                              ck::tensor_layout::convolution::GNWC,
                                              ck::tensor_layout::convolution::GKXC,
                                              ck::tensor_layout::convolution::GNWK>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.spatial_dim, 1);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNWC, TensorLayout::GKXC, TensorLayout::GNWK));
}

TEST(InstanceToConvTraits, ExtractsNwgcLayout)
{
    using DeviceInstance = LayoutTestInstance<1,
                                              ck::tensor_layout::convolution::NWGC,
                                              ck::tensor_layout::convolution::GKXC,
                                              ck::tensor_layout::convolution::NWGK>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.spatial_dim, 1);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::NWGC, TensorLayout::GKXC, TensorLayout::NWGK));
}

TEST(InstanceToConvTraits, ExtractsNgcwLayout)
{
    using DeviceInstance = LayoutTestInstance<1,
                                              ck::tensor_layout::convolution::NGCW,
                                              ck::tensor_layout::convolution::GKXC,
                                              ck::tensor_layout::convolution::NGKW>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.spatial_dim, 1);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::NGCW, TensorLayout::GKXC, TensorLayout::NGKW));
}

// ============================================================================
// Test 2D Convolution Layout Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsGnhwcLayout)
{
    using DeviceInstance = LayoutTestInstance<2,
                                              ck::tensor_layout::convolution::GNHWC,
                                              ck::tensor_layout::convolution::GKYXC,
                                              ck::tensor_layout::convolution::GNHWK>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNHWC, TensorLayout::GKYXC, TensorLayout::GNHWK));
}

TEST(InstanceToConvTraits, ExtractsNhwgcLayout)
{
    using DeviceInstance = LayoutTestInstance<2,
                                              ck::tensor_layout::convolution::NHWGC,
                                              ck::tensor_layout::convolution::GKYXC,
                                              ck::tensor_layout::convolution::NHWGK>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::NHWGC, TensorLayout::GKYXC, TensorLayout::NHWGK));
}

TEST(InstanceToConvTraits, ExtractsNgchwGkyxcLayout)
{
    using DeviceInstance = LayoutTestInstance<2,
                                              ck::tensor_layout::convolution::NGCHW,
                                              ck::tensor_layout::convolution::GKYXC,
                                              ck::tensor_layout::convolution::NGKHW>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::NGCHW, TensorLayout::GKYXC, TensorLayout::NGKHW));
}

TEST(InstanceToConvTraits, ExtractsNgchwGkcyxLayout)
{
    using DeviceInstance = LayoutTestInstance<2,
                                              ck::tensor_layout::convolution::NGCHW,
                                              ck::tensor_layout::convolution::GKCYX,
                                              ck::tensor_layout::convolution::NGKHW>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::NGCHW, TensorLayout::GKCYX, TensorLayout::NGKHW));
}

// ============================================================================
// Test 3D Convolution Layout Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsGndhwcLayout)
{
    using DeviceInstance = LayoutTestInstance<3,
                                              ck::tensor_layout::convolution::GNDHWC,
                                              ck::tensor_layout::convolution::GKZYXC,
                                              ck::tensor_layout::convolution::GNDHWK>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.spatial_dim, 3);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::GNDHWC, TensorLayout::GKZYXC, TensorLayout::GNDHWK));
}

TEST(InstanceToConvTraits, ExtractsNdhwgcLayout)
{
    using DeviceInstance = LayoutTestInstance<3,
                                              ck::tensor_layout::convolution::NDHWGC,
                                              ck::tensor_layout::convolution::GKZYXC,
                                              ck::tensor_layout::convolution::NDHWGK>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.spatial_dim, 3);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::NDHWGC, TensorLayout::GKZYXC, TensorLayout::NDHWGK));
}

TEST(InstanceToConvTraits, ExtractsNgcdhwLayout)
{
    using DeviceInstance = LayoutTestInstance<3,
                                              ck::tensor_layout::convolution::NGCDHW,
                                              ck::tensor_layout::convolution::GKZYXC,
                                              ck::tensor_layout::convolution::NGKDHW>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.spatial_dim, 3);
    EXPECT_THAT(traits.layout,
                ElementsAre(TensorLayout::NGCDHW, TensorLayout::GKZYXC, TensorLayout::NGKDHW));
}

// ============================================================================
// Test Data Type Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsFp16DataType)
{
    using DeviceInstance = DataTypeTestInstance<ck::half_t>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.data_type, DataType::FP16);
}

TEST(InstanceToConvTraits, ExtractsBf16DataType)
{
    using DeviceInstance = DataTypeTestInstance<ck::bhalf_t>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.data_type, DataType::BF16);
}

TEST(InstanceToConvTraits, ExtractsFp32DataType)
{
    using DeviceInstance = DataTypeTestInstance<float, float>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.data_type, DataType::FP32);
}

TEST(InstanceToConvTraits, ExtractsI8DataType)
{
    using DeviceInstance = DataTypeTestInstance<int8_t, int32_t>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.data_type, DataType::I8);
}

// ============================================================================
// Test Pipeline Version Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsPipelineV2)
{
    using DeviceInstance = PipelineVersionTestInstance<ck::BlockGemmPipelineVersion::v2>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V2);
}

TEST(InstanceToConvTraits, ExtractsPipelineV3)
{
    using DeviceInstance = PipelineVersionTestInstance<ck::BlockGemmPipelineVersion::v3>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.pipeline_version, PipelineVersion::V3);
}

TEST(InstanceToConvTraits, ExtractsInterwaveScheduler)
{
    using DeviceInstance = PipelineSchedulerTestInstance<ck::BlockGemmPipelineScheduler::Interwave>;
    const auto traits    = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.pipeline_scheduler, PipelineScheduler::INTERWAVE);
}

// ============================================================================
// Test GEMM Padding Detection
// ============================================================================

TEST(InstanceToConvTraits, ExtractsDefaultGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::Default>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::DEFAULT);
}

TEST(InstanceToConvTraits, ExtractsMnkGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::MNK_PADDING);
}

TEST(InstanceToConvTraits, ExtractsMPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::M_PADDING);
}

TEST(InstanceToConvTraits, ExtractsNPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::NPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::N_PADDING);
}

TEST(InstanceToConvTraits, ExtractsKPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::KPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::K_PADDING);
}

TEST(InstanceToConvTraits, ExtractsMnPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MNPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::MN_PADDING);
}

TEST(InstanceToConvTraits, ExtractsMkPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MKPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::MK_PADDING);
}

TEST(InstanceToConvTraits, ExtractsNkPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::NKPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::NK_PADDING);
}

TEST(InstanceToConvTraits, ExtractsOPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::OPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::O_PADDING);
}

TEST(InstanceToConvTraits, ExtractsMoPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MOPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::MO_PADDING);
}

TEST(InstanceToConvTraits, ExtractsNoPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::NOPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::NO_PADDING);
}

TEST(InstanceToConvTraits, ExtractsKoPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::KOPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::KO_PADDING);
}

TEST(InstanceToConvTraits, ExtractsMnoPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MNOPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::MNO_PADDING);
}

TEST(InstanceToConvTraits, ExtractsMkoPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MKOPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::MKO_PADDING);
}

TEST(InstanceToConvTraits, ExtractsNkoPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::NKOPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::NKO_PADDING);
}

TEST(InstanceToConvTraits, ExtractsMnkoPaddingGemmPadding)
{
    using DeviceInstance =
        GemmPaddingTestInstance<ck::tensor_operation::device::GemmSpecialization::MNKOPadding>;
    const auto traits = ck_tile::reflect::conv::instance_to_conv_traits<DeviceInstance>();
    EXPECT_EQ(traits.gemm_padding, GemmPadding::MNKO_PADDING);
}

} // anonymous namespace

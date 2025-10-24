// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>

namespace {

using ::testing::ElementsAre;
// Test fixture for InstanceTraits tests
class InstanceTraitsTest : public ::testing::Test
{
};

// Test InstanceTraits with DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
TEST_F(InstanceTraitsTest, ConvFwdInstanceTraitsExtraction)
{
    // Define a concrete instance type with specific template parameters
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
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default,                                               // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::Default, // GemmSpec
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // AK1
            8,                                                         // BK1
            32,                                                        // MPerXDL
            32,                                                        // NPerXDL
            4,                                                         // MXdlPerWave
            4,                                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,               // CDEBlockTransferScalarPerVector_NPerBlock
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t>;                               // BComputeDataType

    // Use InstanceTraits to extract compile-time information
    using Traits = ck_tile::reflect::InstanceTraits<DeviceInstance>;

    // Verify spatial dimension
    EXPECT_EQ(Traits::kSpatialDim, 2);

    // Verify block configuration
    EXPECT_EQ(Traits::kBlockSize, 256);
    EXPECT_EQ(Traits::kMPerBlock, 128);
    EXPECT_EQ(Traits::kNPerBlock, 128);
    EXPECT_EQ(Traits::kKPerBlock, 16);

    // Verify tuning parameters
    EXPECT_EQ(Traits::kAK1, 8);
    EXPECT_EQ(Traits::kBK1, 8);
    EXPECT_EQ(Traits::kMPerXDL, 32);
    EXPECT_EQ(Traits::kNPerXDL, 32);
    EXPECT_EQ(Traits::kMXdlPerWave, 4);
    EXPECT_EQ(Traits::kNXdlPerWave, 4);

    // Verify A block transfer parameters
    EXPECT_EQ(Traits::kABlockTransferSrcVectorDim, 2);
    EXPECT_EQ(Traits::kABlockTransferSrcScalarPerVector, 8);
    EXPECT_EQ(Traits::kABlockTransferDstScalarPerVectorK1, 8);
    EXPECT_EQ(Traits::kABlockLdsExtraM, 1);

    // Verify B block transfer parameters
    EXPECT_EQ(Traits::kBBlockTransferSrcVectorDim, 2);
    EXPECT_EQ(Traits::kBBlockTransferSrcScalarPerVector, 8);
    EXPECT_EQ(Traits::kBBlockTransferDstScalarPerVectorK1, 8);
    EXPECT_EQ(Traits::kBBlockLdsExtraN, 1);

    // Verify C shuffle parameters
    EXPECT_EQ(Traits::kCShuffleMXdlPerWavePerShuffle, 1);
    EXPECT_EQ(Traits::kCShuffleNXdlPerWavePerShuffle, 1);
    EXPECT_EQ(Traits::kCBlockTransferScalarPerVector, 8);

    // Verify pipeline configuration
    EXPECT_EQ(Traits::kPipelineScheduler, ck::BlockGemmPipelineScheduler::Intrawave);
    EXPECT_EQ(Traits::kPipelineVersion, ck::BlockGemmPipelineVersion::v1);

    // Verify data types using std::is_same
    EXPECT_TRUE((std::is_same<Traits::ADataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::BDataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::AccDataType, float>::value));
    EXPECT_TRUE((std::is_same<Traits::EDataType, ck::half_t>::value));

    // Verify layout types
    EXPECT_TRUE((std::is_same<Traits::ALayout, ck::tensor_layout::convolution::GNHWC>::value));
    EXPECT_TRUE((std::is_same<Traits::BLayout, ck::tensor_layout::convolution::GKYXC>::value));
    EXPECT_TRUE((std::is_same<Traits::ELayout, ck::tensor_layout::convolution::GNHWK>::value));

    // Verify all array values for thread cluster lengths using googlemock matchers
    EXPECT_THAT(Traits::kAThreadClusterLengths, ElementsAre(4, 64, 1));
    EXPECT_THAT(Traits::kBThreadClusterLengths, ElementsAre(4, 64, 1));
    EXPECT_THAT(Traits::kCThreadClusterLengths, ElementsAre(1, 32, 1, 8));

    // Verify A block transfer arrange order and access order arrays
    EXPECT_THAT(Traits::kAThreadClusterArrangeOrder, ElementsAre(1, 0, 2));
    EXPECT_THAT(Traits::kABlockTransferSrcAccessOrder, ElementsAre(1, 0, 2));

    // Verify B block transfer arrange order and access order arrays
    EXPECT_THAT(Traits::kBThreadClusterArrangeOrder, ElementsAre(1, 0, 2));
    EXPECT_THAT(Traits::kBBlockTransferSrcAccessOrder, ElementsAre(1, 0, 2));

    // Verify additional data types
    EXPECT_TRUE((std::is_same<Traits::CShuffleDataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::DsDataType, ck::Tuple<>>::value));
    EXPECT_TRUE((std::is_same<Traits::AComputeDataType, ck::half_t>::value));
    EXPECT_TRUE((std::is_same<Traits::BComputeDataType, ck::half_t>::value));

    // Verify additional layout types
    EXPECT_TRUE((std::is_same<Traits::DsLayout, ck::Tuple<>>::value));

    // Verify element-wise operations
    EXPECT_TRUE((std::is_same<Traits::AElementwiseOperation,
                              ck::tensor_operation::element_wise::PassThrough>::value));
    EXPECT_TRUE((std::is_same<Traits::BElementwiseOperation,
                              ck::tensor_operation::element_wise::PassThrough>::value));
    EXPECT_TRUE((std::is_same<Traits::CDEElementwiseOperation,
                              ck::tensor_operation::element_wise::PassThrough>::value));
}

// Test instance_string function
TEST_F(InstanceTraitsTest, InstanceStringGeneration)
{
    // Define a concrete instance type with specific template parameters
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
            ck::tensor_operation::device::ConvolutionForwardSpecialization::
                Default,                                               // ConvForwardSpecialization
            ck::tensor_operation::device::GemmSpecialization::Default, // GemmSpec
            256,                                                       // BlockSize
            128,                                                       // MPerBlock
            128,                                                       // NPerBlock
            16,                                                        // KPerBlock
            8,                                                         // AK1
            8,                                                         // BK1
            32,                                                        // MPerXDL
            32,                                                        // NPerXDL
            4,                                                         // MXdlPerWave
            4,                                                         // NXdlPerWave
            ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,                      // ABlockTransferSrcVectorDim
            8,                      // ABlockTransferSrcScalarPerVector
            8,                      // ABlockTransferDstScalarPerVector_AK1
            1,                      // ABlockLdsExtraM
            ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,                      // BBlockTransferSrcVectorDim
            8,                      // BBlockTransferSrcScalarPerVector
            8,                      // BBlockTransferDstScalarPerVector_BK1
            1,                      // BBlockLdsExtraN
            1,                      // CShuffleMXdlPerWavePerShuffle
            1,                      // CShuffleNXdlPerWavePerShuffle
            ck::Sequence<1,
                         32,
                         1,
                         8>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            8,               // CDEBlockTransferScalarPerVector_NPerBlock
            ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
            ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
            ck::half_t,                                // AComputeDataType
            ck::half_t>;                               // BComputeDataType

    // Generate instance string
    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    // Expected string with all template parameters in exact order
    std::string expected_str = "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                               "<2"             // NDimSpatial
                               ",GNHWC"         // ALayout
                               ",GKYXC"         // BLayout
                               ",EmptyTuple"    // DsLayout
                               ",GNHWK"         // ELayout
                               ",fp16"          // ADataType
                               ",fp16"          // BDataType
                               ",fp32"          // AccDataType
                               ",fp16"          // CShuffleDataType
                               ",EmptyTuple"    // DsDataType
                               ",fp16"          // EDataType
                               ",PassThrough"   // AElementwiseOperation
                               ",PassThrough"   // BElementwiseOperation
                               ",PassThrough"   // CDEElementwiseOperation
                               ",Default"       // ConvForwardSpecialization
                               ",Default"       // GemmSpec
                               ",256"           // BlockSize
                               ",128"           // MPerBlock
                               ",128"           // NPerBlock
                               ",16"            // KPerBlock
                               ",8"             // AK1
                               ",8"             // BK1
                               ",32"            // MPerXDL
                               ",32"            // NPerXDL
                               ",4"             // MXdlPerWave
                               ",4"             // NXdlPerWave
                               ",Seq(4,64,1)"   // ABlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               ",2"             // ABlockTransferSrcVectorDim
                               ",8"             // ABlockTransferSrcScalarPerVector
                               ",8"             // ABlockTransferDstScalarPerVector_AK1
                               ",1"             // ABlockLdsExtraM
                               ",Seq(4,64,1)"   // BBlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               ",2"             // BBlockTransferSrcVectorDim
                               ",8"             // BBlockTransferSrcScalarPerVector
                               ",8"             // BBlockTransferDstScalarPerVector_BK1
                               ",1"             // BBlockLdsExtraN
                               ",1"             // CShuffleMXdlPerWavePerShuffle
                               ",1"             // CShuffleNXdlPerWavePerShuffle
                               ",Seq(1,32,1,8)" // CDEBlockTransferClusterLengths
                               ",8"             // CDEBlockTransferScalarPerVector_NPerBlock
                               ",Intrawave"     // BlkGemmPipeSched
                               ",v1"            // BlkGemmPipelineVer
                               ",fp16"          // AComputeDataType
                               ",fp16>";        // BComputeDataType

    // Verify the generated string matches exactly
    EXPECT_EQ(instance_str, expected_str);
}

} // anonymous namespace

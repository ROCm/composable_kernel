// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <ck/ck.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle.hpp>

namespace {

TEST(InstanceTraits, BwdWeightXdlCShuffleInstanceStringReturnsCorrectFormat)
{
    using DeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffle<
        2,                                               // NDimSpatial
        ck::tensor_layout::convolution::GNHWC,           // InLayout
        ck::tensor_layout::convolution::GKYXC,           // WeiLayout
        ck::tensor_layout::convolution::GNHWK,           // OutLayout
        ck::half_t,                                      // InDataType
        ck::half_t,                                      // WeiDataType
        ck::half_t,                                      // OutDataType
        float,                                           // AccDataType
        ck::tensor_operation::element_wise::PassThrough, // InElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // WeiElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough, // OutElementwiseOperation
        ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::
            Default,            // ConvBackwardWeightSpecialization
        256,                    // BlockSize
        128,                    // MPerBlock
        128,                    // NPerBlock
        4,                      // K0PerBlock
        8,                      // K1
        32,                     // MPerXDL
        32,                     // NPerXDL
        2,                      // MXdlPerWave
        2,                      // NXdlPerWave
        ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1
        ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,                      // ABlockTransferSrcVectorDim
        8,                      // ABlockTransferSrcScalarPerVector
        8,                      // ABlockTransferDstScalarPerVector_K1
        false,                  // ABlockLdsAddExtraM
        ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1
        ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,                      // BBlockTransferSrcVectorDim
        8,                      // BBlockTransferSrcScalarPerVector
        8,                      // BBlockTransferDstScalarPerVector_K1
        false,                  // BBlockLdsAddExtraN
        1,                      // CShuffleMXdlPerWavePerShuffle
        1,                      // CShuffleNXdlPerWavePerShuffle
        ck::Sequence<1,
                     32,
                     1,
                     8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,               // CBlockTransferScalarPerVector_NWaveNPerXdl
        ck::half_t,      // ComputeTypeA
        ck::half_t,      // ComputeTypeB
        1,               // MaxTransposeTransferSrcScalarPerVector
        1>;              // MaxTransposeTransferDstScalarPerVector

    std::string instance_str = ck_tile::reflect::instance_string<DeviceInstance>();

    std::string expected_str = "DeviceGroupedConvBwdWeight_Xdl_CShuffle"
                               "<2"             // NDimSpatial
                               ",GNHWC"         // InLayout
                               ",GKYXC"         // WeiLayout
                               ",GNHWK"         // OutLayout
                               ",fp16"          // InDataType
                               ",fp16"          // WeiDataType
                               ",fp16"          // OutDataType
                               ",fp32"          // AccDataType
                               ",PassThrough"   // InElementwiseOperation
                               ",PassThrough"   // WeiElementwiseOperation
                               ",PassThrough"   // OutElementwiseOperation
                               ",Default"       // ConvBackwardWeightSpecialization
                               ",256"           // BlockSize
                               ",128"           // MPerBlock
                               ",128"           // NPerBlock
                               ",4"             // K0PerBlock
                               ",8"             // K1
                               ",32"            // MPerXDL
                               ",32"            // NPerXDL
                               ",2"             // MXdlPerWave
                               ",2"             // NXdlPerWave
                               ",Seq(4,64,1)"   // ABlockTransferThreadClusterLengths_K0_M_K1
                               ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               ",2"             // ABlockTransferSrcVectorDim
                               ",8"             // ABlockTransferSrcScalarPerVector
                               ",8"             // ABlockTransferDstScalarPerVector_K1
                               ",false"         // ABlockLdsAddExtraM
                               ",Seq(4,64,1)"   // BBlockTransferThreadClusterLengths_K0_N_K1
                               ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               ",2"             // BBlockTransferSrcVectorDim
                               ",8"             // BBlockTransferSrcScalarPerVector
                               ",8"             // BBlockTransferDstScalarPerVector_K1
                               ",false"         // BBlockLdsAddExtraN
                               ",1"             // CShuffleMXdlPerWavePerShuffle
                               ",1"             // CShuffleNXdlPerWavePerShuffle
                               ",Seq(1,32,1,8)" // CBlockTransferClusterLengths
                               ",8"             // CBlockTransferScalarPerVector_NWaveNPerXdl
                               ",fp16"          // ComputeTypeA
                               ",fp16"          // ComputeTypeB
                               ",1"             // MaxTransposeTransferSrcScalarPerVector
                               ",1>";           // MaxTransposeTransferDstScalarPerVector

    EXPECT_EQ(instance_str, expected_str);
}

} // anonymous namespace

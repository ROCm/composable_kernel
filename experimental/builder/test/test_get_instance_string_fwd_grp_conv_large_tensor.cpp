// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_large_tensor_instance.hpp>

// Test GetInstanceString through base class pointer for large tensor variant
TEST(GetInstanceString, ReturnsStringForFwdGrpConvLargeTensorInstance)
{
    // Use the template helper to get a working instance configuration
    using InstanceTuple = ck::tensor_operation::device::instance::
        device_grouped_conv_fwd_xdl_large_tensor_f16_instances<
            2,                                                       // NDimSpatial
            ck::tensor_operation::device::instance::GNHWC,           // ALayout
            ck::tensor_operation::device::instance::GKYXC,           // BLayout
            ck::tensor_operation::device::instance::Empty_Tuple,     // DsLayout
            ck::tensor_operation::device::instance::GNHWK,           // ELayout
            ck::tensor_operation::device::instance::ConvFwdDefault>; // ConvForwardSpecialization

    // Get the first instance from the tuple
    using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

    // Define the base class type using DeviceGroupedConvFwdMultipleABD
    using BaseClass = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        2,                                                   // NDimSpatial
        ck::tensor_operation::device::instance::GNHWC,       // ALayout
        ck::tensor_operation::device::instance::GKYXC,       // BLayout
        ck::tensor_operation::device::instance::Empty_Tuple, // DsLayout
        ck::tensor_operation::device::instance::GNHWK,       // ELayout
        ck::half_t,                                          // ADataType
        ck::half_t,                                          // BDataType
        ck::Tuple<>,                                         // DsDataType
        ck::half_t,                                          // EDataType
        ck::tensor_operation::element_wise::PassThrough,     // AElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough,     // BElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough,     // CDEElementwiseOperation
        ck::half_t,                                          // AComputeType
        ck::half_t>;                                         // BComputeType

    // Create an instance of the derived class
    DeviceInstance device_instance;

    // Get a pointer to the base class
    BaseClass* base_ptr = &device_instance;

    // Call GetInstanceString through the base class pointer
    std::string instance_str = base_ptr->GetInstanceString();

    // Expected complete instance string based on the first instance from
    // device_grouped_conv_fwd_xdl_large_tensor_f16_instances
    std::string expected_str = "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor"
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
                               ",MNKPadding"    // GemmSpec
                               ",1"             // NumGemmKPrefetchStage
                               ",64"            // BlockSize
                               ",64"            // MPerBlock
                               ",64"            // NPerBlock
                               ",32"            // KPerBlock
                               ",8"             // AK1
                               ",8"             // BK1
                               ",32"            // MPerXDL
                               ",32"            // NPerXDL
                               ",2"             // MXdlPerWave
                               ",2"             // NXdlPerWave
                               ",Seq(4,16,1)"   // ABlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               ",2"             // ABlockTransferSrcVectorDim
                               ",1"             // ABlockTransferSrcScalarPerVector
                               ",8"             // ABlockTransferDstScalarPerVector_AK1
                               ",1"             // ABlockLdsExtraM
                               ",Seq(4,16,1)"   // BBlockTransferThreadClusterLengths
                               ",Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               ",2"             // BBlockTransferSrcVectorDim
                               ",1"             // BBlockTransferSrcScalarPerVector
                               ",8"             // BBlockTransferDstScalarPerVector_BK1
                               ",1"             // BBlockLdsExtraN
                               ",1"             // CShuffleMXdlPerWavePerShuffle
                               ",1"             // CShuffleNXdlPerWavePerShuffle
                               ",Seq(1,16,1,4)" // CDEBlockTransferClusterLengths
                               ",1"             // CDEBlockTransferScalarPerVector_NPerBlock
                               ",fp16"          // AComputeDataType
                               ",fp16"          // BComputeDataType
                               ",Default>";     // LoopScheduler
    EXPECT_EQ(instance_str, expected_str);
}

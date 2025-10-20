// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp>

// Test GetInstanceString through base class pointer
TEST(GetInstanceStringTest, GetInstanceStringThroughBaseClass)
{
    // Use the template helper to get a working instance configuration
    using InstanceTuple =
        ck::tensor_operation::device::instance::device_grouped_conv_fwd_xdl_f16_comp_instances<
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
    // device_grouped_conv_fwd_xdl_f16_comp_instances This corresponds to the configuration with
    // BlockSize=256, MPerBlock=128, NPerBlock=128, KPerBlock=64, etc.
    std::string expected_str = "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                               "_2"             // NDimSpatial
                               "_GNHWC"         // ALayout
                               "_GKYXC"         // BLayout
                               "_EmptyTuple"    // DsLayout
                               "_GNHWK"         // ELayout
                               "_fp16"          // ADataType
                               "_fp16"          // BDataType
                               "_fp32"          // AccDataType
                               "_fp16"          // CShuffleDataType
                               "_EmptyTuple"    // DsDataType
                               "_fp16"          // EDataType
                               "_PassThrough"   // AElementwiseOperation
                               "_PassThrough"   // BElementwiseOperation
                               "_PassThrough"   // CDEElementwiseOperation
                               "_Default"       // ConvForwardSpecialization
                               "_MNKPadding"    // GemmSpec
                               "_256"           // BlockSize
                               "_128"           // MPerBlock
                               "_128"           // NPerBlock
                               "_64"            // KPerBlock
                               "_8"             // AK1
                               "_8"             // BK1
                               "_32"            // MPerXDL
                               "_32"            // NPerXDL
                               "_2"             // MXdlPerWave
                               "_2"             // NXdlPerWave
                               "_Seq(8,32,1)"   // ABlockTransferThreadClusterLengths
                               "_Seq(1,0,2)"    // ABlockTransferThreadClusterArrangeOrder
                               "_Seq(1,0,2)"    // ABlockTransferSrcAccessOrder
                               "_2"             // ABlockTransferSrcVectorDim
                               "_8"             // ABlockTransferSrcScalarPerVector
                               "_8"             // ABlockTransferDstScalarPerVector_AK1
                               "_0"             // ABlockLdsExtraM
                               "_Seq(8,32,1)"   // BBlockTransferThreadClusterLengths
                               "_Seq(1,0,2)"    // BBlockTransferThreadClusterArrangeOrder
                               "_Seq(1,0,2)"    // BBlockTransferSrcAccessOrder
                               "_2"             // BBlockTransferSrcVectorDim
                               "_8"             // BBlockTransferSrcScalarPerVector
                               "_8"             // BBlockTransferDstScalarPerVector_BK1
                               "_0"             // BBlockLdsExtraN
                               "_1"             // CShuffleMXdlPerWavePerShuffle
                               "_1"             // CShuffleNXdlPerWavePerShuffle
                               "_Seq(1,32,1,8)" // CDEBlockTransferClusterLengths
                               "_8"             // CDEBlockTransferScalarPerVector_NPerBlock
                               "_Intrawave"     // BlkGemmPipeSched
                               "_v4"            // BlkGemmPipelineVer
                               "_fp16"          // AComputeDataType
                               "_fp16";         // BComputeDataType
    EXPECT_EQ(instance_str, expected_str);
}

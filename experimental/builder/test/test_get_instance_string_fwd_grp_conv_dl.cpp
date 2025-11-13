// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck/tensor_operation/gpu/device/device_base.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_dl_instance.hpp>

// Test GetInstanceString through base class pointer for DL variant
TEST(GetInstanceString, ReturnsStringForFwdGrpConvDlInstance)
{
    // Use the template helper to get a working instance configuration
    using InstanceTuple =
        ck::tensor_operation::device::instance::device_grouped_conv2d_fwd_dl_f16_instances<
            ck::tensor_operation::device::instance::GNHWC,           // InLayout
            ck::tensor_operation::device::instance::GKYXC,           // WeiLayout
            ck::tensor_operation::device::instance::Empty_Tuple,     // DsLayout
            ck::tensor_operation::device::instance::GNHWK,           // OutLayout
            ck::Tuple<>,                                             // DsDatatype
            ck::tensor_operation::element_wise::PassThrough,         // CDEElementOp
            ck::tensor_operation::device::instance::ConvFwdDefault>; // ConvSpec

    // Get the first instance from the tuple
    using DeviceInstance = typename std::tuple_element<0, InstanceTuple>::type;

    // Define the base class type using the most general operator base
    using BaseClass = ck::tensor_operation::device::BaseOperator;

    // Create an instance of the derived class
    DeviceInstance device_instance;

    // Get a pointer to the base class
    BaseClass* base_ptr = &device_instance;

    // Call GetInstanceString through the base class pointer
    std::string instance_str = base_ptr->GetInstanceString();

    // Expected complete instance string based on the first instance from
    // device_grouped_conv2d_fwd_dl_f16_instances
    std::string expected_str = "DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK"
                               "<2"            // NDimSpatial
                               ",fp16"         // ADataType
                               ",fp16"         // BDataType
                               ",EmptyTuple"   // DsDataType
                               ",fp16"         // EDataType
                               ",fp32"         // AccDataType
                               ",GNHWC"        // ALayout
                               ",GKYXC"        // BLayout
                               ",EmptyTuple"   // DsLayout
                               ",GNHWK"        // ELayout
                               ",PassThrough"  // AElementwiseOperation
                               ",PassThrough"  // BElementwiseOperation
                               ",PassThrough"  // CDEElementwiseOperation
                               ",Default"      // ConvForwardSpecialization
                               ",MNKPadding"   // GemmSpec
                               ",8"            // BlockSize
                               ",16"           // MPerBlock
                               ",4"            // NPerBlock
                               ",2"            // K0PerBlock
                               ",1"            // K1
                               ",1"            // M1PerThread
                               ",2"            // N1PerThread
                               ",1"            // KPerThread
                               ",Seq(4,2)"     // M1N1ThreadClusterM1Xs
                               ",Seq(1,1)"     // M1N1ThreadClusterN1Xs
                               ",Seq(2,1,2,1)" // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
                               ",Seq(1,1,8,1)" // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
                               ",Seq(1,2,0,3)" // ABlockTransferThreadClusterArrangeOrder
                               ",Seq(1,2,0,3)" // ABlockTransferSrcAccessOrder
                               ",Seq(1,1,1,1)" // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
                               ",Seq(1,2,0,3)" // ABlockTransferSrcVectorTensorContiguousDimOrder
                               ",Seq(1,1,1,1)" // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
                               ",Seq(1,1,1,1)" // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
                               ",Seq(2,1,4,1)" // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
                               ",Seq(1,2,0,3)" // BBlockTransferThreadClusterArrangeOrder
                               ",Seq(1,2,0,3)" // BBlockTransferSrcAccessOrder
                               ",Seq(1,1,1,1)" // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
                               ",Seq(1,2,0,3)" // BBlockTransferSrcVectorTensorContiguousDimOrder
                               ",Seq(1,1,1,1)" // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
                               ",Seq(0,1,2,3,4,5)" // CThreadTransferSrcDstAccessOrder
                               ",5"                // CThreadTransferSrcDstVectorDim
                               ",1>";              // CThreadTransferDstScalarPerVector

    EXPECT_EQ(instance_str, expected_str);
}

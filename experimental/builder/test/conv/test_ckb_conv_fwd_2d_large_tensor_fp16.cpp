// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_common.hpp"

using namespace ck_tile::builder::test_utils;

namespace ck_tile::builder::testing {

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 128, .k = 32}};

    run_test_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
        FwdConvSignature,
        FwdThreadBlock,
        ConvFwdSpecialization::DEFAULT>();
}

TEST(
    FwdConvInstances,
    Create_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor_Instance_2D_FP16_GNHWC_Filter1x1Pad0)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 128,
                                         .tile_size  = {.m = 128, .n = 128, .k = 32}};

    run_test_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
        FwdConvSignature,
        FwdThreadBlock,
        ConvFwdSpecialization::FILTER_1X1_PAD0>();
}

} // namespace ck_tile::builder::testing

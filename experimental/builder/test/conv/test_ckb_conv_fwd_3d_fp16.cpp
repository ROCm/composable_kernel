// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_common.hpp"

using namespace ck_tile::builder::test_utils;

namespace ck_tile::builder::testing {

// 3D FP16 NDHWGC (channels-last) with Pipeline V4 and FILTER_1X1_PAD0
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP16_NDHWGC_ChannelsLast)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 3,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 128, .n = 128, .k = 32}};

    run_test_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        FwdConvSignature,
        FwdThreadBlock,
        PipelineVersion::V4,
        ConvFwdSpecialization::FILTER_1X1_PAD0>();
}

} // namespace ck_tile::builder::testing

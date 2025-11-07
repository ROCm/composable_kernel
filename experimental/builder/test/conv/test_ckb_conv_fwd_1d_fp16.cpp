// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_common.hpp"

using namespace ck_tile::builder::test_utils;

namespace ck_tile::builder::testing {

// 1D FP16 (channels-last) with DEFAULT specialization
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_1D_FP16_ChannelsFirst_scale)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 1,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout1D::NWGC_GKXC_NWGK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 64,
                                         .tile_size  = {.m = 64, .n = 32, .k = 32}};

    run_test_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<FwdConvSignature,
                                                          FwdThreadBlock,
                                                          ConvFwdSpecialization::DEFAULT>();
}

} // namespace ck_tile::builder::testing

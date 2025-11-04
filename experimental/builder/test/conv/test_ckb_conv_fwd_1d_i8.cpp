// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_common.hpp"

using namespace ck_tile::builder::test_utils;

namespace ck_tile::builder::testing {

// 1D I8 (channels-last) with and DEFAULT specialization
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_Instance_1D_FP32_ChannelsFirst_scale)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 1,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout1D::GNWC_GKXC_GNWK,
        .data_type             = DataType::I8,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 128,
                                         .tile_size  = {.m = 64, .n = 64, .k = 64}};

    run_test_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<FwdConvSignature,
                                                         FwdThreadBlock,
                                                         ConvFwdSpecialization::DEFAULT>();
}

} // namespace ck_tile::builder::testing

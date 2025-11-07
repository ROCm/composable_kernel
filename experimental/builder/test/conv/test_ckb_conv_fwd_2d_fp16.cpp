// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_common.hpp"

using namespace ck_tile::builder::test_utils;

namespace ck_tile::builder::testing {

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_FP16_GNHWC)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::GNHWC_GKYXC_GNHWK,
        .data_type             = DataType::FP16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 256, .k = 32}};

    run_test_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        FwdConvSignature,
        FwdThreadBlock,
        PipelineVersion::V3,
        ConvFwdSpecialization::FILTER_1X1_PAD0>();
}

} // namespace ck_tile::builder::testing

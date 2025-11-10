// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_common.hpp"

using namespace ck_tile::builder::test_utils;

namespace ck_tile::builder::testing {

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_FP32_NGCHW_GKCYX)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 2,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout2D::NGCHW_GKCYX_NGKHW,
        .data_type             = DataType::FP32,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 128, .n = 128, .k = 32}};

    run_test_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        FwdConvSignature,
        FwdThreadBlock,
        PipelineVersion::V4,
        ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0>();
}

} // namespace ck_tile::builder::testing

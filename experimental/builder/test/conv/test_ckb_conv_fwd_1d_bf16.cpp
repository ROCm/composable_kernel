// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_common.hpp"

using namespace ck_tile::builder::test_utils;

namespace ck_tile::builder::testing {

// 1D BF16 (channels-first) with Pipeline V2 and FILTER_1X1_STRIDE1_PAD0 specialization and SCALE
// elementwise op
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_1D_BF16_ChannelsFirst_scale)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim           = 1,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout1D::NGCW_GKXC_NGKW,
        .data_type             = DataType::BF16,
        .elementwise_operation = ElementwiseOperation::SCALE,
        .device_operation =
            FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3};

    constexpr ThreadBlock FwdThreadBlock{.block_size = 256,
                                         .tile_size  = {.m = 256, .n = 256, .k = 32}};

    run_test_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        FwdConvSignature,
        FwdThreadBlock,
        PipelineVersion::V2,
        ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0>();
}

} // namespace ck_tile::builder::testing

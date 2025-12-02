// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 3D FP16 NDHWGC (channels-last) with Pipeline V4 and FILTER_1X1_PAD0
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_FP16_NDHWGC_ChannelsLast)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim            = 3,
        .direction              = ConvDirection::FORWARD,
        .data_type              = DataType::FP16,
        .accumulation_data_type = DataType::FP32,
        .input                  = {.config = {.layout = TensorLayout::NDHWGC}},
        .weight                 = {.config = {.layout = TensorLayout::GKZYXC}},
        .output                 = {.config = {.layout = TensorLayout::NDHWGK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_128x128x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::FILTER_1X1_PAD0,
                                  GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v4_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256,128,128,32",
                       "Filter1x1Pad0",
                       "Intrawave",
                       "v4",
                       "NDHWGC,GKZYXC,EmptyTuple,NDHWGK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

} // namespace

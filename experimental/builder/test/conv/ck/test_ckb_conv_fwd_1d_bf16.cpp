// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 1D BF16 (channels-first) with Pipeline V2 and FILTER_1X1_STRIDE1_PAD0 specialization and SCALE
// elementwise op
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_1D_BF16_ChannelsFirst_scale)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim            = 1,
        .direction              = ConvDirection::FORWARD,
        .data_type              = DataType::BF16,
        .accumulation_data_type = DataType::FP32,
        .input                  = {.config = {.layout = TensorLayout::NGCW}},
        .weight                 = {.config = {.layout = TensorLayout::GKXC}},
        .output                 = {.config    = {.layout = TensorLayout::NGKW},
                                   .operation = {.elementwise_operation = ElementwiseOperation::SCALE}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0,
                                  GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v2_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256,256,256,32",
                       "NGCW,GKXC,EmptyTuple,NGKW",
                       "PassThrough,PassThrough,Scale",
                       "Filter1x1Stride1Pad0",
                       "MNKPadding",
                       "Intrawave",
                       "v2"});
}

} // namespace

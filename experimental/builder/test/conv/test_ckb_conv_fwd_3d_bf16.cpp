// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 3D BF16 GNDHWC (group-first, channels-last) with Pipeline V3 and DEFAULT
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_3D_BF16_GNDHWC)
{
    constexpr ConvSignature FwdConvSignature{
        .spatial_dim            = 3,
        .direction              = ConvDirection::FORWARD,
        .data_type              = DataType::BF16,
        .accumulation_data_type = DataType::FP32,
        .input                  = {.config = {.layout = TensorLayout::GNDHWC}},
        .weight                 = {.config = {.layout = TensorLayout::GKZYXC}},
        .output                 = {.config = {.layout = TensorLayout::GNDHWK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v3_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256,256,256,32",
                       "Default",
                       "Intrawave",
                       "v3",
                       "GNDHWC,GKZYXC,EmptyTuple,GNDHWK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

} // namespace

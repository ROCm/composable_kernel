// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

// 2D BF16 NHWGC (channels-last) with Pipeline V1 and DEFAULT
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_ChannelsLast)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = ConvDirection::FORWARD,
                                             .data_type              = DataType::BF16,
                                             .accumulation_data_type = DataType::FP32,
                                             .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                             .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                             .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v1_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256,256,256,32",
                       "Default",
                       "NHWGC,GKYXC,EmptyTuple,NHWGK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding",
                       "Intrawave",
                       "v1"});
}

// 2D BF16 NHWGC (channels-last) with Pipeline V5 and FILTER_3x3
TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_BF16_NHWGC_Filter3x3)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = ConvDirection::FORWARD,
                                             .data_type              = DataType::BF16,
                                             .accumulation_data_type = DataType::FP32,
                                             .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                             .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                             .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::FILTER_3x3, GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v5_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "Filter3x3",
                       "NHWGC,GKYXC,EmptyTuple,NHWGK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding",
                       "v5"});
}

} // namespace

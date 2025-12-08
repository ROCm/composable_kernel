// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"

namespace {

using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3_Instance_2D_FP16_GNHWC)
{
    constexpr ConvSignature FwdConvSignature{.spatial_dim            = 2,
                                             .direction              = ConvDirection::FORWARD,
                                             .data_type              = DataType::FP16,
                                             .accumulation_data_type = DataType::FP32,
                                             .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                             .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                             .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
            .with_thread_block(FwdThreadBlock_256_256x256x32)
            .with_gemm_config(FwdGemmParams_Xdl_4x4_per_wave)
            .with_transfer(FwdTransfer_4x64x1)
            .with_specializations(ConvFwdSpecialization::FILTER_1X1_PAD0,
                                  GemmSpecialization::MNKPadding)
            .with_block_gemm(BlockGemmDesc_v3_intrawave);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256,256,256,32",
                       "Filter1x1Pad0",
                       "Intrawave",
                       "v3",
                       "GNHWC,GKYXC,EmptyTuple,GNHWK",
                       "PassThrough,PassThrough,PassThrough",
                       "MNKPadding"});
}

} // namespace

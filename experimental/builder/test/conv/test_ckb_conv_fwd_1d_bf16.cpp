// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
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
        .spatial_dim           = 1,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout1D::NGCW_GKXC_NGKW,
        .data_type             = DataType::BF16,
        .elementwise_operation = ElementwiseOperation::SCALE};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock_256x256x32,
        .gridwise_gemm       = FwdGemmParams_Xdl_4x4_per_wave,
        .block_transfer      = FwdBlockTransfer_4x64_1,
        .fwd_specialization  = ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .block_gemm          = BlockGemmDesc_v2_intrawave};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256, 256, 256, 32",
                       "Filter1x1Stride1Pad0",
                       "BlkGemmPipelineScheduler: Intrawave",
                       "BlkGemmPipelineVersion: v2"});
}

} // namespace

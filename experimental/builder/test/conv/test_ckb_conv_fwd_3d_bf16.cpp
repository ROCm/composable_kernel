// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
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
        .spatial_dim           = 3,
        .direction             = ConvDirection::FORWARD,
        .layout                = GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK,
        .data_type             = DataType::BF16,
        .elementwise_operation = ElementwiseOperation::PASS_THROUGH};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock_256x256x32,
        .gridwise_gemm       = FwdGemmParams_Xdl_4x4_per_wave,
        .block_transfer      = FwdBlockTransfer_4x64x1,
        .fwd_specialization  = ConvFwdSpecialization::DEFAULT,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .block_gemm          = BlockGemmDesc_v3_intrawave};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                       "256, 256, 256, 32",
                       "Default",
                       "BlkGemmPipelineScheduler: Intrawave",
                       "BlkGemmPipelineVersion: v3"});
}

} // namespace
